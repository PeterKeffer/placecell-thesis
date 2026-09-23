"""Place-model training stage."""

from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from placecell_research.analysis.directionality import directional_modulation_per_unit
from placecell_research.analysis.within_heading_reliability import (
    compute_within_heading_reliability,
)
from placecell_research.artifacts.compatibility import (
    CompatibilityReference,
    validate_artifact_compatibility,
)
from placecell_research.artifacts.config_snapshots import write_artifact_config_snapshots
from placecell_research.artifacts.ids import generate_artifact_id
from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.collection.stage_support import (
    StageRuntime,
    apply_configured_output_tags,
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config import (
    artifact_match_fingerprint,
    resolve_artifact_reference_id,
    resolve_matching_artifact,
    resolve_reuse_target,
    summarize_reuse,
)
from placecell_research.config.schema import ExperimentConfig, SpatialTrainingConfig
from placecell_research.datasets.dataset import build_training_dataloaders
from placecell_research.evaluation.metrics import (
    compute_heading_tuning_shape_eval_metrics,
)
from placecell_research.evaluation.online_probes import (
    dense_sparse_decode_gap,
    heading_decodability,
    participation_ratio,
)
from placecell_research.evaluation.parallel_sources import (
    SharedArrays,
    SourceDecodeJob,
    decode_sources,
    should_use_worker_pool,
)
from placecell_research.evaluation.source_decode import (
    SourceDecodeSettings,
    source_decode_metrics,
)
from placecell_research.numerics.error_metrics import RMSE_AGGREGATION
from placecell_research.numerics.place_cell_quality import resolve_place_cell_gate_thresholds
from placecell_research.numerics.rate_map_kernels import (
    resolve_place_metric_settings,
)
from placecell_research.objectives import build_objectives, compute_total_loss
from placecell_research.objectives.registry import MetricValue
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.spatial_model.components.sparsifiers import KWinnersSparsifier
from placecell_research.spatial_model.contract import (
    build_model_contract,
    write_parameter_shapes_csv,
)
from placecell_research.spatial_model.loading import select_place_model_checkpoint
from placecell_research.tracking import (
    ConsoleProgressReporter,
    emit_metrics_block,
    emit_text_block,
    managed_stage_run,
    stage_tags,
)
from placecell_research.training.checkpointing import (
    claim_latest_compatible_recovery_checkpoint,
    hold_recovery_checkpoint_lock,
    load_optimizer_state_dict,
)
from placecell_research.training.loop import TrainLoopConfig, train_model
from placecell_research.training.optimizer import build_optimizer
from placecell_research.utils.device import resolve_device
from placecell_research.utils.metrics import (
    materialize_metric_values,
    snapshot_metric_values,
    sum_metric_batches,
)
from placecell_research.utils.seeds import SeedBundle, seed_everything

_CHECKPOINT_FILE_NAMES = (
    "weights_best_primary.pt",
    "weights_best_validation_loss.pt",
    "weights_last.pt",
)

_ROUTING_ACTIVE_EPSILON = 1e-12
_ROUTING_DIAGNOSTIC_MAX_SAMPLES = 8_192


def _extra_world_records(config: ExperimentConfig) -> list[dict[str, str]]:
    """The extra worlds pooled into the training loader, in the order they are concatenated."""
    return [
        {"dataset_id": world.dataset_id, "split_id": world.split_id}
        for world in config.dataset.extra_worlds
    ]


def _place_model_input_artifact_ids(
    config: ExperimentConfig,
    raw_config: dict[str, object],
    *,
    resume_artifact_id: str = "",
) -> list[str]:
    """Every artifact the trained model actually consumed, in provenance order."""
    input_artifact_ids = [config.dataset.artifact_id, config.splits.artifact_id]
    for world in _extra_world_records(config):
        input_artifact_ids.extend((world["dataset_id"], world["split_id"]))
    if config.vision.artifact_id:
        input_artifact_ids.append(config.vision.artifact_id)
    if resume_artifact_id:
        input_artifact_ids.append(resume_artifact_id)
    return input_artifact_ids


def _place_model_compatibility_references(
    config: ExperimentConfig,
    *,
    explicit_reuse_artifact_id: str,
    allow_domain_transfer: bool,
) -> list[CompatibilityReference]:
    """Bundles that must be internally consistent: every pooled world plus any resume artifact."""
    if not (config.dataset.artifact_id and config.splits.artifact_id):
        return []
    references = [
        CompatibilityReference(
            label="train_place_model inputs",
            dataset_artifact_id=config.dataset.artifact_id,
            dataset_artifact_type=config.dataset.artifact_type,
            split_artifact_id=config.splits.artifact_id,
        )
    ]
    references.extend(
        CompatibilityReference(
            label=f"train_place_model extra world {world['dataset_id']}",
            dataset_artifact_id=world["dataset_id"],
            dataset_artifact_type=config.dataset.artifact_type,
            split_artifact_id=world["split_id"],
        )
        for world in _extra_world_records(config)
    )
    if explicit_reuse_artifact_id:
        references.append(
            CompatibilityReference(
                label="train_place_model resume artifact",
                dataset_artifact_id=config.dataset.artifact_id,
                dataset_artifact_type=config.dataset.artifact_type,
                split_artifact_id=config.splits.artifact_id,
                model_artifact_id=str(explicit_reuse_artifact_id),
                allow_model_dataset_mismatch=allow_domain_transfer,
            )
        )
    return references


def _place_model_resume_fingerprint_payload(
    config: ExperimentConfig,
    raw_config: dict[str, object],
) -> dict[str, object]:
    """Inputs that must agree before adopting another run's optimizer and random state."""
    payload: dict[str, object] = {
        "spatial_model": raw_config.get("spatial_model", {}),
        "seed": raw_config.get("seed", {}),
        "dataset_artifact_id": config.dataset.artifact_id,
        "split_artifact_id": config.splits.artifact_id,
        "vision_encoder_artifact_id": config.vision.artifact_id,
    }
    extra_worlds = _extra_world_records(config)
    if extra_worlds:
        payload["extra_worlds"] = extra_worlds
    return payload


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _seed_recovery_checkpoints(resume_checkpoint: Path, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(resume_checkpoint, checkpoint_dir / "weights_last.pt")
    for file_name in _CHECKPOINT_FILE_NAMES[:-1]:
        sibling = resume_checkpoint.parent / file_name
        if sibling.exists():
            _link_or_copy(sibling, checkpoint_dir / file_name)
    _link_or_copy(resume_checkpoint, checkpoint_dir / "weights_best_primary.pt")


def _recovery_adoption_blocker(
    checkpoint_path: Path,
    model: nn.Module,
    auxiliary_heads: nn.ModuleDict,
    training_config: SpatialTrainingConfig,
) -> str | None:
    """Why this recovery checkpoint cannot be adopted, or None when it can."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("optimizer_parameter_identity") is None:
        return "Full optimizer resume requires optimizer_parameter_identity in the checkpoint."
    optimizer_state = payload.get("optimizer_state_dict")
    if not optimizer_state:
        return None
    probe_optimizer, _ = build_optimizer(model, auxiliary_heads, training_config)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_optimizer_state_dict(probe_optimizer, optimizer_state)
    except ValueError as migration_error:
        return str(migration_error)
    return None


def _compute_online_probe_metrics(
    representation_array: np.ndarray,
    position_array: np.ndarray,
    valid_array: np.ndarray,
    heading_array: np.ndarray | None,
    dense_array: np.ndarray | None,
    dense_source: str | None,
    *,
    train_fraction: float,
    ridge_alpha: float,
) -> dict[str, float]:
    """Cheap early-signal probes on the gathered validation activations."""
    valid_flat = valid_array.reshape(-1).astype(bool)
    codes = representation_array.reshape(-1, representation_array.shape[-1])[valid_flat]
    positions = position_array.reshape(-1, 2)[valid_flat]
    if codes.shape[0] < 10:
        return {}
    metrics: dict[str, float] = {
        "validation.participation_ratio": participation_ratio(codes),
    }
    if heading_array is not None:
        heading = heading_array.reshape(-1)[valid_flat]
        metrics["validation.heading_decode_r2"] = heading_decodability(
            codes, heading, train_fraction=train_fraction, alpha=ridge_alpha
        )
    if dense_array is not None:
        dense = dense_array.reshape(-1, dense_array.shape[-1])[valid_flat]
        if dense_source is not None:
            metrics[f"validation.{dense_source}.participation_ratio"] = participation_ratio(dense)
        gap = dense_sparse_decode_gap(
            dense, codes, positions, train_fraction=train_fraction, alpha=ridge_alpha
        )
        metrics["validation.dense_decode_r2"] = gap.dense_r2
        metrics["validation.sparse_decode_r2"] = gap.sparse_r2
        metrics["validation.dense_sparse_decode_gap"] = gap.gap
    return metrics


def _spatial_bin_ids(
    positions: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
) -> np.ndarray:
    minimum = positions.min(axis=0)
    span = np.maximum(positions.max(axis=0) - minimum, np.finfo(np.float32).eps)
    normalized = (positions - minimum) / span
    x_bin = np.clip((normalized[:, 0] * num_bins_x).astype(np.int64), 0, num_bins_x - 1)
    y_bin = np.clip((normalized[:, 1] * num_bins_y).astype(np.int64), 0, num_bins_y - 1)
    return y_bin * num_bins_x + x_bin


def _compute_region_winner_metrics(
    representation_array: np.ndarray,
    position_array: np.ndarray,
    valid_array: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
) -> dict[str, float]:
    """Separate never-selected units from rare regional specialists during validation."""
    valid_flat = valid_array.reshape(-1).astype(bool, copy=False)
    flattened = representation_array.reshape(-1, representation_array.shape[-1])[valid_flat]
    positions = position_array.reshape(-1, 2)[valid_flat]
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(flattened).all(axis=1)
    flattened = flattened[finite]
    positions = positions[finite]
    if flattened.size == 0:
        return {}

    active = np.abs(flattened) > _ROUTING_ACTIVE_EPSILON
    region_ids = _spatial_bin_ids(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )
    order = np.argsort(region_ids, kind="stable")
    sorted_regions = region_ids[order]
    sorted_active = active[order]
    occupied_regions, starts, occupancy = np.unique(
        sorted_regions,
        return_index=True,
        return_counts=True,
    )
    del occupied_regions
    region_win_counts = np.stack(
        [
            sorted_active[start : start + count].sum(axis=0, dtype=np.int64)
            for start, count in zip(starts, occupancy, strict=False)
        ]
    )
    conditional_frequency = region_win_counts / occupancy[:, None]
    global_frequency = active.mean(axis=0)
    observed_target_frequency = float(active.sum(axis=1).mean() / active.shape[1])
    max_conditional_frequency = conditional_frequency.max(axis=0)
    rare_specialist = (
        (global_frequency > 0.0)
        & (global_frequency < 0.5 * observed_target_frequency)
        & (max_conditional_frequency >= observed_target_frequency)
    )
    monopolist = global_frequency > 2.0 * observed_target_frequency

    region_active_fraction = (region_win_counts > 0).mean(axis=1)
    region_win_mass = region_win_counts / np.maximum(
        region_win_counts.sum(axis=1, keepdims=True),
        1,
    )
    entropy_terms = np.where(
        region_win_mass > 0.0,
        region_win_mass * np.log(np.maximum(region_win_mass, np.finfo(np.float64).tiny)),
        0.0,
    )
    entropy_denominator = max(np.log(float(active.shape[1])), np.finfo(np.float64).eps)
    region_entropy = -entropy_terms.sum(axis=1) / entropy_denominator
    return {
        "validation.routing.never_selected_fraction": float((global_frequency == 0.0).mean()),
        "validation.routing.rare_specialist_fraction": float(rare_specialist.mean()),
        "validation.routing.monopolist_fraction": float(monopolist.mean()),
        "validation.routing.occupied_region_count": float(occupancy.size),
        "validation.routing.region_active_unit_fraction_mean": float(
            region_active_fraction.mean()
        ),
        "validation.routing.region_active_unit_fraction_min": float(
            region_active_fraction.min()
        ),
        "validation.routing.region_win_entropy_mean": float(region_entropy.mean()),
        "validation.routing.region_win_entropy_min": float(region_entropy.min()),
    }


def _compute_noise_floor_region_agreement(
    dense_array: np.ndarray,
    position_array: np.ndarray,
    valid_array: np.ndarray,
    *,
    balance_bias: np.ndarray,
    k_fraction: float,
    noise_scale: float,
    num_bins_x: int,
    num_bins_y: int,
    random_seed: int = 0,
) -> dict[str, float]:
    """Simulate final-noise routing without mutating the model or consuming its RNG state."""
    if noise_scale <= 0.0:
        return {}
    valid_flat = valid_array.reshape(-1).astype(bool, copy=False)
    dense = dense_array.reshape(-1, dense_array.shape[-1])[valid_flat]
    positions = position_array.reshape(-1, 2)[valid_flat]
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(dense).all(axis=1)
    dense = dense[finite]
    positions = positions[finite]
    if dense.size == 0 or balance_bias.shape != (dense.shape[1],):
        return {}

    rng = np.random.default_rng(random_seed)
    if dense.shape[0] > _ROUTING_DIAGNOSTIC_MAX_SAMPLES:
        selected = rng.choice(
            dense.shape[0],
            size=_ROUTING_DIAGNOSTIC_MAX_SAMPLES,
            replace=False,
        )
        dense = dense[selected]
        positions = positions[selected]
    spread = dense.std(axis=1, ddof=1, keepdims=True)
    spread = np.maximum(spread, np.finfo(dense.dtype).eps)
    normalized = (dense - dense.mean(axis=1, keepdims=True)) / spread
    deterministic_scores = normalized + balance_bias[None, :]
    noisy_scores = deterministic_scores + noise_scale * rng.standard_normal(
        deterministic_scores.shape
    )
    k_active = max(1, round(dense.shape[1] * k_fraction))
    deterministic_winners = np.argpartition(
        deterministic_scores,
        -k_active,
        axis=1,
    )[:, -k_active:]
    noisy_winners = np.argpartition(noisy_scores, -k_active, axis=1)[:, -k_active:]
    agreement = (
        deterministic_winners[:, :, None] == noisy_winners[:, None, :]
    ).sum(axis=(1, 2)) / k_active
    region_ids = _spatial_bin_ids(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )
    region_agreement = np.asarray(
        [agreement[region_ids == region_id].mean() for region_id in np.unique(region_ids)]
    )
    return {
        "validation.routing.noise_floor_winner_agreement": float(agreement.mean()),
        "validation.routing.noise_floor_region_agreement_mean": float(
            region_agreement.mean()
        ),
        "validation.routing.noise_floor_region_agreement_p10": float(
            np.quantile(region_agreement, 0.1)
        ),
        "validation.routing.noise_floor_region_agreement_min": float(
            region_agreement.min()
        ),
    }


def _encoder_kwinners_sparsifier(model_instance) -> KWinnersSparsifier | None:
    root_model = getattr(model_instance, "root", model_instance)
    encoder_stack = getattr(root_model, "encoder_stack", None)
    sparsifier = getattr(encoder_stack, "sparsifier", None)
    return sparsifier if isinstance(sparsifier, KWinnersSparsifier) else None


def _compute_directionality_eval_metrics(
    representation_array: np.ndarray,
    position_array: np.ndarray,
    valid_array: np.ndarray,
    heading_array: np.ndarray | None,
    *,
    num_bins_x: int,
    num_bins_y: int,
    field_threshold_fraction: float,
) -> dict[str, float]:
    if heading_array is None:
        return {}
    directional_r = directional_modulation_per_unit(
        representation=representation_array,
        position_xy=position_array,
        heading=heading_array,
        valid_mask=valid_array,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=None,
        min_occupancy_per_quadrant=5,
        min_heading_quadrants=3,
        field_threshold_fraction=field_threshold_fraction,
        min_field_bins=3,
        min_fire_rate=0.01,
    )
    heading_tuning_metrics = compute_heading_tuning_shape_eval_metrics(
        representation_array=representation_array,
        valid_array=valid_array,
        heading_array=heading_array,
    )
    assessable = directional_r[np.isfinite(directional_r)]
    within_heading = compute_within_heading_reliability(
        representation=representation_array,
        position_xy=position_array,
        heading=heading_array,
        valid_mask=valid_array,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )
    gain = within_heading.gain[np.isfinite(within_heading.gain)]
    total_units = int(directional_r.size)
    metrics: dict[str, float] = {
        "validation.directionality_total_units": float(total_units),
        "validation.directionality_assessable_units": float(assessable.size),
        "validation.directionality_assessable_fraction": (
            float(assessable.size / total_units) if total_units else 0.0
        ),
    }
    if assessable.size:
        omnidirectionality = 1.0 - assessable
        metrics["validation.place_field_omnidirectionality_mean"] = float(omnidirectionality.mean())
        metrics["validation.place_field_omnidirectionality_median"] = float(
            np.median(omnidirectionality)
        )
        metrics["validation.directionality_mean_r"] = float(assessable.mean())
        metrics["validation.directionality_median_r"] = float(np.median(assessable))
        metrics["validation.fraction_omnidirectional"] = float((assessable < 0.3).mean())
        metrics["validation.fraction_directional"] = float((assessable > 0.6).mean())
        metrics["validation.place_field_fraction_omnidirectional"] = metrics[
            "validation.fraction_omnidirectional"
        ]
        metrics["validation.place_field_fraction_directional"] = metrics[
            "validation.fraction_directional"
        ]
    if gain.size:
        metrics["validation.within_heading_reliability_gain"] = float(gain.mean())
    metrics.update(heading_tuning_metrics)
    return metrics


def _relabel_metric_split(metrics: dict[str, float], new_prefix: str) -> dict[str, float]:
    """Rename the leading validation."""
    prefix = "validation."
    relabeled: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith(prefix):
            relabeled[f"{new_prefix}.{key[len(prefix) :]}"] = value
        else:
            relabeled[key] = value
    return relabeled


def _source_decode_settings(config) -> SourceDecodeSettings:
    """Flatten the rate-map and decode knobs the source kernel reads out of the config."""
    return SourceDecodeSettings(
        num_bins_x=int(config.analysis.num_bins_x),
        num_bins_y=int(config.analysis.num_bins_y),
        smoothing_sigma=float(config.analysis.smoothing_sigma),
        min_occupancy=float(config.analysis.min_occupancy),
        decode_train_fraction=float(config.evaluation.decode_train_fraction),
        decode_ridge_alpha=float(config.evaluation.decode_ridge_alpha),
        decode_include_shuffle=bool(config.evaluation.decode_include_shuffle),
        spatial_info_top_k=int(config.evaluation.spatial_info_top_k),
        split_half_num_random_splits=int(config.evaluation.online_split_half_num_random_splits),
        place_metric_settings=resolve_place_metric_settings(config.analysis),
        place_cell_gate_thresholds=resolve_place_cell_gate_thresholds(config.analysis),
    )


def _decode_every_source(
    primary_array: np.ndarray,
    extra_chunk_lists: dict[str, list[np.ndarray]],
    position_array: np.ndarray,
    valid_array: np.ndarray,
    heading_array: np.ndarray | None,
    kinematics_array: np.ndarray | None,
    *,
    primary_source: str,
    config,
) -> dict[str, dict[str, float]]:
    """Decode metrics for the primary source and every extra source that has a full pass."""
    settings = _source_decode_settings(config)

    def decode(representation: np.ndarray) -> dict[str, float]:
        return source_decode_metrics(
            representation,
            position_array,
            valid_array,
            heading_array,
            kinematics_array,
            settings=settings,
        )

    element_counts = [int(primary_array.size)] + [
        sum(int(chunk.size) for chunk in chunks) for chunks in extra_chunk_lists.values()
    ]
    if not should_use_worker_pool(element_counts):
        decoded = {primary_source: decode(primary_array)}
        for source, chunks in extra_chunk_lists.items():
            decoded[source] = decode(np.concatenate(chunks, axis=0))
            chunks.clear()
        return decoded

    with SharedArrays() as shared:
        shared_inputs = (
            shared.publish(position_array),
            shared.publish(valid_array),
            shared.publish(heading_array),
            shared.publish(kinematics_array),
        )
        jobs = [
            SourceDecodeJob(primary_source, shared.publish(primary_array), *shared_inputs, settings)
        ]
        for source, chunks in extra_chunk_lists.items():
            jobs.append(
                SourceDecodeJob(source, shared.publish_chunks(chunks), *shared_inputs, settings)
            )
            chunks.clear()
        return decode_sources(jobs)


def _to_host(representation: torch.Tensor) -> np.ndarray:
    """Copy one batch of a representation to host RAM."""
    return representation.detach().cpu().numpy()


def _track_duplicate_sources(
    duplicate_sources: dict[str, str],
    batch_representations: dict[str, torch.Tensor],
    *,
    first_batch: bool,
) -> None:
    """Track which evaluated sources hold the same values as an earlier evaluated source."""
    ordered_sources = list(batch_representations)
    if first_batch:
        for index, source in enumerate(ordered_sources):
            for earlier_source in ordered_sources[:index]:
                if torch.equal(
                    batch_representations[source], batch_representations[earlier_source]
                ):
                    duplicate_sources[source] = earlier_source
                    break
        return
    for source, earlier_source in list(duplicate_sources.items()):
        if (
            source not in batch_representations
            or earlier_source not in batch_representations
            or not torch.equal(
                batch_representations[source], batch_representations[earlier_source]
            )
        ):
            del duplicate_sources[source]


def evaluate_place_model_online(
    model_instance,
    objectives,
    validation_loader,
    *,
    config,
    device: torch.device,
    online_source: str,
    extra_sources: Sequence[str] = (),
) -> dict[str, float]:
    if validation_loader is None:
        primary_metric = config.spatial_model.training.selection.primary_metric
        return {"validation.total_loss": 0.0, primary_metric: 0.0}

    model_instance.train(False)
    metric_batches: list[dict[str, float | torch.Tensor]] = []
    batches = 0
    representation_chunks: list[np.ndarray] = []
    position_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    heading_chunks: list[np.ndarray] = []
    kinematics_chunks: list[np.ndarray] = []
    dense_chunks: list[np.ndarray] = []
    extra_chunks: dict[str, list[np.ndarray]] = {source: [] for source in extra_sources}
    duplicate_sources: dict[str, str] = {}
    dense_source = (
        online_source.replace(".place_codes", ".pre_sparsifier")
        if online_source.endswith(".place_codes")
        else None
    )
    collect_dense = dense_source is not None
    max_eval_episodes = int(config.spatial_model.training.max_validation_episodes)
    processed_episodes = 0
    with torch.inference_mode():
        for batch in validation_loader:
            remaining_episodes = (
                max_eval_episodes - processed_episodes if max_eval_episodes > 0 else None
            )
            if remaining_episodes is not None and remaining_episodes <= 0:
                break
            batch_episode_count = int(batch["valid_steps"].shape[0])
            if remaining_episodes is not None and batch_episode_count > remaining_episodes:
                batch = {
                    key: value[:remaining_episodes]
                    if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_episode_count,)
                    else value
                    for key, value in batch.items()
                }
            batch_on_device = {
                key: value.to(device, non_blocking=device.type == "cuda")
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            processed_episodes += int(batch_on_device["valid_steps"].shape[0])
            bundle = model_instance.forward_sequence(batch_on_device)
            _, metrics = compute_total_loss(
                objectives,
                bundle,
                batch_on_device,
                config.spatial_model,
            )
            metric_batches.append(snapshot_metric_values(metrics))
            primary_representation = bundle.get_representation(online_source)
            representation_chunks.append(_to_host(primary_representation))
            batch_representations: dict[str, torch.Tensor] = {
                online_source: primary_representation
            }
            for source in list(extra_chunks):
                try:
                    batch_representations[source] = bundle.get_representation(source)
                except KeyError:
                    del extra_chunks[source]
                    duplicate_sources.pop(source, None)
            _track_duplicate_sources(
                duplicate_sources,
                batch_representations,
                first_batch=batches == 0,
            )
            for source, source_representation in batch_representations.items():
                if source == online_source or source in duplicate_sources:
                    continue
                extra_chunks[source].append(_to_host(source_representation))
            if collect_dense:
                try:
                    dense_chunks.append(
                        bundle.get_representation(dense_source).detach().cpu().numpy()
                    )
                except KeyError:
                    collect_dense = False
                    dense_chunks.clear()
            if "position_xy" in batch:
                position_chunks.append(batch["position_xy"].detach().cpu().numpy())
            valid_chunks.append(batch["valid_steps"].detach().cpu().numpy())
            if "heading" in batch:
                heading_chunks.append(batch["heading"].detach().cpu().numpy())
            if "kinematics" in batch:
                kinematics_chunks.append(batch["kinematics"].detach().cpu().numpy())
            batches += 1

    metric_sums = sum_metric_batches(metric_batches)
    metric_batches.clear()
    averaged = {
        f"validation.{key.removeprefix('loss/')}": value / max(batches, 1)
        for key, value in metric_sums.items()
    }
    averaged.setdefault("validation.total_loss", averaged.get("validation.total", 0.0))
    if representation_chunks:
        representation_array = np.concatenate(representation_chunks, axis=0)
        representation_chunks.clear()
        position_array = np.concatenate(position_chunks, axis=0)
        valid_array = np.concatenate(valid_chunks, axis=0)
        heading_array = np.concatenate(heading_chunks, axis=0) if heading_chunks else None
        kinematics_array = np.concatenate(kinematics_chunks, axis=0) if kinematics_chunks else None
        decodable_sources = [
            source
            for source, source_chunks in extra_chunks.items()
            if len(source_chunks) == batches and source not in duplicate_sources
        ]
        decoded_metrics = _decode_every_source(
            representation_array,
            {source: extra_chunks[source] for source in decodable_sources},
            position_array,
            valid_array,
            heading_array,
            kinematics_array,
            primary_source=online_source,
            config=config,
        )
        label_sources = len(extra_sources) > 0
        for metric_name, value in decoded_metrics[online_source].items():
            averaged[f"validation.{metric_name}"] = value
            if label_sources:
                averaged[f"validation.{online_source}.{metric_name}"] = value
        for source in extra_chunks:
            source_metrics = decoded_metrics.get(
                source, decoded_metrics.get(duplicate_sources.get(source, ""))
            )
            if source_metrics is None:
                continue
            for metric_name, value in source_metrics.items():
                averaged[f"validation.{source}.{metric_name}"] = value
        dense_array = (
            np.concatenate(dense_chunks, axis=0)
            if dense_chunks and len(dense_chunks) == batches
            else None
        )
        averaged.update(
            _compute_online_probe_metrics(
                representation_array,
                position_array,
                valid_array,
                heading_array,
                dense_array,
                dense_source,
                train_fraction=float(config.evaluation.decode_train_fraction),
                ridge_alpha=float(config.evaluation.decode_ridge_alpha),
            )
        )
        num_bins_x = int(config.analysis.num_bins_x)
        num_bins_y = int(config.analysis.num_bins_y)
        averaged.update(
            _compute_region_winner_metrics(
                representation_array,
                position_array,
                valid_array,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
            )
        )
        encoder_sparsifier = _encoder_kwinners_sparsifier(model_instance)
        if dense_array is not None and encoder_sparsifier is not None:
            averaged.update(
                _compute_noise_floor_region_agreement(
                    dense_array,
                    position_array,
                    valid_array,
                    balance_bias=encoder_sparsifier.balance_bias.detach().cpu().numpy(),
                    k_fraction=float(encoder_sparsifier.k_fraction),
                    noise_scale=float(encoder_sparsifier.selection_noise_final_scale),
                    num_bins_x=num_bins_x,
                    num_bins_y=num_bins_y,
                )
            )
        if config.evaluation.eval_directionality:
            averaged.update(
                _compute_directionality_eval_metrics(
                    representation_array,
                    position_array,
                    valid_array,
                    heading_array,
                    num_bins_x=int(config.analysis.num_bins_x),
                    num_bins_y=int(config.analysis.num_bins_y),
                    field_threshold_fraction=float(
                        config.analysis.place_field_threshold_fraction
                    ),
                )
            )
    primary_metric = config.spatial_model.training.selection.primary_metric
    if primary_metric != "validation.xy_decode_rmse":
        averaged.setdefault(
            primary_metric,
            averaged.get("validation.xy_decode_rmse", averaged["validation.total_loss"]),
        )
    return averaged


def run(
    config_path: Path,
    overrides: list[str],
    *,
    on_runtime_created: Callable[[StageRuntime], None] | None = None,
) -> dict[str, object]:
    runtime = initialize_stage_runtime(config_path, overrides, "train_place_model")
    if on_runtime_created is not None:
        on_runtime_created(runtime)
    config = runtime.config
    raw_config = runtime.raw_payload
    policies = config.policies
    stage_log_path = runtime.run_directory.logs_dir / "stage_train_model.log"
    explicit_reuse_artifact_reference = config.reuse.place_model_artifact_id
    explicit_reuse_artifact_id = (
        resolve_artifact_reference_id(
            runtime.artifact_registry, "place_model", explicit_reuse_artifact_reference
        )
        if explicit_reuse_artifact_reference
        else ""
    )
    reuse_target = resolve_reuse_target(
        config.reuse.place_model_artifact_id,
        policies.training_resume,
        registry=runtime.artifact_registry if config.reuse.place_model_artifact_id else None,
        artifact_type="place_model" if config.reuse.place_model_artifact_id else None,
    )
    with (
        ExitStack() as recovery_lock_stack,
        managed_stage_run(
            config=config,
            run_directory=runtime.run_directory,
            stage_name="train_place_model",
            run_name=f"{config.tracking.variant_name}__{runtime.run_directory.identity.run_id}",
            tags=stage_tags(
                "train_place_model",
                config.environment.env_id,
                base_tags=config.tracking.tags,
                study_name=config.tracking.study_name,
                variant_name=config.tracking.variant_name,
                variant_slug=runtime.run_directory.identity.variant_slug,
                seed=config.seed.global_seed,
            ),
        ) as stage_run,
    ):
        stage_run.define_metric("trainer/step")
        stage_run.define_metric("train/*", step_metric="trainer/step")
        stage_run.define_metric("validation/*", step_metric="trainer/step")
        stage_run.update_config(
            {"reuse_summary": summarize_reuse(config, artifact_registry=runtime.artifact_registry)}
        )
        input_artifact_ids = _place_model_input_artifact_ids(
            config, raw_config, resume_artifact_id=explicit_reuse_artifact_id
        )
        allow_domain_transfer = config.reuse.allow_domain_transfer
        fingerprint_payload = _place_model_resume_fingerprint_payload(config, raw_config)
        stage_fingerprint = artifact_match_fingerprint(fingerprint_payload)
        runtime.run_directory.update_run_manifest(
            {"summary": {"place_model_resume_fingerprint": stage_fingerprint}}
        )
        compatibility_references = _place_model_compatibility_references(
            config,
            explicit_reuse_artifact_id=explicit_reuse_artifact_id,
            allow_domain_transfer=allow_domain_transfer,
        )
        if compatibility_references:
            validate_artifact_compatibility(runtime.artifact_registry, compatibility_references)

        def _reuse_existing_artifact(artifact_id: str, *, reuse_mode: str) -> dict[str, object]:
            reused_artifact = runtime.artifact_registry.load("place_model", artifact_id)
            architecture_path = reused_artifact.path / "architecture.txt"
            selected_checkpoint = select_place_model_checkpoint(
                reused_artifact.path,
                selection=config.policies.checkpoint_selection,
            )
            best_primary_checkpoint = reused_artifact.path / "weights_best_primary.pt"
            applied_output_tags = apply_configured_output_tags(
                runtime, "place_model", reused_artifact.path
            )
            runtime.run_directory.update_run_manifest(
                {
                    "status": "reused",
                    "reused_artifact_ids": [artifact_id],
                    "summary": {
                        "place_model_artifact_id": artifact_id,
                        "place_model_artifact_path": str(reused_artifact.path),
                        "checkpoint_selection": config.policies.checkpoint_selection,
                        "selected_checkpoint_name": selected_checkpoint.name,
                        "output_tags": applied_output_tags,
                    },
                },
            )
            runtime.run_directory.write_symlink("results/place_model", reused_artifact.path)
            runtime.run_directory.write_symlink(
                "results/place_model_checkpoint_selected.pt", selected_checkpoint
            )
            if best_primary_checkpoint.exists():
                runtime.run_directory.write_symlink(
                    "results/place_model_checkpoint_best_primary.pt",
                    best_primary_checkpoint,
                )
            if architecture_path.exists():
                runtime.run_directory.write_symlink(
                    "results/place_model_architecture.txt", architecture_path
                )
                emit_text_block(
                    "place_model_architecture",
                    architecture_path.read_text(),
                    metadata={
                        "artifact_id": artifact_id,
                        "reuse_mode": reuse_mode,
                    },
                    log_path=stage_log_path,
                )
            for file_name in ("model_contract.json", "active_objectives.json", "input_routes.json"):
                source_path = reused_artifact.path / file_name
                if source_path.exists():
                    runtime.run_directory.write_symlink(f"results/{file_name}", source_path)
            stage_run.finalize(
                status="reused",
                summary={
                    "place_model_artifact_id": artifact_id,
                    "place_model_artifact_path": str(reused_artifact.path),
                    "checkpoint_selection": config.policies.checkpoint_selection,
                    "selected_checkpoint_name": selected_checkpoint.name,
                    "output_tags": applied_output_tags,
                },
                upload_files=[
                    reused_artifact.path / "manifest.json",
                    reused_artifact.path / "model_contract.json",
                    reused_artifact.path / "active_objectives.json",
                    reused_artifact.path / "input_routes.json",
                    architecture_path,
                    selected_checkpoint,
                    *(
                        [best_primary_checkpoint]
                        if best_primary_checkpoint.exists()
                        and best_primary_checkpoint != selected_checkpoint
                        else []
                    ),
                ],
            )
            return augment_stage_result(
                runtime,
                {
                    "train_place_model.model_artifact_id": artifact_id,
                    "evaluation.model_artifact_id": artifact_id,
                    "analysis.model_artifact_id": artifact_id,
                    "checkpoint_path": str(selected_checkpoint),
                    "checkpoint_selection": config.policies.checkpoint_selection,
                    "selected_checkpoint_name": selected_checkpoint.name,
                    "place_model_artifact_id": artifact_id,
                },
            )

        if reuse_target.stage_behavior == "reuse_existing_artifact":
            return _reuse_existing_artifact(
                reuse_target.artifact_id, reuse_mode=reuse_target.stage_behavior
            )
        if reuse_target.stage_behavior == "train_fresh" and policies.training_resume == "fresh":
            matching_artifact = resolve_matching_artifact(
                runtime.artifact_registry,
                "place_model",
                policies.artifact_reuse,
                config_fingerprint_value=stage_fingerprint,
                input_artifact_ids=input_artifact_ids,
            )
            if matching_artifact is not None:
                return _reuse_existing_artifact(
                    matching_artifact.artifact_id, reuse_mode="reuse_if_config_match"
                )

        seeds = SeedBundle(
            global_seed=config.seed.global_seed,
            collection_seed=config.seed.collection_seed,
            split_seed=config.seed.split_seed,
            training_seed=config.seed.training_seed,
        ).resolve()
        seed_everything(seeds.training_seed)

        device = resolve_device(config.spatial_model.training.device)
        train_loader, validation_loader, dataset_metadata = build_training_dataloaders(
            config,
            artifact_root=runtime.repo_root / config.tracking.artifact_root,
            shuffle_seed=seeds.training_seed,
        )
        training_window = int(config.spatial_model.training.bptt_window)
        sequence_length = int(dataset_metadata["episode_length"])
        chunks_per_batch = (
            1
            if training_window == 0
            else max(1, (sequence_length + training_window - 1) // training_window)
        )
        total_optimizer_steps = (
            config.spatial_model.training.epochs * max(len(train_loader), 1) * chunks_per_batch
        )
        optimizer_steps_per_epoch = max(len(train_loader), 1) * chunks_per_batch
        model_build_context = ModelBuildContext(
            num_actions=int(dataset_metadata["num_actions"]),
            observation_dim=int(dataset_metadata["observation_dim"]),
            kinematics_dim=int(dataset_metadata["kinematics_dim"]),
            total_optimizer_steps=total_optimizer_steps,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        )
        model = build_place_model(
            config=config.spatial_model,
            build_context=model_build_context,
        )
        built_objectives = build_objectives(model, config.spatial_model)
        if hasattr(model, "set_auxiliary_heads"):
            model.set_auxiliary_heads(built_objectives.auxiliary_heads)
        online_source = config.evaluation.online_decode_source
        extra_sources = [source for source in config.evaluation.sources if source != online_source]

        def evaluate_fn(model_instance, auxiliary_heads, objectives, val_loader):
            del auxiliary_heads
            metrics = evaluate_place_model_online(
                model_instance,
                objectives,
                val_loader,
                config=config,
                device=device,
                online_source=online_source,
                extra_sources=extra_sources,
            )
            if config.evaluation.evaluate_training_split:
                training_metrics = evaluate_place_model_online(
                    model_instance,
                    objectives,
                    train_loader,
                    config=config,
                    device=device,
                    online_source=online_source,
                    extra_sources=extra_sources,
                )
                metrics.update(_relabel_metric_split(training_metrics, "training"))
            return metrics

        resume_artifact_id = explicit_reuse_artifact_id
        resume_checkpoint: Path | None = None
        auto_resume_source_run_id: str | None = None
        effective_resume_policy = policies.training_resume
        configured_checkpoint_path = str(config.reuse.place_model_checkpoint_path).strip()
        if configured_checkpoint_path:
            resume_checkpoint = Path(configured_checkpoint_path).expanduser()
            if not resume_checkpoint.is_absolute():
                resume_checkpoint = runtime.repo_root / resume_checkpoint
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Configured place-model checkpoint does not exist: {resume_checkpoint}"
                )
        elif effective_resume_policy != "fresh" and resume_artifact_id:
            resume_artifact = runtime.artifact_registry.load("place_model", str(resume_artifact_id))
            resume_checkpoint = select_place_model_checkpoint(
                resume_artifact.path,
                selection=(
                    "last" if effective_resume_policy == "weights_and_optimizer" else "best"
                ),
            )
        elif config.policies.auto_resume_interrupted:
            auto_resume_candidate = recovery_lock_stack.enter_context(
                claim_latest_compatible_recovery_checkpoint(
                    runtime.run_directory.root,
                    resume_fingerprint=stage_fingerprint,
                    exclude_run_id=runtime.run_directory.identity.run_id,
                )
            )
            if auto_resume_candidate is not None:
                adoption_blocker = _recovery_adoption_blocker(
                    auto_resume_candidate.path,
                    model,
                    built_objectives.auxiliary_heads,
                    config.spatial_model.training,
                )
                if adoption_blocker is None:
                    resume_checkpoint = auto_resume_candidate.path
                    auto_resume_source_run_id = auto_resume_candidate.run_id
                    effective_resume_policy = "weights_and_optimizer"
                else:
                    emit_text_block(
                        "auto_resume_declined",
                        f"Training fresh instead of adopting {auto_resume_candidate.path}: "
                        f"{adoption_blocker}",
                        metadata={"interrupted_run_id": auto_resume_candidate.run_id},
                        log_path=stage_log_path,
                    )

        recovery_checkpoint_dir = (
            runtime.run_directory.results_dir / "place_model_training_checkpoints"
        )
        recovery_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        recovery_lock_stack.enter_context(hold_recovery_checkpoint_lock(recovery_checkpoint_dir))
        if resume_checkpoint is not None and effective_resume_policy == "weights_and_optimizer":
            _seed_recovery_checkpoints(resume_checkpoint, recovery_checkpoint_dir)
        runtime.run_directory.update_run_manifest(
            {
                "summary": {
                    "place_model_resume_fingerprint": stage_fingerprint,
                    "place_model_recovery_checkpoint": str(
                        recovery_checkpoint_dir / "weights_last.pt"
                    ),
                    "auto_resume_source_run_id": auto_resume_source_run_id,
                    "resume_from_checkpoint_path": (
                        str(resume_checkpoint) if resume_checkpoint else None
                    ),
                    "effective_training_resume_policy": effective_resume_policy,
                }
            }
        )
        stage_run.update_config(
            {
                "resolved_training_resume": {
                    "policy": effective_resume_policy,
                    "checkpoint_path": str(resume_checkpoint) if resume_checkpoint else None,
                    "auto_resume_source_run_id": auto_resume_source_run_id,
                }
            }
        )

        artifact_id = generate_artifact_id(
            "place_model",
            config.environment.env_id,
            runtime.run_directory.identity.run_id,
        )
        architecture_text = model.architecture_summary()
        emit_text_block(
            "place_model_architecture",
            architecture_text,
            metadata={
                "artifact_id": artifact_id,
                "encoder_family": config.spatial_model.encoder.family,
                "predictor_family": config.spatial_model.predictor.family,
            },
            log_path=stage_log_path,
        )
        stage_run.update_config(
            {
                "stage_inputs": {
                    "dataset_artifact_id": config.dataset.artifact_id,
                    "split_artifact_id": config.splits.artifact_id,
                    "resume_from_artifact_reference": explicit_reuse_artifact_reference or None,
                    "resume_from_artifact_id": resume_artifact_id or None,
                    "resume_from_checkpoint_path": str(resume_checkpoint)
                    if resume_checkpoint
                    else None,
                },
                "model_contract_preview": build_model_contract(
                    model, sorted(config.spatial_model.objectives.keys())
                ),
                "active_objective_names": sorted(config.spatial_model.objectives.keys()),
            }
        )
        runtime.artifact_registry.root.mkdir(parents=True, exist_ok=True)
        log_interval_steps = max(1, int(config.tracking.log_interval_steps))
        progress_reporter = ConsoleProgressReporter(
            "train_place_model",
            stream=sys.stderr,
        )

        def _log_step_metrics(metrics: dict[str, MetricValue], step_count: int) -> None:
            if step_count % log_interval_steps != 0:
                return
            stage_run.log(
                {"trainer/step": step_count, **materialize_metric_values(metrics)},
                step=step_count,
            )

        def _log_epoch_metrics(metrics: dict[str, float], step_count: int) -> None:
            stage_run.log({"trainer/step": step_count, **metrics}, step=step_count)
            epoch_index = int(metrics.get("epoch", 0.0)) + 1
            emit_metrics_block(
                "train_place_model",
                metrics,
                metadata={
                    "epoch": f"{epoch_index}/{config.spatial_model.training.epochs}",
                    "step": step_count,
                    "primary_metric": config.spatial_model.training.selection.primary_metric,
                },
                log_path=stage_log_path,
            )

        with runtime.artifact_registry.temporary_directory(
            prefix="place_model_",
        ) as temporary_directory_name:
            temporary_dir = temporary_directory_name
            result = train_model(
                model=model,
                built_objectives=built_objectives,
                model_config=config.spatial_model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                loop_config=TrainLoopConfig(
                    training=config.spatial_model.training,
                    checkpoint_dir=recovery_checkpoint_dir,
                    device=device,
                    eval_every_n_epochs=int(config.evaluation.eval_every_n_epochs),
                    eval_schedule=str(config.evaluation.eval_schedule),
                    sequence_length=sequence_length,
                    build_context=model_build_context.to_checkpoint_payload(
                        config.to_dict()["spatial_model"]
                    ),
                    resume_checkpoint=resume_checkpoint,
                    resume_policy=effective_resume_policy,
                    step_metrics_callback=_log_step_metrics,
                    epoch_metrics_callback=_log_epoch_metrics,
                    progress_callback=progress_reporter,
                ),
                evaluate_fn=evaluate_fn,
            )

            for checkpoint_path in recovery_checkpoint_dir.glob("weights_*.pt"):
                _link_or_copy(checkpoint_path, temporary_dir / checkpoint_path.name)

            active_objectives = {
                name: asdict(objective_config)
                if is_dataclass(objective_config)
                else dict(objective_config)
                for name, objective_config in config.spatial_model.objectives.items()
            }
            model_contract = build_model_contract(
                model, sorted(config.spatial_model.objectives.keys())
            )
            (temporary_dir / "architecture.txt").write_text(architecture_text.rstrip() + "\n")
            (temporary_dir / "model_contract.json").write_text(
                json.dumps(model_contract, indent=2, sort_keys=True) + "\n"
            )
            (temporary_dir / "active_objectives.json").write_text(
                json.dumps(active_objectives, indent=2, sort_keys=True) + "\n"
            )
            (temporary_dir / "input_routes.json").write_text(
                json.dumps(
                    {
                        "predictor_input_mode": config.spatial_model.inputs.predictor_input_mode,
                        "predictor_context_channels": (
                            config.spatial_model.inputs.predictor_context_channels
                        ),
                        "append_temporal_offset": (
                            config.spatial_model.inputs.input_corruption.append_temporal_offset
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            write_parameter_shapes_csv(temporary_dir / "parameter_shapes.csv", model)
            write_artifact_config_snapshots(
                temporary_dir,
                raw_config,
                stage_name="train_place_model",
                section_names=[
                    "spatial_model",
                    "dataset",
                    "splits",
                    "environment",
                    "collection",
                    "evaluation",
                    "analysis",
                    "seed",
                    "policies",
                    "reuse",
                    "tracking",
                ],
                extra_payload={
                    "artifact_id": artifact_id,
                    "torch_tf32": {
                        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn": bool(torch.backends.cudnn.allow_tf32),
                    },
                    "dataset_artifact_id": config.dataset.artifact_id,
                    "split_artifact_id": config.splits.artifact_id,
                    "extra_worlds": _extra_world_records(config) or None,
                    "vision_encoder_artifact_id": config.vision.artifact_id or None
                    or None,
                    "resume_from_artifact_reference": explicit_reuse_artifact_reference or None,
                    "resume_from_artifact_id": resume_artifact_id or None,
                    "resume_from_checkpoint_path": str(resume_checkpoint)
                    if resume_checkpoint
                    else None,
                },
            )

            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                artifact_type="place_model",
                created_by=CreatedBy(
                    run_id=runtime.run_directory.identity.run_id, stage_name="train_place_model"
                ),
                input_artifact_ids=[
                    artifact_id for artifact_id in input_artifact_ids if artifact_id
                ],
                config_fingerprint=stage_fingerprint,
                git_commit=str(runtime.git_state.get("commit", "")),
                summary={
                    "rmse_aggregation": RMSE_AGGREGATION,
                    "final_metrics": result.final_metrics,
                    "artifact_reuse_policy": policies.artifact_reuse,
                    "training_resume_policy": effective_resume_policy,
                    "auto_resume_source_run_id": auto_resume_source_run_id,
                    "dataset_artifact_id": config.dataset.artifact_id,
                    "split_artifact_id": config.splits.artifact_id,
                    "extra_worlds": _extra_world_records(config) or None,
                    "vision_encoder_artifact_id": config.vision.artifact_id or None
                    or None,
                    "observation_source": config.spatial_model.inputs.observation_source,
                    "resume_from_artifact_reference": explicit_reuse_artifact_reference or None,
                    "resume_from_artifact_id": resume_artifact_id or None,
                    "resume_from_checkpoint_path": str(resume_checkpoint)
                    if resume_checkpoint
                    else None,
                },
            )
            manifest.write(temporary_dir / "manifest.json")
            destination = runtime.artifact_registry.register_directory(
                "place_model", artifact_id, temporary_dir
            )
        runtime.artifact_registry.mark_artifact_completed("place_model", artifact_id)
        applied_output_tags = apply_configured_output_tags(runtime, "place_model", destination)

        runtime.run_directory.update_run_manifest(
            {
                "status": "completed",
                "produced_artifact_ids": [artifact_id],
                "summary": {
                    "place_model_artifact_id": artifact_id,
                    "place_model_artifact_path": str(destination),
                    "output_tags": applied_output_tags,
                    **result.final_metrics,
                },
            }
        )
        runtime.run_directory.write_symlink("results/place_model", destination)
        runtime.run_directory.write_symlink(
            "results/place_model_architecture.txt", destination / "architecture.txt"
        )
        best_primary_checkpoint = destination / "weights_best_primary.pt"
        if best_primary_checkpoint.exists():
            runtime.run_directory.write_symlink(
                "results/place_model_checkpoint_best_primary.pt",
                best_primary_checkpoint,
            )
        selected_checkpoint = select_place_model_checkpoint(
            destination,
            selection=config.policies.checkpoint_selection,
        )
        runtime.run_directory.write_symlink(
            "results/place_model_checkpoint_selected.pt",
            selected_checkpoint,
        )
        stage_run.finalize(
            status="completed",
            summary={
                "place_model_artifact_id": artifact_id,
                "place_model_artifact_path": str(destination),
                "checkpoint_selection": config.policies.checkpoint_selection,
                "selected_checkpoint_name": selected_checkpoint.name,
                "output_tags": applied_output_tags,
                **result.final_metrics,
            },
            upload_files=[
                destination / "manifest.json",
                destination / "resolved_config.yaml",
                destination / "used_hyperparameters.yaml",
                destination / "architecture.txt",
                destination / "model_contract.json",
                destination / "active_objectives.json",
                destination / "input_routes.json",
                destination / "parameter_shapes.csv",
                *([best_primary_checkpoint] if best_primary_checkpoint.exists() else []),
                *(
                    [selected_checkpoint]
                    if selected_checkpoint != best_primary_checkpoint
                    else []
                ),
            ],
        )
        return augment_stage_result(
            runtime,
            {
                "train_place_model.model_artifact_id": artifact_id,
                "evaluation.model_artifact_id": artifact_id,
                "analysis.model_artifact_id": artifact_id,
                "checkpoint_path": str(selected_checkpoint),
                "checkpoint_selection": config.policies.checkpoint_selection,
                "selected_checkpoint_name": selected_checkpoint.name,
                "place_model_artifact_id": artifact_id,
                **result.final_metrics,
            },
        )
