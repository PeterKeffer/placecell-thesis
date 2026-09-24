"""Model evaluation stage."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from placecell_research.artifacts.compatibility import (
    CompatibilityReference,
    resolve_place_model_observation_source,
    validate_artifact_compatibility,
)
from placecell_research.collection.stage_support import (
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config import artifact_match_fingerprint, resolve_matching_artifact
from placecell_research.datasets.batch_iterator import available_split_names, load_split_indices
from placecell_research.evaluation.decode import (
    RidgeDecoderState,
    fit_ridge_position_decoder,
    score_ridge_position_decoder,
)
from placecell_research.evaluation.inference import collect_representations, load_model_checkpoint
from placecell_research.evaluation.matched_decode import matched_decode_metrics
from placecell_research.evaluation.online import evaluate_representations
from placecell_research.evaluation.representation_store import (
    RepresentationRequest,
    read_representation_manifest,
    resolve_representations,
)
from placecell_research.evaluation.runtime import (
    publish_report,
    resolve_registry_reference,
    resolve_stage_dataset_reference,
    resolve_stage_reference,
    resolve_stage_split_reference,
)
from placecell_research.numerics.error_metrics import RMSE_AGGREGATION
from placecell_research.numerics.place_cell_quality import resolve_place_cell_gate_thresholds
from placecell_research.numerics.rate_map_kernels import (
    RateMapComputation,
    compute_rate_maps,
    resolve_place_metric_settings,
    skaggs_spatial_information,
)
from placecell_research.tracking import (
    ConsoleProgressReporter,
    ProgressTracker,
    emit_metrics_block,
    managed_stage_run,
    stage_tags,
)
from placecell_research.training.loop import apply_tf32_policy
from placecell_research.utils.device import resolve_device
from placecell_research.utils.source_fingerprint import package_source_fingerprint


def _configured_evaluation_split_names(
    configured_splits: list[str],
    *,
    primary_split_name: str,
    available_splits: list[str],
) -> list[str]:
    ordered_splits = [str(split_name) for split_name in configured_splits]
    deduplicated_splits = list(dict.fromkeys(ordered_splits))
    if primary_split_name not in deduplicated_splits:
        deduplicated_splits.insert(0, primary_split_name)
    usable_splits = [
        split_name for split_name in deduplicated_splits if split_name in available_splits
    ]
    if not usable_splits and primary_split_name in available_splits:
        return [primary_split_name]
    return usable_splits


def _split_scoped_metrics(
    metrics: dict[str, float],
    *,
    split_name: str,
    primary_split_name: str,
) -> dict[str, float]:
    scoped = {f"{split_name}.{metric_name}": value for metric_name, value in metrics.items()}
    if split_name == primary_split_name:
        scoped.update(metrics)
    return scoped


def _transfer_decoder_metrics(
    representations: dict[str, np.ndarray],
    position_xy: np.ndarray,
    valid_steps: np.ndarray | None,
    *,
    transfer_decoders: dict[str, RidgeDecoderState],
    is_train_split: bool,
    ridge_alpha: float,
) -> dict[str, float]:
    """Fit the cross-split position decoders on train, score them on every other split."""
    if not is_train_split and not transfer_decoders:
        return {}
    flat_positions = position_xy.reshape(-1, 2)
    flat_valid = (
        valid_steps.reshape(-1).astype(bool, copy=False) if valid_steps is not None else None
    )
    if (
        flat_valid is not None
        and flat_valid.shape[0] == flat_positions.shape[0]
        and bool(flat_valid.all())
    ):
        flat_valid = None
    if flat_valid is not None:
        flat_positions = flat_positions[flat_valid]
    metrics: dict[str, float] = {}
    for source_name, source_values in representations.items():
        if not is_train_split and source_name not in transfer_decoders:
            continue
        flat_codes = source_values.reshape(-1, source_values.shape[-1])
        if flat_valid is not None:
            flat_codes = flat_codes[flat_valid]
        if is_train_split:
            transfer_decoders[source_name] = fit_ridge_position_decoder(
                flat_codes,
                flat_positions,
                alpha=ridge_alpha,
            )
            continue
        transfer_rmse, transfer_r2 = score_ridge_position_decoder(
            transfer_decoders[source_name],
            flat_codes,
            flat_positions,
        )
        metrics[f"{source_name}.decode_transfer_rmse"] = float(transfer_rmse)
        metrics[f"{source_name}.decode_transfer_r2"] = float(transfer_r2)
    return metrics


def _evaluation_upload_files(report_path: Path) -> list[Path]:
    file_names = [
        "manifest.json",
        "resolved_config.yaml",
        "used_hyperparameters.yaml",
        "metrics.json",
        "metrics.csv",
        "evaluated_model_contract.json",
        "evaluated_model_used_hyperparameters.yaml",
    ]
    return [
        report_path / file_name for file_name in file_names if (report_path / file_name).exists()
    ]


def run(config_path: Path, overrides: list[str]) -> dict[str, object]:
    runtime = initialize_stage_runtime(config_path, overrides, "evaluate_model")
    config = runtime.config
    raw_config = runtime.raw_payload
    policies = config.policies
    registry = runtime.artifact_registry
    run_directory = runtime.run_directory
    apply_tf32_policy(config.spatial_model.training.allow_tf32)
    device = resolve_device(config.evaluation.device)
    stage_log_path = run_directory.logs_dir / "stage_evaluate.log"
    with managed_stage_run(
        config=config,
        run_directory=run_directory,
        stage_name="evaluate_model",
        run_name=f"evaluate__{config.name}__{run_directory.identity.run_id}",
        tags=stage_tags(
            "evaluate_model",
            config.environment.env_id,
            base_tags=config.tracking.tags,
            study_name=config.tracking.study_name,
            variant_name=config.name,
            variant_slug=run_directory.identity.variant_slug,
            seed=config.seed.global_seed,
        ),
    ) as stage_run:
        progress_reporter = ConsoleProgressReporter("evaluate_model")
        model_id = resolve_registry_reference(
            registry,
            "place_model",
            resolve_stage_reference(
                config.evaluation,
                "evaluation",
                "model_artifact_id",
                fallback=config.reuse.place_model_artifact_id,
            ),
        ).artifact_id
        model_artifact = registry.load("place_model", model_id)
        model_observation_source = resolve_place_model_observation_source(registry, model_id)
        dataset_id, dataset_type = resolve_stage_dataset_reference(
            registry=registry,
            section=config.evaluation,
            section_name="evaluation",
            fallback_artifact_id=config.dataset.artifact_id,
            fallback_artifact_type=config.dataset.artifact_type,
            fallback_model_artifact_id=model_id,
        )
        split_id = resolve_stage_split_reference(
            registry=registry,
            section=config.evaluation,
            section_name="evaluation",
            fallback_artifact_id=config.splits.artifact_id,
            fallback_model_artifact_id=model_id,
        )
        requested_primary_split_name = config.evaluation.split_name
        split_artifact = registry.load("split_set", split_id)
        representation_set_id = config.reuse.representation_set_artifact_id
        representation_set_directory = (
            registry.load("representation_set", representation_set_id).path
            if representation_set_id
            else None
        )
        stored_inference = (
            read_representation_manifest(representation_set_directory)
            if representation_set_directory is not None else None
        )
        available_evaluation_splits = available_split_names(split_artifact.path)
        requested_evaluation_split_names = _configured_evaluation_split_names(
            config.evaluation.split_names,
            primary_split_name=requested_primary_split_name,
            available_splits=available_evaluation_splits,
        )
        if not requested_evaluation_split_names:
            raise ValueError(
                "No usable evaluation splits were found in "
                f"{split_artifact.path / 'split_indices.json'}."
            )
        evaluation_split_names = requested_evaluation_split_names
        primary_split_name = requested_primary_split_name
        if config.evaluation.matched_decode and not {"train", "validation"}.issubset(
            evaluation_split_names
        ):
            raise ValueError("matched_decode requires train and validation evaluation splits.")
        source_names = list(config.evaluation.sources)
        stage_fingerprint = artifact_match_fingerprint(
            {
                "rmse_aggregation": RMSE_AGGREGATION,
                "implementation_fingerprint": package_source_fingerprint(),
                "evaluation": config.to_dict()["evaluation"],
                "checkpoint_selection": config.policies.checkpoint_selection,
                "model_artifact_id": model_id,
                "dataset_artifact_id": dataset_id,
                "dataset_artifact_type": dataset_type,
                "split_artifact_id": split_id,
                "split_name": primary_split_name,
                "split_names": evaluation_split_names,
                "sources": source_names,
                "model_observation_source": model_observation_source,
                "evaluated_model_provenance_version": 1,
            }
        )
        stage_run.update_config(
            {
                "stage_inputs": {
                    "model_artifact_id": model_id,
                    "dataset_artifact_id": dataset_id,
                    "dataset_artifact_type": dataset_type,
                    "split_artifact_id": split_id,
                    "split_name": primary_split_name,
                    "split_names": evaluation_split_names,
                    "requested_split_name": requested_primary_split_name,
                    "requested_split_names": requested_evaluation_split_names,
                    "sources": source_names,
                    "model_observation_source": model_observation_source,
                }
            }
        )
        validate_artifact_compatibility(
            registry,
            [
                CompatibilityReference(
                    label="evaluate_model inputs",
                    dataset_artifact_id=dataset_id,
                    dataset_artifact_type=dataset_type,
                    split_artifact_id=split_id,
                    model_artifact_id=model_id,
                )
            ],
        )
        matching_report = resolve_matching_artifact(
            registry,
            "evaluation_report",
            policies.artifact_reuse,
            config_fingerprint_value=stage_fingerprint,
            input_artifact_ids=[model_id, dataset_id, split_id] + (
                [representation_set_id] if representation_set_id else []
            ),
        )
        if matching_report is not None:
            flattened_metrics = {}
            metrics_path = matching_report.path / "metrics.json"
            if metrics_path.exists():
                flattened_metrics = {
                    key: float(value) for key, value in json.loads(metrics_path.read_text()).items()
                }
            run_directory.update_run_manifest(
                {
                    "status": "reused",
                    "reused_artifact_ids": [matching_report.artifact_id],
                    "summary": {
                        "evaluation_report_id": matching_report.artifact_id,
                        "evaluation_report_path": str(matching_report.path),
                        **flattened_metrics,
                    },
                }
            )
            run_directory.write_symlink("results/evaluation_report", matching_report.path)
            if flattened_metrics:
                stage_run.log(flattened_metrics)
                emit_metrics_block(
                    "evaluate_model",
                    flattened_metrics,
                    metadata={
                        "split": primary_split_name,
                        "split_names": evaluation_split_names,
                        "sources": len(source_names),
                        "report_id": matching_report.artifact_id,
                        "reuse_mode": "reuse_if_config_match",
                    },
                    log_path=stage_log_path,
                )
            stage_run.finalize(
                status="reused",
                summary={
                    "evaluation_report_id": matching_report.artifact_id,
                    "evaluation_report_path": str(matching_report.path),
                    **flattened_metrics,
                },
                upload_files=_evaluation_upload_files(matching_report.path),
            )
            return augment_stage_result(
                runtime,
                {
                    "evaluation_report_id": matching_report.artifact_id,
                    "evaluation_report_path": str(matching_report.path),
                    **flattened_metrics,
                },
            )
        dataset_artifact = registry.load(dataset_type, dataset_id)
        model = None
        if representation_set_directory is None:
            model, _ = load_model_checkpoint(
                model_artifact.path, device, selection=config.policies.checkpoint_selection,
            )
        flattened_metrics: dict[str, float] = {}
        batch_size = config.evaluation.batch_size

        transfer_decoders: dict[str, RidgeDecoderState] = {}
        matched_decoders = {}
        ordered_split_names = sorted(
            evaluation_split_names,
            key=lambda name: {"train": 0, "validation": 1}.get(name, 2)
        )
        for split_name in ordered_split_names:
            request = None
            if representation_set_directory is not None:
                episode_ids = load_split_indices(split_artifact.path, split_name)
                if config.evaluation.max_eval_episodes > 0:
                    episode_ids = episode_ids[:config.evaluation.max_eval_episodes]
                request = RepresentationRequest(
                    place_model_artifact_id=model_id, dataset_artifact_id=dataset_id,
                    dataset_artifact_type=dataset_type, split_artifact_id=split_id,
                    checkpoint_selection=config.policies.checkpoint_selection,
                    device=stored_inference["device"],
                    batch_size=stored_inference["batch_size"],
                    allow_tf32=config.spatial_model.training.allow_tf32,
                    torch_version=str(torch.__version__), episode_ids=episode_ids,
                )
            representations, metadata = resolve_representations(
                artifact_directory=representation_set_directory,
                request=request,
                split_name=split_name,
                source_names=source_names,
                collect=partial(
                    collect_representations,
                    model,
                    dataset_artifact.path,
                    split_artifact.path,
                    split_name,
                    source_names,
                    device,
                    batch_size=batch_size,
                    max_episodes=int(config.evaluation.max_eval_episodes),
                    progress_callback=progress_reporter,
                ),
            )
            spatial_information_scores: dict[str, np.ndarray] | None = None
            rate_map_results: dict[str, RateMapComputation] | None = None
            if config.evaluation.compute_spatial_info:
                spatial_information_scores = {}
                rate_map_results = {}
                spatial_progress = ProgressTracker(
                    progress_reporter,
                    total=len(representations),
                    unit_name="sources",
                )
                spatial_progress.emit(detail=f"compute spatial information ({split_name})")
                for source_name, source_values in representations.items():
                    rate_map_result = compute_rate_maps(
                        source_values,
                        metadata["position_xy"],
                        metadata.get("valid_steps"),
                        num_bins_x=int(config.analysis.num_bins_x),
                        num_bins_y=int(config.analysis.num_bins_y),
                        smoothing_sigma=float(config.analysis.smoothing_sigma),
                        min_occupancy=float(config.analysis.min_occupancy),
                    )
                    rate_map_results[source_name] = rate_map_result
                    spatial_information_scores[source_name] = np.asarray(
                        skaggs_spatial_information(
                            rate_map_result.rate_maps,
                            rate_map_result.occupancy,
                        ),
                        dtype=np.float32,
                    )
                    spatial_progress.advance(detail=f"spatial info {split_name}:{source_name}")
                del source_values, rate_map_result
            evaluation_results = evaluate_representations(
                representations,
                metadata["position_xy"],
                valid_mask=metadata.get("valid_steps"),
                kinematics=metadata.get("kinematics"),
                heading=metadata.get("heading"),
                train_fraction=float(config.evaluation.decode_train_fraction),
                ridge_alpha=float(config.evaluation.decode_ridge_alpha),
                include_shuffle=bool(config.evaluation.decode_include_shuffle),
                spatial_information_scores=spatial_information_scores,
                rate_map_results=rate_map_results,
                spatial_information_top_k=int(config.evaluation.spatial_info_top_k),
                rate_map_num_bins_x=int(config.analysis.num_bins_x),
                rate_map_num_bins_y=int(config.analysis.num_bins_y),
                rate_map_smoothing_sigma=float(config.analysis.smoothing_sigma),
                rate_map_min_occupancy=float(config.analysis.min_occupancy),
                compute_gridness=bool(config.evaluation.compute_gridness),
                nonlinear_decode_enabled=bool(config.evaluation.nonlinear_decode_enabled),
                nonlinear_decode_hidden_sizes=tuple(
                    int(hidden_size)
                    for hidden_size in config.evaluation.nonlinear_decode_hidden_sizes
                ),
                nonlinear_decode_max_epochs=int(config.evaluation.nonlinear_decode_max_epochs),
                nonlinear_decode_batch_size=int(config.evaluation.nonlinear_decode_batch_size),
                nonlinear_decode_max_train_samples=int(
                    config.evaluation.nonlinear_decode_max_train_samples
                ),
                nonlinear_decode_max_validation_samples=int(
                    config.evaluation.nonlinear_decode_max_validation_samples
                ),
                nonlinear_decode_random_seed=int(config.evaluation.nonlinear_decode_random_seed),
                place_cell_gate_thresholds=resolve_place_cell_gate_thresholds(config.analysis),
                place_metric_settings=resolve_place_metric_settings(config.analysis),
            )
            split_metrics = {
                metric_name: float(metric_value)
                for result in evaluation_results
                for metric_name, metric_value in result.to_metrics().items()
            }
            split_metrics.update(
                _transfer_decoder_metrics(
                    representations,
                    metadata["position_xy"],
                    metadata.get("valid_steps"),
                    transfer_decoders=transfer_decoders,
                    is_train_split=split_name == "train",
                    ridge_alpha=float(config.evaluation.decode_ridge_alpha),
                )
            )
            if config.evaluation.matched_decode:
                split_metrics.update(matched_decode_metrics(
                    representations, metadata["position_xy"], metadata["valid_steps"],
                    split_name, matched_decoders,
                ))
                if split_name != "train":
                    for source_name in representations:
                        for metric in ("decode_rmse", "decode_r2"):
                            key = f"{source_name}.{metric}"
                            within_key = f"{source_name}.within_split_{metric}"
                            split_metrics[within_key] = split_metrics[key]
                            split_metrics[key] = split_metrics[f"{source_name}.matched_{metric}"]
            flattened_metrics.update(
                _split_scoped_metrics(
                    split_metrics,
                    split_name=split_name,
                    primary_split_name=primary_split_name,
                )
            )
            del representations, metadata, rate_map_results, spatial_information_scores
        config_text = json.dumps(raw_config, indent=2, sort_keys=True)

        def _write_report(output_dir: Path, artifact_id: str) -> None:
            del artifact_id
            episode_ids = {
                name: load_split_indices(split_artifact.path, name)
                for name in evaluation_split_names
            }
            if config.evaluation.max_eval_episodes > 0:
                episode_ids = {
                    name: ids[:config.evaluation.max_eval_episodes]
                    for name, ids in episode_ids.items()
                }
            protocol = {
                "rmse_aggregation": RMSE_AGGREGATION,
                "implementation_fingerprint": package_source_fingerprint(),
                "primary_decoder": (
                    "train_standardized_ridge_validation_selected_v1"
                    if config.evaluation.matched_decode else "within_split_episode_holdout_v1"
                ),
                "checkpoint_selection": config.policies.checkpoint_selection,
                "model_artifact_id": model_id,
                "dataset_artifact_id": dataset_id,
                "split_artifact_id": split_id,
                "episode_ids": {name: list(map(int, ids)) for name, ids in episode_ids.items()},
                "evaluation_config": config.to_dict()["evaluation"],
                "representation_set_artifact_id": representation_set_id,
                "representation_inference": (
                    {key: stored_inference[key] for key in
                     ("device", "batch_size", "allow_tf32", "torch_version")}
                    if representation_set_directory is not None else None
                ),
                "implementation": run_directory.load_run_manifest().get("git_state", {}),
            }
            (output_dir / "evaluation_protocol.json").write_text(
                json.dumps(protocol, indent=2, sort_keys=True) + "\n"
            )
            metrics_json = output_dir / "metrics.json"
            metrics_csv = output_dir / "metrics.csv"
            metrics_json.write_text(json.dumps(flattened_metrics, indent=2, sort_keys=True) + "\n")
            metrics_csv.write_text(
                "metric,value\n"
                + "\n".join(f"{key},{value}" for key, value in sorted(flattened_metrics.items()))
                + "\n"
            )
            provenance_files = {
                "model_contract.json": "evaluated_model_contract.json",
                "used_hyperparameters.yaml": "evaluated_model_used_hyperparameters.yaml",
            }
            for source_name, destination_name in provenance_files.items():
                source_path = model_artifact.path / source_name
                if source_path.exists():
                    (output_dir / destination_name).write_text(source_path.read_text())

        report_id, report_path = publish_report(
            registry=registry,
            artifact_type="evaluation_report",
            summary_name=(
                f"{model_id}_{primary_split_name}"
                if len(evaluation_split_names) == 1
                else f"{model_id}_{primary_split_name}_multi_split"
            ),
            run_directory=run_directory,
            stage_name="evaluate_model",
            config_text=config_text,
            input_artifact_ids=[model_id, dataset_id, split_id] + (
                [representation_set_id] if representation_set_id else []
            ),
            files_writer=_write_report,
            config_fingerprint_value=stage_fingerprint,
            raw_config=raw_config,
            active_config=config.to_dict(),
            hyperparameter_sections=[
                "evaluation",
                "dataset",
                "splits",
                "seed",
                "policies",
                "reuse",
                "tracking",
            ],
            hyperparameter_context={
                "model_artifact_id": model_id,
                "dataset_artifact_id": dataset_id,
                "split_artifact_id": split_id,
                "split_name": primary_split_name,
                "split_names": evaluation_split_names,
                "requested_split_name": requested_primary_split_name,
                "requested_split_names": requested_evaluation_split_names,
                "sources": source_names,
                "model_observation_source": model_observation_source,
                "model_config_fingerprint": model_artifact.manifest.config_fingerprint,
            },
        )
        run_directory.update_run_manifest(
            {
                "status": "completed",
                "produced_artifact_ids": [report_id],
                "summary": {
                    "evaluation_report_id": report_id,
                    "evaluation_report_path": str(report_path),
                    **flattened_metrics,
                },
            }
        )
        run_directory.write_symlink("results/evaluation_report", report_path)
        registry.mark_artifact_completed("evaluation_report", report_id)
        stage_run.log(flattened_metrics)
        emit_metrics_block(
            "evaluate_model",
            flattened_metrics,
            metadata={
                "split": primary_split_name,
                "split_names": evaluation_split_names,
                "sources": len(source_names),
                "report_id": report_id,
            },
            log_path=stage_log_path,
        )
        stage_run.finalize(
            status="completed",
            summary={
                "evaluation_report_id": report_id,
                "evaluation_report_path": str(report_path),
                **flattened_metrics,
            },
            upload_files=_evaluation_upload_files(report_path),
        )
        return augment_stage_result(
            runtime,
            {
                "evaluation_report_id": report_id,
                "evaluation_report_path": str(report_path),
                **flattened_metrics,
            },
        )
