"""Offline analysis stage."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Hashable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from placecell_research.analysis.base import AnalysisInput, AnalysisResult
from placecell_research.analysis.registry import (
    ANALYSIS_MODULES,
    COMPARATIVE_MODULES,
)
from placecell_research.analysis.validation_summary import (
    flatten_analysis_results,
    write_summary_csv,
    write_summary_json,
)
from placecell_research.artifacts.compatibility import (
    CompatibilityReference,
    validate_artifact_compatibility,
)
from placecell_research.artifacts.ids import short_fingerprint
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.collection.stage_support import (
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config import artifact_match_fingerprint, resolve_matching_artifact
from placecell_research.config.validator import HARD_K_SPARSIFIER_TYPES
from placecell_research.datasets.batch_iterator import available_split_names, load_split_indices
from placecell_research.datasets.zarr_io import _require_zarr, load_dataset_manifest
from placecell_research.evaluation.inference import (
    InputBatchCache,
    collect_representations,
    configure_open_loop_rollout,
    load_model_checkpoint,
)
from placecell_research.evaluation.metrics import place_code_quality
from placecell_research.evaluation.representation_store import (
    RepresentationRequest,
    read_representation_manifest,
    resolve_representations,
)
from placecell_research.evaluation.runtime import (
    _resolve_unique_direct_input_artifact,
    resolve_registry_reference,
    resolve_stage_dataset_reference,
    resolve_stage_reference,
    resolve_stage_split_reference,
)
from placecell_research.numerics.error_metrics import RMSE_AGGREGATION
from placecell_research.stages._analyze_model_run import (
    build_comparative_work_items,
    build_target_work_items,
    disable_targets_where,
    execute_comparative_work_items,
    execute_single_work_items,
    finalize_report,
)
from placecell_research.tracking import (
    ConsoleProgressReporter,
    ProgressTracker,
    emit_metrics_block,
    managed_stage_run,
    stage_tags,
)
from placecell_research.tracking._run_paths import link_if_absent
from placecell_research.training.loop import apply_tf32_policy
from placecell_research.utils.device import resolve_device
from placecell_research.utils.source_fingerprint import package_source_fingerprint


@dataclass(frozen=True, slots=True)
class _AnalysisSourceReference:
    label: str
    source_name: str
    model_artifact_id: str
    dataset_artifact_id: str
    dataset_artifact_type: str
    split_artifact_id: str
    split_name: str


@dataclass(slots=True)
class _CollectionPlan:
    """Representations and batch metadata needed for one local analysis pass."""

    source_names: tuple[str, ...]
    include_batch_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _SingleAnalysisWorkItem:
    """One concrete single-source analysis execution."""

    progress_label: str
    reference: _AnalysisSourceReference
    module_names: list[str]
    result_key_overrides: dict[str, str] = field(default_factory=dict)

    def result_key(self, module_name: str) -> str:
        return self.result_key_overrides.get(module_name, f"{self.reference.label}.{module_name}")


_CollectionCacheKey = tuple[str, str, str, str, str, tuple[str, ...], tuple[str, ...]]
_CollectionCacheValue = tuple[dict[str, object], dict[str, object], dict[Hashable, object]]
_SourceCollectionGroupKey = tuple[str, str, str, str, str, str]
_COMPARATIVE_SHARED_ANALYSIS_KEYS = (
    "num_bins_x",
    "num_bins_y",
    "smoothing_sigma",
    "min_occupancy",
    "place_field_threshold_fraction",
    "active_peak_rate_threshold",
    "remapping_shuffle_iterations",
    "remapping_shuffle_seed",
    "example_episode_index",
    "example_episode_random_seed",
    "example_episode_top_k",
    "example_episode_frame_duration",
    "sr_oracle_discount_gamma",
    "sr_oracle_num_bins_x",
    "sr_oracle_num_bins_y",
    "successor_return_discount_gamma",
    "successor_return_normalized",
    "successor_return_shuffle_seed",
)
_ANALYSIS_RANDOM_SEED_KEYS = (
    "example_episode_random_seed",
    "probing_shuffle_seed",
    "remapping_shuffle_seed",
    "successor_return_shuffle_seed",
)
_LAYERWISE_HIDDEN_STATE_SOURCE_PATTERN = re.compile(
    r"^(encoder|predictor)\.hidden_state_layer_\d+$"
)
_ROLLOUT_LAYERWISE_HIDDEN_STATE_SOURCE_PATTERN = re.compile(
    r"^predictor_rollout\.hidden_state_layer_\d+$"
)


def _layerwise_availability_source(source_name: str) -> str | None:
    """Return the base hidden-state-layer source that gates source_name's availability."""
    if _LAYERWISE_HIDDEN_STATE_SOURCE_PATTERN.match(source_name):
        return source_name
    if _ROLLOUT_LAYERWISE_HIDDEN_STATE_SOURCE_PATTERN.match(source_name):
        return source_name.replace("predictor_rollout.", "predictor.", 1)
    return None


def _config_value_missing(value: object) -> bool:
    return value is None or value == ""


def _comparative_payload_with_analysis_defaults(
    analysis_config: dict[str, object],
    comparative_payload: dict[str, object],
) -> dict[str, object]:
    shared_defaults = {
        key: analysis_config[key]
        for key in _COMPARATIVE_SHARED_ANALYSIS_KEYS
        if key in analysis_config
    }
    merged = {**shared_defaults, **comparative_payload}
    for key, default_value in shared_defaults.items():
        if _config_value_missing(comparative_payload.get(key)):
            merged[key] = default_value
    return merged


def _analysis_config_with_seed_defaults(
    analysis_config: dict[str, object],
    seed: int,
) -> dict[str, object]:
    resolved = dict(analysis_config)
    for key in _ANALYSIS_RANDOM_SEED_KEYS:
        if _config_value_missing(resolved.get(key)):
            resolved[key] = seed

    comparative_payload = resolved.get("comparative", {})
    if isinstance(comparative_payload, dict):
        resolved["comparative"] = {
            str(analysis_name): (
                _comparative_payload_with_analysis_defaults(resolved, dict(payload))
                if isinstance(payload, dict)
                else payload
            )
            for analysis_name, payload in comparative_payload.items()
        }
    return resolved


def _analysis_config_with_target_order(
    analysis_config: dict[str, object],
) -> dict[str, object]:
    """Reorder analysis targets to match the optional target_order list."""
    target_order = analysis_config.get("target_order") or []
    targets = analysis_config.get("targets")
    if not isinstance(targets, dict) or not isinstance(target_order, (list, tuple)):
        return analysis_config
    leading = [str(name) for name in target_order if str(name) in targets]
    if not leading:
        return analysis_config
    leading_set = set(leading)
    ordered: dict[str, object] = {name: targets[name] for name in leading}
    for name, payload in targets.items():
        if name not in leading_set:
            ordered[name] = payload
    return {**analysis_config, "targets": ordered}


_PLACE_FIELD_OVERLAY_MODULE = "place_field_overlay"
_PLACE_FIELD_OVERLAY_KWINNERS_KEY = "place_field_overlay_kwinners_k_fraction"


def _place_field_overlay_is_enabled(analysis_config: dict[str, object]) -> bool:
    targets = analysis_config.get("targets", {})
    if not isinstance(targets, dict):
        return False
    for target_payload in targets.values():
        if not isinstance(target_payload, dict) or not bool(target_payload.get("enabled", True)):
            continue
        modules = target_payload.get("modules", [])
        if isinstance(modules, (list, tuple)) and _PLACE_FIELD_OVERLAY_MODULE in modules:
            return True
    return False


def _analysis_config_with_overlay_kwinners_default(
    analysis_config: dict[str, object],
    *,
    encoder_sparsifier_type: str,
    encoder_k_fraction: float,
) -> dict[str, object]:
    """Default the place-field overlay's k-winners fraction to the encoder's."""
    if not _config_value_missing(analysis_config.get(_PLACE_FIELD_OVERLAY_KWINNERS_KEY)):
        return analysis_config
    if str(encoder_sparsifier_type) not in HARD_K_SPARSIFIER_TYPES:
        return analysis_config
    if not _place_field_overlay_is_enabled(analysis_config):
        return analysis_config
    return {**analysis_config, _PLACE_FIELD_OVERLAY_KWINNERS_KEY: float(encoder_k_fraction)}


def _analysis_config_with_available_layer_targets(
    analysis_config: dict[str, object],
    *,
    available_representations: set[str] | None,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    if not available_representations:
        return analysis_config, []

    def _layer_source_unavailable(source_name: str, _payload: dict[str, object]) -> bool:
        availability_source = _layerwise_availability_source(source_name)
        return (
            availability_source is not None and availability_source not in available_representations
        )

    return disable_targets_where(analysis_config, _layer_source_unavailable)


def _available_representations_from_model_artifact(
    registry,
    model_artifact_id: str,
) -> set[str] | None:
    model_artifact = registry.load("place_model", model_artifact_id)
    contract_path = model_artifact.path / "model_contract.json"
    if not contract_path.exists():
        return None
    contract = json.loads(contract_path.read_text())
    available_representations = contract.get("available_representations")
    if isinstance(available_representations, list):
        return {str(representation) for representation in available_representations}
    tensor_shapes = contract.get("tensor_shapes")
    if isinstance(tensor_shapes, dict):
        return {str(representation) for representation in tensor_shapes}
    return None


_BELIEF_INPUT_MODES = frozenset({"belief", "dual", "conditional", "gated"})


def _model_consumes_belief(registry, model_artifact_id: str) -> bool:
    try:
        model_artifact = registry.load("place_model", model_artifact_id)
    except Exception:
        return False
    contract_path = model_artifact.path / "model_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        if str(contract.get("predictor_input_mode", "")) in _BELIEF_INPUT_MODES:
            return True
    return False


def _disable_open_loop_targets_when_not_belief_trained(
    analysis_config: dict[str, object],
    *,
    consumes_belief: bool,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Disable predictor_rollout.* targets unless the model trained belief-feeding."""
    if consumes_belief:
        return analysis_config, []

    def _is_open_loop_rollout(source_name: str, _payload: dict[str, object]) -> bool:
        return source_name.startswith("predictor_rollout.")

    return disable_targets_where(analysis_config, _is_open_loop_rollout)


def _reference_from_payload(
    payload: dict[str, object],
    *,
    default_model_artifact_id: str,
    default_dataset_artifact_id: str,
    default_dataset_artifact_type: str,
    default_split_artifact_id: str,
    default_split_name: str,
    default_source_name: str,
) -> _AnalysisSourceReference:
    label = str(
        payload.get("label")
        or f"{payload.get('model_artifact_id', default_model_artifact_id)}__"
        f"{payload.get('dataset_artifact_id', default_dataset_artifact_id)}__"
        f"{payload.get('split_name', default_split_name)}"
    )
    return _AnalysisSourceReference(
        label=label,
        source_name=str(payload.get("source", default_source_name)),
        model_artifact_id=str(payload.get("model_artifact_id", default_model_artifact_id)),
        dataset_artifact_id=str(payload.get("dataset_artifact_id", default_dataset_artifact_id)),
        dataset_artifact_type=str(
            payload.get("dataset_artifact_type", default_dataset_artifact_type)
        ),
        split_artifact_id=str(payload.get("split_artifact_id", default_split_artifact_id)),
        split_name=str(payload.get("split_name", default_split_name)),
    )


def _collection_cache_key(
    reference: _AnalysisSourceReference,
    collection_plan: _CollectionPlan,
) -> _CollectionCacheKey:
    return (
        reference.model_artifact_id,
        reference.dataset_artifact_type,
        reference.dataset_artifact_id,
        reference.split_artifact_id,
        reference.split_name,
        collection_plan.source_names,
        collection_plan.include_batch_keys,
    )


def _source_collection_group_key(
    reference: _AnalysisSourceReference,
) -> _SourceCollectionGroupKey:
    return (
        reference.model_artifact_id,
        reference.dataset_artifact_type,
        reference.dataset_artifact_id,
        reference.split_artifact_id,
        reference.split_name,
        _source_collection_sharing_key(reference.source_name),
    )


def _source_collection_sharing_key(source_name: str) -> str:
    if source_name.endswith(".place_codes"):
        return "place_codes"
    return source_name


def _collection_plans_by_source_group(
    work_items: list[_SingleAnalysisWorkItem],
) -> dict[_SourceCollectionGroupKey, _CollectionPlan]:
    collection_plans: dict[_SourceCollectionGroupKey, _CollectionPlan] = {}
    for group_key, group_work_items in _single_work_items_by_source_group(work_items):
        source_names = tuple(
            sorted({work_item.reference.source_name for work_item in group_work_items})
        )
        required_batch_keys = tuple(
            sorted(
                {
                    batch_key
                    for work_item in group_work_items
                    for batch_key in _required_batch_keys(work_item.module_names)
                }
            )
        )
        collection_plans[group_key] = _CollectionPlan(
            source_names=source_names,
            include_batch_keys=required_batch_keys,
        )
    return collection_plans


def _single_work_items_by_source_group(
    work_items: list[_SingleAnalysisWorkItem],
) -> list[tuple[_SourceCollectionGroupKey, list[_SingleAnalysisWorkItem]]]:
    grouped_items: dict[_SourceCollectionGroupKey, list[_SingleAnalysisWorkItem]] = {}
    group_order: list[_SourceCollectionGroupKey] = []
    for work_item in work_items:
        group_key = _source_collection_group_key(work_item.reference)
        if group_key not in grouped_items:
            grouped_items[group_key] = []
            group_order.append(group_key)
        grouped_items[group_key].append(work_item)
    return [(group_key, grouped_items[group_key]) for group_key in group_order]


def _resolve_analysis_source_reference(
    reference: _AnalysisSourceReference,
    *,
    registry,
) -> _AnalysisSourceReference:
    model_artifact = resolve_registry_reference(
        registry,
        "place_model",
        reference.model_artifact_id,
    )
    dataset_reference = str(reference.dataset_artifact_id).strip()
    dataset_type_reference = str(reference.dataset_artifact_type).strip()
    if dataset_reference == "auto":
        dataset_artifact = _resolve_unique_direct_input_artifact(
            registry,
            source_artifact=model_artifact,
            expected_types=("raw_dataset", "encoded_dataset"),
            label=f"analysis comparative input '{reference.label}' dataset_artifact_id",
        )
    elif ArtifactRegistry.is_tag_reference(dataset_reference):
        if dataset_type_reference:
            dataset_artifact = registry.resolve_completed_reference(
                dataset_type_reference,
                dataset_reference,
            )
        else:
            dataset_artifact = registry.require_completed(
                registry.resolve_tag(dataset_reference.removeprefix("tag:"))
            )
            if dataset_artifact.artifact_type not in {"raw_dataset", "encoded_dataset"}:
                raise ValueError(
                    f"Comparative analysis input '{reference.label}' resolved dataset tag to "
                    f"{dataset_artifact.artifact_type}, expected raw_dataset or encoded_dataset."
                )
    else:
        dataset_artifact = resolve_registry_reference(
            registry,
            dataset_type_reference,
            dataset_reference,
        )

    split_reference = str(reference.split_artifact_id).strip()
    if split_reference == "auto":
        split_artifact = _resolve_unique_direct_input_artifact(
            registry,
            source_artifact=model_artifact,
            expected_types=("split_set",),
            label=f"analysis comparative input '{reference.label}' split_artifact_id",
        )
    else:
        split_artifact = resolve_registry_reference(registry, "split_set", split_reference)
    return _AnalysisSourceReference(
        label=reference.label,
        source_name=reference.source_name,
        model_artifact_id=model_artifact.artifact_id,
        dataset_artifact_id=dataset_artifact.artifact_id,
        dataset_artifact_type=dataset_artifact.artifact_type,
        split_artifact_id=split_artifact.artifact_id,
        split_name=reference.split_name,
    )


def _build_analysis_input(
    reference: _AnalysisSourceReference,
    *,
    registry,
    device,
    model_cache: dict[str, object],
    collection_plan: _CollectionPlan,
    collection_cache: dict[_CollectionCacheKey, _CollectionCacheValue],
    batch_size: int,
    max_episodes: int | None,
    input_batch_cache: InputBatchCache | None = None,
    checkpoint_selection: str = "last",
    representation_set_directory: Path | None = None,
    allow_tf32: bool = False,
) -> AnalysisInput:
    cache_key = _collection_cache_key(reference, collection_plan)
    if cache_key not in collection_cache:
        dataset_artifact = registry.load(
            reference.dataset_artifact_type,
            reference.dataset_artifact_id,
        )
        dataset_summary = load_dataset_manifest(dataset_artifact.path)
        split_artifact = registry.load("split_set", reference.split_artifact_id)

        def collect() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            if reference.model_artifact_id not in model_cache:
                model_artifact = registry.load("place_model", reference.model_artifact_id)
                model_cache[reference.model_artifact_id], _ = load_model_checkpoint(
                    model_artifact.path,
                    device,
                    selection=checkpoint_selection,
                )
            model = model_cache[reference.model_artifact_id]
            configure_open_loop_rollout(model, collection_plan.source_names)
            return collect_representations(
                model,
                dataset_artifact.path,
                split_artifact.path,
                reference.split_name,
                list(collection_plan.source_names),
                device,
                batch_size=batch_size,
                include_batch_keys=list(collection_plan.include_batch_keys),
                max_episodes=max_episodes,
                input_batch_cache=input_batch_cache,
            )

        request = None
        if representation_set_directory is not None:
            stored_inference = read_representation_manifest(representation_set_directory)
            episode_ids = load_split_indices(split_artifact.path, reference.split_name)
            if max_episodes is not None and max_episodes > 0:
                episode_ids = episode_ids[:max_episodes]
            request = RepresentationRequest(
                place_model_artifact_id=reference.model_artifact_id,
                dataset_artifact_id=reference.dataset_artifact_id,
                dataset_artifact_type=reference.dataset_artifact_type,
                split_artifact_id=reference.split_artifact_id,
                checkpoint_selection=checkpoint_selection,
                device=stored_inference["device"],
                batch_size=stored_inference["batch_size"],
                allow_tf32=allow_tf32,
                torch_version=str(torch.__version__),
                episode_ids=episode_ids,
            )
        representations, metadata = resolve_representations(
            artifact_directory=representation_set_directory,
            split_name=reference.split_name,
            source_names=list(collection_plan.source_names),
            require_metadata_keys=[
                key for key in collection_plan.include_batch_keys if key not in {"rgb", "latent"}
            ],
            optional_metadata_keys=[
                key for key in collection_plan.include_batch_keys if key in {"rgb", "latent"}
            ],
            request=request,
            collect=collect,
        )
        metadata["env_id"] = dataset_summary.env_id
        metadata["env_kwargs"] = _dataset_env_kwargs(dataset_artifact.path, dataset_summary.env_id)
        collection_cache[cache_key] = (representations, metadata, {})

    representations, metadata, position_cache = collection_cache[cache_key]
    return AnalysisInput(
        representation=representations[reference.source_name],
        position_xy=metadata["position_xy"],
        heading=metadata.get("heading"),
        kinematics=metadata.get("kinematics"),
        actions=metadata.get("actions"),
        valid_mask=metadata["valid_steps"],
        source_name=reference.source_name,
        label=reference.label,
        split_name=reference.split_name,
        rgb=metadata.get("rgb"),
        latent=metadata.get("latent"),
        position_cache=position_cache,
        metadata={
            "model_artifact_id": reference.model_artifact_id,
            "dataset_artifact_id": reference.dataset_artifact_id,
            "split_artifact_id": reference.split_artifact_id,
            "env_id": metadata.get("env_id"),
            "env_kwargs": metadata.get("env_kwargs"),
        },
    )


def _work_item_needs_model_inference(work_item: _SingleAnalysisWorkItem) -> bool:
    return any(module_name != "dataset_coverage" for module_name in work_item.module_names)


def _dataset_env_kwargs(dataset_path: Path, env_id: str) -> dict | None:
    snapshot = dataset_path / "resolved_config.yaml"
    if not snapshot.exists():
        return None
    environment = yaml.safe_load(snapshot.read_text()).get("environment", {})
    if environment.get("env_id") != env_id:
        raise ValueError(f"Dataset geometry snapshot disagrees with manifest: {dataset_path}")
    return environment.get("env_kwargs", {})


def _analysis_max_episodes(raw_max_episodes: int) -> int | None:
    if raw_max_episodes <= 0:
        return None
    return int(raw_max_episodes)


def _build_dataset_coverage_analysis_input(
    reference: _AnalysisSourceReference,
    *,
    registry,
) -> AnalysisInput:
    dataset_artifact = registry.load(
        reference.dataset_artifact_type,
        reference.dataset_artifact_id,
    )
    dataset_summary = load_dataset_manifest(dataset_artifact.path)
    split_artifact = registry.load("split_set", reference.split_artifact_id)
    episode_ids = load_split_indices(split_artifact.path, reference.split_name)
    if not episode_ids:
        split_indices_path = split_artifact.path / "split_indices.json"
        raise ValueError(
            f"Split '{reference.split_name}' in {split_indices_path} contains no episode ids."
        )

    zarr, _ = _require_zarr()
    dataset_group = zarr.open(str(dataset_artifact.path / "dataset.zarr"), mode="r")
    position_xy = np.asarray(
        dataset_group["state"]["position_xy"][episode_ids],
        dtype=np.float32,
    )
    valid_steps = np.asarray(
        dataset_group["masks"]["valid_steps"][episode_ids],
        dtype=bool,
    )
    representation = np.zeros((*valid_steps.shape, 1), dtype=np.float32)
    return AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_steps,
        source_name=reference.source_name,
        label=reference.label,
        split_name=reference.split_name,
        metadata={
            "model_artifact_id": reference.model_artifact_id,
            "dataset_artifact_id": reference.dataset_artifact_id,
            "split_artifact_id": reference.split_artifact_id,
            "env_id": dataset_summary.env_id,
            "env_kwargs": _dataset_env_kwargs(dataset_artifact.path, dataset_summary.env_id),
        },
    )


def _required_batch_keys(module_names: list[str]) -> list[str]:
    required: set[str] = set()
    for module_name in module_names:
        module = ANALYSIS_MODULES[module_name]()
        required.update(getattr(module, "required_batch_keys", lambda: set())())
    return sorted(required)


def _required_comparative_batch_keys(analysis_payload: dict[str, object]) -> list[str]:
    module_name = str(analysis_payload.get("module", ""))
    if module_name not in COMPARATIVE_MODULES:
        raise KeyError(f"Unknown comparative analysis module: {module_name}")
    module = COMPARATIVE_MODULES[module_name]()
    return sorted(getattr(module, "required_batch_keys", lambda: set())())


def _analysis_population_coding_metrics(
    single_results: dict[str, AnalysisResult],
) -> dict[str, float]:
    """Derive the place-code quality composite from decode and rate-map outputs."""
    derived_metrics: dict[str, float] = {}
    for result_key, decode_result in single_results.items():
        if not result_key.endswith(".decode_xy"):
            continue
        target_prefix = result_key.removesuffix(".decode_xy")
        rate_map_result = single_results.get(f"{target_prefix}.rate_map_coding_purity")
        if rate_map_result is None:
            continue
        fraction = rate_map_result.metrics.get("fraction_place_cells")
        field_coverage = rate_map_result.metrics.get("field_coverage_fraction")
        decode_r2 = decode_result.metrics.get("decode_r2")
        if fraction is None or field_coverage is None or decode_r2 is None:
            continue
        quality = place_code_quality(float(decode_r2), float(fraction), float(field_coverage))
        for metric_name, value in quality.items():
            derived_metrics[f"{target_prefix}.population.{metric_name}"] = value
        derived_metrics[f"{target_prefix}.population.place_code_fraction_place_cells"] = float(
            fraction
        )
        derived_metrics[f"{target_prefix}.population.place_code_field_coverage"] = float(
            field_coverage
        )
    return derived_metrics


def _dataset_coverage_extra_splits(
    analysis_config: dict[str, object],
    *,
    default_split_name: str,
    available_splits: list[str],
) -> list[str]:
    configured_splits = analysis_config.get("dataset_coverage_extra_splits", ["train", "test"])
    if not configured_splits:
        return []
    ordered_splits = [str(split_name) for split_name in configured_splits]
    deduplicated_splits = list(dict.fromkeys(ordered_splits))
    return [
        split_name
        for split_name in deduplicated_splits
        if split_name != default_split_name and split_name in available_splits
    ]


def _analysis_output_group_name(result_key: str) -> str:
    group_name, separator, _ = result_key.partition(".")
    if separator:
        return group_name
    if result_key.startswith("comparative_"):
        return result_key
    return f"comparative_{result_key}"


def _grouped_output_file_name(source_path: Path) -> str:
    stem_parts = source_path.stem.split("__")
    if len(stem_parts) >= 3 and "." in stem_parts[1]:
        shortened_stem = "__".join([stem_parts[0], *stem_parts[2:]])
        return f"{shortened_stem}{source_path.suffix}"
    return source_path.name


def _comparative_input_payloads(
    analysis_name: str,
    analysis_payload: dict[str, object],
) -> list[dict[str, object]]:
    input_payloads = list(analysis_payload.get("inputs", []))
    if not input_payloads:
        raise ValueError(
            f"Comparative analysis '{analysis_name}' is enabled but no explicit inputs "
            "were provided."
        )
    return [dict(payload) for payload in input_payloads]


def _enabled_comparative_items(
    analysis_config: dict[str, object],
) -> list[tuple[str, dict[str, object]]]:
    enabled_items: list[tuple[str, dict[str, object]]] = []
    for analysis_name, comparative_payload in analysis_config.get("comparative", {}).items():
        if not comparative_payload.get("enabled", True):
            continue
        enabled_items.append(
            (
                str(analysis_name),
                _comparative_payload_with_analysis_defaults(
                    analysis_config,
                    dict(comparative_payload),
                ),
            )
        )
    return enabled_items


def _analysis_work_item_count(analysis_config: dict[str, object]) -> int:
    cost_order = {"light": 0, "standard": 1, "heavy": 2}
    configured_max_cost_tier = str(analysis_config["max_cost_tier"])
    total = 0
    for target_payload in analysis_config.get("targets", {}).values():
        if not target_payload.get("enabled", True):
            continue
        total += 1
        module_names = [str(module_name) for module_name in target_payload.get("modules", [])]
        for module_name in module_names:
            module = ANALYSIS_MODULES[str(module_name)]()
            if cost_order[module.cost_tier] <= cost_order[configured_max_cost_tier]:
                total += 1
    for _analysis_name, _comparative_payload in _enabled_comparative_items(analysis_config):
        total += 1
    return max(1, total)


def _analysis_work_item_count_with_coverage_splits(
    analysis_config: dict[str, object],
    *,
    default_split_name: str,
    available_splits: list[str],
) -> int:
    total = _analysis_work_item_count(analysis_config)
    coverage_extra_splits = _dataset_coverage_extra_splits(
        analysis_config,
        default_split_name=default_split_name,
        available_splits=available_splits,
    )
    if not coverage_extra_splits:
        return total
    for target_payload in analysis_config.get("targets", {}).values():
        if not target_payload.get("enabled", True):
            continue
        module_names = [str(module_name) for module_name in target_payload.get("modules", [])]
        if "dataset_coverage" in module_names:
            total += len(coverage_extra_splits) * 2
    return total


def _copy_analysis_outputs(output_dir: Path, results: dict[str, object]) -> None:
    for result_key, result in results.items():
        group_name = _analysis_output_group_name(result_key)
        figures_dir = output_dir / "figures" / group_name
        declared_destinations = getattr(result, "figure_destinations", {})
        for figure_key, source_path in getattr(result, "figures", {}).items():
            if source_path.exists():
                relative_destination = declared_destinations.get(
                    figure_key, Path(_grouped_output_file_name(source_path))
                )
                destination = figures_dir / relative_destination
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
        for source_path in getattr(result, "tables", {}).values():
            if source_path.exists():
                tables_dir = output_dir / "tables" / group_name
                tables_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, tables_dir / _grouped_output_file_name(source_path))


def _write_partial_analysis_snapshot(
    output_dir: Path,
    *,
    single_results: dict[str, AnalysisResult],
    comparative_results: dict[str, AnalysisResult],
    completed_results: dict[str, AnalysisResult],
) -> None:
    """Mirror completed analysis outputs outside the registry temp directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(
        "Partial analysis outputs copied as modules finish. "
        "A completed run publishes the canonical report at results/analysis_report.\n"
    )
    partial_summary = flatten_analysis_results(single_results, comparative_results)
    partial_summary.update(_analysis_population_coding_metrics(single_results))
    write_summary_json(output_dir / "summary.json", partial_summary)
    write_summary_csv(output_dir / "metrics.csv", partial_summary)
    _copy_analysis_outputs(output_dir, completed_results)


def _preserve_unfinished_analysis_workspace(
    output_dir: Path,
    analysis_workspace: Path,
    *,
    reason: str,
) -> None:
    """Copy raw staging outputs into the stable partial-analysis folder."""
    if not analysis_workspace.exists():
        return
    destination = output_dir / "unfinished_workspace"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(analysis_workspace, destination)
    (destination / "UNFINISHED.md").write_text(
        "\n".join(
            [
                "# Unfinished Analysis Workspace",
                "",
                f"Reason: {reason}",
                f"Original staging path: {analysis_workspace}",
                "",
                "These files were copied from the registry staging workspace before cleanup. "
                "They may be incomplete and are not a canonical analysis report.",
            ]
        )
        + "\n"
    )


def _replace_partial_analysis_with_report_link(run_directory, report_path: Path) -> None:
    partial_analysis_path = run_directory.results_dir / "partial_analysis"
    if partial_analysis_path.is_symlink() or partial_analysis_path.is_file():
        partial_analysis_path.unlink()
    elif partial_analysis_path.is_dir():
        shutil.rmtree(partial_analysis_path)
    run_directory.write_symlink("results/partial_analysis", report_path)


def _refresh_analysis_run_shortcuts(run_directory, analysis_output_dir: Path) -> None:
    figures_dir = analysis_output_dir / "figures"
    if figures_dir.exists() or figures_dir.is_symlink():
        run_directory.write_symlink("figures", figures_dir)

    unfinished_workspace = analysis_output_dir / "unfinished_workspace"
    if unfinished_workspace.exists() or unfinished_workspace.is_symlink():
        run_directory.write_symlink("staging_workspace", unfinished_workspace)
    else:
        _remove_run_shortcut(run_directory, "staging_workspace")


def _snapshot_partial_and_refresh(
    output_dir: Path,
    *,
    run_directory,
    single_results: dict[str, AnalysisResult],
    comparative_results: dict[str, AnalysisResult],
    completed_results: dict[str, AnalysisResult],
) -> None:
    """Write the partial-analysis snapshot, then refresh the run's figure/staging shortcuts."""
    _write_partial_analysis_snapshot(
        output_dir,
        single_results=single_results,
        comparative_results=comparative_results,
        completed_results=completed_results,
    )
    _refresh_analysis_run_shortcuts(run_directory, output_dir)


def _preserve_workspace_and_refresh(
    output_dir: Path,
    analysis_workspace: Path,
    *,
    run_directory,
    reason: str,
) -> None:
    """Preserve the unfinished staging workspace, then refresh the run's shortcuts."""
    _preserve_unfinished_analysis_workspace(output_dir, analysis_workspace, reason=reason)
    _refresh_analysis_run_shortcuts(run_directory, output_dir)


def _set_live_analysis_staging_shortcut(run_directory, analysis_workspace: Path) -> None:
    run_directory.write_symlink("staging_workspace", analysis_workspace)


def _remove_run_shortcut(run_directory, relative_path: str) -> None:
    shortcut_path = run_directory.path / relative_path
    if shortcut_path.is_symlink():
        shortcut_path.unlink()


_MAX_BROWSER_LINK_NAME_LENGTH = 180


def _shorten_browser_link_name(
    name: str, *, max_length: int = _MAX_BROWSER_LINK_NAME_LENGTH
) -> str:
    if len(name) <= max_length:
        return name
    digest = short_fingerprint(name, length=10)
    keep = max_length - len(digest) - 2
    if keep <= 0:
        return digest[:max_length]
    return f"{name[:keep]}__{digest}"


def _link_browser_shortcut(link_path: Path, target_path: Path) -> None:
    link_if_absent(link_path.with_name(_shorten_browser_link_name(link_path.name)), target_path)


def _write_analysis_browser_links(
    *,
    registry_root: Path,
    report_path: Path,
    report_id: str,
    run_id: str,
    model_artifact_id: str,
    dataset_artifact_id: str,
    split_name: str,
) -> None:
    analysis_root = registry_root / "reports" / "analysis"
    _link_browser_shortcut(
        analysis_root
        / "by_model"
        / f"{model_artifact_id}__{split_name}__{dataset_artifact_id}__{report_id}",
        report_path,
    )
    _link_browser_shortcut(
        analysis_root
        / "by_dataset"
        / f"{dataset_artifact_id}__{split_name}__{model_artifact_id}__{report_id}",
        report_path,
    )
    _link_browser_shortcut(
        analysis_root / "by_run" / f"{run_id}__{report_id}",
        report_path,
    )


def _write_analysis_report_readme(
    *,
    output_dir: Path,
    model_artifact_id: str,
    dataset_artifact_id: str,
    split_name: str,
    target_names: list[str],
    comparative_names: list[str],
) -> None:
    readme_lines = [
        "# Analysis Report",
        "",
        f"- model: `{model_artifact_id}`",
        f"- dataset: `{dataset_artifact_id}`",
        f"- split: `{split_name}`",
        "",
        "Contents:",
        "",
        "- `summary.json`: full metric payload",
        "- `metrics.csv`: flattened metric table",
        "- `figures/`: figures grouped by target or comparative module",
        "- `tables/`: tables grouped by target or comparative module",
        "",
    ]
    if target_names:
        readme_lines.extend(
            [
                "Single-target groups:",
                "",
                *[f"- `{name}`" for name in target_names],
                "",
            ]
        )
    if comparative_names:
        readme_lines.extend(
            [
                "Comparative groups:",
                "",
                *[f"- `comparative_{name}`" for name in comparative_names],
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n")


def run(config_path: Path, overrides: list[str]) -> dict[str, object]:
    runtime = initialize_stage_runtime(config_path, overrides, "analyze_model")
    config = runtime.config
    raw_config = runtime.raw_payload
    policies = config.policies
    registry = runtime.artifact_registry
    run_directory = runtime.run_directory
    apply_tf32_policy(config.spatial_model.training.allow_tf32)
    device = resolve_device(config.analysis.device)
    stage_log_path = run_directory.logs_dir / "stage_analyze.log"
    with managed_stage_run(
        config=config,
        run_directory=run_directory,
        stage_name="analyze_model",
        run_name=f"analyze__{config.name}__{run_directory.identity.run_id}",
        tags=stage_tags(
            "analyze_model",
            config.environment.env_id,
            base_tags=config.tracking.tags,
            study_name=config.tracking.study_name,
            variant_name=config.name,
            variant_slug=run_directory.identity.variant_slug,
            seed=config.seed.global_seed,
        ),
    ) as stage_run:
        analysis_config = _analysis_config_with_seed_defaults(
            config.to_dict()["analysis"],
            int(config.seed.global_seed),
        )
        analysis_config = _analysis_config_with_overlay_kwinners_default(
            analysis_config,
            encoder_sparsifier_type=str(config.spatial_model.sparsifier.type),
            encoder_k_fraction=float(config.spatial_model.sparsifier.k_fraction),
        )
        analysis_config = _analysis_config_with_target_order(analysis_config)
        default_model_id = resolve_registry_reference(
            registry,
            "place_model",
            resolve_stage_reference(
                config.analysis,
                "analysis",
                "model_artifact_id",
                fallback=config.reuse.place_model_artifact_id,
            ),
        ).artifact_id
        analysis_config, skipped_analysis_targets = _analysis_config_with_available_layer_targets(
            analysis_config,
            available_representations=_available_representations_from_model_artifact(
                registry,
                default_model_id,
            ),
        )
        analysis_config, skipped_open_loop_targets = (
            _disable_open_loop_targets_when_not_belief_trained(
                analysis_config,
                consumes_belief=_model_consumes_belief(registry, default_model_id),
            )
        )
        skipped_analysis_targets = [*skipped_analysis_targets, *skipped_open_loop_targets]
        raw_config = {**raw_config, "analysis": analysis_config}
        default_dataset_id, default_dataset_type = resolve_stage_dataset_reference(
            registry=registry,
            section=config.analysis,
            section_name="analysis",
            fallback_artifact_id=config.dataset.artifact_id,
            fallback_artifact_type=config.dataset.artifact_type,
            fallback_model_artifact_id=default_model_id,
        )
        default_split_id = resolve_stage_split_reference(
            registry=registry,
            section=config.analysis,
            section_name="analysis",
            fallback_artifact_id=config.splits.artifact_id,
            fallback_model_artifact_id=default_model_id,
        )
        default_split_name = config.analysis.split_name
        default_split_artifact = registry.load("split_set", default_split_id)
        available_splits = available_split_names(default_split_artifact.path)
        progress = ProgressTracker(
            ConsoleProgressReporter("analyze_model"),
            total=_analysis_work_item_count_with_coverage_splits(
                analysis_config,
                default_split_name=default_split_name,
                available_splits=available_splits,
            ),
            unit_name="work_items",
        )
        progress.emit(detail="resolve analysis plan")
        batch_size = config.analysis.batch_size
        stage_fingerprint = artifact_match_fingerprint(
            {
                "rmse_aggregation": RMSE_AGGREGATION,
                "implementation_fingerprint": package_source_fingerprint(),
                "analysis": analysis_config,
                "world_geometry_semantics": "generated_kwargs_landmarks_v1",
                "checkpoint_selection": config.policies.checkpoint_selection,
                "model_artifact_id": default_model_id,
                "dataset_artifact_id": default_dataset_id,
                "dataset_artifact_type": default_dataset_type,
                "split_artifact_id": default_split_id,
                "split_name": default_split_name,
            }
        )
        enabled_comparative_items = _enabled_comparative_items(analysis_config)
        referenced_input_artifact_ids = {default_model_id, default_dataset_id, default_split_id}
        if config.reuse.representation_set_artifact_id:
            referenced_input_artifact_ids.add(config.reuse.representation_set_artifact_id)
        for analysis_name, comparative_payload in enabled_comparative_items:
            input_payloads = _comparative_input_payloads(str(analysis_name), comparative_payload)
            for payload in input_payloads:
                reference = _resolve_analysis_source_reference(
                    _reference_from_payload(
                        payload,
                        default_model_artifact_id=default_model_id,
                        default_dataset_artifact_id=default_dataset_id,
                        default_dataset_artifact_type=default_dataset_type,
                        default_split_artifact_id=default_split_id,
                        default_split_name=default_split_name,
                        default_source_name=str(
                            comparative_payload.get("source", "encoder.place_codes")
                        ),
                    ),
                    registry=registry,
                )
                referenced_input_artifact_ids.update(
                    {
                        reference.model_artifact_id,
                        reference.dataset_artifact_id,
                        reference.split_artifact_id,
                    }
                )
        stage_run.update_config(
            {
                "stage_inputs": {
                    "model_artifact_id": default_model_id,
                    "dataset_artifact_id": default_dataset_id,
                    "split_artifact_id": default_split_id,
                    "split_name": default_split_name,
                    "analysis_targets": sorted(analysis_config.get("targets", {}).keys()),
                    "skipped_analysis_targets": skipped_analysis_targets,
                    "comparative_targets": sorted(
                        name for name, _payload in enabled_comparative_items
                    ),
                }
            }
        )
        validate_artifact_compatibility(
            registry,
            [
                CompatibilityReference(
                    label="analyze_model default inputs",
                    dataset_artifact_id=default_dataset_id,
                    dataset_artifact_type=default_dataset_type,
                    split_artifact_id=default_split_id,
                    model_artifact_id=default_model_id,
                )
            ],
        )
        matching_report = resolve_matching_artifact(
            registry,
            "analysis_report",
            policies.artifact_reuse,
            config_fingerprint_value=stage_fingerprint,
            input_artifact_ids=sorted(referenced_input_artifact_ids),
        )
        if matching_report is not None:
            flat_summary = {}
            summary_path = matching_report.path / "summary.json"
            if summary_path.exists():
                flat_summary = {
                    key: float(value) for key, value in json.loads(summary_path.read_text()).items()
                }
            run_directory.update_run_manifest(
                {
                    "status": "reused",
                    "reused_artifact_ids": [matching_report.artifact_id],
                    "summary": {
                        "analysis_report_id": matching_report.artifact_id,
                        "analysis_report_path": str(matching_report.path),
                        **flat_summary,
                    },
                }
            )
            run_directory.write_symlink("results/analysis_report", matching_report.path)
            _replace_partial_analysis_with_report_link(run_directory, matching_report.path)
            _refresh_analysis_run_shortcuts(run_directory, matching_report.path)
            if flat_summary:
                stage_run.log(flat_summary)
                emit_metrics_block(
                    "analyze_model",
                    flat_summary,
                    metadata={
                        "split": default_split_name,
                        "targets": len(analysis_config.get("targets", {})),
                        "comparative": len(enabled_comparative_items),
                        "report_id": matching_report.artifact_id,
                        "reuse_mode": "reuse_if_config_match",
                    },
                    log_path=stage_log_path,
                )
            stage_run.finalize(
                status="reused",
                summary={
                    "analysis_report_id": matching_report.artifact_id,
                    "analysis_report_path": str(matching_report.path),
                    **flat_summary,
                },
                upload_files=[
                    matching_report.path / "manifest.json",
                    matching_report.path / "resolved_config.yaml",
                    matching_report.path / "used_hyperparameters.yaml",
                    matching_report.path / "summary.json",
                    matching_report.path / "metrics.csv",
                ],
            )
            return augment_stage_result(
                runtime,
                {
                    "analysis_report_id": matching_report.artifact_id,
                    "analysis_report_path": str(matching_report.path),
                    **flat_summary,
                },
            )

        model_cache: dict[str, object] = {}
        used_input_artifact_ids = {default_model_id, default_dataset_id, default_split_id}
        if config.reuse.representation_set_artifact_id:
            used_input_artifact_ids.add(config.reuse.representation_set_artifact_id)

        registry.root.mkdir(parents=True, exist_ok=True)
        with registry.temporary_directory(
            prefix="analysis_stage_",
        ) as analysis_workspace_name:
            analysis_workspace = analysis_workspace_name
            _set_live_analysis_staging_shortcut(run_directory, analysis_workspace)

            coverage_extra_splits = _dataset_coverage_extra_splits(
                analysis_config,
                default_split_name=default_split_name,
                available_splits=available_splits,
            )
            target_work_items = build_target_work_items(
                analysis_config.get("targets", {}),
                coverage_extra_splits=coverage_extra_splits,
                default_model_id=default_model_id,
                default_dataset_id=default_dataset_id,
                default_dataset_type=default_dataset_type,
                default_split_id=default_split_id,
                default_split_name=default_split_name,
                make_reference=_AnalysisSourceReference,
                make_work_item=_SingleAnalysisWorkItem,
            )
            comparative_work_items = build_comparative_work_items(
                enabled_comparative_items,
                default_model_id=default_model_id,
                default_dataset_id=default_dataset_id,
                default_dataset_type=default_dataset_type,
                default_split_id=default_split_id,
                default_split_name=default_split_name,
                registry=registry,
                resolve_reference=_resolve_analysis_source_reference,
                reference_from_payload=_reference_from_payload,
                comparative_input_payloads=_comparative_input_payloads,
            )

            partial_analysis_dir = run_directory.results_dir / "partial_analysis"
            single_results: dict[str, AnalysisResult] = {}
            comparative_results: dict[str, AnalysisResult] = {}
            collection_plans_by_group = _collection_plans_by_source_group(target_work_items)
            analysis_max_episodes = _analysis_max_episodes(int(config.analysis.max_eval_episodes))
            _representation_set_id = config.reuse.representation_set_artifact_id
            _build_analysis_input_for_run = partial(
                _build_analysis_input,
                checkpoint_selection=config.policies.checkpoint_selection,
                allow_tf32=config.spatial_model.training.allow_tf32,
                representation_set_directory=(
                    registry.load("representation_set", _representation_set_id).path
                    if _representation_set_id
                    else None
                ),
            )
            execute_single_work_items(
                target_work_items,
                single_results=single_results,
                comparative_results=comparative_results,
                collection_plans_by_group=collection_plans_by_group,
                analysis_config=analysis_config,
                analysis_workspace=analysis_workspace,
                partial_analysis_dir=partial_analysis_dir,
                run_directory=run_directory,
                registry=registry,
                device=device,
                model_cache=model_cache,
                batch_size=batch_size,
                analysis_max_episodes=analysis_max_episodes,
                used_input_artifact_ids=used_input_artifact_ids,
                progress=progress,
                single_work_items_by_source_group=_single_work_items_by_source_group,
                work_item_needs_model_inference=_work_item_needs_model_inference,
                build_analysis_input=_build_analysis_input_for_run,
                build_dataset_coverage_analysis_input=_build_dataset_coverage_analysis_input,
                snapshot_partial_and_refresh=_snapshot_partial_and_refresh,
                preserve_workspace_and_refresh=_preserve_workspace_and_refresh,
            )
            execute_comparative_work_items(
                comparative_work_items,
                single_results=single_results,
                comparative_results=comparative_results,
                analysis_config=analysis_config,
                analysis_workspace=analysis_workspace,
                partial_analysis_dir=partial_analysis_dir,
                run_directory=run_directory,
                registry=registry,
                device=device,
                model_cache=model_cache,
                batch_size=batch_size,
                analysis_max_episodes=analysis_max_episodes,
                used_input_artifact_ids=used_input_artifact_ids,
                progress=progress,
                make_collection_plan=_CollectionPlan,
                required_comparative_batch_keys=_required_comparative_batch_keys,
                build_analysis_input=_build_analysis_input_for_run,
                snapshot_partial_and_refresh=_snapshot_partial_and_refresh,
                preserve_workspace_and_refresh=_preserve_workspace_and_refresh,
            )

            return finalize_report(
                single_results=single_results,
                comparative_results=comparative_results,
                raw_config=raw_config,
                analysis_config=analysis_config,
                enabled_comparative_items=enabled_comparative_items,
                registry=registry,
                run_directory=run_directory,
                stage_run=stage_run,
                stage_fingerprint=stage_fingerprint,
                used_input_artifact_ids=used_input_artifact_ids,
                default_model_id=default_model_id,
                default_dataset_id=default_dataset_id,
                default_dataset_type=default_dataset_type,
                default_split_id=default_split_id,
                default_split_name=default_split_name,
                stage_log_path=stage_log_path,
                runtime=runtime,
                analysis_population_coding_metrics=_analysis_population_coding_metrics,
                copy_analysis_outputs=_copy_analysis_outputs,
                write_analysis_report_readme=_write_analysis_report_readme,
                write_analysis_browser_links=_write_analysis_browser_links,
                replace_partial_analysis_with_report_link=(
                    _replace_partial_analysis_with_report_link
                ),
                refresh_analysis_run_shortcuts=_refresh_analysis_run_shortcuts,
                augment_stage_result=augment_stage_result,
                emit_metrics_block=emit_metrics_block,
            )
