"""Canonical pipeline composition."""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from placecell_research.artifacts.registry import ArtifactRegistry, RegisteredArtifact
from placecell_research.config import (
    load_experiment_config,
    validate_experiment_config,
)
from placecell_research.config.diff import (
    build_comparison_card,
    compute_config_diff,
    compute_salient_diff,
)
from placecell_research.config.loader import (
    _load_yaml,
    _resolve_defaults,
    load_raw_config_payload,
)
from placecell_research.stages.collect_dataset import raw_dataset_stage_fingerprint
from placecell_research.stages.create_split import split_stage_fingerprint
from placecell_research.stages.encode_dataset import encoded_dataset_stage_fingerprint
from placecell_research.stages.train_vision_encoder import vision_encoder_stage_fingerprint
from placecell_research.tracking import (
    RunDirectory,
    RunIdentity,
    capture_git_state,
    generate_signature,
    generate_variant_slug,
    make_run_id,
)
from placecell_research.utils.environment_info import capture_environment_info
from placecell_research.utils.memory_watchdog import read_current_rss_mb
from placecell_research.utils.repo_paths import find_repo_root
from placecell_research.utils.seeds import SeedBundle
from placecell_research.utils.timing import utc_now_iso

STAGE_MODULES = {
    "collect_dataset": "collect_dataset",
    "create_split": "create_split",
    "train_vision_encoder": "train_vision_encoder",
    "encode_dataset": "encode_dataset",
    "train_place_model": "train_place_model",
    "collect_representations": "collect_representations",
    "evaluate_model": "evaluate_model",
    "analyze_model": "analyze_model",
}

_DIRECT_OUTPUT_LINKS = {
    "collect_dataset": [("raw_dataset", "dataset.artifact_id", "raw_dataset")],
    "create_split": [("split_set", "splits.artifact_id", "split_set")],
    "train_vision_encoder": [("vision_encoder", "vision.artifact_id", "vision_encoder")],
    "encode_dataset": [("encoded_dataset", "dataset.artifact_id", "encoded_dataset")],
    "train_place_model": [("place_model", "place_model_artifact_id", "place_model")],
    "collect_representations": [
        ("representation_set", "reuse.representation_set_artifact_id", "representation_set")
    ],
    "evaluate_model": [("evaluation_report", "evaluation_report_id", "evaluation_report")],
    "analyze_model": [("analysis_report", "analysis_report_id", "analysis_report")],
}

_PIPELINE_HANDOFF_KEYS = {
    "dataset.artifact_id",
    "dataset.artifact_type",
    "splits.artifact_id",
    "vision.artifact_id",
    "evaluation.model_artifact_id",
    "analysis.model_artifact_id",
    "reuse.representation_set_artifact_id",
}

_HUMAN_ARTIFACT_ENTRY_LINKS = (
    ("encoded_dataset", "artifacts/model_dataset"),
    ("raw_dataset", "artifacts/source_raw_dataset"),
    ("split_set", "artifacts/split"),
    ("vision_encoder", "artifacts/vision_encoder"),
    ("place_model", "artifacts/place_model"),
    ("representation_set", "artifacts/representation_set"),
    ("evaluation_report", "artifacts/evaluation_report"),
    ("analysis_report", "artifacts/analysis_report"),
)

_HUMAN_FILE_ENTRY_LINKS = (
    ("selected_checkpoint.pt", "files/selected_checkpoint.pt"),
    ("best_checkpoint.pt", "files/best_checkpoint.pt"),
    ("place_model_architecture.txt", "files/place_model_architecture.txt"),
    ("vision_encoder_architecture.txt", "files/vision_encoder_architecture.txt"),
    ("evaluation_metrics.json", "files/evaluation_metrics.json"),
    ("evaluation_metrics.csv", "files/evaluation_metrics.csv"),
    ("analysis_summary.json", "files/analysis_summary.json"),
    ("analysis_metrics.csv", "files/analysis_metrics.csv"),
    ("analysis_figures", "files/analysis_figures"),
    ("analysis_tables", "files/analysis_tables"),
)


def _load_stage_runner(stage_name: str) -> Callable[[Path, list[str]], dict[str, object] | None]:
    module_name = STAGE_MODULES.get(stage_name)
    if module_name is None:
        raise KeyError(f"Unknown stage: {stage_name}")
    module = importlib.import_module(f"placecell_research.stages.{module_name}")
    return module.run


def _stage_is_satisfied_by_explicit_inputs(stage_name: str, config) -> bool:
    dataset_artifact_id = str(config.dataset.artifact_id or "").strip()
    dataset_artifact_type = str(config.dataset.artifact_type or "").strip()
    split_artifact_id = str(config.splits.artifact_id or "").strip()
    vision_artifact_id = str(config.vision.artifact_id or "").strip()
    reused_vision_artifact_id = str(config.reuse.vision_encoder_artifact_id or "").strip()

    if stage_name == "collect_dataset":
        return bool(dataset_artifact_id)
    if stage_name == "create_split":
        return bool(split_artifact_id)
    if stage_name == "train_vision_encoder":
        return bool(
            vision_artifact_id
            or reused_vision_artifact_id
            or (dataset_artifact_id and dataset_artifact_type == "encoded_dataset")
        )
    if stage_name == "encode_dataset":
        return bool(dataset_artifact_id and dataset_artifact_type == "encoded_dataset")
    return False


def _append_handoff_override(overrides: list[str], key: str, value: str) -> None:
    normalized_value = str(value or "").strip()
    if normalized_value:
        overrides.append(f"{key}={normalized_value}")


def _find_matching_raw_dataset(
    *,
    registry: ArtifactRegistry,
    config,
) -> RegisteredArtifact | None:
    stage_fingerprint = raw_dataset_stage_fingerprint(config)
    return registry.find_matching(
        "raw_dataset",
        config_fingerprint=stage_fingerprint,
        input_artifact_ids=[],
        include_pruned=True,
    )


def _find_matching_split(
    *,
    registry: ArtifactRegistry,
    config,
    dataset_artifact_id: str,
    dataset_artifact_type: str,
) -> RegisteredArtifact | None:
    stage_fingerprint = split_stage_fingerprint(
        config,
        dataset_artifact_id=dataset_artifact_id,
        dataset_artifact_type=dataset_artifact_type,
    )
    return registry.find_matching(
        "split_set",
        config_fingerprint=stage_fingerprint,
        input_artifact_ids=[dataset_artifact_id],
    )


def _find_matching_vision_encoder(
    *,
    registry: ArtifactRegistry,
    config,
    raw_dataset_artifact_id: str,
    split_artifact_id: str,
) -> RegisteredArtifact | None:
    input_artifact_ids = [raw_dataset_artifact_id]
    if split_artifact_id:
        input_artifact_ids.append(split_artifact_id)
    stage_fingerprint = vision_encoder_stage_fingerprint(
        config,
        dataset_ids=[raw_dataset_artifact_id],
        split_artifact_id=split_artifact_id or None,
    )
    return registry.find_matching(
        "vision_encoder",
        config_fingerprint=stage_fingerprint,
        input_artifact_ids=input_artifact_ids,
    )


def _find_matching_encoded_dataset(
    *,
    registry: ArtifactRegistry,
    config,
    raw_dataset_artifact_id: str,
    vision_encoder_artifact_id: str,
) -> RegisteredArtifact | None:
    stage_fingerprint = encoded_dataset_stage_fingerprint(
        config,
        source_dataset_artifact_id=raw_dataset_artifact_id,
        vision_encoder_artifact_id=vision_encoder_artifact_id,
    )
    return registry.find_matching(
        "encoded_dataset",
        config_fingerprint=stage_fingerprint,
        input_artifact_ids=[raw_dataset_artifact_id, vision_encoder_artifact_id],
    )


def _inject_automatic_reuse_overrides(
    *,
    config_path: Path,
    active_overrides: list[str],
    registry: ArtifactRegistry,
) -> None:
    config = load_experiment_config(config_path, active_overrides)
    if config.policies.artifact_reuse != "reuse_if_config_match":
        return
    if str(config.dataset.artifact_id or "").strip():
        return
    raw_dataset = _find_matching_raw_dataset(registry=registry, config=config)
    if raw_dataset is None:
        return

    raw_dataset_id = raw_dataset.artifact_id
    split = _find_matching_split(
        registry=registry,
        config=config,
        dataset_artifact_id=raw_dataset_id,
        dataset_artifact_type="raw_dataset",
    )
    split_id = split.artifact_id if split is not None else ""
    vision = _find_matching_vision_encoder(
        registry=registry,
        config=config,
        raw_dataset_artifact_id=raw_dataset_id,
        split_artifact_id=split_id,
    )
    encoded = None
    if vision is not None:
        encoded = _find_matching_encoded_dataset(
            registry=registry,
            config=config,
            raw_dataset_artifact_id=raw_dataset_id,
            vision_encoder_artifact_id=vision.artifact_id,
        )

    if encoded is not None:
        _append_handoff_override(active_overrides, "dataset.artifact_id", encoded.artifact_id)
        _append_handoff_override(active_overrides, "dataset.artifact_type", "encoded_dataset")
    elif not bool(raw_dataset.manifest.metadata.get("payload_pruned", False)):
        _append_handoff_override(active_overrides, "dataset.artifact_id", raw_dataset_id)
        _append_handoff_override(active_overrides, "dataset.artifact_type", "raw_dataset")
    else:
        return

    if split is not None:
        _append_handoff_override(active_overrides, "splits.artifact_id", split.artifact_id)
    if vision is not None:
        _append_handoff_override(active_overrides, "vision.artifact_id", vision.artifact_id)


def find_reusable_data_overrides(config_path: Path, overrides: list[str]) -> list[str]:
    """Dataset, split and vision overrides of the finished data chain matching this config."""
    config_path = config_path.resolve()
    probe_overrides = [*overrides, "policies.artifact_reuse=reuse_if_config_match"]
    config = load_experiment_config(config_path, probe_overrides)
    registry = ArtifactRegistry(find_repo_root(config_path) / config.tracking.artifact_root)
    resolved = list(probe_overrides)
    _inject_automatic_reuse_overrides(
        config_path=config_path, active_overrides=resolved, registry=registry
    )
    found = resolved[len(probe_overrides) :]
    if not any(override.startswith("dataset.artifact_id=") for override in found):
        raise ValueError(
            f"No finished dataset matches {config_path.name}; run its data stages first."
        )
    return found


def _resolve_pinned_dataset_artifact_type(
    *,
    config_path: Path,
    active_overrides: list[str],
    registry: ArtifactRegistry,
) -> None:
    """Derive dataset.artifact_type from the registry when only dataset.artifact_id is pinned."""
    config = load_experiment_config(config_path, active_overrides)
    dataset_artifact_id = str(config.dataset.artifact_id or "").strip()
    if not dataset_artifact_id:
        return
    artifact = registry.find_by_id(dataset_artifact_id)
    if artifact is None:
        return
    actual_type = str(artifact.manifest.artifact_type or "").strip()
    declared_type = str(config.dataset.artifact_type or "").strip()
    if actual_type and actual_type != declared_type:
        _append_handoff_override(active_overrides, "dataset.artifact_type", actual_type)


def _write_pipeline_runtime(
    *,
    config_path: Path,
    config,
    raw_payload: dict[str, Any],
    run_directory: RunDirectory,
    repo_root: Path,
) -> None:
    base_payload = _resolve_defaults(config_path, _load_yaml(config_path))
    slurm_metadata = _capture_slurm_runtime_metadata(
        repo_root=repo_root,
        run_root=Path(config.tracking.run_root),
    )
    run_directory.write_run_manifest(
        {
            "stage_name": "pipeline",
            "run_id": run_directory.identity.run_id,
            "created_at": utc_now_iso(),
            "variant_name": run_directory.identity.variant_name,
            "variant_slug": run_directory.identity.variant_slug,
            "signature": run_directory.identity.signature,
            "status": "initialized",
            "git_state": capture_git_state(repo_root),
            "environment_info": capture_environment_info(),
            "slurm": slurm_metadata or None,
        },
    )
    run_directory.write_yaml("manifests/resolved_config.yaml", raw_payload)
    run_directory.write_yaml(
        "manifests/salient_diff.yaml",
        compute_salient_diff(base_payload, raw_payload),
    )
    run_directory.write_yaml(
        "manifests/config_diff.yaml",
        compute_config_diff(base_payload, raw_payload),
    )
    run_directory.write_comparison_card(
        {
            "variant_name": run_directory.identity.variant_name,
            "variant_slug": run_directory.identity.variant_slug,
            "signature": run_directory.identity.signature,
            **build_comparison_card(config.to_dict()),
        }
    )
    SeedBundle(
        global_seed=config.seed.global_seed,
        collection_seed=config.seed.collection_seed,
        split_seed=config.seed.split_seed,
        training_seed=config.seed.training_seed,
    ).write(run_directory.manifests_dir / "seed_bundle.json")
    run_directory.write_symlink("results/manifests", run_directory.manifests_dir)


APP_EVENT_PREFIX = "[placecell-app]"
APP_MANIFEST_BEGIN = "[placecell-app] === manifest_begin ==="
APP_MANIFEST_END = "[placecell-app] === manifest_end ==="


def _rss_event_fields() -> dict[str, float]:
    """Host RSS at a stage boundary, so a later OOM can be attributed to a stage."""
    rss_mb = read_current_rss_mb()
    return {} if rss_mb is None else {"rss_gib": round(rss_mb / 1024.0, 2)}


def _emit_app_event(event: str, *positional: Any, **fields: Any) -> None:
    """One-line streaming event the app can grep for during a live tail."""
    parts: list[str] = [event]
    parts.extend(str(arg) for arg in positional)
    for key, value in fields.items():
        if isinstance(value, (list, tuple)):
            rendered = json.dumps(list(value), separators=(",", ":"))
        elif isinstance(value, dict):
            rendered = json.dumps(value, separators=(",", ":"))
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    print(f"{APP_EVENT_PREFIX} " + " ".join(parts), flush=True)


_STAGE_PRODUCES: tuple[tuple[str, str, str], ...] = (
    ("collect_dataset", "raw_dataset", "raw_dataset"),
    ("create_split", "split_set", "split_set"),
    ("train_vision_encoder", "vision_encoder", "vision_encoder"),
    ("encode_dataset", "encoded_dataset", "encoded_dataset"),
    ("train_place_model", "place_model", "place_model"),
    ("collect_representations", "representation_set", "representation_set"),
    ("evaluate_model", "evaluation_report", "evaluation_report"),
    ("analyze_model", "analysis_report", "analysis_report"),
)

_INTERMEDIATE_FILES: tuple[tuple[str, str, str], ...] = (
    (
        "place_model_selected_checkpoint",
        "selected_checkpoint.pt",
        "after_train_place_model",
    ),
    ("place_model_architecture", "place_model_architecture.txt", "after_train_place_model_init"),
    ("evaluation_metrics_json", "evaluation_metrics.json", "after_evaluate_model"),
    ("evaluation_metrics_csv", "evaluation_metrics.csv", "after_evaluate_model"),
    ("analysis_summary_json", "analysis_summary.json", "after_analyze_model"),
    ("analysis_metrics_csv", "analysis_metrics.csv", "after_analyze_model"),
    ("analysis_figures", "analysis_figures", "after_analyze_model"),
    ("partial_analysis_dir", "partial_analysis", "during_analyze_model"),
)


def _describe_input_artifact(
    *,
    slot_name: str,
    artifact_id: str,
    artifact_type: str,
    config_keys: tuple[str, ...],
    registry: ArtifactRegistry,
    status_when_empty: str,
) -> dict[str, Any]:
    """Resolve one input slot to {id, type, path, exists, source}."""
    pinned_id = (artifact_id or "").strip()
    entry: dict[str, Any] = {
        "slot": slot_name,
        "config_keys": list(config_keys),
        "artifact_type": artifact_type,
        "artifact_id": pinned_id,
        "artifact_path": None,
        "manifest_path": None,
        "exists": False,
        "source": status_when_empty,
    }
    if pinned_id:
        registered = registry.find_by_id(pinned_id)
        path = registry.artifact_path(artifact_type, pinned_id)
        entry["artifact_path"] = str(path)
        entry["manifest_path"] = str(path / "manifest.json")
        entry["exists"] = registered is not None
        entry["source"] = "pinned_by_override"
    return entry


def _summarize_artifacts(
    *,
    config,
    registry: ArtifactRegistry,
    repo_root: Path,
    run_directory: RunDirectory,
) -> dict[str, Any]:
    """Inputs this run consumes and outputs this run will produce."""
    artifacts_root = repo_root / config.tracking.artifact_root
    results_dir = run_directory.results_dir
    dataset_artifact_type = str(config.dataset.artifact_type or "").strip() or "encoded_dataset"
    dataset_id = str(config.dataset.artifact_id or "").strip()
    splits_id = str(config.splits.artifact_id or "").strip()
    vision_id = str(
        config.vision.artifact_id or config.reuse.vision_encoder_artifact_id or ""
    ).strip()
    place_model_id = str(config.reuse.place_model_artifact_id or "").strip()

    inputs: dict[str, dict[str, Any]] = {
        "dataset": _describe_input_artifact(
            slot_name="dataset",
            artifact_id=dataset_id,
            artifact_type=dataset_artifact_type,
            config_keys=("dataset.artifact_id", "dataset.artifact_type"),
            registry=registry,
            status_when_empty="will_be_collected",
        ),
        "splits": _describe_input_artifact(
            slot_name="splits",
            artifact_id=splits_id,
            artifact_type="split_set",
            config_keys=("splits.artifact_id",),
            registry=registry,
            status_when_empty=(
                "implicit_via_encoded_dataset"
                if dataset_artifact_type == "encoded_dataset" and dataset_id
                else "auto_lookup_or_created"
            ),
        ),
        "vision_encoder": _describe_input_artifact(
            slot_name="vision_encoder",
            artifact_id=vision_id,
            artifact_type="vision_encoder",
            config_keys=("vision.artifact_id", "reuse.vision_encoder_artifact_id"),
            registry=registry,
            status_when_empty=(
                "implicit_via_encoded_dataset"
                if dataset_artifact_type == "encoded_dataset" and dataset_id
                else "auto_lookup_or_trained"
            ),
        ),
        "place_model": _describe_input_artifact(
            slot_name="place_model",
            artifact_id=place_model_id,
            artifact_type="place_model",
            config_keys=("reuse.place_model_artifact_id",),
            registry=registry,
            status_when_empty="will_be_trained",
        ),
    }

    stages_in_run = set(config.pipeline.stages or [])
    outputs_planned: list[dict[str, Any]] = []
    for stage_name, artifact_type, symlink_name in _STAGE_PRODUCES:
        if stage_name not in stages_in_run:
            continue
        outputs_planned.append(
            {
                "stage": stage_name,
                "artifact_type": artifact_type,
                "results_symlink": str(results_dir / symlink_name),
                "results_symlink_relative": f"results/{symlink_name}",
            }
        )

    intermediates = [
        {
            "label": label,
            "path": str(results_dir / relative_path),
            "relative_path": f"results/{relative_path}",
            "available": when,
        }
        for label, relative_path, when in _INTERMEDIATE_FILES
    ]

    return {
        "root": str(artifacts_root),
        "reuse_policy": str(getattr(config.policies, "artifact_reuse", "")),
        "training_resume": str(getattr(config.policies, "training_resume", "")),
        "inputs": inputs,
        "outputs_planned": outputs_planned,
        "intermediates": intermediates,
        "root_by_type": {
            artifact_type: str(registry.artifact_path(artifact_type, "_").parent)
            for _, artifact_type, _ in _STAGE_PRODUCES
        },
    }


def _summarize_objectives(config) -> list[dict[str, Any]]:
    """Active objectives with weight > 0, name + headline knobs."""
    summary: list[dict[str, Any]] = []
    for name, objective in (config.spatial_model.objectives or {}).items():
        weight = float(getattr(objective, "weight", 0.0))
        if weight <= 0.0:
            continue
        summary.append(
            {
                "name": name,
                "type": getattr(objective, "type", "") or name,
                "weight": weight,
                "loss_type": getattr(objective, "loss_type", ""),
                "targets": list(getattr(objective, "targets", []) or []),
                "bootstrap_source": getattr(objective, "bootstrap_source", "") or "",
            }
        )
    return summary


def _summarize_headline_config(config) -> dict[str, Any]:
    """The 'config card' the app shows on the job detail screen."""
    encoder = config.spatial_model.encoder
    sparsifier = config.spatial_model.sparsifier
    training = config.spatial_model.training
    return {
        "encoder": {
            "family": getattr(encoder, "family", ""),
            "layer_sizes": list(getattr(encoder, "layer_sizes", []) or []),
            "context_length": getattr(encoder, "context_length", None),
            "position_encoding": getattr(encoder, "position_encoding", None),
        },
        "sparsifier": {
            "type": getattr(sparsifier, "type", ""),
            "k": getattr(sparsifier, "k", None),
            "temperature": getattr(sparsifier, "temperature", None),
        },
        "training": {
            "epochs": getattr(training, "epochs", None),
            "batch_size": getattr(training, "batch_size", None),
            "learning_rate": getattr(training, "learning_rate", None),
            "mixed_precision": getattr(training, "mixed_precision", None),
            "gradient_clip_norm": getattr(training, "gradient_clip_norm", None),
        },
        "dataset": {
            "artifact_type": getattr(config.dataset, "artifact_type", "") or "",
            "artifact_id": getattr(config.dataset, "artifact_id", "") or "",
        },
        "tracking": {
            "study_name": getattr(config.tracking, "study_name", ""),
            "variant_name": config.name,
            "tags": list(getattr(config.tracking, "tags", []) or []),
            "use_wandb": getattr(config.tracking, "use_wandb", False),
            "wandb_project": getattr(config.tracking, "wandb_project", ""),
        },
        "seed": {
            "global_seed": getattr(config.seed, "global_seed", None),
            "training_seed": getattr(config.seed, "training_seed", None),
        },
    }


def _emit_app_manifest_to_stdout(
    *,
    config_path: Path,
    active_overrides: list[str],
    config,
    salient_diff: dict[str, Any],
    config_diff: dict[str, Any],
    run_directory: RunDirectory,
    repo_root: Path,
    slurm_metadata: dict[str, str],
    registry: ArtifactRegistry,
) -> None:
    """Emit a self-contained identity block to stdout for app discovery."""
    run_path = run_directory.path
    results_dir = run_directory.results_dir
    git_state = capture_git_state(repo_root)
    env_info = capture_environment_info()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "emitted_at": utc_now_iso(),
        "entrypoint": "pipeline",
        "identity": {
            "run_id": run_directory.identity.run_id,
            "variant_name": run_directory.identity.variant_name,
            "variant_slug": run_directory.identity.variant_slug,
            "signature": run_directory.identity.signature,
            "study_name": run_directory.identity.study_name,
        },
        "tags": list(getattr(config.tracking, "tags", []) or []),
        "paths": {
            "repo_root": str(repo_root),
            "config_path": str(config_path),
            "config_path_relative": _relative_or_absolute(config_path, repo_root),
            "run_dir": str(run_path),
            "manifests_dir": str(run_directory.manifests_dir),
            "results_dir": str(results_dir),
            "logs_dir": str(run_directory.logs_dir),
            "partial_analysis_dir": str(results_dir / "partial_analysis"),
            "resolved_config_yaml": str(run_directory.manifests_dir / "resolved_config.yaml"),
            "salient_diff_yaml": str(run_directory.manifests_dir / "salient_diff.yaml"),
            "config_diff_yaml": str(run_directory.manifests_dir / "config_diff.yaml"),
            "run_manifest_json": str(run_directory.manifests_dir / "run_manifest.json"),
        },
        "slurm": slurm_metadata or None,
        "overrides": list(active_overrides),
        "pipeline": {
            "stages": list(config.pipeline.stages or []),
            "stop_after_stage": getattr(config.pipeline, "stop_after_stage", None),
        },
        "config_summary": _summarize_headline_config(config),
        "objectives": _summarize_objectives(config),
        "artifacts": _summarize_artifacts(
            config=config,
            registry=registry,
            repo_root=repo_root,
            run_directory=run_directory,
        ),
        "salient_diff": salient_diff,
        "config_diff": config_diff,
        "git": git_state,
        "environment": env_info,
    }

    identity = manifest["identity"]
    stages = manifest["pipeline"]["stages"]
    summary = manifest["config_summary"]
    dataset = summary["dataset"]
    slurm_job_id = (slurm_metadata or {}).get("job_id", "")
    print(f"{APP_EVENT_PREFIX} === pipeline_start ===", flush=True)
    _emit_app_event(
        "pipeline_start",
        identity["variant_name"],
        run_id=identity["run_id"],
        slurm_job=slurm_job_id or "local",
    )
    _emit_app_event(
        "pipeline_config",
        config=manifest["paths"]["config_path_relative"],
        stages=len(stages),
        overrides=len(active_overrides),
    )
    if dataset.get("artifact_id"):
        _emit_app_event(
            "pipeline_inputs",
            dataset=dataset["artifact_id"],
            dataset_type=dataset.get("artifact_type", ""),
        )
    print(APP_MANIFEST_BEGIN, flush=True)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)
    print(APP_MANIFEST_END, flush=True)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _capture_slurm_runtime_metadata(
    *,
    repo_root: Path,
    run_root: Path,
) -> dict[str, str]:
    job_id = str(
        os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or ""
    ).strip()
    if not job_id:
        return {}
    job_name = str(os.environ.get("SLURM_JOB_NAME", "placecell_research")).strip()
    if not job_name:
        job_name = "placecell_research"
    slurm_log_dir = repo_root / run_root / "slurm_logs"
    preferred_log_path = slurm_log_dir / f"{job_name}_{job_id}.out"
    fallback_log_path = slurm_log_dir / f"placecell_research_{job_id}.out"
    chosen_log_path = preferred_log_path
    if not preferred_log_path.exists() and fallback_log_path.exists():
        chosen_log_path = fallback_log_path
    return {
        "job_id": job_id,
        "job_name": job_name,
        "log_path": str(chosen_log_path),
    }


def _link_registered_artifact(
    run_directory: RunDirectory,
    relative_path: str,
    artifact: RegisteredArtifact,
) -> None:
    run_directory.write_symlink(relative_path, artifact.path)


def _link_selected_place_model_checkpoint(
    run_directory: RunDirectory,
    checkpoint_path: Path,
    *,
    results_directory: str,
) -> None:
    run_directory.write_symlink(
        f"{results_directory}/selected_checkpoint.pt",
        checkpoint_path,
    )
    if checkpoint_path.name.startswith("weights_best_"):
        run_directory.write_symlink(
            f"{results_directory}/best_checkpoint.pt",
            checkpoint_path,
        )


def _link_direct_stage_outputs(
    *,
    run_directory: RunDirectory,
    registry: ArtifactRegistry,
    stage_name: str,
    stage_result: dict[str, object],
) -> list[RegisteredArtifact]:
    linked_artifacts: list[RegisteredArtifact] = []
    direct_artifacts_by_type: dict[str, RegisteredArtifact] = {}
    for artifact_type, result_key, link_name in _DIRECT_OUTPUT_LINKS.get(stage_name, []):
        artifact_id = stage_result.get(result_key)
        if not artifact_id:
            continue
        registered = registry.find_by_id(str(artifact_id))
        if registered is None:
            continue
        direct_artifacts_by_type[artifact_type] = registered
        _link_registered_artifact(run_directory, f"results/{link_name}", registered)
        linked_artifacts.append(registered)

    if stage_name == "train_vision_encoder":
        vision_encoder = direct_artifacts_by_type.get("vision_encoder")
        if vision_encoder is not None:
            architecture_path = vision_encoder.path / "architecture.txt"
            if architecture_path.exists():
                run_directory.write_symlink(
                    "results/vision_encoder_architecture.txt",
                    architecture_path,
                )

    if stage_name == "train_place_model" and stage_result.get("checkpoint_path"):
        checkpoint_path = Path(str(stage_result["checkpoint_path"]))
        if checkpoint_path.exists():
            _link_selected_place_model_checkpoint(
                run_directory,
                checkpoint_path,
                results_directory="results",
            )
    if stage_name == "train_place_model":
        place_model = direct_artifacts_by_type.get("place_model")
        if place_model is not None:
            architecture_path = place_model.path / "architecture.txt"
            if architecture_path.exists():
                run_directory.write_symlink(
                    "results/place_model_architecture.txt",
                    architecture_path,
                )
    if stage_name == "evaluate_model" and stage_result.get("evaluation_report_path"):
        report_path = Path(str(stage_result["evaluation_report_path"]))
        if report_path.exists():
            named_candidates = {
                "evaluation_metrics.json": report_path / "metrics.json",
                "evaluation_metrics.csv": report_path / "metrics.csv",
            }
            for link_name, candidate in named_candidates.items():
                if candidate.exists():
                    run_directory.write_symlink(f"results/{link_name}", candidate)
    if stage_name == "analyze_model" and stage_result.get("analysis_report_path"):
        report_path = Path(str(stage_result["analysis_report_path"]))
        if report_path.exists():
            named_candidates = {
                "analysis_summary.json": report_path / "summary.json",
                "analysis_metrics.csv": report_path / "metrics.csv",
            }
            for link_name, candidate in named_candidates.items():
                if candidate.exists():
                    run_directory.write_symlink(f"results/{link_name}", candidate)
            for name in ("figures", "tables"):
                candidate = report_path / name
                if candidate.exists():
                    run_directory.write_symlink(
                        f"results/analysis_{name}",
                        candidate,
                    )
    return linked_artifacts


def _link_stage_run_bundle(
    *,
    run_directory: RunDirectory,
    registry: ArtifactRegistry,
    stage_name: str,
    stage_result: dict[str, object],
) -> None:
    stage_run_path_value = stage_result.get("stage.run_path")
    if not stage_run_path_value:
        return
    stage_run_path = Path(str(stage_run_path_value))
    if not stage_run_path.exists():
        return
    stage_results_dir = run_directory.results_dir / "stages" / stage_name
    stage_results_dir.mkdir(parents=True, exist_ok=True)
    run_directory.write_symlink(f"results/stages/{stage_name}/run", stage_run_path)
    for child_name in ("logs", "manifests", "results"):
        child_path = stage_run_path / child_name
        if child_path.exists():
            run_directory.write_symlink(f"results/stages/{stage_name}/{child_name}", child_path)
    for artifact_type, result_key, link_name in _DIRECT_OUTPUT_LINKS.get(stage_name, []):
        del artifact_type
        artifact_id = stage_result.get(result_key)
        if not artifact_id:
            continue
        registered = registry.find_by_id(str(artifact_id))
        if registered is not None:
            run_directory.write_symlink(f"results/stages/{stage_name}/{link_name}", registered.path)
    if stage_name == "train_place_model" and stage_result.get("checkpoint_path"):
        checkpoint_path = Path(str(stage_result["checkpoint_path"]))
        if checkpoint_path.exists():
            _link_selected_place_model_checkpoint(
                run_directory,
                checkpoint_path,
                results_directory=f"results/stages/{stage_name}",
            )


def _link_related_lineage(
    *,
    run_directory: RunDirectory,
    registry: ArtifactRegistry,
    run_root: Path,
    root_artifacts: list[RegisteredArtifact],
) -> None:
    visited_artifact_ids: set[str] = set()
    linked_run_ids: set[str] = set()

    def _visit(artifact: RegisteredArtifact) -> None:
        if artifact.artifact_id in visited_artifact_ids:
            return
        visited_artifact_ids.add(artifact.artifact_id)
        run_directory.write_symlink(
            f"results/related/artifacts/{artifact.artifact_type}/{artifact.artifact_id}",
            artifact.path,
        )
        if artifact.manifest.created_by is not None:
            created_by = artifact.manifest.created_by
            if (
                created_by.run_id != run_directory.identity.run_id
                and created_by.run_id not in linked_run_ids
            ):
                originating_run = run_root / "by_id" / created_by.run_id
                if originating_run.exists():
                    linked_run_ids.add(created_by.run_id)
                    safe_name = f"{created_by.run_id}__{created_by.stage_name}"
                    run_directory.write_symlink(
                        f"results/related/runs/{safe_name}",
                        originating_run,
                    )
        for input_artifact_id in artifact.manifest.input_artifact_ids:
            upstream = registry.find_by_id(str(input_artifact_id))
            if upstream is not None:
                _visit(upstream)

    for artifact in root_artifacts:
        _visit(artifact)


def _write_results_index(
    run_directory: RunDirectory,
    linked_artifacts: list[RegisteredArtifact],
    completed_stages: list[str],
) -> None:
    direct_artifact_types = ", ".join(
        sorted({artifact.artifact_type for artifact in linked_artifacts})
    ) or "none"
    direct_links = sorted(
        path.name
        for path in run_directory.results_dir.iterdir()
        if path.name != "README.md" and path.parent == run_directory.results_dir
    )
    related_artifacts_dir = run_directory.results_dir / "related" / "artifacts"
    related_count = (
        sum(1 for path in related_artifacts_dir.rglob("*") if path.is_symlink())
        if related_artifacts_dir.exists()
        else 0
    )
    summary_lines = [
        "# Pipeline Results",
        "",
        "This folder is the human-facing entry point for one pipeline run.",
        "Canonical artifacts still live in `artifacts/`. These are symlinks.",
        "The `stages/` folder groups each stage run, its logs, and its main output.",
        "",
        f"- completed stages: {', '.join(completed_stages)}",
        f"- direct artifacts linked here: {direct_artifact_types}",
        f"- related lineage symlinks: {related_count}",
        "",
        "Direct entries:",
        *[f"- {name}" for name in direct_links],
        "",
        "Debug shortcuts:",
        (
            "- `place_model_architecture.txt` and "
            "`vision_encoder_architecture.txt` expose the exact built "
            "PyTorch module printouts when available."
        ),
        (
            "- Stage logs in `../logs/` also include those model printouts "
            "for debugging failed runs."
        ),
        "",
        "Related lineage:",
        "- `related/artifacts/` contains upstream and reused artifacts.",
        "- `related/runs/` contains the run folders that created them when available.",
    ]
    (run_directory.results_dir / "README.md").write_text("\n".join(summary_lines) + "\n")


def _write_open_me_first_bundle(
    run_directory: RunDirectory,
    completed_stages: list[str],
) -> Path:
    open_dir = run_directory.path / "open_me_first"
    open_dir.mkdir(parents=True, exist_ok=True)
    run_directory.write_symlink("open_me_first/config", run_directory.manifests_dir)

    stage_links = run_directory.results_dir / "stages"
    if stage_links.exists():
        run_directory.write_symlink("open_me_first/stages", stage_links)

    lineage_links = run_directory.results_dir / "related"
    if lineage_links.exists():
        run_directory.write_symlink("open_me_first/lineage", lineage_links)

    for result_entry, human_relative_path in _HUMAN_ARTIFACT_ENTRY_LINKS:
        source_path = run_directory.results_dir / result_entry
        if source_path.exists() or source_path.is_symlink():
            run_directory.write_symlink(f"open_me_first/{human_relative_path}", source_path)

    for result_entry, human_relative_path in _HUMAN_FILE_ENTRY_LINKS:
        source_path = run_directory.results_dir / result_entry
        if source_path.exists() or source_path.is_symlink():
            run_directory.write_symlink(f"open_me_first/{human_relative_path}", source_path)

    _write_open_me_first_logs_bundle(run_directory)

    overview_lines = [
        "# Open Me First",
        "",
        "This is the single folder to open for one full pipeline run.",
        "",
        f"- completed stages: {', '.join(completed_stages) if completed_stages else 'none'}",
        "- `artifacts/` contains the main dataset, split, encoder, model, and reports.",
        "- `files/` contains the most important direct files and figure folders.",
        (
            "- `logs/` contains pipeline logs, stage-log shortcuts, "
            "SLURM logs, and one combined text file."
        ),
        "- `config/` contains the resolved config, manifest, seeds, and comparison card.",
        "- `stages/` contains deeper per-stage shortcuts when you need more detail.",
        "- `lineage/` contains upstream and reused artifacts plus originating runs.",
        "",
        "If you only open one directory, open this one.",
    ]
    (open_dir / "README.md").write_text("\n".join(overview_lines) + "\n")
    return open_dir


def _write_open_me_first_logs_bundle(run_directory: RunDirectory) -> None:
    logs_bundle_dir = run_directory.path / "open_me_first" / "logs"
    if logs_bundle_dir.is_symlink():
        logs_bundle_dir.unlink()
    logs_bundle_dir.mkdir(parents=True, exist_ok=True)
    run_directory.write_symlink("open_me_first/logs/pipeline", run_directory.logs_dir)

    stage_links_dir = run_directory.results_dir / "stages"
    if stage_links_dir.exists():
        for stage_dir in sorted(path for path in stage_links_dir.iterdir() if path.is_dir()):
            stage_logs_dir = stage_dir / "logs"
            if stage_logs_dir.exists() or stage_logs_dir.is_symlink():
                run_directory.write_symlink(
                    f"open_me_first/logs/stages/{stage_dir.name}",
                    stage_logs_dir,
                )

    slurm_log_path = _resolve_pipeline_slurm_log_path(run_directory)
    if slurm_log_path is not None:
        run_directory.write_symlink(
            "open_me_first/logs/slurm_log.txt",
            slurm_log_path,
        )

    combined_log_lines: list[str] = []
    for label, log_path in _iter_combined_pipeline_log_sources(run_directory):
        if not log_path.exists():
            continue
        combined_log_lines.extend(
            [
                f"===== {label} =====",
                log_path.read_text(encoding="utf-8"),
                "",
            ]
        )
    if combined_log_lines:
        combined_logs_path = logs_bundle_dir / "all_stage_logs.txt"
        combined_logs_path.write_text(
            "\n".join(combined_log_lines).rstrip() + "\n",
            encoding="utf-8",
        )


def _resolve_pipeline_slurm_log_path(run_directory: RunDirectory) -> Path | None:
    run_manifest = run_directory.load_run_manifest()
    slurm_payload = run_manifest.get("slurm")
    if isinstance(slurm_payload, dict):
        explicit_log_path = str(slurm_payload.get("log_path", "")).strip()
        if explicit_log_path:
            return Path(explicit_log_path)
        job_id = str(slurm_payload.get("job_id", "")).strip()
        if job_id:
            job_name = str(slurm_payload.get("job_name", "placecell_research")).strip()
            if not job_name:
                job_name = "placecell_research"
            preferred_log_path = run_directory.root / "slurm_logs" / f"{job_name}_{job_id}.out"
            fallback_log_path = (
                run_directory.root / "slurm_logs" / f"placecell_research_{job_id}.out"
            )
            if preferred_log_path.exists() or not fallback_log_path.exists():
                return preferred_log_path
            return fallback_log_path
    return None


def _iter_combined_pipeline_log_sources(
    run_directory: RunDirectory,
) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for log_path in sorted(run_directory.logs_dir.glob("*.log")):
        sources.append((f"pipeline/{log_path.name}", log_path))
    stage_links_dir = run_directory.results_dir / "stages"
    if not stage_links_dir.exists():
        return sources
    for stage_dir in sorted(path for path in stage_links_dir.iterdir() if path.is_dir()):
        stage_logs_dir = stage_dir / "logs"
        if not stage_logs_dir.exists():
            continue
        for log_path in sorted(stage_logs_dir.glob("*.log")):
            sources.append((f"stage/{stage_dir.name}/{log_path.name}", log_path))
    return sources


def run(
    config_path: Path,
    overrides: list[str],
    progress_hook: Callable[[str, str], None] | None = None,
) -> dict[str, object]:
    """Run the configured stages in order and assemble one browseable pipeline bundle."""
    config_path = config_path.resolve()
    repo_root = find_repo_root(config_path)
    initial_config = load_experiment_config(config_path, overrides)
    registry = ArtifactRegistry(repo_root / initial_config.tracking.artifact_root)
    active_overrides = list(overrides)
    _inject_automatic_reuse_overrides(
        config_path=config_path,
        active_overrides=active_overrides,
        registry=registry,
    )
    _resolve_pinned_dataset_artifact_type(
        config_path=config_path,
        active_overrides=active_overrides,
        registry=registry,
    )
    raw_payload = load_raw_config_payload(config_path, active_overrides)
    config = load_experiment_config(config_path, active_overrides)
    validate_experiment_config(config)

    config_payload = config.to_dict()
    variant_slug = generate_variant_slug(
        config_payload,
        fallback_name=config.name,
    )
    pipeline_identity = RunIdentity(
        run_id=_make_pipeline_run_id(repo_root, descriptor=variant_slug),
        study_name=config.tracking.study_name,
        variant_name=config.name,
        variant_slug=variant_slug,
        signature=generate_signature(config_payload),
    )
    pipeline_run_directory = RunDirectory(
        root=repo_root / config.tracking.run_root,
        identity=pipeline_identity,
    )
    pipeline_run_directory.create()
    _write_pipeline_runtime(
        config_path=config_path,
        config=config,
        raw_payload=raw_payload,
        run_directory=pipeline_run_directory,
        repo_root=repo_root,
    )
    _emit_app_manifest_to_stdout(
        config_path=config_path,
        active_overrides=active_overrides,
        config=config,
        salient_diff=compute_salient_diff(
            _resolve_defaults(config_path, _load_yaml(config_path)),
            raw_payload,
        ),
        config_diff=compute_config_diff(
            _resolve_defaults(config_path, _load_yaml(config_path)),
            raw_payload,
        ),
        run_directory=pipeline_run_directory,
        repo_root=repo_root,
        slurm_metadata=_capture_slurm_runtime_metadata(
            repo_root=repo_root,
            run_root=Path(config.tracking.run_root),
        ),
        registry=registry,
    )

    run_root = repo_root / config.tracking.run_root
    completed_stages: list[str] = []
    skipped_stages: list[str] = []
    last_stage_result: dict[str, object] = {}
    linked_artifacts: list[RegisteredArtifact] = []
    stage_results: dict[str, dict[str, object]] = {}

    for stage_name in config.pipeline.stages:
        active_config = load_experiment_config(config_path, active_overrides)
        if (
            active_config.spatial_model.inputs.observation_source == "rgb"
            and stage_name in ("train_vision_encoder", "encode_dataset")
        ):
            skipped_stages.append(stage_name)
            stage_results[stage_name] = {"status": "skipped_rgb_observation_source"}
            _emit_app_event("stage_skip", stage_name, reason="rgb_observation_source")
            if progress_hook is not None:
                progress_hook(stage_name, "skipped")
            if config.pipeline.stop_after_stage == stage_name:
                break
            continue
        if _stage_is_satisfied_by_explicit_inputs(stage_name, active_config):
            skipped_stages.append(stage_name)
            stage_results[stage_name] = {"status": "skipped_reused_input"}
            _emit_app_event("stage_skip", stage_name, reason="reused_input")
            if progress_hook is not None:
                progress_hook(stage_name, "skipped")
            if config.pipeline.stop_after_stage == stage_name:
                break
            continue
        stage_runner = _load_stage_runner(stage_name)
        if progress_hook is not None:
            progress_hook(stage_name, "starting")
        _emit_app_event("stage_start", stage_name, **_rss_event_fields())
        stage_started_at = time.monotonic()
        stage_result = stage_runner(config_path, active_overrides) or {}
        stage_duration_sec = round(time.monotonic() - stage_started_at, 2)
        result_for_log = stage_result if isinstance(stage_result, dict) else {}
        artifact_id = result_for_log.get("artifact_id", "")
        artifact_type = result_for_log.get("artifact_type", "")
        done_fields: dict[str, Any] = {"duration_sec": stage_duration_sec, **_rss_event_fields()}
        if artifact_id:
            done_fields["artifact"] = artifact_id
            if artifact_type:
                done_fields["artifact_type"] = artifact_type
                done_fields["artifact_path"] = str(
                    registry.artifact_path(str(artifact_type), str(artifact_id))
                )
        _emit_app_event("stage_done", stage_name, **done_fields)
        stage_results[stage_name] = dict(stage_result)
        last_stage_result = dict(stage_result)
        completed_stages.append(stage_name)
        linked_artifacts.extend(
            _link_direct_stage_outputs(
                run_directory=pipeline_run_directory,
                registry=registry,
                stage_name=stage_name,
                stage_result=stage_result,
            )
        )
        _link_stage_run_bundle(
            run_directory=pipeline_run_directory,
            registry=registry,
            stage_name=stage_name,
            stage_result=stage_result,
        )
        if progress_hook is not None:
            progress_hook(stage_name, "completed")
        for key, value in stage_result.items():
            if key in _PIPELINE_HANDOFF_KEYS:
                active_overrides.append(f"{key}={value}")
        if config.pipeline.stop_after_stage == stage_name:
            break

    _link_related_lineage(
        run_directory=pipeline_run_directory,
        registry=registry,
        run_root=run_root,
        root_artifacts=linked_artifacts,
    )
    _write_results_index(pipeline_run_directory, linked_artifacts, completed_stages)
    open_me_first_path = _write_open_me_first_bundle(
        pipeline_run_directory,
        completed_stages,
    )
    pipeline_run_directory.update_run_manifest(
        {
            "status": "completed",
            "completed_stages": completed_stages,
            "skipped_stages": skipped_stages,
            "stage_results": stage_results,
            "produced_artifact_ids": [artifact.artifact_id for artifact in linked_artifacts],
        },
    )
    produced_ids = [artifact.artifact_id for artifact in linked_artifacts]
    print(f"{APP_EVENT_PREFIX} === pipeline_done ===", flush=True)
    _emit_app_event(
        "pipeline_done",
        "ok",
        completed=len(completed_stages),
        skipped=len(skipped_stages),
        artifacts=len(produced_ids),
    )
    _emit_app_event(
        "pipeline_done_data",
        run_id=pipeline_identity.run_id,
        completed_stages=completed_stages,
        skipped_stages=skipped_stages,
        produced_artifact_ids=produced_ids,
        run_dir=str(pipeline_run_directory.path),
    )
    return {
        "completed_stages": completed_stages,
        "skipped_stages": skipped_stages,
        "pipeline_run_id": pipeline_identity.run_id,
        "pipeline_run_path": str(pipeline_run_directory.path),
        "pipeline_results_path": str(pipeline_run_directory.results_dir),
        "pipeline_open_path": str(open_me_first_path),
        "slurm_log_path": str(_resolve_pipeline_slurm_log_path(pipeline_run_directory) or ""),
        **last_stage_result,
    }


def _make_pipeline_run_id(repo_root: Path, *, descriptor: str | None = None) -> str:
    configured_run_id = os.environ.get("PLACECELL_RUN_ID", "").strip()
    if configured_run_id:
        return configured_run_id
    return make_run_id(repo_root, descriptor=descriptor)
