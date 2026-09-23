"""Thin orchestration helpers for evaluation and analysis stages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from placecell_research.artifacts.config_snapshots import write_artifact_config_snapshots
from placecell_research.artifacts.ids import config_fingerprint, generate_artifact_id
from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.artifacts.registry import ArtifactRegistry, RegisteredArtifact
from placecell_research.numerics.error_metrics import RMSE_AGGREGATION
from placecell_research.tracking.run_directory import RunDirectory
from placecell_research.utils.device import resolve_device
from placecell_research.utils.source_fingerprint import package_source_fingerprint


def resolve_stage_reference(
    raw_config: dict[str, Any],
    section_name: str,
    key: str,
    fallback: str | None = None,
) -> str:
    """Read a stage-scoped explicit artifact reference."""
    section = raw_config.get(section_name, {})
    value = section.get(key)
    if value in {"", None}:
        value = fallback
    if not value:
        raise ValueError(
            f"Missing explicit `{section_name}.{key}`. This stage does not do implicit artifact "
            "discovery."
        )
    return str(value)


def resolve_registry_reference(
    registry: ArtifactRegistry,
    artifact_type: str,
    artifact_reference: str,
) -> RegisteredArtifact:
    normalized_reference = str(artifact_reference or "").strip()
    if not normalized_reference:
        raise ValueError("Artifact reference must be non-empty.")
    return registry.resolve_completed_reference(artifact_type, normalized_reference)


def _resolve_unique_direct_input_artifact(
    registry: ArtifactRegistry,
    *,
    source_artifact: RegisteredArtifact,
    expected_types: tuple[str, ...],
    label: str,
) -> RegisteredArtifact:
    matches = [
        upstream_artifact
        for input_artifact_id in source_artifact.manifest.input_artifact_ids
        if (upstream_artifact := registry.find_by_id(str(input_artifact_id))) is not None
        and upstream_artifact.artifact_type in expected_types
        and registry.is_completed(upstream_artifact)
    ]
    if not matches:
        expected_text = ", ".join(expected_types)
        raise ValueError(
            f"Could not resolve `{label}=auto` from {source_artifact.artifact_type} "
            f"'{source_artifact.artifact_id}': no direct input artifact of type {expected_text} "
            "was recorded."
        )
    if len(matches) > 1:
        matching_ids = ", ".join(sorted(artifact.artifact_id for artifact in matches))
        raise ValueError(
            f"Could not resolve `{label}=auto` from {source_artifact.artifact_type} "
            f"'{source_artifact.artifact_id}': multiple matching direct input artifacts were "
            "recorded "
            f"({matching_ids}). Set `{label}` explicitly."
        )
    return matches[0]


def resolve_stage_dataset_reference(
    *,
    registry: ArtifactRegistry,
    raw_config: dict[str, Any],
    section_name: str,
    fallback_artifact_id: str | None = None,
    fallback_artifact_type: str | None = None,
    fallback_model_artifact_id: str | None = None,
) -> tuple[str, str]:
    artifact_reference = resolve_stage_reference(
        raw_config,
        section_name,
        "dataset_artifact_id",
        fallback=fallback_artifact_id,
    )
    artifact_type_reference = raw_config.get(section_name, {}).get("dataset_artifact_type")
    if artifact_type_reference in {"", None}:
        artifact_type_reference = fallback_artifact_type
    normalized_reference = str(artifact_reference or "").strip()
    normalized_type_reference = str(artifact_type_reference or "").strip()
    if normalized_reference == "auto":
        model_reference = resolve_stage_reference(
            raw_config,
            section_name,
            "model_artifact_id",
            fallback=fallback_model_artifact_id,
        )
        model_artifact = resolve_registry_reference(registry, "place_model", model_reference)
        dataset_artifact = _resolve_unique_direct_input_artifact(
            registry,
            source_artifact=model_artifact,
            expected_types=("raw_dataset", "encoded_dataset"),
            label=f"{section_name}.dataset_artifact_id",
        )
        return dataset_artifact.artifact_id, dataset_artifact.artifact_type
    if ArtifactRegistry.is_tag_reference(normalized_reference):
        if normalized_type_reference:
            dataset_artifact = registry.resolve_completed_reference(
                normalized_type_reference,
                normalized_reference,
            )
            return dataset_artifact.artifact_id, dataset_artifact.artifact_type
        dataset_artifact = registry.require_completed(
            registry.resolve_tag(normalized_reference.removeprefix("tag:"))
        )
        if dataset_artifact.artifact_type not in {"raw_dataset", "encoded_dataset"}:
            raise ValueError(
                f"`{section_name}.dataset_artifact_id` tag resolved to "
                f"{dataset_artifact.artifact_type}, expected raw_dataset or encoded_dataset."
            )
        return dataset_artifact.artifact_id, dataset_artifact.artifact_type
    if not normalized_type_reference:
        raise ValueError(
            f"`{section_name}.dataset_artifact_type` is required when "
            f"`{section_name}.dataset_artifact_id` "
            "is an explicit artifact id."
        )
    dataset_artifact = resolve_registry_reference(
        registry, normalized_type_reference, normalized_reference
    )
    return dataset_artifact.artifact_id, dataset_artifact.artifact_type


def resolve_stage_split_reference(
    *,
    registry: ArtifactRegistry,
    raw_config: dict[str, Any],
    section_name: str,
    fallback_artifact_id: str | None = None,
    fallback_model_artifact_id: str | None = None,
) -> str:
    artifact_reference = resolve_stage_reference(
        raw_config,
        section_name,
        "split_artifact_id",
        fallback=fallback_artifact_id,
    )
    normalized_reference = str(artifact_reference).strip()
    if normalized_reference == "auto":
        model_reference = resolve_stage_reference(
            raw_config,
            section_name,
            "model_artifact_id",
            fallback=fallback_model_artifact_id,
        )
        model_artifact = resolve_registry_reference(registry, "place_model", model_reference)
        split_artifact = _resolve_unique_direct_input_artifact(
            registry,
            source_artifact=model_artifact,
            expected_types=("split_set",),
            label=f"{section_name}.split_artifact_id",
        )
        return split_artifact.artifact_id
    return resolve_registry_reference(registry, "split_set", normalized_reference).artifact_id


def publish_report(
    registry: ArtifactRegistry,
    artifact_type: str,
    summary_name: str,
    run_directory: RunDirectory,
    stage_name: str,
    config_text: str,
    input_artifact_ids: list[str],
    files_writer: Callable[[Path, str], None],
    config_fingerprint_value: str | None = None,
    raw_config: dict[str, Any] | None = None,
    active_config: dict[str, Any] | None = None,
    hyperparameter_sections: list[str] | None = None,
    hyperparameter_context: dict[str, Any] | None = None,
) -> tuple[str, Path]:
    """Create a report artifact directory, let the caller write files, then publish it."""
    artifact_id = generate_artifact_id(artifact_type, summary_name, run_directory.identity.run_id)
    with registry.temporary_directory(prefix=f"{artifact_type}_") as temporary_dir:
        temporary_path = temporary_dir / artifact_id
        temporary_path.mkdir(parents=True, exist_ok=True)
        files_writer(temporary_path, artifact_id)
        if raw_config is not None:
            write_artifact_config_snapshots(
                temporary_path,
                raw_config,
                active_config or {},
                stage_name=stage_name,
                section_names=hyperparameter_sections or [],
                extra_payload=hyperparameter_context or {"artifact_id": artifact_id},
            )
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            created_by=CreatedBy(run_id=run_directory.identity.run_id, stage_name=stage_name),
            input_artifact_ids=input_artifact_ids,
            config_fingerprint=config_fingerprint_value or config_fingerprint(config_text),
            git_commit=str(
                run_directory.load_run_manifest().get("git_state", {}).get("commit", "")
            ),
            summary={
                "stage_name": stage_name,
                "rmse_aggregation": RMSE_AGGREGATION,
                "implementation_fingerprint": package_source_fingerprint(),
            },
        )
        manifest.write(temporary_path / "manifest.json")
        final_path = registry.register_directory(artifact_type, artifact_id, temporary_path)
    return artifact_id, final_path


def resolve_stage_device(raw_config: dict[str, Any], section_name: str) -> torch.device:
    """Read a stage-scoped device override if present."""
    requested = raw_config.get(section_name, {}).get("device", "auto")
    return resolve_device(str(requested))
