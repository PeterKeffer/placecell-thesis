"""Artifact compatibility validation for dataset/split/model provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import yaml

from .registry import ArtifactRegistry, RegisteredArtifact


@dataclass(frozen=True, slots=True)
class CompatibilityReference:
    """One dataset/split/model bundle that should be internally consistent."""

    label: str
    dataset_artifact_id: str
    dataset_artifact_type: str
    split_artifact_id: str | None = None
    model_artifact_id: str | None = None
    allow_model_dataset_mismatch: bool = False


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    artifact_id: str
    artifact_type: str
    vision_encoder_artifact_id: str | None
    source_dataset_artifact_id: str | None


@dataclass(frozen=True, slots=True)
class PlaceModelProvenance:
    artifact_id: str
    observation_source: str
    dataset_artifact_id: str | None
    split_artifact_id: str | None
    vision_encoder_artifact_id: str | None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def _load_yaml(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text()) or {}
    return payload if isinstance(payload, dict) else {}


def _stage_context(artifact: RegisteredArtifact) -> dict[str, Any]:
    payload = _load_yaml(artifact.path / "used_hyperparameters.yaml")
    context = payload.get("stage_context", {})
    return context if isinstance(context, dict) else {}


def _selected_sections(artifact: RegisteredArtifact) -> dict[str, Any]:
    payload = _load_yaml(artifact.path / "used_hyperparameters.yaml")
    sections = payload.get("selected_config_sections", {})
    return sections if isinstance(sections, dict) else {}


def _split_dataset_artifact_id(split_artifact: RegisteredArtifact) -> str:
    split_indices_path = split_artifact.path / "split_indices.json"
    if split_indices_path.exists():
        payload = json.loads(split_indices_path.read_text())
        dataset_artifact_id = str(payload.get("dataset_artifact_id", "")).strip()
        if dataset_artifact_id:
            return dataset_artifact_id
    for artifact_id in split_artifact.manifest.input_artifact_ids:
        candidate = str(artifact_id).strip()
        if candidate:
            return candidate
    raise ValueError(
        f"Split artifact '{split_artifact.artifact_id}' is missing dataset provenance and cannot "
        "be validated."
    )


def _dataset_provenance(
    registry: ArtifactRegistry, artifact_type: str, artifact_id: str
) -> DatasetProvenance:
    artifact = registry.load(artifact_type, artifact_id)
    if artifact_type != "encoded_dataset":
        return DatasetProvenance(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact_type,
            vision_encoder_artifact_id=None,
            source_dataset_artifact_id=None,
        )
    summary = artifact.manifest.summary if isinstance(artifact.manifest.summary, dict) else {}
    encoder_artifact_id = _optional_string(summary.get("vision_encoder_artifact_id")) or ""
    source_dataset_artifact_id = _optional_string(summary.get("source_dataset_artifact_id"))
    if not encoder_artifact_id:
        encoder_artifact_id = (
            _optional_string(_stage_context(artifact).get("vision_encoder_artifact_id")) or ""
        )
    if not encoder_artifact_id:
        for input_artifact_id in artifact.manifest.input_artifact_ids:
            upstream = registry.find_by_id(str(input_artifact_id))
            if upstream is not None and upstream.artifact_type == "vision_encoder":
                encoder_artifact_id = upstream.artifact_id
                break
    if not encoder_artifact_id:
        raise ValueError(
            f"Encoded dataset '{artifact.artifact_id}' is missing vision encoder provenance and "
            "cannot be validated."
        )
    return DatasetProvenance(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact_type,
        vision_encoder_artifact_id=encoder_artifact_id,
        source_dataset_artifact_id=source_dataset_artifact_id,
    )


def resolve_dataset_provenance(
    registry: ArtifactRegistry,
    artifact_type: str,
    artifact_id: str,
) -> DatasetProvenance:
    """Return the saved dataset provenance used by compatibility checks."""
    return _dataset_provenance(registry, artifact_type, artifact_id)


def _place_model_provenance(registry: ArtifactRegistry, artifact_id: str) -> PlaceModelProvenance:
    artifact = registry.load("place_model", artifact_id)
    context = _stage_context(artifact)
    selected_sections = _selected_sections(artifact)
    spatial_model = selected_sections.get("spatial_model", {})
    inputs = spatial_model.get("inputs", {}) if isinstance(spatial_model, dict) else {}
    selected_observation_source = _optional_string(inputs.get("observation_source"))
    summary = artifact.manifest.summary if isinstance(artifact.manifest.summary, dict) else {}
    manifest_observation_source = _optional_string(summary.get("observation_source"))
    if (
        selected_observation_source
        and manifest_observation_source
        and selected_observation_source != manifest_observation_source
    ):
        raise ValueError(
            f"Place model '{artifact.artifact_id}' has inconsistent observation-source provenance."
        )
    observation_source = selected_observation_source or manifest_observation_source or "latent"
    if observation_source not in {"latent", "rgb", "action"}:
        raise ValueError(
            f"Place model '{artifact.artifact_id}' has unsupported observation source "
            f"'{observation_source}'."
        )
    dataset_artifact_id = _optional_string(context.get("dataset_artifact_id"))
    split_artifact_id = _optional_string(context.get("split_artifact_id"))
    vision_encoder_artifact_id = _optional_string(context.get("vision_encoder_artifact_id"))

    if observation_source == "latent" and not vision_encoder_artifact_id and dataset_artifact_id:
        dataset_artifact = registry.find_by_id(dataset_artifact_id)
        if dataset_artifact is not None and dataset_artifact.artifact_type == "encoded_dataset":
            vision_encoder_artifact_id = _dataset_provenance(
                registry,
                dataset_artifact.artifact_type,
                dataset_artifact.artifact_id,
            ).vision_encoder_artifact_id
    return PlaceModelProvenance(
        artifact_id=artifact.artifact_id,
        observation_source=observation_source,
        dataset_artifact_id=dataset_artifact_id,
        split_artifact_id=split_artifact_id,
        vision_encoder_artifact_id=vision_encoder_artifact_id,
    )


def resolve_place_model_observation_source(
    registry: ArtifactRegistry,
    artifact_id: str,
) -> str:
    """Return the observation source recorded by the saved place-model artifact."""
    return _place_model_provenance(registry, artifact_id).observation_source


def validate_artifact_compatibility(
    registry: ArtifactRegistry,
    references: list[CompatibilityReference],
    *,
    allow_cross_dataset_encoder_mismatch: bool = False,
    encoder_mismatch_remedy: str = "Encode every dataset with the same vision encoder.",
) -> None:
    """Validate that referenced datasets, splits, and models are mutually compatible."""
    encoded_dataset_encoders: dict[str, str] = {}

    for reference in references:
        dataset = _dataset_provenance(
            registry,
            reference.dataset_artifact_type,
            reference.dataset_artifact_id,
        )
        if reference.split_artifact_id:
            split_artifact = registry.load("split_set", reference.split_artifact_id)
            split_dataset_artifact_id = _split_dataset_artifact_id(split_artifact)
            split_matches_requested_dataset = (
                split_dataset_artifact_id == reference.dataset_artifact_id
            )
            split_matches_encoded_source_dataset = (
                dataset.artifact_type == "encoded_dataset"
                and dataset.source_dataset_artifact_id is not None
                and split_dataset_artifact_id == dataset.source_dataset_artifact_id
            )
            if not split_matches_requested_dataset and not split_matches_encoded_source_dataset:
                raise ValueError(
                    f"{reference.label} is inconsistent: split '{reference.split_artifact_id}' "
                    "belongs to dataset "
                    f"'{split_dataset_artifact_id}', but '{reference.dataset_artifact_id}' was "
                    "requested."
                )
        if dataset.artifact_type == "encoded_dataset" and dataset.vision_encoder_artifact_id:
            encoded_dataset_encoders[reference.label] = dataset.vision_encoder_artifact_id
        if not reference.model_artifact_id or reference.allow_model_dataset_mismatch:
            continue

        model = _place_model_provenance(registry, reference.model_artifact_id)
        if model.observation_source in {"rgb", "action"}:
            continue
        if dataset.artifact_type != "encoded_dataset":
            raise ValueError(
                f"{reference.label} is incompatible: place model '{reference.model_artifact_id}' "
                "expects latent inputs, "
                f"but dataset '{reference.dataset_artifact_id}' is a {dataset.artifact_type}."
            )
        if not model.vision_encoder_artifact_id:
            raise ValueError(
                f"Place model '{reference.model_artifact_id}' is missing vision encoder provenance "
                "and cannot be "
                f"validated against dataset '{reference.dataset_artifact_id}'."
            )
        if model.vision_encoder_artifact_id != dataset.vision_encoder_artifact_id:
            raise ValueError(
                f"{reference.label} is incompatible: place model '{reference.model_artifact_id}' "
                "expects latents from "
                f"vision encoder '{model.vision_encoder_artifact_id}', but dataset "
                f"'{reference.dataset_artifact_id}' "
                f"was encoded with '{dataset.vision_encoder_artifact_id}'."
            )

    if allow_cross_dataset_encoder_mismatch or len(encoded_dataset_encoders) < 2:
        return
    unique_encoder_ids = set(encoded_dataset_encoders.values())
    if len(unique_encoder_ids) > 1:
        formatted = ", ".join(
            f"{label} -> {encoder_id}"
            for label, encoder_id in sorted(encoded_dataset_encoders.items())
        )
        raise ValueError(
            "Encoded datasets used together must share the same vision encoder. "
            f"Got: {formatted}. {encoder_mismatch_remedy}"
        )
