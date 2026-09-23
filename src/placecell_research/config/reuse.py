"""Canonical explicit artifact-reuse resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from placecell_research.artifacts.ids import config_fingerprint
from placecell_research.artifacts.registry import ArtifactRegistry, RegisteredArtifact

from .schema import ExperimentConfig

ReuseStageBehavior = Literal["train_fresh", "reuse_existing_artifact", "resume_training"]
ReuseReferenceKind = Literal["none", "artifact_id", "tag", "checkpoint_path"]


@dataclass(frozen=True, slots=True)
class ReuseTarget:
    """Resolved behavior for one reusable artifact type."""

    artifact_reference: str
    artifact_id: str
    reference_kind: ReuseReferenceKind
    stage_behavior: ReuseStageBehavior
    description: str


def artifact_match_fingerprint(payload: dict[str, object]) -> str:
    """Stable fingerprint for artifact matching based on stage-relevant inputs only."""
    return config_fingerprint(json.dumps(payload, indent=2, sort_keys=True, default=str))


def resolve_matching_artifact(
    registry: ArtifactRegistry,
    artifact_type: str,
    artifact_reuse_policy: str,
    config_fingerprint_value: str,
    input_artifact_ids: list[str],
) -> RegisteredArtifact | None:
    """Resolve the configured reuse behavior against the artifact registry."""
    if artifact_reuse_policy == "force_recompute":
        return None
    matching_artifact = registry.find_matching(
        artifact_type,
        config_fingerprint=config_fingerprint_value,
        input_artifact_ids=input_artifact_ids,
    )
    if matching_artifact is None:
        return None
    if artifact_reuse_policy == "error":
        raise FileExistsError(
            f"Found an existing {artifact_type} artifact with matching stage inputs and config: "
            f"{matching_artifact.artifact_id}. Set policies.artifact_reuse=reuse_if_config_match "
            "to reuse it "
            "or policies.artifact_reuse=force_recompute to build a new artifact explicitly."
        )
    return matching_artifact


def resolve_artifact_reference_id(
    registry: ArtifactRegistry,
    artifact_type: str,
    artifact_reference: str,
) -> str:
    normalized_reference = str(artifact_reference or "").strip()
    if not normalized_reference:
        return ""
    return registry.resolve_completed_reference(artifact_type, normalized_reference).artifact_id


def resolve_reuse_target(
    artifact_reference: str,
    training_resume: str,
    *,
    registry: ArtifactRegistry | None = None,
    artifact_type: str | None = None,
) -> ReuseTarget:
    normalized_reference = str(artifact_reference or "").strip()
    if not normalized_reference:
        return ReuseTarget(
            artifact_reference="",
            artifact_id="",
            reference_kind="none",
            stage_behavior="train_fresh",
            description="train a fresh artifact",
        )
    reference_kind: ReuseReferenceKind = (
        "tag" if ArtifactRegistry.is_tag_reference(normalized_reference) else "artifact_id"
    )
    resolved_artifact_id = normalized_reference if reference_kind == "artifact_id" else ""
    tag_resolution_is_pending = reference_kind == "tag" and (
        registry is None or artifact_type is None
    )
    if reference_kind == "tag" and not tag_resolution_is_pending:
        resolved_artifact_id = resolve_artifact_reference_id(
            registry, artifact_type, normalized_reference
        )
    if training_resume == "fresh":
        return ReuseTarget(
            artifact_reference=normalized_reference,
            artifact_id=resolved_artifact_id,
            reference_kind=reference_kind,
            stage_behavior="reuse_existing_artifact",
            description=(
                "skip training and reuse the existing artifact referenced by the configured tag"
                if tag_resolution_is_pending
                else "skip training and reuse the existing artifact resolved from the "
                "configured tag"
                if reference_kind == "tag"
                else "skip training and reuse the existing artifact"
            ),
        )
    return ReuseTarget(
        artifact_reference=normalized_reference,
        artifact_id=resolved_artifact_id,
        reference_kind=reference_kind,
        stage_behavior="resume_training",
        description=(
            "load the existing artifact referenced by the configured tag and continue training "
            "from it"
            if tag_resolution_is_pending
            else "load the existing artifact resolved from the configured tag and continue "
            "training from it"
            if reference_kind == "tag"
            else "load the existing artifact and continue training from it"
        ),
    )


def summarize_reuse(
    config: ExperimentConfig,
    *,
    artifact_registry: ArtifactRegistry | None = None,
) -> dict[str, dict[str, str]]:
    """Return a compact, compare-friendly summary of explicit reuse behavior."""
    targets = {
        "vision_encoder": resolve_reuse_target(
            config.reuse.vision_encoder_artifact_id,
            config.policies.training_resume,
            registry=artifact_registry if config.reuse.vision_encoder_artifact_id else None,
            artifact_type="vision_encoder"
            if artifact_registry and config.reuse.vision_encoder_artifact_id
            else None,
        ),
        "place_model": resolve_reuse_target(
            config.reuse.place_model_artifact_id,
            config.policies.training_resume,
            registry=artifact_registry if config.reuse.place_model_artifact_id else None,
            artifact_type="place_model"
            if artifact_registry and config.reuse.place_model_artifact_id
            else None,
        ),
    }
    summary = {
        name: {
            "artifact_reference": target.artifact_reference,
            "artifact_id": target.artifact_id,
            "reference_kind": target.reference_kind,
            "stage_behavior": target.stage_behavior,
            "description": target.description,
        }
        for name, target in targets.items()
    }
    checkpoint_path = str(config.reuse.place_model_checkpoint_path).strip()
    if checkpoint_path:
        summary["place_model"] = {
            "artifact_reference": checkpoint_path,
            "artifact_id": "",
            "reference_kind": "checkpoint_path",
            "stage_behavior": "resume_training",
            "description": "continue training from a direct recovery checkpoint",
        }
    return summary
