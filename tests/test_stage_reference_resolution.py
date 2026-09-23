from __future__ import annotations

from pathlib import Path

from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.evaluation.runtime import (
    resolve_registry_reference,
    resolve_stage_dataset_reference,
    resolve_stage_split_reference,
)


def _write_manifest_artifact(
    artifact_root: Path,
    artifact_type: str,
    artifact_id: str,
    *,
    input_artifact_ids: list[str] | None = None,
) -> Path:
    type_directories = {
        "raw_dataset": artifact_root / "datasets" / "raw" / artifact_id,
        "encoded_dataset": artifact_root / "datasets" / "encoded" / artifact_id,
        "split_set": artifact_root / "splits" / artifact_id,
        "vision_encoder": artifact_root / "vision_encoders" / artifact_id,
        "place_model": artifact_root / "place_models" / artifact_id,
    }
    artifact_path = type_directories[artifact_type]
    artifact_path.mkdir(parents=True, exist_ok=True)
    ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        created_at="2026-03-25T10:00:00Z",
        created_by=CreatedBy(run_id=f"{artifact_id}_run", stage_name=f"test_{artifact_type}"),
        input_artifact_ids=list(input_artifact_ids or []),
        config_fingerprint="sha256:test",
        git_commit="test",
    ).write(artifact_path / "manifest.json")
    return artifact_path


def test_resolve_stage_dataset_and_split_auto_follow_place_model_lineage(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    encoded_dataset_id = "encoded_demo"
    split_id = "split_demo"
    place_model_id = "place_model_demo"
    _write_manifest_artifact(artifact_root, "encoded_dataset", encoded_dataset_id)
    _write_manifest_artifact(artifact_root, "split_set", split_id)
    _write_manifest_artifact(
        artifact_root,
        "place_model",
        place_model_id,
        input_artifact_ids=[encoded_dataset_id, split_id],
    )
    registry = ArtifactRegistry(artifact_root)

    dataset_id, dataset_type = resolve_stage_dataset_reference(
        registry=registry,
        raw_config={
            "analysis": {"model_artifact_id": place_model_id, "dataset_artifact_id": "auto"}
        },
        section_name="analysis",
        fallback_artifact_id="",
        fallback_artifact_type="encoded_dataset",
        fallback_model_artifact_id=place_model_id,
    )
    split_artifact_id = resolve_stage_split_reference(
        registry=registry,
        raw_config={"analysis": {"model_artifact_id": place_model_id, "split_artifact_id": "auto"}},
        section_name="analysis",
        fallback_artifact_id="",
        fallback_model_artifact_id=place_model_id,
    )

    assert dataset_id == encoded_dataset_id
    assert dataset_type == "encoded_dataset"
    assert split_artifact_id == split_id


def test_resolve_stage_dataset_reference_supports_tag_reference_without_explicit_type(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    encoded_dataset_id = "encoded_demo"
    encoded_dataset_path = _write_manifest_artifact(
        artifact_root, "encoded_dataset", encoded_dataset_id
    )
    registry = ArtifactRegistry(artifact_root)
    registry.write_tag("datasets/demo_latest", encoded_dataset_path)

    dataset_id, dataset_type = resolve_stage_dataset_reference(
        registry=registry,
        raw_config={"evaluation": {"dataset_artifact_id": "tag:datasets/demo_latest"}},
        section_name="evaluation",
        fallback_artifact_id="",
        fallback_artifact_type="",
        fallback_model_artifact_id="",
    )

    assert dataset_id == encoded_dataset_id
    assert dataset_type == "encoded_dataset"


def test_resolve_registry_reference_supports_tag_references(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    place_model_id = "place_model_demo"
    place_model_path = _write_manifest_artifact(artifact_root, "place_model", place_model_id)
    registry = ArtifactRegistry(artifact_root)
    registry.write_tag("models/demo_best", place_model_path)

    resolved = resolve_registry_reference(registry, "place_model", "tag:models/demo_best")

    assert resolved.artifact_id == place_model_id
