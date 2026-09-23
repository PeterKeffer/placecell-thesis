"""Helpers shared by stage entrypoints in the collection stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from placecell_research.artifacts.ids import config_fingerprint
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.collection.policies import load_raw_config_payload
from placecell_research.config import load_experiment_config, validate_experiment_config
from placecell_research.config.diff import build_comparison_card, compute_salient_diff
from placecell_research.config.loader import _load_yaml, _resolve_defaults
from placecell_research.config.schema import ExperimentConfig
from placecell_research.tracking import (
    RunDirectory,
    RunIdentity,
    capture_git_state,
    generate_signature,
    generate_variant_slug,
    make_run_id,
)
from placecell_research.utils.environment_info import capture_environment_info
from placecell_research.utils.repo_paths import find_repo_root
from placecell_research.utils.seeds import SeedBundle
from placecell_research.utils.timing import utc_now_iso

_OUTPUT_TAG_FIELD_BY_ARTIFACT_TYPE = {
    "vision_encoder": "vision_encoder",
    "place_model": "place_model",
}


@dataclass
class StageRuntime:
    repo_root: Path
    config_path: Path
    raw_payload: dict[str, Any]
    config: ExperimentConfig
    run_directory: RunDirectory
    artifact_registry: ArtifactRegistry
    git_state: dict[str, Any]
    config_hash: str
    salient_diff: dict[str, Any]


def augment_stage_result(runtime: StageRuntime, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the current stage run identity to a returned stage payload."""
    return {
        **payload,
        "stage.run_id": runtime.run_directory.identity.run_id,
        "stage.run_path": str(runtime.run_directory.path),
    }


def configured_output_tags(runtime: StageRuntime, artifact_type: str) -> list[str]:
    """Return normalized configured output tags for the given produced artifact type."""
    field_name = _OUTPUT_TAG_FIELD_BY_ARTIFACT_TYPE.get(artifact_type)
    if field_name is None:
        return []
    configured_values = getattr(runtime.config.tracking.output_tags, field_name, [])
    return [str(value).strip() for value in configured_values if str(value).strip()]


def apply_configured_output_tags(
    runtime: StageRuntime, artifact_type: str, artifact_path: Path
) -> list[str]:
    """Write configured output tags to the artifact registry and return the applied tags."""
    tags = configured_output_tags(runtime, artifact_type)
    for tag in tags:
        runtime.artifact_registry.write_tag(tag, artifact_path)
    return tags


def initialize_stage_runtime(
    config_path: Path,
    overrides: list[str],
    stage_name: str,
) -> StageRuntime:
    """Create run folders and load both raw and typed configs."""
    config_path = config_path.resolve()
    repo_root = find_repo_root(config_path)
    base_payload = _resolve_defaults(config_path, _load_yaml(config_path))
    raw_payload = load_raw_config_payload(config_path, overrides)
    config = load_experiment_config(config_path, overrides)
    validate_experiment_config(config)
    config_payload = config.to_dict()
    variant_slug = generate_variant_slug(
        config_payload,
        fallback_name=config.tracking.variant_name,
    )
    run_id = make_run_id(repo_root, descriptor=variant_slug)
    identity = RunIdentity(
        run_id=run_id,
        study_name=config.tracking.study_name,
        variant_name=config.tracking.variant_name,
        variant_slug=variant_slug,
        signature=generate_signature(config_payload),
    )
    run_directory = RunDirectory(root=repo_root / config.tracking.run_root, identity=identity)
    run_directory.create()
    git_state = capture_git_state(repo_root)
    salient_diff = compute_salient_diff(base_payload, raw_payload)
    SeedBundle(
        global_seed=config.seed.global_seed,
        collection_seed=config.seed.collection_seed,
        split_seed=config.seed.split_seed,
        training_seed=config.seed.training_seed,
    ).write(run_directory.manifests_dir / "seed_bundle.json")
    raw_yaml = yaml.safe_dump(raw_payload, sort_keys=False)
    config_hash = config_fingerprint(raw_yaml)
    run_directory.write_yaml("manifests/resolved_config.yaml", raw_payload)
    run_directory.write_yaml("manifests/salient_diff.yaml", salient_diff)
    run_directory.write_run_manifest(
        {
            "run_id": run_id,
            "stage_name": stage_name,
            "created_at": utc_now_iso(),
            "variant_name": identity.variant_name,
            "variant_slug": identity.variant_slug,
            "signature": identity.signature,
            "config_fingerprint": config_hash,
            "status": "initialized",
            "git_state": git_state,
            "environment_info": capture_environment_info(),
        }
    )
    run_directory.write_comparison_card(
        {
            "variant_name": identity.variant_name,
            "variant_slug": identity.variant_slug,
            "signature": identity.signature,
            **build_comparison_card(config_payload),
        }
    )
    return StageRuntime(
        repo_root=repo_root,
        config_path=config_path,
        raw_payload=raw_payload,
        config=config,
        run_directory=run_directory,
        artifact_registry=ArtifactRegistry(repo_root / config.tracking.artifact_root),
        git_state=git_state,
        config_hash=config_hash,
        salient_diff=salient_diff,
    )
