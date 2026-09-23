"""Dataset collection stage."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from placecell_research.artifacts.config_snapshots import write_artifact_config_snapshots
from placecell_research.artifacts.ids import generate_artifact_id
from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.collection.collector import (
    collect_raw_dataset,
)
from placecell_research.collection.policies import resolve_policies
from placecell_research.collection.stage_support import (
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config import artifact_match_fingerprint, resolve_matching_artifact
from placecell_research.tracking import ConsoleProgressReporter


def _collection_config_snapshot(config, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    collection = dict(raw_config.get("collection", {}))
    if config.collection.policy == "continuous_random":
        collection["continuous_motion"] = asdict(config.collection.continuous_motion)
    return collection


def raw_dataset_stage_fingerprint(config, raw_config: Mapping[str, Any]) -> str:
    collection_seed = (
        config.seed.global_seed
        if config.seed.collection_seed is None
        else config.seed.collection_seed
    )
    return artifact_match_fingerprint(
        {
            "environment": raw_config.get("environment", {}),
            "collection": _collection_config_snapshot(config, raw_config),
            "collection_seed": collection_seed,
        }
    )


def run(config_path: Path, overrides: list[str]) -> dict[str, str]:
    runtime = initialize_stage_runtime(config_path, overrides, "collect_dataset")
    config = runtime.config
    raw_config = {
        **runtime.raw_payload,
        "collection": _collection_config_snapshot(config, runtime.raw_payload),
    }
    policies = resolve_policies(raw_config)
    collection_seed = (
        config.seed.global_seed
        if config.seed.collection_seed is None
        else config.seed.collection_seed
    )
    stage_fingerprint = raw_dataset_stage_fingerprint(config, raw_config)
    matching_artifact = resolve_matching_artifact(
        runtime.artifact_registry,
        "raw_dataset",
        policies.artifact_reuse,
        config_fingerprint_value=stage_fingerprint,
        input_artifact_ids=[],
    )
    if matching_artifact is not None:
        runtime.run_directory.update_run_manifest(
            {
                "status": "reused",
                "reused_artifact_ids": [matching_artifact.artifact_id],
                "summary": {
                    "raw_dataset_artifact_id": matching_artifact.artifact_id,
                    "raw_dataset_artifact_path": str(matching_artifact.path),
                },
            }
        )
        runtime.run_directory.write_symlink("results/raw_dataset", matching_artifact.path)
        return augment_stage_result(runtime, {
            "dataset.artifact_id": matching_artifact.artifact_id,
            "dataset.artifact_type": "raw_dataset",
        })
    runtime.artifact_registry.root.mkdir(parents=True, exist_ok=True)
    artifact_id = generate_artifact_id(
        "raw",
        config.environment.env_id,
        runtime.run_directory.identity.run_id,
    )
    progress_reporter = ConsoleProgressReporter("collect_dataset", stream=sys.stderr)
    with runtime.artifact_registry.temporary_directory(
        prefix="raw_dataset_",
    ) as temporary_directory:
        temp_dir = temporary_directory
        collection_result = collect_raw_dataset(
            config.environment,
            config.collection,
            temp_dir,
            collection_seed=collection_seed,
            progress_callback=progress_reporter,
        )
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="raw_dataset",
            created_by=CreatedBy(
                run_id=runtime.run_directory.identity.run_id,
                stage_name="collect_dataset",
            ),
            config_fingerprint=stage_fingerprint,
            git_commit=str(runtime.git_state.get("commit", "")),
            summary=collection_result.summary.to_dict(),
        )
        manifest.write(temp_dir / "manifest.json")
        write_artifact_config_snapshots(
            temp_dir,
            raw_config,
            stage_name="collect_dataset",
            section_names=[
                "environment",
                "collection",
                "seed",
                "policies",
                "tracking",
            ],
            extra_payload={
                "artifact_id": artifact_id,
                "collection_seed": collection_seed,
            },
        )
        destination = runtime.artifact_registry.register_directory(
            "raw_dataset",
            artifact_id,
            temp_dir,
        )
    runtime.run_directory.update_run_manifest(
        {
            "status": "completed",
            "produced_artifact_ids": [artifact_id],
            "summary": {
                "raw_dataset_artifact_id": artifact_id,
                "raw_dataset_artifact_path": str(destination),
            },
        }
    )
    runtime.run_directory.write_symlink("results/raw_dataset", destination)
    runtime.artifact_registry.mark_artifact_completed("raw_dataset", artifact_id)
    return augment_stage_result(runtime, {
        "dataset.artifact_id": artifact_id,
        "dataset.artifact_type": "raw_dataset",
    })
