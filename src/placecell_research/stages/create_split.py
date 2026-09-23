"""Split creation stage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from placecell_research.artifacts.config_snapshots import write_artifact_config_snapshots
from placecell_research.artifacts.ids import short_fingerprint, slugify
from placecell_research.collection.stage_support import (
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config import artifact_match_fingerprint, resolve_matching_artifact
from placecell_research.config.schema import SplitPolicyConfig
from placecell_research.datasets.splits import build_split_artifact, create_split_indices
from placecell_research.datasets.zarr_io import load_dataset_manifest
from placecell_research.tracking import ConsoleProgressReporter, ProgressTracker


def _explicit_split_seed(raw_config: Mapping[str, Any]) -> int | None:
    seed_value = dict(raw_config.get("seed", {}) or {}).get("split_seed")
    return None if seed_value is None else int(seed_value)


def split_stage_fingerprint(
    raw_config: Mapping[str, Any],
    *,
    dataset_artifact_id: str,
    dataset_artifact_type: str,
) -> str:
    payload: dict[str, Any] = {
        "dataset_artifact_id": dataset_artifact_id,
        "dataset_artifact_type": dataset_artifact_type,
        "splits": raw_config.get("splits", {}),
    }
    splits_payload = dict(raw_config.get("splits", {}) or {})
    explicit_split_seed = _explicit_split_seed(raw_config)
    default_splits_seed = SplitPolicyConfig().seed
    if explicit_split_seed is not None and explicit_split_seed != int(
        splits_payload.get("seed", default_splits_seed)
    ):
        payload["split_seed_override"] = explicit_split_seed
    return artifact_match_fingerprint(payload)


def run(config_path: Path, overrides: list[str]) -> dict[str, str]:
    runtime = initialize_stage_runtime(config_path, overrides, "create_split")
    progress = ProgressTracker(
        ConsoleProgressReporter("create_split"),
        total=3,
        unit_name="steps",
    )
    progress.emit(detail="load dataset manifest")
    config = runtime.config
    raw_config = runtime.raw_payload
    policies = config.policies
    if not config.dataset.artifact_id:
        raise ValueError(
            "create_split requires dataset.artifact_id to reference a dataset artifact."
        )
    dataset_type = config.dataset.artifact_type
    if dataset_type not in {"raw_dataset", "encoded_dataset"}:
        raise ValueError(f"Split stage does not support dataset type {dataset_type}.")
    stage_fingerprint = split_stage_fingerprint(
        raw_config,
        dataset_artifact_id=config.dataset.artifact_id,
        dataset_artifact_type=dataset_type,
    )
    matching_artifact = resolve_matching_artifact(
        runtime.artifact_registry,
        "split_set",
        policies.artifact_reuse,
        config_fingerprint_value=stage_fingerprint,
        input_artifact_ids=[config.dataset.artifact_id],
    )
    if matching_artifact is not None:
        runtime.run_directory.update_run_manifest(
            {
                "status": "reused",
                "reused_artifact_ids": [matching_artifact.artifact_id],
                "summary": {
                    "split_artifact_id": matching_artifact.artifact_id,
                    "split_artifact_path": str(matching_artifact.path),
                },
            }
        )
        runtime.run_directory.write_symlink("results/split_set", matching_artifact.path)
        progress.advance(3, detail="reuse existing split artifact")
        return augment_stage_result(runtime, {"splits.artifact_id": matching_artifact.artifact_id})
    dataset_artifact = runtime.artifact_registry.load(dataset_type, config.dataset.artifact_id)
    dataset_summary = load_dataset_manifest(dataset_artifact.path)
    progress.advance(detail="create split indices")
    split_seed = config.splits.seed if config.seed.split_seed is None else config.seed.split_seed
    split_id = (
        f"split_{slugify(dataset_summary.env_id)}_"
        f"{config.splits.strategy}_seed{split_seed}_"
        f"{short_fingerprint(config.dataset.artifact_id, str(split_seed), config.splits.strategy)}"
    )
    split_indices = create_split_indices(
        split_id=split_id,
        dataset_artifact_id=config.dataset.artifact_id,
        strategy=config.splits.strategy,
        seed=split_seed,
        num_episodes=dataset_summary.num_episodes,
        constraints=config.splits.constraints,
        train_fraction=config.splits.train_fraction,
        validation_fraction=config.splits.validation_fraction,
        test_fraction=config.splits.test_fraction,
        manual_ids={
            "train": config.splits.train_episode_ids,
            "validation": config.splits.validation_episode_ids,
            "test": config.splits.test_episode_ids,
        },
    )
    runtime.artifact_registry.root.mkdir(parents=True, exist_ok=True)
    with runtime.artifact_registry.temporary_directory(
        prefix="split_",
    ) as temporary_directory:
        temp_dir = temporary_directory
        manifest = build_split_artifact(
            output_dir=temp_dir,
            split_indices=split_indices,
            run_id=runtime.run_directory.identity.run_id,
            config_fingerprint=stage_fingerprint,
            git_commit=str(runtime.git_state.get("commit", "")),
        )
        write_artifact_config_snapshots(
            temp_dir,
            raw_config,
            config.to_dict(),
            stage_name="create_split",
            section_names=["dataset", "splits", "seed", "policies", "tracking"],
            extra_payload={
                "artifact_id": manifest.artifact_id,
                "dataset_artifact_id": config.dataset.artifact_id,
                "dataset_artifact_type": dataset_type,
            },
        )
        destination = runtime.artifact_registry.register_directory(
            "split_set", manifest.artifact_id, temp_dir
        )
    progress.advance(detail="publish split artifact")
    runtime.run_directory.update_run_manifest(
        {
            "status": "completed",
            "produced_artifact_ids": [manifest.artifact_id],
            "summary": {
                "split_artifact_id": manifest.artifact_id,
                "split_artifact_path": str(destination),
            },
        }
    )
    runtime.run_directory.write_symlink("results/split_set", destination)
    runtime.artifact_registry.mark_artifact_completed("split_set", manifest.artifact_id)
    progress.advance(detail="done")
    return augment_stage_result(runtime, {"splits.artifact_id": manifest.artifact_id})
