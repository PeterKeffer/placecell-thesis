"""Run the model forward pass over a split once and store what it produces."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import torch

from placecell_research.artifacts.compatibility import (
    CompatibilityReference,
    validate_artifact_compatibility,
)
from placecell_research.artifacts.config_snapshots import write_artifact_config_snapshots
from placecell_research.artifacts.ids import generate_artifact_id
from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.collection.stage_support import (
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config.schema import ExperimentConfig
from placecell_research.datasets.batch_iterator import available_split_names, load_split_indices
from placecell_research.evaluation.inference import (
    _configure_open_loop_rollout,
    iter_representation_batches,
    load_model_checkpoint,
)
from placecell_research.evaluation.representation_store import (
    write_representation_batches,
    write_representation_manifest,
)
from placecell_research.evaluation.runtime import (
    resolve_registry_reference,
    resolve_stage_dataset_reference,
    resolve_stage_device,
    resolve_stage_reference,
    resolve_stage_split_reference,
)
from placecell_research.training.loop import apply_tf32_policy

_STAGE_NAME = "collect_representations"


def resolve_source_names(config: ExperimentConfig) -> list[str]:
    """Every source the downstream stages will ask for, in a stable order."""
    names: list[str] = list(config.evaluation.sources)
    for target in config.analysis.targets.values():
        enabled = getattr(target, "enabled", True)
        source = getattr(target, "source", None)
        if enabled and source:
            names.append(str(source))
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def run(config_path: Path, overrides: list[str]) -> dict[str, object]:
    runtime = initialize_stage_runtime(config_path, overrides, _STAGE_NAME)
    config = runtime.config
    raw_config = runtime.raw_payload
    registry = runtime.artifact_registry
    apply_tf32_policy(config.spatial_model.training.allow_tf32)
    device = resolve_stage_device(raw_config, "representation_collection")

    model_id = resolve_registry_reference(
        registry,
        "place_model",
        resolve_stage_reference(
            raw_config,
            "representation_collection",
            "model_artifact_id",
            fallback=config.reuse.place_model_artifact_id,
        ),
    ).artifact_id
    model_artifact = registry.load("place_model", model_id)
    dataset_id, dataset_type = resolve_stage_dataset_reference(
        registry=registry,
        raw_config=raw_config,
        section_name="representation_collection",
        fallback_artifact_id=config.dataset.artifact_id,
        fallback_artifact_type=config.dataset.artifact_type,
        fallback_model_artifact_id=model_id,
    )
    split_id = resolve_stage_split_reference(
        registry=registry,
        raw_config=raw_config,
        section_name="representation_collection",
        fallback_artifact_id=config.splits.artifact_id,
        fallback_model_artifact_id=model_id,
    )
    validate_artifact_compatibility(
        registry,
        [
            CompatibilityReference(
                label="collect_representations inputs",
                model_artifact_id=model_id,
                dataset_artifact_id=dataset_id,
                dataset_artifact_type=dataset_type,
                split_artifact_id=split_id,
            )
        ],
    )
    dataset_artifact = registry.load(dataset_type, dataset_id)
    split_artifact = registry.load("split_set", split_id)

    stage_config = config.representation_collection
    source_names = list(stage_config.sources) or resolve_source_names(config)
    available_splits = set(available_split_names(split_artifact.path))
    split_names = [name for name in stage_config.split_names if name in available_splits]
    if not split_names:
        raise ValueError(
            f"none of {stage_config.split_names} exist in {split_artifact.path}; "
            f"it holds {sorted(available_splits)}."
        )

    model, _contract = load_model_checkpoint(
        model_artifact.path, device, selection=config.policies.checkpoint_selection
    )
    _configure_open_loop_rollout(model, tuple(source_names))
    episode_ids_by_split: dict[str, list[int]] = {}
    artifact_id = generate_artifact_id(
        "representation_set", config.environment.env_id, runtime.run_directory.identity.run_id
    )
    registry.root.mkdir(parents=True, exist_ok=True)
    with registry.temporary_directory(prefix="representation_set_") as temp_dir:
        for split_name in split_names:
            episode_ids = load_split_indices(split_artifact.path, split_name)
            if stage_config.max_episodes > 0:
                episode_ids = episode_ids[: stage_config.max_episodes]
            episode_ids_by_split[split_name] = episode_ids
            batches = iter_representation_batches(
                model,
                dataset_artifact.path,
                split_artifact.path,
                split_name,
                source_names,
                device,
                stage_config.batch_size,
                include_batch_keys=list(stage_config.include_batch_keys),
                max_episodes=stage_config.max_episodes or None,
            )
            with closing(batches):
                write_representation_batches(
                    temp_dir,
                    split_name=split_name,
                    episode_count=len(episode_ids),
                    batches=batches,
                )
        write_representation_manifest(
            temp_dir,
            {
                "device": str(device).split(":")[0],
                "sources": source_names,
                "episode_ids": episode_ids_by_split,
                "torch_version": str(torch.__version__),
                "allow_tf32": config.spatial_model.training.allow_tf32,
                "split_names": split_names,
                "place_model_artifact_id": model_id,
                "dataset_artifact_id": dataset_id,
                "dataset_artifact_type": dataset_type,
                "split_artifact_id": split_id,
                "batch_size": stage_config.batch_size,
                "include_batch_keys": list(stage_config.include_batch_keys),
                "max_episodes": stage_config.max_episodes,
                "checkpoint_selection": config.policies.checkpoint_selection,
            },
        )
        ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="representation_set",
            created_by=CreatedBy(
                run_id=runtime.run_directory.identity.run_id, stage_name=_STAGE_NAME
            ),
            input_artifact_ids=[model_id, dataset_id, split_id],
            git_commit=str(runtime.git_state.get("commit", "")),
            summary={
                "device": str(device).split(":")[0],
                "sources": source_names,
                "split_names": split_names,
            },
        ).write(temp_dir / "manifest.json")
        write_artifact_config_snapshots(
            temp_dir,
            raw_config,
            stage_name=_STAGE_NAME,
            section_names=["representation_collection", "seed", "policies", "reuse", "tracking"],
            extra_payload={"artifact_id": artifact_id},
        )
        destination = registry.register_directory("representation_set", artifact_id, temp_dir)
    runtime.run_directory.update_run_manifest(
        {
            "status": "completed",
            "produced_artifact_ids": [artifact_id],
            "summary": {"representation_set_artifact_id": artifact_id},
        }
    )
    runtime.run_directory.write_symlink("results/representation_set", destination)
    registry.mark_artifact_completed("representation_set", artifact_id)
    return augment_stage_result(
        runtime,
        {
            "artifact_id": artifact_id,
            "artifact_type": "representation_set",
            "reuse.representation_set_artifact_id": artifact_id,
            "sources": source_names,
            "split_names": split_names,
        },
    )
