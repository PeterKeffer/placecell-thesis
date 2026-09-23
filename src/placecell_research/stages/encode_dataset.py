"""Frozen encoding stage."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml

from placecell_research.artifacts.config_snapshots import write_artifact_config_snapshots
from placecell_research.artifacts.ids import generate_artifact_id
from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.collection.policies import resolve_policies
from placecell_research.collection.stage_support import (
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config import (
    artifact_match_fingerprint,
    materialize_dataclass,
    resolve_matching_artifact,
)
from placecell_research.config.schema import VisionConfig
from placecell_research.datasets.dataset import TrajectoryDataset
from placecell_research.datasets.schema import DatasetSummary
from placecell_research.datasets.staging import try_create_dataset_archive
from placecell_research.datasets.zarr_io import (
    load_dataset_manifest,
    save_dataset_zarr_streaming_atomic,
)
from placecell_research.tracking import ConsoleProgressReporter, ProgressUpdate, emit_text_block
from placecell_research.utils.cpu_budget import allocated_cpu_count, parallel_read_worker_count
from placecell_research.utils.device import resolve_device
from placecell_research.utils.timing import utc_now_iso
from placecell_research.vision.builder import (
    build_vision_model,
    iter_encoded_episodes_with_model,
    write_dataset_reconstruction_previews,
)


def encoded_dataset_stage_fingerprint(
    raw_payload: Mapping[str, Any],
    *,
    source_dataset_artifact_id: str,
    vision_encoder_artifact_id: str,
) -> str:
    dataset_config = {
        key: value
        for key, value in raw_payload.get("dataset", {}).items()
        if key not in ("artifact_id", "artifact_type")
    }
    return artifact_match_fingerprint(
        {
            "dataset": dataset_config,
            "source_dataset_artifact_id": source_dataset_artifact_id,
            "vision_encoder_artifact_id": vision_encoder_artifact_id,
        }
    )


def _should_prune_source_raw_dataset(config) -> bool:
    return config.dataset.canonicality_policy == "latent_canonical" and not bool(
        config.dataset.keep_rgb
    )


def _prune_raw_dataset_payload(raw_dataset_artifact, *, replacing_artifact_id: str) -> None:
    for relative_name in ("dataset.zarr", "previews"):
        path = raw_dataset_artifact.path / relative_name
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    manifest = raw_dataset_artifact.manifest
    manifest.metadata = {
        **manifest.metadata,
        "payload_pruned": True,
        "payload_pruned_at": utc_now_iso(),
        "replaced_by_artifact_id": replacing_artifact_id,
        "removed_entries": ["dataset.zarr", "previews"],
    }
    manifest.write(raw_dataset_artifact.path / "manifest.json")
    (raw_dataset_artifact.path / "PAYLOAD_PRUNED.txt").write_text(
        "This raw dataset artifact had its heavy RGB payload pruned after a latent-canonical "
        f"encoded dataset was published.\nEncoded replacement artifact: {replacing_artifact_id}\n"
    )


def run(config_path: Path, overrides: list[str]) -> dict[str, str]:
    runtime = initialize_stage_runtime(config_path, overrides, "encode_dataset")
    progress_reporter = ConsoleProgressReporter("encode_dataset")
    config = runtime.config
    raw_payload = runtime.raw_payload
    policies = resolve_policies(raw_payload)
    if not config.dataset.artifact_id or config.dataset.artifact_type != "raw_dataset":
        raise ValueError(
            "encode_dataset requires dataset.artifact_id pointing to a raw_dataset artifact."
        )
    encoder_artifact_id = (
        raw_payload.get("vision", {}).get("artifact_id") or config.reuse.vision_encoder_artifact_id
    )
    if not encoder_artifact_id:
        raise ValueError(
            "encode_dataset requires an explicit vision encoder artifact id via "
            "`reuse.vision_encoder_artifact_id` "
            "or pipeline injection into `vision.artifact_id`."
        )
    stage_fingerprint = encoded_dataset_stage_fingerprint(
        raw_payload,
        source_dataset_artifact_id=config.dataset.artifact_id,
        vision_encoder_artifact_id=encoder_artifact_id,
    )
    matching_artifact = resolve_matching_artifact(
        runtime.artifact_registry,
        "encoded_dataset",
        policies.artifact_reuse,
        config_fingerprint_value=stage_fingerprint,
        input_artifact_ids=[config.dataset.artifact_id, encoder_artifact_id],
    )
    if matching_artifact is not None:
        runtime.run_directory.update_run_manifest(
            {
                "status": "reused",
                "reused_artifact_ids": [matching_artifact.artifact_id],
                "summary": {
                    "encoded_dataset_artifact_id": matching_artifact.artifact_id,
                    "encoded_dataset_artifact_path": str(matching_artifact.path),
                },
            }
        )
        runtime.run_directory.write_symlink("results/encoded_dataset", matching_artifact.path)
        return augment_stage_result(
            runtime,
            {
                "dataset.artifact_id": matching_artifact.artifact_id,
                "dataset.artifact_type": "encoded_dataset",
            },
        )
    raw_dataset_artifact = runtime.artifact_registry.load("raw_dataset", config.dataset.artifact_id)
    encoder_artifact = runtime.artifact_registry.load("vision_encoder", encoder_artifact_id)
    dataset = TrajectoryDataset(raw_dataset_artifact.path, include_rgb=True, include_latent=False)
    summary = load_dataset_manifest(raw_dataset_artifact.path)
    sample_rgb = dataset[0]["rgb"]
    if sample_rgb is None:
        raise ValueError("encode_dataset requires source RGB observations.")
    if sample_rgb.ndim != 4:
        raise ValueError(
            f"Expected episodic RGB tensor shaped [T, C, H, W], got {tuple(sample_rgb.shape)}."
        )
    vision_config_path = encoder_artifact.path / "training_config.yaml"
    vision_config = (
        materialize_dataclass(VisionConfig, yaml.safe_load(vision_config_path.read_text()) or {})
        if vision_config_path.exists()
        else config.vision
    )
    model = build_vision_model(vision_config, tuple(int(value) for value in sample_rgb.shape[1:]))
    if not isinstance(model, torch.nn.Identity):
        checkpoint = torch.load(encoder_artifact.path / "weights.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    encode_settings = raw_payload.get("encode_dataset", {})
    requested_device = str(encode_settings.get("device", "auto"))
    device = resolve_device(requested_device)
    read_workers = int(encode_settings.get("read_workers", parallel_read_worker_count()))
    episode_batch_size = int(encode_settings.get("encode_episode_batch", 1))
    emit_text_block(
        "encode_dataset_runtime",
        "\n".join(
            [
                f"requested_device: {requested_device}",
                f"resolved_device: {device}",
                f"slurm_cpus_per_task: {allocated_cpu_count()}",
                f"read_workers: {read_workers}",
                f"encode_episode_batch: {episode_batch_size}",
            ]
        ),
    )
    encoded_episode_iterator = iter_encoded_episodes_with_model(
        model,
        dataset,
        device=device,
        read_workers=read_workers,
        episode_batch_size=episode_batch_size,
    )
    encode_started_at = perf_counter()

    def _emit_progress(completed: int, detail: str) -> None:
        progress_reporter(
            ProgressUpdate(
                completed=completed,
                total=int(summary.num_episodes),
                elapsed_seconds=perf_counter() - encode_started_at,
                unit_name="episodes",
                detail=detail,
            )
        )

    _emit_progress(0, "encode dataset episodes")
    try:
        first_encoded_episode = next(encoded_episode_iterator)
    except StopIteration as exc:
        raise ValueError("encode_dataset requires at least one source episode.") from exc
    _emit_progress(1, "encode dataset episodes")

    def _select_modalities(episode_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if config.dataset.keep_rgb:
            return episode_arrays
        return {key: value for key, value in episode_arrays.items() if key != "observations/rgb"}

    first_selected_episode = _select_modalities(first_encoded_episode)
    array_specs = {
        key: ((int(summary.num_episodes), *tuple(int(size) for size in value.shape)), value.dtype)
        for key, value in first_selected_episode.items()
    }
    modalities = ["latent"]
    if "observations/rgb" in first_selected_episode:
        modalities.insert(0, "rgb")
    encoded_summary = DatasetSummary(
        env_id=summary.env_id,
        num_episodes=summary.num_episodes,
        episode_length=summary.episode_length,
        num_actions=summary.num_actions,
        dataset_dt=summary.dataset_dt,
        modalities=modalities,
        episode_environment_ids=summary.episode_environment_ids,
    )
    artifact_id = raw_payload.get("dataset", {}).get("output_artifact_id") or generate_artifact_id(
        "encoded",
        summary.env_id,
        runtime.run_directory.identity.run_id,
    )
    runtime.artifact_registry.root.mkdir(parents=True, exist_ok=True)
    with runtime.artifact_registry.temporary_directory(
        prefix="encoded_dataset_",
    ) as temporary_directory:
        temp_dir = temporary_directory

        def _remaining_encoded_episodes():
            completed_episodes = 1
            for episode_arrays in encoded_episode_iterator:
                completed_episodes += 1
                _emit_progress(completed_episodes, "encode dataset episodes")
                yield episode_arrays

        save_dataset_zarr_streaming_atomic(
            temp_dir / "dataset.zarr",
            array_specs,
            chain(
                [(0, first_selected_episode)],
                (
                    (episode_index, _select_modalities(episode_arrays))
                    for episode_index, episode_arrays in enumerate(
                        _remaining_encoded_episodes(),
                        start=1,
                    )
                ),
            ),
            encoded_summary,
        )
        write_dataset_reconstruction_previews(
            model, [dataset], temp_dir / "previews", device=device
        )
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="encoded_dataset",
            created_by=CreatedBy(
                run_id=runtime.run_directory.identity.run_id, stage_name="encode_dataset"
            ),
            input_artifact_ids=[config.dataset.artifact_id, encoder_artifact_id],
            config_fingerprint=stage_fingerprint,
            git_commit=str(runtime.git_state.get("commit", "")),
            summary={
                "modalities": modalities,
                "source_dataset_artifact_id": config.dataset.artifact_id,
                "vision_encoder_artifact_id": encoder_artifact_id,
                "artifact_reuse_policy": policies.artifact_reuse,
            },
        )
        manifest.write(temp_dir / "manifest.json")
        write_artifact_config_snapshots(
            temp_dir,
            raw_payload,
            stage_name="encode_dataset",
            section_names=["dataset", "vision", "seed", "policies", "reuse", "tracking"],
            extra_payload={
                "artifact_id": artifact_id,
                "source_dataset_artifact_id": config.dataset.artifact_id,
                "vision_encoder_artifact_id": encoder_artifact_id,
            },
        )
        destination = runtime.artifact_registry.register_directory(
            "encoded_dataset", artifact_id, temp_dir
        )
    if _should_prune_source_raw_dataset(config):
        _prune_raw_dataset_payload(
            raw_dataset_artifact,
            replacing_artifact_id=artifact_id,
        )
    runtime.run_directory.update_run_manifest(
        {
            "status": "completed",
            "produced_artifact_ids": [artifact_id],
            "summary": {
                "encoded_dataset_artifact_id": artifact_id,
                "encoded_dataset_artifact_path": str(destination),
                "source_raw_dataset_pruned": _should_prune_source_raw_dataset(config),
            },
        }
    )
    runtime.run_directory.write_symlink("results/encoded_dataset", destination)
    runtime.artifact_registry.mark_artifact_completed("encoded_dataset", artifact_id)
    try_create_dataset_archive(destination, "encoded_dataset")
    return augment_stage_result(
        runtime,
        {
            "dataset.artifact_id": artifact_id,
            "dataset.artifact_type": "encoded_dataset",
            "source_raw_dataset_pruned": _should_prune_source_raw_dataset(config),
        },
    )
