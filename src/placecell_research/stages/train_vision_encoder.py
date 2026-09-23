"""Vision encoder training stage."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from placecell_research.artifacts.config_snapshots import write_artifact_config_snapshots
from placecell_research.artifacts.ids import generate_artifact_id
from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.collection.stage_support import (
    apply_configured_output_tags,
    augment_stage_result,
    initialize_stage_runtime,
)
from placecell_research.config import (
    artifact_match_fingerprint,
    resolve_artifact_reference_id,
    resolve_matching_artifact,
    resolve_reuse_target,
    summarize_reuse,
)
from placecell_research.datasets.dataset import TrajectoryDataset
from placecell_research.datasets.splits import create_split_indices
from placecell_research.tracking import (
    ConsoleProgressReporter,
    emit_metrics_block,
    emit_text_block,
    managed_stage_run,
    stage_tags,
)
from placecell_research.utils.device import resolve_device
from placecell_research.utils.seeds import seed_everything
from placecell_research.vision.artifacts import export_effective_vision_config_payload
from placecell_research.vision.builder import train_vision_model

VISION_ARCHITECTURE_FINGERPRINT = "conv_projection_flatten_v3"


def _prefix_reconstruction_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def vision_encoder_stage_fingerprint(
    config,
    *,
    dataset_ids: list[str],
    split_artifact_id: str | None,
) -> str:
    vision_payload = asdict(config.vision)
    for recipe in vision_payload["datasets"]:
        if not recipe["split_id"]:
            recipe.pop("split_id")
    stage_fingerprint_payload = {
        "vision": vision_payload,
        "dataset_artifact_ids": sorted(dataset_ids),
        "vision_architecture": VISION_ARCHITECTURE_FINGERPRINT,
    }
    if split_artifact_id is not None:
        stage_fingerprint_payload["split_artifact_id"] = split_artifact_id
    else:
        stage_fingerprint_payload["implicit_split"] = {
            "strategy": config.splits.strategy,
            "seed": config.splits.seed,
            "train_fraction": config.splits.train_fraction,
            "validation_fraction": config.splits.validation_fraction,
            "test_fraction": config.splits.test_fraction,
            "train_episode_ids": config.splits.train_episode_ids,
            "validation_episode_ids": config.splits.validation_episode_ids,
            "test_episode_ids": config.splits.test_episode_ids,
            "constraints": config.splits.constraints,
        }
    if len(dataset_ids) > 1:
        stage_fingerprint_payload["multi_dataset_split_semantics"] = "episode_splits_v1"
        stage_fingerprint_payload["vision_training_seed"] = int(config.seed.global_seed)
    return artifact_match_fingerprint(stage_fingerprint_payload)


def _limit_vision_training_dataset(
    dataset: TrajectoryDataset,
    *,
    max_episodes: int,
) -> TrajectoryDataset:
    if max_episodes <= 0 or len(dataset) <= max_episodes:
        return dataset
    limited_episode_ids = list(dataset.iter_episode_ids())[:max_episodes]
    return TrajectoryDataset(
        dataset.dataset_dir,
        episode_ids=limited_episode_ids,
        include_rgb=dataset.include_rgb,
        include_latent=dataset.include_latent,
    )


def _vision_dataset_from_artifact(
    artifact_path: Path,
    *,
    episode_ids: list[int] | None,
) -> TrajectoryDataset:
    return TrajectoryDataset(
        artifact_path,
        episode_ids=episode_ids,
        include_rgb=True,
        include_latent=False,
    )


def _default_vision_split_payload(
    config,
    *,
    dataset_artifact_id: str,
    num_episodes: int,
) -> dict[str, object]:
    split_indices = create_split_indices(
        split_id=f"implicit_train_vision_{dataset_artifact_id}",
        dataset_artifact_id=dataset_artifact_id,
        strategy=config.splits.strategy,
        seed=int(config.splits.seed),
        num_episodes=num_episodes,
        constraints=dict(config.splits.constraints),
        train_fraction=float(config.splits.train_fraction),
        validation_fraction=float(config.splits.validation_fraction),
        test_fraction=float(config.splits.test_fraction),
        manual_ids={
            "train": config.splits.train_episode_ids,
            "validation": config.splits.validation_episode_ids,
            "test": config.splits.test_episode_ids,
        },
    )
    return split_indices.to_dict()


def _newest_raw_dataset_for_environment(artifact_registry, environment: str) -> str:
    """Resolve environment to the newest raw_<environment>_* artifact id."""
    candidates = [
        artifact.artifact_id
        for artifact in artifact_registry.iter_artifacts("raw_dataset")
        if artifact.artifact_id.startswith(f"raw_{environment}_")
    ]
    if not candidates:
        raise ValueError(
            f"vision.datasets: no raw dataset found for environment '{environment}' "
            f"(looked for raw_{environment}_*). Collect one first or pin an artifact_id."
        )
    resolved = sorted(candidates)[-1]
    print(f"[train_vision_encoder] environment '{environment}' -> {resolved}")
    return resolved


def _resolve_vision_training_datasets(
    runtime,
    *,
    dataset_ids: list[str],
    split_artifact_id: str | None = None,
) -> tuple[list[TrajectoryDataset], list[TrajectoryDataset] | None, str | None]:
    config = runtime.config
    train_datasets = []
    validation_datasets = []
    recipes = config.vision.datasets
    for index, dataset_id in enumerate(dataset_ids):
        artifact = runtime.artifact_registry.load("raw_dataset", dataset_id)
        recipe_split = recipes[index].split_id if recipes else ""
        selected_split = recipe_split or (split_artifact_id if len(dataset_ids) == 1 else None)
        if selected_split:
            split = runtime.artifact_registry.load("split_set", selected_split)
            if dataset_id not in split.manifest.input_artifact_ids:
                raise ValueError(
                    f"Vision split {selected_split!r} does not belong to {dataset_id!r}."
                )
            payload = json.loads((split.path / "split_indices.json").read_text())
        else:
            payload = _default_vision_split_payload(
                config,
                dataset_artifact_id=dataset_id,
                num_episodes=len(_vision_dataset_from_artifact(artifact.path, episode_ids=None)),
            )
        train_ids = [int(value) for value in payload.get("train_episode_ids", [])]
        validation_ids = [int(value) for value in payload.get("validation_episode_ids", [])]
        test_ids = [int(value) for value in payload.get("test_episode_ids", [])]
        if not train_ids:
            raise ValueError(f"Vision dataset {dataset_id!r} has no training episodes.")
        if (
            set(train_ids) & set(validation_ids)
            or set(train_ids) & set(test_ids)
            or set(validation_ids) & set(test_ids)
        ):
            raise ValueError(f"Vision split for {dataset_id!r} has overlapping episode partitions.")
        partitions = ((train_ids, train_datasets), (validation_ids, validation_datasets))
        for ids, destination in partitions:
            if ids:
                destination.append(
                    _limit_vision_training_dataset(
                        _vision_dataset_from_artifact(artifact.path, episode_ids=ids),
                        max_episodes=int(config.vision.max_episodes),
                    )
                )
    return train_datasets, validation_datasets or None, split_artifact_id


def _resolve_vision_split_artifact_id(
    runtime,
    *,
    dataset_ids: list[str],
) -> str | None:
    if len(dataset_ids) != 1:
        return None
    recipes = runtime.config.vision.datasets
    if recipes and recipes[0].split_id:
        return recipes[0].split_id
    split_artifact_id = str(runtime.config.splits.artifact_id).strip()
    if not split_artifact_id:
        return None
    split_artifact = runtime.artifact_registry.load("split_set", split_artifact_id)
    if dataset_ids[0] not in split_artifact.manifest.input_artifact_ids:
        return None
    return split_artifact_id


def run(config_path: Path, overrides: list[str]) -> dict[str, str]:
    runtime = initialize_stage_runtime(config_path, overrides, "train_vision_encoder")
    config = runtime.config
    raw_payload = runtime.raw_payload
    policies = config.policies
    stage_log_path = runtime.run_directory.logs_dir / "stage_train_vision.log"
    reuse_target = resolve_reuse_target(
        config.reuse.vision_encoder_artifact_id,
        policies.training_resume,
        registry=runtime.artifact_registry if config.reuse.vision_encoder_artifact_id else None,
        artifact_type="vision_encoder" if config.reuse.vision_encoder_artifact_id else None,
    )
    resume_artifact_reference = config.reuse.vision_encoder_artifact_id
    resume_artifact_id = (
        resolve_artifact_reference_id(
            runtime.artifact_registry,
            "vision_encoder",
            resume_artifact_reference,
        )
        if resume_artifact_reference
        else ""
    )
    with managed_stage_run(
        config=config,
        run_directory=runtime.run_directory,
        stage_name="train_vision_encoder",
        run_name=f"vision__{config.tracking.variant_name}__{runtime.run_directory.identity.run_id}",
        tags=stage_tags(
            "train_vision_encoder",
            config.environment.env_id,
            base_tags=config.tracking.tags,
            study_name=config.tracking.study_name,
            variant_name=config.tracking.variant_name,
            variant_slug=runtime.run_directory.identity.variant_slug,
            seed=config.seed.global_seed,
        ),
    ) as stage_run:
        def _reuse_existing_artifact(artifact_id: str, *, reuse_mode: str) -> dict[str, str]:
            reused_artifact = runtime.artifact_registry.load("vision_encoder", artifact_id)
            architecture_path = reused_artifact.path / "architecture.txt"
            applied_output_tags = apply_configured_output_tags(
                runtime,
                "vision_encoder",
                reused_artifact.path,
            )
            runtime.run_directory.update_run_manifest(
                {
                    "status": "reused",
                    "reused_artifact_ids": [artifact_id],
                    "summary": {
                        "vision_encoder_artifact_id": artifact_id,
                        "vision_encoder_artifact_path": str(reused_artifact.path),
                        "output_tags": applied_output_tags,
                    },
                },
            )
            runtime.run_directory.write_symlink("results/vision_encoder", reused_artifact.path)
            if architecture_path.exists():
                runtime.run_directory.write_symlink(
                    "results/vision_encoder_architecture.txt",
                    architecture_path,
                )
                emit_text_block(
                    "vision_encoder_architecture",
                    architecture_path.read_text(),
                    metadata={
                        "artifact_id": artifact_id,
                        "reuse_mode": reuse_mode,
                    },
                    log_path=stage_log_path,
                )
            stage_run.update_config(
                {
                    "reuse_summary": summarize_reuse(
                        config,
                        artifact_registry=runtime.artifact_registry,
                    ),
                    "stage_inputs": {"reused_vision_encoder_artifact_id": artifact_id},
                }
            )
            stage_run.finalize(
                status="reused",
                summary={
                    "vision_encoder_artifact_id": artifact_id,
                    "vision_encoder_artifact_path": str(reused_artifact.path),
                    "output_tags": applied_output_tags,
                },
                upload_files=[
                    reused_artifact.path / "manifest.json",
                    architecture_path,
                ],
            )
            return augment_stage_result(runtime, {"vision.artifact_id": artifact_id})

        if reuse_target.stage_behavior == "reuse_existing_artifact":
            return _reuse_existing_artifact(
                reuse_target.artifact_id,
                reuse_mode=reuse_target.stage_behavior,
            )

        dataset_ids = [
            recipe.artifact_id
            if recipe.artifact_id
            else _newest_raw_dataset_for_environment(runtime.artifact_registry, recipe.environment)
            for recipe in config.vision.datasets
        ]
        if (
            not dataset_ids
            and config.dataset.artifact_type == "raw_dataset"
            and config.dataset.artifact_id
        ):
            dataset_ids = [config.dataset.artifact_id]
        if not dataset_ids:
            raise ValueError("train_vision_encoder requires explicit raw dataset artifact ids.")
        vision_split_artifact_id = _resolve_vision_split_artifact_id(
            runtime,
            dataset_ids=dataset_ids,
        )
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("vision.datasets must resolve to distinct raw datasets.")
        input_artifact_ids = list(dataset_ids)
        input_artifact_ids.extend(
            recipe.split_id for recipe in config.vision.datasets if recipe.split_id
        )
        if (
            vision_split_artifact_id is not None
            and vision_split_artifact_id not in input_artifact_ids
        ):
            input_artifact_ids.append(vision_split_artifact_id)
        stage_fingerprint = vision_encoder_stage_fingerprint(
            runtime.config,
            dataset_ids=dataset_ids,
            split_artifact_id=vision_split_artifact_id,
        )
        if reuse_target.stage_behavior == "train_fresh" and policies.training_resume == "fresh":
            matching_artifact = resolve_matching_artifact(
                runtime.artifact_registry,
                "vision_encoder",
                policies.artifact_reuse,
                config_fingerprint_value=stage_fingerprint,
                input_artifact_ids=input_artifact_ids,
            )
            if matching_artifact is not None:
                return _reuse_existing_artifact(
                    matching_artifact.artifact_id,
                    reuse_mode="reuse_if_config_match",
                )
        seed_everything(int(config.seed.global_seed))
        datasets, validation_datasets, vision_split_artifact_id = _resolve_vision_training_datasets(
            runtime,
            dataset_ids=dataset_ids,
            split_artifact_id=vision_split_artifact_id,
        )
        implicit_split_enabled = (
            validation_datasets is not None
            and vision_split_artifact_id is None
            and (
                not config.vision.datasets
                or any(not recipe.split_id for recipe in config.vision.datasets)
            )
        )
        requested_device = config.vision.device or (
            "cpu" if config.collection.safety.cpu_encoder_during_collection else "auto"
        )
        device = resolve_device(requested_device)
        effective_data_loader_num_workers = int(config.vision.data_loader_num_workers)
        effective_data_loader_pin_memory = bool(
            config.vision.data_loader_pin_memory and device.type == "cuda"
        )
        effective_data_loader_persistent_workers = bool(
            config.vision.data_loader_persistent_workers and effective_data_loader_num_workers > 0
        )
        device_summary = {
            "requested_device": requested_device,
            "resolved_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "vision_batch_size": int(config.vision.batch_size),
            "vision_epochs": int(config.vision.epochs),
            "max_episodes_per_dataset": int(config.vision.max_episodes),
            "max_frames_per_episode": int(config.vision.max_frames_per_episode),
            "frame_cache_mode": str(config.vision.frame_cache_mode),
            "frame_cache_memory_fraction": float(config.vision.frame_cache_memory_fraction),
            "data_loader_num_workers": effective_data_loader_num_workers,
            "data_loader_pin_memory": effective_data_loader_pin_memory,
            "data_loader_persistent_workers": effective_data_loader_persistent_workers,
        }
        if device.type == "cuda" and torch.cuda.is_available():
            device_summary["cuda_device_name"] = torch.cuda.get_device_name(device)
        resume_checkpoint = None
        if policies.training_resume != "fresh" and resume_artifact_id:
            resume_artifact = runtime.artifact_registry.load("vision_encoder", resume_artifact_id)
            resume_checkpoint = resume_artifact.path / "weights.pt"
        artifact_id = generate_artifact_id(
            "vision_encoder",
            config.environment.env_id,
            runtime.run_directory.identity.run_id,
        )
        stage_run.define_metric("vision/step")
        stage_run.define_metric("train/*", step_metric="vision/step")
        stage_run.define_metric("validation/*", step_metric="vision/step")
        stage_run.update_config(
            {
                "reuse_summary": summarize_reuse(
                    config,
                    artifact_registry=runtime.artifact_registry,
                ),
                "stage_inputs": {
                    "dataset_artifact_ids": dataset_ids,
                    "split_artifact_id": vision_split_artifact_id,
                    "validation_enabled": validation_datasets is not None,
                    "implicit_split_enabled": implicit_split_enabled,
                    "max_episodes_per_dataset": int(config.vision.max_episodes),
                    "max_frames_per_episode": int(config.vision.max_frames_per_episode),
                    "resume_from_artifact_reference": resume_artifact_reference or None,
                    "resume_from_artifact_id": resume_artifact_id or None,
                },
            }
        )
        runtime.artifact_registry.root.mkdir(parents=True, exist_ok=True)
        log_interval_steps = max(1, int(config.tracking.log_interval_steps))
        progress_reporter = ConsoleProgressReporter(
            "train_vision_encoder",
            stream=sys.stderr,
        )
        emit_text_block(
            "train_vision_encoder_runtime",
            "\n".join(f"{key}: {value}" for key, value in device_summary.items()),
            metadata={
                "dataset_count": len(dataset_ids),
            },
            log_path=stage_log_path,
        )

        def _log_step_metrics(metrics: dict[str, float], step_count: int) -> None:
            if (
                not any(key.startswith("validation/") for key in metrics)
                and step_count % log_interval_steps != 0
            ):
                return
            stage_run.log({"vision/step": step_count, **metrics}, step=step_count)

        with runtime.artifact_registry.temporary_directory(
            prefix="vision_encoder_",
        ) as temporary_directory:
            temp_dir = temporary_directory
            training_result = train_vision_model(
                config.vision,
                datasets,
                temp_dir,
                device=device,
                validation_datasets=validation_datasets,
                resume_checkpoint=resume_checkpoint,
                shuffle_seed=int(config.seed.global_seed),
                step_metrics_callback=_log_step_metrics,
                metric_interval_steps=log_interval_steps,
                progress_callback=progress_reporter,
            )
            emit_text_block(
                "train_vision_encoder_data_pipeline",
                "\n".join(
                    f"{key}: {value}" for key, value in training_result.training_summary.items()
                ),
                metadata={
                    "artifact_id": artifact_id,
                },
                log_path=stage_log_path,
            )
            model_parameter_device = next(
                training_result.model.parameters(),
                torch.empty((), device=device),
            ).device
            runtime_confirmation = {
                "model_parameter_device": str(model_parameter_device),
                "cuda_runtime_confirmed": model_parameter_device.type == "cuda",
            }
            if model_parameter_device.type == "cuda" and torch.cuda.is_available():
                runtime_confirmation["cuda_device_name"] = torch.cuda.get_device_name(
                    model_parameter_device
                )
            emit_text_block(
                "train_vision_encoder_device_confirmation",
                "\n".join(f"{key}: {value}" for key, value in runtime_confirmation.items()),
                metadata={
                    "artifact_id": artifact_id,
                },
                log_path=stage_log_path,
            )
            architecture_text = str(training_result.model)
            artifact_vision_config = export_effective_vision_config_payload(
                config.vision,
                training_result.model,
            )
            if not isinstance(training_result.model, torch.nn.Identity):
                torch.save(
                    {"model_state_dict": training_result.model.state_dict()},
                    temp_dir / "weights.pt",
                )
            (temp_dir / "training_config.yaml").write_text(
                yaml.safe_dump(artifact_vision_config, sort_keys=False)
            )
            (temp_dir / "architecture.txt").write_text(architecture_text.rstrip() + "\n")
            write_artifact_config_snapshots(
                temp_dir,
                raw_payload,
                config.to_dict(),
                stage_name="train_vision_encoder",
                section_names=["vision", "dataset", "seed", "policies", "reuse", "tracking"],
                extra_payload={
                    "artifact_id": artifact_id,
                    "dataset_artifact_ids": dataset_ids,
                    "split_artifact_id": vision_split_artifact_id,
                    "validation_enabled": validation_datasets is not None,
                    "implicit_split_enabled": implicit_split_enabled,
                    "max_episodes_per_dataset": int(config.vision.max_episodes),
                    "max_frames_per_episode": int(config.vision.max_frames_per_episode),
                    "resume_from_artifact_reference": resume_artifact_reference or None,
                    "resume_from_artifact_id": resume_artifact_id or None,
                },
            )
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                artifact_type="vision_encoder",
                created_by=CreatedBy(
                    run_id=runtime.run_directory.identity.run_id,
                    stage_name="train_vision_encoder",
                ),
                input_artifact_ids=input_artifact_ids,
                config_fingerprint=stage_fingerprint,
                git_commit=str(runtime.git_state.get("commit", "")),
                summary={
                    "model_type": artifact_vision_config["type"],
                    "latent_dim": artifact_vision_config["latent_dim"],
                    "channels": artifact_vision_config.get("channels", []),
                    "input_shape": [int(value) for value in training_result.input_shape],
                    "epochs": config.vision.epochs,
                    "max_episodes_per_dataset": int(config.vision.max_episodes),
                    "max_frames_per_episode": int(config.vision.max_frames_per_episode),
                    "validation_enabled": validation_datasets is not None,
                    "split_artifact_id": vision_split_artifact_id,
                    "implicit_split_enabled": implicit_split_enabled,
                    "best_validation_loss": (
                        float(training_result.best_validation_loss)
                        if training_result.best_validation_loss is not None
                        else None
                    ),
                    **_prefix_reconstruction_metrics("final_train", training_result.train_metrics),
                    **_prefix_reconstruction_metrics(
                        "best_validation", training_result.validation_metrics
                    ),
                    "vision_architecture": VISION_ARCHITECTURE_FINGERPRINT,
                    "artifact_reuse_policy": policies.artifact_reuse,
                    "training_resume_policy": policies.training_resume,
                },
            )
            manifest.write(temp_dir / "manifest.json")
            destination = runtime.artifact_registry.register_directory(
                "vision_encoder",
                artifact_id,
                temp_dir,
            )
        runtime.artifact_registry.mark_artifact_completed("vision_encoder", artifact_id)
        applied_output_tags = apply_configured_output_tags(runtime, "vision_encoder", destination)
        runtime.run_directory.update_run_manifest(
            {
                "status": "completed",
                "produced_artifact_ids": [artifact_id],
                "summary": {
                    "vision_encoder_artifact_id": artifact_id,
                    "vision_encoder_artifact_path": str(destination),
                    "output_tags": applied_output_tags,
                },
            }
        )
        runtime.run_directory.write_symlink("results/vision_encoder", destination)
        runtime.run_directory.write_symlink(
            "results/vision_encoder_architecture.txt",
            destination / "architecture.txt",
        )
        final_loss = training_result.loss_history[-1] if training_result.loss_history else 0.0
        best_validation_loss = (
            float(training_result.best_validation_loss)
            if training_result.best_validation_loss is not None
            else None
        )
        emit_text_block(
            "vision_encoder_architecture",
            architecture_text,
            metadata={
                "artifact_id": artifact_id,
                "vision_type": config.vision.type,
                "latent_dim": config.vision.latent_dim,
            },
            log_path=stage_log_path,
        )
        emit_metrics_block(
            "train_vision_encoder",
            {
                "vision/final_loss": float(final_loss),
                **_prefix_reconstruction_metrics(
                    "vision/final_train", training_result.train_metrics
                ),
                "vision/epochs": config.vision.epochs,
                "vision/latent_dim": config.vision.latent_dim,
                **(
                    {
                        "vision/best_validation_loss": best_validation_loss,
                        **_prefix_reconstruction_metrics(
                            "vision/best_validation", training_result.validation_metrics
                        ),
                    }
                    if best_validation_loss is not None
                    else {}
                ),
            },
            metadata={
                "artifact_id": artifact_id,
                "dataset_count": len(dataset_ids),
            },
            log_path=stage_log_path,
        )
        best_validation_metric_summary: dict[str, float | None] = (
            _prefix_reconstruction_metrics(
                "vision/best_validation", training_result.validation_metrics
            )
            if best_validation_loss is not None
            else {f"vision/best_validation_{key}": None for key in training_result.train_metrics}
        )
        stage_run.finalize(
            status="completed",
            summary={
                "vision_encoder_artifact_id": artifact_id,
                "vision_encoder_artifact_path": str(destination),
                "vision/final_loss": float(final_loss),
                **_prefix_reconstruction_metrics(
                    "vision/final_train", training_result.train_metrics
                ),
                "vision/best_validation_loss": best_validation_loss,
                **best_validation_metric_summary,
                "vision/implicit_split_enabled": implicit_split_enabled,
                "output_tags": applied_output_tags,
            },
            upload_files=[
                destination / "manifest.json",
                destination / "resolved_config.yaml",
                destination / "used_hyperparameters.yaml",
                destination / "training_config.yaml",
                destination / "architecture.txt",
                destination / "weights.pt",
                destination / "previews" / "loss_curve.png",
                destination / "previews" / "reconstruction_grid.png",
                destination / "previews" / "reconstruction_samples.gif",
                destination / "previews" / "validation" / "reconstruction_grid.png",
                destination / "previews" / "validation" / "reconstruction_samples.gif",
            ],
        )
        return augment_stage_result(runtime, {"vision.artifact_id": artifact_id})
