"""Shared lifecycle helper for stage-local tracking."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from .run_directory import RunDirectory
from .tags import default_wandb_group
from .wandb_logger import WandbLogger, build_wandb_run_config


def _existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _default_manifest_uploads(run_directory: RunDirectory) -> list[Path]:
    manifests_dir = run_directory.manifests_dir
    return _existing_paths(
        [
            manifests_dir / "resolved_config.yaml",
            manifests_dir / "salient_diff.yaml",
            manifests_dir / "seed_bundle.json",
        ]
    )


def _load_git_state(run_directory: RunDirectory) -> dict[str, Any]:
    return dict(run_directory.load_run_manifest().get("git_state", {}))


def _load_salient_diff(run_directory: RunDirectory) -> dict[str, Any]:
    salient_diff_path = run_directory.manifests_dir / "salient_diff.yaml"
    if not salient_diff_path.exists():
        return {}
    return yaml.safe_load(salient_diff_path.read_text()) or {}


class ManagedStageRun:
    """Small context manager for stage W&B lifecycle and failure handling."""

    def __init__(
        self,
        *,
        logger: WandbLogger,
        run_directory: RunDirectory,
    ) -> None:
        self.logger = logger
        self.run_directory = run_directory
        self._finalized = False

    def __enter__(self) -> ManagedStageRun:
        self.logger.start()
        self.logger.upload_files(_default_manifest_uploads(self.run_directory))
        return self

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        self.logger.define_metric(name, step_metric=step_metric)

    def update_config(self, payload: dict[str, Any]) -> None:
        self.logger.update_config(payload)

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        self.logger.log(payload, step=step)

    def upload_files(self, paths: Iterable[Path]) -> None:
        self.logger.upload_files(_existing_paths(paths))

    def finalize(
        self,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
        upload_files: Iterable[Path] = (),
    ) -> None:
        payload = {"status": status}
        if summary:
            payload.update(summary)
        self.logger.set_summary(payload)
        self.upload_files([self.run_directory.manifests_dir / "run_manifest.json", *upload_files])
        self._finalized = True

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and not self._finalized:
            self.logger.set_summary({"status": "failed", "error": str(exc)})
        self.logger.finish()
        return False


def managed_stage_run(
    *,
    config: Any,
    run_directory: RunDirectory,
    stage_name: str,
    run_name: str,
    job_type: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    git_state: dict[str, Any] | None = None,
    salient_diff: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> ManagedStageRun:
    """Build a managed W&B lifecycle for one stage run."""
    resolved_git_state = git_state if git_state is not None else _load_git_state(run_directory)
    resolved_salient_diff = (
        salient_diff if salient_diff is not None else _load_salient_diff(run_directory)
    )
    logger = WandbLogger(
        enabled=bool(config.tracking.use_wandb and config.tracking.wandb_mode != "disabled"),
        project=config.tracking.wandb_project,
        name=run_name,
        group=group
        or default_wandb_group(config.tracking.study_name, run_directory.identity.variant_slug),
        job_type=job_type or stage_name,
        tags=sorted(set(tags or [*config.tracking.tags, stage_name])),
        config=build_wandb_run_config(
            config.to_dict(),
            identity=run_directory.identity,
            stage_name=stage_name,
            git_state=resolved_git_state,
            salient_diff=resolved_salient_diff,
            extra_config=extra_config,
        ),
        mode=config.tracking.wandb_mode,
        directory=run_directory.path / "wandb",
        failure_mode=config.tracking.wandb_failure_mode,
    )
    return ManagedStageRun(logger=logger, run_directory=run_directory)
