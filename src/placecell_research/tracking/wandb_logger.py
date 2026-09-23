"""Lean W&B integration for stage runs."""

from __future__ import annotations

import logging
import sys
import threading
import warnings
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from placecell_research.tracking.naming import RunIdentity

_LOGGER = logging.getLogger(__name__)
_WANDB_FINISH_TIMEOUT_SECONDS = 30.0


def _to_wandb_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_wandb_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_wandb_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_wandb_value(item) for item in value]
    if isinstance(value, list):
        return [_to_wandb_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def flatten_wandb_config(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config payloads into compare-friendly dotted keys."""
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_wandb_config(value, dotted))
        else:
            flattened[dotted] = _to_wandb_value(value)
    return flattened


def build_wandb_run_config(
    resolved_config: dict[str, Any],
    *,
    identity: RunIdentity,
    stage_name: str,
    git_state: dict[str, Any],
    salient_diff: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat W&B config payload optimized for run comparison."""
    run_config = flatten_wandb_config(resolved_config)
    run_config.update(
        {
            "runtime.stage_name": stage_name,
            "runtime.run_id": identity.run_id,
            "runtime.study_name": identity.study_name,
            "runtime.variant_name": identity.variant_name,
            "runtime.variant_slug": identity.variant_slug,
            "runtime.signature": identity.signature,
            "runtime.git_commit": git_state.get("commit", ""),
            "runtime.git_branch": git_state.get("branch", ""),
            "runtime.git_dirty": bool(git_state.get("dirty", False)),
        }
    )
    if salient_diff:
        run_config["runtime.changed_salient_fields"] = sorted(salient_diff.keys())
    if extra_config:
        run_config.update(flatten_wandb_config(extra_config))
    return run_config


class WandbLogger:
    """Small optional wrapper around one W&B run."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        name: str,
        group: str,
        job_type: str,
        tags: list[str],
        config: dict[str, Any],
        mode: str = "online",
        directory: Path | None = None,
        failure_mode: str = "warn",
    ) -> None:
        self.enabled = enabled
        self.project = project
        self.name = name
        self.group = group
        self.job_type = job_type
        self.tags = tags
        self.config = config
        self.mode = mode
        self.directory = directory
        self.failure_mode = failure_mode
        self.run: Any | None = None
        self._wandb: Any | None = None
        self.last_error: str | None = None

    def _failure_message(self, operation: str, exc: Exception) -> str:
        return (
            "\n"
            "============================================================\n"
            f"W&B FAILURE: {operation} failed for '{self.name}'\n"
            f"Reason: {exc}\n"
            "W&B tracking is disabled for the remainder of this run.\n"
            "Local artifacts and manifests still remain the source of truth.\n"
            "Set tracking.wandb_failure_mode=raise to fail fast.\n"
            "============================================================\n"
        )

    def _announce_failure(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        warnings.warn(message.strip(), RuntimeWarning, stacklevel=3)

    def _finish_run_quietly(self) -> None:
        run = self.run
        self.run = None
        if run is None:
            return
        finished = threading.Event()

        def _finish() -> None:
            try:
                run.finish()
            except Exception:
                return
            finally:
                finished.set()

        finish_thread = threading.Thread(
            target=_finish,
            name=f"wandb-finish-{self.name}",
            daemon=True,
        )
        finish_thread.start()
        finish_thread.join(timeout=float(_WANDB_FINISH_TIMEOUT_SECONDS))
        if finished.is_set():
            return
        timeout_error = TimeoutError(f"finish timed out after {_WANDB_FINISH_TIMEOUT_SECONDS:.1f}s")
        self.last_error = f"finish: {timeout_error}"
        self.enabled = False
        self._wandb = None
        _LOGGER.warning(
            "W&B finish timed out for '%s' after %.1fs", self.name, _WANDB_FINISH_TIMEOUT_SECONDS
        )
        self._announce_failure(self._failure_message("finish", timeout_error))

    def _disable(self, operation: str, exc: Exception) -> None:
        self.last_error = f"{operation}: {exc}"
        message = self._failure_message(operation, exc)
        _LOGGER.warning("W&B disabled for '%s' after %s failed: %s", self.name, operation, exc)
        self._announce_failure(message)
        self.enabled = False
        self._wandb = None
        self._finish_run_quietly()
        if self.failure_mode == "raise":
            raise RuntimeError(message.strip()) from exc

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import wandb
        except Exception as exc:
            self._disable("import", exc)
            return
        self._wandb = wandb
        try:
            init_kwargs: dict[str, Any] = {
                "project": self.project,
                "name": self.name,
                "group": self.group,
                "job_type": self.job_type,
                "tags": self.tags,
                "config": self.config,
                "mode": self.mode,
            }
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                init_kwargs["dir"] = str(self.directory)
            self.run = wandb.init(**init_kwargs)
        except Exception as exc:
            self._disable("init", exc)

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        if self.run is None or self._wandb is None:
            return
        try:
            if step_metric is None:
                self._wandb.define_metric(name)
            else:
                self._wandb.define_metric(name, step_metric=step_metric)
        except Exception as exc:
            self._disable("define_metric", exc)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log({key: _to_wandb_value(value) for key, value in payload.items()}, step=step)
        except Exception as exc:
            self._disable("log", exc)

    def update_config(self, payload: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.config.update(flatten_wandb_config(payload), allow_val_change=True)
        except Exception as exc:
            self._disable("config.update", exc)

    def set_summary(self, payload: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            for key, value in payload.items():
                self.run.summary[key] = _to_wandb_value(value)
        except Exception as exc:
            self._disable("summary", exc)

    def upload_file(self, path: Path) -> None:
        if self.run is None or not path.exists():
            return
        try:
            self.run.save(str(path.resolve()), base_path=str(path.parent.resolve()), policy="now")
        except Exception as exc:
            self._disable("save", exc)

    def upload_files(self, paths: Iterable[Path]) -> None:
        for path in paths:
            self.upload_file(path)

    def finish(self) -> None:
        self._finish_run_quietly()
