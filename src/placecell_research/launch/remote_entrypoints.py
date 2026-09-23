"""CLI commands that run as one SLURM job, and the order of launcher overrides."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placecell_research.config import (
    load_downstream_run_config,
    load_experiment_config,
    load_study_config,
    validate_downstream_run_config,
    validate_experiment_config,
    validate_study_config,
)

from .user_settings import user_launcher_overrides

CONFIG_LOADERS: dict[str, tuple[Callable[[Path, list[str]], Any], Callable[[Any], list[str]]]] = {
    "experiment": (load_experiment_config, validate_experiment_config),
    "study": (load_study_config, validate_study_config),
    "downstream": (load_downstream_run_config, validate_downstream_run_config),
}
ENTRYPOINT_CONFIG_KINDS = {
    "pipeline": "experiment",
    "collect": "experiment",
    "create-split": "experiment",
    "train-vision": "experiment",
    "encode-dataset": "experiment",
    "train-model": "experiment",
    "evaluate": "experiment",
    "analyze": "experiment",
    "sweep": "study",
    "curriculum": "study",
    "downstream-rollout": "downstream",
    "downstream-train": "downstream",
}
FORCE_RECOMPUTE_ENTRYPOINTS = {"pipeline", "train-vision", "train-model"}


@dataclass(frozen=True, slots=True)
class ResolvedRemoteCliEntrypoint:
    cli_command: str
    config: Any
    overrides: list[str]


def default_launcher_overrides(overrides: list[str], launcher_profile: str) -> list[str]:
    """Profile first, then the user's launcher values, then the explicit overrides."""
    profile_selection = [override for override in overrides if override.startswith("launcher=")]
    remaining = [override for override in overrides if not override.startswith("launcher=")]
    if not profile_selection and not any(
        override.startswith("launcher.type=") for override in overrides
    ):
        profile_selection = [f"launcher={launcher_profile}"]
    return [*profile_selection, *user_launcher_overrides(), *remaining]


def _infer_remote_cli_entrypoint(config_path: Path) -> str:
    normalized_parts = {part.lower() for part in config_path.parts}
    if "downstream" in normalized_parts:
        return "downstream-train"
    if "study" in normalized_parts:
        return "sweep"
    return "pipeline"


def resolve_remote_cli_entrypoint(
    config_path: str | Path,
    entrypoint: str,
    overrides: list[str] | None = None,
    *,
    force_recompute: bool = False,
) -> ResolvedRemoteCliEntrypoint:
    resolved_config_path = Path(config_path).resolve()
    normalized_entrypoint = str(entrypoint or "auto").strip().lower()
    if normalized_entrypoint in {"", "auto"}:
        normalized_entrypoint = _infer_remote_cli_entrypoint(resolved_config_path)
    config_kind = ENTRYPOINT_CONFIG_KINDS.get(normalized_entrypoint)
    if config_kind is None:
        available_entrypoints = ", ".join(sorted(ENTRYPOINT_CONFIG_KINDS))
        raise ValueError(
            f"Unsupported remote entrypoint {normalized_entrypoint!r}. Available: "
            f"{available_entrypoints}."
        )
    resolved_overrides = default_launcher_overrides(list(overrides or []), "slurm")
    if force_recompute:
        if normalized_entrypoint not in FORCE_RECOMPUTE_ENTRYPOINTS:
            raise ValueError(
                "--force-recompute is not supported for remote entrypoint "
                f"{normalized_entrypoint!r}."
            )
        resolved_overrides = [*resolved_overrides, "policies.artifact_reuse=force_recompute"]
    load_config, validate_config = CONFIG_LOADERS[config_kind]
    config = load_config(resolved_config_path, resolved_overrides)
    validate_config(config)
    return ResolvedRemoteCliEntrypoint(
        cli_command=normalized_entrypoint,
        config=config,
        overrides=resolved_overrides,
    )
