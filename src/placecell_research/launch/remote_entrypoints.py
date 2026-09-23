"""Shared registry for CLI entrypoints that can be launched on SLURM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placecell_research.config import (
    load_downstream_run_config,
    load_experiment_config,
    validate_downstream_run_config,
    validate_experiment_config,
)

ConfigLoader = Callable[[Path, list[str]], Any]
ConfigValidator = Callable[[Any], list[str]]
OverrideNormalizer = Callable[[list[str]], list[str]]


@dataclass(frozen=True, slots=True)
class RemoteCliEntrypoint:
    cli_command: str
    load_config: ConfigLoader
    validate_config: ConfigValidator
    normalize_remote_overrides: OverrideNormalizer
    supports_force_recompute: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedRemoteCliEntrypoint:
    cli_command: str
    config: Any
    overrides: list[str]


def default_remote_launcher_overrides(overrides: list[str]) -> list[str]:
    return default_launcher_overrides(overrides, "slurm")


def default_launcher_overrides(overrides: list[str], launcher_profile: str) -> list[str]:
    if any(
        override.startswith("launcher=") or override.startswith("launcher.type=")
        for override in overrides
    ):
        return list(overrides)
    return [f"launcher={launcher_profile}", *overrides]


def _identity_overrides(overrides: list[str]) -> list[str]:
    return list(overrides)


def _load_validated_experiment_config(config_path: Path, overrides: list[str]) -> Any:
    config = load_experiment_config(config_path, overrides)
    validate_experiment_config(config)
    return config


def _load_validated_downstream_run_config(config_path: Path, overrides: list[str]) -> Any:
    config = load_downstream_run_config(config_path, overrides)
    validate_downstream_run_config(config)
    return config


REMOTE_CLI_ENTRYPOINTS: dict[str, RemoteCliEntrypoint] = {
    "pipeline": RemoteCliEntrypoint(
        cli_command="pipeline",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
        supports_force_recompute=True,
    ),
    "collect": RemoteCliEntrypoint(
        cli_command="collect",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
    ),
    "create-split": RemoteCliEntrypoint(
        cli_command="create-split",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
    ),
    "train-vision": RemoteCliEntrypoint(
        cli_command="train-vision",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
        supports_force_recompute=True,
    ),
    "encode-dataset": RemoteCliEntrypoint(
        cli_command="encode-dataset",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
    ),
    "train-model": RemoteCliEntrypoint(
        cli_command="train-model",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
        supports_force_recompute=True,
    ),
    "evaluate": RemoteCliEntrypoint(
        cli_command="evaluate",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
    ),
    "analyze": RemoteCliEntrypoint(
        cli_command="analyze",
        load_config=_load_validated_experiment_config,
        validate_config=validate_experiment_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
    ),
    "downstream-rollout": RemoteCliEntrypoint(
        cli_command="downstream-rollout",
        load_config=_load_validated_downstream_run_config,
        validate_config=validate_downstream_run_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
    ),
    "downstream-train": RemoteCliEntrypoint(
        cli_command="downstream-train",
        load_config=_load_validated_downstream_run_config,
        validate_config=validate_downstream_run_config,
        normalize_remote_overrides=default_remote_launcher_overrides,
    ),
}


def _infer_remote_cli_entrypoint(config_path: Path) -> str:
    normalized_parts = {part.lower() for part in config_path.parts}
    if "downstream" in normalized_parts:
        return "downstream-train"
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
    spec = REMOTE_CLI_ENTRYPOINTS.get(normalized_entrypoint)
    if spec is None:
        available_entrypoints = ", ".join(sorted(REMOTE_CLI_ENTRYPOINTS))
        raise ValueError(
            f"Unsupported remote entrypoint {normalized_entrypoint!r}. Available: "
            f"{available_entrypoints}."
        )
    resolved_overrides = spec.normalize_remote_overrides(list(overrides or []))
    if force_recompute:
        if not spec.supports_force_recompute:
            raise ValueError(
                "--force-recompute is not supported for remote entrypoint "
                f"{normalized_entrypoint!r}."
            )
        resolved_overrides = [*resolved_overrides, "policies.artifact_reuse=force_recompute"]
    config = spec.load_config(resolved_config_path, resolved_overrides)
    return ResolvedRemoteCliEntrypoint(
        cli_command=spec.cli_command,
        config=config,
        overrides=resolved_overrides,
    )
