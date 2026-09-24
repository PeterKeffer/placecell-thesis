"""Config loading and composition."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import TypeAdapter

from .downstream_schema import DownstreamRunConfig
from .schema import ExperimentConfig, StudyConfig

ConfigT = TypeVar("ConfigT")


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_defaults(config_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    defaults = payload.pop("defaults", [])
    merged: dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, str):
            default_path = (config_path.parent / item).with_suffix(".yaml")
            merged = _merge(merged, resolve_defaults(default_path, load_yaml(default_path)))
            continue
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported defaults entry in {config_path}: {item!r}")
        for group, name in item.items():
            if group == "_self_":
                continue
            default_path = config_path.parents[1] / group / f"{name}.yaml"
            resolved_group_payload = resolve_defaults(default_path, load_yaml(default_path))
            if (
                isinstance(resolved_group_payload, dict)
                and set(resolved_group_payload) == {group}
                and isinstance(resolved_group_payload[group], dict)
            ):
                nested_payload = resolved_group_payload
            else:
                nested_payload = {group: resolved_group_payload}
            merged = _merge(merged, nested_payload)
    return _merge(merged, payload)


def _parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        pass
    if text.startswith("[") or text.startswith("{"):
        return yaml.safe_load(text)
    return text


def _resolve_group_override_payload(
    config_path: Path,
    group: str,
    name: str,
) -> dict[str, Any] | None:
    candidates = [parent / group / f"{name}.yaml" for parent in config_path.parents[1:]]
    group_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if group_path is None:
        return None
    resolved_group_payload = resolve_defaults(group_path, load_yaml(group_path))
    if (
        isinstance(resolved_group_payload, dict)
        and set(resolved_group_payload) == {group}
        and isinstance(resolved_group_payload[group], dict)
    ):
        return resolved_group_payload
    return {group: resolved_group_payload}


def _override_list_index(part: str, *, dotted_path: str, length: int) -> int:
    if not part.isdigit():
        raise TypeError(
            f"Override path {dotted_path!r} traverses a list and requires a non-negative "
            f"integer index, got {part!r}."
        )
    index = int(part)
    if index >= length:
        raise IndexError(
            f"Override path {dotted_path!r} uses list index {index}, but the list has "
            f"length {length}."
        )
    return index


def apply_overrides(
    payload: dict[str, Any],
    overrides: list[str] | None,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Apply dotted-path overrides, using non-negative numeric segments to index lists."""
    updated = dict(payload)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use path=value syntax: {override}")
        dotted_path, raw_value = override.split("=", 1)
        if config_path is not None and "." not in dotted_path:
            group_override_payload = _resolve_group_override_payload(
                config_path,
                dotted_path,
                raw_value,
            )
            if group_override_payload is not None:
                updated = _merge(updated, group_override_payload)
                continue
        cursor: Any = updated
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            if isinstance(cursor, dict):
                cursor = cursor.setdefault(part, {})
                continue
            if isinstance(cursor, list):
                cursor = cursor[
                    _override_list_index(part, dotted_path=dotted_path, length=len(cursor))
                ]
                continue
            raise TypeError(
                f"Override path {dotted_path!r} cannot traverse {part!r} through "
                f"{type(cursor).__name__}."
            )
        final_part = parts[-1]
        value = _parse_scalar(raw_value)
        if isinstance(cursor, dict):
            cursor[final_part] = value
        elif isinstance(cursor, list):
            cursor[
                _override_list_index(final_part, dotted_path=dotted_path, length=len(cursor))
            ] = value
        else:
            raise TypeError(
                f"Override path {dotted_path!r} cannot assign {final_part!r} on "
                f"{type(cursor).__name__}."
            )
    return updated


def _materialize_dataclass(cls: type[ConfigT], payload: Any) -> ConfigT:
    if payload is None:
        return cls()  # type: ignore[misc]
    return TypeAdapter(cls).validate_python(payload)


def revalidate_config(config: ConfigT) -> ConfigT:
    """Validate a config object again, so assignments made after loading are checked."""
    adapter = TypeAdapter(type(config))
    payload = asdict(config) if is_dataclass(config) else adapter.dump_python(config, mode="python")
    return adapter.validate_python(payload)


def materialize_dataclass(cls: type[ConfigT], payload: Any) -> ConfigT:
    """Public wrapper for nested dataclass materialization from dict payloads."""
    return _materialize_dataclass(cls, payload)


def load_raw_config_payload(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    """Load config including keys not represented in typed dataclasses."""
    payload = resolve_defaults(config_path, load_yaml(config_path))
    return apply_overrides(payload, overrides, config_path=config_path)


def load_experiment_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load and compose an experiment config."""
    path = Path(config_path)
    payload = resolve_defaults(path, load_yaml(path))
    payload = apply_overrides(payload, overrides, config_path=path)
    return _materialize_dataclass(ExperimentConfig, payload)


def load_study_config(config_path: str | Path, overrides: list[str] | None = None) -> StudyConfig:
    """Load and compose a study config."""
    path = Path(config_path)
    payload = resolve_defaults(path, load_yaml(path))
    payload = apply_overrides(payload, overrides, config_path=path)
    return _materialize_dataclass(StudyConfig, payload)


def load_downstream_run_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> DownstreamRunConfig:
    """Load a downstream RL single-run config."""
    path = Path(config_path)
    payload = resolve_defaults(path, load_yaml(path))
    payload = apply_overrides(payload, overrides, config_path=path)
    return _materialize_dataclass(DownstreamRunConfig, payload)
