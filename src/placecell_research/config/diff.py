"""Curated config diff utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComparisonField:
    path: str
    label: str
    token: str


COMPARISON_FIELDS = (
    ComparisonField("environment.env_id", "environment", "env"),
    ComparisonField("spatial_model.inputs.observation_source", "observation_source", "obs"),
    ComparisonField("spatial_model.encoder.family", "encoder_family", "enc"),
    ComparisonField("spatial_model.predictor.family", "predictor_family", "pred"),
    ComparisonField("spatial_model.training.code_dim", "code_dim", "code"),
    ComparisonField("spatial_model.sparsifier.type", "sparsifier", "sparse"),
    ComparisonField("spatial_model.sparsifier.temperature", "sparsifier_temperature", "temp"),
    ComparisonField("spatial_model.teacher_student.mode", "teacher_student_mode", "teacher"),
    ComparisonField("seed.global_seed", "seed", "seed"),
)

SALIENT_PATHS = {field.path for field in COMPARISON_FIELDS}
ACTIVE_OBJECTIVES_PATH = "spatial_model.objectives.active_names"

INFRA_TOP_LEVEL_NAMESPACES = frozenset({"launcher", "tracking", "reuse", "policies", "pipeline"})
SCIENTIFIC_POLICY_PATHS = frozenset({"policies.checkpoint_selection"})


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, dotted))
        else:
            flattened[dotted] = value
    return flattened


def _active_objective_names(payload: dict[str, Any]) -> list[str]:
    objectives = payload.get("spatial_model", {}).get("objectives", {})
    if not isinstance(objectives, dict):
        return []
    return sorted(str(name) for name in objectives)


def extract_comparison_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Return curated comparison values used for naming and summaries."""
    flattened = _flatten(payload)
    extracted = {
        field.path: flattened[field.path] for field in COMPARISON_FIELDS if field.path in flattened
    }
    extracted[ACTIVE_OBJECTIVES_PATH] = _active_objective_names(payload)
    return extracted


def build_comparison_card(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a small compare-friendly manifest payload."""
    comparison_values = extract_comparison_values(payload)
    card = {
        field.label: comparison_values.get(field.path)
        for field in COMPARISON_FIELDS
        if field.path in comparison_values
    }
    card["active_objectives"] = comparison_values.get(ACTIVE_OBJECTIVES_PATH, [])
    return card


def _diff_leaves(
    base_flat: dict[str, Any],
    resolved_flat: dict[str, Any],
    *,
    include: Callable[[str], bool],
) -> dict[str, Any]:
    """{key: {base, current}} for every included leaf whose value differs between the two."""
    return {
        key: {"base": base_flat.get(key), "current": resolved_flat.get(key)}
        for key in sorted(set(base_flat) | set(resolved_flat))
        if include(key) and base_flat.get(key) != resolved_flat.get(key)
    }


def compute_salient_diff(base: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    """Return curated config differences likely to matter scientifically."""
    base_flat = _flatten(base)
    resolved_flat = _flatten(resolved)
    base_flat.update(extract_comparison_values(base))
    resolved_flat.update(extract_comparison_values(resolved))

    def is_salient(key: str) -> bool:
        return (
            key in SALIENT_PATHS
            or key == ACTIVE_OBJECTIVES_PATH
            or key.startswith("spatial_model.objectives.")
        )

    return _diff_leaves(base_flat, resolved_flat, include=is_salient)


def compute_config_diff(base: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    """Return every differing config leaf except pure-infrastructure namespaces."""
    return _diff_leaves(
        _flatten(base),
        _flatten(resolved),
        include=lambda key: (
            key in SCIENTIFIC_POLICY_PATHS or key.split(".", 1)[0] not in INFRA_TOP_LEVEL_NAMESPACES
        ),
    )
