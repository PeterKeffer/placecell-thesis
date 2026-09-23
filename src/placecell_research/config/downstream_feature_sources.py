"""Canonical downstream feature-source metadata."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

PlaceCodeTransformMode = Literal[
    "identity", "l2", "binary", "active_rms", "active_rms_l2", "zscore"
]
PlaceCodePreScale = Literal["none", "head_row_norm"]


class PlaceCodeStatsPathConfig(Protocol):
    place_code_stats_path: str


@dataclass(frozen=True, slots=True)
class PlaceCodeSourceSpec:
    transform: PlaceCodeTransformMode
    pre_scale: PlaceCodePreScale = "none"
    requires_stats: bool = False


PLACE_CODE_SOURCES: dict[str, PlaceCodeSourceSpec] = {
    "place_codes": PlaceCodeSourceSpec("identity"),
    "place_codes_l2": PlaceCodeSourceSpec("l2"),
    "place_codes_binary": PlaceCodeSourceSpec("binary"),
    "place_codes_active_rms": PlaceCodeSourceSpec("active_rms", requires_stats=True),
    "place_codes_zscore": PlaceCodeSourceSpec("zscore", requires_stats=True),
    "place_codes_rms_l2": PlaceCodeSourceSpec("active_rms_l2", requires_stats=True),
    "place_codes_headnorm_l2": PlaceCodeSourceSpec("l2", pre_scale="head_row_norm"),
}
PLACE_CODE_FEATURE_SOURCES = frozenset(PLACE_CODE_SOURCES)
STATS_REQUIRED_PLACE_CODE_SOURCES = frozenset(
    name for name, spec in PLACE_CODE_SOURCES.items() if spec.requires_stats
)


def missing_place_code_stats_sources(
    source_names: Iterable[str],
    *,
    stats_available: bool,
) -> list[str]:
    if stats_available:
        return []
    return sorted(
        source_name
        for source_name in source_names
        if source_name in PLACE_CODE_SOURCES and PLACE_CODE_SOURCES[source_name].requires_stats
    )


def resolved_place_code_stats_path(models: PlaceCodeStatsPathConfig) -> str | None:
    path = models.place_code_stats_path.strip()
    return path or None
