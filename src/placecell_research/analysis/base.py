"""Stable analysis interfaces."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np

from placecell_research.utils.angles import wrap_radians

CostTier = Literal["light", "standard", "heavy"]
_CACHE_MISS = object()
_CachedValue = TypeVar("_CachedValue")


@dataclass(slots=True)
class AnalysisInput:
    """Standard single-source analysis input."""

    representation: np.ndarray
    position_xy: np.ndarray
    heading: np.ndarray | None
    kinematics: np.ndarray | None
    actions: np.ndarray | None
    valid_mask: np.ndarray
    source_name: str
    label: str
    split_name: str
    rgb: np.ndarray | None = None
    latent: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    position_cache: dict[Hashable, Any] = field(default_factory=dict, repr=False)
    _rate_map_cache: dict[Hashable, Any] = field(default_factory=dict, repr=False)
    _metric_cache: dict[Hashable, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.heading is not None:
            self.heading = wrap_radians(self.heading)

    def get_cached_rate_map_computation(
        self,
        cache_key: Hashable,
        builder: Callable[[], _CachedValue],
    ) -> _CachedValue:
        cached = self._rate_map_cache.get(cache_key, _CACHE_MISS)
        if cached is _CACHE_MISS:
            cached = builder()
            self._rate_map_cache[cache_key] = cached
        return cast(_CachedValue, cached)

    def get_cached_position_computation(
        self,
        cache_key: Hashable,
        builder: Callable[[], _CachedValue],
    ) -> _CachedValue:
        """Cache a product of the trajectory alone, shared by every target of a collection."""
        cached = self.position_cache.get(cache_key, _CACHE_MISS)
        if cached is _CACHE_MISS:
            cached = builder()
            self.position_cache[cache_key] = cached
        return cast(_CachedValue, cached)

    def get_cached_metric(
        self,
        cache_key: Hashable,
        builder: Callable[[], _CachedValue],
    ) -> _CachedValue:
        cached = self._metric_cache.get(cache_key, _CACHE_MISS)
        if cached is _CACHE_MISS:
            cached = builder()
            self._metric_cache[cache_key] = cached
        return cast(_CachedValue, cached)


@dataclass(slots=True)
class AnalysisResult:
    """Output payload for single-source analyses."""

    metrics: dict[str, float]
    per_unit_metrics: dict[str, np.ndarray]
    figures: dict[str, Path]
    tables: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)
    figure_destinations: dict[str, Path] = field(default_factory=dict)


class AnalysisModule(Protocol):
    """Analysis contract for a single representation source."""

    @property
    def name(self) -> str: ...

    @property
    def cost_tier(self) -> CostTier: ...

    def required_representations(self) -> set[str]: ...

    def run(
        self,
        analysis_input: AnalysisInput,
        output_dir: Path,
        config: dict[str, Any],
    ) -> AnalysisResult: ...


class ComparativeAnalysisModule(Protocol):
    """Analysis contract for multi-input comparisons like remapping."""

    @property
    def name(self) -> str: ...

    @property
    def cost_tier(self) -> CostTier: ...

    def run(
        self,
        inputs: list[AnalysisInput],
        labels: list[str],
        output_dir: Path,
        config: dict[str, Any],
    ) -> AnalysisResult: ...
