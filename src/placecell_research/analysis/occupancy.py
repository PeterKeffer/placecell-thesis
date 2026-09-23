"""The stage-scoped position cache over the episode/bin occupancy kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..numerics.occupancy import (
    EpisodeBinOccupancy,
    EpisodeBinStatistics,
    _prepare_episode_bin_statistics,
)

if TYPE_CHECKING:
    from .base import AnalysisInput


def get_or_compute_episode_bin_statistics(
    analysis_input: AnalysisInput,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> EpisodeBinStatistics | None:
    """Prepare episode/bin statistics, reusing the collection's shared bin-level counts."""
    cache_key = ("episode_bin_occupancy", num_bins_x, num_bins_y, bounds)
    cached_occupancy = analysis_input.position_cache.get(cache_key)
    statistics = _prepare_episode_bin_statistics(
        analysis_input.representation,
        analysis_input.position_xy,
        analysis_input.valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
        episode_bin_occupancy=cached_occupancy,
    )
    if cached_occupancy is None and statistics is not None:
        analysis_input.position_cache[cache_key] = EpisodeBinOccupancy(
            episode_bin_step_counts=statistics.episode_bin_step_counts,
            visited_episode_counts=statistics.visited_episode_counts,
            bounds=statistics.bounds,
        )
    return statistics
