"""Analysis-side rate-map plumbing: cached maps, episode-statistics maps, small file writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.manifold import trustworthiness

from ..numerics.rate_map_kernels import (
    RateMapComputation,
    compute_rate_maps,
    compute_spatial_bin_assignments,
    flatten_positions,
)

if TYPE_CHECKING:
    from ..numerics.occupancy import EpisodeBinStatistics
    from .base import AnalysisInput


@dataclass(slots=True)
class OccupancyComputation:
    """Computed occupancy map plus resolved spatial bounds."""

    occupancy: np.ndarray
    raw_occupancy: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float]]


def _subsample_indices(num_points: int, max_points: int, random_seed: int) -> np.ndarray:
    """Sorted index subsample (all indices when within budget)."""
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(num_points, size=max_points, replace=False)
    return np.sort(chosen.astype(np.int64, copy=False))


def _compute_trustworthiness(
    features: np.ndarray,
    embedding: np.ndarray,
    neighbor_count: int,
    *,
    degenerate_value: float = 0.0,
) -> float:
    """trustworthiness with a small-sample k-clamp guard."""
    num_points = int(len(features))
    if num_points < 3:
        return degenerate_value
    safe_neighbor_count = max(
        1,
        min(int(neighbor_count), num_points - 1, max(1, (num_points - 1) // 2)),
    )
    return float(trustworthiness(features, embedding, n_neighbors=safe_neighbor_count))


def compute_rate_maps_from_episode_statistics(
    statistics: EpisodeBinStatistics,
    *,
    smoothing_sigma: float,
    min_occupancy: float,
    unit_chunk_size: int = 64,
) -> RateMapComputation:
    """Compute global rate maps from already prepared episode/bin statistics."""
    num_bins = statistics.num_bins_x * statistics.num_bins_y
    linear_bins = (statistics.episode_bin_codes % num_bins).astype(np.int64, copy=False)
    raw_occupancy = statistics.episode_bin_step_counts.sum(axis=0, dtype=np.float32).reshape(
        statistics.num_bins_y,
        statistics.num_bins_x,
    )
    occupancy = raw_occupancy
    if smoothing_sigma > 0.0:
        occupancy = gaussian_filter(raw_occupancy, sigma=smoothing_sigma)

    activity_maps = np.zeros((statistics.num_units, num_bins), dtype=np.float32)
    chunk_size = max(1, min(int(unit_chunk_size), statistics.num_units))
    for start_index in range(0, statistics.num_units, chunk_size):
        stop_index = min(start_index + chunk_size, statistics.num_units)
        chunk_values = statistics.flat_values[:, start_index:stop_index].astype(
            np.float32,
            copy=False,
        )
        chunk_width = stop_index - start_index
        flat_bin_indices = (
            linear_bins[:, None] + np.arange(chunk_width, dtype=np.int64)[None, :] * num_bins
        )
        chunk_activity = np.bincount(
            flat_bin_indices.reshape(-1),
            weights=chunk_values.reshape(-1).astype(np.float64, copy=False),
            minlength=chunk_width * num_bins,
        ).reshape(chunk_width, num_bins)
        activity_maps[start_index:stop_index] = chunk_activity.astype(np.float32, copy=False)

    activity_maps = activity_maps.reshape(
        statistics.num_units,
        statistics.num_bins_y,
        statistics.num_bins_x,
    )
    if smoothing_sigma > 0.0:
        activity_maps = gaussian_filter(
            activity_maps,
            sigma=(0.0, smoothing_sigma, smoothing_sigma),
        )
    safe_occupancy = np.where(occupancy >= min_occupancy, occupancy, np.nan)
    return RateMapComputation(
        rate_maps=(activity_maps / safe_occupancy[None, :, :]).astype(np.float32, copy=False),
        occupancy=occupancy.astype(np.float32, copy=False),
        raw_occupancy=raw_occupancy.astype(np.float32, copy=False),
        reliability_maps=None,
        visited_episode_counts=None,
        bounds=statistics.bounds,
    )


def compute_occupancy_only(
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> OccupancyComputation:
    """Compute only the smoothed occupancy map for a position trace."""
    positions = flatten_positions(position_xy, valid_mask)
    if positions.size == 0:
        raise ValueError("Cannot compute occupancy without any valid timesteps.")
    linear_bins, _, _, resolved_bounds = compute_spatial_bin_assignments(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    raw_occupancy = (
        np.bincount(
            linear_bins,
            minlength=num_bins_x * num_bins_y,
        )
        .reshape(num_bins_y, num_bins_x)
        .astype(np.float32, copy=False)
    )
    occupancy = raw_occupancy
    if smoothing_sigma > 0.0:
        occupancy = gaussian_filter(raw_occupancy, sigma=smoothing_sigma)
    return OccupancyComputation(
        occupancy=occupancy.astype(np.float32, copy=False),
        raw_occupancy=raw_occupancy,
        bounds=resolved_bounds,
    )


def get_or_compute_occupancy(
    analysis_input: AnalysisInput,
    *,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> OccupancyComputation:
    """Return the collection's occupancy map, computed once per binning for the stage."""
    cache_key = ("occupancy", num_bins_x, num_bins_y, smoothing_sigma, bounds)
    return analysis_input.get_cached_position_computation(
        cache_key,
        lambda: compute_occupancy_only(
            analysis_input.position_xy,
            analysis_input.valid_mask,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            bounds=bounds,
        ),
    )


def get_or_compute_rate_maps(
    analysis_input: AnalysisInput,
    *,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    unit_chunk_size: int = 64,
    episode_statistics: EpisodeBinStatistics | None = None,
) -> RateMapComputation:
    """Return cached rate maps for the current analysis input."""
    cache_key = (
        "rate_map_computation",
        num_bins_x,
        num_bins_y,
        smoothing_sigma,
        min_occupancy,
        bounds,
        unit_chunk_size,
    )
    return analysis_input.get_cached_rate_map_computation(
        cache_key,
        lambda: (
            compute_rate_maps_from_episode_statistics(
                episode_statistics,
                smoothing_sigma=smoothing_sigma,
                min_occupancy=min_occupancy,
                unit_chunk_size=unit_chunk_size,
            )
            if episode_statistics is not None
            else compute_rate_maps(
                analysis_input.representation,
                analysis_input.position_xy,
                analysis_input.valid_mask,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                smoothing_sigma=smoothing_sigma,
                min_occupancy=min_occupancy,
                bounds=bounds,
                unit_chunk_size=unit_chunk_size,
            )
        ),
    )


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    """Write a small CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")
    return path


def save_heatmap(
    path: Path, values: np.ndarray, title: str, x_labels: list[str], y_labels: list[str]
) -> Path:
    """Save a small heatmap figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xticks(range(len(x_labels)), labels=x_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(y_labels)), labels=y_labels)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
