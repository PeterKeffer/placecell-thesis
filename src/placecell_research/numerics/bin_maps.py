"""Turning binned activity sums into smoothed rate maps, and correlating those maps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _rate_map_from_activity_sum(
    activity_sum: np.ndarray,
    occupancy_counts: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
    min_occupancy: float,
) -> np.ndarray:
    """Convert one [bins] episode/pool activity sum into a smoothed occupancy-normalized map."""
    activity_map = activity_sum.reshape(num_bins_y, num_bins_x).astype(np.float32, copy=False)
    occupancy_map = occupancy_counts.reshape(num_bins_y, num_bins_x).astype(np.float32, copy=False)
    if smoothing_sigma > 0.0:
        activity_map = gaussian_filter(activity_map, sigma=smoothing_sigma)
        occupancy_map = gaussian_filter(occupancy_map, sigma=smoothing_sigma)
    safe_occupancy = np.where(occupancy_map >= min_occupancy, occupancy_map, np.nan)
    return (activity_map / safe_occupancy).astype(np.float32, copy=False)


def _smooth_flat_bin_maps(
    flat_maps: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
) -> np.ndarray:
    """Smooth one or more flattened spatial maps along only the spatial axes."""
    maps = flat_maps.reshape(*flat_maps.shape[:-1], num_bins_y, num_bins_x).astype(
        np.float32,
        copy=False,
    )
    if smoothing_sigma > 0.0:
        sigma = (0.0,) * (maps.ndim - 2) + (
            float(smoothing_sigma),
            float(smoothing_sigma),
        )
        maps = gaussian_filter(maps, sigma=sigma)
    return maps.astype(np.float32, copy=False)


def _rate_maps_from_activity_sums(
    activity_sums: np.ndarray,
    occupancy_counts: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
    min_occupancy: float,
) -> np.ndarray:
    """Convert flattened activity sums into smoothed occupancy-normalized maps."""
    occupancy_maps = _smooth_flat_bin_maps(
        occupancy_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    safe_occupancy = np.where(occupancy_maps >= min_occupancy, occupancy_maps, np.nan)
    return _rate_maps_from_activity_sums_with_safe_occupancy(
        activity_sums,
        safe_occupancy,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )


def _rate_maps_from_activity_sums_with_safe_occupancy(
    activity_sums: np.ndarray,
    safe_occupancy: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
) -> np.ndarray:
    """Convert flattened activity sums using a pre-smoothed safe occupancy map."""
    activity_maps = _smooth_flat_bin_maps(
        activity_sums,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    return (activity_maps / safe_occupancy).astype(np.float32, copy=False)


MIN_MAP_CORRELATION_OVERLAP_BINS = 10


def _batched_masked_map_correlation(
    first_maps: np.ndarray,
    second_maps: np.ndarray,
    *,
    epsilon: float = 1e-8,
    min_overlap_bins: int = MIN_MAP_CORRELATION_OVERLAP_BINS,
) -> np.ndarray:
    """Rowwise Pearson correlation over overlapping finite bins; NaN below min overlap."""
    first_values = first_maps.reshape(first_maps.shape[0], -1).astype(
        np.float32,
        copy=False,
    )
    second_values = second_maps.reshape(second_maps.shape[0], -1).astype(
        np.float32,
        copy=False,
    )
    overlap_mask = np.isfinite(first_values) & np.isfinite(second_values)
    overlap_counts = np.sum(overlap_mask, axis=1, dtype=np.int32)
    masked_first = np.where(overlap_mask, first_values, 0.0)
    masked_second = np.where(overlap_mask, second_values, 0.0)
    safe_counts = np.maximum(overlap_counts, 1).astype(np.float32, copy=False)
    first_means = masked_first.sum(axis=1, dtype=np.float32) / safe_counts
    second_means = masked_second.sum(axis=1, dtype=np.float32) / safe_counts
    first_centered = np.where(overlap_mask, first_values - first_means[:, None], 0.0)
    second_centered = np.where(overlap_mask, second_values - second_means[:, None], 0.0)
    denominator = np.sqrt(
        np.square(first_centered, dtype=np.float32).sum(axis=1, dtype=np.float32)
        * np.square(second_centered, dtype=np.float32).sum(axis=1, dtype=np.float32)
    )
    correlations = np.full((first_values.shape[0],), np.nan, dtype=np.float32)
    usable = (overlap_counts >= max(2, int(min_overlap_bins))) & (denominator > float(epsilon))
    correlations[usable] = (
        (first_centered[usable] * second_centered[usable]).sum(axis=1, dtype=np.float32)
        / denominator[usable]
    ).astype(np.float32, copy=False)
    return correlations


def _smoothed_safe_occupancy(
    occupancy_counts: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
    min_occupancy: float,
) -> np.ndarray:
    occupancy_maps = _smooth_flat_bin_maps(
        occupancy_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    return np.where(occupancy_maps >= min_occupancy, occupancy_maps, np.nan)


def _masked_map_correlation(
    first_map: np.ndarray,
    second_map: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Compute Pearson correlation over overlapping finite bins of two maps."""
    overlap_mask = np.isfinite(first_map) & np.isfinite(second_map)
    if int(overlap_mask.sum()) < 2:
        return 0.0
    first_values = first_map[overlap_mask].astype(np.float32, copy=False)
    second_values = second_map[overlap_mask].astype(np.float32, copy=False)
    first_centered = first_values - float(first_values.mean())
    second_centered = second_values - float(second_values.mean())
    denominator = float(
        np.sqrt(np.square(first_centered).sum() * np.square(second_centered).sum())
    )
    if denominator <= epsilon:
        return 0.0
    return float((first_centered * second_centered).sum() / denominator)
