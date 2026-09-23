"""Reliability of a rate map across episode subsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.bin_maps import (
    _smooth_flat_bin_maps,
    batched_masked_map_correlation,
    rate_maps_from_activity_sums_with_safe_occupancy,
    smoothed_safe_occupancy,
)
from ..numerics.occupancy import (
    EpisodeBinStatistics,
    _episode_activity_sums,
    _unit_chunk_bounds,
    iter_episode_activity_sum_chunks,
    prepare_episode_bin_statistics,
)
from ..numerics.split_half import (
    _accumulate_multi_split_correlations,
    _finalize_multi_split_correlations,
    _prepare_balanced_split_supports,
)
from ..numerics.work_blocks import run_over_index_blocks


@dataclass(slots=True)
class RevisitActivityMetrics:
    """Rate-map revisit metrics computed from shared episode/bin activity sums."""

    bin_consistency_maps: np.ndarray
    bin_coefficient_of_variation_maps: np.ndarray
    consistency_visit_counts: np.ndarray
    split_half_agreement_maps: np.ndarray
    split_half_agreement_support_counts: np.ndarray
    split_half_rate_map_correlation: np.ndarray
    episode_rate_map_correlation: np.ndarray


def _empty_episode_metric_maps(
    *,
    num_units: int,
    num_bins_y: int,
    num_bins_x: int,
) -> tuple[np.ndarray, np.ndarray]:
    empty_maps = np.full((num_units, num_bins_y, num_bins_x), np.nan, dtype=np.float32)
    empty_visits = np.zeros((num_bins_y, num_bins_x), dtype=np.float32)
    return empty_maps, empty_visits


_THRESHOLD_BUFFER_BYTES = 128 * 1024**2


def _compute_global_unit_thresholds(
    flat_values: np.ndarray,
    *,
    threshold_mode: str,
    threshold_fraction: float,
    threshold_quantile: float,
    use_absolute_activations: bool,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute exact per-unit thresholds with at most one unit block in float64."""
    normalized_mode = str(threshold_mode).strip().lower()
    if normalized_mode == "quantile_per_unit":
        level, scale = threshold_quantile, 1.0
    elif normalized_mode == "peak_fraction":
        level, scale = 0.995, float(threshold_fraction)
    else:
        raise ValueError(
            "threshold_mode must be one of {'peak_fraction', 'quantile_per_unit'}, "
            f"got {threshold_mode!r}."
        )
    num_samples, num_units = flat_values.shape
    bytes_per_unit = max(1, num_samples * np.dtype(np.float64).itemsize)
    units_per_block = max(1, _THRESHOLD_BUFFER_BYTES // bytes_per_unit)
    thresholds = np.empty(num_units, dtype=np.float32)
    for start in range(0, num_units, units_per_block):
        stop = min(start + units_per_block, num_units)
        quantile_source = np.array(flat_values[:, start:stop].T, dtype=np.float64, order="C")
        if use_absolute_activations:
            np.abs(quantile_source, out=quantile_source)
        block_thresholds = np.quantile(
            quantile_source, np.float64(level), axis=1, overwrite_input=True
        ) * scale
        thresholds[start:stop] = np.maximum(block_thresholds, float(epsilon))
        del quantile_source
    return thresholds


def _compute_bin_consistency_chunk(
    chunk_activity_sums: np.ndarray,
    step_counts: np.ndarray,
    visited_mask_by_episode: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    minimum_visited_episodes: int,
    active_bin_peak_fraction: float,
    active_episode_threshold_fraction_of_bin_mean: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_bins = num_bins_x * num_bins_y
    chunk_width = chunk_activity_sums.shape[0]
    episode_means = np.full_like(chunk_activity_sums, np.nan, dtype=np.float32)
    np.divide(
        chunk_activity_sums,
        step_counts[None, :, :],
        out=episode_means,
        where=visited_mask_by_episode[None, :, :],
    )
    finite_episode_mask = np.isfinite(episode_means)
    finite_episode_means = np.where(finite_episode_mask, episode_means, 0.0)
    visit_count = np.sum(finite_episode_mask, axis=1, dtype=np.int32)
    sum_means = finite_episode_means.sum(axis=1, dtype=np.float32)
    sum_squares = np.square(finite_episode_means, dtype=np.float32).sum(axis=1, dtype=np.float32)
    valid_bins = visit_count >= int(minimum_visited_episodes)

    mean_per_bin = np.zeros((chunk_width, num_bins), dtype=np.float32)
    np.divide(
        sum_means,
        visit_count,
        out=mean_per_bin,
        where=valid_bins,
    )
    variance_per_bin = np.zeros((chunk_width, num_bins), dtype=np.float32)
    np.divide(
        sum_squares,
        visit_count,
        out=variance_per_bin,
        where=valid_bins,
    )
    variance_per_bin = np.maximum(variance_per_bin - np.square(mean_per_bin), 0.0)

    coefficient_of_variation = np.full((chunk_width, num_bins), np.nan, dtype=np.float32)
    np.divide(
        np.sqrt(variance_per_bin).astype(np.float32, copy=False),
        np.abs(mean_per_bin) + float(epsilon),
        out=coefficient_of_variation,
        where=valid_bins,
    )

    positive_mean_per_bin = np.maximum(mean_per_bin, 0.0).astype(np.float32, copy=False)
    positive_peak = np.max(np.where(valid_bins, positive_mean_per_bin, 0.0), axis=1)
    active_bin_threshold = np.maximum(
        positive_peak[:, None] * float(active_bin_peak_fraction),
        float(epsilon),
    )
    active_bin_mask = valid_bins & (positive_mean_per_bin >= active_bin_threshold)
    coefficient_of_variation[~active_bin_mask] = np.nan

    consistency = np.full((chunk_width, num_bins), np.nan, dtype=np.float32)
    consistency[valid_bins] = 0.0
    local_activity_threshold = np.maximum(
        positive_mean_per_bin * float(active_episode_threshold_fraction_of_bin_mean),
        float(epsilon),
    ).astype(np.float32, copy=False)
    locally_active_episode_counts = np.sum(
        finite_episode_mask & (episode_means >= local_activity_threshold[:, None, :]),
        axis=1,
        dtype=np.int32,
    )
    np.divide(
        locally_active_episode_counts,
        visit_count,
        out=consistency,
        where=active_bin_mask,
    )
    return (
        consistency.reshape(chunk_width, num_bins_y, num_bins_x).astype(np.float32, copy=False),
        coefficient_of_variation.reshape(chunk_width, num_bins_y, num_bins_x).astype(
            np.float32,
            copy=False,
        ),
    )


def _compute_split_half_agreement_chunk(
    even_activity: np.ndarray,
    odd_activity: np.ndarray,
    safe_even_occupancy: np.ndarray,
    safe_odd_occupancy: np.ndarray,
    support_mask: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    even_rate_maps = rate_maps_from_activity_sums_with_safe_occupancy(
        even_activity,
        safe_even_occupancy,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    odd_rate_maps = rate_maps_from_activity_sums_with_safe_occupancy(
        odd_activity,
        safe_odd_occupancy,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    valid_mask_by_bin = np.isfinite(even_rate_maps) & np.isfinite(odd_rate_maps)
    valid_mask_by_bin &= support_mask[None, :, :]
    denominator = np.abs(even_rate_maps) + np.abs(odd_rate_maps) + float(epsilon)
    agreement_chunk = np.full_like(even_rate_maps, np.nan, dtype=np.float32)
    agreement_chunk[valid_mask_by_bin] = np.clip(
        1.0
        - (
            np.abs(even_rate_maps[valid_mask_by_bin] - odd_rate_maps[valid_mask_by_bin])
            / denominator[valid_mask_by_bin]
        ),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    correlations = batched_masked_map_correlation(even_rate_maps, odd_rate_maps)
    return agreement_chunk, correlations


def _episode_rate_map_correlation_support(
    step_counts: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
    min_occupancy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    total_occupancy = step_counts.sum(axis=0, dtype=np.float32)
    visited_bin_counts = np.count_nonzero(step_counts > 0, axis=1)
    other_visited_bin_counts = np.count_nonzero(
        (total_occupancy[None, :] - step_counts) > 0,
        axis=1,
    )
    supported_episode_mask = (visited_bin_counts >= 2) & (other_visited_bin_counts >= 2)
    if not np.any(supported_episode_mask):
        return None

    smoothed_episode_occupancy = _smooth_flat_bin_maps(
        step_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    smoothed_total_occupancy = _smooth_flat_bin_maps(
        total_occupancy[None, :],
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )[0]
    safe_episode_occupancy = np.where(
        smoothed_episode_occupancy >= min_occupancy,
        smoothed_episode_occupancy,
        np.nan,
    )
    smoothed_other_occupancy = smoothed_total_occupancy[None, :, :] - smoothed_episode_occupancy
    safe_other_occupancy = np.where(
        smoothed_other_occupancy >= min_occupancy,
        smoothed_other_occupancy,
        np.nan,
    )
    return supported_episode_mask, safe_episode_occupancy, safe_other_occupancy


def _compute_episode_rate_map_correlation_chunk(
    chunk_activity_sums: np.ndarray,
    supported_episode_mask: np.ndarray,
    safe_episode_occupancy: np.ndarray,
    safe_other_occupancy: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
) -> np.ndarray:
    total_activity = chunk_activity_sums.sum(axis=1, dtype=np.float32)
    smoothed_episode_activity = _smooth_flat_bin_maps(
        chunk_activity_sums,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    smoothed_total_activity = _smooth_flat_bin_maps(
        total_activity,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    episode_rate_maps = smoothed_episode_activity / safe_episode_occupancy[None, :, :, :]
    other_rate_maps = (
        smoothed_total_activity[:, None, :, :] - smoothed_episode_activity
    ) / safe_other_occupancy[None, :, :, :]
    pair_correlations = batched_masked_map_correlation(
        episode_rate_maps.reshape(-1, num_bins_y, num_bins_x),
        other_rate_maps.reshape(-1, num_bins_y, num_bins_x),
    ).reshape(chunk_activity_sums.shape[0], chunk_activity_sums.shape[1])
    supported_correlations = pair_correlations[:, supported_episode_mask]
    finite = np.isfinite(supported_correlations)
    finite_counts = finite.sum(axis=1)
    means = np.full(supported_correlations.shape[0], np.nan, dtype=np.float32)
    has_support = finite_counts > 0
    means[has_support] = (
        np.where(finite, supported_correlations, 0.0).sum(axis=1, dtype=np.float32)[has_support]
        / finite_counts[has_support]
    )
    return means


def _split_half_support(
    step_counts: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
    min_occupancy: float,
    minimum_episodes_per_half: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    visited_by_episode = step_counts > 0
    even_episode_mask = (np.arange(step_counts.shape[0], dtype=np.int32) % 2) == 0
    odd_episode_mask = ~even_episode_mask
    even_occupancy = step_counts[even_episode_mask].sum(axis=0, dtype=np.float32)
    odd_occupancy = step_counts[odd_episode_mask].sum(axis=0, dtype=np.float32)
    even_episode_visits = np.sum(visited_by_episode[even_episode_mask], axis=0, dtype=np.int32)
    odd_episode_visits = np.sum(visited_by_episode[odd_episode_mask], axis=0, dtype=np.int32)
    support_counts = (
        np.minimum(even_episode_visits, odd_episode_visits)
        .reshape(
            num_bins_y,
            num_bins_x,
        )
        .astype(np.float32, copy=False)
    )
    support_mask = (
        (even_episode_visits >= int(minimum_episodes_per_half))
        & (odd_episode_visits >= int(minimum_episodes_per_half))
    ).reshape(num_bins_y, num_bins_x)
    safe_even_occupancy = smoothed_safe_occupancy(
        even_occupancy,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
    )
    safe_odd_occupancy = smoothed_safe_occupancy(
        odd_occupancy,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
    )
    return (
        even_episode_mask,
        odd_episode_mask,
        safe_even_occupancy,
        safe_odd_occupancy,
        support_counts,
        support_mask,
    )


def compute_reliability_maps(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    threshold_mode: str = "peak_fraction",
    threshold_fraction: float = 0.3,
    threshold_quantile: float = 0.95,
    use_absolute_activations: bool = False,
    unit_chunk_size: int = 64,
    episode_statistics: EpisodeBinStatistics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin firing reliability across episodes."""
    statistics = episode_statistics or prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    if statistics is None:
        return _empty_episode_metric_maps(
            num_units=representation.shape[-1],
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
        )

    num_bins = num_bins_x * num_bins_y
    thresholds = _compute_global_unit_thresholds(
        statistics.flat_values,
        threshold_mode=threshold_mode,
        threshold_fraction=threshold_fraction,
        threshold_quantile=threshold_quantile,
        use_absolute_activations=use_absolute_activations,
    )

    spike_episode_counts = np.zeros((statistics.num_units, num_bins), dtype=np.int32)
    num_episode_bins = statistics.num_episodes * num_bins
    chunk_size = max(1, min(int(unit_chunk_size), statistics.num_units))
    for start_index in range(0, statistics.num_units, chunk_size):
        stop_index = min(start_index + chunk_size, statistics.num_units)
        flat_chunk = statistics.flat_values[:, start_index:stop_index]
        strong_chunk = (
            np.abs(flat_chunk) >= thresholds[None, start_index:stop_index]
            if use_absolute_activations
            else flat_chunk >= thresholds[None, start_index:stop_index]
        )
        if not np.any(strong_chunk):
            continue
        step_indices, local_unit_indices = np.nonzero(strong_chunk)
        active_codes = statistics.episode_bin_codes[step_indices] + (
            local_unit_indices.astype(np.int64, copy=False) * num_episode_bins
        )
        unique_active_codes = np.unique(active_codes)
        active_local_units = unique_active_codes // num_episode_bins
        active_spatial_bins = unique_active_codes % num_bins
        count_codes = active_local_units * num_bins + active_spatial_bins
        chunk_counts = np.bincount(
            count_codes,
            minlength=(stop_index - start_index) * num_bins,
        ).reshape(stop_index - start_index, num_bins)
        spike_episode_counts[start_index:stop_index] = chunk_counts.astype(
            np.int32,
            copy=False,
        )

    reliability_maps = np.full(
        (statistics.num_units, num_bins_y, num_bins_x), np.nan, dtype=np.float32
    )
    visited_mask = statistics.visited_episode_counts > 0
    if np.any(visited_mask):
        reliability_flat = reliability_maps.reshape(statistics.num_units, -1)
        reliability_flat[:, visited_mask] = (
            spike_episode_counts[:, visited_mask] / statistics.visited_episode_counts[visited_mask]
        ).astype(np.float32, copy=False)
    return (
        reliability_maps,
        statistics.visited_episode_counts.reshape(num_bins_y, num_bins_x).astype(
            np.float32, copy=False
        ),
    )


def compute_field_traversal_reliability(
    statistics: EpisodeBinStatistics,
    field_masks: np.ndarray,
    *,
    threshold_mode: str = "peak_fraction",
    threshold_fraction: float = 0.3,
    threshold_quantile: float = 0.95,
    use_absolute_activations: bool = False,
    minimum_traversals: int = 5,
    flat_headings: np.ndarray | None = None,
    num_heading_sectors: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per unit: fraction of field traversals with at least one strong activation."""
    if field_masks.shape[0] != statistics.num_units:
        raise ValueError(
            f"field_masks has {field_masks.shape[0]} units, statistics has {statistics.num_units}."
        )
    num_bins = statistics.num_bins_x * statistics.num_bins_y
    episode_ids = statistics.episode_bin_codes // num_bins
    spatial_bins = statistics.episode_bin_codes % num_bins
    thresholds = _compute_global_unit_thresholds(
        statistics.flat_values,
        threshold_mode=threshold_mode,
        threshold_fraction=threshold_fraction,
        threshold_quantile=threshold_quantile,
        use_absolute_activations=use_absolute_activations,
    )
    segment_start = statistics.segment_start
    flat_field_masks = field_masks.reshape(statistics.num_units, -1).astype(bool, copy=False)
    traversal_reliability = np.full(statistics.num_units, np.nan, dtype=np.float32)
    traversal_counts = np.zeros(statistics.num_units, dtype=np.int32)
    directional_reliability = np.full(statistics.num_units, np.nan, dtype=np.float32)
    directional_counts = np.zeros(statistics.num_units, dtype=np.int32)
    headings = None
    if flat_headings is not None:
        headings = np.asarray(flat_headings, dtype=np.float64).reshape(-1)
        if headings.shape[0] != statistics.flat_values.shape[0]:
            raise ValueError(
                f"flat_headings has {headings.shape[0]} steps, statistics has "
                f"{statistics.flat_values.shape[0]}."
            )
    for unit_index in range(statistics.num_units):
        field_flat = flat_field_masks[unit_index]
        if not field_flat.any():
            continue
        in_field = field_flat[spatial_bins]
        if not in_field.any():
            continue
        left_field = np.empty_like(in_field)
        left_field[0] = True
        left_field[1:] = ~in_field[:-1]
        traversal_starts = in_field & (left_field | segment_start)
        num_traversals = int(traversal_starts.sum())
        traversal_counts[unit_index] = num_traversals
        if num_traversals < minimum_traversals:
            continue
        unit_values = statistics.flat_values[:, unit_index]
        strong = (
            np.abs(unit_values) >= thresholds[unit_index]
            if use_absolute_activations
            else unit_values >= thresholds[unit_index]
        )
        active_steps = in_field & strong
        traversal_id = np.cumsum(traversal_starts) - 1
        if active_steps.any():
            hit_traversals = np.unique(traversal_id[active_steps])
        else:
            hit_traversals = np.empty(0, dtype=np.int64)
        traversal_reliability[unit_index] = float(len(hit_traversals) / num_traversals)
        if headings is None:
            continue
        in_field_ids = traversal_id[in_field]
        sin_sums = np.zeros(num_traversals, dtype=np.float64)
        cos_sums = np.zeros(num_traversals, dtype=np.float64)
        np.add.at(sin_sums, in_field_ids, np.sin(headings[in_field]))
        np.add.at(cos_sums, in_field_ids, np.cos(headings[in_field]))
        traversal_angles = np.arctan2(sin_sums, cos_sums)
        sector_width = 2.0 * np.pi / num_heading_sectors
        traversal_sectors = (
            np.floor((traversal_angles + np.pi) / sector_width).astype(np.int64)
            % num_heading_sectors
        )
        traversal_is_hit = np.zeros(num_traversals, dtype=bool)
        traversal_is_hit[hit_traversals] = True
        traversal_episodes = episode_ids[traversal_starts]
        held_out = (traversal_episodes % 2) == 1
        selection_sectors = traversal_sectors[~held_out]
        held_out_sectors = traversal_sectors[held_out]
        selection_totals = np.bincount(selection_sectors, minlength=num_heading_sectors)
        selection_hits = np.bincount(
            selection_sectors[traversal_is_hit[~held_out]], minlength=num_heading_sectors
        )
        held_out_totals = np.bincount(held_out_sectors, minlength=num_heading_sectors)
        held_out_hits = np.bincount(
            held_out_sectors[traversal_is_hit[held_out]], minlength=num_heading_sectors
        )
        qualifying_sectors = (selection_totals >= minimum_traversals) & (
            held_out_totals >= minimum_traversals
        )
        if not qualifying_sectors.any():
            continue
        selection_rates = np.where(
            qualifying_sectors, selection_hits / np.maximum(selection_totals, 1), -1.0
        )
        best_sector = int(np.argmax(selection_rates))
        directional_reliability[unit_index] = float(
            held_out_hits[best_sector] / held_out_totals[best_sector]
        )
        directional_counts[unit_index] = int(held_out_totals[best_sector])
    return traversal_reliability, traversal_counts, directional_reliability, directional_counts


def compute_reliability_lift_maps(
    reliability_maps: np.ndarray,
    visited_episode_counts: np.ndarray,
) -> np.ndarray:
    """Per-bin reliability minus the unit's visit-weighted overall hit rate."""
    finite = np.isfinite(reliability_maps)
    weights = np.where(finite, visited_episode_counts[None, :, :], 0.0).astype(np.float64)
    weighted_hits = np.where(finite, reliability_maps, 0.0).astype(np.float64) * weights
    total_weights = weights.sum(axis=(1, 2))
    overall_hit_rate = np.zeros(reliability_maps.shape[0], dtype=np.float64)
    has_visits = total_weights > 0
    overall_hit_rate[has_visits] = (
        weighted_hits.sum(axis=(1, 2))[has_visits] / total_weights[has_visits]
    )
    return (reliability_maps - overall_hit_rate[:, None, None].astype(np.float32)).astype(
        np.float32, copy=False
    )


def compute_bin_consistency_maps(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    minimum_visited_episodes: int = 2,
    active_bin_peak_fraction: float = 0.05,
    active_episode_threshold_fraction_of_bin_mean: float = 0.5,
    unit_chunk_size: int = 64,
    epsilon: float = 1e-6,
    episode_statistics: EpisodeBinStatistics | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin-local revisit consistency from episode-level mean responses."""
    statistics = episode_statistics or prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    if statistics is None:
        empty_maps, empty_visits = _empty_episode_metric_maps(
            num_units=representation.shape[-1],
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
        )
        return empty_maps, empty_maps.copy(), empty_visits

    consistency_maps = np.full(
        (statistics.num_units, num_bins_y, num_bins_x), np.nan, dtype=np.float32
    )
    coefficient_of_variation_maps = np.full_like(consistency_maps, np.nan)
    step_counts = statistics.episode_bin_step_counts.astype(np.float32, copy=False)
    visited_mask_by_episode = step_counts > 0
    for start_index, stop_index, chunk_activity_sums in iter_episode_activity_sum_chunks(
        statistics,
        unit_chunk_size=unit_chunk_size,
    ):
        (
            consistency_maps[start_index:stop_index],
            coefficient_of_variation_maps[start_index:stop_index],
        ) = _compute_bin_consistency_chunk(
            chunk_activity_sums,
            step_counts,
            visited_mask_by_episode,
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
            minimum_visited_episodes=minimum_visited_episodes,
            active_bin_peak_fraction=active_bin_peak_fraction,
            active_episode_threshold_fraction_of_bin_mean=(
                active_episode_threshold_fraction_of_bin_mean
            ),
            epsilon=epsilon,
        )
    return (
        consistency_maps,
        coefficient_of_variation_maps,
        statistics.visited_episode_counts.reshape(num_bins_y, num_bins_x).astype(
            np.float32, copy=False
        ),
    )


def compute_split_half_agreement_maps(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    *,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    minimum_episodes_per_half: int = 1,
    unit_chunk_size: int = 16,
    epsilon: float = 1e-6,
    episode_statistics: EpisodeBinStatistics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin even/odd split-half agreement maps in [0, 1]."""
    (
        agreement_maps,
        support_counts,
        _correlations,
    ) = compute_split_half_agreement_maps_and_correlations(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
        bounds=bounds,
        minimum_episodes_per_half=minimum_episodes_per_half,
        unit_chunk_size=unit_chunk_size,
        epsilon=epsilon,
        episode_statistics=episode_statistics,
    )
    return agreement_maps, support_counts


def compute_split_half_agreement_maps_and_correlations(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    *,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    minimum_episodes_per_half: int = 1,
    unit_chunk_size: int = 16,
    epsilon: float = 1e-6,
    episode_statistics: EpisodeBinStatistics | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute split-half agreement maps and whole-map correlations in one pass."""
    statistics = episode_statistics or prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    if statistics is None:
        empty_maps, support_counts = _empty_episode_metric_maps(
            num_units=representation.shape[-1],
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
        )
        return (
            empty_maps,
            support_counts,
            np.zeros((representation.shape[-1],), dtype=np.float32),
        )

    agreement_maps = np.full(
        (statistics.num_units, num_bins_y, num_bins_x), np.nan, dtype=np.float32
    )
    correlations = np.zeros((statistics.num_units,), dtype=np.float32)
    step_counts = statistics.episode_bin_step_counts.astype(np.float32, copy=False)
    (
        even_episode_mask,
        odd_episode_mask,
        safe_even_occupancy,
        safe_odd_occupancy,
        support_counts,
        support_mask,
    ) = _split_half_support(
        step_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
        minimum_episodes_per_half=minimum_episodes_per_half,
    )

    for start_index, stop_index, chunk_activity_sums in iter_episode_activity_sum_chunks(
        statistics,
        unit_chunk_size=unit_chunk_size,
    ):
        agreement_chunk, correlation_chunk = _compute_split_half_agreement_chunk(
            chunk_activity_sums[:, even_episode_mask, :].sum(axis=1, dtype=np.float32),
            chunk_activity_sums[:, odd_episode_mask, :].sum(axis=1, dtype=np.float32),
            safe_even_occupancy,
            safe_odd_occupancy,
            support_mask,
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
            smoothing_sigma=smoothing_sigma,
            epsilon=epsilon,
        )
        agreement_maps[start_index:stop_index] = agreement_chunk
        correlations[start_index:stop_index] = correlation_chunk
    return agreement_maps, support_counts, correlations


def compute_episode_rate_map_correlations(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    *,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    unit_chunk_size: int = 8,
    episode_statistics: EpisodeBinStatistics | None = None,
) -> np.ndarray:
    """Compute one leave-one-episode-out rate-map correlation per unit."""
    statistics = episode_statistics or prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    if statistics is None:
        return np.zeros((representation.shape[-1],), dtype=np.float32)

    correlations = np.zeros((statistics.num_units,), dtype=np.float32)
    step_counts = statistics.episode_bin_step_counts.astype(np.float32, copy=False)
    episode_support = _episode_rate_map_correlation_support(
        step_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
    )
    if episode_support is None:
        return correlations
    supported_episode_mask, safe_episode_occupancy, safe_other_occupancy = episode_support
    for start_index, stop_index, chunk_activity_sums in iter_episode_activity_sum_chunks(
        statistics,
        unit_chunk_size=unit_chunk_size,
    ):
        correlations[start_index:stop_index] = _compute_episode_rate_map_correlation_chunk(
            chunk_activity_sums,
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
            smoothing_sigma=smoothing_sigma,
            supported_episode_mask=supported_episode_mask,
            safe_episode_occupancy=safe_episode_occupancy,
            safe_other_occupancy=safe_other_occupancy,
        )
    return correlations.astype(np.float32, copy=False)


def compute_revisit_activity_metrics(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    *,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    minimum_visited_episodes: int = 2,
    active_bin_peak_fraction: float = 0.05,
    active_episode_threshold_fraction_of_bin_mean: float = 0.5,
    minimum_episodes_per_half: int = 1,
    unit_chunk_size: int = 8,
    epsilon: float = 1e-6,
    episode_statistics: EpisodeBinStatistics | None = None,
    num_random_splits: int = 20,
    rng_seed: int = 0,
) -> RevisitActivityMetrics:
    """Compute revisit metrics in one pass over episode/bin activity sums."""
    statistics = episode_statistics or prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    if statistics is None:
        empty_maps, empty_visits = _empty_episode_metric_maps(
            num_units=representation.shape[-1],
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
        )
        empty_correlations = np.full((representation.shape[-1],), np.nan, dtype=np.float32)
        return RevisitActivityMetrics(
            bin_consistency_maps=empty_maps,
            bin_coefficient_of_variation_maps=empty_maps.copy(),
            consistency_visit_counts=empty_visits,
            split_half_agreement_maps=empty_maps.copy(),
            split_half_agreement_support_counts=empty_visits.copy(),
            split_half_rate_map_correlation=empty_correlations,
            episode_rate_map_correlation=empty_correlations.copy(),
        )

    bin_consistency_maps = np.full(
        (statistics.num_units, num_bins_y, num_bins_x),
        np.nan,
        dtype=np.float32,
    )
    bin_coefficient_of_variation_maps = np.full_like(bin_consistency_maps, np.nan)
    split_half_agreement_maps = np.full_like(bin_consistency_maps, np.nan)
    episode_rate_map_correlation = np.full((statistics.num_units,), np.nan, dtype=np.float32)

    step_counts = statistics.episode_bin_step_counts.astype(np.float32, copy=False)
    visited_mask_by_episode = step_counts > 0
    (
        even_episode_mask,
        odd_episode_mask,
        safe_even_occupancy,
        safe_odd_occupancy,
        split_half_agreement_support_counts,
        support_mask,
    ) = _split_half_support(
        step_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
        minimum_episodes_per_half=minimum_episodes_per_half,
    )
    episode_support = _episode_rate_map_correlation_support(
        step_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
    )
    split_supports = _prepare_balanced_split_supports(
        step_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
        minimum_episodes_per_half=max(int(minimum_episodes_per_half), 1),
        num_random_splits=num_random_splits,
        rng_seed=rng_seed,
    )
    split_correlation_sums = np.zeros((statistics.num_units,), dtype=np.float64)
    split_correlation_counts = np.zeros((statistics.num_units,), dtype=np.int64)

    unit_chunk_bounds = _unit_chunk_bounds(statistics.num_units, unit_chunk_size)

    def process_unit_chunks(chunk_indices: range) -> None:
        for chunk_index in chunk_indices:
            start_index, stop_index = unit_chunk_bounds[chunk_index]
            chunk_activity_sums = _episode_activity_sums(statistics, start_index, stop_index)
            (
                bin_consistency_maps[start_index:stop_index],
                bin_coefficient_of_variation_maps[start_index:stop_index],
            ) = _compute_bin_consistency_chunk(
                chunk_activity_sums,
                step_counts,
                visited_mask_by_episode,
                num_bins_y=num_bins_y,
                num_bins_x=num_bins_x,
                minimum_visited_episodes=minimum_visited_episodes,
                active_bin_peak_fraction=active_bin_peak_fraction,
                active_episode_threshold_fraction_of_bin_mean=(
                    active_episode_threshold_fraction_of_bin_mean
                ),
                epsilon=epsilon,
            )
            (
                split_half_agreement_maps[start_index:stop_index],
                _even_odd_correlation_chunk,
            ) = _compute_split_half_agreement_chunk(
                chunk_activity_sums[:, even_episode_mask, :].sum(axis=1, dtype=np.float32),
                chunk_activity_sums[:, odd_episode_mask, :].sum(axis=1, dtype=np.float32),
                safe_even_occupancy,
                safe_odd_occupancy,
                support_mask,
                num_bins_y=num_bins_y,
                num_bins_x=num_bins_x,
                smoothing_sigma=smoothing_sigma,
                epsilon=epsilon,
            )
            _accumulate_multi_split_correlations(
                chunk_activity_sums,
                split_supports,
                split_correlation_sums[start_index:stop_index],
                split_correlation_counts[start_index:stop_index],
                num_bins_y=num_bins_y,
                num_bins_x=num_bins_x,
                smoothing_sigma=smoothing_sigma,
            )
            if episode_support is not None:
                supported_episode_mask, safe_episode_occupancy, safe_other_occupancy = (
                    episode_support
                )
                episode_rate_map_correlation[start_index:stop_index] = (
                    _compute_episode_rate_map_correlation_chunk(
                        chunk_activity_sums,
                        supported_episode_mask,
                        safe_episode_occupancy,
                        safe_other_occupancy,
                        num_bins_y=num_bins_y,
                        num_bins_x=num_bins_x,
                        smoothing_sigma=smoothing_sigma,
                    )
                )

    run_over_index_blocks(len(unit_chunk_bounds), process_unit_chunks)

    split_half_rate_map_correlation = _finalize_multi_split_correlations(
        split_correlation_sums, split_correlation_counts
    )
    visit_counts = statistics.visited_episode_counts.reshape(num_bins_y, num_bins_x).astype(
        np.float32,
        copy=False,
    )
    return RevisitActivityMetrics(
        bin_consistency_maps=bin_consistency_maps,
        bin_coefficient_of_variation_maps=bin_coefficient_of_variation_maps,
        consistency_visit_counts=visit_counts,
        split_half_agreement_maps=split_half_agreement_maps,
        split_half_agreement_support_counts=split_half_agreement_support_counts,
        split_half_rate_map_correlation=split_half_rate_map_correlation,
        episode_rate_map_correlation=episode_rate_map_correlation.astype(np.float32, copy=False),
    )
