"""Split-half rate-map reliability: does a unit reproduce its map on half the episodes?"""

from __future__ import annotations

import numpy as np

from .bin_maps import (
    _smooth_flat_bin_maps,
    batched_masked_map_correlation,
    smoothed_safe_occupancy,
)
from .occupancy import (
    EpisodeBinStatistics,
    iter_episode_activity_sum_chunks,
    prepare_episode_bin_statistics,
)


def _balanced_split_masks(
    num_episodes: int,
    num_random_splits: int,
    rng_seed: int,
) -> list[np.ndarray]:
    """Even/odd anchor split plus num_random_splits random balanced episode splits."""
    split_masks = [(np.arange(num_episodes, dtype=np.int64) % 2) == 0]
    if num_episodes >= 4 and num_random_splits > 0:
        rng = np.random.default_rng(rng_seed)
        for _ in range(int(num_random_splits)):
            permuted = rng.permutation(num_episodes)
            mask = np.zeros(num_episodes, dtype=bool)
            mask[permuted[: num_episodes // 2]] = True
            split_masks.append(mask)
    return split_masks


def _prepare_balanced_split_supports(
    step_counts: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
    min_occupancy: float,
    minimum_episodes_per_half: int,
    num_random_splits: int,
    rng_seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per-split (mask, support mask, safe half occupancies) for multi-split correlations."""
    split_supports: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for split_mask in _balanced_split_masks(step_counts.shape[0], num_random_splits, rng_seed):
        first_visits = np.sum(step_counts[split_mask] > 0, axis=0, dtype=np.int32)
        second_visits = np.sum(step_counts[~split_mask] > 0, axis=0, dtype=np.int32)
        support_mask = (
            (first_visits >= int(minimum_episodes_per_half))
            & (second_visits >= int(minimum_episodes_per_half))
        ).reshape(num_bins_y, num_bins_x)
        if not support_mask.any():
            continue
        split_supports.append(
            (
                split_mask,
                support_mask,
                smoothed_safe_occupancy(
                    step_counts[split_mask].sum(axis=0, dtype=np.float32),
                    num_bins_y=num_bins_y,
                    num_bins_x=num_bins_x,
                    smoothing_sigma=smoothing_sigma,
                    min_occupancy=min_occupancy,
                ),
                smoothed_safe_occupancy(
                    step_counts[~split_mask].sum(axis=0, dtype=np.float32),
                    num_bins_y=num_bins_y,
                    num_bins_x=num_bins_x,
                    smoothing_sigma=smoothing_sigma,
                    min_occupancy=min_occupancy,
                ),
            )
        )
    return split_supports


def _accumulate_multi_split_correlations(
    chunk_activity_sums: np.ndarray,
    split_supports: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    correlation_sums: np.ndarray,
    correlation_counts: np.ndarray,
    *,
    num_bins_y: int,
    num_bins_x: int,
    smoothing_sigma: float,
) -> None:
    """Add each supported split's rate-map correlation into the running per-unit sums."""
    if not split_supports:
        return
    half_activity_sums: list[np.ndarray] = []
    safe_half_occupancies: list[np.ndarray] = []
    supported_half_bins: list[np.ndarray] = []
    for split_mask, support_mask, safe_first_occupancy, safe_second_occupancy in split_supports:
        half_activity_sums.append(
            chunk_activity_sums[:, split_mask, :].sum(axis=1, dtype=np.float32)
        )
        half_activity_sums.append(
            chunk_activity_sums[:, ~split_mask, :].sum(axis=1, dtype=np.float32)
        )
        safe_half_occupancies += [safe_first_occupancy, safe_second_occupancy]
        supported_half_bins += [support_mask, support_mask]

    half_maps = _smooth_flat_bin_maps(
        np.stack(half_activity_sums),
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    ) / np.stack(safe_half_occupancies)[:, None, :, :]
    half_maps = np.where(np.stack(supported_half_bins)[:, None, :, :], half_maps, np.nan)

    chunk_width = chunk_activity_sums.shape[0]
    split_correlations = batched_masked_map_correlation(
        half_maps[0::2].reshape(-1, num_bins_y, num_bins_x),
        half_maps[1::2].reshape(-1, num_bins_y, num_bins_x),
    ).reshape(len(split_supports), chunk_width)
    for correlations in split_correlations:
        finite = np.isfinite(correlations)
        correlation_sums[finite] += correlations[finite]
        correlation_counts[finite] += 1


def _finalize_multi_split_correlations(
    correlation_sums: np.ndarray,
    correlation_counts: np.ndarray,
) -> np.ndarray:
    """Mean over supported splits; NaN for units with no supported split."""
    correlations = np.full(correlation_sums.shape, np.nan, dtype=np.float32)
    supported_units = correlation_counts > 0
    correlations[supported_units] = (
        correlation_sums[supported_units] / correlation_counts[supported_units]
    ).astype(np.float32)
    return correlations


def compute_split_half_rate_map_correlations(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    *,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    unit_chunk_size: int = 16,
    episode_statistics: EpisodeBinStatistics | None = None,
    num_random_splits: int = 20,
    rng_seed: int = 0,
    minimum_episodes_per_half: int = 2,
) -> np.ndarray:
    """Split-half rate-map correlation per unit, averaged over balanced episode splits."""
    statistics = episode_statistics or prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    if statistics is None:
        return np.full((representation.shape[-1],), np.nan, dtype=np.float32)

    split_supports = _prepare_balanced_split_supports(
        statistics.episode_bin_step_counts.astype(np.float32, copy=False),
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
        minimum_episodes_per_half=minimum_episodes_per_half,
        num_random_splits=num_random_splits,
        rng_seed=rng_seed,
    )
    correlation_sums = np.zeros((statistics.num_units,), dtype=np.float64)
    correlation_counts = np.zeros((statistics.num_units,), dtype=np.int64)
    for start_index, stop_index, chunk_activity_sums in iter_episode_activity_sum_chunks(
        statistics,
        unit_chunk_size=unit_chunk_size,
    ):
        _accumulate_multi_split_correlations(
            chunk_activity_sums,
            split_supports,
            correlation_sums[start_index:stop_index],
            correlation_counts[start_index:stop_index],
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
            smoothing_sigma=smoothing_sigma,
        )
    return _finalize_multi_split_correlations(correlation_sums, correlation_counts)
