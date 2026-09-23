"""Episode/bin occupancy bookkeeping shared by the revisit analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rate_map_kernels import (
    _all_steps_valid,
    compute_spatial_bin_assignments,
    flatten_positions,
    infer_bounds,
)


@dataclass(slots=True)
class EpisodeBinStatistics:
    """Flattened episode/bin bookkeeping shared across reliability analyses."""

    flat_values: np.ndarray
    episode_bin_codes: np.ndarray
    segment_start: np.ndarray
    episode_bin_step_counts: np.ndarray
    visited_episode_counts: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float]]
    num_episodes: int
    num_units: int
    num_bins_x: int
    num_bins_y: int


@dataclass(slots=True)
class EpisodeBinOccupancy:
    """The bin-level half of EpisodeBinStatistics, derived from positions alone."""

    episode_bin_step_counts: np.ndarray
    visited_episode_counts: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float]]


@dataclass(slots=True)
class EpisodeCoverageGate:
    """Which episodes cleared the per-episode gates, and how the rest were turned away."""

    qualifying_episodes: np.ndarray
    episodes_skipped_short: int
    episodes_skipped_sparse: int


def _prepare_episode_bin_statistics(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    episode_bin_occupancy: EpisodeBinOccupancy | None = None,
) -> EpisodeBinStatistics | None:
    """Resolve valid-step episode/bin bookkeeping once for all revisit metrics."""
    num_episodes, num_steps, num_units = representation.shape
    if _all_steps_valid(valid_mask):
        flat_positions = position_xy.reshape(-1, position_xy.shape[-1])
        flat_values = representation.reshape(-1, num_units).astype(np.float32, copy=False)
        episode_ids = np.repeat(np.arange(num_episodes, dtype=np.int32), num_steps)
        segment_start = np.zeros(episode_ids.shape[0], dtype=bool)
        if num_steps > 0:
            segment_start[::num_steps] = True
    else:
        flattened_valid_mask = valid_mask.reshape(-1).astype(bool, copy=False)
        if not np.any(flattened_valid_mask):
            return None
        flat_positions = position_xy.reshape(-1, position_xy.shape[-1])[flattened_valid_mask]
        flat_values = representation.reshape(-1, num_units)[flattened_valid_mask].astype(
            np.float32,
            copy=False,
        )
        episode_ids = np.repeat(np.arange(num_episodes, dtype=np.int32), num_steps)[
            flattened_valid_mask
        ]
        kept_step_ids = np.tile(np.arange(num_steps, dtype=np.int32), num_episodes)[
            flattened_valid_mask
        ]
        segment_start = np.ones(episode_ids.shape[0], dtype=bool)
        segment_start[1:] = (episode_ids[1:] != episode_ids[:-1]) | (
            kept_step_ids[1:] != kept_step_ids[:-1] + 1
        )
    if episode_bin_occupancy is not None:
        resolved_bounds = episode_bin_occupancy.bounds
    else:
        resolved_bounds = infer_bounds(flat_positions) if bounds is None else bounds
    num_bins = num_bins_x * num_bins_y
    linear_bins, _, _, _ = compute_spatial_bin_assignments(
        flat_positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=resolved_bounds,
    )
    episode_bin_codes = episode_ids.astype(np.int64, copy=False) * num_bins + linear_bins.astype(
        np.int64,
        copy=False,
    )
    if episode_bin_occupancy is not None:
        episode_bin_step_counts = episode_bin_occupancy.episode_bin_step_counts
        visited_episode_counts = episode_bin_occupancy.visited_episode_counts
    else:
        episode_bin_step_counts = (
            np.bincount(episode_bin_codes, minlength=num_episodes * num_bins)
            .reshape(num_episodes, num_bins)
            .astype(np.int32, copy=False)
        )
        visited_episode_counts = np.sum(episode_bin_step_counts > 0, axis=0, dtype=np.int32)
    return EpisodeBinStatistics(
        flat_values=flat_values,
        episode_bin_codes=episode_bin_codes,
        segment_start=segment_start,
        episode_bin_step_counts=episode_bin_step_counts,
        visited_episode_counts=visited_episode_counts,
        bounds=resolved_bounds,
        num_episodes=num_episodes,
        num_units=num_units,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )


def reachable_bin_visited_fractions(episode_bin_step_counts: np.ndarray) -> np.ndarray:
    """Per-episode share of the reachable arena each episode actually entered."""
    visited_bins = np.asarray(episode_bin_step_counts) > 0
    reachable_bins = visited_bins.any(axis=0)
    if not reachable_bins.any():
        return np.zeros(visited_bins.shape[0], dtype=np.float64)
    return visited_bins[:, reachable_bins].mean(axis=1)


def gate_episodes_by_coverage(
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    minimum_valid_steps: int,
    minimum_visited_fraction: float,
) -> EpisodeCoverageGate:
    """Episodes with enough valid steps and enough of the reachable arena visited."""
    num_episodes, num_steps = position_xy.shape[:2]
    total_bins = max(num_bins_x * num_bins_y, 1)
    valid_step_counts = (
        np.full(num_episodes, num_steps, dtype=np.int64)
        if valid_mask is None
        else np.count_nonzero(valid_mask, axis=1)
    )
    candidate_episodes = np.flatnonzero(valid_step_counts >= minimum_valid_steps)
    episodes_skipped_short = int(num_episodes - candidate_episodes.size)
    if candidate_episodes.size == 0:
        return EpisodeCoverageGate(
            qualifying_episodes=candidate_episodes.astype(np.int64),
            episodes_skipped_short=episodes_skipped_short,
            episodes_skipped_sparse=0,
        )

    candidate_valid_mask = (
        None if valid_mask is None else valid_mask[candidate_episodes].astype(bool, copy=False)
    )
    candidate_positions = flatten_positions(
        position_xy[candidate_episodes],
        candidate_valid_mask,
    )
    candidate_ids = np.repeat(np.arange(candidate_episodes.size, dtype=np.int64), num_steps)
    if candidate_valid_mask is not None:
        candidate_ids = candidate_ids[candidate_valid_mask.reshape(-1)]
    linear_bins, _, _, _ = compute_spatial_bin_assignments(
        candidate_positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    candidate_bin_step_counts = np.bincount(
        candidate_ids * total_bins + linear_bins,
        minlength=candidate_episodes.size * total_bins,
    ).reshape(candidate_episodes.size, total_bins)
    visited_fractions = reachable_bin_visited_fractions(candidate_bin_step_counts)
    qualifying_episodes = candidate_episodes[
        visited_fractions >= minimum_visited_fraction
    ].astype(np.int64)
    return EpisodeCoverageGate(
        qualifying_episodes=qualifying_episodes,
        episodes_skipped_short=episodes_skipped_short,
        episodes_skipped_sparse=int(candidate_episodes.size - qualifying_episodes.size),
    )


def _unit_chunk_bounds(num_units: int, unit_chunk_size: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) unit ranges the revisit metrics are computed in."""
    chunk_size = max(1, min(int(unit_chunk_size), num_units))
    return [
        (start_index, min(start_index + chunk_size, num_units))
        for start_index in range(0, num_units, chunk_size)
    ]


def _episode_activity_sums(
    statistics: EpisodeBinStatistics,
    start_index: int,
    stop_index: int,
) -> np.ndarray:
    """[units, episodes, bins] activity sums for one unit chunk."""
    num_episode_bins = statistics.num_episodes * statistics.num_bins_x * statistics.num_bins_y
    episode_bin_codes = statistics.episode_bin_codes.astype(np.int64, copy=False)
    chunk_values = statistics.flat_values[:, start_index:stop_index].astype(
        np.float32,
        copy=False,
    )
    chunk_width = stop_index - start_index
    flat_codes = (
        np.arange(chunk_width, dtype=np.int64)[:, None] * num_episode_bins
        + episode_bin_codes[None, :]
    )
    summed = np.bincount(
        flat_codes.reshape(-1),
        weights=chunk_values.T.reshape(-1).astype(np.float64, copy=False),
        minlength=chunk_width * num_episode_bins,
    )
    return summed.reshape(
        chunk_width,
        statistics.num_episodes,
        statistics.num_bins_x * statistics.num_bins_y,
    ).astype(np.float32, copy=False)


def _iter_episode_activity_sum_chunks(
    statistics: EpisodeBinStatistics,
    *,
    unit_chunk_size: int,
):
    """Yield chunked [units, episodes, bins] activity sums for revisit metrics."""
    for start_index, stop_index in _unit_chunk_bounds(statistics.num_units, unit_chunk_size):
        yield start_index, stop_index, _episode_activity_sums(statistics, start_index, stop_index)
