"""Episode-preserving circular-shift nulls for spatial-information scores."""

from __future__ import annotations

import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix

from placecell_research.numerics.work_blocks import analysis_worker_count, run_over_index_blocks

from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    skaggs_spatial_information,
)


def circular_shift_step_layout(
    episode_lengths: np.ndarray,
    step_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Everything a circular roll needs about each flat step, listed in step_order."""
    episode_starts = np.concatenate(([0], np.cumsum(episode_lengths)[:-1]))
    step_episode = np.repeat(np.arange(episode_lengths.size), episode_lengths)
    step_within_episode = np.arange(step_episode.size) - episode_starts[step_episode]
    return (
        step_episode[step_order],
        episode_starts[step_episode][step_order],
        episode_lengths[step_episode][step_order],
        step_within_episode[step_order],
    )


_CIRCULAR_SHIFT_GATHER_ELEMENT_BUDGET = 1 << 25


def _flat_valid_activity_and_positions(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid: np.ndarray,
    *,
    contributing_episodes: np.ndarray,
    episode_lengths: np.ndarray,
    selected_units: np.ndarray,
    num_selected_units: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid steps of every contributing episode, concatenated, as [steps, selected units]."""
    num_units = representation.shape[-1]
    all_units_selected = num_selected_units == num_units
    if all_units_selected and contributing_episodes.size == representation.shape[0] and valid.all():
        return (
            representation.reshape(-1, num_units).astype(np.float32, copy=False),
            position_xy.reshape(-1, position_xy.shape[-1]),
        )
    total_valid_steps = int(episode_lengths.sum())
    flat_activity = np.empty((total_valid_steps, num_selected_units), dtype=np.float32)
    flat_positions = np.empty((total_valid_steps, position_xy.shape[-1]), dtype=position_xy.dtype)
    next_row = 0
    for episode_index, episode_valid_steps in zip(
        contributing_episodes, episode_lengths, strict=False
    ):
        episode_valid = valid[episode_index]
        episode_activity = representation[episode_index, episode_valid]
        flat_activity[next_row : next_row + episode_valid_steps] = (
            episode_activity if all_units_selected else episode_activity[:, selected_units]
        )
        flat_positions[next_row : next_row + episode_valid_steps] = position_xy[
            episode_index, episode_valid
        ]
        next_row += int(episode_valid_steps)
    return flat_activity, flat_positions


def _segment_blocks(segment_bounds: np.ndarray, rows_per_block: int) -> list[tuple[int, int]]:
    """Group whole bin segments into blocks of at most rows_per_block rows."""
    blocks: list[tuple[int, int]] = []
    num_segments = int(segment_bounds.size) - 1
    block_start = 0
    for segment_index in range(num_segments):
        if (
            segment_index > block_start
            and segment_bounds[segment_index + 1] - segment_bounds[block_start] > rows_per_block
        ):
            blocks.append((block_start, segment_index))
            block_start = segment_index
    if block_start < num_segments:
        blocks.append((block_start, num_segments))
    return blocks


def _log_circular_shift_working_set(
    *,
    total_valid_steps: int,
    num_selected_units: int,
    activity_bytes: int,
    max_block_rows: int,
    num_blocks: int,
    num_shuffles: int,
    gather_bytes_per_row: int,
) -> None:
    """Estimate gather storage, excluding sparse pointers, rate maps and other working arrays."""
    workers = analysis_worker_count(num_shuffles)
    activity_gib = activity_bytes / 1024**3
    worker_gib = max_block_rows * gather_bytes_per_row / 1024**3
    print(
        "[circular_shift_null] "
        f"valid_steps={total_valid_steps} units={num_selected_units} "
        f"activity_gib={activity_gib:.2f} "
        f"block_rows={max_block_rows} blocks={num_blocks} workers={workers} "
        f"worker_gib={worker_gib:.2f} "
        f"peak_gib={activity_gib + workers * worker_gib:.2f}",
        file=sys.stderr,
        flush=True,
    )


def draw_circular_shift_offsets(
    episode_lengths: np.ndarray,
    rng: np.random.Generator,
    min_shift_fraction: float,
) -> np.ndarray:
    """One roll offset per episode; episodes shorter than two steps are never shifted."""
    offsets = np.zeros(episode_lengths.size, dtype=np.int64)
    for episode_index, episode_length in enumerate(episode_lengths):
        if episode_length < 2:
            continue
        minimum_shift = max(1, int(episode_length * min_shift_fraction))
        if 2 * minimum_shift >= episode_length:
            minimum_shift = 1
        offsets[episode_index] = int(
            rng.integers(minimum_shift, episode_length - minimum_shift + 1)
        )
    return offsets


def circular_shift_spatial_information_null(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    num_shuffles: int = 100,
    rng_seed: int = 0,
    min_shift_fraction: float = 0.05,
    unit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Null Skaggs distribution from episode-preserving circular activity shifts."""
    valid = (
        np.ones(representation.shape[:2], dtype=bool)
        if valid_mask is None
        else valid_mask.astype(bool, copy=False)
    )
    num_units = representation.shape[-1]
    selected_units = (
        np.ones(num_units, dtype=bool) if unit_mask is None else np.asarray(unit_mask, dtype=bool)
    )
    null_scores = np.full((num_shuffles, num_units), np.nan, dtype=np.float32)
    episode_lengths = np.count_nonzero(valid, axis=1).astype(np.int64)
    contributing_episodes = np.flatnonzero(episode_lengths > 0)
    if contributing_episodes.size == 0 or not selected_units.any():
        return null_scores
    episode_lengths = episode_lengths[contributing_episodes]
    num_selected_units = int(selected_units.sum())
    flat_activity, flat_positions = _flat_valid_activity_and_positions(
        representation,
        position_xy,
        valid,
        contributing_episodes=contributing_episodes,
        episode_lengths=episode_lengths,
        selected_units=selected_units,
        num_selected_units=num_selected_units,
    )
    linear_bins, _, _, _ = compute_spatial_bin_assignments(
        flat_positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    num_bins = num_bins_x * num_bins_y
    occupancy = (
        np.bincount(linear_bins, minlength=num_bins)
        .reshape(num_bins_y, num_bins_x)
        .astype(np.float32, copy=False)
    )
    if smoothing_sigma > 0.0:
        occupancy = gaussian_filter(occupancy, sigma=smoothing_sigma)
    safe_occupancy = np.where(occupancy >= min_occupancy, occupancy, np.nan)

    sort_order = np.argsort(linear_bins, kind="stable")
    occupied_bins, segment_starts = np.unique(linear_bins[sort_order], return_index=True)

    (
        step_episode,
        episode_start,
        episode_length,
        step_within_episode,
    ) = circular_shift_step_layout(episode_lengths, sort_order)

    rng = np.random.default_rng(rng_seed)
    shuffle_offsets = [
        draw_circular_shift_offsets(episode_lengths, rng, min_shift_fraction)
        for _ in range(num_shuffles)
    ]
    total_valid_steps = int(flat_activity.shape[0])
    use_sparse = np.count_nonzero(flat_activity) <= flat_activity.size // 16
    gather_bytes_per_row = 2 * num_selected_units * 4
    activity_bytes = flat_activity.nbytes
    if use_sparse:
        flat_activity = csr_matrix(flat_activity)
        activity_bytes = (
            flat_activity.data.nbytes + flat_activity.indices.nbytes + flat_activity.indptr.nbytes
        )
        gather_bytes_per_row = num_selected_units * (4 + 2 * (4 + flat_activity.indices.itemsize))
    segment_bounds = np.append(segment_starts, total_valid_steps)
    rows_per_block = max(1, 4 * _CIRCULAR_SHIFT_GATHER_ELEMENT_BUDGET // gather_bytes_per_row)
    segment_blocks = _segment_blocks(segment_bounds, rows_per_block)
    max_block_rows = max(
        (
            int(segment_bounds[last_segment] - segment_bounds[first_segment])
            for first_segment, last_segment in segment_blocks
        ),
        default=0,
    )
    _log_circular_shift_working_set(
        total_valid_steps=total_valid_steps,
        num_selected_units=num_selected_units,
        activity_bytes=activity_bytes,
        max_block_rows=max_block_rows,
        num_blocks=len(segment_blocks),
        num_shuffles=num_shuffles,
        gather_bytes_per_row=gather_bytes_per_row,
    )

    def score_shuffles(shuffle_indices: range) -> None:
        activity_sums = np.empty((occupied_bins.size, num_selected_units), dtype=np.float32)
        for shuffle_index in shuffle_indices:
            offsets = shuffle_offsets[shuffle_index]
            for first_segment, last_segment in segment_blocks:
                block = slice(int(segment_bounds[first_segment]), int(segment_bounds[last_segment]))
                block_activity = flat_activity[
                    episode_start[block]
                    + (
                        (step_within_episode[block] - offsets[step_episode[block]])
                        % episode_length[block]
                    )
                ]
                block_activity = (
                    block_activity.toarray(order="F").T
                    if use_sparse
                    else np.ascontiguousarray(block_activity.T)
                )
                activity_sums[first_segment:last_segment] = np.add.reduceat(
                    block_activity,
                    segment_bounds[first_segment:last_segment] - segment_bounds[first_segment],
                    axis=1,
                ).T
                del block_activity
            activity_maps = np.zeros((num_bins, num_selected_units), dtype=np.float32)
            activity_maps[occupied_bins] = activity_sums
            activity_maps = activity_maps.T.reshape(
                num_selected_units,
                num_bins_y,
                num_bins_x,
            )
            if smoothing_sigma > 0.0:
                activity_maps = gaussian_filter(
                    activity_maps, sigma=(0.0, smoothing_sigma, smoothing_sigma)
                )
            rate_maps = activity_maps / safe_occupancy[None, :, :]
            null_scores[shuffle_index, selected_units] = skaggs_spatial_information(
                rate_maps, occupancy
            )

    run_over_index_blocks(num_shuffles, score_shuffles)
    return null_scores
