"""Composite place-cell quality metrics built from robust ingredients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.angles import wrap_radians
from .rate_map_kernels import (
    correlation_from_centered_parts,
    flatten_valid_steps,
    flatten_vector_field,
    read_float_setting,
)
from .work_blocks import run_over_index_blocks


def spatial_coherence(rate_map: np.ndarray, epsilon: float = 1e-8) -> float:
    """Return first-order spatial coherence for a single rate map."""
    if rate_map.ndim != 2:
        raise ValueError(f"spatial_coherence expects a [Y, X] map, got {rate_map.shape}.")
    finite_mask = np.isfinite(rate_map)
    if not np.any(finite_mask):
        return float("nan")

    neighbor_sum = np.zeros_like(rate_map, dtype=np.float32)
    neighbor_count = np.zeros_like(rate_map, dtype=np.int32)

    north = rate_map[:-1, :]
    north_valid = np.isfinite(north)
    south_targets_sum = neighbor_sum[1:, :]
    south_targets_count = neighbor_count[1:, :]
    south_targets_sum[north_valid] += north[north_valid]
    south_targets_count[north_valid] += 1

    south = rate_map[1:, :]
    south_valid = np.isfinite(south)
    north_targets_sum = neighbor_sum[:-1, :]
    north_targets_count = neighbor_count[:-1, :]
    north_targets_sum[south_valid] += south[south_valid]
    north_targets_count[south_valid] += 1

    west = rate_map[:, :-1]
    west_valid = np.isfinite(west)
    east_targets_sum = neighbor_sum[:, 1:]
    east_targets_count = neighbor_count[:, 1:]
    east_targets_sum[west_valid] += west[west_valid]
    east_targets_count[west_valid] += 1

    east = rate_map[:, 1:]
    east_valid = np.isfinite(east)
    west_targets_sum = neighbor_sum[:, :-1]
    west_targets_count = neighbor_count[:, :-1]
    west_targets_sum[east_valid] += east[east_valid]
    west_targets_count[east_valid] += 1

    valid_pairs = finite_mask & (neighbor_count > 0)
    if int(valid_pairs.sum()) < 2:
        return 0.0

    neighbor_mean = np.divide(
        neighbor_sum,
        neighbor_count,
        out=np.zeros_like(neighbor_sum, dtype=np.float32),
        where=neighbor_count > 0,
    )
    values = rate_map[valid_pairs].astype(np.float32, copy=False)
    neighbor_values = neighbor_mean[valid_pairs].astype(np.float32, copy=False)
    centered_values = values - values.mean()
    centered_neighbors = neighbor_values - neighbor_values.mean()
    denominator = float(
        np.sqrt(np.square(centered_values).sum() * np.square(centered_neighbors).sum())
    )
    if denominator <= epsilon:
        return 0.0
    return float((centered_values @ centered_neighbors) / denominator)


def batched_spatial_coherence(rate_maps: np.ndarray) -> np.ndarray:
    """Return spatial coherence for [N, Y, X] rate maps."""
    if rate_maps.ndim != 3:
        raise ValueError(f"batched_spatial_coherence expects [N, Y, X], got {rate_maps.shape}.")
    return np.asarray([spatial_coherence(rate_map) for rate_map in rate_maps], dtype=np.float32)


def _position_bin_indices(flat_positions: np.ndarray, bins_per_axis: int) -> np.ndarray:
    """Flat row-major bin index per sample on a uniform grid over the visited extent."""
    lower = flat_positions.min(axis=0)
    span = np.maximum(flat_positions.max(axis=0) - lower, 1e-9)
    scaled = ((flat_positions - lower) / span * bins_per_axis).astype(np.int64)
    scaled = np.clip(scaled, 0, bins_per_axis - 1)
    return scaled[:, 1] * bins_per_axis + scaled[:, 0]


def _multiple_correlation(
    centered_units: np.ndarray,
    centered_regressors: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Per-unit multiple correlation R against the joint regressor set."""
    coefficients, *_ = np.linalg.lstsq(centered_regressors, centered_units, rcond=None)
    fitted = centered_regressors @ coefficients
    residual_sum_of_squares = np.square(centered_units - fitted).sum(axis=0)
    total_sum_of_squares = np.square(centered_units).sum(axis=0)
    r_squared = np.zeros(centered_units.shape[1], dtype=np.float32)
    well_defined = total_sum_of_squares > epsilon
    r_squared[well_defined] = 1.0 - (
        residual_sum_of_squares[well_defined] / total_sum_of_squares[well_defined]
    )
    return np.sqrt(np.clip(r_squared, 0.0, 1.0)).astype(np.float32, copy=False)


@dataclass(slots=True)
class _PositionBinLayout:
    """Sort and segment bookkeeping for position-bin means, resolved once per call."""

    bin_indices: np.ndarray
    sort_order: np.ndarray
    occupied_bins: np.ndarray
    segment_starts: np.ndarray
    segment_counts: np.ndarray
    num_bins_total: int


def _position_bin_layout(bin_indices: np.ndarray, num_bins_total: int) -> _PositionBinLayout:
    """Sort the samples by position bin and record where each bin's run of samples starts."""
    sort_order = np.argsort(bin_indices, kind="stable")
    sorted_bins = bin_indices[sort_order]
    occupied_bins, segment_starts = np.unique(sorted_bins, return_index=True)
    return _PositionBinLayout(
        bin_indices=bin_indices,
        sort_order=sort_order,
        occupied_bins=occupied_bins,
        segment_starts=segment_starts,
        segment_counts=np.diff(np.append(segment_starts, len(sorted_bins))).astype(np.float32),
        num_bins_total=num_bins_total,
    )


def _subtract_position_bin_means(
    columns: np.ndarray,
    layout: _PositionBinLayout,
) -> np.ndarray:
    """Subtract each sample's position-bin mean from a [T, D] block of columns."""
    block = columns.astype(np.float32, copy=False)
    bin_sums = np.add.reduceat(block[layout.sort_order], layout.segment_starts, axis=0)
    bin_means = np.zeros((layout.num_bins_total, block.shape[1]), dtype=np.float32)
    bin_means[layout.occupied_bins] = bin_sums / layout.segment_counts[:, None]
    return block - bin_means[layout.bin_indices]


def _position_bin_residuals(values: np.ndarray, layout: _PositionBinLayout) -> np.ndarray:
    """Subtract each sample's position-bin mean; values is [T] or [T, D]."""
    flattened = values.reshape(len(values), -1)
    return _subtract_position_bin_means(flattened, layout).reshape(values.shape)


_CONFOUND_CHUNK_ELEMENT_BUDGET = 1 << 22


def _confound_unit_chunks(num_units: int, num_steps: int) -> list[tuple[int, int]]:
    """Near-equal [start, stop) unit ranges the confound scores are computed in."""
    if num_units <= 3:
        return [(0, num_units)] if num_units else []
    target_width = max(2, min(num_units, _CONFOUND_CHUNK_ELEMENT_BUDGET // max(1, num_steps)))
    chunk_count = min(-(-num_units // target_width), num_units // 2)
    edges = [num_units * index // chunk_count for index in range(chunk_count + 1)]
    return list(zip(edges[:-1], edges[1:], strict=False))


def _confound_correlation(
    centered_unit_major: np.ndarray,
    unit_sum_of_squares: np.ndarray,
    confound: np.ndarray,
) -> np.ndarray:
    """|corr| of every unit against one residualized confound, from the shared centered matrix."""
    centered_confound = confound - confound.mean()
    return np.abs(
        correlation_from_centered_parts(
            centered_unit_major @ centered_confound,
            unit_sum_of_squares,
            float(np.square(centered_confound).sum()),
        )
    )


def compute_available_confound_scores(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    kinematics: np.ndarray | None = None,
    heading: np.ndarray | None = None,
    position_bins_per_axis: int = 20,
) -> dict[str, np.ndarray]:
    """Return per-unit confound scores as PARTIAL correlations controlling for position."""
    flattened, flat_positions = flatten_valid_steps(representation, position_xy, valid_mask)
    layout = _position_bin_layout(
        _position_bin_indices(flat_positions, position_bins_per_axis),
        position_bins_per_axis * position_bins_per_axis,
    )
    num_steps, num_units = flattened.shape
    time_index = _position_bin_residuals(np.arange(num_steps, dtype=np.float32), layout)
    flattened_kinematics = flatten_vector_field(kinematics, valid_mask)
    flattened_heading = flatten_vector_field(heading, valid_mask)

    step_displacement = None
    if (
        flattened_kinematics is not None
        and flattened_kinematics.ndim == 2
        and flattened_kinematics.shape[-1] > 0
    ):
        step_displacement = _position_bin_residuals(
            flattened_kinematics[:, 0].astype(np.float32, copy=False),
            layout,
        )
    centered_heading = None
    if flattened_heading is not None:
        heading_components = np.asarray(
            flattened_heading,
            dtype=np.float32,
        ).reshape(num_steps, -1)
        if heading_components.shape[1] == 1:
            wrapped_heading = wrap_radians(heading_components[:, 0])
            heading_components = np.stack(
                [np.sin(wrapped_heading), np.cos(wrapped_heading)],
                axis=-1,
            ).astype(np.float32, copy=False)
        heading_values = _position_bin_residuals(heading_components, layout)
        centered_heading = heading_values - heading_values.mean(axis=0, keepdims=True)

    heading_score = np.full(num_units, np.nan, dtype=np.float32)
    centered_units = np.empty((num_steps, num_units), dtype=np.float32)
    unit_sum_of_squares = np.empty(num_units, dtype=np.float32)
    unit_chunks = _confound_unit_chunks(num_units, num_steps)

    def center_and_regress_unit_chunks(chunk_indices: range) -> None:
        for chunk_index in chunk_indices:
            start, stop = unit_chunks[chunk_index]
            residual_units = _subtract_position_bin_means(flattened[:, start:stop], layout)
            chunk_centered = residual_units - residual_units.mean(axis=0, keepdims=True)
            centered_units[:, start:stop] = chunk_centered
            unit_sum_of_squares[start:stop] = np.sum(np.square(chunk_centered.T), axis=1)
            if centered_heading is not None:
                heading_score[start:stop] = _multiple_correlation(chunk_centered, centered_heading)

    run_over_index_blocks(len(unit_chunks), center_and_regress_unit_chunks)

    centered_unit_major = centered_units.T
    time_score = _confound_correlation(centered_unit_major, unit_sum_of_squares, time_index)
    step_displacement_score = (
        np.full(num_units, np.nan, dtype=np.float32)
        if step_displacement is None
        else _confound_correlation(centered_unit_major, unit_sum_of_squares, step_displacement)
    )
    max_available_confound_score = np.fmax.reduce(
        [step_displacement_score, heading_score, time_score]
    ).astype(
        np.float32,
        copy=False,
    )
    return {
        "step_displacement_score": step_displacement_score,
        "heading_score": heading_score,
        "time_score": time_score,
        "max_available_confound_score": max_available_confound_score,
    }


def reliability_weighted_information(
    spatial_information_bits: np.ndarray,
    split_half_correlation: np.ndarray,
) -> np.ndarray:
    """Trust spatial information only to the extent it replicates."""
    return np.asarray(
        spatial_information_bits * np.maximum(split_half_correlation, 0.0),
        dtype=np.float32,
    )


def reliability_weighted_information_excess(
    spatial_information_bits: np.ndarray,
    spatial_information_null_95: np.ndarray,
    split_half_correlation: np.ndarray,
) -> np.ndarray:
    """RWI with the bits factor zero-referenced at the unit's own shuffle null."""
    excess_bits = spatial_information_bits - spatial_information_null_95
    return np.asarray(
        np.maximum(excess_bits, 0.0) * np.maximum(split_half_correlation, 0.0),
        dtype=np.float32,
    )


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH step-up FDR control: boolean rejected mask."""
    p_array = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(p_array)
    rejected = np.zeros(p_array.shape, dtype=bool)
    num_tests = int(finite.sum())
    if num_tests == 0:
        return rejected
    sorted_p = np.sort(p_array[finite])
    thresholds = alpha * np.arange(1, num_tests + 1) / num_tests
    below = sorted_p <= thresholds
    if not below.any():
        return rejected
    cutoff = sorted_p[np.max(np.nonzero(below)[0])]
    rejected[finite] = p_array[finite] <= cutoff
    return rejected


DEFAULT_GATE_MINIMUM_SPLIT_HALF = 0.8


DEFAULT_GATE_MINIMUM_COHERENCE = 0.3


DEFAULT_GATE_MAXIMUM_CONFOUND = 0.5


@dataclass(frozen=True, slots=True)
class PlaceCellGateThresholds:
    """The three thresholds the gate reads, resolved once and carried to every path."""

    minimum_split_half: float = DEFAULT_GATE_MINIMUM_SPLIT_HALF
    minimum_coherence: float = DEFAULT_GATE_MINIMUM_COHERENCE
    maximum_confound: float = DEFAULT_GATE_MAXIMUM_CONFOUND


DEFAULT_PLACE_CELL_GATE_THRESHOLDS = PlaceCellGateThresholds()


def resolve_place_cell_gate_thresholds(analysis_config: Any) -> PlaceCellGateThresholds:
    return PlaceCellGateThresholds(
        minimum_split_half=read_float_setting(
            analysis_config, "place_cell_gate_minimum_split_half", DEFAULT_GATE_MINIMUM_SPLIT_HALF
        ),
        minimum_coherence=read_float_setting(
            analysis_config, "place_cell_gate_minimum_coherence", DEFAULT_GATE_MINIMUM_COHERENCE
        ),
        maximum_confound=read_float_setting(
            analysis_config, "place_cell_gate_maximum_confound", DEFAULT_GATE_MAXIMUM_CONFOUND
        ),
    )


def place_cell_pass_mask(
    split_half_correlation: np.ndarray,
    spatial_coherence_scores: np.ndarray,
    max_available_confound_score: np.ndarray,
    *,
    supported_mask: np.ndarray,
    minimum_split_half: float = DEFAULT_GATE_MINIMUM_SPLIT_HALF,
    minimum_coherence: float = DEFAULT_GATE_MINIMUM_COHERENCE,
    maximum_confound: float = DEFAULT_GATE_MAXIMUM_CONFOUND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit gate decisions: (passes, assessable, replicates, coherent, unconfounded)."""
    supported = np.asarray(supported_mask, dtype=bool)
    replicates = np.asarray(split_half_correlation) >= minimum_split_half
    coherent = np.asarray(spatial_coherence_scores) >= minimum_coherence
    unconfounded = np.asarray(max_available_confound_score) <= maximum_confound
    assessable = (
        supported
        & np.isfinite(split_half_correlation)
        & np.isfinite(spatial_coherence_scores)
        & np.isfinite(max_available_confound_score)
    )
    passes = replicates & coherent & unconfounded & assessable
    return passes, assessable, replicates, coherent, unconfounded


def field_coverage_fraction(
    rate_maps: np.ndarray,
    raw_occupancy: np.ndarray,
    qualifying_mask: np.ndarray,
    *,
    threshold_fraction: float = 0.2,
) -> float:
    """Fraction of visited bins inside at least one qualifying unit's field."""
    if rate_maps.ndim != 3:
        raise ValueError(f"field_coverage_fraction expects [N, Y, X] maps, got {rate_maps.shape}.")
    visited = np.asarray(raw_occupancy) > 0.0
    qualifying = np.asarray(qualifying_mask, dtype=bool)
    if not visited.any():
        return float("nan")
    if not qualifying.any():
        return 0.0
    maps = np.clip(np.nan_to_num(rate_maps[qualifying], nan=0.0), 0.0, None)
    peaks = maps.reshape(maps.shape[0], -1).max(axis=1)
    has_signal = peaks > 1e-8
    if not has_signal.any():
        return 0.0
    covered = (maps[has_signal] >= threshold_fraction * peaks[has_signal][:, None, None]).any(
        axis=0
    )
    return float(covered[visited].mean())


def fraction_place_cells(
    split_half_correlation: np.ndarray,
    spatial_coherence_scores: np.ndarray,
    max_available_confound_score: np.ndarray,
    *,
    supported_mask: np.ndarray,
    minimum_split_half: float = DEFAULT_GATE_MINIMUM_SPLIT_HALF,
    minimum_coherence: float = DEFAULT_GATE_MINIMUM_COHERENCE,
    maximum_confound: float = DEFAULT_GATE_MAXIMUM_CONFOUND,
) -> dict[str, float]:
    """What FRACTION of units qualify as place cells, plus the reason any of them failed."""
    passes, assessable, replicates, coherent, unconfounded = place_cell_pass_mask(
        split_half_correlation,
        spatial_coherence_scores,
        max_available_confound_score,
        supported_mask=supported_mask,
        minimum_split_half=minimum_split_half,
        minimum_coherence=minimum_coherence,
        maximum_confound=maximum_confound,
    )
    if not assessable.any():
        return {
            "fraction_place_cells": float("nan"),
            "fraction_passing_split_half": float("nan"),
            "fraction_passing_coherence": float("nan"),
            "fraction_passing_confound": float("nan"),
            "place_cell_assessable_units": 0.0,
        }
    return {
        "fraction_place_cells": float(passes.sum() / assessable.sum()),
        "fraction_passing_split_half": float((replicates & assessable).sum() / assessable.sum()),
        "fraction_passing_coherence": float((coherent & assessable).sum() / assessable.sum()),
        "fraction_passing_confound": float((unconfounded & assessable).sum() / assessable.sum()),
        "place_cell_assessable_units": float(assessable.sum()),
    }


def coding_purity_score(
    spatial_information_bits: np.ndarray,
    spatial_coherence_scores: np.ndarray,
    split_half_correlation: np.ndarray,
    max_available_confound_score: np.ndarray,
) -> np.ndarray:
    """Strict place-cell quality score from reliable, field-free ingredients."""
    purity_weight = 1.0 - np.clip(max_available_confound_score, 0.0, 1.0)
    return np.asarray(
        spatial_information_bits
        * np.maximum(spatial_coherence_scores, 0.0)
        * np.maximum(split_half_correlation, 0.0)
        * purity_weight,
        dtype=np.float32,
    )
