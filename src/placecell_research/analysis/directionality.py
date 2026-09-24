"""Directional modulation of place fields."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.place_cell_quality import benjamini_hochberg
from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    scatter_add_over_units,
)
from ..numerics.work_blocks import run_over_index_blocks
from .base import AnalysisInput, AnalysisResult
from .helpers import write_csv
from .shift_nulls import circular_shift_step_layout, draw_circular_shift_offsets
from .world_overlay import overlay_bounds, resolve_world_overlay

_EPS = 1e-9
_NUM_QUADRANTS = 4
_TWO_PI = 2.0 * np.pi
_ACTIVITY_THRESHOLD = 1e-4
_NULL_CHUNK_NONZERO_BUDGET = 8_000_000
_NULL_MIN_SHIFT_FRACTION = 0.05
_NULL_RNG_SEED = 0


@dataclass(slots=True)
class _DirectionalStatistics:
    """Shared per-(spatial-bin, quadrant) sufficient statistics for observed R and its null."""

    bin_index: np.ndarray
    quadrant: np.ndarray
    rates: np.ndarray
    occupancy_bq: np.ndarray
    occupancy_bin: np.ndarray
    marginal: np.ndarray
    sampled_quadrant: np.ndarray
    sampled_quadrant_count: np.ndarray
    fire_rate: np.ndarray
    negative_fraction: float
    num_spatial_bins: int
    episode_lengths: np.ndarray


def valid_steps_per_episode(
    representation: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    """Valid steps of each contributing episode, in the order the flat rows are laid out."""
    if valid_mask is None:
        num_episodes = representation.shape[0] if representation.ndim == 3 else 1
        num_steps = representation.shape[1] if representation.ndim == 3 else representation.shape[0]
        return np.full(num_episodes, num_steps, dtype=np.int64)
    lengths = np.count_nonzero(np.atleast_2d(valid_mask.astype(bool, copy=False)), axis=1)
    return lengths[lengths > 0].astype(np.int64)


def _prepare_directional_statistics(
    *,
    representation: np.ndarray,
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    min_occupancy_per_quadrant: int,
) -> _DirectionalStatistics | None:
    num_units = int(representation.shape[-1])
    codes = representation.reshape(-1, num_units)
    positions = position_xy.reshape(-1, 2)
    headings = heading.reshape(-1)
    if valid_mask is not None:
        keep = valid_mask.reshape(-1).astype(bool, copy=False)
        codes, positions, headings = codes[keep], positions[keep], headings[keep]
    if codes.shape[0] == 0:
        return None

    negative_fraction = float((codes < 0).mean())
    rates = np.clip(codes, 0.0, None)
    fire_rate = (np.abs(codes) > _ACTIVITY_THRESHOLD).mean(axis=0)

    bin_index, _, _, _ = compute_spatial_bin_assignments(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    quadrant = (np.floor((headings % _TWO_PI) / (np.pi / 2.0)).astype(int)) % _NUM_QUADRANTS
    num_spatial_bins = num_bins_x * num_bins_y
    combined = bin_index * _NUM_QUADRANTS + quadrant

    occupancy_bq = np.bincount(
        combined, minlength=num_spatial_bins * _NUM_QUADRANTS
    ).astype(np.float64).reshape(num_spatial_bins, _NUM_QUADRANTS)
    occupancy_bin = occupancy_bq.sum(axis=1)
    activity = scatter_add_over_units(combined, rates, num_spatial_bins * _NUM_QUADRANTS)
    activity = activity.reshape(num_spatial_bins, _NUM_QUADRANTS, num_units)

    safe_occupancy_bin = np.where(occupancy_bin > 0, occupancy_bin, np.nan)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        marginal = activity.sum(axis=1) / safe_occupancy_bin

    sampled_quadrant = occupancy_bq >= min_occupancy_per_quadrant
    return _DirectionalStatistics(
        bin_index=bin_index,
        quadrant=quadrant,
        rates=rates,
        occupancy_bq=occupancy_bq,
        occupancy_bin=occupancy_bin,
        marginal=marginal,
        sampled_quadrant=sampled_quadrant,
        sampled_quadrant_count=sampled_quadrant.sum(axis=1),
        fire_rate=fire_rate,
        negative_fraction=negative_fraction,
        num_spatial_bins=num_spatial_bins,
        episode_lengths=valid_steps_per_episode(representation, valid_mask),
    )


def _quadrant_means_from_labels(
    statistics: _DirectionalStatistics,
    quadrant_labels: np.ndarray,
) -> np.ndarray:
    """Per-(bin, quadrant, unit) mean rates for a given heading-label assignment."""
    combined = statistics.bin_index * _NUM_QUADRANTS + quadrant_labels
    activity = scatter_add_over_units(
        combined,
        statistics.rates,
        statistics.num_spatial_bins * _NUM_QUADRANTS,
    ).reshape(statistics.num_spatial_bins, _NUM_QUADRANTS, statistics.rates.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return activity / statistics.occupancy_bq[:, :, None]


@dataclass(slots=True)
class _FieldGates:
    """Everything R needs about the bins, resolved once per scoring pass."""

    bin_selection: np.ndarray
    field_mask: np.ndarray
    sampled_quadrant: np.ndarray
    unsampled_quadrant: np.ndarray
    occupancy_bq: np.ndarray
    occupancy_bin: np.ndarray
    peak: np.ndarray
    fire_rate: np.ndarray


def _field_mask_and_peak(
    marginal: np.ndarray,
    bin_selection: np.ndarray,
    field_threshold_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Each unit's peak over every bin of marginal, and its field bins among the selected."""
    finite_marginal = np.isfinite(marginal)
    peak = np.max(np.where(finite_marginal, marginal, -np.inf), axis=0)
    selected_marginal = marginal[bin_selection]
    field_mask = np.isfinite(selected_marginal) & (
        selected_marginal >= float(field_threshold_fraction) * peak[None, :]
    )
    return field_mask, peak


def _resolve_field_gates(
    statistics: _DirectionalStatistics,
    *,
    field_threshold_fraction: float,
    min_heading_quadrants: int,
) -> _FieldGates:
    shared_bin_gate = (
        (statistics.sampled_quadrant_count >= int(min_heading_quadrants))
        & (statistics.occupancy_bin > 0)
    )
    bin_selection = np.flatnonzero(shared_bin_gate)
    field_mask, peak = _field_mask_and_peak(
        statistics.marginal, bin_selection, field_threshold_fraction
    )
    sampled_quadrant = statistics.sampled_quadrant[bin_selection][:, :, None]
    return _FieldGates(
        bin_selection=bin_selection,
        field_mask=field_mask,
        sampled_quadrant=sampled_quadrant,
        unsampled_quadrant=~sampled_quadrant,
        occupancy_bq=statistics.occupancy_bq[bin_selection],
        occupancy_bin=statistics.occupancy_bin[bin_selection],
        peak=peak,
        fire_rate=statistics.fire_rate,
    )


def _modulation_r_from_field_gates(
    gates: _FieldGates,
    mean_per_quadrant: np.ndarray,
    *,
    min_field_bins: int,
    min_fire_rate: float,
) -> np.ndarray:
    """Occupancy-weighted directional modulation R per unit over the gated bins."""
    quadrant_max = np.full(gates.field_mask.shape, -np.inf)
    quadrant_min = np.full(gates.field_mask.shape, np.inf)
    for quadrant in range(_NUM_QUADRANTS):
        means = mean_per_quadrant[:, quadrant]
        sampled = gates.sampled_quadrant[:, quadrant]
        np.maximum(quadrant_max, means, out=quadrant_max, where=sampled)
        np.minimum(quadrant_min, means, out=quadrant_min, where=sampled)
    all_sampled_finite = np.all(
        np.isfinite(mean_per_quadrant) | gates.unsampled_quadrant, axis=1
    )
    usable_bin_unit = gates.field_mask & all_sampled_finite & (quadrant_max > _EPS)

    with np.errstate(invalid="ignore"):
        modulation = (quadrant_max - quadrant_min) / (quadrant_max + quadrant_min + _EPS)
    weighted = np.where(usable_bin_unit, modulation, 0.0) * gates.occupancy_bin[:, None]
    weight_sums = (usable_bin_unit * gates.occupancy_bin[:, None]).sum(axis=0)
    usable_bin_counts = usable_bin_unit.sum(axis=0)

    r_values = np.full(gates.peak.size, np.nan, dtype=np.float32)
    assessable = (
        (gates.fire_rate >= float(min_fire_rate))
        & np.isfinite(gates.peak)
        & (gates.peak > _EPS)
        & (usable_bin_counts >= int(min_field_bins))
        & (weight_sums > 0.0)
    )
    r_values[assessable] = (
        weighted.sum(axis=0)[assessable] / weight_sums[assessable]
    ).astype(np.float32)
    return r_values


def _modulation_r_from_quadrant_means(
    statistics: _DirectionalStatistics,
    mean_per_quadrant: np.ndarray,
    *,
    field_threshold_fraction: float,
    min_heading_quadrants: int,
    min_field_bins: int,
    min_fire_rate: float,
) -> np.ndarray:
    """Occupancy-weighted directional modulation R per unit, fully vectorized."""
    gates = _resolve_field_gates(
        statistics,
        field_threshold_fraction=field_threshold_fraction,
        min_heading_quadrants=min_heading_quadrants,
    )
    return _modulation_r_from_field_gates(
        gates,
        mean_per_quadrant[gates.bin_selection],
        min_field_bins=min_field_bins,
        min_fire_rate=min_fire_rate,
    )


@dataclass(slots=True)
class _OccupiedBinLayout:
    """The occupied spatial bins, and where each sample's (bin, quadrant) lands among them."""

    occupancy_bin: np.ndarray
    selection_rank: np.ndarray
    combined_of_step: np.ndarray
    num_groups: int


def _occupied_bin_layout(
    statistics: _DirectionalStatistics,
    gates: _FieldGates,
) -> _OccupiedBinLayout:
    occupied_bins = np.flatnonzero(statistics.occupancy_bin > 0)
    rank_of_bin = np.full(statistics.num_spatial_bins, -1, dtype=np.int64)
    rank_of_bin[occupied_bins] = np.arange(occupied_bins.size)
    return _OccupiedBinLayout(
        occupancy_bin=statistics.occupancy_bin[occupied_bins],
        selection_rank=rank_of_bin[gates.bin_selection],
        combined_of_step=rank_of_bin[statistics.bin_index] * _NUM_QUADRANTS + statistics.quadrant,
        num_groups=occupied_bins.size * _NUM_QUADRANTS,
    )


@dataclass(slots=True)
class _NonzeroRates:
    """Nonzero clipped rates of one unit chunk."""

    sample_index: np.ndarray
    unit_offset: np.ndarray
    value: np.ndarray


def _nonzero_rates(statistics: _DirectionalStatistics, unit_slice: slice) -> _NonzeroRates:
    chunk = statistics.rates[:, unit_slice]
    sample_index, unit_offset = np.nonzero(chunk)
    return _NonzeroRates(
        sample_index=sample_index,
        unit_offset=unit_offset,
        value=chunk[sample_index, unit_offset].astype(np.float64, copy=False),
    )


def _shifted_activity_sums(
    nonzero_rates: _NonzeroRates,
    layout: _OccupiedBinLayout,
    destination_step: np.ndarray,
    num_units: int,
) -> np.ndarray:
    """Per-(occupied bin, quadrant, unit) summed rates after one circular shift."""
    groups = layout.combined_of_step[destination_step[nonzero_rates.sample_index]]
    return np.bincount(
        groups * num_units + nonzero_rates.unit_offset,
        weights=nonzero_rates.value,
        minlength=layout.num_groups * num_units,
    ).reshape(-1, _NUM_QUADRANTS, num_units)


def _null_modulation_matrix(
    statistics: _DirectionalStatistics,
    gates: _FieldGates,
    *,
    num_shuffles: int,
    field_threshold_fraction: float,
    min_field_bins: int,
    min_fire_rate: float,
) -> np.ndarray:
    """[num_shuffles, num_units] R under episode-preserving circular shifts of the activity."""
    layout = _occupied_bin_layout(statistics, gates)
    step_episode, episode_start, episode_length, step_within_episode = (
        circular_shift_step_layout(
            statistics.episode_lengths, np.arange(statistics.bin_index.size)
        )
    )
    rng = np.random.default_rng(_NULL_RNG_SEED)
    shuffle_offsets = [
        draw_circular_shift_offsets(statistics.episode_lengths, rng, _NULL_MIN_SHIFT_FRACTION)
        for _ in range(num_shuffles)
    ]

    total_units = statistics.rates.shape[1]
    nonzeros_per_unit = max(1, int(np.count_nonzero(statistics.rates)) // total_units)
    chunk_width = int(
        np.clip(_NULL_CHUNK_NONZERO_BUDGET // nonzeros_per_unit, 1, total_units)
    )

    null_matrix = np.full((num_shuffles, total_units), np.nan, dtype=np.float32)
    chunk_starts = list(range(0, total_units, chunk_width))

    def score_unit_chunks(chunk_indices: range) -> None:
        for chunk_index in chunk_indices:
            start = chunk_starts[chunk_index]
            unit_slice = slice(start, min(start + chunk_width, total_units))
            chunk_units = unit_slice.stop - unit_slice.start
            nonzero = _nonzero_rates(statistics, unit_slice)
            chunk_fire_rate = gates.fire_rate[unit_slice]
            for shuffle_index in range(num_shuffles):
                offsets = shuffle_offsets[shuffle_index]
                destination_step = episode_start + (
                    (step_within_episode + offsets[step_episode]) % episode_length
                )
                activity = _shifted_activity_sums(
                    nonzero, layout, destination_step, chunk_units
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    marginal = activity.sum(axis=1) / layout.occupancy_bin[:, None]
                    mean_per_quadrant = (
                        activity[layout.selection_rank] / gates.occupancy_bq[:, :, None]
                    )
                field_mask, peak = _field_mask_and_peak(
                    marginal, layout.selection_rank, field_threshold_fraction
                )
                null_matrix[shuffle_index, unit_slice] = _modulation_r_from_field_gates(
                    _FieldGates(
                        bin_selection=gates.bin_selection,
                        field_mask=field_mask,
                        sampled_quadrant=gates.sampled_quadrant,
                        unsampled_quadrant=gates.unsampled_quadrant,
                        occupancy_bq=gates.occupancy_bq,
                        occupancy_bin=gates.occupancy_bin,
                        peak=peak,
                        fire_rate=chunk_fire_rate,
                    ),
                    mean_per_quadrant,
                    min_field_bins=min_field_bins,
                    min_fire_rate=min_fire_rate,
                )

    run_over_index_blocks(len(chunk_starts), score_unit_chunks)
    return null_matrix


def directional_modulation_per_unit(
    *,
    representation: np.ndarray,
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    min_occupancy_per_quadrant: int,
    min_heading_quadrants: int,
    field_threshold_fraction: float,
    min_field_bins: int,
    min_fire_rate: float,
) -> np.ndarray:
    """Observed directional modulation R per unit (cheap path used by training-time eval)."""
    statistics = _prepare_directional_statistics(
        representation=representation,
        position_xy=position_xy,
        heading=heading,
        valid_mask=valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
        min_occupancy_per_quadrant=min_occupancy_per_quadrant,
    )
    if statistics is None:
        return np.full(int(representation.shape[-1]), np.nan, dtype=np.float32)
    mean_per_quadrant = _quadrant_means_from_labels(statistics, statistics.quadrant)
    return _modulation_r_from_quadrant_means(
        statistics,
        mean_per_quadrant,
        field_threshold_fraction=field_threshold_fraction,
        min_heading_quadrants=min_heading_quadrants,
        min_field_bins=min_field_bins,
        min_fire_rate=min_fire_rate,
    )


@dataclass(slots=True)
class DirectionalityModule:
    """Per-unit directional modulation (place cell vs head-direction-conjunctive)."""

    name: str = "directionality"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_units = int(analysis_input.representation.shape[-1])
        omnidirectional_threshold = float(config.get("omnidirectional_threshold", 0.30))
        directional_threshold = float(config.get("directional_threshold", 0.60))
        num_bins_x = int(config["directionality_num_bins_x"])
        num_bins_y = int(config["directionality_num_bins_y"])
        num_null_shuffles = int(config.get("directionality_null_shuffles", 999))
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None

        modulation_kwargs = {
            "field_threshold_fraction": float(
                config.get("place_field_threshold_fraction", 0.2)
            ),
            "min_heading_quadrants": int(config.get("min_heading_quadrants", 3)),
            "min_field_bins": int(config.get("min_field_bins", 3)),
            "min_fire_rate": float(config.get("min_fire_rate", 0.01)),
        }

        r_values = np.full(num_units, np.nan, dtype=np.float32)
        null_mean = np.full(num_units, np.nan, dtype=np.float32)
        null_95 = np.full(num_units, np.nan, dtype=np.float32)
        excess = np.full(num_units, np.nan, dtype=np.float32)
        null_p = np.full(num_units, np.nan, dtype=np.float32)
        significant = np.zeros(num_units, dtype=bool)
        negative_fraction = float("nan")
        statistics = None
        if analysis_input.heading is not None:
            statistics = _prepare_directional_statistics(
                representation=analysis_input.representation,
                position_xy=analysis_input.position_xy,
                heading=analysis_input.heading,
                valid_mask=analysis_input.valid_mask,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                bounds=world_bounds,
                min_occupancy_per_quadrant=int(config.get("min_occupancy_per_quadrant", 5)),
            )
        if statistics is not None:
            negative_fraction = statistics.negative_fraction
            r_values = _modulation_r_from_quadrant_means(
                statistics,
                _quadrant_means_from_labels(statistics, statistics.quadrant),
                **modulation_kwargs,
            )
            if num_null_shuffles > 0 and np.isfinite(r_values).any():
                null_matrix = _null_modulation_matrix(
                    statistics,
                    _resolve_field_gates(
                        statistics,
                        field_threshold_fraction=modulation_kwargs["field_threshold_fraction"],
                        min_heading_quadrants=modulation_kwargs["min_heading_quadrants"],
                    ),
                    num_shuffles=num_null_shuffles,
                    field_threshold_fraction=modulation_kwargs["field_threshold_fraction"],
                    min_field_bins=modulation_kwargs["min_field_bins"],
                    min_fire_rate=modulation_kwargs["min_fire_rate"],
                )
                null_finite = np.isfinite(null_matrix)
                draw_counts = null_finite.sum(axis=0)
                calibrated = np.isfinite(r_values) & (draw_counts > 0)
                exceed_counts = np.sum(
                    null_finite & (null_matrix >= r_values[None, :]), axis=0
                )
                null_p[calibrated] = (
                    (1.0 + exceed_counts[calibrated]) / (1.0 + draw_counts[calibrated])
                ).astype(np.float32)
                for unit_index in np.flatnonzero(calibrated):
                    unit_null = null_matrix[null_finite[:, unit_index], unit_index]
                    null_mean[unit_index] = float(unit_null.mean())
                    null_95[unit_index] = float(np.percentile(unit_null, 95.0))
                excess = r_values - null_mean
                significant = benjamini_hochberg(null_p)

        assessable = r_values[np.isfinite(r_values)]
        num_assessable = int(assessable.size)

        def fraction(predicate: np.ndarray) -> float:
            return float(predicate.mean()) if num_assessable else float("nan")

        finite_excess = excess[np.isfinite(excess)]
        metrics = {
            "median_directional_modulation_r": (
                float(np.median(assessable)) if num_assessable else float("nan")
            ),
            "mean_directional_modulation_r": (
                float(assessable.mean()) if num_assessable else float("nan")
            ),
            "fraction_omnidirectional": fraction(assessable < omnidirectional_threshold),
            "fraction_directional": fraction(assessable > directional_threshold),
            "median_directional_modulation_r_excess": (
                float(np.median(finite_excess)) if finite_excess.size else float("nan")
            ),
            "fraction_directional_significant": (
                float(significant[np.isfinite(r_values)].mean())
                if num_assessable and num_null_shuffles > 0
                else float("nan")
            ),
            "directionality_clipped_negative_fraction": negative_fraction,
        }
        per_unit_metrics = {
            "directional_modulation_r": r_values,
            "directional_modulation_r_null_mean": null_mean,
            "directional_modulation_r_null_95": null_95,
            "directional_modulation_r_excess": excess,
            "directional_modulation_r_null_p": null_p,
            "directional_modulation_r_significant": significant.astype(np.float32, copy=False),
        }
        per_unit_table = _write_per_unit_table(output_dir, analysis_input, per_unit_metrics)
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics=per_unit_metrics,
            figures={},
            tables={"per_unit_metrics": per_unit_table},
            metadata={
                "directionality_assessable_unit_count": num_assessable,
                "directionality_total_unit_count": num_units,
                "directionality_num_bins_x": num_bins_x,
                "directionality_num_bins_y": num_bins_y,
                "directionality_null_shuffles": num_null_shuffles,
                "omnidirectional_threshold": omnidirectional_threshold,
                "directional_threshold": directional_threshold,
            },
        )


def _write_per_unit_table(
    output_dir: Path,
    analysis_input: AnalysisInput,
    per_unit_metrics: dict[str, np.ndarray],
) -> Path:
    r_values = per_unit_metrics["directional_modulation_r"]
    rows = [
        [
            unit_index,
            float(value) if np.isfinite(value) else "nan",
            float(per_unit_metrics["directional_modulation_r_excess"][unit_index])
            if np.isfinite(per_unit_metrics["directional_modulation_r_excess"][unit_index])
            else "nan",
            float(per_unit_metrics["directional_modulation_r_null_p"][unit_index])
            if np.isfinite(per_unit_metrics["directional_modulation_r_null_p"][unit_index])
            else "nan",
            bool(per_unit_metrics["directional_modulation_r_significant"][unit_index]),
            bool(np.isfinite(value)),
        ]
        for unit_index, value in enumerate(r_values)
    ]
    return write_csv(
        output_dir
        / "directionality"
        / f"directionality_per_unit__{analysis_input.source_name}__{analysis_input.split_name}.csv",
        [
            "unit_index",
            "directional_modulation_r",
            "directional_modulation_r_excess",
            "directional_modulation_r_null_p",
            "directional_modulation_r_significant",
            "assessable",
        ],
        rows,
    )
