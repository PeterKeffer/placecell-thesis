"""Within-heading firing reliability of place codes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import compute_spatial_bin_assignments
from .base import AnalysisInput, AnalysisResult
from .helpers import write_csv
from .rate_map_metrics import nanmean_or_nan
from .world_overlay import overlay_bounds, resolve_world_overlay

_EPS = 1e-9
_TWO_PI = 2.0 * np.pi


@dataclass(slots=True)
class WithinHeadingReliability:
    """Per-unit reliability conditioned on heading vs pooled over heading."""

    pooled: np.ndarray
    within_heading: np.ndarray
    gain: np.ndarray


def compute_within_heading_reliability(
    *,
    representation: np.ndarray,
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    threshold_fraction: float = 0.2,
    num_heading_bins: int = 4,
    min_steps_per_cell: int = 5,
) -> WithinHeadingReliability:
    num_units = int(representation.shape[-1])
    codes = representation.reshape(-1, num_units)
    positions = position_xy.reshape(-1, 2)
    headings = heading.reshape(-1)
    if valid_mask is not None:
        keep = valid_mask.reshape(-1).astype(bool, copy=False)
        codes, positions, headings = codes[keep], positions[keep], headings[keep]

    nan_units = np.full(num_units, np.nan, dtype=np.float64)
    if codes.shape[0] == 0:
        return WithinHeadingReliability(nan_units.copy(), nan_units.copy(), nan_units.copy())

    rates = np.clip(codes, 0.0, None).astype(np.float64)
    peak = rates.max(axis=0)
    assessable = peak > _EPS
    thresholds = threshold_fraction * peak
    strong = (rates >= thresholds[None, :]) & assessable[None, :]
    strong = strong.astype(np.float64)

    bin_index, _, _, _ = compute_spatial_bin_assignments(
        positions, num_bins_x=num_bins_x, num_bins_y=num_bins_y, bounds=bounds
    )
    num_bins = num_bins_x * num_bins_y
    radians_per_bin = _TWO_PI / num_heading_bins
    quadrant = np.floor((headings % _TWO_PI) / radians_per_bin).astype(int) % num_heading_bins

    pooled_rel = _reliability_per_cell(strong, bin_index, num_bins, min_steps_per_cell)

    combined = bin_index * num_heading_bins + quadrant
    rel_cell = _reliability_per_cell(
        strong, combined, num_bins * num_heading_bins, min_steps_per_cell
    ).reshape(num_bins, num_heading_bins, num_units)
    within_rel = _nanmax_over_axis(rel_cell, axis=1)

    pooled_per_unit = nan_units.copy()
    within_per_unit = nan_units.copy()
    for unit in range(num_units):
        if not assessable[unit]:
            continue
        pooled_column = pooled_rel[:, unit]
        within_column = within_rel[:, unit]
        comparable = np.isfinite(pooled_column) & np.isfinite(within_column)
        if not comparable.any():
            continue
        pooled_per_unit[unit] = float(pooled_column[comparable].mean())
        within_per_unit[unit] = float(within_column[comparable].mean())

    return WithinHeadingReliability(
        pooled=pooled_per_unit,
        within_heading=within_per_unit,
        gain=within_per_unit - pooled_per_unit,
    )


def _reliability_per_cell(
    strong: np.ndarray, cell_index: np.ndarray, num_cells: int, min_steps_per_cell: int
) -> np.ndarray:
    """P(strong activation | visit) per (cell, unit); NaN where under-sampled."""
    num_units = strong.shape[-1]
    strong_counts = np.zeros((num_cells, num_units), dtype=np.float64)
    np.add.at(strong_counts, cell_index, strong)
    step_counts = np.bincount(cell_index, minlength=num_cells).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        reliability = strong_counts / step_counts[:, None]
    reliability[step_counts < min_steps_per_cell, :] = np.nan
    return reliability


def _nanmax_over_axis(values: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmax without the all-NaN-slice warning; all-NaN slices stay NaN."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, -np.inf)
    maximum = filled.max(axis=axis)
    maximum[~finite.any(axis=axis)] = np.nan
    return maximum


@dataclass(slots=True)
class WithinHeadingReliabilityModule:
    """Reliability of each unit conditioned on heading vs pooled over heading."""

    name: str = "within_heading_reliability"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_units = int(analysis_input.representation.shape[-1])
        num_bins_x = int(config["directionality_num_bins_x"])
        num_bins_y = int(config["directionality_num_bins_y"])
        conjunctive_gain_threshold = float(config.get("within_heading_conjunctive_gain", 0.2))
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None

        if analysis_input.heading is None:
            nan_units = np.full(num_units, np.nan, dtype=np.float64)
            result = WithinHeadingReliability(nan_units, nan_units.copy(), nan_units.copy())
        else:
            result = compute_within_heading_reliability(
                representation=analysis_input.representation,
                position_xy=analysis_input.position_xy,
                heading=analysis_input.heading,
                valid_mask=analysis_input.valid_mask,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                bounds=world_bounds,
                threshold_fraction=float(config.get("reliability_threshold_fraction", 0.2)),
                num_heading_bins=int(config.get("within_heading_num_bins", 4)),
                min_steps_per_cell=int(config.get("within_heading_min_steps_per_cell", 5)),
            )

        assessable = np.isfinite(result.gain)
        num_assessable = int(assessable.sum())
        fraction_conjunctive = (
            float((result.gain[assessable] > conjunctive_gain_threshold).mean())
            if num_assessable
            else float("nan")
        )
        metrics = {
            "mean_pooled_reliability": nanmean_or_nan(result.pooled),
            "mean_within_heading_reliability": nanmean_or_nan(result.within_heading),
            "mean_within_heading_reliability_gain": nanmean_or_nan(result.gain),
            "fraction_conjunctive": fraction_conjunctive,
        }
        table = _write_per_unit_table(output_dir, analysis_input, result)
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={
                "pooled_reliability": result.pooled,
                "within_heading_reliability": result.within_heading,
                "within_heading_reliability_gain": result.gain,
            },
            figures={},
            tables={"per_unit_metrics": table},
            metadata={
                "within_heading_assessable_unit_count": num_assessable,
                "within_heading_total_unit_count": num_units,
                "within_heading_num_bins": int(config.get("within_heading_num_bins", 4)),
                "within_heading_conjunctive_gain": conjunctive_gain_threshold,
            },
        )


def _write_per_unit_table(
    output_dir: Path, analysis_input: AnalysisInput, result: WithinHeadingReliability
) -> Path:
    rows = [
        [
            unit_index,
            float(result.pooled[unit_index]) if np.isfinite(result.pooled[unit_index]) else "nan",
            float(result.within_heading[unit_index])
            if np.isfinite(result.within_heading[unit_index])
            else "nan",
            float(result.gain[unit_index]) if np.isfinite(result.gain[unit_index]) else "nan",
        ]
        for unit_index in range(result.pooled.shape[0])
    ]
    table_path = (
        output_dir
        / "within_heading_reliability"
        / (
            "within_heading_reliability_per_unit__"
            f"{analysis_input.source_name}__{analysis_input.split_name}.csv"
        )
    )
    columns = [
        "unit_index",
        "pooled_reliability",
        "within_heading_reliability",
        "within_heading_reliability_gain",
    ]
    return write_csv(table_path, columns, rows)
