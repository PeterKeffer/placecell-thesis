"""Per-model rows of the thesis measures and their mean and SD over training seeds."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

MEDIAN_MEASURES = (
    "split_half_correlation",
    "field_component_count",
    "field_area_over_visited_bins",
    "spatial_activity_fraction",
    "lifetime_activity_fraction",
    "traversal_hit_rate",
    "directional_traversal_hit_rate",
    "variance_explained_held_out",
)

FULL_ANALYSIS_METRICS = {
    "full_decode_linear_rmse": "encoder_place_cells.decode_xy.decode_rmse",
    "full_decode_linear_r2": "encoder_place_cells.decode_xy.decode_r2",
    "full_decode_nonlinear_rmse": "encoder_place_cells.decode_xy.nonlinear_decode_rmse",
    "full_silent_fraction": "encoder_place_cells.redundancy_metrics.dead_unit_fraction",
    "full_spatial_information_bits_mean": (
        "encoder_place_cells.rate_map_coding_purity.mean_spatial_information_bits"
    ),
    "full_field_unit_fraction": (
        "encoder_place_cells.population_coverage.active_field_unit_fraction"
    ),
    "participation_ratio": "encoder_place_cells.effective_dimensionality.participation_ratio",
    "step_trustworthiness": (
        "encoder_place_cells.neighborhood_preservation.step_world_code_trustworthiness"
    ),
    "step_continuity": "encoder_place_cells.neighborhood_preservation.step_world_code_continuity",
    "partial_spearman_euclidean": (
        "encoder_place_cells.cognitive_map_geometry.cognitive_map_partial_spearman_euclidean"
    ),
    "mean_units_per_visited_location": (
        "encoder_place_cells.population_coverage.mean_population_coverage_units"
    ),
}

IDENTITY_COLUMNS = ("condition", "training_seed", "model", "model_trained_as")


def single_unit_row(
    per_unit: dict[str, np.ndarray], visited_bin_count: int
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Summaries of the per-unit arrays; silent units count as zero where stated."""
    information = np.nan_to_num(per_unit["spatial_information_bits"], nan=0.0)
    null_95 = np.nan_to_num(per_unit["spatial_information_null_95"], nan=0.0)
    arrays = dict(per_unit)
    arrays["information_above_null95"] = np.maximum(information - null_95, 0.0)
    arrays["field_area_over_visited_bins"] = per_unit["field_area_bins"] / visited_bin_count
    row = {
        "units": int(information.size),
        "silent_fraction": float((per_unit["active_step_count"] == 0).mean()),
        "spatial_information_bits_all_units_mean": float(information.mean()),
        "information_above_null95_all_units_mean": float(arrays["information_above_null95"].mean()),
    }
    for key in MEDIAN_MEASURES:
        values = arrays[key].astype(np.float64)
        finite = values[np.isfinite(values)]
        row[f"{key}_finite_units"] = int(finite.size)
        row[f"{key}_finite_median"] = float(np.median(finite)) if finite.size else float("nan")
        row[f"{key}_all_units_zero_filled_median"] = float(
            np.median(np.nan_to_num(values, nan=0.0))
        )
    area = arrays["field_area_over_visited_bins"].astype(np.float64)
    finite_area = area[np.isfinite(area)]
    quartiles = np.percentile(finite_area, [25, 50, 75]) * 100 if finite_area.size else [np.nan] * 3
    for name, value in zip(("q25", "median", "q75"), quartiles, strict=True):
        row[f"field_area_percent_finite_{name}"] = float(value)
    return row, arrays


def full_analysis_row(summary: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    row, missing = {}, []
    for name, key in FULL_ANALYSIS_METRICS.items():
        if key in summary:
            row[name] = float(summary[key])
        else:
            row[name] = float("nan")
            missing.append(key)
    return row, missing


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def summarize(paths: list[Path]) -> list[dict]:
    """One row per condition: n, mean and sample SD of every numeric column over its rows."""
    rows = [row for path in paths for row in csv.DictReader(path.open())]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    skip = {"training_seed", "seed", "units", "episodes"}
    table = []
    for condition, members in groups.items():
        entry = {"condition": condition, "n": len(members)}
        columns = [key for key in members[0] if key not in (*IDENTITY_COLUMNS, *skip)]
        for column in columns:
            values = [_number(member.get(column, "")) for member in members]
            numbers = [value for value in values if value is not None]
            if not numbers or len(numbers) != len(values):
                continue
            entry[f"{column}_mean"] = statistics.mean(numbers)
            entry[f"{column}_sd"] = statistics.stdev(numbers) if len(numbers) > 1 else float("nan")
        table.append(entry)
    return table
