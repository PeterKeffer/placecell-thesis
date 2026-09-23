"""Pair-level remapping metrics and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.rate_map_kernels import (
    RateMapComputation,
    compute_place_field_mask,
    rowwise_correlation,
)
from .helpers import write_csv
from .world_overlay import POSITION_X_LABEL, POSITION_Y_LABEL

Bounds = tuple[tuple[float, float], tuple[float, float]]


@dataclass(slots=True)
class FieldMetrics:
    field_count: np.ndarray
    field_area: np.ndarray
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    peak_rate: np.ndarray


@dataclass(slots=True)
class PairDiagnostics:
    unit_correlations: np.ndarray
    population_vector_correlation: np.ndarray
    field_metrics_a: FieldMetrics
    field_metrics_b: FieldMetrics
    field_center_shift: np.ndarray
    field_area_ratio: np.ndarray
    peak_rate_ratio: np.ndarray
    null_mean_correlations: np.ndarray
    occupancy_correlation: float
    shared_visited_fraction: float
    active_both_units: int
    active_a_only_units: int
    active_b_only_units: int
    silent_both_units: int
    active_jaccard: float


def _pairwise_map_correlation(
    first_maps: np.ndarray,
    second_maps: np.ndarray,
) -> tuple[np.ndarray, float]:
    if first_maps.shape[0] != second_maps.shape[0]:
        raise ValueError(
            "Remapping comparison requires the same number of units, "
            f"got {first_maps.shape[0]} and {second_maps.shape[0]}."
        )
    if first_maps.shape[1:] != second_maps.shape[1:]:
        raise ValueError(
            "Remapping comparison requires matching rate-map grids, "
            f"got {first_maps.shape[1:]} and {second_maps.shape[1:]}."
        )
    num_units = first_maps.shape[0]
    if num_units == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    first_flat = first_maps.reshape(num_units, -1)
    second_flat = second_maps.reshape(num_units, -1)
    correlations = rowwise_correlation(
        first_flat,
        second_flat,
        valid_mask=np.isfinite(first_flat) & np.isfinite(second_flat),
    )
    return correlations, float(correlations.mean())


def _safe_slug(value: str) -> str:
    return (
        "".join(character if character.isalnum() else "_" for character in value).strip("_")
        or "source"
    )


def remapping_pair_name(first_label: str, second_label: str) -> str:
    return f"{_safe_slug(first_label)}__{_safe_slug(second_label)}"


def _finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _finite_mean(values: np.ndarray) -> float:
    finite_values = _finite_values(values)
    return float(np.mean(finite_values)) if finite_values.size else 0.0


def _finite_median(values: np.ndarray) -> float:
    finite_values = _finite_values(values)
    return float(np.median(finite_values)) if finite_values.size else 0.0


def _finite_iqr(values: np.ndarray) -> float:
    finite_values = _finite_values(values)
    if finite_values.size == 0:
        return 0.0
    return float(np.percentile(finite_values, 75.0) - np.percentile(finite_values, 25.0))


def _finite_fraction(values: np.ndarray, predicate: np.ndarray) -> float:
    finite_values = np.isfinite(values)
    if not np.any(finite_values):
        return 0.0
    return float(np.count_nonzero(predicate & finite_values) / np.count_nonzero(finite_values))


def _population_vector_correlation(
    first_maps: np.ndarray,
    second_maps: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    if first_maps.shape != second_maps.shape:
        raise ValueError(
            "Population vector correlation requires matching map shapes, "
            f"got {first_maps.shape} and {second_maps.shape}."
        )
    _, num_bins_y, num_bins_x = first_maps.shape
    correlations = np.full((num_bins_y, num_bins_x), np.nan, dtype=np.float32)
    first_by_bin = first_maps.reshape(first_maps.shape[0], -1)
    second_by_bin = second_maps.reshape(second_maps.shape[0], -1)
    flat_correlations = correlations.reshape(-1)
    for bin_index in range(first_by_bin.shape[1]):
        first_values = first_by_bin[:, bin_index]
        second_values = second_by_bin[:, bin_index]
        valid_mask = np.isfinite(first_values) & np.isfinite(second_values)
        if np.count_nonzero(valid_mask) < 2:
            continue
        first_valid = first_values[valid_mask]
        second_valid = second_values[valid_mask]
        first_centered = first_valid - float(np.mean(first_valid))
        second_centered = second_valid - float(np.mean(second_valid))
        denominator = float(
            np.sqrt(np.sum(np.square(first_centered)) * np.sum(np.square(second_centered)))
        )
        flat_correlations[bin_index] = (
            float(np.sum(first_centered * second_centered) / denominator)
            if denominator > epsilon
            else 0.0
        )
    return correlations


def _peak_rates(rate_maps: np.ndarray) -> np.ndarray:
    peaks = np.zeros(rate_maps.shape[0], dtype=np.float32)
    for unit_index, rate_map in enumerate(rate_maps):
        finite_values = rate_map[np.isfinite(rate_map)]
        peaks[unit_index] = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
    return peaks


def _active_overlap(
    first_maps: np.ndarray,
    second_maps: np.ndarray,
    active_peak_rate_threshold: float,
) -> tuple[int, int, int, int, float]:
    first_active = _peak_rates(first_maps) > active_peak_rate_threshold
    second_active = _peak_rates(second_maps) > active_peak_rate_threshold
    active_both = int(np.count_nonzero(first_active & second_active))
    active_a_only = int(np.count_nonzero(first_active & ~second_active))
    active_b_only = int(np.count_nonzero(~first_active & second_active))
    silent_both = int(np.count_nonzero(~first_active & ~second_active))
    active_union = active_both + active_a_only + active_b_only
    active_jaccard = float(active_both / active_union) if active_union else 0.0
    return active_both, active_a_only, active_b_only, silent_both, active_jaccard


def _bin_center_grids(
    bounds: Bounds,
    *,
    num_bins_x: int,
    num_bins_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_edges = np.linspace(bounds[0][0], bounds[0][1], num_bins_x + 1, dtype=np.float32)
    y_edges = np.linspace(bounds[1][0], bounds[1][1], num_bins_y + 1, dtype=np.float32)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return np.meshgrid(x_centers, y_centers)


def _field_centroid(
    rate_map: np.ndarray,
    field_mask: np.ndarray,
    x_center_grid: np.ndarray,
    y_center_grid: np.ndarray,
) -> tuple[float, float] | None:
    weights = np.maximum(np.nan_to_num(rate_map, nan=0.0), 0.0).astype(np.float32, copy=False)
    weights *= field_mask.astype(np.float32, copy=False)
    total_weight = float(weights.sum())
    if total_weight <= 1e-8:
        return None
    return (
        float((weights * x_center_grid).sum() / total_weight),
        float((weights * y_center_grid).sum() / total_weight),
    )


def _field_metrics(
    rate_maps: np.ndarray,
    bounds: Bounds,
    threshold_fraction: float,
) -> FieldMetrics:
    num_units, num_bins_y, num_bins_x = rate_maps.shape
    x_center_grid, y_center_grid = _bin_center_grids(
        bounds,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )
    field_count = np.zeros(num_units, dtype=np.float32)
    field_area = np.zeros(num_units, dtype=np.float32)
    centroid_x = np.full(num_units, np.nan, dtype=np.float32)
    centroid_y = np.full(num_units, np.nan, dtype=np.float32)
    peak_rate = _peak_rates(rate_maps)
    for unit_index, rate_map in enumerate(rate_maps):
        field_mask, count, area = compute_place_field_mask(rate_map, threshold_fraction)
        field_count[unit_index] = float(count)
        field_area[unit_index] = float(area)
        if area <= 0.0:
            continue
        centroid = _field_centroid(rate_map, field_mask, x_center_grid, y_center_grid)
        if centroid is None:
            continue
        centroid_x[unit_index] = centroid[0]
        centroid_y[unit_index] = centroid[1]
    return FieldMetrics(
        field_count=field_count,
        field_area=field_area,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        peak_rate=peak_rate,
    )


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    values = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = (numerator > 0.0) & (denominator > 0.0)
    values[valid] = (numerator[valid] / denominator[valid]).astype(np.float32, copy=False)
    return values


def _field_comparisons(
    first_fields: FieldMetrics,
    second_fields: FieldMetrics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_shift = np.full(first_fields.field_count.shape, np.nan, dtype=np.float32)
    valid_center = (
        np.isfinite(first_fields.centroid_x)
        & np.isfinite(first_fields.centroid_y)
        & np.isfinite(second_fields.centroid_x)
        & np.isfinite(second_fields.centroid_y)
    )
    center_shift[valid_center] = np.sqrt(
        np.square(first_fields.centroid_x[valid_center] - second_fields.centroid_x[valid_center])
        + np.square(first_fields.centroid_y[valid_center] - second_fields.centroid_y[valid_center])
    ).astype(np.float32, copy=False)
    return (
        center_shift,
        _ratio(second_fields.field_area, first_fields.field_area),
        _ratio(second_fields.peak_rate, first_fields.peak_rate),
    )


def _shuffled_null_means(
    first_maps: np.ndarray,
    second_maps: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    if iterations <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    flat_first = first_maps.reshape(first_maps.shape[0], -1)
    shuffled_means = np.zeros(iterations, dtype=np.float32)
    for iteration_index in range(iterations):
        shuffled = np.empty_like(flat_first)
        for unit_index, unit_values in enumerate(flat_first):
            shift = int(rng.integers(1, unit_values.size)) if unit_values.size > 1 else 0
            shuffled[unit_index] = np.roll(unit_values, shift)
        _, mean_correlation = _pairwise_map_correlation(
            shuffled.reshape(first_maps.shape),
            second_maps,
        )
        shuffled_means[iteration_index] = float(mean_correlation)
    return shuffled_means


def _occupancy_similarity(
    first_result: RateMapComputation,
    second_result: RateMapComputation,
) -> tuple[float, float]:
    first_flat = first_result.occupancy.reshape(1, -1)
    second_flat = second_result.occupancy.reshape(1, -1)
    valid_mask = np.isfinite(first_flat) & np.isfinite(second_flat)
    correlation = rowwise_correlation(first_flat, second_flat, valid_mask=valid_mask)
    shared_visited_fraction = float(
        np.mean((first_result.raw_occupancy > 0.0) & (second_result.raw_occupancy > 0.0))
    )
    return float(correlation[0]), shared_visited_fraction


def pair_diagnostics(
    first_result: RateMapComputation,
    second_result: RateMapComputation,
    *,
    field_threshold_fraction: float,
    active_peak_rate_threshold: float,
    shuffle_iterations: int,
    shuffle_seed: int,
) -> PairDiagnostics:
    unit_correlations, _ = _pairwise_map_correlation(
        first_result.rate_maps,
        second_result.rate_maps,
    )
    population_vector_correlation = _population_vector_correlation(
        first_result.rate_maps,
        second_result.rate_maps,
    )
    field_metrics_a = _field_metrics(
        first_result.rate_maps,
        first_result.bounds,
        field_threshold_fraction,
    )
    field_metrics_b = _field_metrics(
        second_result.rate_maps,
        second_result.bounds,
        field_threshold_fraction,
    )
    field_center_shift, field_area_ratio, peak_rate_ratio = _field_comparisons(
        field_metrics_a,
        field_metrics_b,
    )
    null_mean_correlations = _shuffled_null_means(
        first_result.rate_maps,
        second_result.rate_maps,
        iterations=shuffle_iterations,
        seed=shuffle_seed,
    )
    occupancy_correlation, shared_visited_fraction = _occupancy_similarity(
        first_result,
        second_result,
    )
    active_both, active_a_only, active_b_only, silent_both, active_jaccard = _active_overlap(
        first_result.rate_maps,
        second_result.rate_maps,
        active_peak_rate_threshold,
    )
    return PairDiagnostics(
        unit_correlations=unit_correlations,
        population_vector_correlation=population_vector_correlation,
        field_metrics_a=field_metrics_a,
        field_metrics_b=field_metrics_b,
        field_center_shift=field_center_shift,
        field_area_ratio=field_area_ratio,
        peak_rate_ratio=peak_rate_ratio,
        null_mean_correlations=null_mean_correlations,
        occupancy_correlation=occupancy_correlation,
        shared_visited_fraction=shared_visited_fraction,
        active_both_units=active_both,
        active_a_only_units=active_a_only,
        active_b_only_units=active_b_only,
        silent_both_units=silent_both,
        active_jaccard=active_jaccard,
    )


def save_map(
    path: Path, values: np.ndarray, title: str, colorbar_label: str, bounds: Bounds
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    extent = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
    image = axis.imshow(values, origin="lower", extent=extent, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel(POSITION_X_LABEL)
    axis.set_ylabel(POSITION_Y_LABEL)
    figure.colorbar(image, ax=axis, shrink=0.8, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def add_pair_metrics(
    metrics: dict[str, float],
    pair_name: str,
    diagnostics: PairDiagnostics,
) -> None:
    unit_correlations = diagnostics.unit_correlations
    pvc_values = diagnostics.population_vector_correlation
    null_values = diagnostics.null_mean_correlations
    metrics.update(
        {
            f"{pair_name}__unit_correlation_mean": _finite_mean(unit_correlations),
            f"{pair_name}__unit_correlation_median": _finite_median(unit_correlations),
            f"{pair_name}__unit_correlation_iqr": _finite_iqr(unit_correlations),
            f"{pair_name}__unit_correlation_fraction_gt_0_5": _finite_fraction(
                unit_correlations,
                unit_correlations > 0.5,
            ),
            f"{pair_name}__unit_correlation_fraction_lt_0_2": _finite_fraction(
                unit_correlations,
                unit_correlations < 0.2,
            ),
            f"{pair_name}__mean_population_vector_correlation": _finite_mean(pvc_values),
            f"{pair_name}__median_population_vector_correlation": _finite_median(pvc_values),
            f"{pair_name}__population_vector_correlation_iqr": _finite_iqr(pvc_values),
            f"{pair_name}__active_both_units": float(diagnostics.active_both_units),
            f"{pair_name}__active_a_only_units": float(diagnostics.active_a_only_units),
            f"{pair_name}__active_b_only_units": float(diagnostics.active_b_only_units),
            f"{pair_name}__silent_both_units": float(diagnostics.silent_both_units),
            f"{pair_name}__active_jaccard": diagnostics.active_jaccard,
            f"{pair_name}__mean_field_center_shift": _finite_mean(diagnostics.field_center_shift),
            f"{pair_name}__median_field_center_shift": _finite_median(
                diagnostics.field_center_shift
            ),
            f"{pair_name}__mean_field_area_ratio": _finite_mean(diagnostics.field_area_ratio),
            f"{pair_name}__mean_peak_rate_ratio": _finite_mean(diagnostics.peak_rate_ratio),
            f"{pair_name}__null_mean_unit_correlation": _finite_mean(null_values),
            f"{pair_name}__null_std_unit_correlation": (
                float(np.std(null_values)) if null_values.size else 0.0
            ),
            f"{pair_name}__observed_minus_null_mean_unit_correlation": (
                _finite_mean(unit_correlations) - _finite_mean(null_values)
            ),
            f"{pair_name}__occupancy_correlation": diagnostics.occupancy_correlation,
            f"{pair_name}__shared_visited_fraction": diagnostics.shared_visited_fraction,
        }
    )


def add_pair_per_unit_metrics(
    per_unit_metrics: dict[str, np.ndarray],
    pair_name: str,
    diagnostics: PairDiagnostics,
) -> None:
    per_unit_metrics.update(
        {
            f"{pair_name}__unit_correlation": diagnostics.unit_correlations,
            f"{pair_name}__field_count_a": diagnostics.field_metrics_a.field_count,
            f"{pair_name}__field_count_b": diagnostics.field_metrics_b.field_count,
            f"{pair_name}__field_center_shift": diagnostics.field_center_shift,
            f"{pair_name}__field_area_ratio": diagnostics.field_area_ratio,
            f"{pair_name}__peak_rate_ratio": diagnostics.peak_rate_ratio,
            f"{pair_name}__peak_rate_a": diagnostics.field_metrics_a.peak_rate,
            f"{pair_name}__peak_rate_b": diagnostics.field_metrics_b.peak_rate,
        }
    )


def write_pair_tables(
    module_dir: Path,
    pair_name: str,
    diagnostics: PairDiagnostics,
) -> dict[str, Path]:
    per_unit_path = write_csv(
        module_dir / f"remapping_per_unit__{pair_name}.csv",
        header=[
            "unit_index",
            "unit_correlation",
            "field_count_a",
            "field_count_b",
            "field_center_shift",
            "field_area_ratio",
            "peak_rate_a",
            "peak_rate_b",
            "peak_rate_ratio",
        ],
        rows=[
            [
                int(unit_index),
                float(diagnostics.unit_correlations[unit_index]),
                float(diagnostics.field_metrics_a.field_count[unit_index]),
                float(diagnostics.field_metrics_b.field_count[unit_index]),
                float(diagnostics.field_center_shift[unit_index]),
                float(diagnostics.field_area_ratio[unit_index]),
                float(diagnostics.field_metrics_a.peak_rate[unit_index]),
                float(diagnostics.field_metrics_b.peak_rate[unit_index]),
                float(diagnostics.peak_rate_ratio[unit_index]),
            ]
            for unit_index in range(diagnostics.unit_correlations.shape[0])
        ],
    )
    null_path = write_csv(
        module_dir / f"remapping_null_distribution__{pair_name}.csv",
        header=["iteration", "mean_unit_correlation"],
        rows=[
            [int(iteration_index), float(value)]
            for iteration_index, value in enumerate(diagnostics.null_mean_correlations)
        ],
    )
    return {
        f"{pair_name}__per_unit_metrics": per_unit_path,
        f"{pair_name}__null_distribution": null_path,
    }
