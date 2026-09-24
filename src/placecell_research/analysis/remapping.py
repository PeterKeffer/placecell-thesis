"""Comparative remapping analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import (
    RateMapComputation,
    flatten_positions,
    infer_bounds,
)
from .base import AnalysisInput, AnalysisResult
from .figures import save_histogram
from .helpers import (
    get_or_compute_rate_maps,
    save_heatmap,
    write_csv,
)
from .remapping_metrics import (
    add_pair_metrics,
    add_pair_per_unit_metrics,
    finite_entries,
    pair_diagnostics,
    pairwise_map_correlation,
    remapping_pair_name,
    save_map,
    write_pair_tables,
)
from .world_overlay import overlay_bounds, resolve_world_overlay

Bounds = tuple[tuple[float, float], tuple[float, float]]
NORMALIZED_WORLD_BOUNDS: Bounds = ((0.0, 1.0), (0.0, 1.0))


def _same_bounds(first_bounds: Bounds, second_bounds: Bounds) -> bool:
    return bool(
        np.allclose(
            np.asarray(first_bounds),
            np.asarray(second_bounds),
            rtol=0.0,
            atol=1e-6,
        )
    )


def _source_world_bounds(source: AnalysisInput) -> Bounds | None:
    world_overlay = resolve_world_overlay(
        str(source.metadata.get("env_id", "")),
        source.metadata.get("env_kwargs"),
    )
    return overlay_bounds(world_overlay) if world_overlay is not None else None


def _normalize_positions(position_xy: np.ndarray, bounds: Bounds) -> np.ndarray:
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError(f"Cannot normalize remapping coordinates with invalid bounds {bounds}.")
    normalized = position_xy.astype(np.float32, copy=True)
    normalized[..., 0] = (position_xy[..., 0] - x_min) / x_span
    normalized[..., 1] = (position_xy[..., 1] - y_min) / y_span
    return normalized


def _with_position_xy(source: AnalysisInput, position_xy: np.ndarray) -> AnalysisInput:
    return replace(source, position_xy=position_xy, position_cache={}, _rate_map_cache={})


def _common_observed_bounds(inputs: list[AnalysisInput]) -> Bounds:
    positions = [flatten_positions(source.position_xy, source.valid_mask) for source in inputs]
    return infer_bounds(np.concatenate(positions, axis=0))


def _rate_map_sources_and_bounds(inputs: list[AnalysisInput]) -> list[tuple[AnalysisInput, Bounds]]:
    source_bounds = [_source_world_bounds(source) for source in inputs]
    known_bounds = [bounds for bounds in source_bounds if bounds is not None]
    if not known_bounds:
        common_bounds = _common_observed_bounds(inputs)
        return [(source, common_bounds) for source in inputs]
    if len(known_bounds) != len(inputs):
        raise ValueError(
            "Remapping comparison cannot mix known environment geometry "
            "with inferred-position bounds."
        )
    if all(_same_bounds(known_bounds[0], bounds) for bounds in known_bounds[1:]):
        return [(source, known_bounds[0]) for source in inputs]
    return [
        (
            _with_position_xy(source, _normalize_positions(source.position_xy, bounds)),
            NORMALIZED_WORLD_BOUNDS,
        )
        for source, bounds in zip(inputs, known_bounds, strict=False)
    ]


@dataclass(slots=True)
class RemappingComparisonModule:
    """Cross-source remapping comparison."""

    name: str = "remapping_comparison"
    cost_tier: str = "heavy"

    def run(
        self,
        inputs: list[AnalysisInput],
        labels: list[str],
        output_dir: Path,
        config: dict,
    ) -> AnalysisResult:
        if len(inputs) < 2:
            raise ValueError("Remapping comparison needs at least two sources.")
        num_bins_x = int(config.get("num_bins_x", 60))
        num_bins_y = int(config.get("num_bins_y", 60))
        smoothing_sigma = float(config["smoothing_sigma"])
        min_occupancy = float(config.get("min_occupancy", 1e-6))
        field_threshold_fraction = float(config.get("place_field_threshold_fraction", 0.2))
        active_peak_rate_threshold = float(config.get("active_peak_rate_threshold", 1e-6))
        shuffle_iterations = int(config.get("remapping_shuffle_iterations", 100))
        shuffle_seed = int(config["remapping_shuffle_seed"])

        rate_maps_by_label: dict[str, RateMapComputation] = {}
        for source, bounds in _rate_map_sources_and_bounds(inputs):
            rate_maps = get_or_compute_rate_maps(
                source,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                smoothing_sigma=smoothing_sigma,
                min_occupancy=min_occupancy,
                bounds=bounds,
            )
            rate_maps_by_label[source.label] = rate_maps

        module_dir = output_dir / self.name
        pairwise_matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)
        metrics: dict[str, float] = {}
        per_unit_metrics: dict[str, np.ndarray] = {}
        figures: dict[str, Path] = {}
        tables: dict[str, Path] = {}
        summary_rows: list[list[object]] = []
        for row_index, first_label in enumerate(labels):
            for column_index, second_label in enumerate(labels):
                if column_index < row_index:
                    continue
                _, mean_correlation = pairwise_map_correlation(
                    rate_maps_by_label[first_label].rate_maps,
                    rate_maps_by_label[second_label].rate_maps,
                )
                pairwise_matrix[row_index, column_index] = mean_correlation
                pairwise_matrix[column_index, row_index] = mean_correlation
                if column_index == row_index:
                    continue
                pair_name = remapping_pair_name(first_label, second_label)
                diagnostics = pair_diagnostics(
                    rate_maps_by_label[first_label],
                    rate_maps_by_label[second_label],
                    field_threshold_fraction=field_threshold_fraction,
                    active_peak_rate_threshold=active_peak_rate_threshold,
                    shuffle_iterations=shuffle_iterations,
                    shuffle_seed=shuffle_seed + row_index * len(labels) + column_index,
                )
                add_pair_metrics(metrics, pair_name, diagnostics)
                add_pair_per_unit_metrics(per_unit_metrics, pair_name, diagnostics)
                figures[f"{pair_name}__unit_correlation_histogram"] = save_histogram(
                    module_dir / f"remapping_unit_correlation_histogram__{pair_name}.png",
                    finite_entries(diagnostics.unit_correlations),
                    f"Unit remapping correlations: {first_label} vs {second_label}",
                    "unit correlation",
                )
                figures[f"{pair_name}__population_vector_correlation"] = save_map(
                    module_dir / f"remapping_population_vector_correlation__{pair_name}.png",
                    diagnostics.population_vector_correlation,
                    f"PVC: {first_label} vs {second_label}",
                    "population vector correlation",
                    rate_maps_by_label[first_label].bounds,
                )
                tables.update(write_pair_tables(module_dir, pair_name, diagnostics))
                summary_rows.append(
                    [
                        first_label,
                        second_label,
                        float(mean_correlation),
                        metrics[f"{pair_name}__unit_correlation_median"],
                        metrics[f"{pair_name}__unit_correlation_iqr"],
                        metrics[f"{pair_name}__unit_correlation_fraction_gt_0_5"],
                        metrics[f"{pair_name}__unit_correlation_fraction_lt_0_2"],
                        metrics[f"{pair_name}__mean_population_vector_correlation"],
                        metrics[f"{pair_name}__active_jaccard"],
                        metrics[f"{pair_name}__mean_field_center_shift"],
                        metrics[f"{pair_name}__null_mean_unit_correlation"],
                        metrics[f"{pair_name}__observed_minus_null_mean_unit_correlation"],
                        metrics[f"{pair_name}__occupancy_correlation"],
                        metrics[f"{pair_name}__shared_visited_fraction"],
                    ]
                )

        heatmap_path = save_heatmap(
            module_dir / f"remapping_summary__{inputs[0].source_name}__{inputs[0].split_name}.png",
            pairwise_matrix,
            "Pairwise remapping correlation",
            labels,
            labels,
        )
        figures["pairwise_heatmap"] = heatmap_path
        pair_summary_name = (
            f"remapping_pair_summary__{inputs[0].source_name}__{inputs[0].split_name}.csv"
        )
        pair_summary_path = module_dir / pair_summary_name
        tables["pair_summary"] = write_csv(
            pair_summary_path,
            header=[
                "first_label",
                "second_label",
                "mean_unit_correlation",
                "median_unit_correlation",
                "unit_correlation_iqr",
                "unit_correlation_fraction_gt_0_5",
                "unit_correlation_fraction_lt_0_2",
                "mean_population_vector_correlation",
                "active_jaccard",
                "mean_field_center_shift",
                "null_mean_unit_correlation",
                "observed_minus_null_mean_unit_correlation",
                "occupancy_correlation",
                "shared_visited_fraction",
            ],
            rows=summary_rows,
        )
        diagonal_values = np.diag(pairwise_matrix)
        off_diagonal_values = pairwise_matrix[np.triu_indices(len(labels), k=1)]
        metrics.update(
            {
                "mean_pairwise_correlation": float(off_diagonal_values.mean())
                if off_diagonal_values.size
                else 0.0,
                "diagonal_mean_correlation": (
                    float(diagonal_values.mean()) if diagonal_values.size else 0.0
                ),
                "off_diagonal_mean_correlation": float(off_diagonal_values.mean())
                if off_diagonal_values.size
                else 0.0,
                "diagonal_minus_off_diagonal_mean_correlation": (
                    float(diagonal_values.mean() - off_diagonal_values.mean())
                    if off_diagonal_values.size and diagonal_values.size
                    else 0.0
                ),
            }
        )
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics=per_unit_metrics,
            figures=figures,
            tables=tables,
        )
