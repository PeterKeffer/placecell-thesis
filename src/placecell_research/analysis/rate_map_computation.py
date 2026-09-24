"""Which units a rate-map figure shows, and the metric records it reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

from ..numerics.rate_map_kernels import (
    flatten_valid_steps,
)
from .base import AnalysisInput
from .rate_map_metrics import (
    RateMapMetricBundle,
    nanmax_or_nan,
    nanmean_or_nan,
)


def _normalize_rate_map_colormap_mode(raw_value: object) -> str:
    normalized_value = str(raw_value).strip().lower()
    if normalized_value in {
        "reds",
        "inferno",
        "turbo",
        "viridis",
        "magma",
        "cividis",
        "coolwarm",
        "plasma",
        "auto",
    }:
        return normalized_value
    return "reds"


def normalize_panel_reliability_metric(raw_value: object) -> str:
    normalized_value = str(raw_value).strip().lower()
    if normalized_value in {
        "thresholded_reliability",
        "quantile_thresholded_reliability",
        "bin_consistency",
        "split_half_agreement",
    }:
        return normalized_value
    return "quantile_thresholded_reliability"


def _field_center_xy(
    rate_map: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """World (x, y) center of mass of a unit's positive activity."""
    weights = np.clip(np.nan_to_num(rate_map, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total_mass = float(weights.sum())
    if total_mass <= 1e-12:
        return None
    num_rows, num_columns = weights.shape
    row_center = float(weights.sum(axis=1) @ np.arange(num_rows)) / total_mass
    column_center = float(weights.sum(axis=0) @ np.arange(num_columns)) / total_mass
    (x_min, x_max), (y_min, y_max) = bounds
    world_x = x_min + (column_center + 0.5) / num_columns * (x_max - x_min)
    world_y = y_min + (row_center + 0.5) / num_rows * (y_max - y_min)
    return world_x, world_y


def _order_indices_by_field_position(
    selected_indices: np.ndarray,
    rate_maps: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Reorder already-selected units so nearby place fields land adjacent."""
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if selected_indices.size <= 2:
        return selected_indices
    centered_units: list[int] = []
    centers: list[tuple[float, float]] = []
    uncentered_units: list[int] = []
    for unit_index in selected_indices.tolist():
        center = _field_center_xy(rate_maps[unit_index], bounds)
        if center is None:
            uncentered_units.append(unit_index)
        else:
            centered_units.append(unit_index)
            centers.append(center)
    if len(centered_units) <= 2:
        return np.asarray(centered_units + uncentered_units, dtype=np.int64)
    linkage_matrix = linkage(
        np.asarray(centers, dtype=np.float64), method="average", optimal_ordering=True
    )
    ordered_centered = [centered_units[leaf] for leaf in leaves_list(linkage_matrix).tolist()]
    return np.asarray(ordered_centered + uncentered_units, dtype=np.int64)


def select_high_activation_positions(
    analysis_input: AnalysisInput,
    unit_indices: np.ndarray,
    *,
    threshold_fraction: float,
    max_points_per_unit: int,
    use_absolute_activations: bool,
) -> dict[int, np.ndarray]:
    flattened_values, flattened_positions = flatten_valid_steps(
        analysis_input.representation,
        analysis_input.position_xy,
        analysis_input.valid_mask,
    )
    if flattened_values.size == 0 or unit_indices.size == 0:
        return {}

    selected_values = flattened_values[:, unit_indices]
    if use_absolute_activations:
        magnitude = np.abs(selected_values)
    else:
        magnitude = selected_values
    thresholds = np.maximum(np.max(magnitude, axis=0) * threshold_fraction, 1e-8)

    selected_positions: dict[int, np.ndarray] = {}
    for local_index, unit_index in enumerate(unit_indices.tolist()):
        active_mask = magnitude[:, local_index] >= thresholds[local_index]
        active_positions = flattened_positions[active_mask]
        if active_positions.shape[0] > max_points_per_unit:
            sample_indices = np.linspace(
                0,
                active_positions.shape[0] - 1,
                num=max_points_per_unit,
                dtype=np.int32,
            )
            active_positions = active_positions[sample_indices]
        selected_positions[int(unit_index)] = active_positions.astype(np.float32, copy=False)
    return selected_positions


def use_shared_rate_map_color_scale(config: dict) -> bool:
    raw_value = config.get("rate_map_shared_color_scale", False)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw_value)


def resolve_rate_map_colormap_mode(config: dict) -> str:
    return _normalize_rate_map_colormap_mode(config.get("rate_map_colormap_mode", "reds"))


@dataclass(frozen=True, slots=True)
class PanelSettings:
    """Every rendering knob the module reads out of the analysis config, resolved once."""

    panel_metric_name: str
    panel_metric_fill_sigma_bins: float
    threshold_fraction: float
    threshold_quantile: float
    per_bin_cv_min_episodes: int
    split_half_agreement_min_episodes_per_half: int
    use_absolute_activations: bool
    shared_color_scale: bool
    colormap_mode: str
    emit_thresholded_reliability_panel: bool
    emit_quantile_thresholded_reliability_panel: bool
    negative_tolerance: float
    max_negative_bin_fraction: float
    max_negative_peak_fraction: float
    export_per_unit: bool
    spike_overlay_max_points: int
    panel_top_k: int
    grid_top_k: int
    unit_order_mode: str


@dataclass(frozen=True, slots=True)
class RankedUnits:
    """Which units the panel and the grid show, and in what order."""

    summary_indices: np.ndarray
    grid_indices: np.ndarray
    render_all_panel_units: bool
    render_all_grid_units: bool


@dataclass(slots=True)
class PanelFamilyOutcome:
    """What one panel family (primary, thresholded, quantile) wrote."""

    first_path: Path | None = None
    first_clean_path: Path | None = None
    page_paths: list[str] = field(default_factory=list)
    clean_page_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GridOutcome:
    """What the companion grid wrote."""

    first_path: Path | None = None
    first_clean_path: Path | None = None
    page_paths: list[str] = field(default_factory=list)
    clean_page_paths: list[str] = field(default_factory=list)


METRIC_KEYS_BY_GROUP: dict[str, set[str]] = {
    "fields": {
        "mean_peak_rate",
        "max_peak_rate",
        "median_negative_fraction",
        "fraction_units_majority_negative",
    },
    "reliability": {
        "mean_reliability",
        "mean_reliability_lift",
        "max_reliability_lift",
        "mean_reliability_inside_fields",
        "mean_reliability_lift_inside_fields",
        "mean_field_traversal_reliability",
        "field_traversal_assessable_units",
        "mean_field_traversal_reliability_directional",
        "field_traversal_directional_assessable_units",
        "mean_field_core_traversal_reliability",
        "field_core_traversal_assessable_units",
        "mean_quantile_thresholded_reliability",
        "mean_quantile_thresholded_reliability_lift",
        "mean_quantile_thresholded_reliability_inside_fields",
    },
    "bin_consistency": {
        "mean_bin_consistency",
        "mean_bin_consistency_inside_fields",
        "mean_bin_consistency_supported_field_fraction",
        "mean_bin_coefficient_of_variation",
    },
    "split_half": {
        "mean_split_half_agreement",
        "mean_split_half_agreement_inside_fields",
        "mean_split_half_agreement_supported_field_fraction",
        "mean_split_half_rate_map_correlation",
    },
    "episode_correlation": {
        "mean_episode_rate_map_correlation",
    },
    "coding_purity": {
        "mean_spatial_information_bits",
        "mean_spatial_coherence",
        "mean_max_available_confound_score",
        "mean_reliability_weighted_information",
        "max_reliability_weighted_information",
        "mean_reliability_weighted_information_all_units",
        "mean_reliability_weighted_information_place_cells",
        "mean_reliability_weighted_information_excess_all_units",
        "mean_reliability_weighted_information_excess_place_cells",
        "mean_coding_purity_score",
        "max_coding_purity_score",
        "fraction_place_cells",
        "fraction_passing_split_half",
        "fraction_passing_coherence",
        "fraction_passing_confound",
        "place_cell_assessable_units",
        "field_coverage_fraction",
        "fraction_significant_spatial_information",
        "fraction_place_cells_strict",
        "place_cell_gate_minimum_split_half",
        "place_cell_gate_minimum_coherence",
        "place_cell_gate_maximum_confound",
    },
}

PER_UNIT_KEYS_BY_GROUP: dict[str, set[str]] = {
    "fields": {
        "place_field_metrics_supported",
        "peak_rate",
        "peak_abs_activation",
        "negative_fraction",
        "field_count",
        "field_area_bins",
    },
    "reliability": {
        "mean_reliability",
        "max_reliability",
        "mean_reliability_lift",
        "max_reliability_lift",
        "mean_quantile_thresholded_reliability_lift",
        "max_quantile_thresholded_reliability_lift",
        "mean_reliability_inside_fields",
        "mean_reliability_lift_inside_fields",
        "field_traversal_reliability",
        "field_traversal_count",
        "field_traversal_reliability_directional",
        "field_traversal_directional_count",
        "field_core_traversal_reliability",
        "field_core_traversal_count",
        "mean_quantile_thresholded_reliability",
        "max_quantile_thresholded_reliability",
        "mean_quantile_thresholded_reliability_inside_fields",
    },
    "bin_consistency": {
        "mean_bin_consistency",
        "max_bin_consistency",
        "mean_bin_consistency_inside_fields",
        "bin_consistency_supported_field_fraction",
        "mean_bin_coefficient_of_variation",
    },
    "split_half": {
        "mean_split_half_agreement",
        "max_split_half_agreement",
        "mean_split_half_agreement_inside_fields",
        "split_half_agreement_supported_field_fraction",
        "split_half_rate_map_correlation",
    },
    "episode_correlation": {
        "episode_rate_map_correlation",
    },
    "coding_purity": {
        "spatial_information_bits",
        "spatial_coherence",
        "max_available_confound_score",
        "reliability_weighted_information",
        "reliability_weighted_information_excess",
        "coding_purity_score",
        "is_place_cell",
        "spatial_information_null_p",
        "spatial_information_null_95",
        "spatial_information_significant",
        "split_half_rate_map_correlation",
    },
}


def panel_metric_family(
    bundle: RateMapMetricBundle,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The per-bin maps, in-field means and in-field support for one panel metric."""
    if metric_name == "thresholded_reliability":
        return (
            bundle.reliability.thresholded_maps,
            bundle.fields.reliability,
            bundle.fields.reliability_supported_fraction,
        )
    if metric_name == "quantile_thresholded_reliability":
        return (
            bundle.reliability.quantile_maps,
            bundle.fields.quantile_reliability,
            bundle.fields.quantile_reliability_supported_fraction,
        )
    if metric_name == "split_half_agreement":
        return (
            bundle.reliability.split_half_agreement_maps,
            bundle.fields.split_half_agreement,
            bundle.fields.split_half_agreement_supported_fraction,
        )
    return (
        bundle.reliability.bin_consistency_maps,
        bundle.fields.bin_consistency,
        bundle.fields.bin_consistency_supported_fraction,
    )


def select_ranked_units(
    bundle: RateMapMetricBundle,
    settings: PanelSettings,
    config: dict,
) -> RankedUnits:
    """Take the top-k of the ranking for each figure, then optionally regroup by field position."""
    ranked_indices = bundle.ranked_indices
    render_all_panel_units = bool(config["rate_map_panel_show_all_units"])
    summary_indices = (
        ranked_indices
        if render_all_panel_units
        else ranked_indices[: min(settings.panel_top_k, len(ranked_indices))]
    )
    render_all_grid_units = bool(config["rate_map_grid_show_all_units"])
    grid_indices = (
        ranked_indices
        if render_all_grid_units
        else ranked_indices[: min(settings.grid_top_k, len(ranked_indices))]
    )
    if settings.unit_order_mode == "field_position":
        rate_maps = bundle.rate_map_result.rate_maps
        bounds = bundle.rate_map_result.bounds
        summary_indices = _order_indices_by_field_position(summary_indices, rate_maps, bounds)
        grid_indices = _order_indices_by_field_position(grid_indices, rate_maps, bounds)
    return RankedUnits(
        summary_indices=summary_indices,
        grid_indices=grid_indices,
        render_all_panel_units=render_all_panel_units,
        render_all_grid_units=render_all_grid_units,
    )


def rate_map_population_metrics(bundle: RateMapMetricBundle) -> dict[str, float]:
    """Population headlines, before the module's metric groups filter them."""
    place_metrics = bundle.place_metrics
    fields = bundle.fields
    summaries = bundle.summaries
    reliability = bundle.reliability
    signed = bundle.signed_diagnostics
    place_cell_passes = place_metrics.passes_gates
    reliability_weighted = place_metrics.reliability_weighted_information
    reliability_weighted_excess = place_metrics.reliability_weighted_information_excess
    gate_summary = place_metrics.gate_summary
    return {
        "mean_peak_rate": (
            float(signed.peak_magnitude.mean()) if len(signed.peak_magnitude) else 0.0
        ),
        "max_peak_rate": (
            float(signed.peak_magnitude.max()) if len(signed.peak_magnitude) else 0.0
        ),
        "mean_spatial_information_bits": nanmean_or_nan(place_metrics.spatial_information_bits),
        "mean_spatial_coherence": nanmean_or_nan(place_metrics.spatial_coherence),
        "mean_max_available_confound_score": nanmean_or_nan(place_metrics.max_available_confound),
        "mean_reliability_weighted_information": nanmean_or_nan(reliability_weighted),
        "max_reliability_weighted_information": nanmax_or_nan(reliability_weighted),
        "mean_reliability_weighted_information_all_units": (
            float(np.nan_to_num(reliability_weighted, nan=0.0).mean())
            if len(reliability_weighted)
            else 0.0
        ),
        "mean_reliability_weighted_information_place_cells": nanmean_or_nan(
            reliability_weighted[place_cell_passes]
        ),
        "mean_reliability_weighted_information_excess_all_units": (
            float(np.nan_to_num(reliability_weighted_excess, nan=0.0).mean())
            if np.isfinite(reliability_weighted_excess).any()
            else float("nan")
        ),
        "mean_reliability_weighted_information_excess_place_cells": nanmean_or_nan(
            reliability_weighted_excess[place_cell_passes]
        ),
        "mean_coding_purity_score": nanmean_or_nan(place_metrics.coding_purity),
        "max_coding_purity_score": nanmax_or_nan(place_metrics.coding_purity),
        "fraction_place_cells": gate_summary["fraction_place_cells"],
        "fraction_passing_split_half": gate_summary["fraction_passing_split_half"],
        "fraction_passing_coherence": gate_summary["fraction_passing_coherence"],
        "fraction_passing_confound": gate_summary["fraction_passing_confound"],
        "place_cell_assessable_units": gate_summary["place_cell_assessable_units"],
        "place_cell_gate_minimum_split_half": (bundle.settings.place_cell_gate_minimum_split_half),
        "place_cell_gate_minimum_coherence": bundle.settings.place_cell_gate_minimum_coherence,
        "place_cell_gate_maximum_confound": bundle.settings.place_cell_gate_maximum_confound,
        "field_coverage_fraction": place_metrics.field_coverage_fraction,
        "fraction_significant_spatial_information": (
            bundle.spatial_information_null.fraction_significant
        ),
        "fraction_place_cells_strict": (
            bundle.spatial_information_null.fraction_place_cells_strict
        ),
        "mean_reliability": (
            float(np.mean(summaries.mean_reliability)) if len(summaries.mean_reliability) else 0.0
        ),
        "mean_reliability_lift": (
            float(np.mean(summaries.mean_reliability_lift))
            if len(summaries.mean_reliability_lift)
            else 0.0
        ),
        "max_reliability_lift": (
            float(np.max(summaries.max_reliability_lift))
            if len(summaries.max_reliability_lift)
            else 0.0
        ),
        "mean_quantile_thresholded_reliability_lift": (
            float(np.mean(summaries.mean_quantile_reliability_lift))
            if len(summaries.mean_quantile_reliability_lift)
            else 0.0
        ),
        "mean_reliability_inside_fields": nanmean_or_nan(fields.reliability),
        "mean_reliability_lift_inside_fields": nanmean_or_nan(fields.reliability_lift),
        "mean_field_traversal_reliability": nanmean_or_nan(fields.traversal_reliability),
        "mean_field_traversal_reliability_directional": nanmean_or_nan(
            fields.traversal_reliability_directional
        ),
        "mean_field_core_traversal_reliability": nanmean_or_nan(fields.core_traversal_reliability),
        "field_core_traversal_assessable_units": float(
            np.isfinite(fields.core_traversal_reliability).sum()
        ),
        "field_traversal_directional_assessable_units": float(
            np.isfinite(fields.traversal_reliability_directional).sum()
        ),
        "field_traversal_assessable_units": float(np.isfinite(fields.traversal_reliability).sum()),
        "mean_quantile_thresholded_reliability": (
            float(np.mean(summaries.mean_quantile_reliability))
            if len(summaries.mean_quantile_reliability)
            else 0.0
        ),
        "mean_quantile_thresholded_reliability_inside_fields": nanmean_or_nan(
            fields.quantile_reliability
        ),
        "mean_bin_consistency": (
            float(np.mean(summaries.mean_bin_consistency))
            if len(summaries.mean_bin_consistency)
            else 0.0
        ),
        "mean_bin_consistency_inside_fields": nanmean_or_nan(fields.bin_consistency),
        "mean_bin_consistency_supported_field_fraction": nanmean_or_nan(
            fields.bin_consistency_supported_fraction
        ),
        "mean_split_half_agreement": (
            float(np.mean(summaries.mean_split_half_agreement))
            if len(summaries.mean_split_half_agreement)
            else 0.0
        ),
        "mean_split_half_agreement_inside_fields": nanmean_or_nan(fields.split_half_agreement),
        "mean_split_half_agreement_supported_field_fraction": nanmean_or_nan(
            fields.split_half_agreement_supported_fraction
        ),
        "mean_bin_coefficient_of_variation": nanmean_or_nan(
            summaries.mean_bin_coefficient_of_variation
        ),
        "mean_split_half_rate_map_correlation": (
            float(np.mean(reliability.split_half_rate_map_correlation))
            if len(reliability.split_half_rate_map_correlation)
            else 0.0
        ),
        "mean_episode_rate_map_correlation": (
            float(np.mean(reliability.episode_rate_map_correlation))
            if len(reliability.episode_rate_map_correlation)
            else 0.0
        ),
        "median_negative_fraction": signed.median_negative_fraction,
        "fraction_units_majority_negative": signed.fraction_units_majority_negative,
    }


def rate_map_per_unit_metrics(bundle: RateMapMetricBundle) -> dict[str, np.ndarray]:
    """One vector per metric, before the module's metric groups filter them."""
    place_metrics = bundle.place_metrics
    fields = bundle.fields
    summaries = bundle.summaries
    reliability = bundle.reliability
    signed = bundle.signed_diagnostics
    null = bundle.spatial_information_null
    return {
        "place_field_metrics_supported": place_metrics.supported.astype(np.float32, copy=False),
        "peak_rate": signed.peak_magnitude.astype(np.float32),
        "peak_abs_activation": signed.peak_abs_activation.astype(np.float32, copy=False),
        "negative_fraction": signed.negative_fraction.astype(np.float32, copy=False),
        "spatial_information_bits": place_metrics.spatial_information_bits,
        "spatial_coherence": place_metrics.spatial_coherence,
        "max_available_confound_score": place_metrics.max_available_confound,
        "reliability_weighted_information": place_metrics.reliability_weighted_information,
        "reliability_weighted_information_excess": (
            place_metrics.reliability_weighted_information_excess
        ),
        "coding_purity_score": place_metrics.coding_purity,
        "is_place_cell": place_metrics.passes_gates.astype(np.float32, copy=False),
        "spatial_information_null_p": null.null_p,
        "spatial_information_null_95": null.null_95,
        "spatial_information_significant": null.significant.astype(np.float32, copy=False),
        "mean_reliability": summaries.mean_reliability,
        "max_reliability": summaries.max_reliability,
        "mean_reliability_lift": summaries.mean_reliability_lift,
        "max_reliability_lift": summaries.max_reliability_lift,
        "mean_quantile_thresholded_reliability_lift": summaries.mean_quantile_reliability_lift,
        "max_quantile_thresholded_reliability_lift": summaries.max_quantile_reliability_lift,
        "mean_reliability_inside_fields": fields.reliability,
        "mean_reliability_lift_inside_fields": fields.reliability_lift,
        "field_traversal_reliability": fields.traversal_reliability,
        "field_traversal_count": fields.traversal_counts.astype(np.float32, copy=False),
        "field_traversal_reliability_directional": fields.traversal_reliability_directional,
        "field_core_traversal_reliability": fields.core_traversal_reliability,
        "field_core_traversal_count": fields.core_traversal_counts.astype(np.float32, copy=False),
        "field_traversal_directional_count": fields.traversal_directional_counts.astype(
            np.float32, copy=False
        ),
        "mean_quantile_thresholded_reliability": summaries.mean_quantile_reliability,
        "max_quantile_thresholded_reliability": summaries.max_quantile_reliability,
        "mean_quantile_thresholded_reliability_inside_fields": fields.quantile_reliability,
        "mean_bin_consistency": summaries.mean_bin_consistency,
        "max_bin_consistency": summaries.max_bin_consistency,
        "mean_bin_consistency_inside_fields": fields.bin_consistency,
        "bin_consistency_supported_field_fraction": fields.bin_consistency_supported_fraction,
        "mean_split_half_agreement": summaries.mean_split_half_agreement,
        "max_split_half_agreement": summaries.max_split_half_agreement,
        "mean_split_half_agreement_inside_fields": fields.split_half_agreement,
        "split_half_agreement_supported_field_fraction": (
            fields.split_half_agreement_supported_fraction
        ),
        "mean_bin_coefficient_of_variation": summaries.mean_bin_coefficient_of_variation,
        "split_half_rate_map_correlation": reliability.split_half_rate_map_correlation,
        "episode_rate_map_correlation": reliability.episode_rate_map_correlation,
        "field_count": fields.counts,
        "field_area_bins": fields.areas,
    }
