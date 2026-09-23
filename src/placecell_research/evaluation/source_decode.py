"""Decode / sparsity / spatial-information metrics for one representation source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from placecell_research.numerics.place_cell_quality import (
    DEFAULT_PLACE_CELL_GATE_THRESHOLDS,
    PlaceCellGateThresholds,
)
from placecell_research.numerics.rate_map_kernels import (
    PlaceMetricSettings,
    compute_rate_maps,
    skaggs_spatial_information,
)

from .online import evaluate_representations


@dataclass(frozen=True)
class SourceDecodeSettings:
    """The rate-map and decode knobs the kernel reads, flattened out of the experiment config."""

    num_bins_x: int
    num_bins_y: int
    smoothing_sigma: float
    min_occupancy: float
    decode_train_fraction: float
    decode_ridge_alpha: float
    decode_include_shuffle: bool
    spatial_info_top_k: int
    split_half_num_random_splits: int
    place_metric_settings: PlaceMetricSettings
    place_cell_gate_thresholds: PlaceCellGateThresholds = DEFAULT_PLACE_CELL_GATE_THRESHOLDS


def source_decode_metrics(
    representation_array: np.ndarray,
    position_array: np.ndarray,
    valid_array: np.ndarray,
    heading_array: np.ndarray | None,
    kinematics_array: np.ndarray | None,
    *,
    settings: SourceDecodeSettings,
) -> dict[str, float]:
    """Decode / sparsity / spatial-information / quality metrics for ONE representation source."""
    rate_map_result = compute_rate_maps(
        representation_array,
        position_array,
        valid_array,
        num_bins_x=settings.num_bins_x,
        num_bins_y=settings.num_bins_y,
        smoothing_sigma=settings.smoothing_sigma,
        min_occupancy=settings.min_occupancy,
    )
    spatial_information = np.asarray(
        skaggs_spatial_information(rate_map_result.rate_maps, rate_map_result.occupancy),
        dtype=np.float32,
    )
    result = evaluate_representations(
        {"source": representation_array},
        position_array,
        valid_mask=valid_array,
        kinematics=kinematics_array,
        heading=heading_array,
        train_fraction=settings.decode_train_fraction,
        ridge_alpha=settings.decode_ridge_alpha,
        include_shuffle=settings.decode_include_shuffle,
        spatial_information_scores={"source": spatial_information},
        rate_map_results={"source": rate_map_result},
        spatial_information_top_k=settings.spatial_info_top_k,
        rate_map_num_bins_x=settings.num_bins_x,
        rate_map_num_bins_y=settings.num_bins_y,
        rate_map_smoothing_sigma=settings.smoothing_sigma,
        rate_map_min_occupancy=settings.min_occupancy,
        split_half_num_random_splits=settings.split_half_num_random_splits,
        place_cell_gate_thresholds=settings.place_cell_gate_thresholds,
        place_metric_settings=settings.place_metric_settings,
    )[0]
    metrics: dict[str, float] = {
        "code_sparsity_mean": float(result.sparsity["mean_activation"]),
        "code_fraction_active": float(result.sparsity["fraction_active"]),
    }
    if result.decode is not None:
        metrics["xy_decode_rmse"] = float(result.decode.rmse)
        metrics["xy_decode_r2"] = float(result.decode.r2)
    if result.spatial_information is not None:
        metrics["spatial_info_mean_top_k"] = float(result.spatial_information["mean_top_k"])
        metrics["spatial_info_max"] = float(result.spatial_information["max_score"])
    if result.place_cell_quality is not None:
        for metric_name, summary in result.place_cell_quality.items():
            metrics[f"{metric_name}_mean_top_k"] = float(summary["mean_top_k"])
            metrics[f"{metric_name}_max"] = float(summary["max_score"])
    if result.population_metrics is not None:
        for metric_name, value in result.population_metrics.items():
            metrics[metric_name] = float(value)
    return metrics
