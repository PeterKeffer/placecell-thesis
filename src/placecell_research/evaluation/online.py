"""Online evaluation on already-computed representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from placecell_research.numerics.fourier_ring import ring_metrics
from placecell_research.numerics.place_cell_quality import (
    DEFAULT_PLACE_CELL_GATE_THRESHOLDS,
    PlaceCellGateThresholds,
    batched_spatial_coherence,
    coding_purity_score,
    compute_available_confound_scores,
    field_coverage_fraction,
    fraction_place_cells,
    place_cell_pass_mask,
    reliability_weighted_information,
)
from placecell_research.numerics.rate_map_kernels import (
    DEFAULT_PLACE_METRIC_SETTINGS,
    PlaceMetricSettings,
    RateMapComputation,
    compute_rate_maps,
    gridness_score,
    gridness_summary_metrics,
    prepare_place_metric_rate_maps,
)
from placecell_research.numerics.split_half import (
    compute_split_half_rate_map_correlations,
)

from .decode import (
    DecodeResult,
    linear_decode_position,
    nonlinear_decode_position,
    supports_episode_level_decode,
)
from .metrics import (
    place_code_quality,
    summarize_code_sparsity,
    summarize_topk_scores,
    topk_spatial_information,
)


@dataclass(slots=True)
class OnlineEvaluationResult:
    """Evaluation bundle for one representation source."""

    source_name: str
    decode: DecodeResult | None
    nonlinear_decode: DecodeResult | None
    sparsity: dict[str, Any]
    spatial_information: dict[str, float] | None = None
    place_cell_quality: dict[str, dict[str, float]] | None = None
    population_metrics: dict[str, float] | None = None
    gridness: dict[str, float] | None = None
    band_pass: dict[str, float] | None = None

    def to_metrics(self) -> dict[str, float]:
        metrics = {
            f"{self.source_name}.fraction_active": float(self.sparsity["fraction_active"]),
        }
        if self.decode is not None:
            metrics.update(
                {
                    f"{self.source_name}.decode_rmse": self.decode.rmse,
                    f"{self.source_name}.decode_r2": self.decode.r2,
                }
            )
        if self.nonlinear_decode is not None:
            metrics.update(
                {
                    f"{self.source_name}.nonlinear_decode_rmse": self.nonlinear_decode.rmse,
                    f"{self.source_name}.nonlinear_decode_r2": self.nonlinear_decode.r2,
                }
            )
        if self.spatial_information:
            metrics.update(
                {
                    f"{self.source_name}.spatial_info_mean": (
                        self.spatial_information["mean_score"]
                    ),
                    f"{self.source_name}.spatial_info_mean_top_k": (
                        self.spatial_information["mean_top_k"]
                    ),
                    f"{self.source_name}.spatial_info_mean_top_k_selected": (
                        self.spatial_information.get("mean_top_k_selected", float("nan"))
                    ),
                    f"{self.source_name}.spatial_info_max": self.spatial_information["max_score"],
                }
            )
        if self.place_cell_quality:
            for metric_name, summary in self.place_cell_quality.items():
                metrics[f"{self.source_name}.{metric_name}_mean"] = float(summary["mean_score"])
                metrics[f"{self.source_name}.{metric_name}_mean_top_k"] = float(
                    summary["mean_top_k"]
                )
                metrics[f"{self.source_name}.{metric_name}_max"] = float(summary["max_score"])
        if self.population_metrics:
            for metric_name, value in self.population_metrics.items():
                metrics[f"{self.source_name}.{metric_name}"] = float(value)
        if self.gridness:
            for metric_name, value in self.gridness.items():
                metrics[f"{self.source_name}.{metric_name}"] = float(value)
        if self.band_pass:
            for metric_name, value in self.band_pass.items():
                metrics[f"{self.source_name}.{metric_name}"] = float(value)
        return metrics


def evaluate_representations(
    representations: dict[str, np.ndarray],
    positions_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    kinematics: np.ndarray | None = None,
    heading: np.ndarray | None = None,
    train_fraction: float,
    ridge_alpha: float,
    include_shuffle: bool,
    spatial_information_scores: dict[str, np.ndarray] | None = None,
    rate_map_results: dict[str, RateMapComputation] | None = None,
    spatial_information_top_k: int = 16,
    rate_map_num_bins_x: int = 60,
    rate_map_num_bins_y: int = 60,
    rate_map_smoothing_sigma: float = 0.4,
    rate_map_min_occupancy: float = 1e-6,
    compute_gridness: bool = False,
    nonlinear_decode_enabled: bool = False,
    nonlinear_decode_hidden_sizes: tuple[int, ...] = (128, 128),
    nonlinear_decode_max_epochs: int = 200,
    nonlinear_decode_batch_size: int = 1024,
    nonlinear_decode_max_train_samples: int = 65_536,
    nonlinear_decode_max_validation_samples: int = 16_384,
    nonlinear_decode_random_seed: int = 0,
    split_half_num_random_splits: int = 20,
    place_cell_gate_thresholds: PlaceCellGateThresholds = DEFAULT_PLACE_CELL_GATE_THRESHOLDS,
    place_metric_settings: PlaceMetricSettings = DEFAULT_PLACE_METRIC_SETTINGS,
) -> list[OnlineEvaluationResult]:
    """Evaluate several representation sources against the same positions."""
    results: list[OnlineEvaluationResult] = []
    flat_positions = positions_xy.reshape(-1, 2)
    flattened_mask = (
        valid_mask.reshape(-1).astype(bool, copy=False) if valid_mask is not None else None
    )
    if (
        flattened_mask is not None
        and flattened_mask.shape[0] == flat_positions.shape[0]
        and bool(flattened_mask.all())
    ):
        flattened_mask = None
    episode_ids = np.repeat(
        np.arange(positions_xy.shape[0], dtype=np.int64),
        positions_xy.shape[1],
    )
    if flattened_mask is not None:
        flat_positions = flat_positions[flattened_mask]
        episode_ids = episode_ids[flattened_mask]
    for source_name, source_values in representations.items():
        flattened = source_values.reshape(-1, source_values.shape[-1])
        if flattened_mask is not None:
            flattened = flattened[flattened_mask]
        decode = None
        nonlinear_decode = None
        if supports_episode_level_decode(episode_ids):
            decode = linear_decode_position(
                flattened,
                flat_positions,
                train_fraction=train_fraction,
                alpha=ridge_alpha,
                include_shuffle=include_shuffle,
                episode_ids=episode_ids,
            )
            if nonlinear_decode_enabled:
                nonlinear_decode = nonlinear_decode_position(
                    flattened,
                    flat_positions,
                    train_fraction=train_fraction,
                    hidden_sizes=nonlinear_decode_hidden_sizes,
                    max_epochs=nonlinear_decode_max_epochs,
                    batch_size=nonlinear_decode_batch_size,
                    max_train_samples=nonlinear_decode_max_train_samples,
                    max_validation_samples=nonlinear_decode_max_validation_samples,
                    random_seed=nonlinear_decode_random_seed,
                    include_shuffle=False,
                    episode_ids=episode_ids,
                )
        sparsity = summarize_code_sparsity(flattened)
        spatial_info_payload = None
        place_cell_quality_payload = None
        population_metrics_payload = None
        gridness_payload = None
        band_pass_payload = None
        quality_scores: dict[str, np.ndarray] = {}
        quality_summaries: dict[str, dict[str, float]] = {}
        spatial_information_values = (
            np.asarray(spatial_information_scores[source_name], dtype=np.float32)
            if spatial_information_scores and source_name in spatial_information_scores
            else None
        )
        if spatial_information_values is not None:
            summary = topk_spatial_information(
                spatial_information_values,
                spatial_information_top_k,
            )
            if np.isfinite(summary["mean_top_k"]) and np.isfinite(summary["max_score"]):
                spatial_info_payload = summary
        if spatial_information_values is not None:
            rate_map_result = (
                rate_map_results.get(source_name) if rate_map_results is not None else None
            )
            if rate_map_result is None:
                rate_map_result = compute_rate_maps(
                    source_values,
                    positions_xy,
                    valid_mask,
                    num_bins_x=rate_map_num_bins_x,
                    num_bins_y=rate_map_num_bins_y,
                    smoothing_sigma=rate_map_smoothing_sigma,
                    min_occupancy=rate_map_min_occupancy,
                )
            if compute_gridness:
                gridness_scores = np.asarray(
                    [gridness_score(rate_map) for rate_map in rate_map_result.rate_maps],
                    dtype=np.float32,
                )
                gridness_payload = gridness_summary_metrics(gridness_scores)
                band_pass_payload = ring_metrics(rate_map_result.rate_maps)
            confound_scores = compute_available_confound_scores(
                source_values,
                positions_xy,
                valid_mask,
                kinematics=kinematics,
                heading=heading,
            )
            split_half_rate_map_correlation = compute_split_half_rate_map_correlations(
                source_values,
                positions_xy,
                valid_mask,
                num_bins_x=rate_map_num_bins_x,
                num_bins_y=rate_map_num_bins_y,
                smoothing_sigma=rate_map_smoothing_sigma,
                min_occupancy=rate_map_min_occupancy,
                bounds=rate_map_result.bounds,
                num_random_splits=split_half_num_random_splits,
            )
            spatial_coherence_scores = batched_spatial_coherence(rate_map_result.rate_maps)
            quality_scores["reliability_weighted_information"] = reliability_weighted_information(
                spatial_information_values,
                split_half_rate_map_correlation,
            )
            quality_scores["coding_purity_score"] = coding_purity_score(
                spatial_information_values,
                spatial_coherence_scores,
                split_half_rate_map_correlation,
                confound_scores["max_available_confound_score"],
            )
            for metric_name, scores in quality_scores.items():
                quality_summaries[metric_name] = summarize_topk_scores(
                    scores, spatial_information_top_k
                )
            place_cell_quality_payload = quality_summaries
            supported_mask = prepare_place_metric_rate_maps(
                rate_map_result.rate_maps,
                negative_tolerance=place_metric_settings.negative_tolerance,
                max_negative_bin_fraction=place_metric_settings.max_negative_bin_fraction,
                max_negative_peak_fraction=place_metric_settings.max_negative_peak_fraction,
            ).supported_mask
            gate_summary = fraction_place_cells(
                split_half_rate_map_correlation,
                spatial_coherence_scores,
                confound_scores["max_available_confound_score"],
                supported_mask=supported_mask,
                minimum_split_half=place_cell_gate_thresholds.minimum_split_half,
                minimum_coherence=place_cell_gate_thresholds.minimum_coherence,
                maximum_confound=place_cell_gate_thresholds.maximum_confound,
            )
            pass_mask, _, _, _, _ = place_cell_pass_mask(
                split_half_rate_map_correlation,
                spatial_coherence_scores,
                confound_scores["max_available_confound_score"],
                supported_mask=supported_mask,
                minimum_split_half=place_cell_gate_thresholds.minimum_split_half,
                minimum_coherence=place_cell_gate_thresholds.minimum_coherence,
                maximum_confound=place_cell_gate_thresholds.maximum_confound,
            )
            field_coverage = field_coverage_fraction(
                rate_map_result.rate_maps,
                rate_map_result.raw_occupancy,
                pass_mask,
                threshold_fraction=place_metric_settings.field_threshold_fraction,
            )
            population_metrics_payload = {
                "place_code_fraction_place_cells": gate_summary["fraction_place_cells"],
                "place_code_field_coverage": field_coverage,
                "fraction_passing_split_half": gate_summary["fraction_passing_split_half"],
                "fraction_passing_coherence": gate_summary["fraction_passing_coherence"],
                "fraction_passing_confound": gate_summary["fraction_passing_confound"],
                "place_cell_assessable_units": gate_summary["place_cell_assessable_units"],
            }
            if decode is not None:
                population_metrics_payload.update(
                    place_code_quality(
                        float(decode.r2),
                        gate_summary["fraction_place_cells"],
                        field_coverage,
                    )
                )
        results.append(
            OnlineEvaluationResult(
                source_name=source_name,
                decode=decode,
                nonlinear_decode=nonlinear_decode,
                sparsity=sparsity,
                spatial_information=spatial_info_payload,
                place_cell_quality=place_cell_quality_payload,
                population_metrics=population_metrics_payload,
                gridness=gridness_payload,
                band_pass=band_pass_payload,
            )
        )
    return results
