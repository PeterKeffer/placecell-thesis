"""Per-episode field-center and field-size stability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.bin_maps import bin_center_grids, smooth_flat_bin_maps
from ..numerics.occupancy import (
    iter_episode_activity_sum_chunks,
    reachable_bin_visited_fractions,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .occupancy import get_or_compute_episode_bin_statistics
from .timing import log_timing, record_timing
from .world_overlay import (
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    finalize_arena_axis,
    overlay_bounds,
    resolve_world_overlay,
)


@dataclass(slots=True)
class _FieldMapMetrics:
    """Peak rate plus field area and field centroid for a whole stack of rate maps."""

    peak_rates: np.ndarray
    field_areas: np.ndarray
    centroids_x: np.ndarray
    centroids_y: np.ndarray


def field_map_metrics(
    rate_maps: np.ndarray,
    threshold_fraction: float,
    *,
    x_center_grid: np.ndarray,
    y_center_grid: np.ndarray,
) -> _FieldMapMetrics:
    """Peak-thresholded field area and activity-weighted centroid for every map at once."""
    spatial_axes = (-2, -1)
    field_values = np.nan_to_num(rate_maps, nan=0.0)
    peak_rates = field_values.max(axis=spatial_axes)
    widened_peak_rates = peak_rates.astype(np.float64)
    thresholds = (threshold_fraction * widened_peak_rates).astype(np.float32)
    field_masks = field_values >= thresholds[..., None, None]
    field_masks &= (widened_peak_rates > 1e-8)[..., None, None]
    field_areas = field_masks.sum(axis=spatial_axes).astype(np.float32)

    weights = np.maximum(field_values, 0.0, out=field_values)
    weights *= field_masks
    total_weights = weights.sum(axis=spatial_axes).astype(np.float64)
    x_moments = (weights * x_center_grid).sum(axis=spatial_axes).astype(np.float64)
    y_moments = (weights * y_center_grid).sum(axis=spatial_axes).astype(np.float64)

    measurable = (field_areas > 0.0) & (total_weights > 1e-8)
    centroids_x = np.full(peak_rates.shape, np.nan, dtype=np.float64)
    centroids_y = np.full(peak_rates.shape, np.nan, dtype=np.float64)
    np.divide(x_moments, total_weights, out=centroids_x, where=measurable)
    np.divide(y_moments, total_weights, out=centroids_y, where=measurable)
    return _FieldMapMetrics(
        peak_rates=peak_rates,
        field_areas=np.where(measurable, field_areas, np.nan),
        centroids_x=centroids_x.astype(np.float32),
        centroids_y=centroids_y.astype(np.float32),
    )


@dataclass(slots=True)
class _FieldStabilityEpisodeMetrics:
    episode_centroids_x: np.ndarray
    episode_centroids_y: np.ndarray
    episode_field_areas: np.ndarray
    valid_episode_mask: np.ndarray


def _compute_field_stability_episode_metrics(
    analysis_input: AnalysisInput,
    *,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    min_occupancy: float,
    minimum_valid_steps: int,
    minimum_visited_fraction: float,
    field_threshold_fraction: float,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    x_center_grid: np.ndarray,
    y_center_grid: np.ndarray,
) -> _FieldStabilityEpisodeMetrics:
    num_units = analysis_input.representation.shape[-1]
    num_episodes = analysis_input.representation.shape[0]
    episode_centroids_x = np.full((num_episodes, num_units), np.nan, dtype=np.float32)
    episode_centroids_y = np.full((num_episodes, num_units), np.nan, dtype=np.float32)
    episode_field_areas = np.full((num_episodes, num_units), np.nan, dtype=np.float32)
    valid_episode_mask = np.zeros((num_episodes,), dtype=bool)

    episode_statistics = get_or_compute_episode_bin_statistics(
        analysis_input,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    if episode_statistics is None:
        return _FieldStabilityEpisodeMetrics(
            episode_centroids_x=episode_centroids_x,
            episode_centroids_y=episode_centroids_y,
            episode_field_areas=episode_field_areas,
            valid_episode_mask=valid_episode_mask,
        )

    episode_step_counts = episode_statistics.episode_bin_step_counts.astype(
        np.float32,
        copy=False,
    )
    episode_occupancy = smooth_flat_bin_maps(
        episode_step_counts,
        num_bins_y=num_bins_y,
        num_bins_x=num_bins_x,
        smoothing_sigma=smoothing_sigma,
    )
    safe_episode_occupancy = np.where(
        episode_occupancy >= min_occupancy,
        episode_occupancy,
        np.nan,
    )
    valid_step_counts = np.count_nonzero(analysis_input.valid_mask, axis=1)
    visited_fractions = reachable_bin_visited_fractions(
        episode_statistics.episode_bin_step_counts,
    )
    valid_episode_mask = (valid_step_counts >= minimum_valid_steps) & (
        visited_fractions >= minimum_visited_fraction
    )
    valid_episode_indices = np.flatnonzero(valid_episode_mask)
    if valid_episode_indices.size == 0:
        return _FieldStabilityEpisodeMetrics(
            episode_centroids_x=episode_centroids_x,
            episode_centroids_y=episode_centroids_y,
            episode_field_areas=episode_field_areas,
            valid_episode_mask=valid_episode_mask,
        )
    safe_valid_occupancy = safe_episode_occupancy[valid_episode_indices]

    for start_index, stop_index, chunk_activity_sums in iter_episode_activity_sum_chunks(
        episode_statistics,
        unit_chunk_size=16,
    ):
        chunk_rate_maps = (
            smooth_flat_bin_maps(
                chunk_activity_sums[:, valid_episode_indices],
                num_bins_y=num_bins_y,
                num_bins_x=num_bins_x,
                smoothing_sigma=smoothing_sigma,
            )
            / safe_valid_occupancy[None, :, :, :]
        ).astype(np.float32, copy=False)
        chunk_metrics = field_map_metrics(
            chunk_rate_maps,
            field_threshold_fraction,
            x_center_grid=x_center_grid,
            y_center_grid=y_center_grid,
        )
        episode_centroids_x[valid_episode_indices, start_index:stop_index] = (
            chunk_metrics.centroids_x.T
        )
        episode_centroids_y[valid_episode_indices, start_index:stop_index] = (
            chunk_metrics.centroids_y.T
        )
        episode_field_areas[valid_episode_indices, start_index:stop_index] = (
            chunk_metrics.field_areas.T
        )

    return _FieldStabilityEpisodeMetrics(
        episode_centroids_x=episode_centroids_x,
        episode_centroids_y=episode_centroids_y,
        episode_field_areas=episode_field_areas,
        valid_episode_mask=valid_episode_mask,
    )


@dataclass(slots=True)
class _FieldStabilityModuleBase:
    """Summarize per-episode field drift and size stability."""

    name: str = "field_stability_bundle"
    cost_tier: str = "standard"
    emit_metrics: bool = True
    emit_summary: bool = True
    emit_trajectories: bool = True

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}

        num_bins_x = int(config.get("per_episode_num_bins_x", 20))
        num_bins_y = int(config.get("per_episode_num_bins_y", 20))
        smoothing_sigma = float(config.get("per_episode_smoothing_sigma", 1.5))
        min_occupancy = float(config.get("per_episode_min_occupancy", 1e-6))
        minimum_visited_fraction = float(config.get("per_episode_minimum_visited_fraction", 0.05))
        configured_minimum_valid_steps = int(config.get("per_episode_minimum_valid_steps", 200))
        minimum_valid_steps = min(
            configured_minimum_valid_steps, int(analysis_input.representation.shape[1])
        )
        field_threshold_fraction = float(config.get("place_field_threshold_fraction", 0.2))
        trajectory_top_k = int(config.get("field_stability_top_k", 6))

        section_started_at = perf_counter()
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        pooled_rate_maps = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=world_bounds,
        )
        x_center_grid, y_center_grid = bin_center_grids(
            pooled_rate_maps.bounds,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
        )
        record_timing(timing_seconds, "pooled_rate_maps", section_started_at)

        section_started_at = perf_counter()
        num_units = analysis_input.representation.shape[-1]
        num_episodes = analysis_input.representation.shape[0]
        pooled_metrics = field_map_metrics(
            pooled_rate_maps.rate_maps,
            field_threshold_fraction,
            x_center_grid=x_center_grid,
            y_center_grid=y_center_grid,
        )
        pooled_centroids_x = pooled_metrics.centroids_x
        pooled_centroids_y = pooled_metrics.centroids_y
        pooled_peak_rate = pooled_metrics.peak_rates
        record_timing(timing_seconds, "pooled_field_metrics", section_started_at)

        section_started_at = perf_counter()
        episode_metrics = analysis_input.get_cached_metric(
            (
                "field_stability_episode_metrics",
                num_bins_x,
                num_bins_y,
                smoothing_sigma,
                min_occupancy,
                minimum_valid_steps,
                minimum_visited_fraction,
                field_threshold_fraction,
                pooled_rate_maps.bounds,
            ),
            lambda: _compute_field_stability_episode_metrics(
                analysis_input,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                smoothing_sigma=smoothing_sigma,
                min_occupancy=min_occupancy,
                minimum_valid_steps=minimum_valid_steps,
                minimum_visited_fraction=minimum_visited_fraction,
                field_threshold_fraction=field_threshold_fraction,
                bounds=pooled_rate_maps.bounds,
                x_center_grid=x_center_grid,
                y_center_grid=y_center_grid,
            ),
        )
        episode_centroids_x = episode_metrics.episode_centroids_x
        episode_centroids_y = episode_metrics.episode_centroids_y
        episode_field_areas = episode_metrics.episode_field_areas
        valid_episode_mask = episode_metrics.valid_episode_mask
        record_timing(timing_seconds, "episode_field_metrics", section_started_at)

        section_started_at = perf_counter()
        pooled_centroid_valid_mask = np.isfinite(pooled_centroids_x) & np.isfinite(
            pooled_centroids_y
        )
        per_unit_drift = np.zeros((num_units,), dtype=np.float32)
        per_unit_median_drift = np.zeros((num_units,), dtype=np.float32)
        per_unit_area_mean = np.zeros((num_units,), dtype=np.float32)
        per_unit_area_cv = np.zeros((num_units,), dtype=np.float32)
        per_unit_valid_episode_fraction = np.zeros((num_units,), dtype=np.float32)
        for unit_index in range(num_units):
            valid_centroid_mask = (
                np.isfinite(episode_centroids_x[:, unit_index])
                & np.isfinite(episode_centroids_y[:, unit_index])
                & pooled_centroid_valid_mask[unit_index]
            )
            valid_area_mask = np.isfinite(episode_field_areas[:, unit_index])
            if np.any(valid_centroid_mask):
                drift = np.sqrt(
                    np.square(
                        episode_centroids_x[valid_centroid_mask, unit_index]
                        - pooled_centroids_x[unit_index]
                    )
                    + np.square(
                        episode_centroids_y[valid_centroid_mask, unit_index]
                        - pooled_centroids_y[unit_index]
                    )
                ).astype(np.float32, copy=False)
                per_unit_drift[unit_index] = float(np.mean(drift))
                per_unit_median_drift[unit_index] = float(np.median(drift))
            if np.any(valid_area_mask):
                areas = episode_field_areas[valid_area_mask, unit_index]
                area_mean = float(np.mean(areas))
                per_unit_area_mean[unit_index] = area_mean
                if area_mean > 1e-8:
                    per_unit_area_cv[unit_index] = float(np.std(areas) / area_mean)
            per_unit_valid_episode_fraction[unit_index] = float(
                max(np.count_nonzero(valid_centroid_mask), np.count_nonzero(valid_area_mask))
                / max(num_episodes, 1)
            )
        record_timing(timing_seconds, "summaries", section_started_at)

        module_dir = output_dir / self.name
        summary_figure_path = (
            module_dir
            / f"field_stability__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        trajectory_figure_path = (
            module_dir / f"field_center_trajectories__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        drift_values = per_unit_drift[per_unit_valid_episode_fraction > 0.0]
        area_cv_values = per_unit_area_cv[per_unit_valid_episode_fraction > 0.0]
        figures: dict[str, Path] = {}
        if self.emit_summary:
            section_started_at = perf_counter()
            summary_figure_path.parent.mkdir(parents=True, exist_ok=True)
            summary_figure, summary_axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
            summary_axes[0].hist(drift_values, bins=24, color="#1F77B4", alpha=0.85)
            summary_axes[0].set_title("Field-Center Drift")
            summary_axes[0].set_xlabel("Mean drift distance (world units)")
            summary_axes[0].set_ylabel("Unit count")
            summary_axes[0].grid(alpha=0.3)
            summary_axes[1].hist(area_cv_values, bins=24, color="#2CA02C", alpha=0.85)
            summary_axes[1].set_title("Field-Size Stability")
            summary_axes[1].set_xlabel("Area CV across valid episodes")
            summary_axes[1].set_ylabel("Unit count")
            summary_axes[1].grid(alpha=0.3)
            summary_figure.suptitle(
                f"{analysis_input.source_name} field stability\n"
                f"valid episodes {int(np.count_nonzero(valid_episode_mask))}/{num_episodes}",
                fontsize=12,
            )
            summary_figure.tight_layout()
            summary_figure.savefig(summary_figure_path, dpi=160)
            plt.close(summary_figure)
            figures["field_stability_summary"] = summary_figure_path
            record_timing(timing_seconds, "render_summary", section_started_at)

        candidate_units = np.where(per_unit_valid_episode_fraction > 0.0)[0]
        ranked_units = candidate_units[
            np.argsort(
                5.0 * per_unit_valid_episode_fraction[candidate_units]
                + 1e-3 * pooled_peak_rate[candidate_units]
            )[::-1]
        ]
        selected_units = ranked_units[:trajectory_top_k]
        if self.emit_trajectories:
            section_started_at = perf_counter()
            trajectory_figure_path.parent.mkdir(parents=True, exist_ok=True)
            if len(selected_units) == 0:
                trajectory_figure, trajectory_axis = plt.subplots(figsize=(6.0, 4.0))
                trajectory_axis.axis("off")
                trajectory_axis.text(
                    0.5,
                    0.5,
                    "No units met the field-stability support thresholds.",
                    ha="center",
                    va="center",
                )
                trajectory_figure.tight_layout()
                trajectory_figure.savefig(trajectory_figure_path, dpi=160)
                plt.close(trajectory_figure)
            else:
                rows = int(np.ceil(len(selected_units) / 3.0))
                columns = min(3, len(selected_units))
                trajectory_figure, trajectory_axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(columns * 4.2, rows * 3.8),
                    squeeze=False,
                )
                flat_axes = trajectory_axes.reshape(-1)
                x_bounds, y_bounds = pooled_rate_maps.bounds
                for axis, unit_index in zip(flat_axes, selected_units, strict=False):
                    if world_overlay is not None:
                        draw_world_segments_on_axis(
                            axis, world_overlay.segments, line_color="#505050", line_width=1.0
                        )
                        draw_landmarks_on_axis(axis, world_overlay, marker_size=15.0)
                    valid_points = np.isfinite(episode_centroids_x[:, unit_index]) & np.isfinite(
                        episode_centroids_y[:, unit_index]
                    )
                    axis.plot(
                        episode_centroids_x[valid_points, unit_index],
                        episode_centroids_y[valid_points, unit_index],
                        color="#1F77B4",
                        linewidth=1.2,
                        alpha=0.8,
                    )
                    axis.scatter(
                        episode_centroids_x[valid_points, unit_index],
                        episode_centroids_y[valid_points, unit_index],
                        s=18.0,
                        color="#1F77B4",
                        alpha=0.85,
                    )
                    if pooled_centroid_valid_mask[unit_index]:
                        axis.scatter(
                            [pooled_centroids_x[unit_index]],
                            [pooled_centroids_y[unit_index]],
                            s=70.0,
                            marker="*",
                            color="#D62728",
                            edgecolors="#111111",
                            linewidths=0.4,
                            zorder=5,
                        )
                    finalize_arena_axis(
                        axis,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        world_overlay=world_overlay,
                    )
                    axis.set_title(
                        f"Unit {int(unit_index)}\n"
                        f"drift {per_unit_drift[unit_index]:.2f} | area CV "
                        f"{per_unit_area_cv[unit_index]:.2f}",
                        fontsize=9,
                    )
                    axis.grid(alpha=0.3)
                for axis in flat_axes[len(selected_units) :]:
                    axis.axis("off")
                trajectory_figure.suptitle(
                    f"{analysis_input.source_name} field-center trajectories",
                    fontsize=12,
                )
                trajectory_figure.tight_layout()
                trajectory_figure.savefig(trajectory_figure_path, dpi=160)
                plt.close(trajectory_figure)
            figures["field_center_trajectories"] = trajectory_figure_path
            record_timing(timing_seconds, "render_trajectories", section_started_at)
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )

        valid_stability_episodes = int(np.count_nonzero(valid_episode_mask))
        metrics = {
            "mean_field_center_drift_distance": float(np.mean(drift_values))
            if drift_values.size
            else float("nan"),
            "median_field_center_drift_distance": float(np.median(drift_values))
            if drift_values.size
            else float("nan"),
            "mean_field_area_cv": float(np.mean(area_cv_values))
            if area_cv_values.size
            else float("nan"),
            "mean_valid_field_episode_fraction": float(np.mean(per_unit_valid_episode_fraction))
            if len(per_unit_valid_episode_fraction)
            else float("nan"),
            "valid_stability_episodes": float(valid_stability_episodes),
        }
        per_unit_metrics = {
            "mean_field_center_drift_distance": per_unit_drift,
            "median_field_center_drift_distance": per_unit_median_drift,
            "mean_episode_field_area": per_unit_area_mean,
            "field_area_cv": per_unit_area_cv,
            "valid_field_episode_fraction": per_unit_valid_episode_fraction,
        }
        return AnalysisResult(
            metrics=metrics if self.emit_metrics else {},
            per_unit_metrics=per_unit_metrics if self.emit_metrics else {},
            figures=figures,
            tables={},
            metadata={
                "visualization": "field_stability",
                "valid_stability_episodes": valid_stability_episodes,
                "per_episode_num_bins_x": num_bins_x,
                "per_episode_num_bins_y": num_bins_y,
                "field_stability_timing_seconds": timing_seconds,
            },
        )


@dataclass(slots=True)
class FieldStabilityMetricsModule(_FieldStabilityModuleBase):
    """Compute per-episode field drift and area-stability metrics."""

    name: str = "field_stability_metrics"
    emit_summary: bool = False
    emit_trajectories: bool = False


@dataclass(slots=True)
class FieldStabilitySummaryModule(_FieldStabilityModuleBase):
    """Render field-stability summary histograms."""

    name: str = "field_stability_summary"
    emit_metrics: bool = False
    emit_trajectories: bool = False


@dataclass(slots=True)
class FieldStabilityTrajectoriesModule(_FieldStabilityModuleBase):
    """Render field-center trajectory panels."""

    name: str = "field_stability_trajectories"
    emit_metrics: bool = False
    emit_summary: bool = False
