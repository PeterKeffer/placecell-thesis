"""Population coverage analysis for pooled place fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from ..numerics.rate_map_kernels import compute_place_field_mask
from .base import AnalysisInput, AnalysisResult
from .figures import despine
from .helpers import get_or_compute_rate_maps
from .world_overlay import (
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    finalize_arena_axis,
    overlay_bounds,
    resolve_world_overlay,
    style_arena_axes,
)


def _positive_occupancy_norm(occupancy: np.ndarray) -> colors.Normalize:
    occupied_values = occupancy[occupancy > 0.0]
    if occupied_values.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0)
    return colors.PowerNorm(
        gamma=0.55,
        vmin=float(occupied_values.min()),
        vmax=float(occupied_values.max()),
    )


def _coverage_norm(coverage_counts: np.ndarray, visited_mask: np.ndarray) -> colors.Normalize:
    if not np.any(visited_mask):
        return colors.Normalize(vmin=0.0, vmax=1.0)
    visited_coverage = coverage_counts[visited_mask]
    return colors.PowerNorm(
        gamma=0.6,
        vmin=max(float(np.min(visited_coverage)), 1.0),
        vmax=max(float(np.max(visited_coverage)), 1.0),
    )


def _coverage_gap_mask(
    occupancy: np.ndarray,
    coverage_counts: np.ndarray,
    *,
    visited_mask: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    occupied_values = occupancy[visited_mask]
    if occupied_values.size == 0:
        return np.zeros_like(coverage_counts, dtype=bool), 0.0, 0
    occupancy_threshold = float(np.median(occupied_values))
    high_occupancy_mask = visited_mask & (occupancy > occupancy_threshold)
    gap_mask = high_occupancy_mask & (coverage_counts <= 1.0)
    high_occupancy_count = int(np.count_nonzero(high_occupancy_mask))
    gap_fraction = (
        float(np.count_nonzero(gap_mask) / high_occupancy_count) if high_occupancy_count else 0.0
    )
    return gap_mask, gap_fraction, high_occupancy_count


def _log_occupancy_coverage_correlation(
    occupancy: np.ndarray,
    coverage_counts: np.ndarray,
    visited_mask: np.ndarray,
) -> float:
    if np.count_nonzero(visited_mask) < 2:
        return 0.0
    log_occupancy = np.log1p(occupancy[visited_mask]).astype(np.float32, copy=False)
    visited_coverage = coverage_counts[visited_mask].astype(np.float32, copy=False)
    if float(np.std(log_occupancy)) <= 1e-8 or float(np.std(visited_coverage)) <= 1e-8:
        return 0.0
    return float(np.corrcoef(log_occupancy, visited_coverage)[0, 1])


def _coverage_residuals(log_occupancy: np.ndarray, coverage_counts: np.ndarray) -> np.ndarray:
    if len(log_occupancy) < 2 or float(np.std(log_occupancy)) <= 1e-8:
        return np.zeros_like(coverage_counts, dtype=np.float32)
    slope, intercept = np.polyfit(log_occupancy, coverage_counts, deg=1)
    expected_coverage = slope * log_occupancy + intercept
    return (coverage_counts - expected_coverage).astype(np.float32, copy=False)


def _draw_spatial_context(
    axis: plt.Axes,
    *,
    world_overlay: object,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    if world_overlay is not None:
        draw_world_segments_on_axis(
            axis,
            world_overlay.segments,
            line_color="#F2F2F2",
            line_width=1.1,
        )
        draw_landmarks_on_axis(axis, world_overlay, marker_size=18.0)
    finalize_arena_axis(
        axis,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        world_overlay=world_overlay,
    )
    style_arena_axes(axis, label_x=False, label_y=False)


def _write_coverage_gap_summary(
    output_path: Path,
    *,
    occupancy: np.ndarray,
    coverage_counts: np.ndarray,
    visited_mask: np.ndarray,
    gap_mask: np.ndarray,
    occupancy_norm: colors.Normalize,
    coverage_norm: colors.Normalize,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    world_overlay: object,
    title_prefix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_bounds, y_bounds = bounds
    extent = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])
    occupancy_display = np.ma.masked_where(~visited_mask, occupancy)
    coverage_display = np.ma.masked_where(~visited_mask, coverage_counts)
    gap_display = np.ma.masked_where(~gap_mask, gap_mask.astype(np.float32, copy=False))

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.8))
    occupancy_image = axes[0, 0].imshow(
        occupancy_display,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="YlOrRd",
        norm=occupancy_norm,
    )
    axes[0, 0].set_title("Dataset occupancy")
    _draw_spatial_context(
        axes[0, 0],
        world_overlay=world_overlay,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )
    figure.colorbar(occupancy_image, ax=axes[0, 0], shrink=0.82).set_label(
        "Valid steps per spatial bin"
    )

    coverage_image = axes[0, 1].imshow(
        coverage_display,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="turbo",
        norm=coverage_norm,
    )
    axes[0, 1].set_title("Population field coverage")
    _draw_spatial_context(
        axes[0, 1],
        world_overlay=world_overlay,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )
    figure.colorbar(coverage_image, ax=axes[0, 1], shrink=0.82).set_label("Covering units")

    gap_image = axes[1, 0].imshow(
        gap_display,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="Reds",
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
    )
    axes[1, 0].set_title("High-occupancy, weakly covered bins")
    _draw_spatial_context(
        axes[1, 0],
        world_overlay=world_overlay,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )
    figure.colorbar(gap_image, ax=axes[1, 0], shrink=0.82).set_label("Coverage gap")

    if np.any(visited_mask):
        log_occupancy = np.log1p(occupancy[visited_mask])
        visited_coverage = coverage_counts[visited_mask]
        axes[1, 1].scatter(log_occupancy, visited_coverage, s=12.0, alpha=0.55, linewidths=0.0)
        if len(log_occupancy) >= 2 and float(np.std(log_occupancy)) > 1e-8:
            slope, intercept = np.polyfit(log_occupancy, visited_coverage, deg=1)
            x_line = np.linspace(float(log_occupancy.min()), float(log_occupancy.max()), num=64)
            axes[1, 1].plot(x_line, slope * x_line + intercept, color="#2B2B2B", linewidth=1.2)
    axes[1, 1].set_title("Occupancy vs field coverage")
    axes[1, 1].set_xlabel("log(1 + valid steps) per bin")
    axes[1, 1].set_ylabel("covering units")
    axes[1, 1].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    figure.suptitle(title_prefix, fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_coverage_correlation_figure(
    output_path: Path,
    *,
    occupancy: np.ndarray,
    coverage_counts: np.ndarray,
    visited_mask: np.ndarray,
    gap_mask: np.ndarray,
    correlation: float,
    title_prefix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.6, 5.4))
    if np.any(visited_mask):
        log_occupancy = np.log1p(occupancy[visited_mask])
        visited_coverage = coverage_counts[visited_mask]
        residuals = _coverage_residuals(log_occupancy, visited_coverage)
        residual_limit = max(float(np.max(np.abs(residuals))), 1.0)
        scatter = axis.scatter(
            log_occupancy,
            visited_coverage,
            c=residuals,
            s=18.0,
            alpha=0.72,
            linewidths=0.0,
            cmap="coolwarm_r",
            norm=colors.TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit),
        )
        visited_gaps = gap_mask[visited_mask]
        if np.any(visited_gaps):
            axis.scatter(
                log_occupancy[visited_gaps],
                visited_coverage[visited_gaps],
                s=40.0,
                facecolors="none",
                edgecolors="#1F1F1F",
                linewidths=0.8,
            )
        if len(log_occupancy) >= 2 and float(np.std(log_occupancy)) > 1e-8:
            slope, intercept = np.polyfit(log_occupancy, visited_coverage, deg=1)
            x_line = np.linspace(float(log_occupancy.min()), float(log_occupancy.max()), num=96)
            axis.plot(x_line, slope * x_line + intercept, color="#1F1F1F", linewidth=1.4)
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("coverage residual: actual - expected units")

    axis.set_title(f"{title_prefix}\nr={correlation:.3f}")
    axis.set_xlabel("dataset coverage: log(1 + valid steps) per bin")
    axis.set_ylabel("population coverage: covering units per bin")
    axis.grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)
    despine(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@dataclass(slots=True)
class PopulationCoverageModule:
    """Render where pooled place fields cover the environment."""

    name: str = "population_coverage"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        field_threshold_fraction = float(config.get("place_field_threshold_fraction", 0.2))
        num_bins_x = int(config.get("num_bins_x", 60))
        num_bins_y = int(config.get("num_bins_y", 60))
        smoothing_sigma = float(config.get("smoothing_sigma", 0.3))
        min_occupancy = float(config.get("min_occupancy", 1e-6))
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        rate_map_result = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=world_bounds,
        )
        coverage_counts = np.zeros((num_bins_y, num_bins_x), dtype=np.float32)
        active_unit_mask = np.zeros((rate_map_result.rate_maps.shape[0],), dtype=bool)
        for unit_index, rate_map in enumerate(rate_map_result.rate_maps):
            field_mask, _, field_area = compute_place_field_mask(rate_map, field_threshold_fraction)
            if field_area <= 0.0:
                continue
            active_unit_mask[unit_index] = True
            coverage_counts += field_mask.astype(np.float32, copy=False)

        active_unit_count = int(np.count_nonzero(active_unit_mask))
        active_unit_fraction = (
            float(active_unit_count / len(active_unit_mask)) if len(active_unit_mask) else 0.0
        )
        coverage_fraction = (
            coverage_counts / active_unit_count
            if active_unit_count > 0
            else np.zeros_like(coverage_counts, dtype=np.float32)
        )
        visited_mask = np.asarray(rate_map_result.raw_occupancy > 0.0, dtype=bool)
        visited_coverage = coverage_counts[visited_mask]
        visited_fraction = coverage_fraction[visited_mask]
        gap_mask, gap_fraction, high_occupancy_count = _coverage_gap_mask(
            rate_map_result.occupancy,
            coverage_counts,
            visited_mask=visited_mask,
        )
        occupancy_coverage_correlation = _log_occupancy_coverage_correlation(
            rate_map_result.occupancy,
            coverage_counts,
            visited_mask,
        )
        uncovered_visited_bin_fraction = (
            float(np.mean(visited_coverage <= 0.0)) if visited_coverage.size else 0.0
        )

        module_dir = output_dir / self.name
        figure_path = module_dir / (
            f"population_coverage__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        x_bounds, y_bounds = rate_map_result.bounds
        coverage_display = np.ma.masked_where(~visited_mask, coverage_counts)
        occupancy_norm = _positive_occupancy_norm(rate_map_result.occupancy)
        coverage_norm = _coverage_norm(coverage_counts, visited_mask)

        figure, axis = plt.subplots(figsize=(6.2, 5.8))
        image = axis.imshow(
            coverage_display,
            origin="lower",
            extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
            aspect="equal",
            cmap="turbo",
            norm=coverage_norm,
        )
        if world_overlay is not None:
            draw_world_segments_on_axis(
                axis,
                world_overlay.segments,
                line_color="#F2F2F2",
                line_width=1.1,
            )
            draw_landmarks_on_axis(axis, world_overlay, marker_size=18.0)
        finalize_arena_axis(
            axis,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            world_overlay=world_overlay,
        )
        axis.set_title(
            f"{analysis_input.source_name} population coverage\n"
            f"active field units {active_unit_count}/{len(active_unit_mask)}"
        )
        style_arena_axes(axis)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
        colorbar.set_label("Units whose pooled field covers this bin")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)

        gap_summary_path = module_dir / (
            f"coverage_gap_summary__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        _write_coverage_gap_summary(
            gap_summary_path,
            occupancy=rate_map_result.occupancy,
            coverage_counts=coverage_counts,
            visited_mask=visited_mask,
            gap_mask=gap_mask,
            occupancy_norm=occupancy_norm,
            coverage_norm=coverage_norm,
            bounds=rate_map_result.bounds,
            world_overlay=world_overlay,
            title_prefix=f"{analysis_input.source_name} experience vs population coverage",
        )

        correlation_figure_path = module_dir / (
            f"coverage_correlation__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        _write_coverage_correlation_figure(
            correlation_figure_path,
            occupancy=rate_map_result.occupancy,
            coverage_counts=coverage_counts,
            visited_mask=visited_mask,
            gap_mask=gap_mask,
            correlation=occupancy_coverage_correlation,
            title_prefix=f"{analysis_input.source_name} dataset vs population coverage",
        )

        return AnalysisResult(
            metrics={
                "active_field_unit_count": float(active_unit_count),
                "active_field_unit_fraction": active_unit_fraction,
                "mean_population_coverage_units": (
                    float(np.mean(visited_coverage)) if visited_coverage.size else 0.0
                ),
                "mean_population_coverage_fraction": (
                    float(np.mean(visited_fraction)) if visited_fraction.size else 0.0
                ),
                "max_population_coverage_units": (
                    float(np.max(visited_coverage)) if visited_coverage.size else 0.0
                ),
                "uncovered_visited_bin_fraction": uncovered_visited_bin_fraction,
                "high_occupancy_low_coverage_bin_fraction": gap_fraction,
                "high_occupancy_bin_count": float(high_occupancy_count),
                "log_occupancy_coverage_correlation": occupancy_coverage_correlation,
            },
            per_unit_metrics={
                "active_field_unit_mask": active_unit_mask.astype(np.float32, copy=False),
            },
            figures={
                "population_coverage_map": figure_path,
                "coverage_gap_summary": gap_summary_path,
                "coverage_correlation_scatter": correlation_figure_path,
            },
            tables={},
            metadata={
                "visualization": "population_field_coverage",
            },
        )
