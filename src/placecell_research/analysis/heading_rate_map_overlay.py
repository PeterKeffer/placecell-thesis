"""Per-neuron rate maps with heading overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    scatter_add_over_units,
)
from .base import AnalysisInput, AnalysisResult
from .figures import save_figure
from .helpers import (
    get_or_compute_rate_maps,
    write_csv,
)
from .rate_map_rendering import draw_unit_rate_map
from .world_overlay import (
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    finalize_arena_axis,
    overlay_bounds,
    resolve_world_overlay,
)

_EPS = 1e-9
_TWO_PI = 2.0 * np.pi


@dataclass(slots=True)
class HeadingRateMapOverlayMetrics:
    heading_bin_rates: np.ndarray
    heading_bin_reliability: np.ndarray
    heading_bin_support: np.ndarray
    heading_bin_occupancy: np.ndarray
    heading_bin_field_occupancy: np.ndarray
    preferred_heading_rad: np.ndarray
    heading_vector_length: np.ndarray
    activation_corridor_deg: np.ndarray
    peak_to_mean: np.ndarray
    heading_support_fraction: np.ndarray
    local_preferred_heading: np.ndarray
    local_selectivity: np.ndarray
    local_support_fraction: np.ndarray


@dataclass(slots=True)
class HeadingRateMapOverlayModule:
    """Render rate map, local heading overlay, and heading tuning curve per unit."""

    name: str = "heading_rate_map_overlay"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_units = int(analysis_input.representation.shape[-1])
        if analysis_input.heading is None:
            nan_values = np.full(num_units, np.nan, dtype=np.float32)
            return AnalysisResult(
                metrics={"units_rendered": 0.0},
                per_unit_metrics={
                    "activation_corridor_deg": nan_values.copy(),
                    "heading_peak_to_mean": nan_values.copy(),
                    "heading_vector_length": nan_values.copy(),
                    "heading_support_fraction": nan_values.copy(),
                },
                figures={},
                tables={},
                metadata={"reason": "heading_missing"},
            )

        num_bins_x = int(config["heading_rate_map_overlay_num_bins_x"])
        num_bins_y = int(config["heading_rate_map_overlay_num_bins_y"])
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        rate_maps = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=float(config["heading_rate_map_overlay_smoothing_sigma"]),
            min_occupancy=float(config["heading_rate_map_overlay_min_occupancy"]),
            bounds=world_bounds,
        )
        overlay = _compute_heading_rate_map_overlay_metrics(
            analysis_input,
            config=config,
            bounds=rate_maps.bounds,
        )

        ranked_units = _rank_units(rate_maps.rate_maps, overlay.activation_corridor_deg)
        top_k = int(config["heading_rate_map_overlay_top_k"])
        ranked_units = ranked_units[:top_k] if top_k > 0 else ranked_units
        module_dir = output_dir / self.name
        figures = _render_pages(
            module_dir,
            analysis_input=analysis_input,
            rate_maps=rate_maps.rate_maps,
            bounds=rate_maps.bounds,
            world_overlay=world_overlay,
            overlay=overlay,
            unit_indices=ranked_units,
            page_size=int(config.get("heading_rate_map_overlay_page_size", 8)),
            render_dpi=int(config["heading_rate_map_overlay_render_dpi"]),
        )
        table_path = _write_table(module_dir, analysis_input, overlay)
        rendered_corridors = overlay.activation_corridor_deg[ranked_units]
        rendered_corridors = rendered_corridors[np.isfinite(rendered_corridors)]
        return AnalysisResult(
            metrics={
                "units_rendered": float(ranked_units.size),
                "median_activation_corridor_deg": float(np.median(rendered_corridors))
                if rendered_corridors.size
                else float("nan"),
            },
            per_unit_metrics={
                "activation_corridor_deg": overlay.activation_corridor_deg,
                "heading_peak_to_mean": overlay.peak_to_mean,
                "heading_vector_length": overlay.heading_vector_length,
                "heading_support_fraction": overlay.heading_support_fraction,
            },
            figures=figures,
            tables={"per_unit_metrics": table_path},
            metadata={
                "heading_rate_map_overlay_num_bins_x": num_bins_x,
                "heading_rate_map_overlay_num_bins_y": num_bins_y,
                "heading_rate_map_overlay_page_size": max(
                    1, int(config.get("heading_rate_map_overlay_page_size", 8))
                ),
            },
        )


def _compute_heading_rate_map_overlay_metrics(
    analysis_input: AnalysisInput,
    *,
    config: dict,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
) -> HeadingRateMapOverlayMetrics:
    num_units = int(analysis_input.representation.shape[-1])
    num_bins_x = int(config["heading_rate_map_overlay_local_num_bins_x"])
    num_bins_y = int(config["heading_rate_map_overlay_local_num_bins_y"])
    num_heading_bins = int(config.get("heading_rate_map_overlay_num_heading_bins", 36))
    min_heading_occupancy = int(
        config.get("heading_rate_map_overlay_min_occupancy_per_heading_bin", 5)
    )
    empty = _empty_metrics(num_units, num_bins_y, num_bins_x, num_heading_bins)
    if analysis_input.heading is None:
        return empty

    valid_flat = analysis_input.valid_mask.reshape(-1).astype(bool, copy=False)
    codes = analysis_input.representation.reshape(-1, num_units)[valid_flat]
    positions = analysis_input.position_xy.reshape(-1, 2)[valid_flat]
    headings = analysis_input.heading.reshape(-1)[valid_flat]
    if codes.shape[0] == 0:
        return empty

    rates = np.clip(codes, 0.0, None).astype(np.float64)
    active_threshold = float(config["heading_rate_map_overlay_active_threshold"])
    fired = (rates > active_threshold).astype(np.float64)
    field_threshold_fraction = float(config["heading_rate_map_overlay_field_threshold_fraction"])
    heading_bins = (
        np.floor((headings % _TWO_PI) / (_TWO_PI / num_heading_bins)).astype(int)
    ) % num_heading_bins

    spatial_bins, _, _, _ = compute_spatial_bin_assignments(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    num_spatial_bins = num_bins_x * num_bins_y
    combined_bins = spatial_bins * num_heading_bins + heading_bins
    combined_occupancy = (
        np.bincount(combined_bins, minlength=num_spatial_bins * num_heading_bins)
        .astype(np.float64)
        .reshape(num_spatial_bins, num_heading_bins)
    )
    combined_activity = scatter_add_over_units(
        combined_bins, rates, num_spatial_bins * num_heading_bins
    ).reshape(num_spatial_bins, num_heading_bins, num_units)
    combined_fired = scatter_add_over_units(
        combined_bins, fired, num_spatial_bins * num_heading_bins
    ).reshape(num_spatial_bins, num_heading_bins, num_units)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_heading_mean = combined_activity / combined_occupancy[:, :, None]

    spatial_occupancy = combined_occupancy.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        spatial_mean = combined_activity.sum(axis=1) / spatial_occupancy[:, None]
    occupied = spatial_occupancy > 0.0
    peak_per_unit = np.nanmax(
        np.where(occupied[:, None] & np.isfinite(spatial_mean), spatial_mean, -np.inf), axis=0
    )
    safe_peak = np.where(peak_per_unit > _EPS, peak_per_unit, np.inf)
    field_mask = (
        occupied[:, None]
        & np.isfinite(spatial_mean)
        & (spatial_mean >= field_threshold_fraction * safe_peak[None, :])
    ).astype(np.float64)

    global_heading_occupancy = np.bincount(heading_bins, minlength=num_heading_bins).astype(
        np.float64
    )

    infield_occupancy = field_mask.T @ combined_occupancy
    infield_activity = np.einsum("su,shu->uh", field_mask, combined_activity)
    infield_fired = np.einsum("su,shu->uh", field_mask, combined_fired)
    del combined_activity
    del combined_fired
    with np.errstate(divide="ignore", invalid="ignore"):
        infield_rates = np.where(
            infield_occupancy > 0.0, infield_activity / infield_occupancy, np.nan
        )
        infield_reliability = np.where(
            infield_occupancy > 0.0, infield_fired / infield_occupancy, np.nan
        )

    heading_centers = (
        (np.arange(num_heading_bins, dtype=np.float64) + 0.5) * _TWO_PI / num_heading_bins
    )
    heading_vectors = np.exp(1j * heading_centers)
    sampled_heading = infield_occupancy >= float(min_heading_occupancy)
    out = empty
    out.heading_bin_rates[:] = infield_rates.astype(np.float32, copy=False)
    out.heading_bin_reliability[:] = infield_reliability.astype(np.float32, copy=False)
    out.heading_bin_support[:] = sampled_heading.astype(np.float32, copy=False)
    out.heading_bin_occupancy[:] = global_heading_occupancy.astype(np.float32, copy=False)
    out.heading_bin_field_occupancy[:] = infield_occupancy.astype(np.float32, copy=False)

    for unit_index in range(num_units):
        valid_heading = sampled_heading[unit_index] & np.isfinite(infield_rates[unit_index])
        unit_curve = infield_rates[unit_index]
        valid_values = unit_curve[valid_heading]
        if valid_values.size == 0:
            continue
        total_value = float(valid_values.sum())
        peak_value = float(valid_values.max())
        mean_value = float(valid_values.mean())
        if total_value <= _EPS or peak_value <= _EPS or mean_value <= _EPS:
            continue
        vector = np.sum(unit_curve[valid_heading] * heading_vectors[valid_heading]) / total_value
        peak_bin = int(np.nanargmax(np.where(valid_heading, unit_curve, np.nan)))
        out.preferred_heading_rad[unit_index] = float(np.angle(vector))
        out.heading_vector_length[unit_index] = float(abs(vector))
        out.activation_corridor_deg[unit_index] = float(
            _circular_true_run_length(valid_heading & (unit_curve >= 0.5 * peak_value), peak_bin)
            * 360.0
            / num_heading_bins
        )
        out.peak_to_mean[unit_index] = float(peak_value / mean_value)
        out.heading_support_fraction[unit_index] = float(
            np.count_nonzero(valid_heading) / num_heading_bins
        )

        for spatial_bin in range(num_spatial_bins):
            local_support = combined_occupancy[spatial_bin] >= float(min_heading_occupancy)
            local_rates = local_heading_mean[spatial_bin, :, unit_index]
            valid_local = local_support & np.isfinite(local_rates)
            y_index, x_index = divmod(spatial_bin, num_bins_x)
            out.local_support_fraction[unit_index, y_index, x_index] = float(
                np.count_nonzero(valid_local) / num_heading_bins
            )
            if np.count_nonzero(valid_local) < 2 or float(local_rates[valid_local].sum()) <= _EPS:
                continue
            local_vector = np.sum(local_rates[valid_local] * heading_vectors[valid_local]) / float(
                local_rates[valid_local].sum()
            )
            out.local_preferred_heading[unit_index, y_index, x_index] = float(
                np.angle(local_vector)
            )
            out.local_selectivity[unit_index, y_index, x_index] = float(abs(local_vector))
    return out


def _empty_metrics(
    num_units: int, num_bins_y: int, num_bins_x: int, num_heading_bins: int
) -> HeadingRateMapOverlayMetrics:
    nan_units = np.full(num_units, np.nan, dtype=np.float32)
    return HeadingRateMapOverlayMetrics(
        heading_bin_rates=np.full((num_units, num_heading_bins), np.nan, dtype=np.float32),
        heading_bin_reliability=np.full((num_units, num_heading_bins), np.nan, dtype=np.float32),
        heading_bin_support=np.zeros((num_units, num_heading_bins), dtype=np.float32),
        heading_bin_occupancy=np.zeros(num_heading_bins, dtype=np.float32),
        heading_bin_field_occupancy=np.zeros((num_units, num_heading_bins), dtype=np.float32),
        preferred_heading_rad=nan_units.copy(),
        heading_vector_length=nan_units.copy(),
        activation_corridor_deg=nan_units.copy(),
        peak_to_mean=nan_units.copy(),
        heading_support_fraction=nan_units.copy(),
        local_preferred_heading=np.full(
            (num_units, num_bins_y, num_bins_x), np.nan, dtype=np.float32
        ),
        local_selectivity=np.zeros((num_units, num_bins_y, num_bins_x), dtype=np.float32),
        local_support_fraction=np.zeros((num_units, num_bins_y, num_bins_x), dtype=np.float32),
    )


def _render_pages(
    output_dir: Path,
    *,
    analysis_input: AnalysisInput,
    rate_maps: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    world_overlay,
    overlay: HeadingRateMapOverlayMetrics,
    unit_indices: np.ndarray,
    page_size: int,
    render_dpi: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    page_size = max(1, int(page_size))
    for page_index, start in enumerate(range(0, unit_indices.size, page_size)):
        page_units = unit_indices[start : start + page_size]
        figure, axes = plt.subplots(
            len(page_units), 3, figsize=(12.2, max(3.1, 3.0 * len(page_units))), squeeze=False
        )
        figure.suptitle(
            "Heading / rate-map overlay — "
            f"{analysis_input.source_name.replace('_', ' ').replace('.', ' ')} "
            f"({analysis_input.split_name})",
            fontsize=13,
        )
        for row, unit_index in enumerate(page_units.astype(int, copy=False)):
            _draw_unit_row(
                axes[row], unit_index, rate_maps[unit_index], bounds, world_overlay, overlay
            )
        figure.subplots_adjust(
            left=0.055, right=0.985, bottom=0.075, top=0.92, wspace=0.28, hspace=0.36
        )
        path = (
            output_dir / f"heading_rate_map_overlay__{analysis_input.source_name}"
            f"__{analysis_input.split_name}__{start}_{start + len(page_units)}.png"
        )
        save_figure(figure, path, dpi=render_dpi)
        plt.close(figure)
        figures[f"heading_rate_map_overlay_page_{page_index}"] = path
    return figures


def _draw_unit_row(
    axes: np.ndarray,
    unit_index: int,
    rate_map: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    world_overlay,
    overlay: HeadingRateMapOverlayMetrics,
) -> None:
    extent = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
    values = np.nan_to_num(rate_map, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = max(float(values.max()), 1e-6)
    norm = colors.PowerNorm(gamma=0.7, vmin=0.0, vmax=vmax)

    draw_unit_rate_map(axes[0], rate_map, bounds=bounds, world_overlay=world_overlay)
    _draw_contour(axes[0], values, bounds, color="#333333")
    axes[0].set_title(f"unit {unit_index} rate map", fontsize=9)

    axes[1].imshow(
        values, origin="lower", extent=extent, cmap="gray_r", norm=norm, interpolation="nearest"
    )
    _draw_contour(axes[1], values, bounds, color="#E24A33")
    heading_caption = (
        f"corridor {_fmt_deg(overlay.activation_corridor_deg[unit_index])}"
        f"   ·   peak/mean {_fmt(overlay.peak_to_mean[unit_index])}×"
        f"   ·   R {_fmt(overlay.heading_vector_length[unit_index])}"
        f"   ·   support {_fmt_pct(overlay.heading_support_fraction[unit_index])}"
    )
    _style_spatial_axis(axes[1], bounds, world_overlay, x_label=heading_caption, y_label="")
    axes[1].set_title("rate map + local heading", fontsize=9)

    local_heading = overlay.local_preferred_heading[unit_index]
    local_selectivity = overlay.local_selectivity[unit_index]
    local_support = overlay.local_support_fraction[unit_index]
    y_count, x_count = local_heading.shape
    xs = (
        np.linspace(bounds[0][0], bounds[0][1], x_count, endpoint=False)
        + (bounds[0][1] - bounds[0][0]) / max(1, x_count) * 0.5
    )
    ys = (
        np.linspace(bounds[1][0], bounds[1][1], y_count, endpoint=False)
        + (bounds[1][1] - bounds[1][0]) / max(1, y_count) * 0.5
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    arrow_mask = np.isfinite(local_heading) & (local_selectivity > 0.05) & (local_support >= 0.25)
    if np.any(arrow_mask):
        axes[1].quiver(
            grid_x[arrow_mask],
            grid_y[arrow_mask],
            np.cos(local_heading[arrow_mask]) * local_selectivity[arrow_mask],
            np.sin(local_heading[arrow_mask]) * local_selectivity[arrow_mask],
            local_selectivity[arrow_mask],
            cmap="viridis",
            clim=(0.0, 1.0),
            angles="xy",
            scale_units="xy",
            scale=7.0,
            width=0.006,
            headwidth=4.0,
            headlength=5.0,
            zorder=4,
        )
    _draw_rose(
        inset_axes(
            axes[1],
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.0, 0.66, 0.28, 0.28),
            bbox_transform=axes[1].transAxes,
            borderpad=0.0,
            axes_class=plt.PolarAxes,
        ),
        unit_index,
        overlay,
        compact=True,
    )
    _draw_curve(axes[2], unit_index, overlay)


def _draw_curve(axis: plt.Axes, unit_index: int, overlay: HeadingRateMapOverlayMetrics) -> None:
    normalized_rates, valid = _normalized_heading_rates(overlay, unit_index)
    degrees = (
        (np.arange(normalized_rates.size, dtype=np.float64) + 0.5) * 360.0 / normalized_rates.size
    )

    occupancy = np.asarray(overlay.heading_bin_field_occupancy[unit_index], dtype=np.float64)
    occupancy_axis = axis.twinx()
    occupancy_axis.fill_between(
        degrees, 0.0, occupancy, step="mid", color="#C9C9C9", alpha=0.5, linewidth=0.0, zorder=0
    )
    occupancy_axis.set_ylim(0.0, max(float(occupancy.max()) * 1.15, 1.0))
    occupancy_axis.set_ylabel("in-field samples / bin", fontsize=7, color="#9A9A9A")
    occupancy_axis.tick_params(axis="y", labelsize=6, colors="#9A9A9A", length=2.0)
    occupancy_axis.spines["top"].set_visible(False)
    occupancy_axis.spines["right"].set_color("#C9C9C9")

    axis.plot(degrees[valid], normalized_rates[valid], color="#2B6CB0", linewidth=1.9, zorder=3)
    axis.fill_between(
        degrees,
        0.0,
        normalized_rates,
        where=valid & (normalized_rates >= 0.5),
        color="#2B6CB0",
        alpha=0.22,
        zorder=2,
    )
    axis.axhline(0.5, color="#666666", linewidth=0.8, linestyle=":", zorder=1)
    if _has_preferred_heading(overlay, unit_index):
        axis.axvline(
            (float(np.degrees(overlay.preferred_heading_rad[unit_index])) + 360.0) % 360.0,
            color="#F46D43",
            linewidth=1.8,
            zorder=3,
        )
    axis.set(
        xlim=(0, 360),
        ylim=(0, 1.05),
        title="in-field heading rate",
        xlabel="heading angle (deg)",
        ylabel="normalized rate",
    )
    axis.tick_params(labelsize=7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_zorder(occupancy_axis.get_zorder() + 1)
    axis.patch.set_visible(False)


def _draw_rose(
    axis: plt.Axes, unit_index: int, overlay: HeadingRateMapOverlayMetrics, *, compact: bool
) -> None:
    normalized_rates, valid = _normalized_heading_rates(overlay, unit_index)
    centers = (
        (np.arange(normalized_rates.size, dtype=np.float64) + 0.5) * _TWO_PI / normalized_rates.size
    )
    colors_by_bin = [
        "#2b8cbe" if valid[i] and normalized_rates[i] >= 0.5 else "#d8e8f2"
        for i in range(normalized_rates.size)
    ]
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.bar(
        centers[valid],
        normalized_rates[valid],
        width=_TWO_PI / normalized_rates.size * 0.9,
        color=[colors_by_bin[i] for i in np.flatnonzero(valid)],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.95,
    )
    if _has_preferred_heading(overlay, unit_index):
        preferred = overlay.preferred_heading_rad[unit_index]
        axis.plot([preferred, preferred], [0.0, 1.05], color="#f46d43", linewidth=1.8)
    axis.set_ylim(0.0, 1.05)
    axis.set_yticks([] if compact else [0.5, 1.0])
    axis.set_xticks([] if compact else np.deg2rad([0, 90, 180, 270]))
    axis.grid(color="#999999", alpha=0.25)


def _style_spatial_axis(
    axis: plt.Axes,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    world_overlay,
    *,
    x_label: str,
    y_label: str,
) -> None:
    finalize_arena_axis(axis, x_bounds=bounds[0], y_bounds=bounds[1], world_overlay=world_overlay)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlabel(x_label, fontsize=8)
    axis.set_ylabel(y_label, fontsize=8)
    axis.tick_params(labelsize=7)
    if world_overlay is not None:
        draw_world_segments_on_axis(
            axis, world_overlay.segments, line_color="#d9d9d9", line_width=0.8
        )
        draw_landmarks_on_axis(axis, world_overlay, marker_size=10.0)


def _draw_contour(
    axis: plt.Axes,
    values: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    color: str,
) -> None:
    peak = float(values.max()) if values.size else 0.0
    if peak <= _EPS:
        return
    xs = (
        np.linspace(bounds[0][0], bounds[0][1], values.shape[1], endpoint=False)
        + (bounds[0][1] - bounds[0][0]) / max(1, values.shape[1]) * 0.5
    )
    ys = (
        np.linspace(bounds[1][0], bounds[1][1], values.shape[0], endpoint=False)
        + (bounds[1][1] - bounds[1][0]) / max(1, values.shape[0]) * 0.5
    )
    axis.contour(xs, ys, values, levels=[0.5 * peak], colors=color, linewidths=1.0, zorder=3)


def _rank_units(rate_maps: np.ndarray, activation_corridor_deg: np.ndarray) -> np.ndarray:
    peaks = (
        np.nan_to_num(rate_maps, nan=0.0, posinf=0.0, neginf=0.0)
        .reshape(rate_maps.shape[0], -1)
        .max(axis=1)
    )
    return np.argsort(peaks * np.isfinite(activation_corridor_deg), kind="stable")[::-1].astype(
        np.int64
    )


def _has_preferred_heading(overlay: HeadingRateMapOverlayMetrics, unit_index: int) -> bool:
    vector_length = float(overlay.heading_vector_length[unit_index])
    return bool(np.isfinite(vector_length) and vector_length >= 0.05)


def _normalized_heading_rates(
    overlay: HeadingRateMapOverlayMetrics,
    unit_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.nan_to_num(overlay.heading_bin_rates[unit_index], nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(overlay.heading_bin_rates[unit_index]) & (
        overlay.heading_bin_support[unit_index] > 0.0
    )
    normalized = np.zeros_like(rates, dtype=np.float64)
    if np.any(valid):
        peak = float(rates[valid].max())
        if peak > _EPS:
            normalized = rates.astype(np.float64, copy=False) / peak
    return normalized, valid


def _circular_true_run_length(mask: np.ndarray, index: int) -> int:
    if not bool(mask[index]):
        return 0
    length = 1
    for direction in (-1, 1):
        cursor = (index + direction) % mask.size
        while cursor != index and bool(mask[cursor]):
            length += 1
            cursor = (cursor + direction) % mask.size
    return min(length, int(mask.size))


def _write_table(
    output_dir: Path, analysis_input: AnalysisInput, overlay: HeadingRateMapOverlayMetrics
) -> Path:
    rows = [
        [
            unit_index,
            float(overlay.activation_corridor_deg[unit_index]),
            float(overlay.peak_to_mean[unit_index]),
            float(overlay.heading_vector_length[unit_index]),
            float(overlay.heading_support_fraction[unit_index]),
            float(overlay.preferred_heading_rad[unit_index]),
        ]
        for unit_index in range(overlay.activation_corridor_deg.shape[0])
    ]
    return write_csv(
        output_dir / f"heading_rate_map_overlay_per_unit__{analysis_input.source_name}"
        f"__{analysis_input.split_name}.csv",
        [
            "unit_index",
            "activation_corridor_deg",
            "heading_peak_to_mean",
            "heading_vector_length",
            "heading_support_fraction",
            "preferred_heading_rad",
        ],
        rows,
    )


def _fmt(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{float(value):.2f}"


def _fmt_deg(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{float(value):.0f} deg"


def _fmt_pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * float(value):.0f}%"
