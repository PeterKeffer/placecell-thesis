"""Drawing rate maps: colour scales, arena context, and the figures built from them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.image import AxesImage
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.ndimage import binary_dilation, gaussian_filter

from ..figure_style import despine
from .world_overlay import (
    POSITION_X_LABEL,
    POSITION_Y_LABEL,
    WorldOverlay,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    finalize_arena_axis,
)

_DIVERGING_COLORMAP = "RdBu_r"
_SIGNED_NEGATIVE_FRACTION = 0.02


def _sequential_colormap_name(colormap_mode: str) -> str:
    if colormap_mode == "reds":
        return "Reds"
    if colormap_mode in {"inferno", "turbo", "viridis", "magma", "cividis", "coolwarm", "plasma"}:
        return colormap_mode
    return "Reds"


def _population_is_signed(
    values: np.ndarray, *, negative_fraction: float = _SIGNED_NEGATIVE_FRACTION
) -> bool:
    """Decide once, from the whole population, whether maps should use a diverging map."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False
    absolute_limit = float(np.max(np.abs(finite_values)))
    if absolute_limit <= 1e-9:
        return False
    return float(np.min(finite_values)) < -negative_fraction * absolute_limit


def _style_for_family(
    values: np.ndarray, *, is_signed: bool, colormap_mode: str
) -> tuple[str, colors.Normalize]:
    """Colormap + norm for the chosen family."""
    return _style_for_limit(
        _family_limit(values, is_signed=is_signed),
        is_signed=is_signed,
        colormap_mode=colormap_mode,
    )


def _family_limit(values: np.ndarray, *, is_signed: bool) -> float:
    """The peak the family's norm is scaled to: largest magnitude, or largest value."""
    finite_values = values[np.isfinite(values)]
    if is_signed:
        return max(float(np.max(np.abs(finite_values))) if finite_values.size else 1.0, 1e-6)
    return max(float(np.max(finite_values)) if finite_values.size else 1.0, 1e-6)


def _style_for_limit(
    limit: float, *, is_signed: bool, colormap_mode: str
) -> tuple[str, colors.Normalize]:
    """Colormap + norm for a family already reduced to its peak."""
    if is_signed:
        return _DIVERGING_COLORMAP, colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return _sequential_colormap_name(colormap_mode), colors.PowerNorm(
        gamma=0.6, vmin=0.0, vmax=limit
    )


def _panel_metric_cmap_name(*, panel_metric_name: str) -> str:
    if panel_metric_name in {"thresholded_reliability", "quantile_thresholded_reliability"}:
        return "viridis"
    if panel_metric_name == "split_half_agreement":
        return "plasma"
    return "turbo"


def _panel_metric_label(metric_name: str) -> tuple[str, str, str]:
    """Return (combined-panel title, reliability-panel title, caption label)."""
    if metric_name == "split_half_agreement":
        return ("Activity × agreement", "Split-half agreement", "agr")
    if metric_name == "bin_consistency":
        return ("Activity × consistency", "Bin consistency", "con")
    return ("Activity × reliability", "Reliability", "rel")


def _normalized_activity_alpha(rate_map: np.ndarray) -> np.ndarray:
    finite_magnitude = np.abs(np.nan_to_num(rate_map, nan=0.0)).astype(np.float32, copy=False)
    maximum_magnitude = float(np.max(finite_magnitude)) if finite_magnitude.size else 0.0
    if maximum_magnitude <= 1e-8:
        return np.zeros_like(finite_magnitude, dtype=np.float32)
    return np.clip(finite_magnitude / maximum_magnitude, 0.0, 1.0).astype(np.float32, copy=False)


def _combined_rgba_with_style(
    rate_map: np.ndarray,
    panel_metric_map: np.ndarray,
    *,
    panel_metric_name: str,
    activity_cmap_name: str,
    activity_norm: colors.Normalize,
    metric_cmap_name: str,
    metric_norm: colors.Normalize,
) -> np.ndarray:
    if panel_metric_name in {"thresholded_reliability", "quantile_thresholded_reliability"}:
        rgba = plt.get_cmap(activity_cmap_name)(activity_norm(np.nan_to_num(rate_map, nan=0.0)))
        rgba[..., 3] = np.clip(np.nan_to_num(panel_metric_map, nan=0.0), 0.0, 1.0)
        rgba[..., 3] *= np.isfinite(rate_map).astype(np.float32, copy=False)
        return rgba
    rgba = plt.get_cmap(metric_cmap_name)(metric_norm(np.nan_to_num(panel_metric_map, nan=0.0)))
    rgba[..., 3] = _normalized_activity_alpha(rate_map)
    rgba[..., 3] *= np.isfinite(panel_metric_map).astype(np.float32, copy=False)
    return rgba


def _fill_panel_metric_from_neighbors(
    panel_metric_map: np.ndarray,
    *,
    field_mask: np.ndarray,
    fill_sigma_bins: float,
) -> np.ndarray:
    """Optionally fill unsupported field-edge bins from neighboring supported bins."""
    if fill_sigma_bins <= 0.0:
        return panel_metric_map
    support_mask = np.isfinite(panel_metric_map)
    if not np.any(support_mask):
        return panel_metric_map
    candidate_mask = (~support_mask) & binary_dilation(
        field_mask.astype(bool, copy=False), iterations=1
    )
    if not np.any(candidate_mask):
        return panel_metric_map
    support_weights = support_mask.astype(np.float32, copy=False)
    supported_values = (
        np.nan_to_num(panel_metric_map, nan=0.0).astype(np.float32, copy=False) * support_weights
    )
    smoothed_value_sum = gaussian_filter(supported_values, sigma=fill_sigma_bins)
    smoothed_support = gaussian_filter(support_weights, sigma=fill_sigma_bins)
    local_mean = np.divide(
        smoothed_value_sum,
        smoothed_support,
        out=np.zeros_like(panel_metric_map, dtype=np.float32),
        where=smoothed_support > 1e-6,
    )
    filled_map = panel_metric_map.copy()
    fill_mask = candidate_mask & (smoothed_support > 1e-6)
    filled_map[fill_mask] = local_mean[fill_mask]
    return filled_map


def support_map_style(support_counts: np.ndarray) -> tuple[str, colors.Normalize]:
    positive_counts = support_counts[support_counts > 0]
    if positive_counts.size == 0:
        return "YlOrRd", colors.Normalize(vmin=0.0, vmax=1.0)
    return (
        "YlOrRd",
        colors.PowerNorm(
            gamma=0.55,
            vmin=float(positive_counts.min()),
            vmax=float(positive_counts.max()),
        ),
    )


def _colorbar_ticks_for_norm(norm: colors.Normalize) -> list[float]:
    if isinstance(norm, colors.TwoSlopeNorm):
        return [float(norm.vmin), 0.0, float(norm.vmax)]
    return [float(norm.vmin), float(norm.vmax)]


def _bounds_to_extent(
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float, float, float]:
    x_bounds, y_bounds = bounds
    return (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])


def _apply_world_context(
    axis: plt.Axes,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    world_overlay: WorldOverlay | None,
    show_x_tick_labels: bool,
    show_y_tick_labels: bool,
    facecolor: str = "#ECECEC",
) -> None:
    x_bounds, y_bounds = bounds
    finalize_arena_axis(
        axis,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        world_overlay=world_overlay,
    )
    axis.set_facecolor(facecolor)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    axis.grid(False)
    despine(axis)
    axis.tick_params(
        labelsize=6.5,
        pad=1.5,
        labelbottom=show_x_tick_labels,
        labelleft=show_y_tick_labels,
        labeltop=False,
        labelright=False,
    )
    axis.set_xlabel(POSITION_X_LABEL if show_x_tick_labels else "", fontsize=7.0)
    axis.set_ylabel(POSITION_Y_LABEL if show_y_tick_labels else "", fontsize=7.0)
    if world_overlay is None:
        return
    draw_world_segments_on_axis(
        axis,
        world_overlay.segments,
        line_color="#505050",
        line_width=1.0,
        alpha=0.9,
    )
    draw_landmarks_on_axis(
        axis,
        world_overlay,
        marker_size=16.0,
        edge_line_width=0.3,
        alpha=0.88,
    )


def _apply_heatmap_grid_context(
    axis: plt.Axes,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    world_overlay: WorldOverlay | None,
    facecolor: str = "#ECECEC",
) -> None:
    x_bounds, y_bounds = bounds
    finalize_arena_axis(
        axis,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        world_overlay=world_overlay,
    )
    axis.set_facecolor(facecolor)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.set_xlabel("")
    axis.set_ylabel("")
    for spine in axis.spines.values():
        spine.set_visible(False)
    if world_overlay is None:
        return
    draw_world_segments_on_axis(
        axis,
        world_overlay.segments,
        line_color="#505050",
        line_width=0.9,
        alpha=0.9,
    )
    draw_landmarks_on_axis(
        axis,
        world_overlay,
        marker_size=12.0,
        edge_line_width=0.25,
        alpha=0.84,
    )


def draw_unit_rate_map(
    axis: plt.Axes,
    rate_map: np.ndarray,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    world_overlay: WorldOverlay | None,
    signed: bool = False,
    draw_colorbar: bool = False,
) -> AxesImage:
    """Draw one per-unit rate map on axis and return the image."""
    values = np.nan_to_num(rate_map, nan=0.0, posinf=0.0, neginf=0.0)
    cmap_name, norm = _style_for_family(values, is_signed=signed, colormap_mode="reds")
    image = axis.imshow(
        values,
        origin="lower",
        extent=_bounds_to_extent(bounds),
        cmap=cmap_name,
        norm=norm,
        interpolation="nearest",
    )
    _apply_heatmap_grid_context(axis, bounds=bounds, world_overlay=world_overlay)
    if draw_colorbar:
        _attach_gutter_colorbar(axis.figure, axis, mappable=image, norm=norm, label="activity")
    return image


def _format_metric_value(label: str, value: float, *, precision: int = 2) -> str:
    if not np.isfinite(value):
        return f"{label} n/a"
    return f"{label} {value:.{precision}f}"


def _format_mean_skaggs_title(mean_spatial_information_bits: float) -> str:
    if not np.isfinite(mean_spatial_information_bits):
        return "spatial information unavailable for signed rate maps"
    return f"mean spatial information {mean_spatial_information_bits:.2f} bits"


def _choose_grid_shape(num_items: int, target_aspect_ratio: float = 16.0 / 9.0) -> tuple[int, int]:
    if num_items <= 0:
        return 0, 0
    best_rows = 1
    best_columns = num_items
    best_score = float("inf")
    for rows in range(1, num_items + 1):
        columns = int(np.ceil(num_items / rows))
        aspect_score = abs((columns / rows) - target_aspect_ratio)
        waste_score = (rows * columns - num_items) * 0.05
        score = aspect_score + waste_score
        if score < best_score:
            best_rows = rows
            best_columns = columns
            best_score = score
    return best_rows, best_columns


_AGG_SAFE_PIXELS = 64000


def agg_safe_dpi(figure: plt.Figure, requested_dpi: int) -> int:
    longest_axis_inches = float(max(figure.get_size_inches()))
    if longest_axis_inches <= 0.0:
        return requested_dpi
    return max(1, min(requested_dpi, int(_AGG_SAFE_PIXELS / longest_axis_inches)))


def _grid_metric_caption(
    *,
    unit_index: int,
    spatial_information_bits: np.ndarray,
    mean_panel_metric_inside_fields: np.ndarray,
    metric_short_label: str,
) -> str:
    """One-line metric caption shown below a grid cell (option B)."""
    panel_metric = _format_metric_value(
        metric_short_label, float(mean_panel_metric_inside_fields[unit_index])
    )
    return (
        f"{_format_metric_value('SI', float(spatial_information_bits[unit_index]))}"
        f"   ·   "
        f"{panel_metric}"
    )


def _panel_metric_caption(
    *,
    unit_index: int,
    spatial_information_bits: np.ndarray,
    mean_panel_metric_inside_fields: np.ndarray,
    metric_short_label: str,
    panel_metric_supported_field_fraction: np.ndarray,
    split_half_correlations: np.ndarray,
    episode_rate_map_correlations: np.ndarray,
    field_counts: np.ndarray,
) -> str:
    """One-line per-unit metric caption shown below the panel rate map (option B)."""
    panel_metric = _format_metric_value(
        metric_short_label, float(mean_panel_metric_inside_fields[unit_index])
    )
    supported_fraction = _format_metric_value(
        "supp", float(panel_metric_supported_field_fraction[unit_index])
    )
    return (
        f"u{int(unit_index):03d}   ·   "
        f"{_format_metric_value('SI', float(spatial_information_bits[unit_index]))}   ·   "
        f"{panel_metric}   ·   "
        f"{supported_fraction}   ·   "
        f"split {float(split_half_correlations[unit_index]):.2f}   ·   "
        f"ep {float(episode_rate_map_correlations[unit_index]):.2f}   ·   "
        f"{_format_metric_value('fields', float(field_counts[unit_index]), precision=0)}"
    )


def _attach_gutter_colorbar(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    mappable,
    norm: colors.Normalize,
    tick_fontsize: float = 6.0,
    label: str | None = None,
) -> None:
    """Attach a slim colorbar in a reserved gutter beside axis."""
    colorbar = figure.colorbar(mappable, ax=axis, fraction=0.046, pad=0.02)
    colorbar.set_ticks(_colorbar_ticks_for_norm(norm))
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    colorbar.ax.tick_params(labelsize=tick_fontsize, length=2.0, pad=0.8)
    colorbar.outline.set_linewidth(0.5)
    if label:
        colorbar.set_label(label, fontsize=7.0)


def empty_rate_map_figure(message: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return figure


def build_summary_panel_figure(
    *,
    source_name: str,
    split_name: str,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    rate_maps: np.ndarray,
    panel_metric_maps: np.ndarray,
    ranked_indices: np.ndarray,
    field_masks: np.ndarray,
    spatial_information_bits: np.ndarray,
    mean_panel_metric_inside_fields: np.ndarray,
    panel_metric_supported_field_fraction: np.ndarray,
    split_half_correlations: np.ndarray,
    episode_rate_map_correlations: np.ndarray,
    field_counts: np.ndarray,
    world_overlay: WorldOverlay | None,
    mean_spatial_information_bits: float,
    shared_color_scale: bool,
    colormap_mode: str,
    panel_metric_name: str,
    panel_metric_fill_sigma_bins: float,
    title_prefix: str,
    show_captions: bool,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Build the 3-column per-unit summary panel (rate / combined / reliability)."""
    population_is_signed = _population_is_signed(rate_maps[ranked_indices])
    common_cmap_name = None
    common_norm = None
    if shared_color_scale:
        common_cmap_name, common_norm = _style_for_family(
            rate_maps[ranked_indices],
            is_signed=population_is_signed,
            colormap_mode=colormap_mode,
        )

    figure, axes = plt.subplots(
        len(ranked_indices),
        3,
        figsize=(12.15, max(3.4, 3.1 * len(ranked_indices))),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        f"{source_name.replace('_', ' ').replace('.', ' ')} ({split_name})\n"
        f"{_format_mean_skaggs_title(mean_spatial_information_bits)}",
        fontsize=12,
    )
    combined_title, reliability_title, metric_short_label = _panel_metric_label(panel_metric_name)
    axes[0, 0].set_title("Rate map", fontsize=10)
    axes[0, 1].set_title(combined_title, fontsize=10)
    axes[0, 2].set_title(reliability_title, fontsize=10)

    extent = _bounds_to_extent(bounds)
    reliability_cmap_name = _panel_metric_cmap_name(panel_metric_name=panel_metric_name)
    reliability_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    combined_uses_activity_colors = panel_metric_name in {
        "thresholded_reliability",
        "quantile_thresholded_reliability",
    }

    rate_axes: list[plt.Axes] = []
    for row_index, unit_index in enumerate(ranked_indices):
        rate_map = rate_maps[unit_index]
        raw_panel_metric_map = panel_metric_maps[unit_index]
        displayed_panel_metric_map = (
            _fill_panel_metric_from_neighbors(
                raw_panel_metric_map,
                field_mask=field_masks[unit_index],
                fill_sigma_bins=panel_metric_fill_sigma_bins,
            )
            if panel_metric_name
            not in {"thresholded_reliability", "quantile_thresholded_reliability"}
            else raw_panel_metric_map
        )
        if shared_color_scale:
            unit_cmap_name = str(common_cmap_name)
            unit_norm = common_norm
        else:
            unit_cmap_name, unit_norm = _style_for_family(
                rate_map, is_signed=population_is_signed, colormap_mode=colormap_mode
            )

        rate_axis = axes[row_index, 0]
        combined_axis = axes[row_index, 1]
        reliability_axis = axes[row_index, 2]

        rate_image = rate_axis.imshow(
            np.ma.masked_invalid(rate_map),
            origin="lower",
            cmap=unit_cmap_name,
            norm=unit_norm,
            interpolation="nearest",
            extent=extent,
        )
        combined_axis.imshow(
            _combined_rgba_with_style(
                rate_map,
                displayed_panel_metric_map,
                panel_metric_name=panel_metric_name,
                activity_cmap_name=unit_cmap_name,
                activity_norm=unit_norm,
                metric_cmap_name=reliability_cmap_name,
                metric_norm=reliability_norm,
            ),
            origin="lower",
            interpolation="nearest",
            extent=extent,
        )
        reliability_axis.imshow(
            np.ma.masked_invalid(displayed_panel_metric_map),
            origin="lower",
            cmap=reliability_cmap_name,
            norm=reliability_norm,
            interpolation="nearest",
            extent=extent,
        )

        show_x_tick_labels = row_index == len(ranked_indices) - 1
        _apply_world_context(
            rate_axis,
            bounds=bounds,
            world_overlay=world_overlay,
            show_x_tick_labels=show_x_tick_labels,
            show_y_tick_labels=True,
        )
        _apply_world_context(
            combined_axis,
            bounds=bounds,
            world_overlay=world_overlay,
            show_x_tick_labels=show_x_tick_labels,
            show_y_tick_labels=False,
        )
        _apply_world_context(
            reliability_axis,
            bounds=bounds,
            world_overlay=world_overlay,
            show_x_tick_labels=show_x_tick_labels,
            show_y_tick_labels=False,
        )

        if show_captions:
            rate_axis.set_xlabel(
                _panel_metric_caption(
                    unit_index=int(unit_index),
                    spatial_information_bits=spatial_information_bits,
                    mean_panel_metric_inside_fields=mean_panel_metric_inside_fields,
                    metric_short_label=metric_short_label,
                    panel_metric_supported_field_fraction=panel_metric_supported_field_fraction,
                    split_half_correlations=split_half_correlations,
                    episode_rate_map_correlations=episode_rate_map_correlations,
                    field_counts=field_counts,
                ),
                fontsize=8.0,
                labelpad=2.5,
            )

        _attach_gutter_colorbar(
            figure, rate_axis, mappable=rate_image, norm=unit_norm, label="activity"
        )
        if combined_uses_activity_colors:
            _attach_gutter_colorbar(
                figure,
                combined_axis,
                mappable=plt.cm.ScalarMappable(norm=unit_norm, cmap=unit_cmap_name),
                norm=unit_norm,
                label="activity",
            )
        else:
            _attach_gutter_colorbar(
                figure,
                combined_axis,
                mappable=plt.cm.ScalarMappable(norm=reliability_norm, cmap=reliability_cmap_name),
                norm=reliability_norm,
                label=reliability_title.lower(),
            )
        _attach_gutter_colorbar(
            figure,
            reliability_axis,
            mappable=plt.cm.ScalarMappable(norm=reliability_norm, cmap=reliability_cmap_name),
            norm=reliability_norm,
            label=reliability_title.lower(),
        )
        rate_axes.append(rate_axis)

    return figure, rate_axes


def build_rate_map_grid_figure(
    *,
    source_name: str,
    split_name: str,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    rate_maps: np.ndarray,
    ranked_indices: np.ndarray,
    spatial_information_bits: np.ndarray,
    mean_panel_metric_inside_fields: np.ndarray,
    world_overlay: WorldOverlay | None,
    mean_spatial_information_bits: float,
    shared_color_scale: bool,
    colormap_mode: str,
    panel_metric_name: str,
    title_prefix: str,
    show_captions: bool,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Build the scan-friendly rate-map grid."""
    rows, columns = _choose_grid_shape(len(ranked_indices))
    population_is_signed = _population_is_signed(rate_maps[ranked_indices])
    common_cmap_name = None
    common_norm = None
    if shared_color_scale:
        common_cmap_name, common_norm = _style_for_family(
            rate_maps[ranked_indices],
            is_signed=population_is_signed,
            colormap_mode=colormap_mode,
        )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.4, rows * 2.4),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        f"{source_name} {title_prefix} ({split_name})\n"
        f"{_format_mean_skaggs_title(mean_spatial_information_bits)}",
        fontsize=15,
    )
    _, _, metric_short_label = _panel_metric_label(panel_metric_name)

    activity_image = None
    extent = _bounds_to_extent(bounds)
    heatmap_axes: list[plt.Axes] = []
    for flat_index, unit_index in enumerate(ranked_indices):
        row_index = flat_index // columns
        column_index = flat_index % columns
        axis = axes[row_index, column_index]
        rate_map = rate_maps[unit_index]
        if shared_color_scale:
            unit_cmap_name = str(common_cmap_name)
            unit_norm = common_norm
        else:
            unit_cmap_name, unit_norm = _style_for_family(
                rate_map, is_signed=population_is_signed, colormap_mode=colormap_mode
            )
        rate_image = axis.imshow(
            np.ma.masked_invalid(rate_map),
            origin="lower",
            cmap=unit_cmap_name,
            norm=unit_norm,
            interpolation="nearest",
            extent=extent,
        )
        activity_image = rate_image
        _apply_heatmap_grid_context(axis, bounds=bounds, world_overlay=world_overlay)
        axis.set_title(f"Unit {int(unit_index)}", fontsize=9, pad=3.0)
        if show_captions:
            axis.set_xlabel(
                _grid_metric_caption(
                    unit_index=int(unit_index),
                    spatial_information_bits=spatial_information_bits,
                    mean_panel_metric_inside_fields=mean_panel_metric_inside_fields,
                    metric_short_label=metric_short_label,
                ),
                fontsize=7.5,
                labelpad=2.0,
            )
        if not shared_color_scale:
            _attach_gutter_colorbar(
                figure, axis, mappable=rate_image, norm=unit_norm, label="activity"
            )
        heatmap_axes.append(axis)

    for flat_index in range(len(ranked_indices), rows * columns):
        row_index = flat_index // columns
        column_index = flat_index % columns
        axes[row_index, column_index].axis("off")

    if (
        shared_color_scale
        and common_cmap_name is not None
        and common_norm is not None
        and activity_image is not None
    ):
        activity_colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=common_norm, cmap=common_cmap_name),
            ax=axes,
            shrink=0.72,
            pad=0.02,
            location="bottom",
        )
        activity_colorbar.set_label("activity", fontsize=9)
        activity_colorbar.ax.tick_params(labelsize=8)

    return figure, heatmap_axes
