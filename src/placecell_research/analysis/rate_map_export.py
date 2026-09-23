"""Writing rate-map figures and tables to disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.work_blocks import index_blocks
from .base import AnalysisInput
from .helpers import write_csv
from .parallel import render_block_count, render_pool
from .rate_map_computation import (
    GridOutcome,
    PanelFamilyOutcome,
    _PanelSettings,
    _RankedUnits,
)
from .rate_map_metrics import (
    RateMapMetricBundle,
    nanmean_or_nan,
)
from .rate_map_rendering import (
    _apply_world_context,
    _attach_gutter_colorbar,
    _bounds_to_extent,
    _family_limit,
    _population_is_signed,
    _style_for_limit,
    agg_safe_dpi,
    build_rate_map_grid_figure,
    build_summary_panel_figure,
    empty_rate_map_figure,
    support_map_style,
)
from .world_overlay import (
    POSITION_X_LABEL,
    WorldOverlay,
    finalize_arena_axis,
)


def render_rate_map_grid_pages(
    *,
    output_dir: Path,
    file_prefix: str,
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
    render_dpi: int,
    page_size: int,
) -> list[Path]:
    """Render paged rate-map grids with the same layout used by analysis modules."""
    output_dir.mkdir(parents=True, exist_ok=True)
    page_size = max(1, int(page_size))
    ranked_indices = np.asarray(ranked_indices, dtype=np.int64).reshape(-1)
    if ranked_indices.size == 0:
        empty_path = output_dir / f"rate_map_grid__{file_prefix}__0_0.png"
        _render_rate_map_grid(
            empty_path,
            source_name=source_name,
            split_name=split_name,
            bounds=bounds,
            rate_maps=rate_maps,
            ranked_indices=ranked_indices,
            spatial_information_bits=spatial_information_bits,
            mean_panel_metric_inside_fields=mean_panel_metric_inside_fields,
            world_overlay=world_overlay,
            mean_spatial_information_bits=mean_spatial_information_bits,
            shared_color_scale=shared_color_scale,
            colormap_mode=colormap_mode,
            panel_metric_name=panel_metric_name,
            title_prefix=title_prefix,
            render_dpi=render_dpi,
        )
        return [empty_path]

    pages: list[Path] = []
    for start_index in range(0, ranked_indices.size, page_size):
        end_index = min(start_index + page_size, ranked_indices.size)
        path = output_dir / f"rate_map_grid__{file_prefix}__{start_index}_{end_index}.png"
        _render_rate_map_grid(
            path,
            source_name=source_name,
            split_name=split_name,
            bounds=bounds,
            rate_maps=rate_maps,
            ranked_indices=ranked_indices[start_index:end_index],
            spatial_information_bits=spatial_information_bits,
            mean_panel_metric_inside_fields=mean_panel_metric_inside_fields,
            world_overlay=world_overlay,
            mean_spatial_information_bits=mean_spatial_information_bits,
            shared_color_scale=shared_color_scale,
            colormap_mode=colormap_mode,
            panel_metric_name=panel_metric_name,
            title_prefix=title_prefix,
            render_dpi=render_dpi,
        )
        pages.append(path)
    return pages

def _rate_map_render_dpi(*, render_all_units: bool) -> int:
    if render_all_units:
        return 72
    return 170

def _rate_map_page_ranges(num_items: int, *, page_size: int) -> list[tuple[int, int]]:
    if num_items <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    for start_index in range(0, num_items, page_size):
        end_index = min(start_index + page_size, num_items)
        ranges.append((start_index, end_index))
    return ranges


def _rate_map_page_path(base_path: Path, *, start_index: int, end_index: int) -> Path:
    return base_path.with_name(f"{base_path.stem}__{start_index}_{end_index}{base_path.suffix}")

def _render_summary_panel(
    path: Path,
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
    spike_positions_by_unit: dict[int, np.ndarray],
    world_overlay: WorldOverlay | None,
    use_absolute_activations: bool,
    mean_spatial_information_bits: float,
    shared_color_scale: bool,
    colormap_mode: str,
    panel_metric_name: str,
    panel_metric_fill_sigma_bins: float,
    show_spike_positions: bool,
    title_prefix: str,
    render_dpi: int,
    clean_path: Path | None = None,
) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if clean_path is not None:
        clean_path.parent.mkdir(parents=True, exist_ok=True)
    if len(ranked_indices) == 0:
        figure = empty_rate_map_figure(
            f"No units available for {source_name} {title_prefix} ({split_name})"
        )
        if clean_path is not None:
            figure.savefig(clean_path, dpi=render_dpi, bbox_inches="tight")
        figure.savefig(path, dpi=render_dpi, bbox_inches="tight")
        plt.close(figure)
        return path, clean_path

    figure, rate_axes = build_summary_panel_figure(
        source_name=source_name,
        split_name=split_name,
        bounds=bounds,
        rate_maps=rate_maps,
        panel_metric_maps=panel_metric_maps,
        ranked_indices=ranked_indices,
        field_masks=field_masks,
        spatial_information_bits=spatial_information_bits,
        mean_panel_metric_inside_fields=mean_panel_metric_inside_fields,
        panel_metric_supported_field_fraction=panel_metric_supported_field_fraction,
        split_half_correlations=split_half_correlations,
        episode_rate_map_correlations=episode_rate_map_correlations,
        field_counts=field_counts,
        world_overlay=world_overlay,
        mean_spatial_information_bits=mean_spatial_information_bits,
        shared_color_scale=shared_color_scale,
        colormap_mode=colormap_mode,
        panel_metric_name=panel_metric_name,
        panel_metric_fill_sigma_bins=panel_metric_fill_sigma_bins,
        title_prefix=title_prefix,
        show_captions=True,
    )
    panel_dpi = agg_safe_dpi(figure, render_dpi)

    if clean_path is not None:
        metric_captions = [axis.get_xlabel() for axis in rate_axes]
        for axis in rate_axes:
            axis.set_xlabel(POSITION_X_LABEL, fontsize=7.0)
        figure.savefig(clean_path, dpi=panel_dpi, bbox_inches="tight")
        for axis, caption in zip(rate_axes, metric_captions, strict=False):
            axis.set_xlabel(caption, fontsize=8.0, labelpad=2.5)

    if show_spike_positions:
        for row_index, unit_index in enumerate(ranked_indices):
            spike_positions = spike_positions_by_unit.get(int(unit_index))
            if spike_positions is None or spike_positions.size == 0:
                continue
            rate_axes[row_index].scatter(
                spike_positions[:, 0],
                spike_positions[:, 1],
                s=9.0,
                c="#8DEBFF",
                edgecolors="#111111",
                linewidths=0.25,
                alpha=0.7,
                zorder=6,
            )

    figure.savefig(path, dpi=panel_dpi, bbox_inches="tight")
    plt.close(figure)
    return path, clean_path

def _render_rate_map_grid(
    path: Path,
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
    render_dpi: int,
    clean_path: Path | None = None,
) -> tuple[Path, Path | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if clean_path is not None:
        clean_path.parent.mkdir(parents=True, exist_ok=True)
    if len(ranked_indices) == 0:
        figure = empty_rate_map_figure(
            f"No units available for {source_name} {title_prefix} ({split_name})"
        )
        if clean_path is not None:
            figure.savefig(clean_path, dpi=render_dpi, bbox_inches="tight")
        figure.savefig(path, dpi=render_dpi, bbox_inches="tight")
        plt.close(figure)
        return path, clean_path

    figure, heatmap_axes = build_rate_map_grid_figure(
        source_name=source_name,
        split_name=split_name,
        bounds=bounds,
        rate_maps=rate_maps,
        ranked_indices=ranked_indices,
        spatial_information_bits=spatial_information_bits,
        mean_panel_metric_inside_fields=mean_panel_metric_inside_fields,
        world_overlay=world_overlay,
        mean_spatial_information_bits=mean_spatial_information_bits,
        shared_color_scale=shared_color_scale,
        colormap_mode=colormap_mode,
        panel_metric_name=panel_metric_name,
        title_prefix=title_prefix,
        show_captions=True,
    )

    figure.savefig(path, dpi=render_dpi, bbox_inches="tight")
    if clean_path is not None:
        for axis in heatmap_axes:
            axis.set_xlabel("")
        figure.savefig(clean_path, dpi=render_dpi, bbox_inches="tight")
    plt.close(figure)
    return path, clean_path

def render_support_map(
    path: Path,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    support_counts: np.ndarray,
    world_overlay: WorldOverlay | None,
    title: str,
    render_dpi: int,
) -> Path:
    """Render the occupancy/support map as a standalone single figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap_name, norm = support_map_style(support_counts)
    display_map = np.ma.masked_where(support_counts <= 0.0, support_counts)
    figure, axis = plt.subplots(figsize=(5.6, 5.6), constrained_layout=True)
    axis.imshow(
        display_map,
        origin="lower",
        cmap=cmap_name,
        norm=norm,
        interpolation="nearest",
        extent=_bounds_to_extent(bounds),
    )
    _apply_world_context(
        axis,
        bounds=bounds,
        world_overlay=world_overlay,
        show_x_tick_labels=True,
        show_y_tick_labels=True,
    )
    axis.set_title(title, fontsize=11)
    _attach_gutter_colorbar(
        figure,
        axis,
        mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap_name),
        norm=norm,
        label="episodes",
    )
    figure.savefig(path, dpi=render_dpi, bbox_inches="tight")
    plt.close(figure)
    return path

_PER_UNIT_EXPORT_DIR_NAME = "rate_map_units"
_PER_UNIT_EXPORT_DPI = 160


@dataclass(frozen=True, slots=True)
class _PerUnitRenderChunk:
    """One worker's share of the per-unit export: its maps, its target paths, its styling."""

    rate_maps: np.ndarray
    png_paths: tuple[Path, ...]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    world_overlay: WorldOverlay | None
    common_style_limit: float | None
    population_is_signed: bool
    colormap_mode: str


def _render_per_unit_chunk(chunk: _PerUnitRenderChunk) -> None:
    extent = _bounds_to_extent(chunk.bounds)
    for rate_map, png_path in zip(chunk.rate_maps, chunk.png_paths, strict=False):
        limit = (
            chunk.common_style_limit
            if chunk.common_style_limit is not None
            else _family_limit(rate_map, is_signed=chunk.population_is_signed)
        )
        cmap_name, norm = _style_for_limit(
            limit,
            is_signed=chunk.population_is_signed,
            colormap_mode=chunk.colormap_mode,
        )
        figure, axis = plt.subplots(figsize=(3.0, 3.0))
        axis.imshow(
            np.ma.masked_invalid(rate_map),
            origin="lower",
            cmap=cmap_name,
            norm=norm,
            interpolation="nearest",
            extent=extent,
        )
        finalize_arena_axis(
            axis,
            x_bounds=chunk.bounds[0],
            y_bounds=chunk.bounds[1],
            world_overlay=chunk.world_overlay,
        )
        axis.axis("off")
        figure.savefig(
            png_path,
            dpi=_PER_UNIT_EXPORT_DPI,
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )
        plt.close(figure)


def _render_per_unit_chunks(chunks: list[_PerUnitRenderChunk]) -> None:
    """Render every chunk, one render-pool process each; a lone chunk stays in this process."""
    if len(chunks) < 2:
        for chunk in chunks:
            _render_per_unit_chunk(chunk)
        return
    pool = render_pool()
    for future in [pool.submit(_render_per_unit_chunk, chunk) for chunk in chunks]:
        future.result()


def export_per_unit_rate_maps(
    module_dir: Path,
    *,
    source_name: str,
    split_name: str,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    rate_maps: np.ndarray,
    occupancy: np.ndarray,
    skaggs_bits: np.ndarray,
    unit_indices: np.ndarray,
    world_overlay: WorldOverlay | None,
    shared_color_scale: bool,
    colormap_mode: str,
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    """Export the panel-selected units as clean single-map PNGs plus one npz bundle."""
    units_dir = module_dir / _PER_UNIT_EXPORT_DIR_NAME
    units_dir.mkdir(parents=True, exist_ok=True)
    export_indices = np.sort(np.asarray(unit_indices, dtype=np.int64))
    population_is_signed = _population_is_signed(rate_maps[export_indices])
    common_style_limit = (
        _family_limit(rate_maps[export_indices], is_signed=population_is_signed)
        if shared_color_scale
        else None
    )
    png_paths = tuple(
        units_dir / f"unit_{unit_index:04d}__{source_name}__{split_name}.png"
        for unit_index in export_indices.tolist()
    )
    exported_figures = {
        f"rate_map_unit_{unit_index:04d}": png_path
        for unit_index, png_path in zip(export_indices.tolist(), png_paths, strict=False)
    }
    exported_destinations = {
        figure_key: Path(_PER_UNIT_EXPORT_DIR_NAME) / png_path.name
        for figure_key, png_path in exported_figures.items()
    }
    _render_per_unit_chunks(
        [
            _PerUnitRenderChunk(
                rate_maps=rate_maps[export_indices[block.start : block.stop]],
                png_paths=png_paths[block.start : block.stop],
                bounds=bounds,
                world_overlay=world_overlay,
                common_style_limit=common_style_limit,
                population_is_signed=population_is_signed,
                colormap_mode=colormap_mode,
            )
            for block in index_blocks(
                export_indices.size, render_block_count(export_indices.size)
            )
        ]
    )
    table_path = module_dir / f"rate_map_units__{source_name}__{split_name}.npz"
    np.savez_compressed(
        table_path,
        unit_ids=export_indices,
        rate_maps=rate_maps[export_indices].astype(np.float32, copy=False),
        occupancy=occupancy.astype(np.float32, copy=False),
        skaggs_bits=skaggs_bits[export_indices].astype(np.float32, copy=False),
    )
    return exported_figures, exported_destinations, table_path

_METRICS_TABLE_HEADER = [
    "unit_index",
    "place_field_metrics_supported",
    "spatial_information_bits",
    "spatial_coherence",
    "max_available_confound_score",
    "reliability_weighted_information",
    "reliability_weighted_information_excess",
    "coding_purity_score",
    "peak_rate",
    "field_count",
    "field_area_bins",
    "mean_thresholded_reliability",
    "mean_thresholded_reliability_inside_fields",
    "mean_reliability_lift_inside_fields",
    "field_traversal_reliability",
    "field_traversal_count",
    "field_traversal_reliability_directional",
    "field_traversal_directional_count",
    "field_core_traversal_reliability",
    "field_core_traversal_count",
    "mean_quantile_thresholded_reliability",
    "mean_quantile_thresholded_reliability_inside_fields",
    "mean_bin_consistency",
    "mean_bin_consistency_inside_fields",
    "bin_consistency_supported_field_fraction",
    "mean_split_half_agreement",
    "mean_split_half_agreement_inside_fields",
    "split_half_agreement_supported_field_fraction",
    "mean_bin_coefficient_of_variation",
    "split_half_rate_map_correlation",
    "episode_rate_map_correlation",
]

def render_panel_family(
    *,
    analysis_input: AnalysisInput,
    bundle: RateMapMetricBundle,
    settings: _PanelSettings,
    ranked_units: _RankedUnits,
    page_ranges: list[tuple[int, int]],
    figures: dict[str, Path],
    figure_key_prefix: str,
    panel_base_path: Path,
    clean_panel_base_path: Path,
    metric_name: str,
    metric_maps: np.ndarray,
    metric_inside_fields: np.ndarray,
    metric_supported_field_fraction: np.ndarray,
    title_prefix_root: str,
    spike_positions_by_unit: dict[int, np.ndarray],
    world_overlay: WorldOverlay | None,
) -> PanelFamilyOutcome:
    """Render one panel family across its pages; the first page keeps the un-suffixed name."""
    outcome = PanelFamilyOutcome()
    for page_index, (start_index, end_index) in enumerate(page_ranges):
        page_ranked_indices = ranked_units.summary_indices[start_index:end_index]
        page_path = (
            _rate_map_page_path(panel_base_path, start_index=start_index, end_index=end_index)
            if len(page_ranges) > 1
            else panel_base_path
        )
        clean_page_path = (
            _rate_map_page_path(
                clean_panel_base_path, start_index=start_index, end_index=end_index
            )
            if len(page_ranges) > 1
            else clean_panel_base_path
        )
        rendered_panel_path, rendered_clean_panel_path = _render_summary_panel(
            page_path,
            source_name=analysis_input.source_name,
            split_name=analysis_input.split_name,
            bounds=bundle.rate_map_result.bounds,
            rate_maps=bundle.rate_map_result.rate_maps,
            panel_metric_maps=metric_maps,
            ranked_indices=page_ranked_indices,
            field_masks=bundle.fields.masks,
            spatial_information_bits=bundle.place_metrics.spatial_information_bits,
            mean_panel_metric_inside_fields=metric_inside_fields,
            panel_metric_supported_field_fraction=metric_supported_field_fraction,
            split_half_correlations=bundle.reliability.split_half_rate_map_correlation,
            episode_rate_map_correlations=bundle.reliability.episode_rate_map_correlation,
            field_counts=bundle.fields.counts,
            spike_positions_by_unit=spike_positions_by_unit,
            world_overlay=world_overlay,
            use_absolute_activations=settings.use_absolute_activations,
            mean_spatial_information_bits=nanmean_or_nan(
                bundle.place_metrics.spatial_information_bits
            ),
            shared_color_scale=settings.shared_color_scale,
            colormap_mode=settings.colormap_mode,
            panel_metric_name=metric_name,
            panel_metric_fill_sigma_bins=settings.panel_metric_fill_sigma_bins,
            show_spike_positions=True,
            title_prefix=(
                f"{title_prefix_root} {start_index}_{end_index}"
                if len(page_ranges) > 1
                else title_prefix_root
            ),
            render_dpi=_rate_map_render_dpi(
                render_all_units=ranked_units.render_all_panel_units
            ),
            clean_path=clean_page_path,
        )
        outcome.page_paths.append(str(rendered_panel_path))
        if rendered_clean_panel_path is not None:
            outcome.clean_page_paths.append(str(rendered_clean_panel_path))
        if page_index == 0:
            outcome.first_path = rendered_panel_path
            outcome.first_clean_path = rendered_clean_panel_path
        else:
            figures[f"{figure_key_prefix}_{start_index}_{end_index}"] = rendered_panel_path
            if rendered_clean_panel_path is not None:
                figures[f"{figure_key_prefix}_clean_{start_index}_{end_index}"] = (
                    rendered_clean_panel_path
                )
    return outcome

def render_grid_pages(
    *,
    analysis_input: AnalysisInput,
    bundle: RateMapMetricBundle,
    settings: _PanelSettings,
    ranked_units: _RankedUnits,
    panel_metric_inside_fields: np.ndarray,
    module_dir: Path,
    all_units_page_size: int,
    figures: dict[str, Path],
    world_overlay: WorldOverlay | None,
) -> GridOutcome:
    """Render the companion rate-map grid across its pages."""
    grid_indices = ranked_units.grid_indices
    page_ranges = (
        _rate_map_page_ranges(len(grid_indices), page_size=all_units_page_size)
        if ranked_units.render_all_grid_units and len(grid_indices) > all_units_page_size
        else [(0, len(grid_indices))]
    )
    outcome = GridOutcome()
    grid_base_path = (
        module_dir
        / f"rate_map_grid__{analysis_input.source_name}__{analysis_input.split_name}.png"
    )
    clean_grid_base_path = (
        module_dir
        / f"rate_map_grid_clean__{analysis_input.source_name}__{analysis_input.split_name}.png"
    )
    for page_index, (start_index, end_index) in enumerate(page_ranges):
        page_ranked_indices = grid_indices[start_index:end_index]
        page_path = (
            _rate_map_page_path(grid_base_path, start_index=start_index, end_index=end_index)
            if len(page_ranges) > 1
            else grid_base_path
        )
        clean_page_path = (
            _rate_map_page_path(
                clean_grid_base_path, start_index=start_index, end_index=end_index
            )
            if len(page_ranges) > 1
            else clean_grid_base_path
        )
        rendered_grid_path, rendered_clean_grid_path = _render_rate_map_grid(
            page_path,
            source_name=analysis_input.source_name,
            split_name=analysis_input.split_name,
            bounds=bundle.rate_map_result.bounds,
            rate_maps=bundle.rate_map_result.rate_maps,
            ranked_indices=page_ranked_indices,
            spatial_information_bits=bundle.place_metrics.spatial_information_bits,
            mean_panel_metric_inside_fields=panel_metric_inside_fields,
            world_overlay=world_overlay,
            mean_spatial_information_bits=nanmean_or_nan(
                bundle.place_metrics.spatial_information_bits
            ),
            shared_color_scale=settings.shared_color_scale,
            colormap_mode=settings.colormap_mode,
            panel_metric_name=settings.panel_metric_name,
            title_prefix=(
                f"rate maps {start_index}_{end_index}" if len(page_ranges) > 1 else "rate maps"
            ),
            render_dpi=_rate_map_render_dpi(render_all_units=ranked_units.render_all_grid_units),
            clean_path=clean_page_path,
        )
        outcome.page_paths.append(str(rendered_grid_path))
        if rendered_clean_grid_path is not None:
            outcome.clean_page_paths.append(str(rendered_clean_grid_path))
        if page_index == 0:
            outcome.first_path = rendered_grid_path
            outcome.first_clean_path = rendered_clean_grid_path
        else:
            figures[f"rate_map_grid_{start_index}_{end_index}"] = rendered_grid_path
            if rendered_clean_grid_path is not None:
                figures[f"rate_map_grid_clean_{start_index}_{end_index}"] = (
                    rendered_clean_grid_path
                )
    return outcome

def write_rate_map_metrics_table(
    path: Path,
    bundle: RateMapMetricBundle,
) -> Path:
    """One row per unit, in ranking order, of every scalar the panels put on a figure."""
    place_metrics = bundle.place_metrics
    fields = bundle.fields
    summaries = bundle.summaries
    reliability = bundle.reliability
    return write_csv(
        path,
        header=_METRICS_TABLE_HEADER,
        rows=[
            [
                int(unit_index),
                bool(place_metrics.supported[unit_index]),
                float(place_metrics.spatial_information_bits[unit_index]),
                float(place_metrics.spatial_coherence[unit_index]),
                float(place_metrics.max_available_confound[unit_index]),
                float(place_metrics.reliability_weighted_information[unit_index]),
                float(place_metrics.reliability_weighted_information_excess[unit_index]),
                float(place_metrics.coding_purity[unit_index]),
                float(bundle.signed_diagnostics.peak_magnitude[unit_index]),
                float(fields.counts[unit_index]),
                float(fields.areas[unit_index]),
                float(summaries.mean_reliability[unit_index]),
                float(fields.reliability[unit_index]),
                float(fields.reliability_lift[unit_index]),
                float(fields.traversal_reliability[unit_index]),
                float(fields.traversal_counts[unit_index]),
                float(fields.traversal_reliability_directional[unit_index]),
                float(fields.traversal_directional_counts[unit_index]),
                float(fields.core_traversal_reliability[unit_index]),
                float(fields.core_traversal_counts[unit_index]),
                float(summaries.mean_quantile_reliability[unit_index]),
                float(fields.quantile_reliability[unit_index]),
                float(summaries.mean_bin_consistency[unit_index]),
                float(fields.bin_consistency[unit_index]),
                float(fields.bin_consistency_supported_fraction[unit_index]),
                float(summaries.mean_split_half_agreement[unit_index]),
                float(fields.split_half_agreement[unit_index]),
                float(fields.split_half_agreement_supported_fraction[unit_index]),
                float(summaries.mean_bin_coefficient_of_variation[unit_index]),
                float(reliability.split_half_rate_map_correlation[unit_index]),
                float(reliability.episode_rate_map_correlation[unit_index]),
            ]
            for unit_index in bundle.ranked_indices.tolist()
        ],
    )
