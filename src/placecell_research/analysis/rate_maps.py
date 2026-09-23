"""Panel-based place-map summaries and scan-friendly grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..numerics.rate_map_kernels import (
    place_metric_support_rule,
)
from .base import AnalysisInput, AnalysisResult
from .rate_map_computation import (
    _METRIC_KEYS_BY_GROUP,
    _PER_UNIT_KEYS_BY_GROUP,
    _GridOutcome,
    _normalize_panel_reliability_metric,
    _panel_metric_family,
    _PanelFamilyOutcome,
    _PanelSettings,
    _per_unit_metrics,
    _population_metrics,
    _RankedUnits,
    _resolve_rate_map_colormap_mode,
    _select_high_activation_positions,
    _select_ranked_units,
    _use_shared_rate_map_color_scale,
)
from .rate_map_export import (
    _export_per_unit_rate_maps,
    _rate_map_page_ranges,
    _render_grid_pages,
    _render_panel_family,
    _render_support_map,
    _write_rate_map_metrics_table,
)
from .rate_map_metrics import (
    RateMapMetricBundle,
    compute_rate_map_metric_bundle,
)
from .timing import log_timing, record_timing
from .world_overlay import (
    WorldOverlay,
    overlay_bounds,
    resolve_world_overlay,
)


@dataclass(slots=True)
class _RateMapModuleBase:
    """Compute summary panels and scan-friendly rate-map grids."""

    name: str = "rate_map_bundle"
    cost_tier: str = "standard"
    all_units_page_size: int = 64
    emit_metrics: bool = True
    emit_panel: bool = True
    emit_grid: bool = True
    emit_extra_reliability_panels: bool = True
    emit_primary_metric_extra_panel: bool = True
    emit_metrics_table: bool = True
    metric_groups: tuple[str, ...] = (
        "fields",
        "reliability",
        "bin_consistency",
        "split_half",
        "episode_correlation",
        "coding_purity",
    )

    def required_representations(self) -> set[str]:
        return set()

    def _resolve_panel_settings(self, config: dict) -> _PanelSettings:
        panel_metric_name = _normalize_panel_reliability_metric(
            config.get("rate_map_panel_reliability_metric", "quantile_thresholded_reliability")
        )
        return _PanelSettings(
            panel_metric_name=panel_metric_name,
            panel_metric_fill_sigma_bins=float(
                config.get("rate_map_panel_metric_fill_sigma_bins", 1.0)
            ),
            threshold_fraction=float(config.get("reliability_threshold_fraction", 0.2)),
            threshold_quantile=float(config.get("reliability_threshold_quantile", 0.95)),
            per_bin_cv_min_episodes=int(config.get("per_bin_cv_min_episodes", 3)),
            split_half_agreement_min_episodes_per_half=int(
                config.get("split_half_agreement_min_episodes_per_half", 2)
            ),
            use_absolute_activations=bool(
                config.get("reliability_use_absolute_activations", False)
            ),
            shared_color_scale=_use_shared_rate_map_color_scale(config),
            colormap_mode=_resolve_rate_map_colormap_mode(config),
            emit_thresholded_reliability_panel=self.emit_extra_reliability_panels
            and bool(config.get("rate_map_panel_emit_thresholded_reliability_figure", True)),
            emit_quantile_thresholded_reliability_panel=self.emit_extra_reliability_panels
            and bool(
                config.get("rate_map_panel_emit_quantile_thresholded_reliability_figure", True)
            ),
            negative_tolerance=float(config.get("place_metric_negative_tolerance", 1e-8)),
            max_negative_bin_fraction=float(
                config.get("place_metric_max_negative_bin_fraction", 0.01)
            ),
            max_negative_peak_fraction=float(
                config.get("place_metric_max_negative_peak_fraction", 0.05)
            ),
            export_per_unit=self.emit_panel
            and bool(config.get("rate_map_export_per_unit", False)),
            spike_overlay_max_points=int(config.get("rate_map_spike_overlay_max_points", 180)),
            panel_top_k=int(config.get("rate_map_panel_top_k", 8)),
            grid_top_k=int(config.get("rate_map_grid_top_k", 24)),
            unit_order_mode=str(config.get("rate_map_unit_order", "field_position"))
            .strip()
            .lower(),
        )

    def _renders_extra_family(self, settings: _PanelSettings, metric_name: str) -> bool:
        """Whether a non-primary panel family is drawn for metric_name."""
        emits_family = (
            settings.emit_thresholded_reliability_panel
            if metric_name == "thresholded_reliability"
            else settings.emit_quantile_thresholded_reliability_panel
        )
        return emits_family and (
            self.emit_primary_metric_extra_panel or settings.panel_metric_name != metric_name
        )

    def _render_all_panels(
        self,
        analysis_input: AnalysisInput,
        bundle: RateMapMetricBundle,
        settings: _PanelSettings,
        ranked_units: _RankedUnits,
        *,
        module_dir: Path,
        figures: dict[str, Path],
        world_overlay: WorldOverlay | None,
        timing_seconds: dict[str, float],
    ) -> dict[str, _PanelFamilyOutcome]:
        """Draw the primary panel family, the support map, and any extra reliability families."""
        summary_indices = ranked_units.summary_indices
        page_ranges = (
            _rate_map_page_ranges(len(summary_indices), page_size=self.all_units_page_size)
            if ranked_units.render_all_panel_units
            and len(summary_indices) > self.all_units_page_size
            else [(0, len(summary_indices))]
        )
        renders_thresholded = (
            self.emit_panel and settings.panel_metric_name == "thresholded_reliability"
        ) or self._renders_extra_family(settings, "thresholded_reliability")
        renders_quantile = (
            self.emit_panel and settings.panel_metric_name == "quantile_thresholded_reliability"
        ) or self._renders_extra_family(settings, "quantile_thresholded_reliability")

        section_started_at = perf_counter()
        spike_positions_by_unit = (
            _select_high_activation_positions(
                analysis_input,
                summary_indices,
                threshold_fraction=settings.threshold_fraction,
                max_points_per_unit=settings.spike_overlay_max_points,
                use_absolute_activations=settings.use_absolute_activations,
            )
            if (self.emit_panel or renders_thresholded or renders_quantile)
            else {}
        )
        record_timing(timing_seconds, "select_spike_positions", section_started_at)

        def render_family(
            *,
            figure_key_prefix: str,
            base_stem: str,
            metric_name: str,
            title_prefix_root: str,
        ) -> _PanelFamilyOutcome:
            metric_maps, inside_fields, supported_fraction = _panel_metric_family(
                bundle, metric_name
            )
            suffix = f"__{analysis_input.source_name}__{analysis_input.split_name}.png"
            return _render_panel_family(
                analysis_input=analysis_input,
                bundle=bundle,
                settings=settings,
                ranked_units=ranked_units,
                page_ranges=page_ranges,
                figures=figures,
                figure_key_prefix=figure_key_prefix,
                panel_base_path=module_dir / f"{base_stem}{suffix}",
                clean_panel_base_path=module_dir / f"{base_stem}_clean{suffix}",
                metric_name=metric_name,
                metric_maps=metric_maps,
                metric_inside_fields=inside_fields,
                metric_supported_field_fraction=supported_fraction,
                title_prefix_root=title_prefix_root,
                spike_positions_by_unit=spike_positions_by_unit,
                world_overlay=world_overlay,
            )

        section_started_at = perf_counter()
        primary = _PanelFamilyOutcome()
        if self.emit_panel:
            primary = render_family(
                figure_key_prefix="rate_map_panel",
                base_stem="rate_map_panel",
                metric_name=settings.panel_metric_name,
                title_prefix_root=(
                    "spatial firing summary "
                    "(PREVIEW: ranked by peak activation, no stability-supported units)"
                    if bundle.ranking_is_peak_preview
                    else "spatial firing summary"
                ),
            )
        record_timing(timing_seconds, "render_primary_panel", section_started_at)

        if self.emit_panel:
            primary_support_counts = {
                "thresholded_reliability": bundle.reliability.thresholded_visit_counts,
                "quantile_thresholded_reliability": bundle.reliability.quantile_visit_counts,
                "split_half_agreement": (
                    bundle.reliability.split_half_agreement_support_counts
                ),
            }.get(settings.panel_metric_name, bundle.reliability.bin_consistency_visit_counts)
            figures["support_map"] = _render_support_map(
                module_dir
                / f"support_map__{analysis_input.source_name}__{analysis_input.split_name}.png",
                bounds=bundle.rate_map_result.bounds,
                support_counts=primary_support_counts,
                world_overlay=world_overlay,
                title="Support Map\nColor = visiting episodes",
                render_dpi=170,
            )

        families = {"primary": primary}
        for metric_name, base_stem, title_prefix_root in (
            (
                "thresholded_reliability",
                "rate_map_panel_thresholded_reliability",
                "thresholded reliability summary",
            ),
            (
                "quantile_thresholded_reliability",
                "rate_map_panel_quantile_thresholded_reliability",
                "quantile-thresholded reliability summary",
            ),
        ):
            if self.emit_panel and settings.panel_metric_name == metric_name:
                families[metric_name] = _PanelFamilyOutcome(
                    first_path=primary.first_path,
                    first_clean_path=primary.first_clean_path,
                    page_paths=list(primary.page_paths),
                    clean_page_paths=list(primary.clean_page_paths),
                )
                continue
            if not self._renders_extra_family(settings, metric_name):
                families[metric_name] = _PanelFamilyOutcome()
                continue
            section_started_at = perf_counter()
            families[metric_name] = render_family(
                figure_key_prefix=base_stem,
                base_stem=base_stem,
                metric_name=metric_name,
                title_prefix_root=title_prefix_root,
            )
            timing_seconds["render_extra_panels"] = (
                timing_seconds.get("render_extra_panels", 0.0)
                + perf_counter()
                - section_started_at
            )
        timing_seconds.setdefault("render_extra_panels", 0.0)
        return families

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        module_started_at = perf_counter()
        timing_seconds: dict[str, float] = {}
        settings = self._resolve_panel_settings(config)
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None

        section_started_at = perf_counter()
        bundle = analysis_input.get_cached_rate_map_computation(
            (
                "rate_map_metric_bundle",
                tuple(sorted((str(key), repr(value)) for key, value in config.items())),
                world_bounds,
            ),
            lambda: compute_rate_map_metric_bundle(
                analysis_input,
                config,
                bounds=world_bounds,
            ),
        )
        record_timing(timing_seconds, "metric_bundle", section_started_at)
        for section_name, elapsed_seconds in bundle.timing_seconds.items():
            timing_seconds[f"metric_bundle.{section_name}"] = elapsed_seconds

        module_dir = output_dir / self.name
        ranked_units = _select_ranked_units(bundle, settings, config)
        figures: dict[str, Path] = {}
        figure_destinations: dict[str, Path] = {}
        panel_families = self._render_all_panels(
            analysis_input,
            bundle,
            settings,
            ranked_units,
            module_dir=module_dir,
            figures=figures,
            world_overlay=world_overlay,
            timing_seconds=timing_seconds,
        )

        per_unit_table_path: Path | None = None
        if settings.export_per_unit:
            section_started_at = perf_counter()
            (
                per_unit_figures,
                per_unit_destinations,
                per_unit_table_path,
            ) = _export_per_unit_rate_maps(
                module_dir,
                source_name=analysis_input.source_name,
                split_name=analysis_input.split_name,
                bounds=bundle.rate_map_result.bounds,
                rate_maps=bundle.rate_map_result.rate_maps,
                occupancy=bundle.rate_map_result.occupancy,
                skaggs_bits=bundle.place_metrics.spatial_information_bits,
                unit_indices=ranked_units.summary_indices,
                world_overlay=world_overlay,
                shared_color_scale=settings.shared_color_scale,
                colormap_mode=settings.colormap_mode,
            )
            figures.update(per_unit_figures)
            figure_destinations.update(per_unit_destinations)
            record_timing(timing_seconds, "export_per_unit_maps", section_started_at)

        _, panel_metric_inside_fields, _ = _panel_metric_family(
            bundle, settings.panel_metric_name
        )
        section_started_at = perf_counter()
        grid = (
            _render_grid_pages(
                analysis_input=analysis_input,
                bundle=bundle,
                settings=settings,
                ranked_units=ranked_units,
                panel_metric_inside_fields=panel_metric_inside_fields,
                module_dir=module_dir,
                all_units_page_size=self.all_units_page_size,
                figures=figures,
                world_overlay=world_overlay,
            )
            if self.emit_grid
            else _GridOutcome()
        )
        record_timing(timing_seconds, "render_primary_grid", section_started_at)

        primary = panel_families["primary"]
        if self.emit_panel and primary.first_path is None:
            raise RuntimeError("Rate-map rendering did not produce primary panel output.")
        for figure_key, figure_path in (
            ("rate_map_panel", primary.first_path),
            ("rate_map_panel_clean", primary.first_clean_path),
            ("rate_map_grid", grid.first_path),
            ("rate_map_grid_clean", grid.first_clean_path),
            (
                "rate_map_panel_thresholded_reliability",
                panel_families["thresholded_reliability"].first_path,
            ),
            (
                "rate_map_panel_thresholded_reliability_clean",
                panel_families["thresholded_reliability"].first_clean_path,
            ),
            (
                "rate_map_panel_quantile_thresholded_reliability",
                panel_families["quantile_thresholded_reliability"].first_path,
            ),
            (
                "rate_map_panel_quantile_thresholded_reliability_clean",
                panel_families["quantile_thresholded_reliability"].first_clean_path,
            ),
        ):
            if figure_path is not None:
                figures[figure_key] = figure_path

        section_started_at = perf_counter()
        metrics_table_path: Path | None = None
        if self.emit_metrics and self.emit_metrics_table:
            metrics_table_path = _write_rate_map_metrics_table(
                module_dir
                / (
                    "rate_map_panel_metrics"
                    f"__{analysis_input.source_name}__{analysis_input.split_name}.csv"
                ),
                bundle,
            )
        record_timing(timing_seconds, "write_metrics_table", section_started_at)
        timing_seconds["total"] = perf_counter() - module_started_at
        timing_order = (
            "total",
            "metric_bundle",
            "select_spike_positions",
            "render_primary_panel",
            "render_extra_panels",
            "export_per_unit_maps",
            "render_primary_grid",
            "write_metrics_table",
        )
        metric_bundle_timing_order = tuple(
            sorted(
                section_name
                for section_name in timing_seconds
                if section_name.startswith("metric_bundle.")
            )
        )
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
            section_names=(*timing_order, *metric_bundle_timing_order),
        )
        selected_metric_keys = set().union(
            *(_METRIC_KEYS_BY_GROUP[group] for group in self.metric_groups)
        )
        selected_per_unit_keys = set().union(
            *(_PER_UNIT_KEYS_BY_GROUP[group] for group in self.metric_groups)
        )
        filtered_metrics = {
            key: value
            for key, value in _population_metrics(bundle).items()
            if key in selected_metric_keys
        }
        filtered_per_unit_metrics = {
            key: value
            for key, value in _per_unit_metrics(bundle).items()
            if key in selected_per_unit_keys
        }
        tables: dict[str, Path] = {}
        if metrics_table_path is not None:
            tables["rate_map_panel_metrics"] = metrics_table_path
        if per_unit_table_path is not None:
            tables["rate_map_units"] = per_unit_table_path
        return AnalysisResult(
            metrics=filtered_metrics if self.emit_metrics else {},
            per_unit_metrics=filtered_per_unit_metrics if self.emit_metrics else {},
            figures=figures,
            tables=tables,
            metadata=self._result_metadata(
                analysis_input,
                bundle,
                settings,
                ranked_units,
                panel_families=panel_families,
                grid=grid,
                world_overlay=world_overlay,
                timing_seconds=timing_seconds,
            ),
            figure_destinations=figure_destinations,
        )

    def _result_metadata(
        self,
        analysis_input: AnalysisInput,
        bundle: RateMapMetricBundle,
        settings: _PanelSettings,
        ranked_units: _RankedUnits,
        *,
        panel_families: dict[str, _PanelFamilyOutcome],
        grid: _GridOutcome,
        world_overlay: WorldOverlay | None,
        timing_seconds: dict[str, float],
    ) -> dict[str, object]:
        """Provenance the report carries alongside the numbers: knobs, support, page lists."""
        del analysis_input
        supports_place_metrics = bundle.place_metrics.supported
        thresholded = panel_families["thresholded_reliability"]
        quantile = panel_families["quantile_thresholded_reliability"]
        return {
            "bounds": bundle.rate_map_result.bounds,
            "world_overlay_applied": world_overlay is not None,
            "rate_map_colormap_mode": settings.colormap_mode,
            "rate_map_panel_reliability_metric": settings.panel_metric_name,
            "rate_map_panel_emit_thresholded_reliability_figure": (
                settings.emit_thresholded_reliability_panel
            ),
            "rate_map_panel_emit_quantile_thresholded_reliability_figure": (
                settings.emit_quantile_thresholded_reliability_panel
            ),
            "reliability_threshold_quantile": settings.threshold_quantile,
            "rate_map_panel_metric_fill_sigma_bins": settings.panel_metric_fill_sigma_bins,
            "rate_map_export_per_unit": settings.export_per_unit,
            "place_field_metrics_supported_unit_count": int(
                np.count_nonzero(supports_place_metrics)
            ),
            "place_field_metrics_total_unit_count": int(len(supports_place_metrics)),
            "place_field_metrics_support_rule": place_metric_support_rule(
                negative_tolerance=settings.negative_tolerance,
                max_negative_bin_fraction=settings.max_negative_bin_fraction,
                max_negative_peak_fraction=settings.max_negative_peak_fraction,
            ),
            "place_field_metrics_skipped_for_signed_rate_maps": bool(
                np.any(~supports_place_metrics)
            ),
            "place_metric_negative_tolerance": settings.negative_tolerance,
            "place_metric_max_negative_bin_fraction": settings.max_negative_bin_fraction,
            "place_metric_max_negative_peak_fraction": settings.max_negative_peak_fraction,
            "place_metric_negative_bin_fraction": (
                bundle.place_metrics.prepared_maps.negative_bin_fraction.astype(
                    np.float32, copy=False
                )
            ),
            "place_metric_negative_peak_fraction": (
                bundle.place_metrics.prepared_maps.negative_peak_fraction.astype(
                    np.float32, copy=False
                )
            ),
            "per_bin_cv_min_episodes": settings.per_bin_cv_min_episodes,
            "split_half_agreement_min_episodes_per_half": (
                settings.split_half_agreement_min_episodes_per_half
            ),
            "rate_map_panel_has_support_column": False,
            "rate_map_has_per_unit_colorbars": not settings.shared_color_scale,
            "rate_map_all_units_page_size": self.all_units_page_size,
            "thresholded_reliability_visit_counts": (
                bundle.reliability.thresholded_visit_counts
            ),
            "quantile_thresholded_reliability_visit_counts": (
                bundle.reliability.quantile_visit_counts
            ),
            "bin_consistency_visit_counts": bundle.reliability.bin_consistency_visit_counts,
            "split_half_agreement_support_counts": (
                bundle.reliability.split_half_agreement_support_counts
            ),
            "rate_map_panel_pages": panel_families["primary"].page_paths,
            "rate_map_panel_clean_pages": panel_families["primary"].clean_page_paths,
            "rate_map_panel_thresholded_reliability_pages": thresholded.page_paths,
            "rate_map_panel_thresholded_reliability_clean_pages": thresholded.clean_page_paths,
            "rate_map_panel_quantile_thresholded_reliability_pages": quantile.page_paths,
            "rate_map_panel_quantile_thresholded_reliability_clean_pages": (
                quantile.clean_page_paths
            ),
            "rate_map_grid_pages": grid.page_paths,
            "rate_map_grid_clean_pages": grid.clean_page_paths,
            "ranked_units_summary": ranked_units.summary_indices.astype(int).tolist(),
            "ranked_units_grid": (
                ranked_units.grid_indices.astype(int).tolist() if self.emit_grid else []
            ),
            "rate_map_ranking_metric": (
                "peak_activation_preview_no_assessable_units"
                if bundle.ranking_is_peak_preview
                else "eligibility_lexicographic"
            ),
            "rate_map_ranking_is_peak_preview": bundle.ranking_is_peak_preview,
            "spatial_information_null_shuffles": (
                bundle.spatial_information_null.num_shuffles
            ),
            "rate_map_timing_seconds": timing_seconds,
        }


@dataclass(slots=True)
class RateMapFieldsModule(_RateMapModuleBase):
    """Compute pooled rate-map field and peak-rate metrics."""

    name: str = "rate_map_fields"
    emit_panel: bool = False
    emit_grid: bool = False
    emit_extra_reliability_panels: bool = False
    metric_groups: tuple[str, ...] = ("fields",)


@dataclass(slots=True)
class RateMapReliabilityModule(_RateMapModuleBase):
    """Compute thresholded and quantile-thresholded reliability summaries."""

    name: str = "rate_map_reliability"
    emit_panel: bool = False
    emit_grid: bool = False
    emit_extra_reliability_panels: bool = False
    emit_metrics_table: bool = False
    metric_groups: tuple[str, ...] = ("reliability",)


@dataclass(slots=True)
class RateMapBinConsistencyModule(_RateMapModuleBase):
    """Compute per-bin consistency and coefficient-of-variation summaries."""

    name: str = "rate_map_bin_consistency"
    emit_panel: bool = False
    emit_grid: bool = False
    emit_extra_reliability_panels: bool = False
    emit_metrics_table: bool = False
    metric_groups: tuple[str, ...] = ("bin_consistency",)


@dataclass(slots=True)
class RateMapSplitHalfModule(_RateMapModuleBase):
    """Compute split-half agreement summaries."""

    name: str = "rate_map_split_half"
    emit_panel: bool = False
    emit_grid: bool = False
    emit_extra_reliability_panels: bool = False
    emit_metrics_table: bool = False
    metric_groups: tuple[str, ...] = ("split_half",)


@dataclass(slots=True)
class RateMapEpisodeCorrelationModule(_RateMapModuleBase):
    """Compute episode-to-episode rate-map correlation summaries."""

    name: str = "rate_map_episode_correlation"
    emit_panel: bool = False
    emit_grid: bool = False
    emit_extra_reliability_panels: bool = False
    emit_metrics_table: bool = False
    metric_groups: tuple[str, ...] = ("episode_correlation",)


@dataclass(slots=True)
class RateMapCodingPurityModule(_RateMapModuleBase):
    """Compute reliability-weighted information and coding-purity summaries."""

    name: str = "rate_map_coding_purity"
    emit_panel: bool = False
    emit_grid: bool = False
    emit_extra_reliability_panels: bool = False
    emit_metrics_table: bool = False
    metric_groups: tuple[str, ...] = ("coding_purity",)


@dataclass(slots=True)
class RateMapPanelModule(_RateMapModuleBase):
    """Render the ranked rate-map summary panel."""

    name: str = "rate_map_panel"
    emit_metrics: bool = False
    emit_grid: bool = False
    emit_extra_reliability_panels: bool = False


@dataclass(slots=True)
class RateMapGridModule(_RateMapModuleBase):
    """Render the companion rate-map grid."""

    name: str = "rate_map_grid"
    emit_metrics: bool = False
    emit_panel: bool = False
    emit_extra_reliability_panels: bool = False


@dataclass(slots=True)
class RateMapExtraReliabilityPanelsModule(_RateMapModuleBase):
    """Render non-primary reliability panel families."""

    name: str = "rate_map_extra_reliability_panels"
    emit_metrics: bool = False
    emit_panel: bool = False
    emit_grid: bool = False
    emit_primary_metric_extra_panel: bool = False
