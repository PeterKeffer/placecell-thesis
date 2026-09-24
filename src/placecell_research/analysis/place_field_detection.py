"""Place field detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import (
    compute_largest_field_metrics,
    compute_place_field_mask,
    prepare_place_metric_rate_maps,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .rate_map_metrics import nanmean_or_nan
from .world_overlay import overlay_bounds, resolve_world_overlay


@dataclass(slots=True)
class PlaceFieldDetectionModule:
    """Threshold-based place field counting."""

    name: str = "place_field_detection"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        threshold_fraction = float(config.get("place_field_threshold_fraction", 0.2))
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        rate_map_result = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=int(config.get("num_bins_x", 60)),
            num_bins_y=int(config.get("num_bins_y", 60)),
            smoothing_sigma=float(config.get("smoothing_sigma", 0.3)),
            min_occupancy=float(config.get("min_occupancy", 1e-6)),
            bounds=world_bounds,
        )
        prepared_maps = prepare_place_metric_rate_maps(rate_map_result.rate_maps)
        unit_count = rate_map_result.rate_maps.shape[0]
        counts_array = np.full(unit_count, np.nan, dtype=np.float32)
        areas_array = np.full(unit_count, np.nan, dtype=np.float32)
        radii_array = np.full(unit_count, np.nan, dtype=np.float32)
        coherences_array = np.full(unit_count, np.nan, dtype=np.float32)
        for unit_index, rate_map in enumerate(prepared_maps.clipped_rate_maps):
            if not prepared_maps.supported_mask[unit_index]:
                continue
            _, field_count, field_area = compute_place_field_mask(rate_map, threshold_fraction)
            _, coherence, radius = compute_largest_field_metrics(rate_map, threshold_fraction)
            counts_array[unit_index] = float(field_count)
            areas_array[unit_index] = float(field_area)
            radii_array[unit_index] = float(radius)
            coherences_array[unit_index] = float(coherence)
        supported_count = int(np.count_nonzero(prepared_maps.supported_mask))

        return AnalysisResult(
            metrics={
                "mean_field_count": nanmean_or_nan(counts_array),
                "mean_field_area": nanmean_or_nan(areas_array),
                "mean_field_radius": nanmean_or_nan(radii_array),
                "mean_field_coherence": nanmean_or_nan(coherences_array),
                "field_metrics_supported_fraction": (
                    float(supported_count / unit_count) if unit_count else float("nan")
                ),
            },
            per_unit_metrics={
                "field_count": counts_array,
                "field_area": areas_array,
                "field_radius": radii_array,
                "field_coherence": coherences_array,
                "field_metrics_supported": prepared_maps.supported_mask.astype(
                    np.float32, copy=False
                ),
            },
            figures={},
            tables={},
        )
