"""Spatial information analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import (
    place_metric_support_rule,
    skaggs_spatial_information,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .world_overlay import overlay_bounds, resolve_world_overlay


@dataclass(slots=True)
class SpatialInfoModule:
    """Per-unit Skaggs spatial information."""

    name: str = "spatial_info"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        rate_map_result = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=int(config.get("num_bins_x", 60)),
            num_bins_y=int(config.get("num_bins_y", 60)),
            smoothing_sigma=float(config.get("smoothing_sigma", 0.4)),
            min_occupancy=float(config.get("min_occupancy", 1e-6)),
            bounds=world_bounds,
        )
        negative_tolerance = float(config.get("place_metric_negative_tolerance", 1e-8))
        max_negative_bin_fraction = float(
            config.get("place_metric_max_negative_bin_fraction", 0.01)
        )
        max_negative_peak_fraction = float(
            config.get("place_metric_max_negative_peak_fraction", 0.05)
        )
        scores = np.asarray(
            skaggs_spatial_information(
                rate_map_result.rate_maps,
                rate_map_result.occupancy,
                negative_tolerance=negative_tolerance,
                max_negative_bin_fraction=max_negative_bin_fraction,
                max_negative_peak_fraction=max_negative_peak_fraction,
            ),
            dtype=np.float32,
        )
        finite_scores = scores[np.isfinite(scores)]
        return AnalysisResult(
            metrics={
                "mean_spatial_information_bits": (
                    float(finite_scores.mean()) if len(finite_scores) else float("nan")
                ),
                "max_spatial_information_bits": (
                    float(finite_scores.max()) if len(finite_scores) else float("nan")
                ),
            },
            per_unit_metrics={"spatial_information_bits": scores},
            figures={},
            tables={},
            metadata={
                "spatial_information_supported_unit_count": int(finite_scores.size),
                "spatial_information_total_unit_count": int(scores.size),
                "spatial_information_support_rule": place_metric_support_rule(
                    negative_tolerance=negative_tolerance,
                    max_negative_bin_fraction=max_negative_bin_fraction,
                    max_negative_peak_fraction=max_negative_peak_fraction,
                ),
                "place_metric_negative_tolerance": negative_tolerance,
                "place_metric_max_negative_bin_fraction": max_negative_bin_fraction,
                "place_metric_max_negative_peak_fraction": max_negative_peak_fraction,
            },
        )
