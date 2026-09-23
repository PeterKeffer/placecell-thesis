"""Gridness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import (
    gridness_score,
    gridness_summary_metrics,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .world_overlay import overlay_bounds, resolve_world_overlay


@dataclass(slots=True)
class GridnessModule:
    """Approximate gridness scoring."""

    name: str = "gridness"
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
        scores = np.asarray(
            [gridness_score(rate_map) for rate_map in rate_map_result.rate_maps],
            dtype=np.float32,
        )
        return AnalysisResult(
            metrics=gridness_summary_metrics(scores),
            per_unit_metrics={"gridness": scores},
            figures={},
            tables={},
        )
