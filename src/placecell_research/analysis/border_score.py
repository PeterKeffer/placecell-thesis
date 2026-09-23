"""Border/boundary-cell score (Solstad et al. 2008)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .world_overlay import overlay_bounds, resolve_world_overlay


def border_score(
    rate_map: np.ndarray, *, threshold_fraction: float = 0.3, min_field_bins: int = 4
) -> float:
    """Solstad border score for one rate map."""
    grid = np.nan_to_num(np.asarray(rate_map, dtype=np.float64), nan=0.0)
    grid = np.clip(grid, a_min=0.0, a_max=None)
    peak = grid.max()
    if peak <= 0:
        return 0.0
    height, width = grid.shape
    active = grid >= threshold_fraction * peak
    labels, num_fields = label(active)
    if num_fields == 0:
        return 0.0

    coverage_max = 0.0
    for field_id in range(1, num_fields + 1):
        field = labels == field_id
        if field.sum() < min_field_bins:
            continue
        wall_coverage = (
            field[0, :].mean(),
            field[-1, :].mean(),
            field[:, 0].mean(),
            field[:, -1].mean(),
        )
        coverage_max = max(coverage_max, max(wall_coverage))

    rows, cols = np.indices((height, width))
    distance_to_wall = np.minimum.reduce(
        [rows, height - 1 - rows, cols, width - 1 - cols]
    ).astype(np.float64)
    max_distance = max((min(height, width) - 1) / 2.0, 1.0)
    weights = np.where(active, grid, 0.0)
    weight_total = weights.sum()
    if weight_total <= 0:
        return 0.0
    mean_distance = float((weights * distance_to_wall).sum() / weight_total) / max_distance

    denominator = coverage_max + mean_distance
    if denominator <= 0:
        return 0.0
    return float((coverage_max - mean_distance) / denominator)


@dataclass(slots=True)
class BorderScoreModule:
    """Per-unit border/boundary-cell scoring."""

    name: str = "border_score"
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
        threshold_fraction = float(config.get("border_field_threshold_fraction", 0.3))
        scores = np.asarray(
            [border_score(rate_map, threshold_fraction=threshold_fraction)
             for rate_map in rate_map_result.rate_maps],
            dtype=np.float32,
        )
        threshold = float(config.get("border_score_threshold", 0.5))
        return AnalysisResult(
            metrics={
                "mean_border_score": float(scores.mean()) if len(scores) else 0.0,
                "max_border_score": float(scores.max()) if len(scores) else 0.0,
                "fraction_border": float((scores > threshold).mean()) if len(scores) else 0.0,
            },
            per_unit_metrics={"border_score": scores},
            figures={},
            tables={},
        )
