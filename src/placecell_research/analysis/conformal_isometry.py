"""Conformal-isometry / metric-distortion of a representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from ..numerics.rate_map_kernels import flatten_valid_steps
from .base import AnalysisInput, AnalysisResult


@dataclass(slots=True)
class ConformalIsometryModule:
    """Metric-distortion / conformal-isometry scoring of the population code."""

    name: str = "conformal_isometry"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        flat, positions = flatten_valid_steps(
            analysis_input.representation, analysis_input.position_xy, analysis_input.valid_mask
        )
        max_points = int(config.get("conformal_max_points", 1500))
        if flat.shape[0] < 50:
            return AnalysisResult(
                metrics={}, per_unit_metrics={}, figures={}, tables={},
                metadata={"conformal_isometry_skipped": "too few samples"},
            )
        rng = np.random.default_rng(0)
        if flat.shape[0] > max_points:
            index = rng.choice(flat.shape[0], max_points, replace=False)
            flat, positions = flat[index], positions[index]
        neural = np.asarray(flat, dtype=np.float64)
        spatial = np.asarray(positions, dtype=np.float64)
        neural_distance = cdist(neural, neural)
        spatial_distance = cdist(spatial, spatial)

        upper = np.triu_indices(neural.shape[0], k=1)
        spatial_pairs = spatial_distance[upper]
        neural_pairs = neural_distance[upper]
        positive = spatial_pairs > 1e-9
        if positive.sum() < 10:
            return AnalysisResult(
                metrics={}, per_unit_metrics={}, figures={}, tables={},
                metadata={"conformal_isometry_skipped": "degenerate positions"},
            )
        correlation = float(np.corrcoef(spatial_pairs[positive], neural_pairs[positive])[0, 1])

        k = max(2, int(config.get("conformal_neighbor_k", 8)))
        local_scales = []
        for i in range(neural.shape[0]):
            order = np.argsort(spatial_distance[i])
            neighbors = [j for j in order if j != i and spatial_distance[i, j] > 1e-9][:k]
            if not neighbors:
                continue
            ratios = neural_distance[i, neighbors] / spatial_distance[i, neighbors]
            local_scales.append(ratios.mean())
        local_scales = np.asarray(local_scales, dtype=np.float64)
        if local_scales.size == 0 or local_scales.mean() <= 0:
            scale_cv = float("nan")
        else:
            scale_cv = float(local_scales.std() / local_scales.mean())
        return AnalysisResult(
            metrics={
                "metric_distance_correlation": correlation,
                "metric_scale_cv": scale_cv,
                "metric_local_scale_mean": float(local_scales.mean()) if local_scales.size else 0.0,
            },
            per_unit_metrics={},
            figures={},
            tables={},
        )
