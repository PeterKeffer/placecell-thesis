"""Manifold topology via persistent homology (Betti numbers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import flatten_valid_steps
from .base import AnalysisInput, AnalysisResult


def _significant_count(lifetimes: np.ndarray, fraction: float) -> int:
    finite = lifetimes[np.isfinite(lifetimes)]
    if finite.size == 0:
        return 0
    longest = float(finite.max())
    if longest <= 0:
        return 0
    return int((finite > fraction * longest).sum())


@dataclass(slots=True)
class ManifoldTopologyModule:
    """Persistent-homology Betti numbers of the population activity manifold."""

    name: str = "manifold_topology"
    cost_tier: str = "heavy"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        try:
            from ripser import ripser
        except ImportError:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={"manifold_topology_skipped": "ripser not installed"},
            )
        flat, _positions = flatten_valid_steps(
            analysis_input.representation, analysis_input.position_xy, analysis_input.valid_mask
        )
        if flat.shape[0] < 50:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={"manifold_topology_skipped": "too few samples"},
            )
        max_points = int(config.get("topology_max_points", 700))
        rng = np.random.default_rng(0)
        points = np.asarray(flat, dtype=np.float64)
        if points.shape[0] > max_points:
            points = points[rng.choice(points.shape[0], max_points, replace=False)]
        std = points.std(axis=0)
        points = points[:, std > 1e-9]
        if points.shape[1] < 2:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={"manifold_topology_skipped": "degenerate (no live units)"},
            )
        points = (points - points.mean(axis=0)) / points.std(axis=0)
        diagrams = ripser(points, maxdim=2)["dgms"]
        fraction = float(config.get("topology_lifetime_fraction", 0.5))
        lifetimes = [(diagram[:, 1] - diagram[:, 0]) for diagram in diagrams]
        betti = [_significant_count(life, fraction) for life in lifetimes]
        h1_finite = lifetimes[1][np.isfinite(lifetimes[1])] if len(lifetimes) > 1 else np.array([])
        h1_sorted = np.sort(h1_finite)[::-1]
        return AnalysisResult(
            metrics={
                "betti_0": float(betti[0]) if len(betti) > 0 else 0.0,
                "betti_1": float(betti[1]) if len(betti) > 1 else 0.0,
                "betti_2": float(betti[2]) if len(betti) > 2 else 0.0,
                "h1_top1_lifetime": float(h1_sorted[0]) if h1_sorted.size > 0 else 0.0,
                "h1_top2_lifetime": float(h1_sorted[1]) if h1_sorted.size > 1 else 0.0,
            },
            per_unit_metrics={},
            figures={},
            tables={},
        )
