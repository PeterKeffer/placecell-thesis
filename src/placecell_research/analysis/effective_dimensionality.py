"""Effective dimensionality (participation ratio) of a representation's population code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import flatten_valid_steps
from .base import AnalysisInput, AnalysisResult


@dataclass(slots=True)
class EffectiveDimensionalityModule:
    """Participation ratio of the PCA spectrum of the population code."""

    name: str = "effective_dimensionality"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        flat, _positions = flatten_valid_steps(
            analysis_input.representation, analysis_input.position_xy, analysis_input.valid_mask
        )
        if flat.shape[0] < 2 or flat.shape[1] < 1:
            return AnalysisResult(
                metrics={}, per_unit_metrics={}, figures={}, tables={},
                metadata={"effective_dim_skipped": "too few samples"},
            )
        max_samples = int(config.get("geometry_max_samples", 5000))
        rows = np.asarray(flat, dtype=np.float64)
        if rows.shape[0] > max_samples:
            index = np.random.default_rng(0).choice(rows.shape[0], max_samples, replace=False)
            rows = rows[index]
        centered = rows - rows.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = singular_values.astype(np.float64) ** 2
        total = eigenvalues.sum()
        if total <= 0:
            return AnalysisResult(
                metrics={"participation_ratio": 0.0}, per_unit_metrics={}, figures={}, tables={},
                metadata={"effective_dim_skipped": "zero variance"},
            )
        participation_ratio = float(total ** 2 / (eigenvalues ** 2).sum())
        fraction = eigenvalues / total
        cumulative = np.cumsum(fraction)
        return AnalysisResult(
            metrics={
                "participation_ratio": participation_ratio,
                "variance_fraction_top2": float(fraction[:2].sum()),
                "variance_fraction_top6": float(fraction[:6].sum()),
                "dims_for_90pct_variance": float(int(np.searchsorted(cumulative, 0.9)) + 1),
                "num_units": float(flat.shape[1]),
            },
            per_unit_metrics={},
            figures={},
            tables={},
        )
