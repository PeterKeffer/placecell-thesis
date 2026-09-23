"""Sparsity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import AnalysisInput, AnalysisResult


def _masked_sparsity_scores(
    representation: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, float]:
    values = representation.astype(np.float32, copy=False)
    valid = (
        np.ones(values.shape[:2], dtype=bool)
        if valid_mask is None
        else valid_mask.astype(bool, copy=False)
    )
    valid_count = int(valid.sum())
    if valid_count == 0:
        return np.zeros((values.shape[-1],), dtype=np.float32), 0.0

    valid_weights = valid.astype(np.float32, copy=False)
    unit_sum = np.einsum("btd,bt->d", values, valid_weights, optimize=True)
    unit_square_sum = np.einsum("btd,btd,bt->d", values, values, valid_weights, optimize=True)
    unit_mean = unit_sum / float(valid_count)
    unit_mean_square = unit_square_sum / float(valid_count)
    lifetime_scores = np.divide(
        np.square(unit_mean, dtype=np.float32),
        unit_mean_square,
        out=np.zeros_like(unit_mean, dtype=np.float32),
        where=unit_mean_square > float(epsilon),
    )

    sample_mean = values.mean(axis=-1)
    sample_mean_square = np.einsum("btd,btd->bt", values, values, optimize=True) / float(
        values.shape[-1]
    )
    valid_samples = valid & (sample_mean_square > float(epsilon))
    population_values = np.divide(
        np.square(sample_mean, dtype=np.float32),
        sample_mean_square,
        out=np.zeros_like(sample_mean, dtype=np.float32),
        where=sample_mean_square > float(epsilon),
    )
    population_score = (
        float(population_values[valid_samples].mean()) if np.any(valid_samples) else 0.0
    )
    return lifetime_scores.astype(np.float32, copy=False), population_score


@dataclass(slots=True)
class SparsityModule:
    """Lifetime and population sparsity metrics."""

    name: str = "sparsity"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        lifetime_scores, population_score = _masked_sparsity_scores(
            analysis_input.representation,
            analysis_input.valid_mask,
        )
        return AnalysisResult(
            metrics={
                "mean_lifetime_activity_fraction": (
                    float(lifetime_scores.mean()) if len(lifetime_scores) else 0.0
                ),
                "population_activity_fraction": float(population_score),
            },
            per_unit_metrics={"lifetime_activity_fraction": lifetime_scores},
            figures={},
            tables={},
        )
