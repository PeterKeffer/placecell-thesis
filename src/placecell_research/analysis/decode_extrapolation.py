"""Spatial-extrapolation decoding: does the code generalise to UNVISITED regions?"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from ..numerics.rate_map_kernels import sample_valid_steps
from .base import AnalysisInput, AnalysisResult


def _ridge_r2(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    alpha: float,
) -> float:
    decoder = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    decoder.fit(train_features, train_targets)
    prediction = decoder.predict(test_features)
    residual = ((test_targets - prediction) ** 2).sum()
    total = ((test_targets - test_targets.mean(axis=0)) ** 2).sum()
    return float(1.0 - residual / total) if total > 0 else 0.0


@dataclass(slots=True)
class DecodeExtrapolationModule:
    """Train-on-one-region, test-on-the-unvisited-region position decoding."""

    name: str = "decode_extrapolation"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        max_samples = int(config.get("geometry_max_samples", 5000))
        sampled_steps = sample_valid_steps(
            analysis_input.representation,
            analysis_input.position_xy,
            analysis_input.valid_mask,
            max_samples=max_samples,
            random_seed=0,
        )
        features = sampled_steps.values
        targets = sampled_steps.positions
        if features.shape[0] < 50:
            return AnalysisResult(
                metrics={}, per_unit_metrics={}, figures={}, tables={},
                metadata={
                    "decode_extrapolation_skipped": "too few samples",
                    "sampled_valid_steps": int(features.shape[0]),
                    "total_valid_steps": sampled_steps.total_valid_steps,
                },
            )
        alpha = float(config.get("decode_ridge_alpha", 1e-3))
        axis = 0 if str(config.get("decode_extrapolation_axis", "x")) == "x" else 1
        split_value = float(np.median(targets[:, axis]))
        train_mask = targets[:, axis] < split_value
        test_mask = ~train_mask
        metadata: dict = {}
        if train_mask.sum() < 20 or test_mask.sum() < 20:
            extrapolation_r2 = float("nan")
            metadata["decode_extrapolation_note"] = "insufficient points on one side of the split"
        else:
            extrapolation_r2 = _ridge_r2(
                features[train_mask],
                targets[train_mask],
                features[test_mask],
                targets[test_mask],
                alpha,
            )
        order = np.random.default_rng(0).permutation(features.shape[0])
        cut = int(0.8 * len(order))
        train_index, test_index = order[:cut], order[cut:]
        interpolation_r2 = _ridge_r2(
            features[train_index],
            targets[train_index],
            features[test_index],
            targets[test_index],
            alpha,
        )
        gap = (
            float(interpolation_r2 - extrapolation_r2)
            if np.isfinite(extrapolation_r2)
            else float("nan")
        )
        return AnalysisResult(
            metrics={
                "extrapolation_r2": extrapolation_r2,
                "interpolation_r2": interpolation_r2,
                "extrapolation_gap": gap,
            },
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={
                **metadata,
                "sampled_valid_steps": int(features.shape[0]),
                "total_valid_steps": sampled_steps.total_valid_steps,
            },
        )
