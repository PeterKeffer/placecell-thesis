"""Code timescale: how fast does a representation decorrelate?"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import AnalysisInput, AnalysisResult

DEFAULT_LAGS = (1, 2, 5, 10, 20, 50, 100)
_INVERSE_E = 0.36787944117144233


def autocorrelation_by_lag(
    codes: np.ndarray, valid: np.ndarray, lags: tuple[int, ...]
) -> dict[int, float]:
    """Normalised autocorrelation r(k) of the time-centered code, per lag."""
    codes = codes.astype(np.float32, copy=False)
    valid = valid.astype(bool, copy=False)
    weights = valid.astype(np.float32)[..., None]
    totals = weights.sum(axis=1, keepdims=True)
    masked_codes = np.where(valid[..., None], codes, 0.0)
    mean = masked_codes.sum(axis=1, keepdims=True) / np.clip(totals, 1.0, None)
    centered = np.where(valid[..., None], codes - mean, 0.0)
    energy = (centered * centered).sum(axis=2)
    denominator = float(energy.sum() / max(int(valid.sum()), 1))
    if denominator <= 0.0:
        return {}
    time = centered.shape[1]
    result: dict[int, float] = {}
    for lag in lags:
        if lag >= time:
            break
        pair_is_valid = valid[:, : time - lag] & valid[:, lag:]
        pair_count = int(pair_is_valid.sum())
        if pair_count == 0:
            continue
        paired = (centered[:, : time - lag] * centered[:, lag:]).sum(axis=2)
        numerator = float(paired[pair_is_valid].sum() / pair_count)
        result[lag] = numerator / denominator
    return result


def decorrelation_time(curve: dict[int, float]) -> float:
    """First lag whose autocorrelation drops below 1/e; NaN if it never does within the lags."""
    for lag in sorted(curve):
        if curve[lag] < _INVERSE_E:
            return float(lag)
    return float("nan")


@dataclass(slots=True)
class CodeTimescaleModule:
    """Autocorrelation time of a per-step code."""

    name: str = "code_timescale"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(
        self, analysis_input: AnalysisInput, output_dir: Path, config: dict[str, Any]
    ) -> AnalysisResult:
        codes = np.asarray(analysis_input.representation)
        if codes.ndim != 3:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "code_timescale_skipped": True,
                    "code_timescale_skip_reason": (
                        f"needs (episodes, time, units), got {codes.shape}"
                    ),
                },
            )
        lags = tuple(config.get("code_timescale_lags", DEFAULT_LAGS))
        valid = np.asarray(analysis_input.valid_mask, dtype=bool)
        curve = autocorrelation_by_lag(codes, valid, lags)
        if not curve:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "code_timescale_skipped": True,
                    "code_timescale_skip_reason": "code has no time-varying component",
                },
            )
        metrics = {f"code_autocorrelation_lag_{lag}": value for lag, value in curve.items()}
        metrics["code_t_dec"] = decorrelation_time(curve)
        return AnalysisResult(metrics=metrics, per_unit_metrics={}, figures={}, tables={})
