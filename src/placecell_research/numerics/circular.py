"""Circular statistics on binned tuning curves."""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def circular_vector(rates: np.ndarray, heading_vectors: np.ndarray) -> tuple[float, float]:
    total_rate = float(np.sum(rates))
    if total_rate <= _EPS:
        return float("nan"), float("nan")
    vector = np.sum(rates * heading_vectors) / total_rate
    return float(np.abs(vector)), float(np.angle(vector))


def peak_to_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    mean_value = float(finite.mean())
    if mean_value <= _EPS:
        return float("nan")
    return float(finite.max() / mean_value)
