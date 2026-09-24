"""Cheap per-validation probes for catching representation quality early."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decode import linear_decode_position

_EPS = 1e-12


@dataclass
class DecodeGap:
    """Position decodability before vs after the sparsifier."""

    dense_r2: float
    sparse_r2: float
    gap: float


def _decode_r2(
    codes: np.ndarray, targets: np.ndarray, train_fraction: float, alpha: float
) -> float:
    result = linear_decode_position(
        codes,
        targets,
        train_fraction=train_fraction,
        alpha=alpha,
        include_shuffle=False,
    )
    return float(result.r2)


def heading_decodability(
    codes: np.ndarray, heading: np.ndarray, train_fraction: float = 0.8, alpha: float = 1e-3
) -> float:
    """R^2 of a ridge probe predicting (sin, cos) of heading from the code."""
    heading_targets = np.stack([np.sin(heading), np.cos(heading)], axis=-1)
    return _decode_r2(codes, heading_targets, train_fraction, alpha)


def dense_sparse_decode_gap(
    dense_codes: np.ndarray,
    sparse_codes: np.ndarray,
    positions: np.ndarray,
    train_fraction: float = 0.8,
    alpha: float = 1e-3,
) -> DecodeGap:
    """Position-decode R^2 from the pre-sparsifier code minus the post-sparsifier code."""
    dense_r2 = _decode_r2(dense_codes, positions, train_fraction, alpha)
    sparse_r2 = _decode_r2(sparse_codes, positions, train_fraction, alpha)
    return DecodeGap(dense_r2=dense_r2, sparse_r2=sparse_r2, gap=dense_r2 - sparse_r2)


def participation_ratio(codes: np.ndarray) -> float:
    """Effective number of active dimensions: (sum eig)^2 / sum(eig^2)."""
    if codes.shape[0] < 2:
        return float("nan")
    centered = codes - codes.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / (codes.shape[0] - 1)
    trace = float(np.trace(covariance))
    frobenius_squared = float((covariance * covariance).sum())
    if frobenius_squared <= _EPS:
        return float("nan")
    return trace * trace / frobenius_squared
