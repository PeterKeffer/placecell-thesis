"""Fixed normalization statistics for frozen place-code features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class PlaceCodeStats:
    mean: np.ndarray
    std: np.ndarray
    active_rms: np.ndarray
    sample_count: int
    active_count: np.ndarray


class PlaceCodeStatsAccumulator:
    """Streaming accumulator for fixed place-code normalization statistics."""

    def __init__(self, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.feature_dim = int(feature_dim)
        self.sample_count = 0
        self.sum_values = np.zeros((self.feature_dim,), dtype=np.float64)
        self.sum_square = np.zeros((self.feature_dim,), dtype=np.float64)
        self.active_sum_square = np.zeros((self.feature_dim,), dtype=np.float64)
        self.active_count = np.zeros((self.feature_dim,), dtype=np.int64)

    def update(self, codes: np.ndarray) -> None:
        matrix = np.asarray(codes, dtype=np.float32)
        if matrix.size == 0:
            return
        if matrix.ndim != 2:
            raise ValueError(f"Expected place-code matrix with shape (N, D), got {matrix.shape}.")
        if int(matrix.shape[1]) != self.feature_dim:
            raise ValueError(
                f"Expected place-code width {self.feature_dim}, got {int(matrix.shape[1])}."
            )
        self.sample_count += int(matrix.shape[0])
        matrix64 = matrix.astype(np.float64)
        self.sum_values += matrix64.sum(axis=0)
        self.sum_square += np.square(matrix64).sum(axis=0)
        active_mask = np.abs(matrix) > _EPSILON
        self.active_count += active_mask.sum(axis=0)
        self.active_sum_square += np.where(active_mask, np.square(matrix64), 0.0).sum(axis=0)

    def to_stats(self) -> PlaceCodeStats:
        if self.sample_count == 0:
            raise ValueError("Cannot compute place-code stats from an empty matrix.")
        mean = self.sum_values / float(self.sample_count)
        variance = self.sum_square / float(self.sample_count) - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std <= _EPSILON, 1.0, std).astype(np.float32)

        active_rms = np.ones_like(std, dtype=np.float32)
        active_units = self.active_count > 0
        active_rms[active_units] = np.sqrt(
            self.active_sum_square[active_units] / self.active_count[active_units],
        ).astype(np.float32)
        active_rms = np.where(active_rms <= _EPSILON, 1.0, active_rms).astype(
            np.float32,
            copy=False,
        )
        return PlaceCodeStats(
            mean=mean.astype(np.float32),
            std=std,
            active_rms=active_rms,
            sample_count=int(self.sample_count),
            active_count=self.active_count.astype(np.int64, copy=False),
        )


def compute_place_code_stats(codes: np.ndarray) -> PlaceCodeStats:
    matrix = np.asarray(codes, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected place-code matrix with shape (N, D), got {matrix.shape}.")
    accumulator = PlaceCodeStatsAccumulator(feature_dim=int(matrix.shape[1]))
    accumulator.update(matrix)
    return accumulator.to_stats()


def save_place_code_stats(stats: PlaceCodeStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=stats.mean.astype(np.float32, copy=False),
        std=stats.std.astype(np.float32, copy=False),
        active_rms=stats.active_rms.astype(np.float32, copy=False),
        sample_count=np.asarray([stats.sample_count], dtype=np.int64),
        active_count=stats.active_count.astype(np.int64, copy=False),
    )


def load_place_code_stats(path: Path | str) -> PlaceCodeStats:
    payload = np.load(Path(path), allow_pickle=False)
    return PlaceCodeStats(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        active_rms=np.asarray(payload["active_rms"], dtype=np.float32),
        sample_count=int(np.asarray(payload["sample_count"]).reshape(-1)[0]),
        active_count=np.asarray(payload["active_count"], dtype=np.int64),
    )
