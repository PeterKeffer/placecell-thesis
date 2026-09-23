"""Solstad-style synthetic grid-cell baseline for downstream RL."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .synthetic_place_cells import resolve_environment_xz_bounds

if TYPE_CHECKING:
    from placecell_research.config.downstream_schema import SyntheticGridCellsConfig

_DEFAULT_PERIOD_RATIO = 1.42
_PLANE_WAVE_OFFSETS_DEG = (0.0, 60.0, 120.0)


def _module_cell_counts(num_cells: int, num_modules: int) -> list[int]:
    """Split num_cells across num_modules as evenly as possible (remainder to the front)."""
    if num_cells < 1:
        raise ValueError(f"num_cells must be >= 1, got {num_cells}.")
    if num_modules < 1:
        raise ValueError(f"num_modules must be >= 1, got {num_modules}.")
    if num_modules > num_cells:
        raise ValueError(f"num_modules ({num_modules}) cannot exceed num_cells ({num_cells}).")
    base, remainder = divmod(int(num_cells), int(num_modules))
    return [base + (1 if module_index < remainder else 0) for module_index in range(num_modules)]


def _module_periods(
    bounds_xz: Sequence[float],
    num_modules: int,
    min_period: float,
    period_ratio: float,
) -> np.ndarray:
    """Grid periods per module (ascending), geometric by period_ratio."""
    if period_ratio <= 0.0:
        raise ValueError(f"period_ratio must be > 0, got {period_ratio}.")
    min_x, max_x, min_z, max_z = (float(value) for value in bounds_xz)
    exponents = np.arange(int(num_modules), dtype=np.float64)
    if min_period > 0.0:
        return float(min_period) * (period_ratio**exponents)
    largest_period = max(max_x - min_x, max_z - min_z)
    return largest_period * (period_ratio ** (exponents - (int(num_modules) - 1)))


def build_synthetic_grid_cell_bank(
    bounds_xz: Sequence[float],
    num_cells: int,
    num_modules: int,
    min_period: float,
    period_ratio: float,
    orientation_degrees: float,
    orientation_jitter_degrees: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed grid-cell bank: per-cell wave vectors and phase offsets."""
    counts = _module_cell_counts(num_cells, num_modules)
    periods = _module_periods(bounds_xz, num_modules, min_period, period_ratio)
    rng = np.random.default_rng(int(seed))
    base_orientation = np.radians(float(orientation_degrees))
    max_jitter = np.radians(float(orientation_jitter_degrees))
    plane_wave_offsets = np.radians(_PLANE_WAVE_OFFSETS_DEG)

    wave_vector_blocks: list[np.ndarray] = []
    phase_blocks: list[np.ndarray] = []
    for count, period in zip(counts, periods, strict=False):
        orientation = base_orientation
        if max_jitter > 0.0:
            orientation = base_orientation + rng.uniform(-max_jitter, max_jitter)
        angles = orientation + plane_wave_offsets
        magnitude = 4.0 * np.pi / (np.sqrt(3.0) * float(period))
        module_waves = magnitude * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        wave_vector_blocks.append(np.broadcast_to(module_waves, (count, 3, 2)).copy())
        phase_blocks.append(rng.uniform(0.0, float(period), size=(count, 2)))

    wave_vectors = np.concatenate(wave_vector_blocks, axis=0).astype(np.float32, copy=False)
    phases = np.concatenate(phase_blocks, axis=0).astype(np.float32, copy=False)
    return wave_vectors, phases


def build_grid_bank_from_config(
    grid_config: SyntheticGridCellsConfig, env_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve arena bounds then build the fixed grid bank (wave_vectors, phases) from a config."""
    bounds_xz = resolve_environment_xz_bounds(env_id, grid_config.bounds_xz)
    return build_synthetic_grid_cell_bank(
        bounds_xz,
        grid_config.num_cells,
        grid_config.num_modules,
        grid_config.min_period,
        grid_config.period_ratio,
        grid_config.orientation_degrees,
        grid_config.orientation_jitter_degrees,
        grid_config.seed,
    )


def synthetic_grid_cell_code(
    positions_xz: np.ndarray,
    wave_vectors: np.ndarray,
    phases: np.ndarray,
    normalization: str = "none",
) -> np.ndarray:
    """Three-plane-wave hexagonal grid code for a batch of (x, z) positions."""
    positions = np.asarray(positions_xz, dtype=np.float32).reshape(-1, 2)
    wave_vectors = np.asarray(wave_vectors, dtype=np.float32)
    phases = np.asarray(phases, dtype=np.float32)
    deltas = positions[:, None, :] - phases[None, :, :]
    projections = np.einsum("nid,bnd->bni", wave_vectors, deltas)
    code = (2.0 / 3.0) * ((1.0 / 3.0) * np.cos(projections).sum(axis=2) + 0.5)
    if normalization == "l2":
        norms = np.linalg.norm(code, axis=1, keepdims=True)
        code = code / np.where(norms <= 1e-8, 1.0, norms)
    elif normalization != "none":
        raise ValueError(f"unknown synthetic grid-cell normalization: {normalization!r}")
    return code.astype(np.float32, copy=False)


@dataclass
class GridCodeEncoder:
    """Deterministic grid-code encoder over (x, z) positions, sharing one fixed bank."""

    wave_vectors: np.ndarray
    phases: np.ndarray
    normalization: str = "l2"
    distance_metric: str = "l2"
    normalize_codes: bool = True
    success_threshold: float = 0.35

    @property
    def feature_dim(self) -> int:
        return int(np.asarray(self.wave_vectors).shape[0])

    def encode(self, position_xy) -> np.ndarray:
        position = np.asarray(position_xy, dtype=np.float32).reshape(1, 2)
        return synthetic_grid_cell_code(
            position, self.wave_vectors, self.phases, self.normalization
        ).reshape(-1)
