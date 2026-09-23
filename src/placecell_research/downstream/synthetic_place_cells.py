"""Sorscher-style synthetic place-cell baseline for downstream RL."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from placecell_research.config.downstream_schema import SyntheticPlaceCellsConfig

_SORSCHER_SIGMA_FRACTION = 0.2 / 2.2

_KNOWN_ENVIRONMENT_XZ_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "MiniWorld-WallGapAsymLarge-v0": (-24.0, 24.0, -30.0, 36.0),
    "simple": (0.0, 9.0, 0.0, 5.0),
    "museum-gallery": (0.0, 43.0, 0.0, 25.0),
    "warren": (0.0, 44.0, 0.0, 27.0),
    "warren-hard": (0.0, 44.0, 0.0, 27.0),
    "cave": (0.0, 53.0, 0.0, 35.0),
    "cave-landmarks": (0.0, 53.0, 0.0, 35.0),
}


def resolve_environment_xz_bounds(
    env_id: str,
    explicit_bounds_xz: Sequence[float] | None = None,
) -> tuple[float, float, float, float]:
    """Resolve the whole-environment (min_x, max_x, min_z, max_z) box."""
    if explicit_bounds_xz:
        bounds = tuple(float(value) for value in explicit_bounds_xz)
        if len(bounds) != 4:
            raise ValueError(
                "synthetic_place_cells.bounds_xz must be [min_x, max_x, min_z, max_z], "
                f"got {len(bounds)} values."
            )
        return bounds
    if env_id in _KNOWN_ENVIRONMENT_XZ_BOUNDS:
        return _KNOWN_ENVIRONMENT_XZ_BOUNDS[env_id]
    raise ValueError(
        f"No known (x, z) bounds for env_id '{env_id}'. "
        "Set observation.synthetic_place_cells.bounds_xz = [min_x, max_x, min_z, max_z]."
    )


def resolve_environment_reachable_regions(
    env_id: str,
) -> tuple[tuple[float, float, float, float], ...] | None:
    """Disjoint (min_x, max_x, min_z, max_z) rectangles the agent can actually occupy."""
    if env_id == "MiniWorld-WallGapAsymLarge-v0":
        from placecell_research.envs.miniworld_wallgap_asym_large import (
            _FULL_ROOM_BOUNDS_BY_NAME,
        )

        return tuple(
            tuple(float(value) for value in bounds)  # type: ignore[misc]
            for bounds in _FULL_ROOM_BOUNDS_BY_NAME.values()
        )
    return None


def build_synthetic_place_cell_centers(
    bounds_xz: Sequence[float],
    num_cells: int,
    seed: int,
    reachable_regions: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Sample num_cells place-field centers with a fixed seed."""
    if num_cells < 1:
        raise ValueError(f"num_cells must be >= 1, got {num_cells}.")
    rng = np.random.default_rng(int(seed))
    if reachable_regions is None or len(reachable_regions) == 0:
        min_x, max_x, min_z, max_z = (float(value) for value in bounds_xz)
        centers = rng.uniform(low=[min_x, min_z], high=[max_x, max_z], size=(int(num_cells), 2))
        return centers.astype(np.float32, copy=False)
    rectangles = np.asarray(reachable_regions, dtype=np.float64).reshape(-1, 4)
    areas = (rectangles[:, 1] - rectangles[:, 0]) * (rectangles[:, 3] - rectangles[:, 2])
    if np.any(areas <= 0.0):
        raise ValueError(f"reachable_regions must all have positive area, got {rectangles}.")
    chosen = rng.choice(len(rectangles), size=int(num_cells), p=areas / areas.sum())
    unit_square = rng.uniform(size=(int(num_cells), 2))
    picked = rectangles[chosen]
    centers = np.stack(
        [
            picked[:, 0] + unit_square[:, 0] * (picked[:, 1] - picked[:, 0]),
            picked[:, 2] + unit_square[:, 1] * (picked[:, 3] - picked[:, 2]),
        ],
        axis=1,
    )
    return centers.astype(np.float32, copy=False)


def resolve_synthetic_sigmas(
    bounds_xz: Sequence[float],
    sigma_center: float,
    surround_scale: float,
) -> tuple[float, float]:
    """Resolve (sigma_center, sigma_surround) in environment units."""
    min_x, max_x, min_z, max_z = (float(value) for value in bounds_xz)
    if sigma_center <= 0.0:
        characteristic_size = 0.5 * ((max_x - min_x) + (max_z - min_z))
        sigma_center = _SORSCHER_SIGMA_FRACTION * characteristic_size
    sigma_surround = float(surround_scale) * float(sigma_center)
    return float(sigma_center), sigma_surround


def build_place_bank_from_config(
    place_config: SyntheticPlaceCellsConfig, env_id: str
) -> tuple[np.ndarray, float, float]:
    """Resolve arena bounds then build (centers, sigma_center, sigma_surround) from a config."""
    bounds_xz = resolve_environment_xz_bounds(env_id, place_config.bounds_xz)
    reachable_regions = (
        None if place_config.bounds_xz else resolve_environment_reachable_regions(env_id)
    )
    centers = build_synthetic_place_cell_centers(
        bounds_xz, place_config.num_cells, place_config.seed, reachable_regions
    )
    sigma_center, sigma_surround = resolve_synthetic_sigmas(
        bounds_xz, place_config.sigma_center, place_config.surround_scale
    )
    return centers, sigma_center, sigma_surround


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def synthetic_place_cell_code(
    positions_xz: np.ndarray,
    centers_xz: np.ndarray,
    sigma_center: float,
    sigma_surround: float,
    normalization: str = "none",
) -> np.ndarray:
    """Difference-of-softmax place-cell code for a batch of (x, z) positions."""
    positions = np.asarray(positions_xz, dtype=np.float32).reshape(-1, 2)
    centers = np.asarray(centers_xz, dtype=np.float32)
    deltas = positions[:, None, :] - centers[None, :, :]
    squared_distance = np.square(deltas).sum(axis=2)
    center = _softmax_rows(-squared_distance / (2.0 * sigma_center * sigma_center))
    surround = _softmax_rows(-squared_distance / (2.0 * sigma_surround * sigma_surround))
    code = center - surround
    if normalization == "l2":
        norms = np.linalg.norm(code, axis=1, keepdims=True)
        code = code / np.where(norms <= 1e-8, 1.0, norms)
    elif normalization != "none":
        raise ValueError(f"unknown synthetic place-cell normalization: {normalization!r}")
    return code.astype(np.float32, copy=False)


@dataclass
class SyntheticPlaceCodeEncoder:
    """Deterministic place-code encoder over (x, z) positions, sharing one fixed bank."""

    centers: np.ndarray
    sigma_center: float
    sigma_surround: float
    normalization: str = "l2"
    distance_metric: str = "l2"
    normalize_codes: bool = True
    success_threshold: float = 0.35

    @property
    def feature_dim(self) -> int:
        return int(np.asarray(self.centers).shape[0])

    def encode(self, position_xy) -> np.ndarray:
        position = np.asarray(position_xy, dtype=np.float32).reshape(1, 2)
        return synthetic_place_cell_code(
            position, self.centers, self.sigma_center, self.sigma_surround, self.normalization
        ).reshape(-1)


class PlaceCodeGoalCodebook:
    """A coordinate-free bank of place-code goals recorded during offline exploration."""

    def __init__(self, codes: np.ndarray) -> None:
        self.codes = np.asarray(codes, dtype=np.float32)
        if self.codes.ndim != 2 or self.codes.shape[0] < 1:
            raise ValueError(
                f"codebook codes must be (num_entries, feature_dim), got {self.codes.shape}."
            )

    @property
    def feature_dim(self) -> int:
        return int(self.codes.shape[1])

    @classmethod
    def load(cls, path: str | Path) -> PlaceCodeGoalCodebook:
        codebook_path = Path(path)
        if not codebook_path.exists():
            raise FileNotFoundError(f"Goal codebook not found: {codebook_path}")
        with np.load(codebook_path) as payload:
            if "codes" not in payload:
                raise ValueError(
                    f"Goal codebook '{codebook_path}' must contain a 'codes' array, "
                    f"found {list(payload.files)!r}."
                )
            codes = np.asarray(payload["codes"], dtype=np.float32)
        return cls(codes=codes)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        index = int(rng.integers(self.codes.shape[0]))
        return self.codes[index].copy()
