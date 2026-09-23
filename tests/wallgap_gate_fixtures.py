"""Shared fixtures for the per-episode coverage-gate regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay
from placecell_research.config import load_experiment_config
from placecell_research.numerics.rate_map_kernels import compute_spatial_bin_assignments

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WALLGAP_RECIPE_CONFIG = REPOSITORY_ROOT / "configs" / "experiment" / "wallgap.yaml"
WALLGAP_ENVIRONMENT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "environment" / "miniworld_wallgap_asym_large.yaml"
)

REAL_EPISODE_STEPS = 2048
REAL_FORWARD_STEP = 0.26
REAL_TURN_RADIANS_PER_STEP = 1.0
REAL_MAX_FULL_GRID_COVERAGE = 0.16
REAL_MAX_REACHABLE_COVERAGE = 0.20


def shipped_analysis_config() -> dict:
    """Analysis knobs as a WallGap run resolves them."""
    return load_experiment_config(WALLGAP_RECIPE_CONFIG, []).to_dict()["analysis"]


def wallgap_env_id() -> str:
    return str(yaml.safe_load(WALLGAP_ENVIRONMENT_CONFIG.read_text())["env_id"])


def wallgap_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    return overlay_bounds(resolve_world_overlay(wallgap_env_id()))


def miniworld_scale_walk(
    num_episodes: int,
    num_steps: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    seed: int,
) -> np.ndarray:
    """Momentum-carrying random walks at MiniWorld's step scale, clipped to the arena."""
    generator = np.random.default_rng(seed)
    (min_x, max_x), (min_y, max_y) = bounds
    positions = np.empty((num_episodes, num_steps, 2), dtype=np.float32)
    for episode_index in range(num_episodes):
        x = generator.uniform(min_x, max_x)
        y = generator.uniform(min_y, max_y)
        heading = generator.uniform(0.0, 2.0 * np.pi)
        for step_index in range(num_steps):
            positions[episode_index, step_index] = (x, y)
            heading += generator.normal(0.0, REAL_TURN_RADIANS_PER_STEP)
            x = float(np.clip(x + REAL_FORWARD_STEP * np.cos(heading), min_x, max_x))
            y = float(np.clip(y + REAL_FORWARD_STEP * np.sin(heading), min_y, max_y))
    return positions


def kwinners_place_codes(
    positions: np.ndarray,
    *,
    num_units: int,
    active_units: int,
) -> np.ndarray:
    """Sparse place codes with exactly active_units non-zero units per step."""
    generator = np.random.default_rng(0)
    (min_x, max_x) = positions[..., 0].min(), positions[..., 0].max()
    (min_y, max_y) = positions[..., 1].min(), positions[..., 1].max()
    centers = np.stack(
        [
            generator.uniform(min_x, max_x, size=num_units),
            generator.uniform(min_y, max_y, size=num_units),
        ],
        axis=-1,
    ).astype(np.float32)
    squared_distances = np.square(positions[..., None, :] - centers).sum(axis=-1)
    tuning = np.exp(-squared_distances / (2.0 * 6.0**2)).astype(np.float32)
    keep_from = np.argpartition(tuning, -active_units, axis=-1)[..., -active_units:]
    codes = np.zeros_like(tuning)
    np.put_along_axis(codes, keep_from, np.take_along_axis(tuning, keep_from, axis=-1), axis=-1)
    return codes


def build_analysis_input(
    positions: np.ndarray,
    codes: np.ndarray,
    valid_mask: np.ndarray,
) -> AnalysisInput:
    return AnalysisInput(
        representation=codes,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        label="encoder place cells",
        split_name="test",
        metadata={"env_id": wallgap_env_id()},
    )


def realistic_wallgap_input(num_episodes: int = 12, *, num_units: int = 48) -> AnalysisInput:
    positions = miniworld_scale_walk(
        num_episodes,
        REAL_EPISODE_STEPS,
        wallgap_bounds(),
        seed=7,
    )
    codes = kwinners_place_codes(positions, num_units=num_units, active_units=4)
    return build_analysis_input(positions, codes, np.ones(positions.shape[:2], dtype=bool))


def episode_bin_step_counts(
    positions: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    num_episodes, num_steps = positions.shape[:2]
    total_bins = num_bins_x * num_bins_y
    linear_bins, _, _, _ = compute_spatial_bin_assignments(
        positions.reshape(-1, 2),
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    episode_ids = np.repeat(np.arange(num_episodes), num_steps)
    return np.bincount(
        episode_ids * total_bins + linear_bins,
        minlength=num_episodes * total_bins,
    ).reshape(num_episodes, total_bins)
