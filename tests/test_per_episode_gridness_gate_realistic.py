"""The per-episode gridness coverage gate, pinned to the values the WallGap recipe ships."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from wallgap_gate_fixtures import (
    REAL_EPISODE_STEPS,
    REAL_FORWARD_STEP,
    REAL_TURN_RADIANS_PER_STEP,
    build_analysis_input,
    episode_bin_step_counts,
    kwinners_place_codes,
    realistic_wallgap_input,
    shipped_analysis_config,
    wallgap_bounds,
)

from placecell_research.analysis.per_episode_gridness import PerEpisodeGridnessModule
from placecell_research.numerics.occupancy import reachable_bin_visited_fractions
from placecell_research.numerics.rate_map_kernels import (
    flatten_positions,
    infer_bounds,
)

PILLAR_INNER_RADIUS_FRACTION = 0.55
PILLAR_ARENA_EPISODES = 12
GATE_TEST_UNITS = 8


def _pillar_arena_walk(num_episodes: int, num_steps: int, *, seed: int) -> np.ndarray:
    """MiniWorld-scale walks in a WallGap-sized arena with an impassable central pillar."""
    generator = np.random.default_rng(seed)
    (min_x, max_x), (min_y, max_y) = wallgap_bounds()
    center_x, center_y = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    radius_x, radius_y = 0.5 * (max_x - min_x), 0.5 * (max_y - min_y)

    def is_walkable(x: float, y: float) -> bool:
        normalized_radius = float(np.hypot((x - center_x) / radius_x, (y - center_y) / radius_y))
        return PILLAR_INNER_RADIUS_FRACTION <= normalized_radius <= 1.0

    positions = np.empty((num_episodes, num_steps, 2), dtype=np.float32)
    for episode_index in range(num_episodes):
        x, y = center_x, center_y
        while not is_walkable(x, y):
            x = generator.uniform(min_x, max_x)
            y = generator.uniform(min_y, max_y)
        heading = generator.uniform(0.0, 2.0 * np.pi)
        for step_index in range(num_steps):
            positions[episode_index, step_index] = (x, y)
            heading += generator.normal(0.0, REAL_TURN_RADIANS_PER_STEP)
            next_x = x + REAL_FORWARD_STEP * np.cos(heading)
            next_y = y + REAL_FORWARD_STEP * np.sin(heading)
            if is_walkable(next_x, next_y):
                x, y = float(next_x), float(next_y)
    return positions


def _module_bin_step_counts(positions: np.ndarray, config: dict) -> np.ndarray:
    """Raw per-episode bin counts on the grid the module itself bins over."""
    valid_mask = np.ones(positions.shape[:2], dtype=bool)
    return episode_bin_step_counts(
        positions,
        num_bins_x=int(config["per_episode_num_bins_x"]),
        num_bins_y=int(config["per_episode_num_bins_y"]),
        bounds=infer_bounds(flatten_positions(positions, valid_mask)),
    )


def test_gate_counts_reachable_bins_not_the_whole_grid(tmp_path: Path) -> None:
    """The discriminating case: a threshold only the reachable denominator can clear."""
    config = shipped_analysis_config()
    positions = _pillar_arena_walk(PILLAR_ARENA_EPISODES, REAL_EPISODE_STEPS, seed=7)
    step_counts = _module_bin_step_counts(positions, config)

    full_grid_coverage = (step_counts > 0).mean(axis=1)
    reachable_coverage = reachable_bin_visited_fractions(step_counts)
    reachable_only_threshold = float(0.5 * (full_grid_coverage.max() + reachable_coverage.min()))
    assert full_grid_coverage.max() < reachable_only_threshold < reachable_coverage.min()

    codes = kwinners_place_codes(positions, num_units=GATE_TEST_UNITS, active_units=2)
    analysis_input = build_analysis_input(
        positions,
        codes,
        np.ones(positions.shape[:2], dtype=bool),
    )
    result = PerEpisodeGridnessModule().run(
        analysis_input,
        tmp_path,
        {**config, "per_episode_minimum_visited_fraction": reachable_only_threshold},
    )
    assert result.metrics["per_episode_gridness_episodes_used"] == PILLAR_ARENA_EPISODES
    assert result.metrics["per_episode_gridness_episodes_skipped_sparse"] == 0
    assert result.metrics["per_episode_gridness_episodes_skipped_short"] == 0


def test_shipped_wallgap_config_admits_real_scale_episodes(tmp_path: Path) -> None:
    """The shipped 0.05 admits real WallGap episodes rather than emptying the analysis."""
    config = shipped_analysis_config()
    analysis_input = realistic_wallgap_input(num_units=GATE_TEST_UNITS)

    result = PerEpisodeGridnessModule().run(analysis_input, tmp_path, config)

    assert result.metrics["per_episode_gridness_episodes_used"] > 0
    assert np.isfinite(result.metrics["mean_per_episode_gridness"])
    assert np.all(np.isfinite(result.per_unit_metrics["per_episode_gridness"]))


def test_episodes_parked_in_single_bins_are_all_rejected(tmp_path: Path) -> None:
    """The failure path: no episode sees enough of the arena, and the counters say so."""
    config = shipped_analysis_config()
    (min_x, max_x), (min_y, max_y) = wallgap_bounds()
    lattice_side = 8
    num_steps = int(config["per_episode_minimum_valid_steps"]) + 8
    lattice_x, lattice_y = np.meshgrid(
        np.linspace(min_x + 1.0, max_x - 1.0, lattice_side, dtype=np.float32),
        np.linspace(min_y + 1.0, max_y - 1.0, lattice_side, dtype=np.float32),
    )
    parked = np.stack([lattice_x.reshape(-1), lattice_y.reshape(-1)], axis=-1)
    positions = np.repeat(parked[:, None, :], num_steps, axis=1)
    codes = kwinners_place_codes(positions, num_units=GATE_TEST_UNITS, active_units=2)
    analysis_input = build_analysis_input(
        positions,
        codes,
        np.ones(positions.shape[:2], dtype=bool),
    )

    result = PerEpisodeGridnessModule().run(analysis_input, tmp_path, config)

    assert result.metrics["per_episode_gridness_episodes_used"] == 0
    assert result.metrics["per_episode_gridness_episodes_skipped_sparse"] == lattice_side**2
    assert result.metrics["per_episode_gridness_episodes_skipped_short"] == 0


def test_gate_is_identical_at_every_smoothing_sigma(tmp_path: Path) -> None:
    """Smoothing normalizes rates; it must not decide which episodes are eligible."""
    config = shipped_analysis_config()
    assert config["per_episode_smoothing_sigma"] == 0.0, "recipe no longer pins sigma 0.0"

    episodes_used = [
        PerEpisodeGridnessModule()
        .run(
            realistic_wallgap_input(num_units=GATE_TEST_UNITS),
            tmp_path,
            {**config, "per_episode_smoothing_sigma": smoothing_sigma},
        )
        .metrics["per_episode_gridness_episodes_used"]
        for smoothing_sigma in (0.0, 1.5)
    ]
    assert episodes_used[0] == episodes_used[1]
