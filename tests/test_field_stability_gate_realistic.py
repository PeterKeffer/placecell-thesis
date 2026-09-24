"""The per-episode coverage gate, pinned to the values the WallGap recipe actually ships."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from wallgap_gate_fixtures import (
    REAL_MAX_FULL_GRID_COVERAGE,
    REAL_MAX_REACHABLE_COVERAGE,
    build_analysis_input,
    episode_bin_step_counts,
    kwinners_place_codes,
    realistic_wallgap_input,
    shipped_analysis_config,
    wallgap_bounds,
)

from placecell_research.analysis.field_stability import FieldStabilityMetricsModule
from placecell_research.analysis.per_episode_rate_maps import PerEpisodeRateMapsModule
from placecell_research.numerics.bin_maps import smooth_flat_bin_maps
from placecell_research.numerics.occupancy import reachable_bin_visited_fractions


def test_fixture_coverage_stays_at_real_wallgap_scale() -> None:
    """Guard the fixture itself: a roomier walk would re-create the false green."""
    config = shipped_analysis_config()
    analysis_input = realistic_wallgap_input()
    step_counts = episode_bin_step_counts(
        analysis_input.position_xy,
        num_bins_x=config["per_episode_num_bins_x"],
        num_bins_y=config["per_episode_num_bins_y"],
        bounds=wallgap_bounds(),
    )
    full_grid_coverage = (step_counts > 0).mean(axis=1)
    assert full_grid_coverage.max() <= REAL_MAX_FULL_GRID_COVERAGE
    reachable_coverage = reachable_bin_visited_fractions(step_counts)
    assert reachable_coverage.max() <= REAL_MAX_REACHABLE_COVERAGE


def test_shipped_wallgap_config_admits_real_scale_episodes(tmp_path: Path) -> None:
    """(a) Sparse k-winners episodes at the recipe's sigma 0.0 now clear the gate."""
    config = shipped_analysis_config()
    assert config["per_episode_smoothing_sigma"] == 0.0, "recipe no longer pins sigma 0.0"

    stability = FieldStabilityMetricsModule().run(realistic_wallgap_input(), tmp_path, config)
    assert stability.metrics["valid_stability_episodes"] > 0
    assert np.isfinite(stability.metrics["mean_field_center_drift_distance"])
    assert np.isfinite(stability.metrics["mean_field_area_cv"])

    rate_maps = PerEpisodeRateMapsModule().run(realistic_wallgap_input(), tmp_path, config)
    assert rate_maps.metrics["per_episode_rate_maps_qualifying_episodes"] > 0
    assert rate_maps.metrics["per_episode_rate_maps_units_rendered"] > 0


def test_gate_failure_reports_nan_not_zero(tmp_path: Path) -> None:
    """(b) An unmeasurable input says so, instead of passing for perfect stability."""
    config = shipped_analysis_config()
    bounds = wallgap_bounds()
    num_episodes = 64
    num_steps = int(config["per_episode_minimum_valid_steps"]) + 8
    parked_x = np.linspace(bounds[0][0] + 1.0, bounds[0][1] - 1.0, num_episodes, dtype=np.float32)
    parked_y = np.linspace(bounds[1][0] + 1.0, bounds[1][1] - 1.0, num_episodes, dtype=np.float32)
    positions = np.repeat(
        np.stack([parked_x, parked_y], axis=-1)[:, None, :],
        num_steps,
        axis=1,
    )
    codes = kwinners_place_codes(positions, num_units=16, active_units=2)
    valid_mask = np.ones(positions.shape[:2], dtype=bool)
    analysis_input = build_analysis_input(positions, codes, valid_mask)

    stability = FieldStabilityMetricsModule().run(analysis_input, tmp_path, config)
    assert stability.metrics["valid_stability_episodes"] == 0.0
    assert np.isnan(stability.metrics["mean_field_center_drift_distance"])
    assert np.isnan(stability.metrics["median_field_center_drift_distance"])
    assert np.isnan(stability.metrics["mean_field_area_cv"])

    rate_maps = PerEpisodeRateMapsModule().run(analysis_input, tmp_path, config)
    assert rate_maps.metrics["per_episode_rate_maps_qualifying_episodes"] == 0.0
    assert np.isnan(rate_maps.metrics["median_reanchoring_ratio"])
    assert "coverage" in rate_maps.metadata["reason"]


def test_gate_is_identical_at_every_smoothing_sigma(tmp_path: Path) -> None:
    """(c) Smoothing normalizes rates; it must not decide which episodes are eligible."""
    config = shipped_analysis_config()
    analysis_input = realistic_wallgap_input()
    num_bins_x = int(config["per_episode_num_bins_x"])
    num_bins_y = int(config["per_episode_num_bins_y"])

    step_counts = episode_bin_step_counts(
        analysis_input.position_xy,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=wallgap_bounds(),
    )
    smoothed_coverage = np.mean(
        smooth_flat_bin_maps(
            step_counts.astype(np.float32),
            num_bins_y=num_bins_y,
            num_bins_x=num_bins_x,
            smoothing_sigma=1.5,
        )
        > 0.0,
        axis=(1, 2),
    )
    raw_coverage = reachable_bin_visited_fractions(step_counts)
    halo_only_threshold = float(0.5 * (raw_coverage.max() + smoothed_coverage.min()))
    assert raw_coverage.max() < halo_only_threshold < smoothed_coverage.min()

    for minimum_visited_fraction in (
        float(config["per_episode_minimum_visited_fraction"]),
        halo_only_threshold,
    ):
        counts = []
        qualifying = []
        for smoothing_sigma in (0.0, 1.5):
            sigma_config = {
                **config,
                "per_episode_smoothing_sigma": smoothing_sigma,
                "per_episode_minimum_visited_fraction": minimum_visited_fraction,
            }
            stability = FieldStabilityMetricsModule().run(
                realistic_wallgap_input(),
                tmp_path,
                sigma_config,
            )
            rate_maps = PerEpisodeRateMapsModule().run(
                realistic_wallgap_input(),
                tmp_path,
                sigma_config,
            )
            counts.append(stability.metrics["valid_stability_episodes"])
            qualifying.append(rate_maps.metrics["per_episode_rate_maps_qualifying_episodes"])
        moved = f"gate moved with sigma at fraction {minimum_visited_fraction}"
        assert counts[0] == counts[1], moved
        assert qualifying[0] == qualifying[1], moved
