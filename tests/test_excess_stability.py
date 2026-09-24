from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.excess_stability import (
    ExcessStabilityModule,
    compute_twin_psi,
)
from placecell_research.numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    infer_bounds,
)

_NUM_BINS = 12
_CONFIG = {
    "num_bins_x": _NUM_BINS,
    "num_bins_y": _NUM_BINS,
    "smoothing_sigma": 0.0,
    "min_occupancy": 1e-6,
    "excess_stability_lag": 16,
}


def _analysis_input(representation, positions, heading=None, kinematics=None) -> AnalysisInput:
    valid_mask = np.ones(representation.shape[:2], dtype=bool)
    return AnalysisInput(
        representation=representation.astype(np.float32),
        position_xy=positions.astype(np.float32),
        heading=heading,
        kinematics=kinematics,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="test",
    )


def _random_walk_positions(
    rng: np.random.Generator, num_episodes: int, num_steps: int
) -> np.ndarray:
    positions = np.zeros((num_episodes, num_steps, 2), dtype=np.float32)
    positions[:, 0] = rng.uniform(2.0, 18.0, size=(num_episodes, 2))
    steps = rng.normal(scale=0.8, size=(num_episodes, num_steps - 1, 2))
    positions[:, 1:] = np.clip(positions[:, :1] + np.cumsum(steps, axis=1), 0.0, 20.0)
    return positions


def _linear_bins(positions: np.ndarray) -> np.ndarray:
    linear_bins, _, _, _ = compute_spatial_bin_assignments(
        positions.reshape(-1, 2), num_bins_x=_NUM_BINS, num_bins_y=_NUM_BINS
    )
    return linear_bins


def _bin_table_activity(
    rng: np.random.Generator, positions: np.ndarray, num_units: int
) -> np.ndarray:
    """Non-negative activity that is an exact function of the module's spatial bin."""
    table = rng.uniform(0.1, 1.0, size=(_NUM_BINS * _NUM_BINS, num_units)).astype(np.float32)
    return table[_linear_bins(positions)].reshape(*positions.shape[:2], num_units)


def test_memoryless_spatial_cells_have_ratio_one(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    positions = _random_walk_positions(rng, num_episodes=8, num_steps=256)
    representation = _bin_table_activity(rng, positions, num_units=5)

    result = ExcessStabilityModule().run(
        _analysis_input(representation, positions), tmp_path, _CONFIG
    )

    ratios = result.per_unit_metrics["excess_stability_ratio"]
    assert np.all(np.isfinite(ratios))
    assert np.allclose(ratios, 1.0, atol=1e-5)
    assert abs(result.metrics["median_excess_stability_ratio"] - 1.0) < 1e-5
    assert result.metrics["fraction_units_ratio_below_0_9"] == 0.0
    assert result.metrics["fraction_units_ratio_above_1_1"] == 0.0
    assert result.metrics["fraction_units_dropped"] == 0.0
    assert np.isnan(result.metrics["median_excess_vs_pose_twin"])
    assert np.isnan(result.metrics["median_pose_vs_xy_twin"])
    assert np.isnan(result.metrics["rotation_variance_fraction"])
    for sweep_bins in (20, 40):
        assert np.isfinite(result.metrics[f"median_excess_stability_ratio_bins{sweep_bins}"])
    assert result.metadata["held_out_twin"] is True
    assert result.figures["excess_stability_histogram"].exists()


def test_temporal_integrators_fall_below_ratio_0_9(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    positions = _random_walk_positions(rng, num_episodes=8, num_steps=256)
    spatial_signal = _bin_table_activity(rng, positions, num_units=6)
    smoothing_factor = 0.02
    integrated = np.zeros_like(spatial_signal)
    integrated[:, 0] = spatial_signal[:, 0]
    for step in range(1, spatial_signal.shape[1]):
        integrated[:, step] = (1.0 - smoothing_factor) * integrated[
            :, step - 1
        ] + smoothing_factor * spatial_signal[:, step]

    result = ExcessStabilityModule().run(_analysis_input(integrated, positions), tmp_path, _CONFIG)

    assert result.metrics["median_excess_stability_ratio"] < 0.9
    assert result.metrics["fraction_units_ratio_below_0_9"] >= 0.5


def test_pose_twin_isolates_heading_component(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    num_episodes, num_steps, num_units = 8, 256, 5
    positions = _random_walk_positions(rng, num_episodes, num_steps)
    heading = rng.uniform(0.0, 2.0 * np.pi, size=(num_episodes, num_steps)).astype(np.float32)
    linear_bins = _linear_bins(positions)
    ramp = ((linear_bins % _NUM_BINS) + (linear_bins // _NUM_BINS)) / (2.0 * (_NUM_BINS - 1))
    spatial = np.repeat(
        (0.2 + ramp).astype(np.float32).reshape(num_episodes, num_steps, 1), num_units, axis=-1
    )
    quadrants = (np.floor((heading % (2.0 * np.pi)) / (np.pi / 2.0)).astype(int)) % 4
    quadrant_table = rng.uniform(0.5, 1.5, size=(4, num_units)).astype(np.float32)
    representation = spatial + quadrant_table[quadrants]

    result = ExcessStabilityModule().run(
        _analysis_input(representation, positions, heading=heading), tmp_path, _CONFIG
    )

    assert abs(result.metrics["median_excess_vs_pose_twin"] - 1.0) < 1e-4
    assert result.metrics["median_pose_vs_xy_twin"] > 1.1
    assert result.metrics["median_excess_stability_ratio"] > 1.1


def test_held_out_twin_removes_in_sample_memorization_bias() -> None:
    rng = np.random.default_rng(4)
    num_episodes, num_steps = 6, 120
    positions = _random_walk_positions(rng, num_episodes, num_steps)
    spatial = _bin_table_activity(rng, positions, num_units=6)
    noisy = np.clip(
        spatial + rng.normal(scale=0.5, size=spatial.shape).astype(np.float32), 0.0, None
    )
    valid = np.ones((num_episodes, num_steps), dtype=bool)
    shared_kwargs = {
        "num_bins_x": _NUM_BINS,
        "num_bins_y": _NUM_BINS,
        "smoothing_sigma": 0.0,
        "min_occupancy": 1e-6,
        "bounds": infer_bounds(positions.reshape(-1, 2)),
    }
    all_episodes = np.arange(num_episodes)

    in_sample = compute_twin_psi(
        noisy,
        positions,
        None,
        valid,
        16,
        fit_episodes=all_episodes,
        eval_episodes=all_episodes,
        **shared_kwargs,
    )
    held_out = compute_twin_psi(
        noisy,
        positions,
        None,
        valid,
        16,
        fit_episodes=all_episodes[: num_episodes // 2],
        eval_episodes=all_episodes[num_episodes // 2 :],
        **shared_kwargs,
    )

    def median_xy_ratio(twin_psi) -> float:
        ratios = twin_psi.psi_observed / (twin_psi.psi_xy_twin + 1e-12)
        return float(np.median(ratios[np.isfinite(ratios)]))

    assert median_xy_ratio(held_out) > median_xy_ratio(in_sample) + 0.02


def test_unit_whose_twin_has_no_lag_variance_is_excluded_not_maximally_stable(
    tmp_path: Path,
) -> None:
    """Psi_obs = Psi_twin = 0 is UNMEASURABLE, and 0/(0 + eps) reads as maximal excess."""
    num_episodes, num_steps, lag = 4, 64, 2
    step_parity = np.arange(num_steps) % 2
    positions = np.zeros((num_episodes, num_steps, 2), dtype=np.float32)
    positions[:, :, 0] = np.where(step_parity == 0, 2.0, 18.0)
    positions[:, :, 1] = 10.0
    representation = np.tile(
        np.where(step_parity == 0, 1.0, 2.0).astype(np.float32)[None, :, None],
        (num_episodes, 1, 1),
    )

    result = ExcessStabilityModule().run(
        _analysis_input(representation, positions),
        tmp_path,
        {**_CONFIG, "excess_stability_lag": lag},
    )

    assert result.per_unit_metrics["psi_observed"] == pytest.approx([0.0])
    assert result.per_unit_metrics["psi_xy_twin"] == pytest.approx([0.0])
    assert np.all(np.isnan(result.per_unit_metrics["excess_stability_ratio"]))
    assert np.isnan(result.metrics["median_excess_stability_ratio"])
    assert np.isnan(result.metrics["fraction_units_ratio_below_0_9"])
    assert result.metrics["fraction_units_dropped"] == 1.0
    assert result.metrics["num_units_unmeasurable_xy_twin"] == 1.0


def test_measurable_units_keep_the_unguarded_quotient_bit_for_bit(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    positions = _random_walk_positions(rng, num_episodes=8, num_steps=256)
    representation = _bin_table_activity(rng, positions, num_units=5)

    result = ExcessStabilityModule().run(
        _analysis_input(representation, positions), tmp_path, _CONFIG
    )

    ratios = result.per_unit_metrics["excess_stability_ratio"]
    unchanged = result.per_unit_metrics["psi_observed"] / (
        result.per_unit_metrics["psi_xy_twin"] + 1e-12
    )
    assert np.array_equal(ratios, unchanged)
    assert result.metrics["num_units_unmeasurable_xy_twin"] == 0.0
    assert result.metrics["num_units_unmeasurable_pose_twin"] == 0.0
    assert result.metrics["fraction_units_dropped"] == 0.0


def test_rotation_variance_fraction_isolates_rotation_steps(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    num_episodes, num_steps = 2, 64
    positions = _random_walk_positions(rng, num_episodes, num_steps)
    step_indices = np.arange(num_steps)
    rotation_transition = (step_indices % 2 == 1) & (step_indices > 0)
    translation_transition = (step_indices % 2 == 0) & (step_indices > 0)
    kinematics = np.zeros((num_episodes, num_steps, 4), dtype=np.float32)
    kinematics[:, rotation_transition, 1] = 0.35
    kinematics[:, translation_transition, 0] = 0.26

    def activity_changing_on(transition_mask: np.ndarray) -> np.ndarray:
        increments = np.where(
            transition_mask[None, :, None],
            rng.normal(size=(num_episodes, num_steps, 2)),
            0.0,
        )
        increments[:, 0] = 0.0
        return np.cumsum(increments, axis=1).astype(np.float32)

    rotation_result = ExcessStabilityModule().run(
        _analysis_input(
            activity_changing_on(rotation_transition), positions, kinematics=kinematics
        ),
        tmp_path,
        _CONFIG,
    )
    translation_result = ExcessStabilityModule().run(
        _analysis_input(
            activity_changing_on(translation_transition), positions, kinematics=kinematics
        ),
        tmp_path,
        _CONFIG,
    )

    assert rotation_result.metrics["rotation_variance_fraction"] == pytest.approx(1.0)
    assert translation_result.metrics["rotation_variance_fraction"] == pytest.approx(0.0)
