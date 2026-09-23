from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis import decode_extrapolation
from placecell_research.analysis.band_score import BandScoreModule, band_score
from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.border_score import BorderScoreModule, border_score
from placecell_research.analysis.conformal_isometry import ConformalIsometryModule
from placecell_research.analysis.decode_extrapolation import DecodeExtrapolationModule
from placecell_research.analysis.effective_dimensionality import EffectiveDimensionalityModule
from placecell_research.analysis.fourier_ring import FourierRingModule
from placecell_research.analysis.manifold_topology import ManifoldTopologyModule
from placecell_research.numerics.fourier_ring import ring_metrics

BINS, BOX = 32, 2.2
_xs = np.linspace(0, BOX, BINS)
_X, _Y = np.meshgrid(_xs, _xs)


def _make_input(representation: np.ndarray, positions: np.ndarray) -> AnalysisInput:
    time_steps = representation.shape[0]
    return AnalysisInput(
        representation=representation[None].astype(np.float32),
        position_xy=positions[None].astype(np.float32),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, time_steps), dtype=bool),
        source_name="grid.hidden_state",
        label="test",
        split_name="test",
    )


def test_band_score_separates_stripes_from_grid_place_noise() -> None:
    stripes = np.maximum(np.cos(2 * np.pi / 0.55 * _X), 0.0)
    grid = sum(
        np.maximum(np.cos(2 * np.pi / 0.7 * (np.cos(a) * _X + np.sin(a) * _Y)), 0.0)
        for a in (0.0, np.pi / 3, 2 * np.pi / 3)
    )
    place = np.exp(-(((_X - BOX / 2) ** 2 + (_Y - BOX / 2) ** 2) / (2 * 0.3**2)))
    noise = np.random.default_rng(0).random((BINS, BINS))
    assert band_score(stripes) > 0.35
    assert band_score(stripes) > band_score(grid)
    assert band_score(stripes) > band_score(place)
    assert band_score(stripes) > band_score(noise)


def test_border_score_high_for_wall_cell_low_for_central_field() -> None:
    wall = np.zeros((BINS, BINS), dtype=np.float64)
    wall[:, :3] = 1.0
    center = np.exp(-(((_X - BOX / 2) ** 2 + (_Y - BOX / 2) ** 2) / (2 * 0.2**2)))
    assert border_score(wall) > 0.4
    assert border_score(wall) > border_score(center)


def test_band_and_border_scores_treat_unvisited_bins_as_zero_rate() -> None:
    stripes = np.maximum(np.cos(2 * np.pi / 0.55 * _X), 0.0)
    wall = np.zeros((BINS, BINS), dtype=np.float64)
    wall[:, :3] = 1.0
    unvisited = np.zeros((BINS, BINS), dtype=bool)
    unvisited[8:12, 8:12] = True

    stripes_with_unvisited_bins = stripes.copy()
    stripes_with_unvisited_bins[unvisited] = np.nan
    wall_with_unvisited_bins = wall.copy()
    wall_with_unvisited_bins[unvisited] = np.nan

    assert np.isfinite(band_score(stripes_with_unvisited_bins))
    assert border_score(wall_with_unvisited_bins) > 0.4


def test_effective_dimensionality_low_for_2d_manifold_high_for_noise() -> None:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, BOX, (2000, 2))
    manifold = np.concatenate([positions, 1e-3 * rng.standard_normal((2000, 30))], axis=1)
    noise = rng.standard_normal((2000, 32))
    pr_manifold = EffectiveDimensionalityModule().run(
        _make_input(manifold, positions), Path("/tmp"), {}
    ).metrics["participation_ratio"]
    pr_noise = EffectiveDimensionalityModule().run(
        _make_input(noise, positions), Path("/tmp"), {}
    ).metrics["participation_ratio"]
    assert pr_manifold < 4.0
    assert pr_noise > 10.0
    assert pr_manifold < pr_noise


def test_conformal_isometry_high_for_position_code_low_for_scramble() -> None:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, BOX, (800, 2))
    isometric = positions + 1e-2 * rng.standard_normal((800, 2))
    scrambled = rng.standard_normal((800, 8))
    iso = ConformalIsometryModule().run(_make_input(isometric, positions), Path("/tmp"), {}).metrics
    scram = ConformalIsometryModule().run(
        _make_input(scrambled, positions), Path("/tmp"), {}
    ).metrics
    assert iso["metric_distance_correlation"] > 0.95
    assert iso["metric_distance_correlation"] > scram["metric_distance_correlation"]
    assert iso["metric_scale_cv"] < scram["metric_scale_cv"]


def test_decode_extrapolation_generalises_for_linear_code() -> None:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, BOX, (3000, 2))
    projection = rng.standard_normal((2, 16))
    linear = positions @ projection + 1e-3 * rng.standard_normal((3000, 16))
    lookup = rng.standard_normal((3000, 16))
    lin = DecodeExtrapolationModule().run(_make_input(linear, positions), Path("/tmp"), {}).metrics
    look = DecodeExtrapolationModule().run(_make_input(lookup, positions), Path("/tmp"), {}).metrics
    assert lin["extrapolation_r2"] > 0.9
    assert lin["extrapolation_r2"] > look["extrapolation_r2"]


def test_decode_extrapolation_caps_samples_before_ridge(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    positions = rng.uniform(0, BOX, (3000, 2))
    representation = rng.standard_normal((3000, 32))
    ridge_sample_counts: list[int] = []

    def record_ridge_samples(
        train_features,
        train_targets,
        test_features,
        test_targets,
        alpha,
    ) -> float:
        del train_targets, test_targets, alpha
        ridge_sample_counts.extend([len(train_features), len(test_features)])
        return 0.0

    monkeypatch.setattr(decode_extrapolation, "_ridge_r2", record_ridge_samples)

    result = DecodeExtrapolationModule().run(
        _make_input(representation, positions),
        Path("/tmp"),
        {"geometry_max_samples": 64},
    )

    assert max(ridge_sample_counts) <= 64
    assert result.metadata["sampled_valid_steps"] == 64
    assert result.metadata["total_valid_steps"] == 3000


def test_manifold_topology_runs_or_skips_cleanly() -> None:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, BOX, (200, 2))
    representation = np.concatenate([positions, 1e-2 * rng.standard_normal((200, 6))], axis=1)
    result = ManifoldTopologyModule().run(_make_input(representation, positions), Path("/tmp"), {})
    assert ("manifold_topology_skipped" in result.metadata) or ("betti_1" in result.metrics)


def test_fourier_ring_detects_band_pass_ring_not_low_pass_blob() -> None:
    grid = sum(
        np.cos(2 * np.pi / 0.55 * (np.cos(a) * _X + np.sin(a) * _Y))
        for a in (0.0, np.pi / 3, 2 * np.pi / 3)
    )
    blob = np.exp(-(((_X - BOX / 2) ** 2 + (_Y - BOX / 2) ** 2) / (2 * 0.3**2)))
    ring = ring_metrics(grid)
    flat = ring_metrics(blob)
    assert ring["is_band_pass"] == 1.0
    assert ring["ring_peak_frequency"] > 1.0
    assert flat["is_band_pass"] == 0.0
    assert ring["ring_score"] > flat["ring_score"]


def test_per_unit_modules_run_smoke() -> None:
    rng = np.random.default_rng(0)
    time_steps = 3000
    positions = rng.uniform(0, BOX, (time_steps, 2))
    representation = rng.standard_normal((time_steps, 12))
    analysis_input = _make_input(representation, positions)
    config = {"num_bins_x": 16, "num_bins_y": 16, "smoothing_sigma": 1.0}
    for module in (BandScoreModule(), BorderScoreModule(), FourierRingModule()):
        result = module.run(analysis_input, Path("/tmp"), config)
        assert len(next(iter(result.per_unit_metrics.values()))) == 12
