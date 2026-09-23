"""Tests for the re-anchoring-aware grid-cell battery."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.reanchoring_gridness import (
    ReanchoringGridnessModule,
    angular_harmonics,
    average_autocorrelograms,
    benjamini_hochberg,
    radial_periodicity_score,
)
from placecell_research.numerics.rate_map_kernels import (
    autocorrelogram,
    gridness_from_autocorrelogram,
    gridness_score,
)


def hex_bump_map(
    size: int = 48,
    spacing: float = 12.0,
    phase: tuple[float, float] = (0.0, 0.0),
    orientation_deg: float = 0.0,
    sigma: float = 2.5,
) -> np.ndarray:
    """Hexagonal grid of localized Gaussian fields (phase/orientation tunable)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    theta = np.deg2rad(orientation_deg)
    basis_a = spacing * np.array([np.cos(theta), np.sin(theta)])
    basis_b = spacing * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
    field = np.zeros((size, size), dtype=np.float64)
    span = range(-4, size // int(spacing) + 5)
    for i in span:
        for j in span:
            cx = i * basis_a[0] + j * basis_b[0] + phase[0]
            cy = i * basis_a[1] + j * basis_b[1] + phase[1]
            field += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return field


def stripe_map(size: int = 48, spacing: float = 12.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    del yy
    return np.clip(np.cos(2.0 * np.pi / spacing * xx), 0.0, None)


def single_blob_map(size: int = 48) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    center = size / 2.0
    return np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * 4.0**2))


def test_gridness_from_autocorrelogram_matches_gridness_score():
    rate_map = hex_bump_map()
    direct = gridness_score(rate_map)
    via_autocorr = gridness_from_autocorrelogram(autocorrelogram(rate_map))
    assert abs(direct - via_autocorr) < 1e-6


def test_gridness_from_autocorrelogram_high_for_hex_low_for_blob():
    assert gridness_from_autocorrelogram(autocorrelogram(hex_bump_map())) > 0.3
    assert gridness_from_autocorrelogram(autocorrelogram(single_blob_map())) < 0.1


def test_averaged_autocorrelogram_translation_invariant():
    rng = np.random.default_rng(0)
    spacing = 12.0
    single = gridness_score(hex_bump_map(spacing=spacing))
    per_episode = np.stack(
        [hex_bump_map(spacing=spacing, phase=tuple(rng.uniform(0, spacing, 2))) for _ in range(16)],
        axis=0,
    )
    averaged = gridness_from_autocorrelogram(average_autocorrelograms(per_episode))
    assert averaged > 0.5
    assert abs(averaged - single) < 0.4


def test_radial_periodicity_survives_orientation_reanchoring():
    rng = np.random.default_rng(1)
    spacing = 12.0
    per_episode = np.stack(
        [
            hex_bump_map(
                spacing=spacing,
                phase=tuple(rng.uniform(0, spacing, 2)),
                orientation_deg=float(rng.uniform(0, 60)),
            )
            for _ in range(16)
        ],
        axis=0,
    )
    averaged = average_autocorrelograms(per_episode)
    assert gridness_from_autocorrelogram(averaged) < 0.3
    assert radial_periodicity_score(averaged) > 0.15


def test_radial_periodicity_high_for_periodic_low_for_blob():
    assert radial_periodicity_score(autocorrelogram(hex_bump_map())) > 0.15
    assert radial_periodicity_score(autocorrelogram(stripe_map())) > 0.15
    assert radial_periodicity_score(autocorrelogram(single_blob_map())) < 0.05


def test_angular_harmonics_separate_stripe_from_hex():
    hex_stripe, hex_lattice = angular_harmonics(autocorrelogram(hex_bump_map()))
    band_stripe, band_lattice = angular_harmonics(autocorrelogram(stripe_map()))
    assert hex_lattice > hex_stripe
    assert band_stripe > band_lattice


def test_benjamini_hochberg_rejects_only_below_threshold():
    reject = benjamini_hochberg(np.array([0.001, 0.2, 0.03, 0.5]), alpha=0.05)
    np.testing.assert_array_equal(reject, np.array([True, False, False, False]))


def test_benjamini_hochberg_all_significant():
    reject = benjamini_hochberg(np.array([0.001, 0.002, 0.003]), alpha=0.05)
    assert reject.all()


def _reanchoring_analysis_input(spacing: float = 8.0) -> AnalysisInput:
    rng = np.random.default_rng(0)
    grid = 8
    coords = [(x, y) for y in range(grid) for x in range(grid)]
    steps = len(coords)
    num_episodes = 9
    positions = np.zeros((num_episodes, steps, 2), dtype=np.float32)
    representation = np.zeros((num_episodes, steps, 2), dtype=np.float32)
    for episode in range(num_episodes):
        phase = tuple(rng.uniform(0, spacing, 2))
        field = hex_bump_map(size=grid * 4, spacing=spacing, phase=phase, sigma=1.5)
        for step, (cx, cy) in enumerate(coords):
            positions[episode, step] = (cx, cy)
            representation[episode, step, 0] = field[cy * 4, cx * 4]
            representation[episode, step, 1] = rng.random()
    valid_mask = np.ones((num_episodes, steps), dtype=bool)
    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="predictor.hidden_state_layer_0",
        label="predictor_hidden_state_layer_0",
        split_name="validation",
    )


def test_module_runs_and_ranks_reanchoring_unit_above_noise(tmp_path: Path):
    config = {
        "reanchoring_num_bins_x": 12,
        "reanchoring_num_bins_y": 12,
        "reanchoring_per_episode_num_bins_x": 12,
        "reanchoring_per_episode_num_bins_y": 12,
        "reanchoring_smoothing_sigma": 0.8,
        "reanchoring_minimum_valid_steps": 4,
        "reanchoring_minimum_visited_fraction": 0.1,
        "reanchoring_render_top_k": 2,
        "reanchoring_shuffle_count": 0,
    }
    result = ReanchoringGridnessModule().run(_reanchoring_analysis_input(), tmp_path, config)

    for key in ("averaged_autocorr_gridness", "reanchoring_index", "radial_periodicity_score"):
        assert key in result.per_unit_metrics
    averaged = result.per_unit_metrics["averaged_autocorr_gridness"]
    assert averaged[0] > averaged[1]
    assert result.figures
    for path in result.figures.values():
        assert Path(path).exists()


def test_module_shuffle_null_produces_pvalues_for_candidates(tmp_path: Path):
    config = {
        "reanchoring_num_bins_x": 12,
        "reanchoring_num_bins_y": 12,
        "reanchoring_per_episode_num_bins_x": 12,
        "reanchoring_per_episode_num_bins_y": 12,
        "reanchoring_smoothing_sigma": 0.8,
        "reanchoring_minimum_valid_steps": 4,
        "reanchoring_minimum_visited_fraction": 0.1,
        "reanchoring_render_top_k": 2,
        "reanchoring_shuffle_count": 24,
        "reanchoring_shuffle_top_k": 2,
    }
    result = ReanchoringGridnessModule().run(_reanchoring_analysis_input(), tmp_path, config)

    p_values = result.per_unit_metrics["shuffle_p_value"]
    assert result.metrics["shuffle_candidates_tested"] == 2
    assert np.all(np.isfinite(p_values))
    assert np.all((p_values > 0.0) & (p_values <= 1.0))
    assert p_values[0] <= p_values[1]
    assert "num_significant_after_fdr" in result.metrics


def test_absent_shuffle_count_draws_the_analysis_config_default(tmp_path: Path, monkeypatch):
    """No shipped config sets this key, so the module fallback is what every run actually draws."""
    from placecell_research.config.schema import AnalysisConfig

    drawn: list[int] = []
    original_shuffle_null = ReanchoringGridnessModule._shuffle_null

    def record_shuffle_count(self, **kwargs):
        drawn.append(kwargs["shuffle_count"])
        return original_shuffle_null(self, **{**kwargs, "shuffle_count": 0})

    monkeypatch.setattr(ReanchoringGridnessModule, "_shuffle_null", record_shuffle_count)
    config = {
        "reanchoring_num_bins_x": 12,
        "reanchoring_num_bins_y": 12,
        "reanchoring_per_episode_num_bins_x": 12,
        "reanchoring_per_episode_num_bins_y": 12,
        "reanchoring_smoothing_sigma": 0.8,
        "reanchoring_minimum_valid_steps": 4,
        "reanchoring_minimum_visited_fraction": 0.1,
        "reanchoring_render_top_k": 2,
    }
    result = ReanchoringGridnessModule().run(_reanchoring_analysis_input(), tmp_path, config)

    assert drawn == [AnalysisConfig().reanchoring_shuffle_count]
    assert result.metadata["reanchoring_shuffle_count"] == drawn[0]
