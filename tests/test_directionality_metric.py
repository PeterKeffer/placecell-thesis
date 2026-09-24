"""Directional-modulation analysis metric: separates omnidirectional from directional cells."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.directionality import (
    _NULL_MIN_SHIFT_FRACTION,
    _NULL_RNG_SEED,
    _NUM_QUADRANTS,
    DirectionalityModule,
    _modulation_r_from_quadrant_means,
    _null_modulation_matrix,
    _prepare_directional_statistics,
    _quadrant_means_from_labels,
    _resolve_field_gates,
)
from placecell_research.analysis.registry import ANALYSIS_MODULES
from placecell_research.analysis.shift_nulls import _draw_circular_shift_offsets
from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay
from placecell_research.numerics.rate_map_kernels import scatter_add_over_units

CORNERS = np.asarray([(0.0, 0.0), (0.0, 3.0), (3.0, 0.0), (3.0, 3.0)], dtype=np.float32)


def test_directionality_is_registered_for_config_use() -> None:
    assert "directionality" in ANALYSIS_MODULES
    assert ANALYSIS_MODULES["directionality"]().name == "directionality"


def _build_input() -> AnalysisInput:
    """Two units over 4 well-separated locations, each visited from all 4 heading quadrants."""
    corners = [(0.0, 0.0), (0.0, 3.0), (3.0, 0.0), (3.0, 3.0)]
    quad_centers = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
    repeats = 6
    positions, headings, omni, directional = [], [], [], []
    for x, y in corners:
        for quadrant, center in enumerate(quad_centers):
            for _ in range(repeats):
                positions.append((x, y))
                headings.append(center)
                omni.append(1.0)
                directional.append(1.0 if quadrant == 0 else 0.0)
    steps = len(positions)
    representation = np.stack([omni, directional], axis=1).reshape(1, steps, 2).astype(np.float32)
    position_xy = np.asarray(positions, dtype=np.float32).reshape(1, steps, 2)
    heading = np.asarray(headings, dtype=np.float32).reshape(1, steps)
    valid_mask = np.ones((1, steps), dtype=bool)
    return AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )


def test_directionality_separates_omnidirectional_from_directional(
    tmp_path, analysis_settings
) -> None:
    result = DirectionalityModule().run(
        _build_input(),
        tmp_path,
        analysis_settings(num_bins=2, min_occupancy_per_quadrant=5, min_field_bins=3),
    )

    r_values = result.per_unit_metrics["directional_modulation_r"]
    assert r_values.shape == (2,)
    assert r_values[0] < 0.1, "omnidirectional unit should have low directional modulation"
    assert r_values[1] > 0.9, "directional unit should have high directional modulation"


def test_directionality_reports_headline_aggregate_metrics(tmp_path, analysis_settings) -> None:
    result = DirectionalityModule().run(
        _build_input(),
        tmp_path,
        analysis_settings(num_bins=2, min_occupancy_per_quadrant=5, min_field_bins=3),
    )

    assert "median_directional_modulation_r" in result.metrics
    assert "fraction_omnidirectional" in result.metrics
    assert "fraction_directional" in result.metrics
    assert result.metrics["fraction_omnidirectional"] == 0.5
    assert result.metrics["fraction_directional"] == 0.5


def test_directionality_uses_canonical_place_field_threshold_config(
    monkeypatch, tmp_path, analysis_settings
) -> None:
    captured_thresholds = []

    def fake_modulation_r(statistics, mean_per_quadrant, **kwargs):
        del statistics, mean_per_quadrant
        captured_thresholds.append(kwargs["field_threshold_fraction"])
        return np.asarray([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "placecell_research.analysis.directionality._modulation_r_from_quadrant_means",
        fake_modulation_r,
    )

    DirectionalityModule().run(
        _build_input(),
        tmp_path,
        analysis_settings(
            num_bins=2,
            place_field_threshold_fraction=0.17,
            directionality_null_shuffles=0,
        ),
    )

    assert captured_thresholds == [0.17]


def test_directionality_ignores_legacy_field_threshold_alias(
    monkeypatch, tmp_path, analysis_settings
) -> None:
    captured_thresholds = []

    def fake_modulation_r(statistics, mean_per_quadrant, **kwargs):
        del statistics, mean_per_quadrant
        captured_thresholds.append(kwargs["field_threshold_fraction"])
        return np.asarray([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "placecell_research.analysis.directionality._modulation_r_from_quadrant_means",
        fake_modulation_r,
    )

    DirectionalityModule().run(
        _build_input(),
        tmp_path,
        {
            **analysis_settings(num_bins=2, directionality_null_shuffles=0),
            "field_threshold_fraction": 0.17,
        },
    )

    assert captured_thresholds == [0.2]


def test_directionality_writes_per_unit_csv(tmp_path, analysis_settings) -> None:
    result = DirectionalityModule().run(
        _build_input(),
        tmp_path,
        analysis_settings(num_bins=2, min_occupancy_per_quadrant=5, min_field_bins=3),
    )

    table_path = result.tables["per_unit_metrics"]
    lines = table_path.read_text().splitlines()
    assert lines[0] == (
        "unit_index,directional_modulation_r,directional_modulation_r_excess,"
        "directional_modulation_r_null_p,directional_modulation_r_significant,assessable"
    )
    assert lines[1].startswith("0,")
    assert lines[2].startswith("1,")


def test_directionality_marks_unassessable_units_when_heading_missing(
    tmp_path, analysis_settings
) -> None:
    analysis_input = _build_input()
    analysis_input.heading = None
    result = DirectionalityModule().run(
        analysis_input,
        tmp_path,
        analysis_settings(num_bins=2, min_occupancy_per_quadrant=5, min_field_bins=3),
    )
    r_values = result.per_unit_metrics["directional_modulation_r"]
    assert r_values.shape == (2,)
    assert np.all(np.isnan(r_values)), "without heading the metric cannot be assessed"


def test_directionality_uses_known_world_bounds(monkeypatch, tmp_path, analysis_settings) -> None:
    env_id = "MiniWorld-WallGapAsym-v0"
    expected_bounds = overlay_bounds(resolve_world_overlay(env_id))
    analysis_input = _build_input()
    analysis_input.metadata["env_id"] = env_id
    captured_bounds = []

    def fake_compute_spatial_bin_assignments(positions, *, num_bins_x, num_bins_y, bounds):
        del positions
        captured_bounds.append(bounds)
        linear_bins = np.arange(96, dtype=np.int32) % (num_bins_x * num_bins_y)
        return linear_bins, np.asarray([]), np.asarray([]), bounds

    monkeypatch.setattr(
        "placecell_research.analysis.directionality.compute_spatial_bin_assignments",
        fake_compute_spatial_bin_assignments,
        raising=False,
    )

    DirectionalityModule().run(
        analysis_input,
        tmp_path,
        analysis_settings(num_bins=2, min_occupancy_per_quadrant=5, min_field_bins=3),
    )

    assert captured_bounds == [expected_bounds]


_NULL_CONFIG = {
    "num_bins": 2,
    "min_occupancy_per_quadrant": 5,
    "min_field_bins": 3,
    "directionality_null_shuffles": 100,
}


def _build_interleaved_heading_input(seed: int = 11) -> AnalysisInput:
    """Interleaved visits: the same four corners under an aperiodic heading random walk."""
    rng = np.random.default_rng(seed)
    num_episodes, num_steps = 4, 500
    headings = np.zeros((num_episodes, num_steps), dtype=np.float32)
    positions = np.zeros((num_episodes, num_steps, 2), dtype=np.float32)
    for episode in range(num_episodes):
        heading = rng.uniform(0.0, 2.0 * np.pi)
        for step in range(num_steps):
            heading += rng.normal(0.0, 0.9)
            headings[episode, step] = heading
            positions[episode, step] = CORNERS[step % 4]
    sharp = np.exp(3.0 * np.cos(headings - np.pi / 2.0)).astype(np.float32)
    graded = (2.0 + np.cos(headings)).astype(np.float32)
    flat = np.ones((num_episodes, num_steps), dtype=np.float32)
    return AnalysisInput(
        representation=np.stack([sharp, graded, flat], axis=2),
        position_xy=positions,
        heading=headings,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((num_episodes, num_steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )


def _build_temporal_bout_input() -> AnalysisInput:
    """One contiguous activity bout under four contiguous 100-step heading quadrant blocks."""
    num_steps = 400
    steps = np.arange(num_steps)
    bout = np.zeros(num_steps, dtype=np.float32)
    bout[150:250] = 1.0
    flat = np.full(num_steps, 0.5, dtype=np.float32)
    return AnalysisInput(
        representation=np.stack([bout, flat], axis=1).reshape(1, num_steps, 2),
        position_xy=CORNERS[steps % 4].reshape(1, num_steps, 2),
        heading=(2.0 * np.pi * steps / num_steps).astype(np.float32).reshape(1, num_steps),
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, num_steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )


def test_directionality_null_flags_heading_tuned_not_omnidirectional(
    tmp_path, analysis_settings
) -> None:
    """Tuning that repeats across interleaved visits survives the circular-shift null."""
    result = DirectionalityModule().run(
        _build_interleaved_heading_input(), tmp_path, analysis_settings(**_NULL_CONFIG)
    )

    significant = result.per_unit_metrics["directional_modulation_r_significant"]
    excess = result.per_unit_metrics["directional_modulation_r_excess"]
    assert significant.tolist() == [1.0, 1.0, 0.0]
    assert excess[0] > 0.5
    assert excess[1] > 0.2
    assert abs(excess[2]) < 0.05
    assert np.isclose(result.metrics["fraction_directional_significant"], 2.0 / 3.0)


def test_directionality_null_does_not_flag_a_temporal_bout(tmp_path, analysis_settings) -> None:
    """A bout of activity is not a directional cell, however cleanly it sits in one quadrant."""
    result = DirectionalityModule().run(
        _build_temporal_bout_input(), tmp_path, analysis_settings(**_NULL_CONFIG)
    )

    r_values = result.per_unit_metrics["directional_modulation_r"]
    null_p = result.per_unit_metrics["directional_modulation_r_null_p"]
    significant = result.per_unit_metrics["directional_modulation_r_significant"]
    assert r_values[0] == 1.0, "the metric itself is unchanged; only its null moved"
    assert null_p[0] == 1.0
    assert significant.tolist() == [0.0, 0.0]
    assert result.metrics["fraction_directional_significant"] == 0.0


def _build_random_walk_input(seed: int = 3) -> AnalysisInput:
    """Random walks over a 6x6 arena with place, conjunctive and noise units."""
    rng = np.random.default_rng(seed)
    num_episodes, num_steps, num_units = 6, 400, 12
    positions = np.zeros((num_episodes, num_steps, 2), dtype=np.float32)
    headings = np.zeros((num_episodes, num_steps), dtype=np.float32)
    for episode in range(num_episodes):
        position = rng.uniform(1.0, 5.0, size=2)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        for step in range(num_steps):
            heading += rng.normal(0.0, 0.4)
            step_vector = 0.3 * np.array([np.cos(heading), np.sin(heading)])
            position = np.clip(position + step_vector, 0.1, 5.9)
            positions[episode, step] = position
            headings[episode, step] = heading

    flat_positions = positions.reshape(-1, 2)
    flat_headings = headings.reshape(-1)
    centers = rng.uniform(0.5, 5.5, size=(num_units, 2))
    preferred = rng.uniform(0.0, 2.0 * np.pi, size=num_units)
    heading_gain = np.tile([0.0, 0.9, 0.4, 0.0], num_units // 4)
    place = np.exp(
        -((flat_positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) / 2.0
    )
    tuning = place * (
        1.0
        - heading_gain[None, :]
        * (1.0 - np.cos(flat_headings[:, None] - preferred[None, :]))
        / 2.0
    )
    tuning += rng.normal(0.0, 0.05, size=tuning.shape)
    cutoff = np.partition(tuning, -4, axis=1)[:, -4][:, None]
    tuning = np.where(tuning >= cutoff, tuning, 0.0)
    return AnalysisInput(
        representation=tuning.reshape(num_episodes, num_steps, num_units).astype(np.float32),
        position_xy=positions,
        heading=headings,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((num_episodes, num_steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )


_NULL_KWARGS = {
    "field_threshold_fraction": 0.2,
    "min_heading_quadrants": 3,
    "min_field_bins": 3,
    "min_fire_rate": 0.01,
}


_REFERENCE_EPS = 1e-9


def _standalone_modulation_r_reference(
    statistics,
    mean_per_quadrant: np.ndarray,
    *,
    field_threshold_fraction: float,
    min_heading_quadrants: int,
    min_field_bins: int,
    min_fire_rate: float,
) -> np.ndarray:
    """R scored the plain way: dense masks over the FULL grid, nothing factored out."""
    marginal = statistics.marginal
    finite_marginal = np.isfinite(marginal)
    peak = np.max(np.where(finite_marginal, marginal, -np.inf), axis=0)
    shared_bin_gate = (statistics.sampled_quadrant_count >= int(min_heading_quadrants)) & (
        statistics.occupancy_bin > 0
    )
    field_mask = (
        finite_marginal
        & (marginal >= float(field_threshold_fraction) * peak[None, :])
        & shared_bin_gate[:, None]
    )
    sampled = statistics.sampled_quadrant[:, :, None]
    all_sampled_finite = np.all(np.where(sampled, np.isfinite(mean_per_quadrant), True), axis=1)
    quadrant_max = np.max(np.where(sampled, mean_per_quadrant, -np.inf), axis=1)
    quadrant_min = np.min(np.where(sampled, mean_per_quadrant, np.inf), axis=1)
    usable_bin_unit = field_mask & all_sampled_finite & (quadrant_max > _REFERENCE_EPS)
    with np.errstate(invalid="ignore"):
        modulation = (quadrant_max - quadrant_min) / (
            quadrant_max + quadrant_min + _REFERENCE_EPS
        )
    weighted = np.where(usable_bin_unit, modulation, 0.0) * statistics.occupancy_bin[:, None]
    weight_sums = (usable_bin_unit * statistics.occupancy_bin[:, None]).sum(axis=0)
    r_values = np.full(statistics.rates.shape[1], np.nan, dtype=np.float32)
    assessable = (
        (statistics.fire_rate >= float(min_fire_rate))
        & np.isfinite(peak)
        & (peak > _REFERENCE_EPS)
        & (usable_bin_unit.sum(axis=0) >= int(min_field_bins))
        & (weight_sums > 0.0)
    )
    r_values[assessable] = (
        weighted.sum(axis=0)[assessable] / weight_sums[assessable]
    ).astype(np.float32)
    return r_values


def _dense_circular_shift_null_reference(statistics, num_shuffles: int) -> np.ndarray:
    """The null the fast path must reproduce: one whole-array np.roll per episode."""
    rng = np.random.default_rng(_NULL_RNG_SEED)
    shuffle_offsets = [
        _draw_circular_shift_offsets(statistics.episode_lengths, rng, _NULL_MIN_SHIFT_FRACTION)
        for _ in range(num_shuffles)
    ]
    episode_bounds = np.concatenate(([0], np.cumsum(statistics.episode_lengths)))
    combined = statistics.bin_index * _NUM_QUADRANTS + statistics.quadrant
    num_units = statistics.rates.shape[1]
    safe_occupancy_bin = np.where(
        statistics.occupancy_bin > 0, statistics.occupancy_bin, np.nan
    )[:, None]

    rows = []
    for offsets in shuffle_offsets:
        rolled = np.concatenate(
            [
                np.roll(statistics.rates[start:stop], int(offset), axis=0)
                for start, stop, offset in zip(
                    episode_bounds[:-1], episode_bounds[1:], offsets, strict=False
                )
            ]
        )
        activity = scatter_add_over_units(
            combined, rolled, statistics.num_spatial_bins * _NUM_QUADRANTS
        ).reshape(statistics.num_spatial_bins, _NUM_QUADRANTS, num_units)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_per_quadrant = activity / statistics.occupancy_bq[:, :, None]
            marginal = activity.sum(axis=1) / safe_occupancy_bin
        rows.append(
            _standalone_modulation_r_reference(
                replace(statistics, rates=rolled, marginal=marginal),
                mean_per_quadrant,
                **_NULL_KWARGS,
            )
        )
    return np.stack(rows)


def _statistics_and_gates(analysis_input: AnalysisInput):
    statistics = _prepare_directional_statistics(
        representation=analysis_input.representation,
        position_xy=analysis_input.position_xy,
        heading=analysis_input.heading,
        valid_mask=analysis_input.valid_mask,
        num_bins_x=5,
        num_bins_y=5,
        bounds=None,
        min_occupancy_per_quadrant=5,
    )
    gates = _resolve_field_gates(
        statistics,
        field_threshold_fraction=_NULL_KWARGS["field_threshold_fraction"],
        min_heading_quadrants=_NULL_KWARGS["min_heading_quadrants"],
    )
    return statistics, gates


def test_null_matrix_matches_the_dense_circular_shift_reference() -> None:
    """Sparse rescatter over occupied bins must reproduce the dense roll draw for draw."""
    statistics, gates = _statistics_and_gates(_build_random_walk_input())
    expected = _dense_circular_shift_null_reference(statistics, 30)
    actual = _null_modulation_matrix(
        statistics,
        gates,
        num_shuffles=30,
        field_threshold_fraction=_NULL_KWARGS["field_threshold_fraction"],
        min_field_bins=_NULL_KWARGS["min_field_bins"],
        min_fire_rate=_NULL_KWARGS["min_fire_rate"],
    )
    assert np.isfinite(expected).any(), "fixture must produce assessable null draws"
    np.testing.assert_array_equal(actual, expected)


def test_null_shifts_respect_episode_boundaries() -> None:
    """Rolling must stay inside an episode: split one episode in two and the null changes."""
    analysis_input = _build_random_walk_input()
    num_episodes, num_steps, _ = analysis_input.representation.shape
    one_episode = replace(
        analysis_input,
        representation=analysis_input.representation.reshape(1, num_episodes * num_steps, -1),
        position_xy=analysis_input.position_xy.reshape(1, num_episodes * num_steps, 2),
        heading=analysis_input.heading.reshape(1, num_episodes * num_steps),
        valid_mask=np.ones((1, num_episodes * num_steps), dtype=bool),
    )
    kwargs = {
        "num_shuffles": 8,
        "field_threshold_fraction": _NULL_KWARGS["field_threshold_fraction"],
        "min_field_bins": _NULL_KWARGS["min_field_bins"],
        "min_fire_rate": _NULL_KWARGS["min_fire_rate"],
    }
    per_episode = _null_modulation_matrix(*_statistics_and_gates(analysis_input), **kwargs)
    pooled = _null_modulation_matrix(*_statistics_and_gates(one_episode), **kwargs)
    assert np.isfinite(per_episode).any()
    assert not np.array_equal(per_episode, pooled, equal_nan=True)


def test_observed_modulation_r_matches_the_standalone_scorer() -> None:
    """The gate-factored scorer must equal the plain dense one on every gate setting."""
    analysis_input = _build_random_walk_input()
    statistics = _prepare_directional_statistics(
        representation=analysis_input.representation,
        position_xy=analysis_input.position_xy,
        heading=analysis_input.heading,
        valid_mask=analysis_input.valid_mask,
        num_bins_x=5,
        num_bins_y=5,
        bounds=None,
        min_occupancy_per_quadrant=5,
    )
    mean_per_quadrant = _quadrant_means_from_labels(statistics, statistics.quadrant)
    for min_heading_quadrants in (0, 2, 3, 4):
        for field_threshold_fraction in (0.0, 0.2, 0.9):
            kwargs = {
                **_NULL_KWARGS,
                "min_heading_quadrants": min_heading_quadrants,
                "field_threshold_fraction": field_threshold_fraction,
            }
            expected = _standalone_modulation_r_reference(
                statistics, mean_per_quadrant, **kwargs
            )
            actual = _modulation_r_from_quadrant_means(statistics, mean_per_quadrant, **kwargs)
            np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(
        _modulation_r_from_quadrant_means(statistics, mean_per_quadrant, **_NULL_KWARGS)
    ).any(), "fixture must produce assessable units"


def test_null_matrix_is_independent_of_the_unit_chunk_width(monkeypatch) -> None:
    statistics, gates = _statistics_and_gates(_build_random_walk_input())
    kwargs = {
        "num_shuffles": 20,
        "field_threshold_fraction": _NULL_KWARGS["field_threshold_fraction"],
        "min_field_bins": _NULL_KWARGS["min_field_bins"],
        "min_fire_rate": _NULL_KWARGS["min_fire_rate"],
    }
    whole = _null_modulation_matrix(statistics, gates, **kwargs)
    monkeypatch.setattr(
        "placecell_research.analysis.directionality._NULL_CHUNK_NONZERO_BUDGET", 1
    )
    np.testing.assert_array_equal(_null_modulation_matrix(statistics, gates, **kwargs), whole)
