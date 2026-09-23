from __future__ import annotations

import numpy as np

from placecell_research.analysis.reliability_splits import (
    _compute_global_unit_thresholds,
    compute_bin_consistency_maps,
    compute_episode_rate_map_correlations,
    compute_field_traversal_reliability,
    compute_reliability_maps,
    compute_revisit_activity_metrics,
    compute_split_half_agreement_maps,
    compute_split_half_agreement_maps_and_correlations,
)
from placecell_research.numerics.bin_maps import MIN_MAP_CORRELATION_OVERLAP_BINS
from placecell_research.numerics.occupancy import (
    EpisodeBinStatistics,
    prepare_episode_bin_statistics,
)
from placecell_research.numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    infer_bounds,
)
from placecell_research.numerics.split_half import compute_split_half_rate_map_correlations


def _reference_compute_reliability_maps(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    num_bins_x: int,
    num_bins_y: int,
    threshold_mode: str,
    threshold_fraction: float,
    threshold_quantile: float,
    use_absolute_activations: bool,
) -> tuple[np.ndarray, np.ndarray]:
    flattened_mask = (
        np.ones(representation.shape[:2], dtype=bool)
        if valid_mask is None
        else valid_mask.astype(bool, copy=False)
    )
    flattened_positions = position_xy[flattened_mask]
    bounds = infer_bounds(flattened_positions)
    flat_values = representation[flattened_mask]
    num_units = representation.shape[-1]
    num_bins = num_bins_x * num_bins_y

    if threshold_mode == "quantile_per_unit":
        threshold_source = np.abs(flat_values) if use_absolute_activations else flat_values
        thresholds = np.maximum(
            np.quantile(
                threshold_source.astype(np.float64, copy=False),
                np.float64(threshold_quantile),
                axis=0,
            ),
            1e-8,
        ).astype(np.float32, copy=False)
    else:
        peak_source = np.abs(flat_values) if use_absolute_activations else flat_values
        global_peaks = np.quantile(
            peak_source.astype(np.float64, copy=False), np.float64(0.995), axis=0
        )
        thresholds = np.maximum(global_peaks * float(threshold_fraction), 1e-8).astype(
            np.float32,
            copy=False,
        )

    visited_episode_counts = np.zeros((num_bins,), dtype=np.int32)
    spike_episode_counts = np.zeros((num_units, num_bins), dtype=np.int32)
    for episode_index in range(representation.shape[0]):
        episode_valid = flattened_mask[episode_index]
        if not np.any(episode_valid):
            continue
        episode_positions = position_xy[episode_index, episode_valid]
        linear_bins, _, _, _ = compute_spatial_bin_assignments(
            episode_positions,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=bounds,
        )
        visited_episode_counts += (np.bincount(linear_bins, minlength=num_bins) > 0).astype(
            np.int32,
            copy=False,
        )
        episode_values = representation[episode_index, episode_valid]
        strong_mask = (
            np.abs(episode_values) >= thresholds[None, :]
            if use_absolute_activations
            else episode_values >= thresholds[None, :]
        )
        linear_bins_int64 = linear_bins.astype(np.int64, copy=False)
        for unit_index in range(num_units):
            active_bins = linear_bins_int64[strong_mask[:, unit_index]]
            if active_bins.size == 0:
                continue
            spike_episode_counts[unit_index] += (
                np.bincount(active_bins, minlength=num_bins) > 0
            ).astype(np.int32, copy=False)

    reliability_maps = np.full((num_units, num_bins_y, num_bins_x), np.nan, dtype=np.float32)
    visited_mask = visited_episode_counts > 0
    if np.any(visited_mask):
        reliability_flat = reliability_maps.reshape(num_units, -1)
        reliability_flat[:, visited_mask] = (
            spike_episode_counts[:, visited_mask] / visited_episode_counts[visited_mask]
        ).astype(np.float32, copy=False)
    return (
        reliability_maps,
        visited_episode_counts.reshape(num_bins_y, num_bins_x).astype(np.float32, copy=False),
    )


def test_compute_reliability_maps_matches_reference_episode_loop() -> None:
    rng = np.random.default_rng(42)
    representation = rng.normal(size=(5, 9, 7)).astype(np.float32)
    position_xy = rng.uniform(low=-1.0, high=1.0, size=(5, 9, 2)).astype(np.float32)
    valid_mask = np.asarray(
        [
            [True, True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False, False],
            [False, True, True, True, True, True, False, False, False],
            [True, False, True, False, True, False, True, False, True],
            [False, False, False, False, False, False, False, False, False],
        ],
        dtype=bool,
    )

    for threshold_mode, use_absolute_activations in (
        ("peak_fraction", True),
        ("peak_fraction", False),
        ("quantile_per_unit", False),
    ):
        expected_reliability, expected_visits = _reference_compute_reliability_maps(
            representation,
            position_xy,
            valid_mask,
            num_bins_x=6,
            num_bins_y=5,
            threshold_mode=threshold_mode,
            threshold_fraction=0.35,
            threshold_quantile=0.72,
            use_absolute_activations=use_absolute_activations,
        )
        actual_reliability, actual_visits = compute_reliability_maps(
            representation,
            position_xy,
            valid_mask,
            num_bins_x=6,
            num_bins_y=5,
            threshold_mode=threshold_mode,
            threshold_fraction=0.35,
            threshold_quantile=0.72,
            use_absolute_activations=use_absolute_activations,
            unit_chunk_size=3,
        )

        np.testing.assert_allclose(actual_reliability, expected_reliability, equal_nan=True)
        np.testing.assert_allclose(actual_visits, expected_visits)


def test_compute_reliability_maps_avoids_dense_episode_presence(monkeypatch) -> None:
    rng = np.random.default_rng(17)
    episodes = 4
    steps = 11
    num_units = 3
    num_bins_x = 5
    num_bins_y = 4
    num_bins = num_bins_x * num_bins_y
    representation = rng.normal(size=(episodes, steps, num_units)).astype(np.float32)
    position_xy = rng.uniform(low=-1.0, high=1.0, size=(episodes, steps, 2)).astype(
        np.float32,
        copy=False,
    )
    valid_mask = rng.random(size=(episodes, steps)) > 0.1
    forbidden_minlength = num_units * episodes * num_bins
    original_bincount = np.bincount

    def guarded_bincount(values, weights=None, minlength=0):
        if int(minlength) == forbidden_minlength:
            raise AssertionError("reliability counting should not allocate dense episode presence")
        return original_bincount(values, weights=weights, minlength=minlength)

    monkeypatch.setattr(
        "placecell_research.analysis.helpers.np.bincount",
        guarded_bincount,
    )

    actual_reliability, actual_visits = compute_reliability_maps(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        threshold_mode="quantile_per_unit",
        threshold_quantile=0.65,
        use_absolute_activations=False,
        unit_chunk_size=num_units,
    )
    expected_reliability, expected_visits = _reference_compute_reliability_maps(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        threshold_mode="quantile_per_unit",
        threshold_fraction=0.3,
        threshold_quantile=0.65,
        use_absolute_activations=False,
    )

    np.testing.assert_allclose(actual_reliability, expected_reliability, equal_nan=True)
    np.testing.assert_allclose(actual_visits, expected_visits)


def test_revisit_stability_metrics_reward_consistent_per_bin_activity() -> None:
    position_xy = np.asarray(
        [
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
        ],
        dtype=np.float32,
    )
    representation = np.asarray(
        [
            [[0.0, 0.0], [1.0, 1.0], [1.0, 0.1], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.1], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [1.0, 0.1], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.1], [1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    consistency_maps, coefficient_of_variation_maps, visit_counts = compute_bin_consistency_maps(
        representation,
        position_xy,
        valid_mask=None,
        num_bins_x=4,
        num_bins_y=1,
        active_bin_peak_fraction=0.05,
        active_episode_threshold_fraction_of_bin_mean=0.5,
    )
    split_half_correlation = compute_split_half_rate_map_correlations(
        representation,
        position_xy,
        valid_mask=None,
        num_bins_x=4,
        num_bins_y=1,
        smoothing_sigma=0.0,
        min_occupancy=1e-6,
    )
    episode_rate_map_correlation = compute_episode_rate_map_correlations(
        representation,
        position_xy,
        valid_mask=None,
        num_bins_x=4,
        num_bins_y=1,
        smoothing_sigma=0.0,
        min_occupancy=1e-6,
    )
    split_half_agreement_maps, split_half_support_counts = compute_split_half_agreement_maps(
        representation,
        position_xy,
        valid_mask=None,
        num_bins_x=4,
        num_bins_y=1,
        smoothing_sigma=0.0,
        min_occupancy=1e-6,
    )

    np.testing.assert_allclose(visit_counts, np.ones((1, 4), dtype=np.float32) * 4.0)
    np.testing.assert_allclose(split_half_support_counts, np.ones((1, 4), dtype=np.float32) * 2.0)
    np.testing.assert_allclose(
        consistency_maps[:, 0, [0, 3]],
        np.zeros((2, 2), dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        consistency_maps[0, 0, 1:3],
        np.ones((2,), dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        coefficient_of_variation_maps[0, 0, 1:3],
        np.zeros((2,), dtype=np.float32),
        atol=1e-6,
    )
    assert float(consistency_maps[1, 0, 1]) < 0.7
    assert float(consistency_maps[1, 0, 2]) < 0.7
    assert float(coefficient_of_variation_maps[1, 0, 1]) > 0.5
    assert float(coefficient_of_variation_maps[1, 0, 2]) > 0.5
    np.testing.assert_allclose(
        split_half_agreement_maps[0, 0, 1:3],
        np.ones((2,), dtype=np.float32),
        atol=1e-6,
    )
    assert float(split_half_agreement_maps[1, 0, 1]) < 0.3
    assert float(split_half_agreement_maps[1, 0, 2]) < 0.3

    assert 4 * 1 < MIN_MAP_CORRELATION_OVERLAP_BINS
    assert np.isnan(split_half_correlation).all()
    assert np.isnan(episode_rate_map_correlation).all()


def test_map_correlations_rank_a_stable_field_above_a_shifting_one() -> None:
    num_bins = MIN_MAP_CORRELATION_OVERLAP_BINS + 2
    bin_centers = (np.arange(num_bins, dtype=np.float32) + 0.5) / num_bins * 2.0 - 1.0
    position_xy = np.stack(
        [
            np.tile(bin_centers, (4, 1)),
            np.zeros((4, num_bins), dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    stable_field = np.full(num_bins, 0.1, dtype=np.float32)
    stable_field[3:6] = 1.0
    shifting_field_even = np.full(num_bins, 0.1, dtype=np.float32)
    shifting_field_even[2:5] = 1.0
    shifting_field_odd = np.full(num_bins, 0.1, dtype=np.float32)
    shifting_field_odd[7:10] = 1.0

    representation = np.zeros((4, num_bins, 2), dtype=np.float32)
    for episode_index in range(4):
        representation[episode_index, :, 0] = stable_field
        representation[episode_index, :, 1] = (
            shifting_field_even if episode_index % 2 == 0 else shifting_field_odd
        )

    split_half_correlation = compute_split_half_rate_map_correlations(
        representation,
        position_xy,
        valid_mask=None,
        num_bins_x=num_bins,
        num_bins_y=1,
        smoothing_sigma=0.0,
        min_occupancy=1e-6,
    )
    episode_rate_map_correlation = compute_episode_rate_map_correlations(
        representation,
        position_xy,
        valid_mask=None,
        num_bins_x=num_bins,
        num_bins_y=1,
        smoothing_sigma=0.0,
        min_occupancy=1e-6,
    )

    assert float(split_half_correlation[0]) > 0.99
    assert float(episode_rate_map_correlation[0]) > 0.99
    assert float(split_half_correlation[1]) < float(split_half_correlation[0])
    assert float(episode_rate_map_correlation[1]) < float(episode_rate_map_correlation[0])


def test_support_thresholds_can_mask_under_sampled_bins() -> None:
    position_xy = np.asarray(
        [
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
        ],
        dtype=np.float32,
    )
    representation = np.asarray(
        [
            [[0.0], [1.0], [1.0], [0.0]],
            [[0.0], [1.0], [1.0], [0.0]],
            [[0.0], [1.0], [1.0], [0.0]],
            [[0.0], [1.0], [1.0], [0.0]],
        ],
        dtype=np.float32,
    )

    sparse_valid_mask = np.asarray(
        [
            [False, True, True, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    consistency_maps, _, _ = compute_bin_consistency_maps(
        representation,
        position_xy,
        sparse_valid_mask,
        num_bins_x=4,
        num_bins_y=1,
        minimum_visited_episodes=3,
        active_bin_peak_fraction=0.05,
        active_episode_threshold_fraction_of_bin_mean=0.5,
    )
    split_half_agreement_maps, _ = compute_split_half_agreement_maps(
        representation,
        position_xy,
        sparse_valid_mask,
        num_bins_x=4,
        num_bins_y=1,
        smoothing_sigma=0.0,
        min_occupancy=1e-6,
        minimum_episodes_per_half=2,
    )

    assert np.isnan(consistency_maps[0, 0, 1])
    assert np.isnan(consistency_maps[0, 0, 2])
    assert np.isnan(split_half_agreement_maps[0, 0, 1])
    assert np.isnan(split_half_agreement_maps[0, 0, 2])


def test_revisit_activity_metrics_match_individual_metric_functions() -> None:
    rng = np.random.default_rng(9)
    representation = rng.normal(size=(5, 8, 4)).astype(np.float32)
    position_xy = rng.uniform(low=-1.0, high=1.0, size=(5, 8, 2)).astype(np.float32)
    valid_mask = rng.random(size=(5, 8)) > 0.15
    kwargs = {
        "num_bins_x": 5,
        "num_bins_y": 4,
        "smoothing_sigma": 0.2,
        "min_occupancy": 1e-6,
        "minimum_visited_episodes": 2,
        "active_bin_peak_fraction": 0.05,
        "active_episode_threshold_fraction_of_bin_mean": 0.5,
        "minimum_episodes_per_half": 2,
        "unit_chunk_size": 2,
    }

    combined = compute_revisit_activity_metrics(
        representation,
        position_xy,
        valid_mask,
        **kwargs,
    )
    expected_consistency, expected_cv, expected_visits = compute_bin_consistency_maps(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=kwargs["num_bins_x"],
        num_bins_y=kwargs["num_bins_y"],
        minimum_visited_episodes=kwargs["minimum_visited_episodes"],
        active_bin_peak_fraction=kwargs["active_bin_peak_fraction"],
        active_episode_threshold_fraction_of_bin_mean=(
            kwargs["active_episode_threshold_fraction_of_bin_mean"]
        ),
        unit_chunk_size=kwargs["unit_chunk_size"],
    )
    expected_agreement, expected_support, _even_odd_correlation = (
        compute_split_half_agreement_maps_and_correlations(
            representation,
            position_xy,
            valid_mask,
            num_bins_x=kwargs["num_bins_x"],
            num_bins_y=kwargs["num_bins_y"],
            smoothing_sigma=kwargs["smoothing_sigma"],
            min_occupancy=kwargs["min_occupancy"],
            minimum_episodes_per_half=kwargs["minimum_episodes_per_half"],
            unit_chunk_size=kwargs["unit_chunk_size"],
        )
    )
    expected_split_correlation = compute_split_half_rate_map_correlations(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=kwargs["num_bins_x"],
        num_bins_y=kwargs["num_bins_y"],
        smoothing_sigma=kwargs["smoothing_sigma"],
        min_occupancy=kwargs["min_occupancy"],
        minimum_episodes_per_half=kwargs["minimum_episodes_per_half"],
        unit_chunk_size=kwargs["unit_chunk_size"],
    )
    expected_episode_correlation = compute_episode_rate_map_correlations(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=kwargs["num_bins_x"],
        num_bins_y=kwargs["num_bins_y"],
        smoothing_sigma=kwargs["smoothing_sigma"],
        min_occupancy=kwargs["min_occupancy"],
        unit_chunk_size=kwargs["unit_chunk_size"],
    )

    np.testing.assert_allclose(
        combined.bin_consistency_maps,
        expected_consistency,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        combined.bin_coefficient_of_variation_maps,
        expected_cv,
        equal_nan=True,
    )
    np.testing.assert_allclose(combined.consistency_visit_counts, expected_visits)
    np.testing.assert_allclose(
        combined.split_half_agreement_maps,
        expected_agreement,
        equal_nan=True,
    )
    np.testing.assert_allclose(combined.split_half_agreement_support_counts, expected_support)
    np.testing.assert_allclose(combined.split_half_rate_map_correlation, expected_split_correlation)
    np.testing.assert_allclose(combined.episode_rate_map_correlation, expected_episode_correlation)


def _quantised_activations(seed: int = 0) -> np.ndarray:
    """k-WTA-like activations: mostly exact zeros over a few repeated levels."""
    generator = np.random.default_rng(seed)
    activations = np.zeros((4096, 64), dtype=np.float32)
    levels = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    for unit_index in range(activations.shape[1]):
        active_steps = generator.choice(activations.shape[0], size=80, replace=False)
        activations[active_steps, unit_index] = generator.choice(levels, size=active_steps.size)
    return activations


def test_unit_thresholds_are_computed_in_float64():
    """Thresholds must not depend on the installed numpy major."""
    activations = _quantised_activations()

    for threshold_mode, threshold_fraction, threshold_quantile in (
        ("peak_fraction", 0.2, 0.99),
        ("quantile_per_unit", 1.0, 0.99),
    ):
        thresholds = _compute_global_unit_thresholds(
            activations,
            threshold_mode=threshold_mode,
            threshold_fraction=threshold_fraction,
            threshold_quantile=threshold_quantile,
            use_absolute_activations=False,
        )

        level = 0.995 if threshold_mode == "peak_fraction" else threshold_quantile
        scale = threshold_fraction if threshold_mode == "peak_fraction" else 1.0
        expected = np.maximum(
            np.quantile(activations.astype(np.float64), np.float64(level), axis=0) * scale,
            1e-8,
        ).astype(np.float32)

        np.testing.assert_array_equal(thresholds, expected)


def test_invalid_gap_splits_a_field_traversal() -> None:
    representation = np.array([[[1.0], [0.0], [1.0]]], dtype=np.float32)
    position_xy = np.full((1, 3, 2), 0.5, dtype=np.float32)
    valid_mask = np.array([[True, False, True]], dtype=bool)
    statistics = prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=1,
        num_bins_y=1,
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    assert statistics is not None
    assert statistics.segment_start.tolist() == [True, True]
    field_masks = np.array([[[True]]], dtype=bool)
    _, counts, _, _ = compute_field_traversal_reliability(
        statistics,
        field_masks,
        minimum_traversals=1,
    )
    assert counts.tolist() == [2]


def _direction_independent_traversal_statistics(
    *,
    num_units: int,
    num_sectors: int,
    num_episodes: int,
    traversals_per_sector_per_episode: int,
    firing_probability: float,
    seed: int,
) -> tuple[EpisodeBinStatistics, np.ndarray, np.ndarray]:
    """Traversals whose hit probability is the same in every heading sector."""
    traversals_per_episode = num_sectors * traversals_per_sector_per_episode
    num_steps = 2 * traversals_per_episode
    sector_width = 2.0 * np.pi / num_sectors
    sector_of_traversal = np.tile(np.arange(num_sectors), traversals_per_sector_per_episode)
    step_headings = np.repeat(-np.pi + (sector_of_traversal + 0.5) * sector_width, 2)
    headings = np.tile(step_headings, num_episodes)
    position_xy = np.zeros((num_episodes, num_steps, 2), dtype=np.float32)
    position_xy[..., 0] = np.tile(np.array([0.5, 1.5], dtype=np.float32), traversals_per_episode)
    position_xy[..., 1] = 0.5
    rng = np.random.default_rng(seed)
    hits = rng.random((num_episodes, traversals_per_episode, num_units)) < firing_probability
    representation = np.zeros((num_episodes, num_steps, num_units), dtype=np.float32)
    representation[:, 0::2, :] = hits.astype(np.float32)
    statistics = prepare_episode_bin_statistics(
        representation,
        position_xy,
        None,
        num_bins_x=2,
        num_bins_y=1,
        bounds=((0.0, 2.0), (0.0, 1.0)),
    )
    assert statistics is not None
    field_masks = np.tile(np.array([[[True, False]]], dtype=bool), (num_units, 1, 1))
    return statistics, field_masks, headings


def test_directional_reliability_scores_held_out_episodes() -> None:
    firing_probability = 0.3
    statistics, field_masks, headings = _direction_independent_traversal_statistics(
        num_units=256,
        num_sectors=8,
        num_episodes=10,
        traversals_per_sector_per_episode=1,
        firing_probability=firing_probability,
        seed=20260903,
    )
    _, _, directional, directional_counts = compute_field_traversal_reliability(
        statistics,
        field_masks,
        minimum_traversals=5,
        flat_headings=headings,
        num_heading_sectors=8,
    )
    assert np.all(np.isfinite(directional))
    assert np.all(directional_counts == 5)
    assert abs(float(np.mean(directional)) - firing_probability) < 0.04
