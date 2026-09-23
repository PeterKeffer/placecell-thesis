from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from placecell_research.analysis import helpers
from placecell_research.analysis.helpers import compute_rate_maps_from_episode_statistics
from placecell_research.analysis.reliability_splits import (
    _compute_bin_consistency_chunk,
    compute_episode_rate_map_correlations,
    compute_split_half_agreement_maps_and_correlations,
)
from placecell_research.analysis.shift_nulls import circular_shift_spatial_information_null
from placecell_research.evaluation.decode import (
    fit_ridge_position_decoder,
    score_ridge_position_decoder,
)
from placecell_research.evaluation.metrics import place_code_quality
from placecell_research.evaluation.online import evaluate_representations
from placecell_research.numerics.occupancy import (
    iter_episode_activity_sum_chunks,
    prepare_episode_bin_statistics,
)
from placecell_research.numerics.place_cell_quality import (
    DEFAULT_GATE_MAXIMUM_CONFOUND,
    DEFAULT_GATE_MINIMUM_COHERENCE,
    DEFAULT_GATE_MINIMUM_SPLIT_HALF,
    batched_spatial_coherence,
    benjamini_hochberg,
    coding_purity_score,
    compute_available_confound_scores,
    field_coverage_fraction,
    fraction_place_cells,
    place_cell_pass_mask,
    reliability_weighted_information,
    spatial_coherence,
)
from placecell_research.numerics.rate_map_kernels import (
    PlaceMetricSettings,
    compute_rate_maps,
    prepare_place_metric_rate_maps,
    resolve_place_metric_settings,
    skaggs_spatial_information,
)


def _synthetic_place_activity(
    positions: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    width: float = 0.18,
) -> np.ndarray:
    squared_distance = (positions[..., 0] - center_x) ** 2 + (positions[..., 1] - center_y) ** 2
    return np.exp(-squared_distance / max(width**2, 1e-6)).astype(np.float32)


def test_spatial_coherence_prefers_smooth_maps() -> None:
    smooth_map = np.asarray(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    scrambled_map = np.asarray(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert spatial_coherence(smooth_map) > 0.9
    assert spatial_coherence(scrambled_map) < spatial_coherence(smooth_map)


def test_composite_quality_metrics_clamp_unreliable_inputs() -> None:
    spatial_information_bits = np.asarray([2.0, 1.5, np.nan], dtype=np.float32)
    split_half_correlation = np.asarray([0.8, -0.3, 0.9], dtype=np.float32)
    coherence_scores = np.asarray([0.7, -0.2, 0.6], dtype=np.float32)
    max_available_confound_score = np.asarray([0.1, 1.2, 0.2], dtype=np.float32)

    rwi = reliability_weighted_information(spatial_information_bits, split_half_correlation)
    cps = coding_purity_score(
        spatial_information_bits,
        coherence_scores,
        split_half_correlation,
        max_available_confound_score,
    )

    np.testing.assert_allclose(rwi[:2], np.asarray([1.6, 0.0], dtype=np.float32))
    np.testing.assert_allclose(cps[:2], np.asarray([1.008, 0.0], dtype=np.float32), atol=1e-6)
    assert np.isnan(rwi[2])
    assert np.isnan(cps[2])


def test_place_code_quality_is_bounded_geometric_mean() -> None:
    perfect = place_code_quality(1.0, 1.0, 1.0)
    assert perfect["place_code_quality"] == 1.0
    assert perfect["place_code_accuracy"] == 1.0

    balanced = place_code_quality(0.512, 1.0, 1.0)
    np.testing.assert_allclose(balanced["place_code_quality"], 0.8, atol=1e-6)

    assert place_code_quality(0.9, 0.0, 0.8)["place_code_quality"] == 0.0
    assert place_code_quality(-3.0, 1.0, 1.0)["place_code_accuracy"] == 0.0
    assert np.isnan(place_code_quality(float("nan"), 1.0, 1.0)["place_code_quality"])


def test_fraction_place_cells_reports_per_gate_pass_rates() -> None:
    split_half = np.asarray([0.8, 0.1, 0.9, np.nan], dtype=np.float32)
    coherence = np.asarray([0.7, 0.6, 0.1, 0.5], dtype=np.float32)
    confound = np.asarray([0.2, 0.3, 0.2, 0.1], dtype=np.float32)

    summary = fraction_place_cells(
        split_half, coherence, confound, supported_mask=np.ones(4, dtype=bool)
    )

    np.testing.assert_allclose(summary["fraction_place_cells"], 1.0 / 3.0)
    np.testing.assert_allclose(summary["fraction_passing_split_half"], 2.0 / 3.0)
    np.testing.assert_allclose(summary["fraction_passing_coherence"], 2.0 / 3.0)
    np.testing.assert_allclose(summary["fraction_passing_confound"], 1.0)
    assert summary["place_cell_assessable_units"] == 3.0


def test_field_coverage_fraction_counts_tiled_qualifying_fields() -> None:
    rate_maps = np.zeros((3, 4, 4), dtype=np.float32)
    rate_maps[0, :2, :] = 1.0
    rate_maps[1, 2:, :] = 1.0
    rate_maps[2, :, :] = 1.0
    occupancy = np.ones((4, 4), dtype=np.float32)

    both_halves = field_coverage_fraction(
        rate_maps, occupancy, np.asarray([True, True, False])
    )
    one_half = field_coverage_fraction(rate_maps, occupancy, np.asarray([True, False, False]))
    none_qualifying = field_coverage_fraction(
        rate_maps, occupancy, np.asarray([False, False, False])
    )

    assert both_halves == 1.0
    assert one_half == 0.5
    assert none_qualifying == 0.0


def test_confound_scores_partial_out_position_coupled_heading() -> None:
    steps = 400
    x_positions = np.tile(np.linspace(-1.0, 1.0, 20, dtype=np.float32), steps // 20)
    position_xy = np.stack([x_positions, np.zeros_like(x_positions)], axis=-1)[None]
    heading = (x_positions * 0.9 * np.pi)[None]
    place_tuned_unit = x_positions.astype(np.float32)[None, :, None]

    scores = compute_available_confound_scores(
        place_tuned_unit,
        position_xy,
        valid_mask=np.ones((1, steps), dtype=bool),
        heading=heading,
    )

    assert float(scores["heading_score"][0]) < 0.1


def test_confound_scores_missing_covariates_are_nan_not_zero() -> None:
    rng = np.random.default_rng(3)
    representation = rng.normal(size=(1, 60, 4)).astype(np.float32)
    position_xy = rng.uniform(-1.0, 1.0, size=(1, 60, 2)).astype(np.float32)

    scores = compute_available_confound_scores(
        representation,
        position_xy,
        valid_mask=np.ones((1, 60), dtype=bool),
    )

    assert np.isnan(scores["heading_score"]).all()
    assert np.isnan(scores["step_displacement_score"]).all()
    assert np.isfinite(scores["time_score"]).all()
    assert np.isfinite(scores["max_available_confound_score"]).all()


def test_heading_confound_score_is_rotation_invariant() -> None:
    steps = 120
    headings = np.linspace(0.0, 6.0 * np.pi, num=steps, dtype=np.float32).reshape(1, -1)
    aligned_unit = np.sin(headings[0])
    rotated_unit = np.sin(headings[0] - np.pi / 4.0)
    representation = np.stack([aligned_unit, rotated_unit], axis=-1)[None].astype(np.float32)
    position_xy = np.zeros((1, steps, 2), dtype=np.float32)

    scores = compute_available_confound_scores(
        representation,
        position_xy,
        valid_mask=np.ones((1, steps), dtype=bool),
        heading=headings,
    )

    assert float(scores["heading_score"][0]) > 0.99
    assert float(scores["heading_score"][1]) > 0.99
    np.testing.assert_allclose(scores["heading_score"][0], scores["heading_score"][1], atol=0.01)


def test_benjamini_hochberg_rejects_small_p_and_ignores_nan() -> None:
    p_values = np.asarray([0.001, 0.002, 0.6, np.nan], dtype=np.float32)
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected.tolist() == [True, True, False, False]
    assert not benjamini_hochberg(np.asarray([0.3, 0.5, 0.9]), alpha=0.05).any()


def test_circular_shift_null_separates_place_cell_from_noise() -> None:
    rng = np.random.default_rng(11)
    episodes, steps = 8, 60
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index * 3)], axis=-1
        )
        position_sequences.append(positions)
        representation_sequences.append(
            np.stack(
                [
                    _synthetic_place_activity(positions, center_x=-0.3, center_y=0.0, width=0.3),
                    rng.random(steps).astype(np.float32),
                ],
                axis=-1,
            )
        )
    representation = np.stack(representation_sequences, axis=0)
    positions_xy = np.stack(position_sequences, axis=0)
    valid_mask = np.ones((episodes, steps), dtype=bool)

    rate_map_result = compute_rate_maps(
        representation,
        positions_xy,
        valid_mask,
        num_bins_x=12,
        num_bins_y=12,
        smoothing_sigma=0.3,
        min_occupancy=1e-6,
    )
    observed = np.asarray(
        skaggs_spatial_information(rate_map_result.rate_maps, rate_map_result.occupancy),
        dtype=np.float32,
    )
    null_matrix = circular_shift_spatial_information_null(
        representation,
        positions_xy,
        valid_mask,
        num_bins_x=12,
        num_bins_y=12,
        smoothing_sigma=0.3,
        min_occupancy=1e-6,
        bounds=rate_map_result.bounds,
        num_shuffles=40,
        rng_seed=0,
    )

    assert null_matrix.shape == (40, 2)
    place_null = null_matrix[np.isfinite(null_matrix[:, 0]), 0]
    noise_null = null_matrix[np.isfinite(null_matrix[:, 1]), 1]
    assert observed[0] > place_null.max()
    assert observed[1] <= noise_null.max()


def test_transfer_decoder_scores_held_out_data_without_refit() -> None:
    rng = np.random.default_rng(5)
    fit_positions = rng.uniform(-1.0, 1.0, size=(400, 2)).astype(np.float32)
    held_out_positions = rng.uniform(-1.0, 1.0, size=(200, 2)).astype(np.float32)
    fit_codes = np.concatenate(
        [fit_positions, rng.normal(scale=0.05, size=(400, 3)).astype(np.float32)], axis=-1
    )
    held_out_codes = np.concatenate(
        [held_out_positions, rng.normal(scale=0.05, size=(200, 3)).astype(np.float32)], axis=-1
    )

    decoder = fit_ridge_position_decoder(fit_codes, fit_positions, alpha=1e-3)
    transfer_rmse, transfer_r2 = score_ridge_position_decoder(
        decoder, held_out_codes, held_out_positions
    )

    assert transfer_r2 > 0.95
    assert transfer_rmse < 0.2


def test_heading_confound_score_uses_circular_heading_components() -> None:
    headings = np.linspace(0.0, 4.0 * np.pi, num=80, dtype=np.float32).reshape(1, -1)
    representation = np.sin(headings)[..., None].astype(np.float32)
    position_xy = np.zeros((1, headings.shape[1], 2), dtype=np.float32)

    scores = compute_available_confound_scores(
        representation,
        position_xy,
        valid_mask=np.ones_like(headings, dtype=bool),
        heading=headings,
    )

    assert float(scores["heading_score"][0]) > 0.99


def test_compute_rate_maps_avoids_unbuffered_add_at(monkeypatch) -> None:
    class _AddSentinel:
        @staticmethod
        def at(*args, **kwargs):
            del args, kwargs
            raise AssertionError(
                "compute_rate_maps should use bincount accumulation, not np.add.at."
            )

    monkeypatch.setattr(helpers.np, "add", _AddSentinel)
    representation = np.asarray(
        [
            [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
            [[4.0, 3.0], [5.0, 4.0], [6.0, 5.0]],
        ],
        dtype=np.float32,
    )
    position_xy = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    result = compute_rate_maps(
        representation,
        position_xy,
        valid_mask=np.ones((2, 3), dtype=bool),
        num_bins_x=3,
        num_bins_y=2,
        smoothing_sigma=0.0,
        min_occupancy=1.0,
        unit_chunk_size=1,
    )

    assert result.rate_maps.shape == (2, 2, 3)


def test_compute_rate_maps_uses_one_activity_bincount_per_chunk(monkeypatch) -> None:
    bincount_calls = 0
    original_bincount = helpers.np.bincount

    def counted_bincount(*args, **kwargs):
        nonlocal bincount_calls
        bincount_calls += 1
        return original_bincount(*args, **kwargs)

    monkeypatch.setattr(helpers.np, "bincount", counted_bincount)
    representation = np.asarray(
        [
            [[1.0, 0.0, 0.5], [2.0, 1.0, 1.5], [3.0, 2.0, 2.5]],
            [[4.0, 3.0, 3.5], [5.0, 4.0, 4.5], [6.0, 5.0, 5.5]],
        ],
        dtype=np.float32,
    )
    position_xy = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    result = compute_rate_maps(
        representation,
        position_xy,
        valid_mask=np.ones((2, 3), dtype=bool),
        num_bins_x=3,
        num_bins_y=2,
        smoothing_sigma=0.0,
        min_occupancy=1.0,
        unit_chunk_size=16,
    )

    assert result.rate_maps.shape == (3, 2, 3)
    assert bincount_calls == 2


def test_compute_rate_maps_from_episode_statistics_matches_canonical_path() -> None:
    rng = np.random.default_rng(7)
    representation = rng.normal(size=(4, 5, 3)).astype(np.float32)
    position_xy = rng.uniform(-1.0, 1.0, size=(4, 5, 2)).astype(np.float32)
    valid_mask = rng.random(size=(4, 5)) > 0.2
    statistics = prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=4,
        num_bins_y=3,
    )
    assert statistics is not None

    canonical = compute_rate_maps(
        representation,
        position_xy,
        valid_mask=valid_mask,
        num_bins_x=4,
        num_bins_y=3,
        smoothing_sigma=0.2,
        min_occupancy=1e-6,
        bounds=statistics.bounds,
        unit_chunk_size=2,
    )
    from_statistics = compute_rate_maps_from_episode_statistics(
        statistics,
        smoothing_sigma=0.2,
        min_occupancy=1e-6,
        unit_chunk_size=2,
    )

    np.testing.assert_allclose(from_statistics.occupancy, canonical.occupancy)
    np.testing.assert_allclose(from_statistics.rate_maps, canonical.rate_maps)
    assert from_statistics.bounds == canonical.bounds


def test_episode_activity_sum_chunks_use_one_bincount_per_chunk(monkeypatch) -> None:
    representation = np.asarray(
        [
            [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
            [[4.0, 3.0], [5.0, 4.0], [6.0, 5.0]],
        ],
        dtype=np.float32,
    )
    position_xy = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    statistics = prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask=np.ones((2, 3), dtype=bool),
        num_bins_x=3,
        num_bins_y=2,
    )
    assert statistics is not None
    bincount_calls = 0

    original_bincount = helpers.np.bincount

    def counted_bincount(*args, **kwargs):
        nonlocal bincount_calls
        bincount_calls += 1
        return original_bincount(*args, **kwargs)

    monkeypatch.setattr(helpers.np, "bincount", counted_bincount)

    chunks = list(iter_episode_activity_sum_chunks(statistics, unit_chunk_size=16))

    assert len(chunks) == 1
    assert chunks[0][2].shape == (2, 2, 6)
    assert bincount_calls == 1


def test_episode_rate_map_correlations_batch_smoothing(monkeypatch) -> None:
    episodes = 4
    steps = 10
    units = 3
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.5, 0.5, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index)],
            axis=-1,
        )
        position_sequences.append(positions)
        representation_sequences.append(
            np.stack(
                [
                    _synthetic_place_activity(positions, center_x=-0.3, center_y=-0.1),
                    _synthetic_place_activity(positions, center_x=0.35, center_y=0.2),
                    np.linspace(0.1, 0.9, steps, dtype=np.float32),
                ],
                axis=-1,
            )
        )
    assert len(representation_sequences[0][0]) == units

    gaussian_calls = 0
    original_gaussian_filter = helpers.gaussian_filter

    def counted_gaussian_filter(*args, **kwargs):
        nonlocal gaussian_calls
        gaussian_calls += 1
        return original_gaussian_filter(*args, **kwargs)

    monkeypatch.setattr(helpers, "gaussian_filter", counted_gaussian_filter)

    correlations = compute_episode_rate_map_correlations(
        np.stack(representation_sequences, axis=0),
        np.stack(position_sequences, axis=0),
        valid_mask=np.ones((episodes, steps), dtype=bool),
        num_bins_x=8,
        num_bins_y=8,
        smoothing_sigma=0.4,
        min_occupancy=1e-6,
    )

    assert correlations.shape == (units,)
    assert gaussian_calls <= 6


def test_split_half_agreement_reuses_smoothed_occupancy(monkeypatch) -> None:
    episodes = 4
    steps = 12
    units = 40
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.5, 0.5, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index)],
            axis=-1,
        )
        position_sequences.append(positions)
        base_activity = _synthetic_place_activity(positions, center_x=0.1, center_y=-0.1)
        representation_sequences.append(
            np.stack(
                [
                    base_activity * (1.0 + 0.01 * unit_index)
                    for unit_index in range(units)
                ],
                axis=-1,
            )
        )

    gaussian_calls = 0
    original_gaussian_filter = helpers.gaussian_filter

    def counted_gaussian_filter(*args, **kwargs):
        nonlocal gaussian_calls
        gaussian_calls += 1
        return original_gaussian_filter(*args, **kwargs)

    monkeypatch.setattr(helpers, "gaussian_filter", counted_gaussian_filter)

    agreement_maps, _, correlations = compute_split_half_agreement_maps_and_correlations(
        np.stack(representation_sequences, axis=0),
        np.stack(position_sequences, axis=0),
        valid_mask=np.ones((episodes, steps), dtype=bool),
        num_bins_x=8,
        num_bins_y=8,
        smoothing_sigma=0.4,
        min_occupancy=1e-6,
        unit_chunk_size=8,
    )

    assert agreement_maps.shape == (units, 8, 8)
    assert correlations.shape == (units,)
    assert gaussian_calls <= 12


def test_bin_consistency_chunk_avoids_per_unit_episode_mean_allocations(monkeypatch) -> None:
    chunk_activity_sums = np.asarray(
        [
            [[1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0]],
            [[0.2, 0.0, 1.0, 0.0], [1.0, 0.0, 0.2, 0.0], [0.2, 0.0, 1.0, 0.0]],
            [[0.5, 0.0, 0.5, 0.0], [0.5, 0.0, 0.5, 0.0], [0.5, 0.0, 0.5, 0.0]],
            [[0.1, 0.0, 0.1, 0.0], [0.2, 0.0, 0.2, 0.0], [0.3, 0.0, 0.3, 0.0]],
        ],
        dtype=np.float32,
    )
    step_counts = np.ones((3, 4), dtype=np.float32)
    visited_mask_by_episode = step_counts > 0
    episode_mean_allocations = 0
    original_full = helpers.np.full

    def counted_full(shape, *args, **kwargs):
        nonlocal episode_mean_allocations
        if tuple(shape) == (3, 4):
            episode_mean_allocations += 1
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(helpers.np, "full", counted_full)

    consistency_maps, coefficient_of_variation_maps = _compute_bin_consistency_chunk(
        chunk_activity_sums,
        step_counts,
        visited_mask_by_episode,
        num_bins_y=1,
        num_bins_x=4,
        minimum_visited_episodes=2,
        active_bin_peak_fraction=0.05,
        active_episode_threshold_fraction_of_bin_mean=0.5,
        epsilon=1e-6,
    )

    assert consistency_maps.shape == (4, 1, 4)
    assert coefficient_of_variation_maps.shape == (4, 1, 4)
    assert episode_mean_allocations <= 1


def test_evaluate_representations_exports_rwi_and_cps_summaries() -> None:
    episodes = 6
    steps = 48
    position_sequences = []
    representation_sequences = []
    heading = np.zeros((episodes, steps), dtype=np.float32)
    kinematics = np.zeros((episodes, steps, 2), dtype=np.float32)
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = 0.7 * np.sin(np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32))
    for episode_index in range(episodes):
        positions = np.stack([x_positions, y_positions], axis=-1)
        position_sequences.append(positions)
        representation_sequences.append(
            np.stack(
                [
                    _synthetic_place_activity(positions, center_x=-0.35, center_y=-0.15),
                    _synthetic_place_activity(positions, center_x=0.45, center_y=0.2),
                ],
                axis=-1,
            )
        )
        kinematics[episode_index, :, 0] = 0.1

    representations = {"encoder.place_codes": np.stack(representation_sequences, axis=0)}
    positions_xy = np.stack(position_sequences, axis=0)
    valid_mask = np.ones((episodes, steps), dtype=bool)
    rate_map_result = compute_rate_maps(
        representations["encoder.place_codes"],
        positions_xy,
        valid_mask,
        num_bins_x=12,
        num_bins_y=12,
        smoothing_sigma=0.6,
        min_occupancy=1e-6,
    )
    spatial_information_scores = {
        "encoder.place_codes": np.asarray(
            skaggs_spatial_information(rate_map_result.rate_maps, rate_map_result.occupancy),
            dtype=np.float32,
        )
    }

    result = evaluate_representations(
        representations,
        positions_xy,
        valid_mask=valid_mask,
        kinematics=kinematics,
        heading=heading,
        train_fraction=0.5,
        ridge_alpha=1e-3,
        include_shuffle=False,
        spatial_information_scores=spatial_information_scores,
        spatial_information_top_k=2,
        rate_map_num_bins_x=12,
        rate_map_num_bins_y=12,
        rate_map_smoothing_sigma=0.6,
        rate_map_min_occupancy=1e-6,
    )[0]

    assert result.place_cell_quality is not None
    assert result.place_cell_quality["reliability_weighted_information"]["mean_top_k"] >= 0.0
    assert result.place_cell_quality["coding_purity_score"]["mean_top_k"] >= 0.0
    flattened_metrics = result.to_metrics()
    assert "encoder.place_codes.reliability_weighted_information_mean_top_k" in flattened_metrics
    assert "encoder.place_codes.coding_purity_score_mean_top_k" in flattened_metrics
    assert "encoder.place_codes.place_code_quality" in flattened_metrics
    assert "encoder.place_codes.place_code_accuracy" in flattened_metrics
    assert "encoder.place_codes.place_code_fraction_place_cells" in flattened_metrics
    assert "encoder.place_codes.place_code_field_coverage" in flattened_metrics
    quality = flattened_metrics["encoder.place_codes.place_code_quality"]
    assert np.isnan(quality) or 0.0 <= quality <= 1.0
    assert batched_spatial_coherence(rate_map_result.rate_maps).shape == (2,)


def test_evaluate_representations_summarizes_sparsity_on_valid_steps_only() -> None:
    representations = {
        "encoder.place_codes": np.asarray(
            [
                [[1.0], [1.0], [0.0], [0.0]],
                [[1.0], [1.0], [0.0], [0.0]],
            ],
            dtype=np.float32,
        )
    }
    positions_xy = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [9.0, 9.0]],
            [[0.0, 1.0], [1.0, 1.0], [9.0, 9.0], [9.0, 9.0]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.asarray(
        [[True, True, False, False], [True, True, False, False]],
        dtype=bool,
    )

    result = evaluate_representations(
        representations,
        positions_xy,
        valid_mask=valid_mask,
        train_fraction=0.5,
        ridge_alpha=1e-3,
        include_shuffle=False,
    )[0]

    assert result.sparsity["fraction_active"] == 1.0


def test_evaluate_representations_exports_nonlinear_decode_metrics() -> None:
    rng = np.random.default_rng(12)
    episodes = 8
    steps = 40
    flat_features = rng.uniform(-1.0, 1.0, size=(episodes * steps, 2)).astype(np.float32)
    flat_positions = np.stack(
        [
            flat_features[:, 0] * flat_features[:, 1],
            np.square(flat_features[:, 0]) - np.square(flat_features[:, 1]),
        ],
        axis=-1,
    ).astype(np.float32)
    representations = {"encoder.place_codes": flat_features.reshape(episodes, steps, 2)}
    positions_xy = flat_positions.reshape(episodes, steps, 2)

    result = evaluate_representations(
        representations,
        positions_xy,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        train_fraction=0.75,
        ridge_alpha=1e-3,
        include_shuffle=False,
        nonlinear_decode_enabled=True,
        nonlinear_decode_hidden_sizes=(64, 64),
        nonlinear_decode_max_epochs=180,
        nonlinear_decode_batch_size=128,
        nonlinear_decode_random_seed=13,
    )[0]

    metrics = result.to_metrics()
    assert "encoder.place_codes.nonlinear_decode_rmse" in metrics
    assert "encoder.place_codes.nonlinear_decode_r2" in metrics
    assert metrics["encoder.place_codes.nonlinear_decode_r2"] > metrics[
        "encoder.place_codes.decode_r2"
    ]


def test_evaluate_representations_skips_decode_with_one_episode() -> None:
    representations = {
        "encoder.place_codes": np.asarray(
            [[[1.0], [0.8], [0.2], [0.0]]],
            dtype=np.float32,
        )
    }
    positions_xy = np.asarray(
        [[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]],
        dtype=np.float32,
    )

    result = evaluate_representations(
        representations,
        positions_xy,
        valid_mask=np.ones((1, 4), dtype=bool),
        train_fraction=0.5,
        ridge_alpha=1e-3,
        include_shuffle=False,
    )[0]

    assert result.decode is None
    assert "encoder.place_codes.decode_rmse" not in result.to_metrics()
    assert result.sparsity["fraction_active"] > 0.0


def test_reliability_lift_zeroes_dense_units_and_keeps_fields() -> None:
    from placecell_research.analysis.reliability_splits import compute_reliability_lift_maps

    dense_unit = np.ones((4, 4), dtype=np.float32)
    field_unit = np.zeros((4, 4), dtype=np.float32)
    field_unit[:2, :2] = 1.0
    reliability_maps = np.stack([dense_unit, field_unit], axis=0)
    visited_counts = np.full((4, 4), 5.0, dtype=np.float32)

    lift = compute_reliability_lift_maps(reliability_maps, visited_counts)

    np.testing.assert_allclose(lift[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(lift[1][:2, :2], 0.75, atol=1e-6)
    np.testing.assert_allclose(lift[1][2:, 2:], -0.25, atol=1e-6)


def test_map_correlation_below_min_overlap_is_nan() -> None:
    from placecell_research.numerics.bin_maps import batched_masked_map_correlation

    sparse_first = np.full((1, 6, 6), np.nan, dtype=np.float32)
    sparse_second = np.full((1, 6, 6), np.nan, dtype=np.float32)
    sparse_first[0, 0, :3] = [1.0, 2.0, 3.0]
    sparse_second[0, 0, :3] = [1.0, 2.0, 3.0]

    dense_first = np.full((1, 6, 6), np.nan, dtype=np.float32)
    dense_second = np.full((1, 6, 6), np.nan, dtype=np.float32)
    dense_first[0, :2, :] = np.arange(12, dtype=np.float32).reshape(2, 6)
    dense_second[0, :2, :] = np.arange(12, dtype=np.float32).reshape(2, 6)

    assert np.isnan(batched_masked_map_correlation(sparse_first, sparse_second)[0])
    np.testing.assert_allclose(
        batched_masked_map_correlation(dense_first, dense_second)[0], 1.0, atol=1e-6
    )


def _smooth_signed_gradient(bins_y: int = 20, bins_x: int = 20) -> np.ndarray:
    """A signed map that is maximally coherent and half negative."""
    _, columns = np.mgrid[0:bins_y, 0:bins_x]
    gradient = (columns - (bins_x - 1) / 2.0).astype(np.float32)
    return gaussian_filter(gradient, sigma=1.0).astype(np.float32)[None, :, :]


def test_unsupported_signed_map_cannot_pass_the_place_cell_gate() -> None:
    rate_maps = _smooth_signed_gradient()
    supported = prepare_place_metric_rate_maps(rate_maps).supported_mask
    coherence = batched_spatial_coherence(rate_maps)
    split_half = np.asarray([0.99], dtype=np.float32)
    confound = np.asarray([0.0], dtype=np.float32)

    assert not bool(supported[0])
    assert coherence[0] > 0.99

    passes, assessable, *_ = place_cell_pass_mask(
        split_half, coherence, confound, supported_mask=supported
    )
    summary = fraction_place_cells(
        split_half, coherence, confound, supported_mask=supported
    )

    assert not bool(passes[0])
    assert not bool(assessable[0])
    assert np.isnan(summary["fraction_place_cells"])
    assert summary["place_cell_assessable_units"] == 0.0
    assert (
        field_coverage_fraction(rate_maps, np.ones((20, 20), dtype=np.float32), passes) == 0.0
    )


def test_supported_map_still_passes_the_place_cell_gate() -> None:
    rate_maps = np.zeros((1, 20, 20), dtype=np.float32)
    rate_maps[0, 8:12, 8:12] = 1.0
    rate_maps[0] = gaussian_filter(rate_maps[0], sigma=1.0)
    supported = prepare_place_metric_rate_maps(rate_maps).supported_mask
    coherence = batched_spatial_coherence(rate_maps)

    passes, assessable, *_ = place_cell_pass_mask(
        np.asarray([0.99], dtype=np.float32),
        coherence,
        np.asarray([0.0], dtype=np.float32),
        supported_mask=supported,
    )

    assert bool(supported[0])
    assert bool(assessable[0])
    assert bool(passes[0])


def test_field_coverage_reads_visited_bins_before_smoothing() -> None:
    raw_occupancy = np.zeros((9, 9), dtype=np.float32)
    raw_occupancy[4, 4] = 100.0
    smoothed_occupancy = gaussian_filter(raw_occupancy, sigma=1.0).astype(np.float32)
    rate_maps = np.zeros((1, 9, 9), dtype=np.float32)
    rate_maps[0, 4, 4] = 1.0
    qualifying = np.asarray([True])

    assert int(np.count_nonzero(raw_occupancy > 0.0)) == 1
    assert int(np.count_nonzero(smoothed_occupancy > 0.0)) == 81

    assert field_coverage_fraction(rate_maps, raw_occupancy, qualifying) == 1.0


def test_rate_map_computation_carries_unsmoothed_occupancy() -> None:
    representation = np.ones((2, 5, 1), dtype=np.float32)
    positions = np.zeros((2, 5, 2), dtype=np.float32)
    result = compute_rate_maps(
        representation,
        positions,
        None,
        num_bins_x=9,
        num_bins_y=9,
        smoothing_sigma=1.0,
        min_occupancy=1e-6,
        bounds=((0.0, 9.0), (0.0, 9.0)),
    )

    assert int(np.count_nonzero(result.raw_occupancy > 0.0)) == 1
    assert int(np.count_nonzero(result.occupancy > 0.0)) > 1
    assert float(result.raw_occupancy.sum()) == 10.0


def test_place_cell_gate_defaults_match_the_analysis_config() -> None:
    """Both paths resolve the gate from the config; these are what an unset config resolves to."""
    from placecell_research.config.schema import AnalysisConfig

    config = AnalysisConfig()
    assert config.place_cell_gate_minimum_split_half == DEFAULT_GATE_MINIMUM_SPLIT_HALF
    assert config.place_cell_gate_minimum_coherence == DEFAULT_GATE_MINIMUM_COHERENCE
    assert config.place_cell_gate_maximum_confound == DEFAULT_GATE_MAXIMUM_CONFOUND


def test_analysis_config_gate_thresholds_reach_the_rate_map_settings() -> None:
    from placecell_research.analysis.rate_map_metrics import (
        resolve_rate_map_metric_settings,
    )

    settings = resolve_rate_map_metric_settings({"place_cell_gate_minimum_split_half": 0.3})

    assert settings.place_cell_gate_minimum_split_half == 0.3
    assert settings.place_cell_gate_minimum_coherence == DEFAULT_GATE_MINIMUM_COHERENCE


def test_absent_null_shuffle_key_draws_the_analysis_config_default() -> None:
    """No shipped config sets this key, so the module fallback is what every run actually draws."""
    from placecell_research.analysis.rate_map_metrics import (
        resolve_rate_map_metric_settings,
    )
    from placecell_research.config.schema import AnalysisConfig

    settings = resolve_rate_map_metric_settings({})

    assert settings.null_num_shuffles == AnalysisConfig().spatial_information_null_shuffles
    overridden = resolve_rate_map_metric_settings({"spatial_information_null_shuffles": 7})
    assert overridden.null_num_shuffles == 7


_GATE_NUM_BINS = 12
_GATE_SMOOTHING = 0.4


def _noisy_place_code_dataset(noise_scale: float = 0.45):
    """Three place-coding units whose split-half correlation sits between 0.5 and 0.8."""
    rng = np.random.default_rng(7)
    positions = rng.uniform(0.5, 19.5, size=(16, 96, 2)).astype(np.float32)
    centers = np.array([[6.0, 6.0], [14.0, 14.0], [6.0, 14.0]], dtype=np.float32)
    distances = np.linalg.norm(positions[..., None, :] - centers[None, None], axis=-1)
    fields = np.exp(-0.5 * (distances / 3.0) ** 2).astype(np.float32)
    noisy = fields + noise_scale * rng.normal(size=fields.shape).astype(np.float32)
    return np.clip(noisy, 0.0, None), positions


def _offline_fraction_place_cells(representation, positions, config) -> float:
    from placecell_research.analysis.base import AnalysisInput
    from placecell_research.analysis.rate_map_metrics import compute_rate_map_metric_bundle

    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones(representation.shape[:2], dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="test",
    )
    bundle = compute_rate_map_metric_bundle(analysis_input, config, bounds=None)
    return float(bundle.place_metrics.gate_summary["fraction_place_cells"])


def _online_fraction_place_cells(representation, positions, config) -> float:
    from placecell_research.numerics.place_cell_quality import (
        resolve_place_cell_gate_thresholds,
    )

    valid_mask = np.ones(representation.shape[:2], dtype=bool)
    rate_map_result = compute_rate_maps(
        representation,
        positions,
        valid_mask,
        num_bins_x=_GATE_NUM_BINS,
        num_bins_y=_GATE_NUM_BINS,
        smoothing_sigma=_GATE_SMOOTHING,
        min_occupancy=1e-6,
    )
    result = evaluate_representations(
        {"source": representation},
        positions,
        valid_mask=valid_mask,
        train_fraction=0.8,
        ridge_alpha=1e-3,
        include_shuffle=False,
        spatial_information_scores={
            "source": np.ones(representation.shape[-1], dtype=np.float32)
        },
        rate_map_results={"source": rate_map_result},
        rate_map_num_bins_x=_GATE_NUM_BINS,
        rate_map_num_bins_y=_GATE_NUM_BINS,
        rate_map_smoothing_sigma=_GATE_SMOOTHING,
        rate_map_min_occupancy=1e-6,
        place_cell_gate_thresholds=resolve_place_cell_gate_thresholds(config),
    )[0]
    return float(result.population_metrics["place_code_fraction_place_cells"])


def test_configured_gate_thresholds_reach_the_online_path_as_well_as_the_offline_one() -> None:
    """One configured gate, one answer, whichever path measured it."""
    representation, positions = _noisy_place_code_dataset()
    base_config = {
        "num_bins_x": _GATE_NUM_BINS,
        "num_bins_y": _GATE_NUM_BINS,
        "smoothing_sigma": _GATE_SMOOTHING,
        "min_occupancy": 1e-6,
        "spatial_information_null_shuffles": 4,
    }
    configured = {**base_config, "place_cell_gate_minimum_split_half": 0.5}

    assert _offline_fraction_place_cells(representation, positions, configured) == 1.0
    assert _online_fraction_place_cells(representation, positions, configured) == 1.0
    assert _offline_fraction_place_cells(representation, positions, base_config) == 0.0
    assert _online_fraction_place_cells(representation, positions, base_config) == 0.0


def test_gate_thresholds_resolve_from_a_mapping_and_from_the_analysis_config() -> None:
    """The analysis stage hands a dict of knobs; the evaluation stages hand AnalysisConfig."""
    from placecell_research.config.schema import AnalysisConfig
    from placecell_research.numerics.place_cell_quality import (
        resolve_place_cell_gate_thresholds,
    )

    from_mapping = resolve_place_cell_gate_thresholds(
        {"place_cell_gate_minimum_split_half": 0.5, "place_cell_gate_maximum_confound": 0.1}
    )
    assert from_mapping.minimum_split_half == 0.5
    assert from_mapping.maximum_confound == 0.1
    assert from_mapping.minimum_coherence == DEFAULT_GATE_MINIMUM_COHERENCE

    from_config = resolve_place_cell_gate_thresholds(
        AnalysisConfig(place_cell_gate_minimum_coherence=0.4)
    )
    assert from_config.minimum_coherence == 0.4
    assert from_config.minimum_split_half == DEFAULT_GATE_MINIMUM_SPLIT_HALF
    assert from_config.maximum_confound == DEFAULT_GATE_MAXIMUM_CONFOUND


def _signed_two_unit_dataset():
    """Two place-like units carrying a negative skirt: their support depends on the tolerance."""
    episodes, steps = 6, 48
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = 0.7 * np.sin(np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32))
    positions = np.stack([x_positions, y_positions], axis=-1)
    unit_a = _synthetic_place_activity(positions, center_x=-0.35, center_y=-0.15) - 0.25
    unit_b = _synthetic_place_activity(positions, center_x=0.45, center_y=0.2) - 0.25
    representation = np.stack([unit_a, unit_b], axis=-1)
    return (
        np.repeat(representation[None], episodes, axis=0),
        np.repeat(positions[None], episodes, axis=0),
        np.ones((episodes, steps), dtype=bool),
    )


def _assessable_units(place_metric_settings) -> float:
    representation, positions, valid_mask = _signed_two_unit_dataset()
    rate_map_result = compute_rate_maps(
        representation, positions, valid_mask, num_bins_x=8, num_bins_y=8,
        smoothing_sigma=0.0, min_occupancy=1e-6,
    )
    result = evaluate_representations(
        {"encoder.place_codes": representation},
        positions,
        valid_mask=valid_mask,
        train_fraction=0.5,
        ridge_alpha=1e-3,
        include_shuffle=False,
        spatial_information_scores={
            "encoder.place_codes": np.asarray(
                skaggs_spatial_information(
                    rate_map_result.rate_maps, rate_map_result.occupancy
                ),
                dtype=np.float32,
            )
        },
        rate_map_results={"encoder.place_codes": rate_map_result},
        place_metric_settings=place_metric_settings,
    )[0]
    return float(result.population_metrics["place_cell_assessable_units"])


def test_configured_signed_support_tolerance_changes_the_online_metric() -> None:
    """The online path must measure support with the run's own tolerances, not the defaults."""
    strict = _assessable_units(
        PlaceMetricSettings(max_negative_bin_fraction=0.0, max_negative_peak_fraction=0.0)
    )
    permissive = _assessable_units(
        PlaceMetricSettings(max_negative_bin_fraction=1.0, max_negative_peak_fraction=10.0)
    )

    assert strict == 0.0
    assert permissive == 2.0


def test_place_metric_settings_come_from_the_analysis_config() -> None:
    from placecell_research.config.schema import AnalysisConfig

    analysis_config = AnalysisConfig()
    analysis_config.place_field_threshold_fraction = 0.55
    analysis_config.place_metric_max_negative_bin_fraction = 0.4

    settings = resolve_place_metric_settings(analysis_config)

    assert settings.field_threshold_fraction == pytest.approx(0.55)
    assert settings.max_negative_bin_fraction == pytest.approx(0.4)
    assert resolve_place_metric_settings(
        {"place_field_threshold_fraction": 0.3}
    ).field_threshold_fraction == pytest.approx(0.3)
