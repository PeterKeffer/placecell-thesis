from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from placecell_research.analysis import helpers, occupancy, place_cell_quality
from placecell_research.analysis import rate_maps as rate_map_module
from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.figures import figure_to_rgb_array
from placecell_research.analysis.registry import (
    run_analysis_modules,
)
from placecell_research.analysis.sparsity_metrics import SparsityModule
from placecell_research.numerics import rate_map_kernels


def _synthetic_input() -> AnalysisInput:
    rng = np.random.default_rng(0)
    representation = rng.normal(size=(8, 16, 6)).astype(np.float32)
    position_xy = rng.normal(size=(8, 16, 2)).astype(np.float32)
    valid_mask = np.ones((8, 16), dtype=bool)
    rgb = rng.integers(0, 255, size=(8, 16, 3, 8, 8), dtype=np.uint8).astype(np.float32)
    return AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        label="synthetic",
        split_name="test",
        rgb=rgb,
    )


def test_run_analysis_modules_executes_requested_modules(tmp_path) -> None:
    completed_modules: list[str] = []
    results = run_analysis_modules(
        _synthetic_input(),
        tmp_path,
        {
            "num_bins_x": 20,
            "num_bins_y": 20,
            "smoothing_sigma": 0.0,
            "transition_geometry_num_bins_x": 8,
            "transition_geometry_num_bins_y": 8,
            "transition_geometry_num_modes": 3,
        },
        [
            "dataset_coverage",
            "sparsity",
            "decode_xy",
            "episode_dynamics",
            "transition_geometry_graph",
            "transition_geometry_alignment",
            "transition_geometry_panel",
        ],
        progress_callback=lambda module_name: completed_modules.append(module_name),
    )
    assert set(results) == {
        "dataset_coverage",
        "sparsity",
        "decode_xy",
        "episode_dynamics",
        "transition_geometry_graph",
        "transition_geometry_alignment",
        "transition_geometry_panel",
    }
    assert completed_modules == [
        "dataset_coverage",
        "sparsity",
        "decode_xy",
        "episode_dynamics",
        "transition_geometry_graph",
        "transition_geometry_alignment",
        "transition_geometry_panel",
    ]
    assert "occupied_bins_fraction" in results["dataset_coverage"].metrics
    assert results["dataset_coverage"].metadata["visualization"] == "occupancy_heatmap"
    assert results["dataset_coverage"].metadata["analysis_module_timing_seconds"]["total"] >= 0.0
    assert "population_activity_fraction" in results["sparsity"].metrics
    assert "decode_rmse" in results["decode_xy"].metrics
    assert results["decode_xy"].metadata["analysis_module_timing_seconds"]["total"] >= 0.0
    assert "selected_episode_length" in results["episode_dynamics"].metrics
    assert "transition_geometry_transitions_used" in results["transition_geometry_graph"].metrics
    assert "transition_geometry_max_unit_mode_abs_correlation" in (
        results["transition_geometry_alignment"].metrics
    )
    assert "transition_geometry" in results["transition_geometry_panel"].figures


def test_analysis_timing_logs_after_progress_callback(tmp_path, capsys) -> None:
    completed_modules: list[str] = []

    def progress_callback(module_name: str) -> None:
        completed_modules.append(module_name)
        print(f"[progress] {module_name}", file=sys.stderr)

    run_analysis_modules(
        _synthetic_input(),
        tmp_path,
        {"decode_train_fraction": 0.8, "decode_ridge_alpha": 1e-3},
        ["decode_xy"],
        progress_callback=progress_callback,
    )

    stderr = capsys.readouterr().err
    progress_index = stderr.index("[progress] decode_xy")
    assert completed_modules == ["decode_xy"]
    assert progress_index < stderr.index("[decode_xy] encoder.place_codes test timings")
    assert progress_index < stderr.index(
        "[analysis_module] encoder.place_codes test decode_xy timings"
    )


def test_run_analysis_modules_reuses_cached_rate_maps_and_occupancy(monkeypatch, tmp_path) -> None:
    rate_map_calls = 0
    episode_statistics_rate_map_calls = 0
    occupancy_calls = 0
    original_compute_rate_maps = helpers.compute_rate_maps
    original_compute_rate_maps_from_episode_statistics = (
        helpers.compute_rate_maps_from_episode_statistics
    )
    original_compute_occupancy_only = helpers.compute_occupancy_only

    def counted_compute_rate_maps(*args, **kwargs):
        nonlocal rate_map_calls
        rate_map_calls += 1
        return original_compute_rate_maps(*args, **kwargs)

    def counted_compute_rate_maps_from_episode_statistics(*args, **kwargs):
        nonlocal episode_statistics_rate_map_calls
        episode_statistics_rate_map_calls += 1
        return original_compute_rate_maps_from_episode_statistics(*args, **kwargs)

    def counted_compute_occupancy_only(*args, **kwargs):
        nonlocal occupancy_calls
        occupancy_calls += 1
        return original_compute_occupancy_only(*args, **kwargs)

    monkeypatch.setattr(helpers, "compute_rate_maps", counted_compute_rate_maps)
    monkeypatch.setattr(
        helpers,
        "compute_rate_maps_from_episode_statistics",
        counted_compute_rate_maps_from_episode_statistics,
    )
    monkeypatch.setattr(helpers, "compute_occupancy_only", counted_compute_occupancy_only)

    results = run_analysis_modules(
        _synthetic_input(),
        tmp_path,
        {"num_bins_x": 20, "num_bins_y": 20, "smoothing_sigma": 0.0, "min_occupancy": 1e-6},
        [
            "dataset_coverage",
            "rate_map_fields",
            "spatial_info",
            "gridness",
            "place_field_detection",
        ],
    )

    assert set(results) == {
        "dataset_coverage",
        "rate_map_fields",
        "spatial_info",
        "gridness",
        "place_field_detection",
    }
    assert occupancy_calls == 1
    assert rate_map_calls == 0
    assert episode_statistics_rate_map_calls == 1


def test_split_rate_map_modules_share_metric_bundle(monkeypatch, tmp_path) -> None:
    metric_bundle_calls = 0
    original_compute_metric_bundle = rate_map_module.compute_rate_map_metric_bundle

    def counted_compute_metric_bundle(*args, **kwargs):
        nonlocal metric_bundle_calls
        metric_bundle_calls += 1
        return original_compute_metric_bundle(*args, **kwargs)

    monkeypatch.setattr(
        rate_map_module,
        "compute_rate_map_metric_bundle",
        counted_compute_metric_bundle,
    )

    results = run_analysis_modules(
        _synthetic_input(),
        tmp_path,
        {"num_bins_x": 10, "num_bins_y": 10, "smoothing_sigma": 0.0, "min_occupancy": 1e-6},
        [
            "rate_map_fields",
            "rate_map_reliability",
            "rate_map_bin_consistency",
            "rate_map_split_half",
            "rate_map_episode_correlation",
            "rate_map_coding_purity",
            "rate_map_panel",
            "rate_map_grid",
            "rate_map_extra_reliability_panels",
        ],
    )

    assert set(results) == {
        "rate_map_fields",
        "rate_map_reliability",
        "rate_map_bin_consistency",
        "rate_map_split_half",
        "rate_map_episode_correlation",
        "rate_map_coding_purity",
        "rate_map_panel",
        "rate_map_grid",
        "rate_map_extra_reliability_panels",
    }
    assert metric_bundle_calls == 1
    assert "mean_peak_rate" in results["rate_map_fields"].metrics
    assert "mean_reliability" in results["rate_map_reliability"].metrics
    assert "mean_bin_consistency" in results["rate_map_bin_consistency"].metrics
    assert "mean_split_half_agreement" in results["rate_map_split_half"].metrics
    assert "mean_episode_rate_map_correlation" in results["rate_map_episode_correlation"].metrics
    assert "mean_coding_purity_score" in results["rate_map_coding_purity"].metrics
    assert "rate_map_panel" in results["rate_map_panel"].figures
    assert "rate_map_grid" not in results["rate_map_panel"].figures
    assert "rate_map_grid" in results["rate_map_grid"].figures
    assert "rate_map_panel_thresholded_reliability" in (
        results["rate_map_extra_reliability_panels"].figures
    )
    assert "rate_map_panel_quantile_thresholded_reliability" not in (
        results["rate_map_extra_reliability_panels"].figures
    )


def test_sparsity_module_matches_reference_formulas(tmp_path) -> None:
    analysis_input = _synthetic_input()
    analysis_input.valid_mask[0, 0] = False
    flattened, _ = rate_map_kernels.flatten_valid_steps(
        analysis_input.representation,
        analysis_input.position_xy,
        analysis_input.valid_mask,
    )
    expected_lifetime = np.asarray(
        [
            rate_map_kernels.lifetime_activity_fraction(flattened[:, unit_index])
            for unit_index in range(flattened.shape[-1])
        ],
        dtype=np.float32,
    )
    expected_population = rate_map_kernels.population_activity_fraction(flattened)

    result = SparsityModule().run(analysis_input, tmp_path, {})

    np.testing.assert_allclose(
        result.per_unit_metrics["lifetime_activity_fraction"],
        expected_lifetime,
        rtol=1e-5,
    )
    assert np.isclose(result.metrics["population_activity_fraction"], expected_population)


def test_sample_valid_steps_samples_without_full_flatten(monkeypatch) -> None:
    representation = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    position_xy = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2)
    valid_mask = np.asarray(
        [
            [True, False, True, True, False, True],
            [False, True, True, False, True, True],
        ],
        dtype=bool,
    )

    def fail_if_full_flatten_is_used(*_args, **_kwargs):
        raise AssertionError("sample_valid_steps should not fully flatten before sampling")

    monkeypatch.setattr(rate_map_kernels, "flatten_valid_steps", fail_if_full_flatten_is_used)

    sampled = rate_map_kernels.sample_valid_steps(
        representation,
        position_xy,
        valid_mask,
        max_samples=4,
        random_seed=7,
    )

    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    expected_indices = np.sort(
        np.random.default_rng(7).choice(valid_indices, size=4, replace=False)
    )
    np.testing.assert_array_equal(sampled.values, representation.reshape(-1, 3)[expected_indices])
    np.testing.assert_array_equal(sampled.positions, position_xy.reshape(-1, 2)[expected_indices])
    np.testing.assert_array_equal(sampled.episode_ids, expected_indices // representation.shape[1])
    assert sampled.total_valid_steps == int(valid_mask.sum())


def test_sample_valid_steps_uses_all_valid_rows_when_uncapped() -> None:
    representation = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    position_xy = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2)
    valid_mask = np.asarray(
        [
            [True, False, True, True, False, True],
            [False, True, True, False, True, True],
        ],
        dtype=bool,
    )

    sampled = rate_map_kernels.sample_valid_steps(
        representation,
        position_xy,
        valid_mask,
        max_samples=0,
        random_seed=7,
    )

    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    np.testing.assert_array_equal(sampled.values, representation.reshape(-1, 3)[valid_indices])
    np.testing.assert_array_equal(sampled.positions, position_xy.reshape(-1, 2)[valid_indices])
    np.testing.assert_array_equal(sampled.episode_ids, valid_indices // representation.shape[1])
    assert sampled.total_valid_steps == int(valid_mask.sum())


def test_flatten_helpers_keep_all_valid_views() -> None:
    representation = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    position_xy = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2)
    vector_field = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2)
    valid_mask = np.ones((2, 4), dtype=bool)

    flattened_values, flattened_positions = rate_map_kernels.flatten_valid_steps(
        representation,
        position_xy,
        valid_mask,
    )
    flattened_only_positions = rate_map_kernels.flatten_positions(position_xy, valid_mask)
    flattened_vector_field = rate_map_kernels.flatten_vector_field(vector_field, valid_mask)
    episode_statistics = occupancy._prepare_episode_bin_statistics(
        representation,
        position_xy,
        valid_mask,
        num_bins_x=4,
        num_bins_y=2,
    )

    assert episode_statistics is not None
    assert np.shares_memory(flattened_values, representation)
    assert np.shares_memory(flattened_positions, position_xy)
    assert np.shares_memory(flattened_only_positions, position_xy)
    assert np.shares_memory(flattened_vector_field, vector_field)
    assert np.shares_memory(episode_statistics.flat_values, representation)


def test_rate_map_coding_purity_and_confounds_share_confound_scores(monkeypatch, tmp_path) -> None:
    confound_calls = 0
    original_compute_confound_scores = place_cell_quality.compute_available_confound_scores

    def counted_compute_confound_scores(*args, **kwargs):
        nonlocal confound_calls
        confound_calls += 1
        return original_compute_confound_scores(*args, **kwargs)

    monkeypatch.setattr(
        place_cell_quality,
        "compute_available_confound_scores",
        counted_compute_confound_scores,
    )

    results = run_analysis_modules(
        _synthetic_input(),
        tmp_path,
        {"num_bins_x": 10, "num_bins_y": 10, "smoothing_sigma": 0.0, "min_occupancy": 1e-6},
        ["rate_map_coding_purity", "confounds"],
    )

    assert set(results) == {"rate_map_coding_purity", "confounds"}
    assert confound_calls == 1
    assert set(results["confounds"].metadata["confounds_timing_seconds"]) >= {
        "compute_scores",
        "assemble_result",
    }


def test_figure_to_rgb_array_uses_actual_buffer_shape_for_scaled_canvas(monkeypatch) -> None:
    class FakeCanvas:
        def draw(self) -> None:
            return

        def get_width_height(self) -> tuple[int, int]:
            return (900, 700)

        def buffer_rgba(self) -> np.ndarray:
            return np.zeros((1400, 1800, 4), dtype=np.uint8)

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

    monkeypatch.setattr(plt, "close", lambda figure: None)
    frame = figure_to_rgb_array(FakeFigure())  # type: ignore[arg-type]
    assert frame.shape == (1400, 1800, 3)
