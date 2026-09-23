from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.dataset_coverage import DatasetCoverageModule
from placecell_research.analysis.decode_xy import DecodeXYModule
from placecell_research.analysis.field_stability import (
    FieldStabilityMetricsModule,
    FieldStabilitySummaryModule,
    FieldStabilityTrajectoriesModule,
)
from placecell_research.analysis.population_coverage import (
    PopulationCoverageModule,
    _coverage_gap_mask,
    _coverage_residuals,
)
from placecell_research.analysis.redundancy_metrics import RedundancyMetricsModule


def _synthetic_place_activity(
    positions: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    width: float = 0.22,
) -> np.ndarray:
    squared_distance = (positions[..., 0] - center_x) ** 2 + (positions[..., 1] - center_y) ** 2
    return np.exp(-squared_distance / max(width ** 2, 1e-6)).astype(np.float32)


def _analysis_input(
    representation: np.ndarray,
    position_xy: np.ndarray,
    *,
    heading: np.ndarray | None = None,
    kinematics: np.ndarray | None = None,
) -> AnalysisInput:
    return AnalysisInput(
        representation=representation.astype(np.float32, copy=False),
        position_xy=position_xy.astype(np.float32, copy=False),
        heading=heading,
        kinematics=kinematics,
        actions=None,
        valid_mask=np.ones(position_xy.shape[:2], dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsym-v0"},
    )


def test_coverage_residuals_are_negative_for_undercovered_bins() -> None:
    log_occupancy = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    coverage_counts = np.asarray([0.0, 1.0, 2.0, 0.0], dtype=np.float32)

    residuals = _coverage_residuals(log_occupancy, coverage_counts)

    assert residuals[-1] < 0.0
    assert residuals[2] > 0.0


def test_population_coverage_module_renders_coverage_map(tmp_path: Path) -> None:
    episodes = 4
    steps = 40
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    positions = []
    representations = []
    for episode_index in range(episodes):
        episode_positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index * 3)],
            axis=-1,
        )
        positions.append(episode_positions)
        representations.append(
            np.stack(
                [
                    _synthetic_place_activity(episode_positions, center_x=-0.35, center_y=-0.15),
                    _synthetic_place_activity(episode_positions, center_x=0.35, center_y=0.15),
                ],
                axis=-1,
            )
        )

    result = PopulationCoverageModule().run(
        _analysis_input(np.stack(representations, axis=0), np.stack(positions, axis=0)),
        tmp_path,
        {"num_bins_x": 24, "num_bins_y": 24, "smoothing_sigma": 0.6, "min_occupancy": 1e-6},
    )

    assert result.figures["population_coverage_map"].exists()
    assert result.figures["coverage_gap_summary"].exists()
    assert result.figures["coverage_correlation_scatter"].exists()
    assert result.metrics["active_field_unit_count"] == 2.0
    assert result.metrics["active_field_unit_fraction"] == 1.0
    assert result.metrics["max_population_coverage_units"] >= 1.0
    assert "high_occupancy_low_coverage_bin_fraction" in result.metrics
    assert "log_occupancy_coverage_correlation" in result.metrics


def test_redundancy_metrics_module_detects_dead_and_redundant_units(tmp_path: Path) -> None:
    steps = 128
    positions = np.stack(
        [
            np.linspace(-1.0, 1.0, steps, dtype=np.float32),
            np.linspace(-0.5, 0.5, steps, dtype=np.float32),
        ],
        axis=-1,
    )[None, :, :]
    base_signal = np.sin(np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32))
    representation = np.stack(
        [
            base_signal,
            base_signal.copy(),
            np.cos(np.linspace(0.0, 3.0 * np.pi, steps, dtype=np.float32)),
            np.zeros((steps,), dtype=np.float32),
        ],
        axis=-1,
    )[None, :, :]

    result = RedundancyMetricsModule().run(_analysis_input(representation, positions), tmp_path, {})

    assert result.figures["redundancy_metrics"].exists()
    assert result.metrics["dead_unit_fraction"] >= 0.24
    assert result.metrics["effective_rank"] < 3.5
    assert result.metrics["mean_abs_pairwise_unit_correlation"] > 0.0
    assert set(result.metadata["redundancy_timing_seconds"]) >= {
        "sample_valid_steps",
        "moments",
        "eigendecomposition",
        "pairwise_correlations",
        "render_figure",
    }
    np.testing.assert_allclose(
        result.per_unit_metrics["dead_unit_mask"][-1],
        np.asarray(1.0, dtype=np.float32),
    )


def test_redundancy_metrics_samples_before_flattening_valid_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    steps = 32
    positions = np.stack(
        [
            np.linspace(-1.0, 1.0, steps, dtype=np.float32),
            np.linspace(-0.5, 0.5, steps, dtype=np.float32),
        ],
        axis=-1,
    )[None, :, :]
    representation = np.stack(
        [
            np.linspace(0.0, 1.0, steps, dtype=np.float32),
            np.linspace(1.0, 0.0, steps, dtype=np.float32),
            np.sin(np.linspace(0.0, np.pi, steps, dtype=np.float32)),
        ],
        axis=-1,
    )[None, :, :]

    def fail_if_full_flatten_is_used(*_args, **_kwargs):
        raise AssertionError("redundancy should sample rows before full valid-step flattening")

    monkeypatch.setattr(
        "placecell_research.numerics.rate_map_kernels.flatten_valid_steps",
        fail_if_full_flatten_is_used,
    )

    result = RedundancyMetricsModule().run(
        _analysis_input(representation, positions),
        tmp_path,
        {"redundancy_max_samples": 8},
    )

    assert result.metadata["redundancy_sample_count"] == 8
    assert result.figures["redundancy_metrics"].exists()


def test_field_stability_module_reports_small_drift_for_stable_fields(tmp_path: Path) -> None:
    episodes = 5
    steps = 64
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.7, 0.7, steps, dtype=np.float32)
    positions = []
    representations = []
    for episode_index in range(episodes):
        episode_positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index)],
            axis=-1,
        )
        positions.append(episode_positions)
        representations.append(
            np.stack(
                [
                    _synthetic_place_activity(
                        episode_positions,
                        center_x=-0.25,
                        center_y=-0.10,
                        width=0.28,
                    ),
                    _synthetic_place_activity(
                        episode_positions,
                        center_x=0.40,
                        center_y=0.20,
                        width=0.25,
                    ),
                ],
                axis=-1,
            )
        )

    analysis_input = _analysis_input(np.stack(representations, axis=0), np.stack(positions, axis=0))
    config = {
        "per_episode_num_bins_x": 16,
        "per_episode_num_bins_y": 16,
        "per_episode_smoothing_sigma": 0.8,
        "per_episode_min_occupancy": 1e-6,
        "per_episode_minimum_visited_fraction": 0.05,
        "per_episode_minimum_valid_steps": 16,
        "place_field_threshold_fraction": 0.35,
    }
    metrics_result = FieldStabilityMetricsModule().run(
        analysis_input,
        tmp_path,
        config,
    )
    summary_result = FieldStabilitySummaryModule().run(
        analysis_input,
        tmp_path,
        config,
    )
    trajectories_result = FieldStabilityTrajectoriesModule().run(
        analysis_input,
        tmp_path,
        config,
    )

    assert summary_result.figures["field_stability_summary"].exists()
    assert trajectories_result.figures["field_center_trajectories"].exists()
    assert metrics_result.metrics["mean_field_center_drift_distance"] < 0.3
    assert metrics_result.metrics["mean_valid_field_episode_fraction"] > 0.5
    assert set(metrics_result.metadata["field_stability_timing_seconds"]) >= {
        "pooled_rate_maps",
        "pooled_field_metrics",
        "episode_field_metrics",
    }
    assert "render_summary" in summary_result.metadata["field_stability_timing_seconds"]
    assert "render_trajectories" in trajectories_result.metadata["field_stability_timing_seconds"]


def test_field_stability_split_modules_share_episode_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episodes = 4
    steps = 48
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    positions = []
    representations = []
    for episode_index in range(episodes):
        episode_positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index * 2)],
            axis=-1,
        )
        positions.append(episode_positions)
        representations.append(
            np.stack(
                [
                    _synthetic_place_activity(episode_positions, center_x=-0.3, center_y=-0.1),
                    _synthetic_place_activity(episode_positions, center_x=0.35, center_y=0.15),
                ],
                axis=-1,
            )
        )

    analysis_input = _analysis_input(
        np.stack(representations, axis=0),
        np.stack(positions, axis=0),
    )
    activity_chunk_passes = 0
    from placecell_research.analysis import field_stability

    original_iter_chunks = field_stability.iter_episode_activity_sum_chunks

    def counted_iter_chunks(*args, **kwargs):
        nonlocal activity_chunk_passes
        activity_chunk_passes += 1
        yield from original_iter_chunks(*args, **kwargs)

    monkeypatch.setattr(
        "placecell_research.analysis.field_stability.iter_episode_activity_sum_chunks",
        counted_iter_chunks,
    )
    config = {
        "per_episode_num_bins_x": 8,
        "per_episode_num_bins_y": 8,
        "per_episode_smoothing_sigma": 0.5,
        "per_episode_min_occupancy": 1e-6,
        "per_episode_minimum_visited_fraction": 0.1,
        "per_episode_minimum_valid_steps": 8,
        "field_stability_top_k": 2,
    }

    FieldStabilityMetricsModule().run(analysis_input, tmp_path, config)
    FieldStabilitySummaryModule().run(analysis_input, tmp_path, config)
    FieldStabilityTrajectoriesModule().run(analysis_input, tmp_path, config)

    assert activity_chunk_passes == 1


def test_field_stability_batches_episode_rate_maps(tmp_path: Path, monkeypatch) -> None:
    episodes = 4
    steps = 48
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    positions = []
    representations = []
    for episode_index in range(episodes):
        episode_positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index * 2)],
            axis=-1,
        )
        positions.append(episode_positions)
        representations.append(
            np.stack(
                [
                    _synthetic_place_activity(episode_positions, center_x=-0.3, center_y=-0.1),
                    _synthetic_place_activity(episode_positions, center_x=0.35, center_y=0.15),
                ],
                axis=-1,
            )
        )

    def fail_per_episode_rate_map_loop(*_args, **_kwargs):
        raise AssertionError("field_stability should batch per-episode rate maps")

    monkeypatch.setattr(
        "placecell_research.analysis.field_stability.compute_rate_maps",
        fail_per_episode_rate_map_loop,
        raising=False,
    )

    result = FieldStabilitySummaryModule().run(
        _analysis_input(np.stack(representations, axis=0), np.stack(positions, axis=0)),
        tmp_path,
        {
            "per_episode_num_bins_x": 8,
            "per_episode_num_bins_y": 8,
            "per_episode_smoothing_sigma": 0.5,
            "per_episode_min_occupancy": 1e-6,
            "per_episode_minimum_visited_fraction": 0.1,
            "per_episode_minimum_valid_steps": 8,
            "field_stability_top_k": 2,
        },
    )

    assert result.figures["field_stability_summary"].exists()


def test_decode_heatmap_and_headline_use_the_same_rmse(tmp_path, monkeypatch):
    import matplotlib.axes

    rng = np.random.default_rng(91)
    positions = rng.normal(size=(6, 32, 2)).astype(np.float32) * [1, 3]
    codes = rng.normal(size=(6, 32, 4)).astype(np.float32)
    images = []
    original = matplotlib.axes.Axes.imshow

    def capture(axis, values, *args, **kwargs):
        images.append(np.asarray(values).copy())
        return original(axis, values, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", capture)
    result = DecodeXYModule().run(_analysis_input(codes, positions), tmp_path, {
        "decode_error_num_bins_x": 1, "decode_error_num_bins_y": 1,
        "decode_bias_min_samples_per_bin": 1,
    })
    assert images[0].shape == (1, 1)
    assert images[0][0, 0] == pytest.approx(result.metrics["decode_rmse"], rel=1e-6)


def test_decode_xy_module_writes_bias_arrow_figure(tmp_path: Path) -> None:
    episodes = 6
    steps = 48
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.8, 0.8, steps, dtype=np.float32)
    positions = []
    representations = []
    for episode_index in range(episodes):
        episode_positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index * 2)],
            axis=-1,
        )
        positions.append(episode_positions)
        representations.append(
            np.stack(
                [
                    episode_positions[:, 0],
                    episode_positions[:, 1],
                    episode_positions[:, 0] + episode_positions[:, 1],
                ],
                axis=-1,
            )
        )

    result = DecodeXYModule().run(
        _analysis_input(np.stack(representations, axis=0), np.stack(positions, axis=0)),
        tmp_path,
        {
            "decode_train_fraction": 0.8,
            "decode_ridge_alpha": 1e-3,
            "decode_error_num_bins_x": 20,
            "decode_error_num_bins_y": 20,
            "decode_bias_min_samples_per_bin": 3,
        },
    )

    assert result.figures["decode_error_heatmap"].exists()
    assert result.figures["decode_bias_arrows"].exists()
    assert result.metrics["decode_r2"] > 0.9
    assert result.metrics["mean_decode_bias_magnitude"] >= 0.0
    assert result.metadata["decode_bias_min_samples_per_bin"] == 3
    assert result.metadata["decode_max_samples"] == 0
    assert result.metadata["decode_sampled_valid_steps"] == episodes * steps
    assert result.metadata["decode_total_valid_steps"] == episodes * steps
    assert set(result.metadata["decode_timing_seconds"]) >= {
        "sample_valid_steps",
        "linear_decode",
        "error_maps",
        "render_figures",
    }


def test_decode_xy_module_can_emit_nonlinear_decode_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    episodes = 8
    steps = 48
    flat_features = rng.uniform(-1.0, 1.0, size=(episodes * steps, 2)).astype(np.float32)
    flat_positions = np.stack(
        [
            flat_features[:, 0] * flat_features[:, 1],
            np.square(flat_features[:, 0]) - np.square(flat_features[:, 1]),
        ],
        axis=-1,
    ).astype(np.float32)

    result = DecodeXYModule().run(
        _analysis_input(
            flat_features.reshape(episodes, steps, 2),
            flat_positions.reshape(episodes, steps, 2),
        ),
        tmp_path,
        {
            "decode_train_fraction": 0.75,
            "decode_ridge_alpha": 1e-3,
            "decode_nonlinear_enabled": True,
            "decode_nonlinear_hidden_sizes": [64, 64],
            "decode_nonlinear_max_epochs": 180,
            "decode_nonlinear_batch_size": 128,
            "decode_nonlinear_random_seed": 11,
            "decode_bias_min_samples_per_bin": 1,
        },
    )

    assert "nonlinear_decode_r2" in result.metrics
    assert result.metrics["nonlinear_decode_r2"] > result.metrics["decode_r2"]
    assert "nonlinear_decode" in result.metadata["decode_timing_seconds"]


def test_decode_xy_module_can_opt_into_bounded_valid_step_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episodes = 4
    steps = 32
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.8, 0.8, steps, dtype=np.float32)
    positions = []
    representations = []
    for episode_index in range(episodes):
        episode_positions = np.stack(
            [x_positions, np.roll(y_positions, shift=episode_index)],
            axis=-1,
        )
        positions.append(episode_positions)
        representations.append(
            np.stack(
                [
                    episode_positions[:, 0],
                    episode_positions[:, 1],
                    episode_positions[:, 0] - episode_positions[:, 1],
                ],
                axis=-1,
            )
        )

    def fail_if_full_flatten_is_used(*_args, **_kwargs):
        raise AssertionError("decode_xy should sample rows before full valid-step flattening")

    monkeypatch.setattr(
        "placecell_research.numerics.rate_map_kernels.flatten_valid_steps",
        fail_if_full_flatten_is_used,
    )

    result = DecodeXYModule().run(
        _analysis_input(np.stack(representations, axis=0), np.stack(positions, axis=0)),
        tmp_path,
        {
            "decode_train_fraction": 0.75,
            "decode_ridge_alpha": 1e-3,
            "decode_max_samples": 32,
            "decode_random_seed": 9,
            "decode_bias_min_samples_per_bin": 1,
        },
    )

    assert result.metrics["decode_r2"] > 0.8
    assert result.metadata["decode_max_samples"] == 32
    assert result.metadata["decode_sampled_valid_steps"] == 32
    assert result.metadata["decode_total_valid_steps"] == episodes * steps
    assert result.metadata["decode_timing_seconds"]["sample_valid_steps"] >= 0.0


def test_decode_xy_module_skips_single_episode_decode(tmp_path: Path) -> None:
    steps = 8
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.8, 0.8, steps, dtype=np.float32)
    episode_positions = np.stack([x_positions, y_positions], axis=-1)
    representation = np.stack(
        [
            episode_positions[:, 0],
            episode_positions[:, 1],
            episode_positions[:, 0] + episode_positions[:, 1],
        ],
        axis=-1,
    )

    result = DecodeXYModule().run(
        _analysis_input(representation[None, ...], episode_positions[None, ...]),
        tmp_path,
        {"decode_train_fraction": 0.8, "decode_ridge_alpha": 1e-3},
    )

    assert result.metrics == {}
    assert result.figures == {}
    assert result.metadata["decode_skipped"] is True
    assert (
        result.metadata["decode_skip_reason"]
        == "Need at least 2 episodes for episode-level XY decoding."
    )
    assert result.metadata["valid_sample_count"] == steps
    assert result.metadata["valid_episode_count"] == 1


def _two_visited_bins_input() -> AnalysisInput:
    """Two far-apart bins, one of them covered by the single unit's field."""
    steps = 40
    positions = np.zeros((2, steps, 2), dtype=np.float32)
    positions[:, ::2, :] = np.asarray([-1.0, -1.0], dtype=np.float32)
    positions[:, 1::2, :] = np.asarray([1.0, 1.0], dtype=np.float32)
    representation = np.zeros((2, steps, 1), dtype=np.float32)
    representation[:, ::2, 0] = 1.0
    return _analysis_input(representation, positions)


def test_population_coverage_reads_visited_bins_before_smoothing(tmp_path: Path) -> None:
    result = PopulationCoverageModule().run(
        _two_visited_bins_input(),
        tmp_path,
        {"num_bins_x": 9, "num_bins_y": 9, "smoothing_sigma": 1.0, "min_occupancy": 4.0},
    )

    assert result.metrics["mean_population_coverage_units"] == 0.5
    assert result.metrics["mean_population_coverage_fraction"] == 0.5


def test_dataset_coverage_counts_visited_bins_before_smoothing(tmp_path: Path) -> None:
    result = DatasetCoverageModule().run(
        _two_visited_bins_input(),
        tmp_path,
        {"num_bins_x": 9, "num_bins_y": 9, "smoothing_sigma": 1.0},
    )

    assert result.metrics["occupied_bins_fraction"] == pytest.approx(2 / 81)


def test_high_occupancy_denominator_excludes_the_smoothing_halo() -> None:
    """The gap statistic ranks bins by occupancy, but only visited bins may be in the ranking."""
    raw_occupancy = np.zeros((9, 9), dtype=np.float32)
    raw_occupancy[2:5, 2:5] = np.asarray(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
        dtype=np.float32,
    )
    smoothed_occupancy = gaussian_filter(raw_occupancy, sigma=1.0).astype(np.float32)
    coverage_counts = np.zeros((9, 9), dtype=np.float32)
    visited_mask = raw_occupancy > 0.0

    assert int(np.count_nonzero(visited_mask)) == 9
    assert int(np.count_nonzero(smoothed_occupancy > 0.0)) > 9

    _, _, visited_high_count = _coverage_gap_mask(
        smoothed_occupancy,
        coverage_counts,
        visited_mask=visited_mask,
    )
    _, _, halo_high_count = _coverage_gap_mask(
        smoothed_occupancy,
        coverage_counts,
        visited_mask=smoothed_occupancy > 0.0,
    )

    assert visited_high_count == 4
    assert halo_high_count > visited_high_count
