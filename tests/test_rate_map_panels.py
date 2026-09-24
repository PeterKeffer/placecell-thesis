from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest

from placecell_research.analysis import occupancy, reliability_splits
from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.rate_map_metrics import compute_rate_map_metric_bundle
from placecell_research.analysis.rate_map_rendering import (
    _population_is_signed,
    _style_for_family,
)
from placecell_research.analysis.rate_maps import (
    RateMapFieldsModule,
    RateMapPanelModule,
    _RateMapModuleBase,
)
from placecell_research.analysis.world_overlay import resolve_world_overlay


def test_orient_topdown_yaxis_inverts_only_for_y_down():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from placecell_research.analysis.world_overlay import WorldOverlay, orient_topdown_yaxis

    figure, axis = plt.subplots()
    axis.set_ylim(0.0, 25.0)
    orient_topdown_yaxis(axis, WorldOverlay("env", (), y_down=True))
    assert axis.get_ylim() == (25.0, 0.0)
    axis.set_ylim(0.0, 25.0)
    orient_topdown_yaxis(axis, WorldOverlay("env", (), y_down=False))
    assert axis.get_ylim() == (0.0, 25.0)
    orient_topdown_yaxis(axis, None)
    assert axis.get_ylim() == (0.0, 25.0)
    plt.close(figure)


def test_finalize_arena_axis_applies_bounds_aspect_and_orientation():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from placecell_research.analysis.world_overlay import WorldOverlay, finalize_arena_axis

    figure, axis = plt.subplots()
    finalize_arena_axis(
        axis,
        x_bounds=(-2.0, 4.0),
        y_bounds=(0.0, 25.0),
        world_overlay=WorldOverlay("env", (), y_down=True),
    )

    assert axis.get_xlim() == (-2.0, 4.0)
    assert axis.get_ylim() == (25.0, 0.0)
    assert float(axis.get_aspect()) == 1.0
    plt.close(figure)


def test_unknown_world_overlay_env_id_stays_quiet():
    from placecell_research.analysis import world_overlay

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert world_overlay.resolve_world_overlay("definitely-not-an-env") is None

    assert list(caught) == []


def test_known_jaxenstein_overlay_warns_when_optional_backend_is_missing(monkeypatch):
    from placecell_research.analysis import world_overlay
    from placecell_research.envs import jaxenstein_maps

    def raise_missing_backend(_env_id):
        raise ImportError("No module named 'jaxenstein'")

    monkeypatch.setattr(jaxenstein_maps, "supports_goal_override", lambda env_id: env_id == "cave")
    monkeypatch.setattr(jaxenstein_maps, "build_jaxenstein_env", raise_missing_backend)

    with pytest.warns(RuntimeWarning, match="JAXenstein overlay.*cave.*optional"):
        assert world_overlay.resolve_world_overlay("cave") is None


def test_known_jaxenstein_overlay_warns_when_build_breaks(monkeypatch):
    from placecell_research.analysis import world_overlay
    from placecell_research.envs import jaxenstein_maps

    def raise_build_error(_env_id):
        raise RuntimeError("bad wall grid")

    monkeypatch.setattr(jaxenstein_maps, "supports_goal_override", lambda env_id: env_id == "cave")
    monkeypatch.setattr(jaxenstein_maps, "build_jaxenstein_env", raise_build_error)

    with pytest.warns(RuntimeWarning, match="Failed to build JAXenstein overlay.*cave"):
        assert world_overlay.resolve_world_overlay("cave") is None


def _synthetic_place_activity(
    positions: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    width: float = 0.18,
) -> np.ndarray:
    squared_distance = (positions[..., 0] - center_x) ** 2 + (positions[..., 1] - center_y) ** 2
    return np.exp(-squared_distance / max(width**2, 1e-6)).astype(np.float32)


def test_rate_map_metric_bundle_computes_without_rendering() -> None:
    positions = np.stack(
        [
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            np.linspace(-0.5, 0.5, 12, dtype=np.float32),
        ],
        axis=-1,
    )[None, ...]
    representation = _synthetic_place_activity(positions, center_x=0.1, center_y=0.1)[..., None]
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, 12), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={},
    )

    bundle = compute_rate_map_metric_bundle(
        analysis_input,
        {
            "num_bins_x": 8,
            "num_bins_y": 8,
            "smoothing_sigma": 0.3,
            "min_occupancy": 1e-6,
        },
        bounds=None,
    )

    assert bundle.rate_map_result.rate_maps.shape == (1, 8, 8)
    assert bundle.place_metrics.spatial_information_bits.shape == (1,)
    assert bundle.ranked_indices.tolist() == [0]


def test_rate_map_metric_bundle_reuses_episode_bin_statistics(monkeypatch) -> None:
    episodes = 4
    steps = 10
    positions = np.stack(
        [
            np.linspace(-1.0, 1.0, steps, dtype=np.float32),
            np.linspace(-0.5, 0.5, steps, dtype=np.float32),
        ],
        axis=-1,
    )
    position_xy = np.stack(
        [np.roll(positions, shift=episode_index, axis=0) for episode_index in range(episodes)]
    )
    representation = np.stack(
        [
            _synthetic_place_activity(position_xy, center_x=-0.2, center_y=-0.1),
            _synthetic_place_activity(position_xy, center_x=0.3, center_y=0.2),
        ],
        axis=-1,
    )
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={},
    )
    prepare_calls = 0
    original_prepare = occupancy.prepare_episode_bin_statistics

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(occupancy, "prepare_episode_bin_statistics", counted_prepare)

    compute_rate_map_metric_bundle(
        analysis_input,
        {
            "num_bins_x": 8,
            "num_bins_y": 8,
            "smoothing_sigma": 0.3,
            "min_occupancy": 1e-6,
        },
        bounds=None,
    )

    assert prepare_calls == 1


def test_rate_map_metric_bundle_reuses_episode_activity_sum_chunks(monkeypatch) -> None:
    episodes = 5
    steps = 12
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.5, 0.5, steps, dtype=np.float32)
    position_xy = np.stack(
        [
            np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
            for episode_index in range(episodes)
        ],
        axis=0,
    )
    representation = np.stack(
        [
            _synthetic_place_activity(position_xy, center_x=-0.2, center_y=-0.1),
            _synthetic_place_activity(position_xy, center_x=0.3, center_y=0.2),
            np.broadcast_to(np.linspace(0.1, 0.9, steps, dtype=np.float32), (episodes, steps)),
        ],
        axis=-1,
    )
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={},
    )
    summed_unit_ranges: list[tuple[int, int]] = []
    original_activity_sums = reliability_splits._episode_activity_sums

    def recorded_activity_sums(statistics, start_index, stop_index):
        summed_unit_ranges.append((start_index, stop_index))
        return original_activity_sums(statistics, start_index, stop_index)

    monkeypatch.setattr(reliability_splits, "_episode_activity_sums", recorded_activity_sums)

    compute_rate_map_metric_bundle(
        analysis_input,
        {
            "num_bins_x": 8,
            "num_bins_y": 8,
            "smoothing_sigma": 0.3,
            "min_occupancy": 1e-6,
        },
        bounds=None,
    )

    summed_units = [
        unit_index for start, stop in summed_unit_ranges for unit_index in range(start, stop)
    ]
    assert sorted(summed_units) == list(range(representation.shape[-1]))


def test_rate_map_module_writes_summary_panel_grid_and_field_reliability(
    tmp_path: Path, analysis_settings
) -> None:
    episodes = 5
    steps = 48
    position_sequences = []
    representation_sequences = []
    for _episode_index in range(episodes):
        x_positions = np.linspace(-7.0, 7.0, steps, dtype=np.float32)
        y_positions = 8.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32))
        positions = np.stack([x_positions, y_positions], axis=-1)
        position_sequences.append(positions)
        unit_0 = _synthetic_place_activity(positions, center_x=-3.5, center_y=-1.5, width=1.8)
        unit_1 = _synthetic_place_activity(positions, center_x=4.5, center_y=2.5, width=1.8)
        unit_2 = np.linspace(0.05, 0.15, steps, dtype=np.float32)
        representation_sequences.append(np.stack([unit_0, unit_1, unit_2], axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=24,
            num_bins_y=24,
            smoothing_sigma=0.8,
            min_occupancy=1e-6,
            reliability_threshold_fraction=0.3,
            place_field_threshold_fraction=0.35,
            rate_map_panel_top_k=3,
            rate_map_grid_top_k=3,
        ),
    )

    assert (
        tmp_path / "rate_map_bundle" / "rate_map_panel__encoder.place_codes__validation.png"
    ).exists()
    assert (
        tmp_path
        / "rate_map_bundle"
        / "rate_map_panel_thresholded_reliability__encoder.place_codes__validation.png"
    ).exists()
    assert (
        tmp_path / "rate_map_bundle" / "rate_map_grid__encoder.place_codes__validation.png"
    ).exists()
    assert (
        tmp_path / "rate_map_bundle" / "rate_map_panel_clean__encoder.place_codes__validation.png"
    ).exists()
    assert (
        tmp_path
        / "rate_map_bundle"
        / "rate_map_panel_thresholded_reliability_clean__encoder.place_codes__validation.png"
    ).exists()
    assert (
        tmp_path / "rate_map_bundle" / "rate_map_grid_clean__encoder.place_codes__validation.png"
    ).exists()
    assert result.tables["rate_map_panel_metrics"].exists()
    assert result.metrics["mean_spatial_information_bits"] > 0.0
    assert result.metrics["mean_spatial_coherence"] > 0.0
    assert result.metrics["mean_reliability_weighted_information"] > 0.0
    all_units_rwi = result.metrics["mean_reliability_weighted_information_all_units"]
    assert 0.0 < all_units_rwi <= result.metrics["mean_reliability_weighted_information"]
    place_cell_rwi = result.metrics["mean_reliability_weighted_information_place_cells"]
    assert np.isnan(place_cell_rwi) or place_cell_rwi >= all_units_rwi
    excess_rwi = result.metrics["mean_reliability_weighted_information_excess_all_units"]
    assert np.isnan(excess_rwi) or 0.0 <= excess_rwi <= all_units_rwi
    per_unit_excess = result.per_unit_metrics["reliability_weighted_information_excess"]
    per_unit_rwi = result.per_unit_metrics["reliability_weighted_information"]
    both_finite = np.isfinite(per_unit_excess) & np.isfinite(per_unit_rwi)
    assert np.all(per_unit_excess[both_finite] <= per_unit_rwi[both_finite] + 1e-6)
    in_field_lift = result.metrics["mean_reliability_lift_inside_fields"]
    assert np.isnan(in_field_lift) or in_field_lift > result.metrics["mean_reliability_lift"]
    traversal_reliability = result.per_unit_metrics["field_traversal_reliability"]
    assert traversal_reliability.shape == (3,)
    finite_traversal = traversal_reliability[np.isfinite(traversal_reliability)]
    assert np.all((finite_traversal >= 0.0) & (finite_traversal <= 1.0))
    mean_traversal = result.metrics["mean_field_traversal_reliability"]
    assert np.isnan(mean_traversal) or 0.0 <= mean_traversal <= 1.0
    assert result.metrics["mean_coding_purity_score"] > 0.0
    assert result.metrics["mean_reliability_inside_fields"] > 0.0
    assert result.metrics["mean_bin_consistency_inside_fields"] > 0.0
    assert result.metrics["mean_bin_consistency_supported_field_fraction"] > 0.0
    assert result.metrics["mean_split_half_rate_map_correlation"] >= 0.0
    assert result.metrics["mean_split_half_agreement_supported_field_fraction"] > 0.0
    assert result.metrics["mean_episode_rate_map_correlation"] >= 0.0
    assert result.per_unit_metrics["field_count"].shape == (3,)
    assert result.per_unit_metrics["spatial_coherence"].shape == (3,)
    assert result.per_unit_metrics["reliability_weighted_information"].shape == (3,)
    assert result.per_unit_metrics["coding_purity_score"].shape == (3,)
    assert result.per_unit_metrics["mean_bin_consistency"].shape == (3,)
    assert result.per_unit_metrics["split_half_rate_map_correlation"].shape == (3,)
    assert float(result.per_unit_metrics["field_count"][0]) >= 1.0
    assert result.metadata["rate_map_colormap_mode"] == "reds"
    assert result.metadata["per_bin_cv_min_episodes"] == 3
    assert (
        result.metadata["rate_map_panel_reliability_metric"] == "quantile_thresholded_reliability"
    )
    assert result.metadata["rate_map_panel_emit_thresholded_reliability_figure"] is True
    assert result.metadata["rate_map_panel_emit_quantile_thresholded_reliability_figure"] is True
    assert result.metadata["reliability_threshold_quantile"] == 0.95
    assert result.metadata["rate_map_panel_metric_fill_sigma_bins"] == 1.0
    assert result.metadata["split_half_agreement_min_episodes_per_half"] == 2
    assert result.metadata["rate_map_panel_has_support_column"] is False
    assert result.metadata["rate_map_has_per_unit_colorbars"] is True
    assert len(result.metadata["ranked_units_summary"]) == 3
    assert len(result.metadata["ranked_units_grid"]) == 3
    assert result.metadata["world_overlay_applied"] is True
    assert result.figures["rate_map_panel_thresholded_reliability"].exists()
    assert result.figures["rate_map_panel_thresholded_reliability_clean"].exists()
    assert result.figures["rate_map_panel_quantile_thresholded_reliability"].exists()
    assert result.figures["rate_map_panel_quantile_thresholded_reliability_clean"].exists()
    assert len(result.metadata["rate_map_panel_thresholded_reliability_pages"]) == 1
    assert len(result.metadata["rate_map_panel_thresholded_reliability_clean_pages"]) == 1
    assert len(result.metadata["rate_map_panel_quantile_thresholded_reliability_pages"]) == 1
    assert len(result.metadata["rate_map_panel_quantile_thresholded_reliability_clean_pages"]) == 1
    assert (
        result.figures["rate_map_panel_quantile_thresholded_reliability"]
        == result.figures["rate_map_panel"]
    )
    assert (
        result.figures["rate_map_panel_quantile_thresholded_reliability_clean"]
        == result.figures["rate_map_panel_clean"]
    )
    metrics_table_text = result.tables["rate_map_panel_metrics"].read_text()
    assert "spatial_coherence" in metrics_table_text
    assert "reliability_weighted_information" in metrics_table_text
    assert "coding_purity_score" in metrics_table_text
    assert "mean_thresholded_reliability" in metrics_table_text
    assert "mean_quantile_thresholded_reliability" in metrics_table_text
    assert "mean_bin_consistency" in metrics_table_text
    assert "mean_split_half_agreement" in metrics_table_text
    assert "bin_consistency_supported_field_fraction" in metrics_table_text
    assert "split_half_agreement_supported_field_fraction" in metrics_table_text
    assert "split_half_rate_map_correlation" in metrics_table_text
    assert "episode_rate_map_correlation" in metrics_table_text


def test_rate_map_module_reports_timing_sections(tmp_path: Path, capsys, analysis_settings) -> None:
    positions = np.stack(
        [
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            np.linspace(-0.5, 0.5, 12, dtype=np.float32),
        ],
        axis=-1,
    )[None, ...]
    representation = _synthetic_place_activity(
        positions,
        center_x=0.1,
        center_y=0.1,
    )[..., None]
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, 12), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=8,
            num_bins_y=8,
            smoothing_sigma=0.3,
            min_occupancy=1e-6,
            rate_map_panel_top_k=1,
            rate_map_grid_top_k=1,
            rate_map_panel_emit_thresholded_reliability_figure=False,
            rate_map_panel_emit_quantile_thresholded_reliability_figure=False,
        ),
    )

    captured = capsys.readouterr()
    timings = result.metadata["rate_map_timing_seconds"]
    assert set(timings) >= {
        "metric_bundle",
        "metric_bundle.prepare_episode_statistics",
        "metric_bundle.reliability_peak_fraction",
        "metric_bundle.reliability_quantile",
        "render_primary_panel",
        "render_primary_grid",
        "write_metrics_table",
        "total",
    }
    assert all(float(value) >= 0.0 for value in timings.values())
    assert "[rate_map_bundle] encoder.place_codes validation timings" in captured.err
    assert "metric_bundle=" in captured.err
    assert "metric_bundle.prepare_episode_statistics=" in captured.err
    assert "render_primary_panel=" in captured.err


def test_rate_map_panel_module_writes_panel_without_grid(tmp_path: Path, analysis_settings) -> None:
    positions = np.stack(
        [
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            np.linspace(-0.5, 0.5, 12, dtype=np.float32),
        ],
        axis=-1,
    )[None, ...]
    representation = _synthetic_place_activity(
        positions,
        center_x=0.1,
        center_y=0.1,
    )[..., None]
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, 12), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={},
    )

    result = RateMapPanelModule().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=8,
            num_bins_y=8,
            smoothing_sigma=0.3,
            min_occupancy=1e-6,
            rate_map_panel_top_k=1,
            rate_map_grid_top_k=1,
            rate_map_panel_emit_thresholded_reliability_figure=False,
            rate_map_panel_emit_quantile_thresholded_reliability_figure=False,
        ),
    )

    panel_path = tmp_path / "rate_map_panel" / "rate_map_panel__encoder.place_codes__validation.png"
    assert panel_path.exists()
    disabled_grid_path = (
        tmp_path / "rate_map_panel" / "rate_map_grid__encoder.place_codes__validation.png"
    )
    assert not disabled_grid_path.exists()
    assert "rate_map_panel" in result.figures
    assert "rate_map_grid" not in result.figures
    assert result.metadata["rate_map_grid_pages"] == []
    assert result.metadata["ranked_units_grid"] == []


def test_rate_map_metrics_module_skips_spike_position_selection(
    tmp_path: Path,
    monkeypatch,
    analysis_settings,
) -> None:
    positions = np.stack(
        [
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            np.linspace(-0.5, 0.5, 12, dtype=np.float32),
        ],
        axis=-1,
    )[None, ...]
    representation = _synthetic_place_activity(
        positions,
        center_x=0.1,
        center_y=0.1,
    )[..., None]
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, 12), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={},
    )

    def fail_if_selected(*_args, **_kwargs):
        raise AssertionError("metrics-only rate-map modules should not select spike overlays")

    monkeypatch.setattr(
        "placecell_research.analysis.rate_maps.select_high_activation_positions",
        fail_if_selected,
    )

    result = RateMapFieldsModule().run(
        analysis_input,
        tmp_path,
        analysis_settings(num_bins_x=8, num_bins_y=8, smoothing_sigma=0.3, min_occupancy=1e-6),
    )

    assert "rate_map_panel" not in result.figures
    assert result.metrics["mean_peak_rate"] > 0.0


def test_rate_map_module_can_switch_panel_back_to_thresholded_reliability(
    tmp_path: Path, analysis_settings
) -> None:
    episodes = 4
    steps = 18
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
        position_sequences.append(positions)
        unit_0 = _synthetic_place_activity(positions, center_x=-0.4, center_y=-0.2)
        unit_1 = _synthetic_place_activity(positions, center_x=0.35, center_y=0.2)
        representation_sequences.append(np.stack([unit_0, unit_1], axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=20,
            num_bins_y=20,
            smoothing_sigma=0.6,
            min_occupancy=1e-6,
            rate_map_panel_reliability_metric="thresholded_reliability",
            rate_map_panel_top_k=2,
            rate_map_grid_top_k=2,
        ),
    )

    assert result.metadata["rate_map_panel_reliability_metric"] == "thresholded_reliability"
    assert result.metrics["mean_reliability_inside_fields"] > 0.0
    assert result.metrics["mean_bin_consistency_inside_fields"] > 0.0
    assert (
        result.figures["rate_map_panel_thresholded_reliability"] == result.figures["rate_map_panel"]
    )
    assert (
        result.metadata["rate_map_panel_thresholded_reliability_pages"]
        == result.metadata["rate_map_panel_pages"]
    )


def test_rate_map_module_can_switch_panel_to_quantile_thresholded_reliability(
    tmp_path: Path,
    analysis_settings,
) -> None:
    episodes = 4
    steps = 18
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
        position_sequences.append(positions)
        unit_0 = _synthetic_place_activity(positions, center_x=-0.4, center_y=-0.2)
        unit_1 = _synthetic_place_activity(positions, center_x=0.35, center_y=0.2)
        representation_sequences.append(np.stack([unit_0, unit_1], axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=20,
            num_bins_y=20,
            smoothing_sigma=0.6,
            min_occupancy=1e-6,
            rate_map_panel_reliability_metric="quantile_thresholded_reliability",
            rate_map_panel_top_k=2,
            rate_map_grid_top_k=2,
        ),
    )

    assert (
        result.metadata["rate_map_panel_reliability_metric"] == "quantile_thresholded_reliability"
    )
    assert result.metrics["mean_quantile_thresholded_reliability_inside_fields"] > 0.0
    assert (
        result.figures["rate_map_panel_quantile_thresholded_reliability"]
        == result.figures["rate_map_panel"]
    )
    assert (
        result.metadata["rate_map_panel_quantile_thresholded_reliability_pages"]
        == result.metadata["rate_map_panel_pages"]
    )


def test_rate_map_module_can_render_split_half_agreement_panel(
    tmp_path: Path, analysis_settings
) -> None:
    episodes = 4
    steps = 18
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
        position_sequences.append(positions)
        unit_0 = _synthetic_place_activity(positions, center_x=-0.4, center_y=-0.2)
        unit_1 = _synthetic_place_activity(positions, center_x=0.35, center_y=0.2)
        representation_sequences.append(np.stack([unit_0, unit_1], axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=20,
            num_bins_y=20,
            smoothing_sigma=0.6,
            min_occupancy=1e-6,
            rate_map_panel_reliability_metric="split_half_agreement",
            rate_map_panel_top_k=2,
            rate_map_grid_top_k=2,
        ),
    )

    assert result.metadata["rate_map_panel_reliability_metric"] == "split_half_agreement"
    assert result.metrics["mean_split_half_agreement_inside_fields"] > 0.0
    assert np.asarray(result.metadata["split_half_agreement_support_counts"]).shape == (20, 20)


def test_rate_map_module_can_disable_extra_thresholded_reliability_panel(
    tmp_path: Path, analysis_settings
) -> None:
    episodes = 4
    steps = 18
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.6, 0.6, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
        position_sequences.append(positions)
        unit_0 = _synthetic_place_activity(positions, center_x=-0.4, center_y=-0.2)
        unit_1 = _synthetic_place_activity(positions, center_x=0.35, center_y=0.2)
        representation_sequences.append(np.stack([unit_0, unit_1], axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=20,
            num_bins_y=20,
            smoothing_sigma=0.6,
            min_occupancy=1e-6,
            rate_map_panel_emit_thresholded_reliability_figure=False,
            rate_map_panel_emit_quantile_thresholded_reliability_figure=False,
            rate_map_panel_top_k=2,
            rate_map_grid_top_k=2,
        ),
    )

    assert result.metadata["rate_map_panel_emit_thresholded_reliability_figure"] is False
    assert result.metadata["rate_map_panel_emit_quantile_thresholded_reliability_figure"] is False
    assert "rate_map_panel_thresholded_reliability" not in result.figures
    assert (
        result.figures["rate_map_panel_quantile_thresholded_reliability"]
        == result.figures["rate_map_panel"]
    )
    assert result.metadata["rate_map_panel_thresholded_reliability_pages"] == []
    assert (
        result.metadata["rate_map_panel_quantile_thresholded_reliability_pages"]
        == result.metadata["rate_map_panel_pages"]
    )


def test_rate_map_metrics_export_peak_rate_ignores_unvisited_nan_bins(
    tmp_path: Path, analysis_settings
) -> None:
    episodes = 4
    steps = 8
    x_positions = np.linspace(-1.0, -0.2, steps, dtype=np.float32)
    y_positions = np.linspace(-0.5, 0.3, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        positions = np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
        position_sequences.append(positions)
        unit_0 = _synthetic_place_activity(positions, center_x=-0.6, center_y=-0.1)
        unit_1 = _synthetic_place_activity(positions, center_x=-0.35, center_y=0.15)
        representation_sequences.append(np.stack([unit_0, unit_1], axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=24,
            num_bins_y=24,
            smoothing_sigma=0.8,
            min_occupancy=1e-6,
            rate_map_panel_top_k=2,
            rate_map_grid_top_k=2,
        ),
    )

    exported_rows = list(csv.DictReader(result.tables["rate_map_panel_metrics"].open()))
    assert len(exported_rows) == 2
    for row in exported_rows:
        exported_peak_rate = float(row["peak_rate"])
        assert np.isfinite(exported_peak_rate)
        assert exported_peak_rate > 0.0


def test_rate_map_module_marks_place_field_metrics_unsupported_for_signed_rate_maps(
    tmp_path: Path,
    analysis_settings,
) -> None:
    episodes = 4
    steps = 48
    x_positions = np.linspace(-7.0, 7.0, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    for episode_index in range(episodes):
        y_positions = 8.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32))
        positions = np.stack([x_positions, y_positions], axis=-1)
        position_sequences.append(positions)
        phase = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float32) + episode_index * 0.2
        unit_0 = np.sin(phase)
        unit_1 = np.cos(phase * 1.3)
        representation_sequences.append(np.stack([unit_0, unit_1], axis=-1).astype(np.float32))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.hidden_state",
        label="encoder_hidden_state",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=20,
            num_bins_y=20,
            smoothing_sigma=0.4,
            min_occupancy=1e-6,
            rate_map_panel_top_k=2,
            rate_map_grid_top_k=2,
        ),
    )

    assert result.metadata["place_field_metrics_skipped_for_signed_rate_maps"] is True
    assert result.metadata["place_field_metrics_supported_unit_count"] == 0
    assert not np.isfinite(result.metrics["mean_spatial_information_bits"])
    assert not np.isfinite(result.metrics["mean_reliability_inside_fields"])
    assert not np.isfinite(result.metrics["mean_bin_coefficient_of_variation"])
    assert np.all(~np.isfinite(result.per_unit_metrics["spatial_information_bits"]))
    assert np.all(~np.isfinite(result.per_unit_metrics["field_count"]))
    assert np.all(np.isfinite(result.per_unit_metrics["split_half_rate_map_correlation"]))

    exported_rows = list(csv.DictReader(result.tables["rate_map_panel_metrics"].open()))
    assert len(exported_rows) == 2
    assert all(row["place_field_metrics_supported"] == "False" for row in exported_rows)
    assert all(row["spatial_information_bits"] == "nan" for row in exported_rows)


def test_population_colormap_family_and_palette() -> None:
    positive_rate_map = np.asarray([[0.0, 0.2], [0.5, 1.0]], dtype=np.float32)
    signed_rate_map = np.asarray([[-0.4, 0.0], [0.3, 0.8]], dtype=np.float32)
    negligible_negative = np.asarray([[-0.001, 0.2], [0.5, 1.0]], dtype=np.float32)

    assert _population_is_signed(signed_rate_map) is True
    assert _population_is_signed(positive_rate_map) is False
    assert _population_is_signed(negligible_negative) is False

    assert _style_for_family(positive_rate_map, is_signed=False, colormap_mode="reds")[0] == "Reds"
    assert (
        _style_for_family(positive_rate_map, is_signed=False, colormap_mode="inferno")[0]
        == "inferno"
    )
    assert (
        _style_for_family(positive_rate_map, is_signed=False, colormap_mode="turbo")[0] == "turbo"
    )
    assert (
        _style_for_family(signed_rate_map, is_signed=True, colormap_mode="inferno")[0] == "RdBu_r"
    )


def test_rate_map_module_supports_near_nonnegative_units_with_tiny_negative_bins(
    tmp_path: Path,
    analysis_settings,
) -> None:
    episodes = 4
    steps = 100
    grid_width = 10
    x_positions = np.tile(np.arange(grid_width, dtype=np.float32), grid_width)
    y_positions = np.repeat(np.arange(grid_width, dtype=np.float32), grid_width)
    positions = np.stack([x_positions, y_positions], axis=-1)

    representation = np.zeros((episodes, steps, 1), dtype=np.float32)
    representation[:, 0, 0] = 2.0
    representation[:, 1, 0] = -0.05
    position_xy = np.repeat(positions[None, :, :], episodes, axis=0)

    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=10,
            num_bins_y=10,
            smoothing_sigma=0.0,
            min_occupancy=1e-6,
            rate_map_panel_top_k=1,
            rate_map_grid_top_k=1,
        ),
    )

    assert result.metadata["place_field_metrics_supported_unit_count"] == 1
    assert np.isfinite(result.metrics["mean_spatial_information_bits"])
    assert np.isfinite(result.per_unit_metrics["spatial_information_bits"][0])
    assert np.isfinite(result.per_unit_metrics["field_count"][0])


def test_world_overlay_for_wallgap_asym_exposes_segments_and_landmarks() -> None:
    overlay = resolve_world_overlay("MiniWorld-WallGapAsymLarge-v0")

    assert overlay is not None
    assert len(overlay.segments) > 0
    assert len(overlay.landmarks) >= 4


def test_rate_map_grid_show_all_units_overrides_grid_top_k(
    tmp_path: Path, analysis_settings
) -> None:
    episodes = 4
    steps = 18
    unit_count = 5
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.8, 0.8, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    centers = np.linspace(-0.6, 0.6, unit_count, dtype=np.float32)
    for episode_index in range(episodes):
        positions = np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
        position_sequences.append(positions)
        units = [
            _synthetic_place_activity(
                positions, center_x=float(center), center_y=float(center * 0.4)
            )
            for center in centers
        ]
        representation_sequences.append(np.stack(units, axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=24,
            num_bins_y=24,
            smoothing_sigma=0.6,
            min_occupancy=1e-6,
            rate_map_panel_top_k=3,
            rate_map_grid_top_k=2,
            rate_map_grid_show_all_units=True,
        ),
    )

    assert len(result.metadata["ranked_units_grid"]) == unit_count


def test_rate_map_panel_show_all_units_overrides_panel_top_k(
    tmp_path: Path,
    analysis_settings,
) -> None:
    episodes = 4
    steps = 18
    unit_count = 5
    x_positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    y_positions = np.linspace(-0.8, 0.8, steps, dtype=np.float32)
    position_sequences = []
    representation_sequences = []
    centers = np.linspace(-0.6, 0.6, unit_count, dtype=np.float32)
    for episode_index in range(episodes):
        positions = np.stack([x_positions, np.roll(y_positions, shift=episode_index)], axis=-1)
        position_sequences.append(positions)
        units = [
            _synthetic_place_activity(
                positions,
                center_x=float(center),
                center_y=float(center * 0.4),
            )
            for center in centers
        ]
        representation_sequences.append(np.stack(units, axis=-1))

    analysis_input = AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = _RateMapModuleBase().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=24,
            num_bins_y=24,
            smoothing_sigma=0.6,
            min_occupancy=1e-6,
            rate_map_panel_top_k=2,
            rate_map_grid_top_k=2,
            rate_map_panel_show_all_units=True,
        ),
    )

    assert len(result.metadata["ranked_units_summary"]) == unit_count


def test_world_overlay_for_wallgap_asym_large_exposes_segments_and_landmarks() -> None:
    overlay = resolve_world_overlay("MiniWorld-WallGapAsymLarge-v0")

    assert overlay is not None
    assert len(overlay.segments) > 0
    assert len(overlay.landmarks) >= 8
    landmark_labels = {layer.label for layer in overlay.landmarks}
    assert {"office_desks", "office_chairs", "medkit"} <= landmark_labels


def test_agg_safe_dpi_clamps_tall_summary_panel(tmp_path):
    """Regression: a 128-unit summary panel must stay under matplotlib Agg's 2**16 px cap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from placecell_research.analysis.rate_map_rendering import agg_safe_dpi

    requested_dpi = 170
    crashing_panel = plt.figure(figsize=(12.15, 3.1 * 128))
    safe_dpi = agg_safe_dpi(crashing_panel, requested_dpi)
    assert safe_dpi < requested_dpi
    assert max(crashing_panel.get_size_inches()) * safe_dpi < 65536
    plt.close(crashing_panel)

    small_panel = plt.figure(figsize=(12.15, 3.1 * 8))
    assert agg_safe_dpi(small_panel, requested_dpi) == requested_dpi
    plt.close(small_panel)

    overflowing = plt.figure(figsize=(0.4, 80.0))
    with pytest.raises(ValueError, match="Image size"):
        overflowing.savefig(tmp_path / "raw.png", dpi=900)
    overflowing.savefig(tmp_path / "clamped.png", dpi=agg_safe_dpi(overflowing, 900))
    assert (tmp_path / "clamped.png").exists()
    plt.close(overflowing)


def test_heading_overlay_spatial_axis_flips_for_y_down():
    """Regression: the heading rate-map overlay must invert y on y-down (jaxenstein) backends."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from placecell_research.analysis.heading_rate_map_overlay import _style_spatial_axis
    from placecell_research.analysis.world_overlay import WorldOverlay

    bounds = ((0.0, 10.0), (0.0, 25.0))

    figure, axis = plt.subplots()
    _style_spatial_axis(
        axis, bounds, WorldOverlay("env", (), y_down=True), x_label="x", y_label="y"
    )
    assert axis.get_ylim() == (25.0, 0.0)
    plt.close(figure)

    figure, axis = plt.subplots()
    _style_spatial_axis(
        axis, bounds, WorldOverlay("env", (), y_down=False), x_label="x", y_label="y"
    )
    assert axis.get_ylim() == (0.0, 25.0)
    plt.close(figure)


def test_field_traversal_reliability_counts_traversals_and_hits() -> None:
    representation = np.array([[[1.0], [0.0], [0.0], [0.0], [0.0]]], dtype=np.float32)
    position_xy = np.array(
        [[[0.5, 0.5], [0.5, 0.5], [1.5, 0.5], [0.5, 0.5], [1.5, 0.5]]], dtype=np.float32
    )
    statistics = occupancy.prepare_episode_bin_statistics(
        representation,
        position_xy,
        None,
        num_bins_x=2,
        num_bins_y=1,
        bounds=((0.0, 2.0), (0.0, 1.0)),
    )
    field_masks = np.array([[[True, False]]], dtype=bool)
    reliability, counts, _, _ = reliability_splits.compute_field_traversal_reliability(
        statistics,
        field_masks,
        minimum_traversals=2,
    )
    assert counts.tolist() == [2]
    assert reliability.tolist() == [0.5]
    gated_reliability, gated_counts, _, _ = reliability_splits.compute_field_traversal_reliability(
        statistics,
        field_masks,
        minimum_traversals=3,
    )
    assert gated_counts.tolist() == [2]
    assert np.isnan(gated_reliability[0])
    two_episodes = occupancy.prepare_episode_bin_statistics(
        np.repeat(representation, 2, axis=0),
        np.repeat(position_xy, 2, axis=0),
        None,
        num_bins_x=2,
        num_bins_y=1,
        bounds=((0.0, 2.0), (0.0, 1.0)),
    )
    headings = np.tile(np.array([0.0, 0.0, 0.0, np.pi, np.pi], dtype=np.float64), 2)
    (
        undirected,
        _,
        directional,
        directional_counts,
    ) = reliability_splits.compute_field_traversal_reliability(
        two_episodes,
        field_masks,
        minimum_traversals=1,
        flat_headings=headings,
    )
    assert undirected.tolist() == [0.5]
    assert directional.tolist() == [1.0]
    assert directional_counts.tolist() == [1]
