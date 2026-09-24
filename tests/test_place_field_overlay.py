from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import rgb_to_hsv

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.place_field_overlay import (
    PlaceFieldOverlayModule,
    _active_unit_indices,
    _assign_frame_colors,
    _composite_active_fields,
    _frame_step_indices,
    _kwinners_k_from_fraction,
    _normalize_fields,
)


def test_active_unit_indices_returns_active_units_sorted_by_activation() -> None:
    step_activation = np.array([0.0, 0.5, 0.0, 0.3, 0.9], dtype=np.float32)

    kept, dropped = _active_unit_indices(step_activation, threshold=0.0, max_cells=10)

    np.testing.assert_array_equal(kept, np.array([4, 1, 3]))
    assert dropped == 0


def test_active_unit_indices_caps_at_max_cells_and_reports_dropped() -> None:
    step_activation = np.array([0.0, 0.5, 0.0, 0.3, 0.9], dtype=np.float32)

    kept, dropped = _active_unit_indices(step_activation, threshold=0.0, max_cells=2)

    np.testing.assert_array_equal(kept, np.array([4, 1]))
    assert dropped == 1


def test_kwinners_k_from_fraction_matches_sparsifier_rounding() -> None:
    assert _kwinners_k_from_fraction(512, 0.02) == 10
    assert _kwinners_k_from_fraction(100, 0.001) == 1
    assert _kwinners_k_from_fraction(512, 0.0) == 0
    assert _kwinners_k_from_fraction(512, None) == 0


def test_active_unit_indices_kwinners_k_overrides_max_cells() -> None:
    step_activation = np.array([0.1, 0.9, 0.5, 0.3, 0.7], dtype=np.float32)

    kept, dropped = _active_unit_indices(
        step_activation, threshold=0.0, max_cells=20, kwinners_k=2
    )

    np.testing.assert_array_equal(kept, np.array([1, 4]))
    assert dropped == 3


def test_active_unit_indices_respects_threshold_and_handles_empty() -> None:
    above_threshold, _ = _active_unit_indices(
        np.array([0.1, 0.5, 0.05], dtype=np.float32), threshold=0.2, max_cells=5
    )
    np.testing.assert_array_equal(above_threshold, np.array([1]))

    empty, dropped = _active_unit_indices(np.zeros(4, dtype=np.float32), threshold=0.0, max_cells=5)
    assert empty.size == 0
    assert dropped == 0


def test_assign_frame_colors_evenly_spaces_hues_for_maximum_separation() -> None:
    colors = _assign_frame_colors(np.array([3, 1, 9, 4]))

    hues = np.sort(rgb_to_hsv(colors)[:, 0])
    np.testing.assert_allclose(hues, [0.0, 0.25, 0.5, 0.75], atol=1e-5)


def test_assign_frame_colors_distinct_at_realistic_active_counts() -> None:
    for active_count in (10, 20):
        active = np.arange(active_count) * 7 + 1
        colors = _assign_frame_colors(active)
        assert len({tuple(np.round(color, 5)) for color in colors}) == active_count


def test_assign_frame_colors_is_deterministic_for_the_same_active_set() -> None:
    first = _assign_frame_colors(np.array([5, 2, 8]))
    second = _assign_frame_colors(np.array([5, 2, 8]))
    np.testing.assert_array_equal(first, second)


def test_assign_frame_colors_handles_empty_active_set() -> None:
    assert _assign_frame_colors(np.array([], dtype=np.int64)).shape == (0, 3)


def test_normalize_fields_self_normalizes_each_unit_and_zeros_nan_and_negatives() -> None:
    rate_maps = np.array(
        [
            [[np.nan, 1.0], [2.0, 4.0]],
            [[-1.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    normalized = _normalize_fields(rate_maps)

    np.testing.assert_allclose(normalized[0], np.array([[0.0, 0.25], [0.5, 1.0]]))
    np.testing.assert_allclose(normalized[1], np.array([[0.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(normalized[2], np.zeros((2, 2)))
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0


def test_composite_max_blend_shows_overlap_as_mixed_color() -> None:
    field_red = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    field_blue = np.array([[0.0, 1.0], [0.0, 0.5]], dtype=np.float32)
    fields = np.stack([field_red, field_blue])
    colors = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    background = np.array([0.05, 0.05, 0.05], dtype=float)

    image = _composite_active_fields(fields, colors, blend_mode="max", background=background)

    np.testing.assert_allclose(image[0, 0], [1.0, 0.05, 0.05])
    np.testing.assert_allclose(image[0, 1], [0.05, 0.05, 1.0])
    np.testing.assert_allclose(image[1, 1], [0.5, 0.05, 0.5])
    np.testing.assert_allclose(image[1, 0], [0.05, 0.05, 0.05])


def test_composite_additive_blend_sums_overlapping_layers() -> None:
    field_red = np.array([[0.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    field_blue = np.array([[0.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    fields = np.stack([field_red, field_blue])
    colors = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)

    image = _composite_active_fields(
        fields, colors, blend_mode="additive", background=np.zeros(3)
    )

    np.testing.assert_allclose(image[1, 1], [0.5, 0.0, 0.5])


def test_composite_with_no_active_layers_returns_background() -> None:
    image = _composite_active_fields(
        np.zeros((0, 2, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=float),
        blend_mode="max",
        background=np.array([0.1, 0.1, 0.1], dtype=float),
    )
    assert image.shape == (2, 3, 3)
    np.testing.assert_allclose(image, 0.1)


def test_frame_step_indices_strides_long_episodes_and_keeps_endpoints() -> None:
    np.testing.assert_array_equal(_frame_step_indices(10, max_frames=0), np.arange(10))
    np.testing.assert_array_equal(_frame_step_indices(10, max_frames=20), np.arange(10))

    strided = _frame_step_indices(100, max_frames=10)
    assert strided.size <= 10
    assert strided[0] == 0
    assert strided[-1] == 99
    assert np.all(np.diff(strided) > 0)


def _sparse_overlay_input() -> AnalysisInput:
    episodes, steps, units = 3, 24, 8
    rng = np.random.default_rng(3)
    positions = rng.uniform(0.0, 5.0, size=(episodes, steps, 2)).astype(np.float32)
    representation = np.zeros((episodes, steps, units), dtype=np.float32)
    for episode in range(episodes):
        for step in range(steps):
            active = rng.choice(units, size=3, replace=False)
            representation[episode, step, active] = rng.uniform(0.5, 1.5, size=3).astype(np.float32)
    heading = rng.uniform(-np.pi, np.pi, size=(episodes, steps)).astype(np.float32)
    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": ""},
    )


def test_place_field_overlay_module_writes_one_gif_for_the_run(
    tmp_path: Path, analysis_settings
) -> None:
    analysis_input = _sparse_overlay_input()

    result = PlaceFieldOverlayModule().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=16,
            num_bins_y=16,
            smoothing_sigma=0.0,
            min_occupancy=1e-6,
            example_episode_index=0,
            place_field_overlay_max_frames=6,
        ),
    )

    gif_path = (
        tmp_path
        / "place_field_overlay"
        / "place_field_overlay__encoder.place_codes__validation.gif"
    )
    assert gif_path.exists()
    assert "place_field_overlay_gif" in result.figures
    assert result.metrics["selected_episode_index"] == 0.0
    assert result.metrics["selected_episode_length"] == 24.0
    assert result.metrics["frames_rendered"] == 6.0
    assert result.metrics["max_active_cells_in_frame"] == 3.0
    assert result.metrics["dropped_cell_count"] == 0.0


def test_place_field_overlay_reports_dropped_cells_when_capped(
    tmp_path: Path, analysis_settings
) -> None:
    analysis_input = _sparse_overlay_input()

    result = PlaceFieldOverlayModule().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=16,
            num_bins_y=16,
            smoothing_sigma=0.0,
            min_occupancy=1e-6,
            example_episode_index=0,
            place_field_overlay_max_frames=6,
            place_field_overlay_max_cells=2,
        ),
    )

    assert result.metrics["max_active_cells_in_frame"] == 2.0
    assert result.metrics["dropped_cell_count"] == 6.0


def test_place_field_overlay_kwinners_sparsifies_dense_codes(
    tmp_path: Path, analysis_settings
) -> None:
    episodes, steps, units = 2, 12, 16
    rng = np.random.default_rng(1)
    representation = rng.uniform(0.1, 1.0, size=(episodes, steps, units)).astype(np.float32)
    positions = rng.uniform(0.0, 5.0, size=(episodes, steps, 2)).astype(np.float32)
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="predictor.place_codes",
        label="predictor_place_cells",
        split_name="validation",
        metadata={"env_id": ""},
    )

    result = PlaceFieldOverlayModule().run(
        analysis_input,
        tmp_path,
        analysis_settings(
            num_bins_x=16,
            num_bins_y=16,
            smoothing_sigma=0.0,
            min_occupancy=1e-6,
            example_episode_index=0,
            place_field_overlay_max_frames=4,
            place_field_overlay_kwinners_k_fraction=0.25,
        ),
    )

    assert result.metrics["active_selection_k"] == 4.0
    assert result.metrics["max_active_cells_in_frame"] == 4.0
