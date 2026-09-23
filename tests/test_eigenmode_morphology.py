from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.eigenmode_morphology import (
    CHECKERBOARD,
    HORIZONTAL_BAND,
    VERTICAL_BAND,
    EigenmodeMorphologyModule,
    _mode_group_scores,
    _normalized_map_values,
    _shuffle_map_p_values,
    cosine_template_bank,
    score_cosine_morphology,
)
from placecell_research.analysis.registry import ANALYSIS_MODULES


def _lattice_analysis_input(grid_size: int = 4) -> AnalysisInput:
    coordinates = np.linspace(-1.5, 1.5, grid_size, dtype=np.float32)
    episodes: list[np.ndarray] = []
    for y_index in range(grid_size):
        row = np.asarray(
            [[coordinates[x_index], coordinates[y_index]] for x_index in range(grid_size)],
            dtype=np.float32,
        )
        episodes.extend([row, row[::-1].copy()])
    for x_index in range(grid_size):
        column = np.asarray(
            [[coordinates[x_index], coordinates[y_index]] for y_index in range(grid_size)],
            dtype=np.float32,
        )
        episodes.extend([column, column[::-1].copy()])
    position_xy = np.stack(episodes)
    representation = np.stack(
        [
            position_xy[..., 0],
            position_xy[..., 1],
            position_xy[..., 0] + position_xy[..., 1],
        ],
        axis=-1,
    )
    return AnalysisInput(
        representation=representation.astype(np.float32),
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones(position_xy.shape[:2], dtype=bool),
        source_name="predictor.hidden_state",
        label="predictor_hidden_state",
        split_name="validation",
    )


def _cosine_patterns(size: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = (np.arange(size, dtype=np.float64) + 0.5) / size
    x, y = np.meshgrid(coordinates, coordinates)
    vertical = np.cos(3.0 * np.pi * x)
    horizontal = np.cos(2.0 * np.pi * y)
    checkerboard = np.cos(2.0 * np.pi * x) * np.cos(3.0 * np.pi * y)
    return vertical, horizontal, checkerboard


def test_cosine_morphology_distinguishes_band_orientation_and_checkerboards() -> None:
    vertical, horizontal, checkerboard = _cosine_patterns()
    scores = score_cosine_morphology(
        np.stack([vertical, horizontal, checkerboard]),
        max_frequency=4,
    )

    np.testing.assert_array_equal(
        scores["best_kind_code"],
        [VERTICAL_BAND, HORIZONTAL_BAND, CHECKERBOARD],
    )
    np.testing.assert_array_equal(scores["best_frequency_x"], [3, 0, 2])
    np.testing.assert_array_equal(scores["best_frequency_y"], [0, 2, 3])
    assert np.all(scores["best_score"] > 0.999)


def test_eigenmode_morphology_is_registered_as_a_standard_analysis_module() -> None:
    assert ANALYSIS_MODULES["eigenmode_morphology"] is EigenmodeMorphologyModule


def test_degenerate_rotated_band_modes_are_scored_as_one_subspace() -> None:
    vertical, horizontal, _checkerboard = _cosine_patterns()
    first = (vertical + horizontal) / np.sqrt(2.0)
    second = (vertical - horizontal) / np.sqrt(2.0)
    mode_maps = np.stack([first, second])
    visited_mask = np.ones(first.shape, dtype=bool)
    bank = cosine_template_bank(first.shape, visited_mask, max_frequency=4)

    groups = _mode_group_scores(
        mode_maps,
        np.asarray([0.1, 0.1001]),
        visited_mask,
        bank,
        relative_tolerance=0.01,
        shuffle_count=0,
        seed=0,
        fdr_alpha=0.05,
    )

    assert len(groups) == 1
    assert groups[0].first_mode_rank == 1
    assert groups[0].last_mode_rank == 2
    assert groups[0].best_kind_code in {VERTICAL_BAND, HORIZONTAL_BAND}
    assert groups[0].best_score > 0.999


def test_spatial_shuffle_null_rejects_a_boundary_anchored_cosine() -> None:
    vertical, _horizontal, _checkerboard = _cosine_patterns()
    maps = vertical[None]
    visited_mask = np.ones(vertical.shape, dtype=bool)
    bank = cosine_template_bank(vertical.shape, visited_mask, max_frequency=4)
    scores = score_cosine_morphology(maps, visited_mask, max_frequency=4)
    normalized = _normalized_map_values(maps, visited_mask)

    p_values = _shuffle_map_p_values(
        normalized,
        scores["best_score"],
        np.asarray([0]),
        bank,
        shuffle_count=199,
        seed=0,
    )

    assert p_values[0] <= 0.01


def test_eigenmode_morphology_module_writes_metrics_table_and_panel(tmp_path: Path) -> None:
    result = EigenmodeMorphologyModule().run(
        _lattice_analysis_input(),
        tmp_path,
        {
            "transition_geometry_num_bins_x": 4,
            "transition_geometry_num_bins_y": 4,
            "transition_geometry_num_modes": 3,
            "transition_geometry_include_self_transitions": False,
            "eigenmode_morphology_max_frequency": 2,
            "eigenmode_morphology_shuffle_count": 19,
            "eigenmode_morphology_shuffle_top_k": 3,
            "eigenmode_morphology_fdr_alpha": 0.05,
            "eigenmode_morphology_render_top_k": 3,
            "smoothing_sigma": 0.0,
            "min_occupancy": 1e-6,
        },
    )

    assert result.metrics["transition_mode_group_count"] >= 1.0
    assert result.metrics["num_morphology_candidates_tested"] == 3.0
    assert result.per_unit_metrics["best_score"].shape == (3,)
    assert result.figures["eigenmode_morphology"].exists()
    assert result.tables["eigenmode_morphology_mode_groups"].exists()
