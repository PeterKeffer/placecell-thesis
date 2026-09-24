from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis import transition_geometry
from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.transition_geometry import (
    TransitionGeometryAlignmentModule,
    TransitionGeometryGraphModule,
    TransitionGeometryPanelModule,
)


def _lattice_analysis_input(grid_size: int = 4) -> AnalysisInput:
    x_coordinates = np.linspace(-1.5, 1.5, grid_size, dtype=np.float32)
    y_coordinates = np.linspace(-1.5, 1.5, grid_size, dtype=np.float32)

    episodes: list[np.ndarray] = []
    for y_index in range(grid_size):
        row = np.asarray(
            [[x_coordinates[x_index], y_coordinates[y_index]] for x_index in range(grid_size)],
            dtype=np.float32,
        )
        if y_index % 2 == 1:
            row = row[::-1]
        episodes.extend([row, row[::-1].copy()])
    for x_index in range(grid_size):
        column = np.asarray(
            [[x_coordinates[x_index], y_coordinates[y_index]] for y_index in range(grid_size)],
            dtype=np.float32,
        )
        if x_index % 2 == 1:
            column = column[::-1]
        episodes.extend([column, column[::-1].copy()])

    position_xy = np.stack(episodes, axis=0).astype(np.float32, copy=False)
    valid_mask = np.ones(position_xy.shape[:2], dtype=bool)
    x_feature = position_xy[..., 0]
    y_feature = position_xy[..., 1]
    diagonal_feature = x_feature + y_feature
    representation = np.stack([x_feature, y_feature, diagonal_feature], axis=-1).astype(
        np.float32,
        copy=False,
    )
    return AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="predictor.hidden_state",
        label="predictor_hidden_state",
        split_name="validation",
    )


def test_transition_geometry_finds_lattice_aligned_modes(tmp_path: Path) -> None:
    analysis_input = _lattice_analysis_input()
    config = {
        "transition_geometry_num_bins_x": 4,
        "transition_geometry_num_bins_y": 4,
        "transition_geometry_num_modes": 3,
        "transition_geometry_alignment_top_k": 3,
        "transition_geometry_include_self_transitions": False,
        "smoothing_sigma": 0.0,
        "min_occupancy": 1e-6,
    }
    graph_result = TransitionGeometryGraphModule().run(
        analysis_input,
        tmp_path,
        config,
    )
    alignment_result = TransitionGeometryAlignmentModule().run(
        analysis_input,
        tmp_path,
        config,
    )
    panel_result = TransitionGeometryPanelModule().run(
        analysis_input,
        tmp_path,
        config,
    )

    assert graph_result.metrics["transition_geometry_transitions_used"] > 0.0
    assert graph_result.metrics["transition_geometry_selected_mode_count"] >= 2.0
    assert alignment_result.metrics["transition_geometry_max_unit_mode_abs_correlation"] > 0.7
    assert alignment_result.metrics["transition_geometry_mean_mode_best_unit_abs_correlation"] > 0.4
    assert panel_result.figures["transition_geometry"].exists()
    assert alignment_result.per_unit_metrics["best_transition_mode_abs_correlation"].shape == (3,)
    assert alignment_result.per_unit_metrics["best_transition_mode_rank"].shape == (3,)
    assert set(graph_result.metadata["transition_geometry_timing_seconds"]) >= {
        "prepare_bins",
        "transition_counts",
        "eigendecomposition",
    }
    assert set(alignment_result.metadata["transition_geometry_timing_seconds"]) >= {
        "rate_maps",
        "mode_alignment",
    }
    assert set(panel_result.metadata["transition_geometry_timing_seconds"]) >= {
        "render_figure",
    }


def test_transition_geometry_modules_share_graph_and_alignment_computation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    analysis_input = _lattice_analysis_input()
    config = {
        "transition_geometry_num_bins_x": 4,
        "transition_geometry_num_bins_y": 4,
        "transition_geometry_num_modes": 3,
        "transition_geometry_alignment_top_k": 3,
        "transition_geometry_include_self_transitions": False,
        "smoothing_sigma": 0.0,
        "min_occupancy": 1e-6,
    }
    real_modes = transition_geometry.transition_laplacian_modes
    mode_calls = 0

    def count_modes(*args, **kwargs):
        nonlocal mode_calls
        mode_calls += 1
        return real_modes(*args, **kwargs)

    monkeypatch.setattr(transition_geometry, "transition_laplacian_modes", count_modes)

    TransitionGeometryGraphModule().run(analysis_input, tmp_path, config)
    TransitionGeometryAlignmentModule().run(analysis_input, tmp_path, config)
    TransitionGeometryPanelModule().run(analysis_input, tmp_path, config)

    assert mode_calls == 1
