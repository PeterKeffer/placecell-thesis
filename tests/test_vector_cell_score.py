"""Tests for object-vector (OVC) / boundary-vector (BVC) cell scoring."""

from __future__ import annotations

import numpy as np

from placecell_research.analysis import vector_cell_score as vcs
from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.world_overlay import LandmarkLayer, WorldOverlay


def test_nearest_object_vectors_distance_and_bearing() -> None:
    points = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    landmarks = np.array([[0.0, 0.0]])
    distance, bearing = vcs._nearest_object_vectors(points, landmarks)
    assert np.allclose(distance, [2.0, 3.0, 1.0])
    assert np.allclose(bearing, [0.0, np.pi / 2, np.pi])


def test_point_segment_distance_clamps_to_endpoints() -> None:
    points = np.array([[0.0, 1.0], [5.0, 1.0]])
    distance, _ = vcs._point_segment_vectors(points, np.array([0.0, 0.0]), np.array([2.0, 0.0]))
    assert np.allclose(distance, [1.0, np.hypot(3.0, 1.0)])


def _object_overlay() -> WorldOverlay:
    return WorldOverlay(
        env_id="test-ovc",
        segments=(((-5.0, -5.0), (5.0, -5.0)), ((-5.0, 5.0), (5.0, 5.0))),
        landmarks=(LandmarkLayer(label="obj", marker="o", color="r", positions=((0.0, 0.0),)),),
    )


def test_object_vector_module_detects_a_synthetic_ovc(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        vcs, "resolve_world_overlay", lambda env_id, env_kwargs=None: _object_overlay()
    )
    rng = np.random.default_rng(0)
    episodes, steps = 8, 600
    positions = rng.uniform(-4.5, 4.5, size=(episodes, steps, 2))
    distance = np.linalg.norm(positions, axis=-1)
    bearing = np.arctan2(positions[..., 1], positions[..., 0])
    angle_diff = np.arctan2(np.sin(bearing), np.cos(bearing))

    ovc = np.exp(-((distance - 2.5) ** 2) / 0.4 - (angle_diff**2) / 0.3)
    near_object = np.exp(-(distance**2) / 0.4)
    noise = rng.standard_normal((episodes, steps))
    representation = np.stack([ovc, near_object, noise], axis=-1).astype(np.float32)

    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions.astype(np.float32),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="grid.hidden_state",
        label="grid_cells",
        split_name="test",
        metadata={"env_id": "test-ovc"},
    )
    result = vcs.ObjectVectorScoreModule().run(analysis_input, tmp_path, {})

    reliability = result.per_unit_metrics["object_vector_score_reliability"]
    peak_distance = result.per_unit_metrics["object_vector_score_peak_distance"]
    assert reliability[0] > 0.6
    assert peak_distance[0] > 1.5
    assert peak_distance[1] < 1.0
    assert reliability[2] < 0.3
    assert result.metrics["fraction_object_vector_cells"] > 0.0
    assert np.isnan(result.metrics["num_object_vector_score_significant"])
    assert result.metadata["shuffle_significance_tested"] is False
    assert result.figures


def test_object_module_skips_gracefully_without_landmarks(monkeypatch, tmp_path) -> None:
    bare = WorldOverlay(env_id="bare", segments=(((-1.0, -1.0), (1.0, -1.0)),), landmarks=())
    monkeypatch.setattr(vcs, "resolve_world_overlay", lambda env_id, env_kwargs=None: bare)
    analysis_input = AnalysisInput(
        representation=np.zeros((2, 50, 3), dtype=np.float32),
        position_xy=np.zeros((2, 50, 2), dtype=np.float32),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((2, 50), dtype=bool),
        source_name="grid.hidden_state",
        label="grid_cells",
        split_name="test",
        metadata={"env_id": "bare"},
    )
    result = vcs.ObjectVectorScoreModule().run(analysis_input, tmp_path, {})
    assert result.metrics["object_vector_score_units_scored"] == 0.0
    assert "skipped_reason" in result.metadata
