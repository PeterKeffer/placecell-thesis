from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.decode_structural_room import DecodeStructuralRoomModule
from placecell_research.analysis.registry import run_analysis_modules


def test_structural_room_decode_uses_episode_held_out_named_rooms(tmp_path: Path) -> None:
    room_positions = np.asarray(
        [[0.0, 12.0], [0.0, 0.0], [-12.0, -12.0], [12.0, -12.0]],
        dtype=np.float32,
    )
    room_codes = np.eye(4, dtype=np.float32)
    episode_count = 8
    analysis_input = AnalysisInput(
        representation=np.broadcast_to(room_codes, (episode_count, 4, 4)).copy(),
        position_xy=np.broadcast_to(room_positions, (episode_count, 4, 2)).copy(),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episode_count, 4), dtype=bool),
        source_name="level_3:encoder.place_codes",
        label="level_3",
        split_name="test",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    result = DecodeStructuralRoomModule().run(analysis_input, tmp_path, {})

    assert result.metrics["structural_room_decode_accuracy"] == pytest.approx(1.0)
    assert result.metrics["structural_room_decode_balanced_accuracy"] == pytest.approx(1.0)
    assert result.metrics["structural_room_decode_macro_f1"] == pytest.approx(1.0)
    assert result.metrics["structural_room_decode_class_count"] == 4
    assert result.metadata["train_episode_count"] == 6
    assert result.metadata["test_episode_count"] == 2


def test_structural_room_decode_skips_unknown_environment(tmp_path: Path) -> None:
    analysis_input = AnalysisInput(
        representation=np.ones((2, 2, 2), dtype=np.float32),
        position_xy=np.zeros((2, 2, 2), dtype=np.float32),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((2, 2), dtype=bool),
        source_name="encoder.place_codes",
        label="root",
        split_name="test",
        metadata={"env_id": "unknown"},
    )

    result = DecodeStructuralRoomModule().run(analysis_input, tmp_path, {})

    assert result.metrics == {}
    assert result.metadata["decode_skipped"] is True


def test_hierarchy_diagnostics_run_together_through_registry(tmp_path: Path) -> None:
    room_positions = np.asarray(
        [[0.0, 12.0], [0.0, 0.0], [-12.0, -12.0], [12.0, -12.0]],
        dtype=np.float32,
    )
    room_codes = np.eye(4, dtype=np.float32)
    episode_count = 8
    analysis_input = AnalysisInput(
        representation=np.broadcast_to(room_codes, (episode_count, 4, 4)).copy(),
        position_xy=np.broadcast_to(room_positions, (episode_count, 4, 2)).copy(),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episode_count, 4), dtype=bool),
        source_name="level_3:encoder.place_codes",
        label="level_3",
        split_name="test",
        metadata={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
    )

    results = run_analysis_modules(
        analysis_input,
        tmp_path,
        {"spatial_code_dynamics_lags": [1, 2]},
        ["decode_structural_room", "spatial_code_dynamics"],
    )

    assert results["decode_structural_room"].metrics[
        "structural_room_decode_balanced_accuracy"
    ] == pytest.approx(1.0)
    assert results["spatial_code_dynamics"].metrics["lag_1_pair_count"] == 24
    assert results["spatial_code_dynamics"].tables["lag_dynamics"].exists()
