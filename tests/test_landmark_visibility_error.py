from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.landmark_visibility_error import (
    LandmarkVisibilityErrorModule,
    landmark_visibility,
)
from placecell_research.analysis.world_overlay import resolve_world_overlay

_ENV_ID = "MiniWorld-WallGapAsymLarge-v0"


def _analysis_input(
    representation, positions, heading, valid_mask, env_id=_ENV_ID
) -> AnalysisInput:
    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="test",
        metadata={"env_id": env_id},
    )


def _wall_segments() -> np.ndarray:
    return np.asarray(resolve_world_overlay(_ENV_ID).segments, dtype=np.float64)


def test_visibility_geometry_hand_cases() -> None:
    walls = _wall_segments()
    positions = np.asarray(
        [
            [12.0, 21.0],
            [12.0, 21.0],
            [12.0, 21.0],
            [-12.0, 4.0],
            [-12.0, 0.0],
        ]
    )
    headings = np.asarray(
        [0.0, np.pi, np.pi, 0.0, np.arctan2(4.0, 10.0)]
    )
    landmarks = np.asarray(
        [
            [15.0, 21.0],
            [-15.0, 28.0],
            [4.0, 4.0],
            [-2.0, -4.0],
        ]
    )

    visible = landmark_visibility(positions, headings, landmarks, walls)

    assert visible[0, 0]
    assert not visible[1, 0]
    assert not visible[2, 1]
    assert not visible[3, 2]
    assert visible[4, 3]


def _dataset_with_drift_after_losing_landmarks(num_episodes: int = 12, num_steps: int = 64):
    """First half: agent faces a nearby cone (visible, low noise)."""
    rng = np.random.default_rng(2)
    half = num_steps // 2
    positions = np.zeros((num_episodes, num_steps, 2), dtype=np.float32)
    heading = np.zeros((num_episodes, num_steps), dtype=np.float32)
    positions[:, :half] = np.asarray([12.0, 21.0]) + rng.uniform(
        -1.0, 1.0, size=(num_episodes, half, 2)
    )
    heading[:, :half] = 0.0
    positions[:, half:] = np.asarray([0.0, 34.0]) + rng.uniform(
        -0.5, 0.5, size=(num_episodes, num_steps - half, 2)
    )
    heading[:, half:] = -np.pi / 2.0
    noise_scale = np.full((num_episodes, num_steps), 0.3, dtype=np.float32)
    noise_scale[:, half:] += 0.08 * np.arange(1, num_steps - half + 1, dtype=np.float32)
    representation = np.zeros((num_episodes, num_steps, 4), dtype=np.float32)
    representation[..., :2] = positions + noise_scale[..., None] * rng.normal(
        size=(num_episodes, num_steps, 2)
    ).astype(np.float32)
    representation[..., 2:] = 0.1 * rng.normal(size=(num_episodes, num_steps, 2)).astype(
        np.float32
    )
    valid_mask = np.ones((num_episodes, num_steps), dtype=bool)
    return representation, positions, heading, valid_mask


def test_error_grows_with_steps_since_landmark(tmp_path: Path) -> None:
    representation, positions, heading, valid_mask = _dataset_with_drift_after_losing_landmarks()

    result = LandmarkVisibilityErrorModule().run(
        _analysis_input(representation, positions, heading, valid_mask), tmp_path, {}
    )

    metrics = result.metrics
    assert metrics["mean_error_steps_since_0"] < metrics["mean_error_steps_since_5_16"]
    assert metrics["mean_error_steps_since_5_16"] < metrics["mean_error_steps_since_17_64"]
    assert np.isnan(metrics["mean_error_steps_since_over_64"])
    assert metrics["error_vs_log_steps_since_slope"] > 0.0
    assert 0.4 < metrics["fraction_steps_landmark_visible"] < 0.6
    assert result.figures["landmark_visibility_error_curve"].exists()
    assert result.metadata["fov_total_angle_degrees"] == 60.0


def test_missing_heading_skips_with_reason(tmp_path: Path) -> None:
    representation, positions, heading, valid_mask = _dataset_with_drift_after_losing_landmarks(
        num_episodes=2, num_steps=8
    )

    result = LandmarkVisibilityErrorModule().run(
        _analysis_input(representation, positions, None, valid_mask), tmp_path, {}
    )

    assert result.metrics == {"landmark_visibility_error_skipped": 1.0}
    assert "landmark_visibility_error_skip_reason" in result.metadata


def test_unknown_env_skips_with_reason(tmp_path: Path) -> None:
    representation, positions, heading, valid_mask = _dataset_with_drift_after_losing_landmarks(
        num_episodes=2, num_steps=8
    )

    result = LandmarkVisibilityErrorModule().run(
        _analysis_input(representation, positions, heading, valid_mask, env_id="Nowhere-v0"),
        tmp_path,
        {},
    )

    assert result.metrics == {"landmark_visibility_error_skipped": 1.0}
    assert "no world overlay" in result.metadata["landmark_visibility_error_skip_reason"].lower()
