from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.decode_convergence import DecodeConvergenceModule


def _analysis_input(representation, positions, valid_mask) -> AnalysisInput:
    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="test",
    )


def _converging_dataset(num_episodes: int = 16, num_steps: int = 64, code_dim: int = 6):
    """Code = position + noise whose scale decays over steps within each episode."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, 20.0, size=(num_episodes, num_steps, 2)).astype(np.float32)
    noise_std = 0.1 + 4.0 * np.exp(-np.arange(num_steps) / 6.0)
    representation = np.zeros((num_episodes, num_steps, code_dim), dtype=np.float32)
    representation[..., :2] = positions + (
        noise_std[None, :, None] * rng.normal(size=(num_episodes, num_steps, 2))
    ).astype(np.float32)
    representation[..., 2:] = rng.normal(size=(num_episodes, num_steps, code_dim - 2)).astype(
        np.float32
    )
    valid_mask = np.ones((num_episodes, num_steps), dtype=bool)
    return representation, positions, valid_mask


def test_decaying_noise_yields_within_episode_convergence(tmp_path: Path) -> None:
    representation, positions, valid_mask = _converging_dataset()

    result = DecodeConvergenceModule().run(
        _analysis_input(representation, positions, valid_mask), tmp_path, {}
    )

    metrics = result.metrics
    assert metrics["mean_error_steps_1_8"] > metrics["asymptote_error"]
    assert metrics["mean_error_steps_9_64"] < metrics["mean_error_steps_1_8"]
    assert metrics["error_ratio_initial_over_asymptote"] > 1.5
    assert np.isfinite(metrics["convergence_step"])
    assert metrics["convergence_step"] > 1.0
    assert result.figures["decode_convergence_curve"].exists()


def test_single_episode_input_skips_with_reason(tmp_path: Path) -> None:
    representation, positions, valid_mask = _converging_dataset(num_episodes=1)

    result = DecodeConvergenceModule().run(
        _analysis_input(representation, positions, valid_mask), tmp_path, {}
    )

    assert result.metrics == {"decode_convergence_skipped": 1.0}
    assert "decode_convergence_skip_reason" in result.metadata


def test_no_step_with_enough_holdout_episodes_skips_with_reason(tmp_path: Path) -> None:
    """Two episodes leave one held-out episode, so every step falls under the minimum."""
    representation, positions, valid_mask = _converging_dataset(num_episodes=2)

    result = DecodeConvergenceModule().run(
        _analysis_input(representation, positions, valid_mask), tmp_path, {}
    )

    assert result.metrics == {"decode_convergence_skipped": 1.0}
    assert "held-out episodes" in result.metadata["decode_convergence_skip_reason"]
    assert result.figures == {}
