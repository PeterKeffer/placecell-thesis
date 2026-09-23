from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.uncertainty_signature import (
    UncertaintySignatureModule,
    _normalized_entropy,
    _spearman_with_block_permutation_p,
    bonferroni_corrected,
)


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


def _dataset_with_uncertainty_channel(
    num_episodes: int = 40, num_steps: int = 32, code_dim: int = 8
):
    """Decode noise varies per step; one large channel scales inversely with the noise."""
    rng = np.random.default_rng(1)
    positions = rng.uniform(0.0, 20.0, size=(num_episodes, num_steps, 2)).astype(np.float32)
    noise_scale = np.exp(
        rng.uniform(np.log(0.2), np.log(3.0), size=(num_episodes, num_steps))
    ).astype(np.float32)
    representation = np.zeros((num_episodes, num_steps, code_dim), dtype=np.float32)
    representation[..., :2] = positions + noise_scale[..., None] * rng.normal(
        size=(num_episodes, num_steps, 2)
    ).astype(np.float32)
    representation[..., 2] = 50.0 / noise_scale
    representation[..., 3:] = 0.1 * rng.normal(
        size=(num_episodes, num_steps, code_dim - 3)
    ).astype(np.float32)
    valid_mask = np.ones((num_episodes, num_steps), dtype=bool)
    return representation, positions, valid_mask


def test_constructed_uncertainty_channel_is_detected(tmp_path: Path) -> None:
    representation, positions, valid_mask = _dataset_with_uncertainty_channel()

    result = UncertaintySignatureModule().run(
        _analysis_input(representation, positions, valid_mask), tmp_path, {}
    )

    metrics = result.metrics
    assert metrics["spearman_error_vs_magnitude"] < -0.3
    assert metrics["spearman_error_vs_magnitude_p_value"] < 0.01
    assert metrics["spearman_error_vs_entropy"] > 0.3
    assert metrics["spearman_error_vs_entropy_p_value"] < 0.01
    assert result.figures["uncertainty_signature_hexbin"].exists()
    assert "Spearman" in result.metadata["uncertainty_signature_caption"]


def _null_dataset(seed: int, num_episodes: int = 24, num_steps: int = 48, code_dim: int = 8):
    """Codes carry position (so the decoder works) but nothing about the momentary error."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 20.0, size=(num_episodes, num_steps, 2)).astype(np.float32)
    representation = rng.normal(size=(num_episodes, num_steps, code_dim)).astype(np.float32)
    representation[..., :2] = positions + rng.normal(
        scale=1.0, size=(num_episodes, num_steps, 2)
    ).astype(np.float32)
    return representation, positions, np.ones((num_episodes, num_steps), dtype=bool)


def test_selection_between_the_two_tests_cannot_manufacture_significance(tmp_path: Path) -> None:
    """A null pair whose winner reads p < 0.05 only because it was the winner."""
    representation, positions, valid_mask = _null_dataset(0)

    result = UncertaintySignatureModule().run(
        _analysis_input(representation, positions, valid_mask), tmp_path, {}
    )

    metrics = result.metrics
    assert metrics["spearman_error_vs_magnitude_p_value"] < 0.05
    assert metrics["spearman_error_vs_entropy_p_value"] > 0.05
    assert result.metadata["uncertainty_signature_selected_statistic"] == "magnitude"
    assert (
        metrics["selected_by_larger_abs_rho_spearman"] == metrics["spearman_error_vs_magnitude"]
    )
    assert metrics["selected_by_larger_abs_rho_p_value_bonferroni"] == pytest.approx(
        2.0 * metrics["spearman_error_vs_magnitude_p_value"]
    )
    assert metrics["selected_by_larger_abs_rho_p_value_bonferroni"] > 0.05
    assert "Bonferroni" in result.metadata["uncertainty_signature_caption"]
    assert "larger |rho|" in result.metadata["uncertainty_signature_selection_rule"]


def test_bonferroni_correction_is_capped_at_one() -> None:
    assert bonferroni_corrected(0.6) == 1.0
    assert bonferroni_corrected(0.01) == 0.02
    assert np.isnan(bonferroni_corrected(float("nan")))


def test_single_episode_input_skips_with_reason(tmp_path: Path) -> None:
    representation, positions, valid_mask = _dataset_with_uncertainty_channel(num_episodes=1)

    result = UncertaintySignatureModule().run(
        _analysis_input(representation, positions, valid_mask), tmp_path, {}
    )

    assert result.metrics == {"uncertainty_signature_skipped": 1.0}
    assert "uncertainty_signature_skip_reason" in result.metadata


def test_dead_code_rows_have_no_entropy() -> None:
    codes = np.array(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    entropy = _normalized_entropy(codes)

    assert entropy[0] == pytest.approx(1.0)
    assert entropy[1] == pytest.approx(0.0)
    assert np.isnan(entropy[2])


def test_correlation_drops_dead_rows_instead_of_ranking_them() -> None:
    """The dead row must not enter the rank correlation as an extreme observation."""
    codes = np.zeros((6, 4), dtype=np.float32)
    for row in range(5):
        codes[row, : row + 1] = 1.0
    errors = np.arange(6, dtype=np.float64)
    entropy = _normalized_entropy(codes)
    blocks = [np.arange(6)]

    rho, _ = _spearman_with_block_permutation_p(
        errors, entropy, blocks, np.random.default_rng(0)
    )
    rho_without_dead_row, _ = _spearman_with_block_permutation_p(
        errors[:5], entropy[:5], [np.arange(5)], np.random.default_rng(0)
    )

    assert np.isnan(entropy[5])
    assert rho == pytest.approx(rho_without_dead_row)
