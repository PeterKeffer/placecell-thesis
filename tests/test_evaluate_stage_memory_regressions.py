from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from placecell_research.evaluation import metrics as evaluation_metrics
from placecell_research.evaluation.decode import (
    fit_ridge_position_decoder,
    score_ridge_position_decoder,
)
from placecell_research.evaluation.metrics import summarize_code_sparsity
from placecell_research.stages.evaluate_model import _transfer_decoder_metrics

CODE_UNITS = 512
ACTIVE_UNITS_PER_STEP = 10
BACKBONE_UNITS = 3537


def _unmaterialized_sparsity(codes: np.ndarray, active_threshold: float = 1e-4) -> dict[str, float]:
    """The pre-chunking implementation, kept here as the reference."""
    flattened = codes.reshape(-1, codes.shape[-1])
    absolute_codes = np.abs(flattened)
    is_active = absolute_codes > active_threshold
    return {
        "mean_activation": float(absolute_codes.mean()),
        "fraction_active": float(is_active.mean()),
        "active_units_mean": float(is_active.sum(axis=1).mean()),
        "active_units_std": float(is_active.sum(axis=1).std()),
    }


def _kwinner_codes(num_rows: int, *, seed: int = 0) -> np.ndarray:
    """[num_rows, 512] with exactly 10 units above threshold per row, the rest exactly zero."""
    generator = np.random.default_rng(seed)
    codes = np.zeros((num_rows, CODE_UNITS), dtype=np.float32)
    ranking = generator.random((num_rows, CODE_UNITS))
    winners = np.argpartition(ranking, -ACTIVE_UNITS_PER_STEP, axis=1)[:, -ACTIVE_UNITS_PER_STEP:]
    codes[np.arange(num_rows)[:, None], winners] = generator.uniform(
        0.05, 1.0, size=(num_rows, ACTIVE_UNITS_PER_STEP)
    ).astype(np.float32)
    return codes


def _dense_backbone(num_rows: int, *, seed: int = 1) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.standard_normal((num_rows, BACKBONE_UNITS), dtype=np.float32)


@pytest.mark.parametrize("chunk_elements", [1 << 22, 1 << 16, 1])
def test_summarize_code_sparsity_chunking_does_not_move_the_scalars(
    monkeypatch, chunk_elements: int
) -> None:
    monkeypatch.setattr(evaluation_metrics, "_SPARSITY_CHUNK_ELEMENTS", chunk_elements)
    for codes in (_kwinner_codes(2048), _dense_backbone(512)):
        expected = _unmaterialized_sparsity(codes)
        summary = summarize_code_sparsity(codes)
        assert summary["fraction_active"] == expected["fraction_active"]
        assert summary["active_units_mean"] == expected["active_units_mean"]
        assert summary["active_units_std"] == expected["active_units_std"]
        assert summary["mean_activation"] == pytest.approx(expected["mean_activation"], rel=1e-6)


def test_summarize_code_sparsity_reports_the_kwinner_budget() -> None:
    summary = summarize_code_sparsity(_kwinner_codes(1024).reshape(8, 128, CODE_UNITS))
    assert summary["fraction_active"] == ACTIVE_UNITS_PER_STEP / CODE_UNITS
    assert summary["active_units_mean"] == float(ACTIVE_UNITS_PER_STEP)
    assert summary["active_units_std"] == 0.0


def test_summarize_code_sparsity_peak_is_a_chunk_not_the_array(monkeypatch) -> None:
    """The four scalars must not cost a full-size |codes| copy plus its boolean mask."""
    monkeypatch.setattr(evaluation_metrics, "_SPARSITY_CHUNK_ELEMENTS", 1 << 16)
    codes = _dense_backbone(1024)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        summarize_code_sparsity(codes)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak_bytes < codes.nbytes // 8


def test_summarize_code_sparsity_returns_nan_on_empty_input() -> None:
    summary = summarize_code_sparsity(np.zeros((0, CODE_UNITS), dtype=np.float32))
    assert all(np.isnan(value) for value in summary.values())


class _MaterializationTrap:
    """Stands in for a representation array and fails if the transfer path copies it."""

    shape = (4, 32, BACKBONE_UNITS)

    def reshape(self, *args: object) -> np.ndarray:
        raise AssertionError("transfer-decoder path materialized a copy for a discarded result")


def test_transfer_decoder_metrics_skips_the_copy_when_nothing_consumes_it() -> None:
    positions = np.zeros((4, 32, 2), dtype=np.float32)
    trap = {"encoder.backbone_output": _MaterializationTrap()}
    assert (
        _transfer_decoder_metrics(
            trap,
            positions,
            None,
            transfer_decoders={},
            is_train_split=False,
            ridge_alpha=1e-3,
        )
        == {}
    )
    assert (
        _transfer_decoder_metrics(
            trap,
            positions,
            None,
            transfer_decoders={"encoder.place_codes": object()},
            is_train_split=False,
            ridge_alpha=1e-3,
        )
        == {}
    )


def _linear_split(num_episodes: int, steps: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    codes = _kwinner_codes(num_episodes * steps, seed=seed)
    projection = np.random.default_rng(99).standard_normal((CODE_UNITS, 2)).astype(np.float32)
    positions = codes @ projection + generator.normal(0.0, 0.01, size=(len(codes), 2))
    return (
        codes.reshape(num_episodes, steps, CODE_UNITS),
        positions.reshape(num_episodes, steps, 2).astype(np.float32),
    )


def test_transfer_decoder_metrics_fits_on_train_and_scores_held_out() -> None:
    train_codes, train_positions = _linear_split(8, 32, seed=3)
    validation_codes, validation_positions = _linear_split(8, 32, seed=4)
    transfer_decoders: dict[str, object] = {}

    fitted = _transfer_decoder_metrics(
        {"encoder.place_codes": train_codes},
        train_positions,
        None,
        transfer_decoders=transfer_decoders,
        is_train_split=True,
        ridge_alpha=1e-3,
    )
    assert fitted == {}
    expected_decoder = fit_ridge_position_decoder(
        train_codes.reshape(-1, CODE_UNITS),
        train_positions.reshape(-1, 2),
        alpha=1e-3,
    )
    np.testing.assert_array_equal(
        transfer_decoders["encoder.place_codes"].weights, expected_decoder.weights
    )

    scored = _transfer_decoder_metrics(
        {"encoder.place_codes": validation_codes},
        validation_positions,
        None,
        transfer_decoders=transfer_decoders,
        is_train_split=False,
        ridge_alpha=1e-3,
    )
    expected_rmse, expected_r2 = score_ridge_position_decoder(
        expected_decoder,
        validation_codes.reshape(-1, CODE_UNITS),
        validation_positions.reshape(-1, 2),
    )
    assert scored == {
        "encoder.place_codes.decode_transfer_rmse": expected_rmse,
        "encoder.place_codes.decode_transfer_r2": expected_r2,
    }


def test_transfer_decoder_metrics_masking_is_unchanged() -> None:
    train_codes, train_positions = _linear_split(8, 32, seed=5)
    validation_codes, validation_positions = _linear_split(8, 32, seed=6)
    all_valid = np.ones(validation_positions.shape[:2], dtype=bool)
    partly_valid = all_valid.copy()
    partly_valid[:, ::3] = False

    def _scored(valid_steps: np.ndarray | None) -> dict[str, float]:
        transfer_decoders: dict[str, object] = {}
        _transfer_decoder_metrics(
            {"encoder.place_codes": train_codes},
            train_positions,
            None,
            transfer_decoders=transfer_decoders,
            is_train_split=True,
            ridge_alpha=1e-3,
        )
        return _transfer_decoder_metrics(
            {"encoder.place_codes": validation_codes},
            validation_positions,
            valid_steps,
            transfer_decoders=transfer_decoders,
            is_train_split=False,
            ridge_alpha=1e-3,
        )

    assert _scored(all_valid) == _scored(None)

    flat_mask = partly_valid.reshape(-1)
    expected_rmse, expected_r2 = score_ridge_position_decoder(
        fit_ridge_position_decoder(
            train_codes.reshape(-1, CODE_UNITS),
            train_positions.reshape(-1, 2),
            alpha=1e-3,
        ),
        validation_codes.reshape(-1, CODE_UNITS)[flat_mask],
        validation_positions.reshape(-1, 2)[flat_mask],
    )
    assert _scored(partly_valid) == {
        "encoder.place_codes.decode_transfer_rmse": expected_rmse,
        "encoder.place_codes.decode_transfer_r2": expected_r2,
    }
