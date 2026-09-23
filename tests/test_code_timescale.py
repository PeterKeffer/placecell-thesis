import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.code_timescale import (
    CodeTimescaleModule,
    autocorrelation_by_lag,
    decorrelation_time,
)


def _ar1(a: float, episodes: int = 2, time: int = 400, units: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((episodes, time, units)).astype(np.float32)
    for t in range(1, time):
        z[:, t] = a * z[:, t - 1] + np.sqrt(1 - a**2) * z[:, t]
    return z


def _input(codes: np.ndarray) -> AnalysisInput:
    episodes, time, _ = codes.shape
    return AnalysisInput(
        representation=codes,
        position_xy=np.zeros((episodes, time, 2), dtype=np.float32),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, time), dtype=bool),
        source_name="test",
        label="test",
        split_name="test",
    )


def test_lag_1_autocorrelation_recovers_the_ar1_coefficient():
    curve = autocorrelation_by_lag(_ar1(0.8), np.ones((2, 400), dtype=bool), (1, 2))
    assert abs(curve[1] - 0.8) < 0.05


def test_autocorrelation_is_invariant_to_invalid_right_padding():
    codes = np.asarray(
        [[[1.0, -1.0], [2.0, 0.5], [-1.0, 2.0], [0.5, -0.5]]],
        dtype=np.float32,
    )
    valid = np.ones((1, 4), dtype=bool)
    reference = autocorrelation_by_lag(codes, valid, (1, 2))

    padded_codes = np.concatenate(
        [codes, np.full((1, 7, 2), 10_000.0, dtype=np.float32)],
        axis=1,
    )
    padded_valid = np.concatenate([valid, np.zeros((1, 7), dtype=bool)], axis=1)

    assert autocorrelation_by_lag(padded_codes, padded_valid, (1, 2)) == reference


def test_fast_code_has_short_decorrelation_time_and_slow_code_long():
    valid = np.ones((2, 400), dtype=bool)
    fast = decorrelation_time(autocorrelation_by_lag(_ar1(0.0), valid, (1, 2, 5, 10, 20, 50)))
    slow = decorrelation_time(autocorrelation_by_lag(_ar1(0.95), valid, (1, 2, 5, 10, 20, 50)))
    assert fast < slow
    assert fast <= 1.0


def test_module_reports_lag_1_and_t_dec():
    result = CodeTimescaleModule().run(_input(_ar1(0.9)), None, {})
    assert "code_autocorrelation_lag_1" in result.metrics
    assert "code_t_dec" in result.metrics
    assert abs(result.metrics["code_autocorrelation_lag_1"] - 0.9) < 0.05


def test_module_skips_cleanly_on_a_constant_code():
    constant = np.ones((2, 50, 4), dtype=np.float32)
    result = CodeTimescaleModule().run(_input(constant), None, {})
    assert result.metrics == {}
    assert result.metadata["code_timescale_skipped"] is True
