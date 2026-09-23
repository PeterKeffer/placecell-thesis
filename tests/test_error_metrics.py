import numpy as np
import pytest

from placecell_research.evaluation.matched_decode import MatchedPositionDecoder
from placecell_research.numerics.error_metrics import (
    rmse_from_coordinate_mse,
    root_mean_squared_error,
)


def test_coordinate_errors_are_pooled_before_square_root():
    targets = np.zeros((3, 2))
    predictions = np.tile([1.0, 3.0], (3, 1))
    assert root_mean_squared_error(targets, predictions) == pytest.approx(np.sqrt(5))
    assert rmse_from_coordinate_mse(np.array([1.0, 9.0])) == pytest.approx(np.sqrt(5))


def test_matched_decoder_selects_by_pooled_error():
    decoder = MatchedPositionDecoder(
        mean=np.zeros(1), target_mean=np.zeros(2),
        weights=np.array([[[0.0, 3.0]], [[1.6, 1.6]]]), alphas=(1.0, 10.0),
    )
    codes = np.ones((7, 1))
    targets = np.zeros((7, 2))
    score = decoder.score(codes, targets, select=True)
    assert score["matched_decode_alpha"] == 10.0
    assert score["matched_decode_rmse"] == pytest.approx(1.6)
    confirmed = decoder.score(codes, targets, select=False)
    assert confirmed["matched_decode_rmse"] == score["matched_decode_rmse"]
    assert confirmed["matched_decode_alpha"] == score["matched_decode_alpha"]


def test_rmse_weights_samples_across_unequal_chunks():
    decoder = MatchedPositionDecoder(
        mean=np.zeros(1), target_mean=np.zeros(2),
        weights=np.zeros((1, 1, 2)), alphas=(1.0,), selected=0,
    )
    targets = np.zeros((65537, 2))
    targets[-1] = [1.0, 3.0]
    score = decoder.score(np.zeros((65537, 1)), targets, select=False)
    assert score["matched_decode_rmse"] == pytest.approx(np.sqrt(5 / 65537))
