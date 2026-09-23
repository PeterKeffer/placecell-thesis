from __future__ import annotations

import numpy as np
import pytest

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.successor_return import (
    SuccessorReturnComparisonModule,
    discounted_feature_returns,
    successor_return_metrics,
)


def test_discounted_feature_returns_stop_at_episode_padding() -> None:
    features = np.array(
        [
            [[1.0], [2.0], [3.0], [99.0]],
            [[4.0], [5.0], [99.0], [99.0]],
        ]
    )
    valid = np.array(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )

    returns = discounted_feature_returns(
        features,
        valid,
        discount_gamma=0.5,
        normalized=True,
    )

    expected = np.array(
        [
            [[1.375], [1.75], [1.5], [0.0]],
            [[3.25], [2.5], [0.0], [0.0]],
        ]
    )
    np.testing.assert_allclose(returns, expected)


def test_successor_return_metrics_recognize_an_exact_successor_feature() -> None:
    features = np.array([[[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 3.0]]])
    valid = np.ones((1, 4), dtype=bool)
    successor = discounted_feature_returns(
        features,
        valid,
        discount_gamma=0.8,
        normalized=True,
    )

    metrics = successor_return_metrics(
        successor,
        features,
        valid,
        discount_gamma=0.8,
        normalized=True,
        shuffle_seed=7,
    )

    assert metrics["mc_return_rmse"] == pytest.approx(0.0)
    assert metrics["mc_return_nrmse"] == pytest.approx(0.0)
    assert metrics["mc_return_r2"] == pytest.approx(1.0)
    assert metrics["mc_return_cosine"] == pytest.approx(1.0)
    assert metrics["bellman_residual_rmse"] == pytest.approx(0.0)
    assert metrics["current_feature_nrmse"] > metrics["mc_return_nrmse"]
    assert metrics["immediate_feature_nrmse"] > metrics["mc_return_nrmse"]
    assert metrics["mc_nrmse_improvement_over_immediate"] > 0.0


def test_successor_return_module_scores_aligned_analysis_inputs(tmp_path) -> None:
    features = np.array([[[1.0], [2.0], [4.0]]])
    valid = np.ones((1, 3), dtype=bool)
    successor = discounted_feature_returns(
        features,
        valid,
        discount_gamma=0.5,
        normalized=False,
    )
    positions = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])

    def analysis_input(representation: np.ndarray, label: str) -> AnalysisInput:
        return AnalysisInput(
            representation=representation,
            position_xy=positions,
            heading=None,
            kinematics=None,
            actions=None,
            valid_mask=valid,
            source_name=label,
            label=label,
            split_name="test",
            metadata={"dataset_artifact_id": "dataset", "split_artifact_id": "split"},
        )

    result = SuccessorReturnComparisonModule().run(
        [analysis_input(successor, "psi"), analysis_input(features, "phi")],
        ["psi", "phi"],
        tmp_path,
        {
            "successor_return_discount_gamma": 0.5,
            "successor_return_normalized": False,
            "successor_return_shuffle_seed": 3,
        },
    )

    assert result.metrics["mc_return_r2"] == pytest.approx(1.0)
    assert result.metadata["successor_return_normalized"] is False
    assert result.metadata["successor_label"] == "psi"
    assert result.metadata["feature_label"] == "phi"


def test_successor_return_module_is_registered_for_comparative_analysis() -> None:
    from placecell_research.analysis.registry import COMPARATIVE_MODULES

    assert COMPARATIVE_MODULES["successor_return"] is SuccessorReturnComparisonModule
