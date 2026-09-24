from __future__ import annotations

import pytest
import torch

from placecell_research.config.schema import ObjectiveConfig, SpatialModelConfig
from placecell_research.objectives.prediction import PredictionAlignmentObjective
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.spatial_model.types import ModuleOutputs, RepresentationBundle


def _loss(
    loss_type: str,
    prediction: list[float],
    target: list[float],
    *,
    support_rank_weight: float = 0.1,
) -> torch.Tensor:
    predictor = torch.tensor([[prediction, prediction]], requires_grad=True)
    teacher = torch.tensor([[target, target]])
    bundle = RepresentationBundle(
        modules={
            "predictor": ModuleOutputs(place_codes=predictor),
            "teacher": ModuleOutputs(place_codes=teacher),
        },
        masks={"valid_steps": torch.ones(1, 2, dtype=torch.bool)},
    )
    objective = PredictionAlignmentObjective(
        name="prediction",
        config=ObjectiveConfig(
            type="prediction_alignment",
            loss_type=loss_type,
            support_rank_weight=support_rank_weight,
        ),
    )
    return objective.compute(bundle, batch={}).loss


def test_balanced_smooth_l1_weights_active_and_inactive_groups_equally() -> None:
    loss = _loss("balanced_smooth_l1", [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0])

    assert loss.item() == pytest.approx(5.0 / 6.0)


def test_support_rank_smooth_l1_prefers_correct_support_with_equal_value_error() -> None:
    target = [1.0, 0.0, 0.0, 0.0]
    correct_support = _loss("support_rank_smooth_l1", [0.8, 0.2, 0.1, 0.0], target)
    wrong_support = _loss("support_rank_smooth_l1", [0.2, 0.8, 0.1, 0.0], target)

    assert correct_support < wrong_support


def test_support_rank_weight_controls_ranking_contribution() -> None:
    prediction = [0.8, 0.1, 0.0, 0.0]
    target = [1.0, 0.0, 0.0, 0.0]

    regression_only = _loss(
        "support_rank_smooth_l1",
        prediction,
        target,
        support_rank_weight=0.0,
    )
    with_ranking = _loss(
        "support_rank_smooth_l1",
        prediction,
        target,
        support_rank_weight=0.01,
    )

    assert with_ranking > regression_only


def test_support_rank_smooth_l1_handles_negative_active_values_without_conflict() -> None:
    target = [-1.0, 0.0, 0.0, 0.0]
    correct_sign = _loss("support_rank_smooth_l1", [-1.0, 0.0, 0.0, 0.0], target)
    wrong_sign = _loss("support_rank_smooth_l1", [1.0, 0.0, 0.0, 0.0], target)

    assert torch.isfinite(correct_sign)
    assert correct_sign < wrong_sign


def test_prediction_alignment_rejects_unknown_loss_type() -> None:
    with pytest.raises(ValueError, match="prediction_alignment.loss_type"):
        _loss("typo", [1.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize("offset,expected", [(0, 0.0), (1, 1.0)])
def test_target_offset_uses_same_predictor_and_valid_transitions(offset, expected):
    prediction = torch.tensor(
        [[[99.0, 99.0], [1.0, 0.0], [0.0, 1.0], [99.0, 99.0]]], requires_grad=True
    )
    teacher = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [99.0, 99.0]]])
    bundle = RepresentationBundle(
        modules={
            "predictor": ModuleOutputs(place_codes=prediction),
            "teacher": ModuleOutputs(place_codes=teacher),
        },
        masks={"valid_steps": torch.tensor([[True, True, True, False]])},
    )
    objective = PredictionAlignmentObjective(
        name="prediction", config=ObjectiveConfig(type="prediction_alignment", target_offset=offset)
    )
    loss = objective.compute(bundle, {}).loss
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[:, 0]) == 0
    assert torch.count_nonzero(prediction.grad[:, 3]) == 0
    if offset == 1:
        assert torch.count_nonzero(prediction.grad[:, 1:3]) > 0


def test_same_step_alignment_backpropagates_through_predictor():
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [8]
    config.predictor.layer_sizes = [8]
    config.training.code_dim = 8
    objective_config = ObjectiveConfig(type="prediction_alignment", target_offset=0)
    config.objectives = {"prediction": objective_config}
    model = build_place_model(
        config,
        ModelBuildContext(
            num_actions=3, observation_dim=6, kinematics_dim=2, total_optimizer_steps=4
        ),
    )
    batch = {
        "latent": torch.randn(2, 6, 6),
        "actions": torch.zeros(2, 6, dtype=torch.long),
        "kinematics": torch.zeros(2, 6, 2),
        "valid_steps": torch.ones(2, 6, dtype=torch.bool),
    }
    result = PredictionAlignmentObjective(name="prediction", config=objective_config).compute(
        model.forward_sequence(batch), batch
    )
    result.loss.backward()
    gradients = [parameter.grad for parameter in model.predictor_temporal.parameters()]
    assert gradients
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
