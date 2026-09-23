"""Binding-lr split (D1): per-group learning rates over a split-out encoder binding layer."""

from __future__ import annotations

import pytest
import torch

from placecell_research.config.schema import ObjectiveConfig, SpatialModelConfig
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.training.optimizer import build_optimizer


def _build_model(group_learning_rates: dict[str, float] | None = None):
    torch.manual_seed(0)
    config = SpatialModelConfig()
    config.inputs.predictor_input_mode = "encoder"
    config.predictor.family = "gru"
    config.predictor.dropout = 0.0
    config.training.code_dim = 8
    if group_learning_rates is not None:
        config.training.group_learning_rates = group_learning_rates
    config.objectives = {
        "prediction_cosine": ObjectiveConfig(type="prediction_alignment", weight=1.0),
    }
    model = build_place_model(
        config,
        ModelBuildContext(
            num_actions=4,
            observation_dim=8,
            kinematics_dim=2,
            total_optimizer_steps=10,
        ),
    )
    return model, config


def test_binding_group_holds_exactly_the_encoder_head() -> None:
    model, _config = _build_model()
    groups = model.parameters_by_group()
    head_ids = {id(p) for p in model.encoder_stack.encoder_head.parameters()}
    assert {id(p) for p in groups["encoder_binding"]} == head_ids
    assert head_ids.isdisjoint({id(p) for p in groups["encoder"]})


def test_group_learning_rate_override_applies_to_binding_and_its_no_decay_twin() -> None:
    model, config = _build_model({"encoder_binding": 3e-3})
    optimizer, _groups = build_optimizer(model, torch.nn.ModuleDict(), config.training)
    lr_by_name = {group["name"]: group["lr"] for group in optimizer.param_groups}
    for name, lr in lr_by_name.items():
        expected = 3e-3 if name.startswith("encoder_binding") else config.training.learning_rate
        assert lr == expected, (name, lr)


def test_unknown_group_name_raises() -> None:
    model, config = _build_model({"encoder_bnding": 1e-3})
    with pytest.raises(ValueError, match="encoder_bnding"):
        build_optimizer(model, torch.nn.ModuleDict(), config.training)


def test_encoder_selector_still_trains_the_binding_layer() -> None:
    model, _config = _build_model()
    model.set_trainable({"encoder"})
    assert all(p.requires_grad for p in model.encoder_stack.encoder_head.parameters())
    model.set_trainable({"predictor"})
    assert not any(p.requires_grad for p in model.encoder_stack.encoder_head.parameters())
