"""Component-mode gradient clipping: the confound it removes, and the invariants it must keep."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from placecell_research.config.schema import SpatialModelConfig, SpatialTrainingConfig
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.training.optimizer import clip_gradients, clip_gradients_by_component

CODE_DIM = 8
CLIP = 1.0


def _build_context() -> ModelBuildContext:
    return ModelBuildContext(
        num_actions=4, observation_dim=12, kinematics_dim=2, total_optimizer_steps=8
    )


def _model():
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [CODE_DIM]
    config.predictor.layer_sizes = [CODE_DIM]
    config.training.code_dim = CODE_DIM
    config.objectives = {}
    config.teacher_student.mode = "ema_byol"
    return build_place_model(config, _build_context())


def _fill_gradients(parameters: list[nn.Parameter], value: float) -> None:
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, value)


def _flat_gradient(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])


def test_default_mode_is_global_so_no_existing_run_changes() -> None:
    assert SpatialTrainingConfig().gradient_clip_mode == "global"


def test_the_causal_groups_partition_every_trainable_parameter_exactly_once() -> None:
    model = _model()
    groups = model.clip_parameter_groups()
    encoder_ids = [id(p) for p in groups["encoder"]]
    predictor_ids = [id(p) for p in groups["predictor"]]

    assert len(encoder_ids) == len(set(encoder_ids)), "duplicate inside the encoder group"
    assert len(predictor_ids) == len(set(predictor_ids)), "duplicate inside the predictor group"
    assert not set(encoder_ids) & set(predictor_ids), "a parameter is in BOTH causal groups"
    assert encoder_ids and predictor_ids


def test_the_predictor_group_is_exactly_what_the_freeze_removes() -> None:
    model = _model()
    predictor_ids = {id(p) for p in model.clip_parameter_groups()["predictor"]}

    model.set_trainable({"encoder"})
    frozen_ids = {id(p) for p in model.parameters() if not p.requires_grad}
    assert predictor_ids <= frozen_ids
    for module_name in ("predictor_temporal", "predictor_head", "predictor_input_assembler"):
        module = getattr(model, module_name, None)
        if isinstance(module, nn.Module):
            assert {id(p) for p in module.parameters()} <= predictor_ids, module_name


def test_component_mode_gives_both_arms_the_same_encoder_step() -> None:
    """THE test."""
    control, treatment = _model(), _model()
    for model in (control, treatment):
        groups = model.clip_parameter_groups()
        _fill_gradients(groups["encoder"], 0.5)
    _fill_gradients(control.clip_parameter_groups()["predictor"], 0.5)

    control_encoder = control.clip_parameter_groups()["encoder"]
    treatment_encoder = treatment.clip_parameter_groups()["encoder"]
    assert torch.allclose(_flat_gradient(control_encoder), _flat_gradient(treatment_encoder)), (
        "the two arms must start from an identical pre-clip encoder gradient"
    )
    pre_clip_norm = float(_flat_gradient(control_encoder).norm())
    assert pre_clip_norm > CLIP, "test is vacuous unless the clip actually fires"

    clip_gradients_by_component(
        [p for p in control.parameters() if p.requires_grad],
        [],
        {k: [p for p in v if p.requires_grad] for k, v in control.clip_parameter_groups().items()},
        CLIP,
    )
    clip_gradients_by_component(
        [p for p in treatment.parameters() if p.requires_grad],
        [],
        {
            "encoder": treatment.clip_parameter_groups()["encoder"],
            "predictor": [],
        },
        CLIP,
    )

    assert torch.allclose(
        _flat_gradient(control_encoder), _flat_gradient(treatment_encoder), atol=1e-6
    ), "component mode still gave the two arms different encoder steps"


def test_global_mode_is_what_makes_the_two_arms_diverge() -> None:
    """The counterpart: the confound is real, not hypothetical."""
    control, treatment = _model(), _model()
    for model in (control, treatment):
        _fill_gradients(model.clip_parameter_groups()["encoder"], 0.5)
    _fill_gradients(control.clip_parameter_groups()["predictor"], 0.5)

    control_encoder = control.clip_parameter_groups()["encoder"]
    treatment_encoder = treatment.clip_parameter_groups()["encoder"]

    clip_gradients([p for p in control.parameters() if p.grad is not None], [], CLIP)
    clip_gradients([p for p in treatment.parameters() if p.grad is not None], [], CLIP)

    control_norm = float(_flat_gradient(control_encoder).norm())
    treatment_norm = float(_flat_gradient(treatment_encoder).norm())
    assert treatment_norm > control_norm, (
        "global clipping did not advantage the frozen arm -- if this ever fails, re-derive the "
        "confound before trusting a frozen-vs-control result under global mode"
    )


def test_an_empty_predictor_group_is_handled_without_an_invalid_norm() -> None:
    """The frozen arm's predictor group is empty every step; that is normal, not an error."""
    model = _model()
    _fill_gradients(model.clip_parameter_groups()["encoder"], 0.1)
    norm, diagnostics = clip_gradients_by_component(
        [p for p in model.parameters() if p.grad is not None],
        [],
        {"encoder": model.clip_parameter_groups()["encoder"], "predictor": []},
        CLIP,
    )
    assert torch.isfinite(norm)
    assert diagnostics["predictor_grad_norm_pre_clip"] == 0.0
    assert diagnostics["predictor_clip_scale"] == 1.0, "an empty group must not report a clip"


def test_diagnostics_report_whether_the_clip_actually_fired() -> None:
    """clip_scale below 1.0 is the evidence that the confound would have been active."""
    model = _model()
    groups = model.clip_parameter_groups()
    _fill_gradients(groups["encoder"], 5.0)
    _fill_gradients(groups["predictor"], 1e-6)
    _, diagnostics = clip_gradients_by_component(
        [p for p in model.parameters() if p.grad is not None], [], groups, CLIP
    )
    assert diagnostics["encoder_grad_norm_pre_clip"] > CLIP
    assert diagnostics["encoder_grad_norm_post_clip"] == pytest.approx(CLIP, rel=1e-5)
    assert diagnostics["encoder_clip_scale"] < 1.0
    assert diagnostics["predictor_clip_scale"] == pytest.approx(1.0)


def test_parameters_outside_the_causal_groups_are_still_clipped() -> None:
    model = _model()
    stray = nn.Parameter(torch.zeros(4))
    stray.grad = torch.full_like(stray, 100.0)
    groups = model.clip_parameter_groups()
    _fill_gradients(groups["encoder"], 0.1)
    _, diagnostics = clip_gradients_by_component(
        [p for p in model.parameters() if p.grad is not None] + [stray], [], groups, CLIP
    )
    assert diagnostics["other_grad_norm_pre_clip"] > CLIP
    assert float(stray.grad.norm()) == pytest.approx(CLIP, rel=1e-5), "a stray parameter escaped"
