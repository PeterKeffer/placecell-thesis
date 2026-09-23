"""Sparsifier and expansion-cage fixes: EMA-safe duty cycle, norm placement, bias guards."""

from __future__ import annotations

import torch
from torch import nn

from placecell_research.config.schema import SpatialModelConfig, TemporalFamilyConfig
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.spatial_model.components.sparsifiers import (
    GroupedKWinnersSparsifier,
    KWinnersSparsifier,
)

CODE_DIM = 8
TRUNK_DIM = 16


def _context(observation_dim: int = CODE_DIM) -> ModelBuildContext:
    return ModelBuildContext(
        num_actions=4,
        observation_dim=observation_dim,
        kinematics_dim=2,
        total_optimizer_steps=10,
        optimizer_steps_per_epoch=4,
    )


def _batch(sequence_length: int = 5, observation_dim: int = CODE_DIM) -> dict[str, torch.Tensor]:
    return {
        "latent": torch.randn(2, sequence_length, observation_dim),
        "actions": torch.randint(0, 4, (2, sequence_length)),
        "kinematics": torch.randn(2, sequence_length, 2),
        "valid_steps": torch.ones(2, sequence_length, dtype=torch.bool),
    }


def test_boosted_kwinners_survives_an_ema_teacher_update_after_train_and_eval_passes() -> None:
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [CODE_DIM]
    config.predictor.layer_sizes = [CODE_DIM]
    config.training.code_dim = CODE_DIM
    config.objectives = {}
    config.teacher_student.mode = "ema_byol"
    config.teacher_student.predictor_ema = True
    config.teacher_student.predictor_ema_decay = 0.5
    config.predictor_sparsifier.type = "kwinners"
    config.predictor_sparsifier.kwinners_boost_strength = 1.0
    model = build_place_model(config, _context())

    teacher_sparsifier = model.teacher_controller.teacher_predictor_stack().predictor_sparsifier
    assert teacher_sparsifier.duty_cycle.shape == (CODE_DIM,)

    model.train()
    model.forward_sequence(_batch())
    model.eval()
    with torch.no_grad():
        model.forward_sequence(_batch())
    model.train()
    model.teacher_controller.update(current_step=1)

    assert teacher_sparsifier.duty_cycle.shape == model.predictor_sparsifier.duty_cycle.shape


def test_code_norm_is_sized_to_the_code_and_normalizes_the_projection_output() -> None:
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [TRUNK_DIM]
    config.predictor.layer_sizes = [TRUNK_DIM]
    config.training.code_dim = CODE_DIM
    config.objectives = {}
    config.encoder_pre_head_norm = "rmsnorm_fixed"
    config.encoder.head_activation = "none"
    config.encoder.normalize_codes = False
    model = build_place_model(config, _context())

    norm = model.encoder_stack.encoder_code_norm
    assert isinstance(norm, nn.RMSNorm)
    assert norm.normalized_shape == (CODE_DIM,)

    model.eval()
    with torch.no_grad():
        bundle = model.forward_sequence(_batch())
    logits = bundle.modules["encoder"].place_logits
    assert logits is not None
    root_mean_square = logits.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(root_mean_square, torch.ones_like(root_mean_square), atol=1e-5,
                               rtol=1e-5)
    hidden = bundle.modules["encoder"].hidden_state
    hidden_rms = hidden.pow(2).mean(dim=-1).sqrt()
    assert not torch.allclose(hidden_rms, torch.ones_like(hidden_rms), atol=1e-3)


def _load_sign_sparsifier(**overrides) -> KWinnersSparsifier:
    kwargs = dict(
        k_fraction=0.25,
        num_units=8,
        balance_bias_rate=0.1,
        balance_bias_strategy="load_sign",
    )
    kwargs.update(overrides)
    return KWinnersSparsifier(**kwargs)


def test_load_sign_bias_is_clamped_under_sustained_imbalance() -> None:
    sparsifier = _load_sign_sparsifier(balance_bias_clamp=0.25)
    sparsifier.train()
    values = torch.zeros(4, 8)
    values[:, 0] = 10.0
    values[:, 1] = 9.0
    for _ in range(50):
        sparsifier(values)
    assert sparsifier.balance_bias.abs().max() <= 0.25 + 1e-6
    assert sparsifier.balance_bias[2:].min() > 0.0


def test_load_sign_bias_leaks_toward_zero_without_load_error() -> None:
    sparsifier = _load_sign_sparsifier(balance_bias_leak=0.1)
    sparsifier.train()
    with torch.no_grad():
        sparsifier.balance_bias.copy_(
            torch.tensor([3.0, -3.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
        )
    magnitude_before = sparsifier.balance_bias.abs().sum().item()
    values = torch.full((4, 8), -1.0)
    for row in range(4):
        values[row, 2 * row] = 1.0
        values[row, 2 * row + 1] = 0.5
    for _ in range(20):
        sparsifier(values)
    assert sparsifier.balance_bias.abs().sum().item() < 0.5 * magnitude_before


def test_grouped_group_bias_guard_recruits_a_group_that_never_wins() -> None:
    sparsifier = GroupedKWinnersSparsifier(
        4 / 64,
        num_units=64,
        num_groups=16,
        group_bias_rate=0.05,
    )
    sparsifier.train()
    values = torch.randn(4, 64)
    values = values.reshape(4, 16, 4)
    values[:, 0] = -20.0
    values = values.reshape(4, 64)
    for _ in range(5):
        sparsifier(values)
    assert sparsifier.group_bias[0] > 0.0
    assert sparsifier.last_auxiliary_outputs["grouped_kwinners.group_bias_max_abs"] > 0.0


def test_grouped_group_bias_is_off_by_default() -> None:
    sparsifier = GroupedKWinnersSparsifier(4 / 64, num_units=64, num_groups=16)
    sparsifier.train()
    values = torch.randn(4, 64)
    for _ in range(5):
        sparsifier(values)
    assert torch.all(sparsifier.group_bias == 0.0)


def _predictor_diagnostic_config() -> SpatialModelConfig:
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [CODE_DIM]
    config.predictor.layer_sizes = [CODE_DIM]
    config.training.code_dim = CODE_DIM
    config.objectives = {}
    config.predictor_sparsifier.type = "kwinners"
    config.predictor_sparsifier.k_fraction = 0.25
    config.predictor_sparsifier.kwinners_balance_bias_rate = 0.01
    return config


def test_fused_predictor_publishes_its_sparsifier_diagnostics() -> None:
    model = build_place_model(_predictor_diagnostic_config(), _context())
    model.train()
    bundle = model.forward_sequence(_batch())
    assert "kwinners.balance_bias_mean_abs" in bundle.modules["predictor"].auxiliary


def test_stepwise_predictor_publishes_its_sparsifier_diagnostics() -> None:
    config = _predictor_diagnostic_config()
    config.predictor = TemporalFamilyConfig(
        family="clockwork",
        layer_sizes=[CODE_DIM],
        clockwork_periods=[1, 2],
    )
    config.inputs.predictor_input_mode = "encoder"
    model = build_place_model(config, _context())
    model.train()
    bundle = model.forward_sequence(_batch())
    assert "kwinners.balance_bias_mean_abs" in bundle.modules["predictor"].auxiliary


def test_balance_bias_does_not_steer_selection_in_eval() -> None:
    sparsifier = _load_sign_sparsifier(balance_bias_rate=0.1)
    values = torch.tensor([[5.0, 4.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]])
    with torch.no_grad():
        sparsifier.balance_bias.copy_(
            torch.tensor([-50.0, -50.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0])
        )

    sparsifier.eval()
    evaluation_output = sparsifier(values)
    assert set(evaluation_output.nonzero()[:, -1].tolist()) == {0, 1}

    sparsifier.train()
    training_output = sparsifier(values)
    assert set(training_output.nonzero()[:, -1].tolist()) == {6, 7}


def test_balance_bias_does_not_change_in_eval() -> None:
    sparsifier = _load_sign_sparsifier()
    sparsifier.eval()
    values = torch.tensor([[5.0, 4.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]])
    for _ in range(10):
        sparsifier(values)
    assert torch.all(sparsifier.balance_bias == 0.0)
