from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from placecell_research.config.schema import SparsifierConfig, SpatialModelConfig
from placecell_research.objectives.registry import compute_total_loss
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.spatial_model.components.sparsifiers import KWinnersSparsifier
from placecell_research.stages.train_place_model import (
    _compute_noise_floor_region_agreement,
    _compute_region_winner_metrics,
)

NUM_UNITS = 8


def _context() -> ModelBuildContext:
    return ModelBuildContext(
        num_actions=4,
        observation_dim=NUM_UNITS,
        kinematics_dim=2,
        total_optimizer_steps=20,
        optimizer_steps_per_epoch=4,
    )


def _config(*, balance_rate: float = 0.01, noise_scale: float = 0.0) -> SpatialModelConfig:
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [NUM_UNITS]
    config.predictor.layer_sizes = [NUM_UNITS]
    config.training.code_dim = NUM_UNITS
    config.objectives = {}
    config.sparsifier.type = "kwinners"
    config.sparsifier.k_fraction = 0.25
    config.sparsifier.kwinners_balance_bias_rate = balance_rate
    config.sparsifier.kwinners_selection_noise_scale = noise_scale
    return config


def _batch(batch_size: int = 2, sequence_length: int = 4) -> dict[str, torch.Tensor]:
    return {
        "latent": torch.randn(batch_size, sequence_length, NUM_UNITS),
        "actions": torch.randint(0, 4, (batch_size, sequence_length)),
        "kinematics": torch.randn(batch_size, sequence_length, 2),
        "valid_steps": torch.ones(batch_size, sequence_length, dtype=torch.bool),
    }


def _selected_units(output: torch.Tensor) -> set[int]:
    return set(output.ne(0).nonzero(as_tuple=False)[:, -1].tolist())


def test_zero_default_preserves_topk_output_state_and_rng() -> None:
    sparsifier = KWinnersSparsifier(k_fraction=0.25, num_units=NUM_UNITS)
    values = torch.randn(3, NUM_UNITS)
    cpu_rng_before = torch.random.get_rng_state().clone()
    mutable_before = {
        name: getattr(sparsifier, name).clone()
        for name in sparsifier.forward_mutable_buffer_names
    }

    output = sparsifier(values)

    cpu_rng_after = torch.random.get_rng_state()
    indices = torch.topk(values, 2, dim=-1).indices
    mask = torch.zeros_like(values, dtype=torch.bool).scatter_(-1, indices, True)
    expected = torch.where(mask, values, torch.zeros_like(values))
    torch.testing.assert_close(output, expected)
    assert torch.equal(output[mask], values[mask])
    assert torch.equal(cpu_rng_after, cpu_rng_before)
    for name, before in mutable_before.items():
        assert torch.equal(getattr(sparsifier, name), before)
    assert sparsifier.last_auxiliary_outputs == {}

    if torch.cuda.is_available():
        cuda_sparsifier = KWinnersSparsifier(
            k_fraction=0.25,
            num_units=NUM_UNITS,
        ).cuda()
        cuda_values = values.cuda()
        cuda_rng_before = torch.cuda.get_rng_state().clone()
        cuda_sparsifier(cuda_values)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)


def test_zero_spread_bias_and_noise_escape_fixed_ties() -> None:
    values = torch.ones(1, NUM_UNITS)
    balanced = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        balance_bias_rate=1.0,
        balance_liveness_patience_steps=1,
    )
    first_winners = _selected_units(balanced(values))
    second_winners = _selected_units(balanced(values))
    assert second_winners != first_winners

    torch.manual_seed(7)
    noisy = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        selection_noise_scale=1.0,
    )
    masks = {frozenset(_selected_units(noisy(values))) for _ in range(20)}
    assert len(masks) > 1


def test_known_normalized_margin_recruits_at_derived_update() -> None:
    values = torch.tensor(
        [
            [3.0, 0.0, 2.9, -2.0],
            [0.0, 3.0, 2.9, -2.0],
        ]
    )
    normalized = (values - values.mean(dim=-1, keepdim=True)) / values.std(
        dim=-1,
        keepdim=True,
    )
    normalized_margin = float(normalized[0, 0] - normalized[0, 2])
    gamma = normalized_margin / 2.5
    expected_updates = math.ceil(normalized_margin / gamma)
    assert expected_updates == 3
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=4,
        balance_bias_rate=gamma,
        balance_liveness_rate_ratio=1.0,
        balance_liveness_patience_steps=1,
    )

    for _ in range(expected_updates):
        assert 2 not in _selected_units(sparsifier(values))
    assert 2 in _selected_units(sparsifier(values))

    rate_zero = KWinnersSparsifier(k_fraction=0.25, num_units=4)
    assert all(2 not in _selected_units(rate_zero(values)) for _ in range(5))


def test_age_liveness_preserves_recent_winners_and_rescues_only_stale_units() -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        balance_bias_rate=0.1,
        balance_liveness_rate_ratio=0.1,
        balance_liveness_patience_steps=2,
    )
    first_mask = torch.zeros(4, NUM_UNITS, dtype=torch.bool)
    first_mask[:, 0] = True
    first_mask[:, 2] = True
    sparsifier._update_balance_bias(first_mask, k_active=2)

    assert sparsifier.balance_bias[0] == pytest.approx(-0.1)
    assert sparsifier.balance_bias[1] == pytest.approx(0.0)
    assert sparsifier.balance_bias[2] == pytest.approx(-0.1)
    assert int(sparsifier._steps_since_win[1]) == 1
    assert int(sparsifier._steps_since_win[2]) == 0

    second_mask = torch.zeros(4, NUM_UNITS, dtype=torch.bool)
    second_mask[:, 0] = True
    second_mask[0, 3] = True
    second_mask[1, 4] = True
    second_mask[2, 5] = True
    second_mask[3, 6] = True
    sparsifier._update_balance_bias(second_mask, k_active=2)

    assert sparsifier.balance_bias[0] == pytest.approx(-0.2)
    assert sparsifier.balance_bias[1] == pytest.approx(0.01)
    assert sparsifier.balance_bias[2] == pytest.approx(-0.1)
    assert int(sparsifier._steps_since_win[1]) == 2
    assert int(sparsifier._steps_since_win[2]) == 1
    assert bool(sparsifier._ever_rescued[1])
    assert not bool(sparsifier._ever_rescued[2])


def test_load_sign_bias_uses_actual_routed_load_and_centers_free_offset() -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        balance_bias_rate=0.002,
        balance_bias_strategy="load_sign",
    )
    mask = torch.zeros(4, NUM_UNITS, dtype=torch.bool)
    mask[:, 0] = True
    mask[0, 1] = True
    mask[1:, 2] = True

    sparsifier._update_balance_bias(mask, k_active=2)

    win_frequency = mask.float().mean(dim=0)
    raw_delta = 0.002 * torch.sign(torch.tensor(0.25) - win_frequency)
    expected = raw_delta - raw_delta.mean()
    torch.testing.assert_close(sparsifier.balance_bias, expected)
    assert float(sparsifier.balance_bias.mean()) == pytest.approx(0.0, abs=1e-9)
    assert sparsifier.balance_bias[0] < sparsifier.balance_bias[1]
    assert sparsifier.balance_bias[3] > sparsifier.balance_bias[1]


def test_load_sign_bias_rejects_zero_rate() -> None:
    with pytest.raises(ValueError, match="positive bias rate"):
        KWinnersSparsifier(
            k_fraction=0.25,
            num_units=NUM_UNITS,
            balance_bias_strategy="load_sign",
        )


def test_age_liveness_tracks_noise_free_winners(monkeypatch: pytest.MonkeyPatch) -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=4,
        balance_bias_rate=0.1,
        balance_liveness_rate_ratio=0.1,
        balance_liveness_patience_steps=1,
        selection_noise_scale=1.0,
    )
    values = torch.tensor([[3.0, 2.0, 1.0, 0.5]]).repeat(4, 1)
    noise = torch.full_like(values, -10.0)
    noise[torch.arange(4), torch.arange(4)] = 10.0
    monkeypatch.setattr(torch, "randn_like", lambda _values: noise)

    output = sparsifier(values)

    assert _selected_units(output) == {0, 1, 2, 3}
    assert sparsifier._steps_since_win.tolist() == [0, 1, 1, 1]
    assert sparsifier.balance_bias.tolist() == pytest.approx([-0.1, 0.01, 0.01, 0.01])
    assert sparsifier.last_auxiliary_outputs[
        "kwinners.deterministic_recruited_unit_fraction"
    ] == pytest.approx(0.25)
    assert sparsifier.last_auxiliary_outputs[
        "kwinners.noisy_deterministic_winner_agreement"
    ] == pytest.approx(0.25)


def test_noise_anneals_to_final_scale_without_consuming_final_rng() -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        selection_noise_scale=1.0,
        selection_noise_anneal_steps=2,
        selection_noise_final_scale=0.0,
    )
    scales = [sparsifier._step_selection_noise_scale() for _ in range(4)]
    assert scales == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert int(sparsifier._noise_step) == 2

    rng_before = torch.random.get_rng_state().clone()
    first = sparsifier(torch.ones(1, NUM_UNITS))
    second = sparsifier(torch.ones(1, NUM_UNITS))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(first, second)


def test_noise_floor_remains_active_after_annealing() -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        selection_noise_scale=1.0,
        selection_noise_anneal_steps=2,
        selection_noise_final_scale=0.1,
    )

    scales = [sparsifier._step_selection_noise_scale() for _ in range(4)]

    assert scales == pytest.approx([1.0, 0.55, 0.1, 0.1])
    rng_before = torch.random.get_rng_state().clone()
    sparsifier(torch.ones(1, NUM_UNITS))
    assert not torch.equal(torch.random.get_rng_state(), rng_before)


def test_evaluation_ignores_the_bias_and_does_not_update_it() -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        balance_bias_rate=0.1,
        balance_liveness_patience_steps=2,
        selection_noise_scale=2.0,
    )
    with torch.no_grad():
        sparsifier.balance_bias[2] = 5.0
    sparsifier.eval()
    values = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
    bias_before = sparsifier.balance_bias.clone()

    first = sparsifier(values)
    second = sparsifier(values)

    assert torch.equal(first, second)
    assert _selected_units(first) == {0, 1}
    assert torch.equal(sparsifier.balance_bias, bias_before)
    assert torch.count_nonzero(sparsifier._steps_since_win) == 0
    assert int(sparsifier._noise_step) == 0


def test_encoder_exports_controller_diagnostics_into_loss_metrics() -> None:
    model = build_place_model(_config(), _context())
    batch = _batch()

    bundle = model.forward_sequence(batch)
    _, metrics = compute_total_loss([], bundle, batch, model.components.config)

    auxiliary = bundle.modules["encoder"].auxiliary
    assert "kwinners.balance_bias_mean_abs" in auxiliary
    assert "kwinners.normalized_boundary_margin_mean" in auxiliary
    assert "kwinners.balance_bias_std" in auxiliary
    assert "kwinners.balance_bias_to_boundary_margin" in auxiliary
    assert "kwinners.mean_absolute_load_error" in auxiliary
    assert "kwinners.max_load_violation" in auxiliary
    assert "kwinners.liveness_ever_rescued_fraction" in auxiliary
    assert "encoder/kwinners.balance_bias_mean_abs" in metrics


def test_ema_teacher_copies_balance_bias_and_does_not_update_it_on_forward() -> None:
    model = build_place_model(_config(), _context())
    controller = model.teacher_controller
    teacher_stack = controller.teacher_encoder_stack
    assert teacher_stack is not None
    online_sparsifier = model.encoder_stack.sparsifier
    teacher_sparsifier = teacher_stack.sparsifier
    with torch.no_grad():
        online_sparsifier.balance_bias.copy_(torch.arange(NUM_UNITS))
        online_sparsifier._steps_since_win.copy_(torch.arange(NUM_UNITS))

    controller.update(current_step=1)

    torch.testing.assert_close(teacher_sparsifier.balance_bias, online_sparsifier.balance_bias)
    assert torch.count_nonzero(teacher_sparsifier._steps_since_win) == 0
    teacher_bias_before = teacher_sparsifier.balance_bias.clone()
    teacher_sparsifier(torch.randn(2, NUM_UNITS))
    torch.testing.assert_close(teacher_sparsifier.balance_bias, teacher_bias_before)


def test_checkpoint_roundtrip_preserves_balance_state() -> None:
    checkpoint_units = 512
    config = _config()
    config.encoder.layer_sizes = [checkpoint_units]
    config.predictor.layer_sizes = [checkpoint_units]
    config.training.code_dim = checkpoint_units
    context = ModelBuildContext(
        num_actions=4,
        observation_dim=NUM_UNITS,
        kinematics_dim=2,
        total_optimizer_steps=20,
    )
    model = build_place_model(config, context)
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("balance_bias"):
                buffer.copy_(torch.arange(buffer.numel(), dtype=buffer.dtype))
            elif name.endswith("_steps_since_win"):
                buffer.copy_(torch.arange(buffer.numel(), dtype=buffer.dtype))
            elif name.endswith("_ever_rescued"):
                buffer[::2] = True
            elif name.endswith("_noise_step"):
                buffer.fill_(7)
    fresh_model = build_place_model(config, context)
    fresh_model.load_state_dict(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(fresh_model.state_dict()[name], value)


def test_schema_and_builder_validate_new_kwinners_knobs() -> None:
    invalid_negative_values = {
        "kwinners_balance_bias_rate": -0.01,
        "kwinners_selection_noise_scale": -0.01,
        "kwinners_selection_noise_anneal_steps": -1,
        "kwinners_selection_noise_final_scale": -0.01,
    }
    for field_name, invalid_value in invalid_negative_values.items():
        with pytest.raises(ValidationError, match=field_name):
            SparsifierConfig(type="kwinners", **{field_name: invalid_value})

    with pytest.raises(ValidationError, match="final_scale"):
        SparsifierConfig(
            type="kwinners",
            kwinners_selection_noise_scale=0.5,
            kwinners_selection_noise_anneal_steps=10,
            kwinners_selection_noise_final_scale=1.0,
        )
    for field_name, invalid_value in {
        "kwinners_balance_liveness_rate_ratio": 0.0,
        "kwinners_balance_liveness_patience_epochs": 0.5,
    }.items():
        with pytest.raises(ValidationError, match=field_name):
            SparsifierConfig(type="kwinners", **{field_name: invalid_value})
    with pytest.raises(ValidationError, match="cannot both"):
        SparsifierConfig(
            type="kwinners",
            kwinners_boost_strength=1.0,
            kwinners_balance_bias_rate=0.01,
        )
    with pytest.raises(ValidationError, match="requires a positive"):
        SparsifierConfig(
            type="kwinners",
            kwinners_balance_bias_strategy="load_sign",
        )
    nondefault_values = {
        "kwinners_balance_bias_rate": 0.01,
        "kwinners_balance_bias_strategy": "load_sign",
        "kwinners_balance_liveness_rate_ratio": 0.2,
        "kwinners_balance_liveness_patience_epochs": 3.0,
        "kwinners_selection_noise_scale": 1.0,
        "kwinners_selection_noise_anneal_steps": 10,
        "kwinners_selection_noise_final_scale": 0.1,
    }
    for field_name, nondefault_value in nondefault_values.items():
        with pytest.raises(ValidationError, match="only used by the 'kwinners'"):
            SparsifierConfig(type="sparsemax", **{field_name: nondefault_value})

    model = build_place_model(_config(), _context())
    sparsifier = model.encoder_stack.sparsifier
    assert sparsifier.balance_bias.shape == (NUM_UNITS,)
    assert sparsifier.balance_liveness_patience_steps == 8
    assert sparsifier.balance_bias_rate * sparsifier.balance_liveness_rate_ratio == pytest.approx(
        0.001
    )


def test_region_metrics_distinguish_dead_specialist_and_monopolist_units() -> None:
    codes = np.zeros((1, 12, 4), dtype=np.float32)
    codes[0, :10, 0] = 1.0
    codes[0, 10, 1] = 1.0
    codes[0, 11, 3] = 1.0
    positions = np.zeros((1, 12, 2), dtype=np.float32)
    positions[0, 10, 0] = 1.0
    positions[0, 11, 0] = 2.0
    valid = np.ones((1, 12), dtype=bool)

    metrics = _compute_region_winner_metrics(
        codes,
        positions,
        valid,
        num_bins_x=3,
        num_bins_y=1,
    )

    assert metrics["validation.routing.never_selected_fraction"] == pytest.approx(0.25)
    assert metrics["validation.routing.rare_specialist_fraction"] == pytest.approx(0.5)
    assert metrics["validation.routing.monopolist_fraction"] == pytest.approx(0.25)
    assert metrics["validation.routing.region_active_unit_fraction_mean"] == pytest.approx(0.25)


def test_noise_floor_agreement_reports_region_conditioned_churn() -> None:
    dense = np.zeros((1, 12, 4), dtype=np.float32)
    dense[0, :6, 0] = 3.0
    positions = np.zeros((1, 12, 2), dtype=np.float32)
    positions[0, 6:, 0] = 1.0
    valid = np.ones((1, 12), dtype=bool)

    metrics = _compute_noise_floor_region_agreement(
        dense,
        positions,
        valid,
        balance_bias=np.zeros(4, dtype=np.float32),
        k_fraction=0.25,
        noise_scale=0.1,
        num_bins_x=2,
        num_bins_y=1,
        random_seed=3,
    )

    assert 0.0 <= metrics["validation.routing.noise_floor_winner_agreement"] <= 1.0
    assert (
        metrics["validation.routing.noise_floor_region_agreement_min"]
        < metrics["validation.routing.noise_floor_region_agreement_mean"]
    )
