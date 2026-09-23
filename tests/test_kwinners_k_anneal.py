from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from placecell_research.config.schema import (
    SparsifierConfig,
    SpatialModelConfig,
    SpatialTrainingConfig,
)
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.spatial_model.components.sparsifiers import KWinnersSparsifier

NUM_UNITS = 8


def _config() -> SpatialModelConfig:
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [NUM_UNITS]
    config.predictor.layer_sizes = [NUM_UNITS]
    config.training.code_dim = NUM_UNITS
    config.objectives = {}
    config.sparsifier = SparsifierConfig(
        type="kwinners",
        k_fraction=0.25,
        kwinners_k_anneal_start=NUM_UNITS,
        kwinners_k_anneal_steps=4,
    )
    return config


def _context() -> ModelBuildContext:
    return ModelBuildContext(
        num_actions=4,
        observation_dim=NUM_UNITS,
        kinematics_dim=2,
        total_optimizer_steps=20,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "latent": torch.randn(2, 4, NUM_UNITS),
        "actions": torch.randint(0, 4, (2, 4)),
        "kinematics": torch.randn(2, 4, 2),
        "valid_steps": torch.ones(2, 4, dtype=torch.bool),
    }


def _active_count(output: torch.Tensor) -> int:
    return int(output.ne(0).sum(dim=-1).item())


def test_k_anneal_zero_defaults_preserve_output_state_and_rng() -> None:
    values = torch.randn(3, NUM_UNITS)
    expected = KWinnersSparsifier(k_fraction=0.25, num_units=NUM_UNITS)
    actual = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        k_anneal_start=0,
        k_anneal_steps=0,
    )
    rng_before = torch.random.get_rng_state().clone()

    torch.testing.assert_close(actual(values), expected(values))

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert int(actual._k_anneal_step) == 0
    assert actual.last_auxiliary_outputs == {}


def test_k_anneal_schedule_controls_mask_cardinality_and_stops_at_final_k() -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        k_anneal_start=NUM_UNITS,
        k_anneal_steps=3,
    )
    values = torch.arange(1, NUM_UNITS + 1, dtype=torch.float32).unsqueeze(0)

    active_counts = [_active_count(sparsifier(values)) for _ in range(5)]

    assert active_counts == [8, 6, 4, 2, 2]
    assert int(sparsifier._k_anneal_step) == 3
    assert float(sparsifier.last_auxiliary_outputs["kwinners.effective_k"]) == 2.0


def test_k_anneal_eval_reads_schedule_without_advancing_it() -> None:
    sparsifier = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        k_anneal_start=NUM_UNITS,
        k_anneal_steps=4,
    )
    values = torch.arange(1, NUM_UNITS + 1, dtype=torch.float32).unsqueeze(0)
    sparsifier(values)
    sparsifier(values)
    sparsifier.eval()

    first = sparsifier(values)
    second = sparsifier(values)

    assert _active_count(first) == 5
    assert torch.equal(first, second)
    assert int(sparsifier._k_anneal_step) == 2


def test_k_anneal_teacher_counter_syncs_post_step() -> None:
    model = build_place_model(_config(), _context())
    controller = model.teacher_controller
    teacher_stack = controller.teacher_encoder_stack
    assert teacher_stack is not None
    online_sparsifier = model.encoder_stack.sparsifier
    teacher_sparsifier = teacher_stack.sparsifier

    model.forward_sequence(_batch())
    assert int(online_sparsifier._k_anneal_step) == 1
    assert int(teacher_sparsifier._k_anneal_step) == 0

    controller.update(current_step=1)
    assert int(teacher_sparsifier._k_anneal_step) == 1
    teacher_sparsifier(torch.randn(2, NUM_UNITS))
    assert int(teacher_sparsifier._k_anneal_step) == 1


def test_k_anneal_checkpoint_mid_schedule_roundtrip() -> None:
    config = _config()
    model = build_place_model(config, _context())
    model.forward_sequence(_batch())
    model.forward_sequence(_batch())
    assert int(model.encoder_stack.sparsifier._k_anneal_step) == 2

    resumed = build_place_model(config, _context())
    resumed.load_state_dict(model.state_dict())
    resumed_sparsifier = resumed.encoder_stack.sparsifier
    assert int(resumed_sparsifier._k_anneal_step) == 2
    resumed_sparsifier.eval()
    values = torch.arange(1, NUM_UNITS + 1, dtype=torch.float32).unsqueeze(0)
    assert _active_count(resumed_sparsifier(values)) == 5


def test_k_anneal_schema_rejects_invalid_or_stacked_schedules() -> None:
    for field_name in ("kwinners_k_anneal_start", "kwinners_k_anneal_steps"):
        with pytest.raises(ValidationError, match=field_name):
            SparsifierConfig(type="kwinners", **{field_name: -1})

    with pytest.raises(ValidationError, match="both be positive"):
        SparsifierConfig(type="kwinners", kwinners_k_anneal_start=8)
    with pytest.raises(ValidationError, match="only used by the 'kwinners'"):
        SparsifierConfig(
            type="sparsemax",
            kwinners_k_anneal_start=8,
            kwinners_k_anneal_steps=10,
        )

    incompatible = (
        {"kwinners_boost_strength": 1.0},
        {"kwinners_balance_bias_rate": 0.01},
        {"kwinners_selection_noise_scale": 1.0},
    )
    for extra_fields in incompatible:
        with pytest.raises(ValidationError, match="cannot be combined"):
            SparsifierConfig(
                type="kwinners",
                kwinners_k_anneal_start=8,
                kwinners_k_anneal_steps=10,
                **extra_fields,
            )

    with pytest.raises(ValidationError, match="final k=4"):
        SpatialModelConfig(
            sparsifier=SparsifierConfig(
                type="kwinners",
                k_fraction=0.5,
                kwinners_k_anneal_start=2,
                kwinners_k_anneal_steps=10,
            ),
            training=SpatialTrainingConfig(code_dim=NUM_UNITS),
        )

    oversized = KWinnersSparsifier(
        k_fraction=0.25,
        num_units=NUM_UNITS,
        k_anneal_start=NUM_UNITS + 1,
        k_anneal_steps=10,
    )
    with pytest.raises(ValueError, match="must not exceed num_units"):
        oversized(torch.ones(1, NUM_UNITS))
