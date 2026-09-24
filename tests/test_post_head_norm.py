"""The second cage: RMSNorm on the code width, directly before the k-winners competition."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from placecell_research.config.schema import (
    ObjectiveConfig,
    SparsifierConfig,
    SpatialModelConfig,
    SpatialTrainingConfig,
    TemporalFamilyConfig,
)
from placecell_research.downstream.frozen_extractor import _OnlineEncoderRuntime
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.spatial_model.components.teacher import PostHeadRMSNorm
from placecell_research.training.optimizer import _named_parameter_lookup, build_optimizer

CODE_DIM = 16
K_FRACTION = 0.25
HIDDEN = 8
NORM_KEY = "encoder_post_head_norm"


def _context() -> ModelBuildContext:
    return ModelBuildContext(
        num_actions=4, observation_dim=8, kinematics_dim=2, total_optimizer_steps=10
    )


def _model_config(post_head_norm: str = "none", **model_knobs: object) -> SpatialModelConfig:
    config = SpatialModelConfig()
    config.sparsifier = SparsifierConfig(type="kwinners", k_fraction=K_FRACTION)
    config.encoder = TemporalFamilyConfig(family="lstm", layer_sizes=[HIDDEN, HIDDEN])
    config.predictor.layer_sizes = [HIDDEN]
    config.training = SpatialTrainingConfig(code_dim=CODE_DIM)
    config.objectives = {
        "prediction_cosine": ObjectiveConfig(type="prediction_alignment", weight=1.0)
    }
    config.encoder_post_head_norm = post_head_norm
    for name, value in model_knobs.items():
        setattr(config, name, value)
    return config


def _model(post_head_norm: str = "none", seed: int = 0, **model_knobs: object):
    torch.manual_seed(seed)
    return build_place_model(_model_config(post_head_norm, **model_knobs), _context()).eval()


def _batch(seed: int = 0, time_steps: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "latent": torch.randn(2, time_steps, 8, generator=generator) * 7.0,
        "actions": torch.randint(0, 4, (2, time_steps), generator=generator),
        "kinematics": torch.randn(2, time_steps, 2, generator=generator),
    }


def _encode(model, batch: dict[str, torch.Tensor]):
    with torch.no_grad():
        outputs, _state = model.encoder_stack.forward_stateful(
            batch["latent"], actions=batch["actions"], kinematics=batch["kinematics"]
        )
    return outputs


def test_default_adds_no_module_no_key_and_no_rng_draw() -> None:
    """(1) The knob is off by default and costs nothing when it is."""
    assert SpatialModelConfig().encoder_post_head_norm == "none"

    disabled = _model("none")
    assert isinstance(disabled.encoder_stack.encoder_post_head_norm, nn.Identity)
    assert not [key for key in disabled.state_dict() if NORM_KEY in key]

    caged = _model("rmsnorm_fixed")
    assert list(disabled.state_dict()) == list(caged.state_dict())
    for key, value in disabled.state_dict().items():
        assert torch.equal(value, caged.state_dict()[key]), key


def test_fixed_cage_pins_the_pre_sparsifier_to_rms_one() -> None:
    """(2) Every row the competition reads has RMS 1, whatever the trunk did."""
    outputs = _encode(_model("rmsnorm_fixed"), _batch())

    root_mean_square = outputs.pre_sparsifier.square().mean(dim=-1).sqrt()
    torch.testing.assert_close(
        root_mean_square, torch.ones_like(root_mean_square), atol=1e-5, rtol=0.0
    )
    uncaged = _encode(_model("none"), _batch()).pre_sparsifier
    assert not torch.allclose(
        uncaged.square().mean(dim=-1).sqrt(),
        torch.ones_like(root_mean_square),
        atol=1e-5,
    )


def test_cage_rescales_amplitudes_but_never_moves_the_winners() -> None:
    """(3) A positive per-row rescale cannot reorder the row, so k-winners picks the same units."""
    batch = _batch()
    uncaged = _encode(_model("none"), batch).place_codes
    caged = _encode(_model("rmsnorm_fixed"), batch).place_codes

    assert torch.equal(uncaged.ne(0), caged.ne(0))
    assert not torch.allclose(uncaged, caged)


def test_checkpoints_survive_the_fixed_cage_and_declare_the_learned_one() -> None:
    uncaged_state = _model("none").state_dict()
    incompatible = _model("rmsnorm_fixed").load_state_dict(uncaged_state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys

    learned = _model("rmsnorm")
    assert [key for key in learned.state_dict() if NORM_KEY in key] == [
        "encoder_stack.encoder_post_head_norm.weight",
        "teacher_controller.teacher_encoder_stack.encoder_post_head_norm.weight",
    ]
    with pytest.raises(RuntimeError, match=NORM_KEY):
        learned.load_state_dict(uncaged_state, strict=True)


def test_frozen_extractor_reproduces_the_caged_path_exactly() -> None:
    """(5) The downstream runtime rebuilds this path by hand; a skipped cage shows up here."""
    model = _model("rmsnorm_fixed")
    runtime = _OnlineEncoderRuntime(model=model, representation_source="encoder.place_codes")

    for episode in range(2):
        batch = _batch(seed=episode, time_steps=4)
        reference = _encode(model, batch)
        runtime.reset()
        stepped = []
        with torch.no_grad():
            for step in range(4):
                bundle = runtime.extract(
                    batch["latent"][:, step : step + 1],
                    actions=batch["actions"][:, step : step + 1],
                    kinematics=batch["kinematics"][:, step : step + 1],
                )
                stepped.append(bundle.modules["encoder"].place_codes[:, 0])
        torch.testing.assert_close(
            torch.stack(stepped, dim=1), reference.place_codes, atol=1e-5, rtol=1e-5
        )


def test_chunked_state_matches_the_full_sequence_under_the_cage() -> None:
    """(5) Stepping the stack in chunks and threading its state is the same arithmetic."""
    model = _model("rmsnorm_fixed")
    batch = _batch(time_steps=6)
    reference = _encode(model, batch)

    chunks = []
    state = None
    with torch.no_grad():
        for start in (0, 3):
            stop = start + 3
            outputs, state = model.encoder_stack.forward_stateful(
                batch["latent"][:, start:stop],
                actions=batch["actions"][:, start:stop],
                kinematics=batch["kinematics"][:, start:stop],
                initial_state=state,
            )
            chunks.append(outputs.place_codes)
    torch.testing.assert_close(
        torch.cat(chunks, dim=1), reference.place_codes, atol=1e-5, rtol=1e-5
    )


def _weight_decay_for(optimizer, parameter: nn.Parameter) -> float:
    owning_groups = [
        group
        for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    ]
    assert len(owning_groups) == 1
    return float(owning_groups[0]["weight_decay"])


def test_learned_gain_is_never_weight_decayed() -> None:
    """(5) The gain is a scale, not a weight matrix."""
    model = _model("rmsnorm")
    training = SpatialTrainingConfig(code_dim=CODE_DIM)
    optimizer, _groups = build_optimizer(model, nn.ModuleDict(), training)

    assert isinstance(model.encoder_stack.encoder_post_head_norm, PostHeadRMSNorm)
    assert _weight_decay_for(optimizer, model.encoder_stack.encoder_post_head_norm.weight) == 0.0
    assert training.weight_decay > 0.0


def test_exemption_is_scoped_to_the_cage_and_leaves_plain_rmsnorm_decayed() -> None:
    """The backbones carry their own nn.RMSNorm."""
    container = nn.ModuleDict({"backbone": nn.RMSNorm(CODE_DIM), "cage": PostHeadRMSNorm(CODE_DIM)})
    _names, no_decay_ids = _named_parameter_lookup([container])

    assert id(container["cage"].weight) in no_decay_ids
    assert id(container["backbone"].weight) not in no_decay_ids
