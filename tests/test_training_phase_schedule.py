from __future__ import annotations

from placecell_research.config.schema import (
    ObjectiveConfig,
    PhaseConfig,
    SpatialModelConfig,
)
from placecell_research.objectives.registry import build_objectives
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.training.loop import active_phase_for_epoch, build_phase_schedule


def _context() -> ModelBuildContext:
    return ModelBuildContext(
        num_actions=4, observation_dim=8, kinematics_dim=4, total_optimizer_steps=10
    )


def _small(config: SpatialModelConfig) -> SpatialModelConfig:
    config.training.code_dim = 16
    config.encoder.layer_sizes = [16]
    config.predictor.layer_sizes = [16]
    return config


def _base_model():
    config = _small(SpatialModelConfig())
    config.objectives = {
        "prediction_cosine": ObjectiveConfig(type="prediction_alignment", weight=1.0)
    }
    return build_place_model(config, _context())


def test_default_schedule_is_single_all_trainable_phase():
    model = _base_model()
    schedule = build_phase_schedule(epochs=8, phases=[], model=model)
    assert len(schedule) == 1
    assert schedule[0].epochs == 8
    assert set(schedule[0].train) == model.available_selectors()


def test_default_all_phase_keeps_targeted_objective_head_trainable():
    config = _small(SpatialModelConfig())
    config.objectives = {
        "reconstruction": ObjectiveConfig(
            type="latent_reconstruction",
            targets=["predictor.place_codes"],
        )
    }
    model = build_place_model(config, _context())
    built_objectives = build_objectives(model, config)
    model.set_auxiliary_heads(built_objectives.auxiliary_heads)
    schedule = build_phase_schedule(epochs=8, phases=[], model=model)

    model.set_trainable(set(schedule[0].train))

    head_parameters = list(model.auxiliary_heads["reconstruction"].parameters())
    assert head_parameters
    assert all(parameter.requires_grad for parameter in head_parameters)

    model.set_trainable({"encoder"})
    assert all(not parameter.requires_grad for parameter in head_parameters)


def test_active_phase_picks_by_cumulative_epochs():
    phases = [
        PhaseConfig(name="place", epochs=3, train=["encoder"]),
        PhaseConfig(name="head", epochs=2, train=["encoder_head"]),
    ]
    assert [active_phase_for_epoch(phases, e)[0].name for e in range(5)] == [
        "place",
        "place",
        "place",
        "head",
        "head",
    ]
    assert [active_phase_for_epoch(phases, e)[1] for e in range(5)] == [0, 1, 2, 0, 1]
