"""The control reconstructs the current observation through encoder Top-10 codes."""

import copy
import dataclasses
from pathlib import Path

import pytest
import torch

from placecell_research.config.loader import load_experiment_config
from placecell_research.config.validator import validate_experiment_config
from placecell_research.objectives.reconstruction import LatentReconstructionObjective
from placecell_research.spatial_model.types import RepresentationBundle

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/wallgap.yaml"
)
RECONSTRUCTION_OVERRIDES = [
    "spatial_model.objectives.prediction_cosine.weight=0",
    "spatial_model.objectives.current_latent_reconstruction.type=latent_reconstruction",
    "spatial_model.objectives.current_latent_reconstruction.targets=[encoder.place_codes]",
    "spatial_model.objectives.current_latent_reconstruction.weight=1",
]


def test_only_objective_changes_from_current_baseline():
    baseline = load_experiment_config(RECIPE)
    control = load_experiment_config(RECIPE, RECONSTRUCTION_OVERRIDES)
    validate_experiment_config(control)
    a, b = dataclasses.asdict(baseline), dataclasses.asdict(control)
    for key in ("dataset", "splits", "seed", "vision", "reuse", "policies"):
        assert a[key] == b[key]
    expected = copy.deepcopy(a["spatial_model"])
    expected["objectives"] = b["spatial_model"]["objectives"]
    assert expected == b["spatial_model"]
    assert control.spatial_model.objectives["prediction_cosine"].weight == 0
    objective = control.spatial_model.objectives["current_latent_reconstruction"]
    assert objective.targets == ["encoder.place_codes"]
    assert objective.weight == 1


def test_target_is_current_timestep_and_padding_is_excluded():
    from placecell_research.config.schema import ObjectiveConfig

    config = ObjectiveConfig(type="latent_reconstruction", targets=["encoder.place_codes"])
    objective = LatentReconstructionObjective(name="current", config=config)
    latent = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    prediction = latent.clone().requires_grad_()
    bundle = RepresentationBundle(auxiliary_outputs={"current.reconstruction": prediction})
    batch = {"latent": latent, "valid_steps": torch.tensor([[True, True, True, False]])}
    assert objective.compute(bundle, batch).loss.item() == 0
    with torch.no_grad():
        prediction[0, 2] += 2
        prediction[0, 3] += 100
    loss = objective.compute(bundle, batch).loss
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert prediction.grad[0, 2].item() != 0
    assert prediction.grad[0, 0].item() == 0
    assert prediction.grad[0, 3].item() == 0


def test_full_model_reconstruction_gradient_reaches_encoder_not_predictor():
    from placecell_research.objectives.registry import build_objectives
    from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model

    config = load_experiment_config(RECIPE, RECONSTRUCTION_OVERRIDES)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = build_place_model(config.spatial_model, ModelBuildContext(3, 64, 4, 1))
        built = build_objectives(model, config.spatial_model)
        model.set_auxiliary_heads(built.auxiliary_heads)
        batch = {
            "latent": torch.randn(2, 4, 64),
            "actions": torch.zeros(2, 4, dtype=torch.long),
            "kinematics": torch.randn(2, 4, 4),
            "valid_steps": torch.ones(2, 4, dtype=torch.bool),
        }
        bundle = model.forward_sequence(batch)
        objective = next(o for o in built.objectives if o.name == "current_latent_reconstruction")
        loss = objective.compute(bundle, batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {name: p.grad for name, p in model.named_parameters()}
        assert any(
            g is not None and g.abs().sum() > 0
            for n, g in gradients.items()
            if n.startswith("encoder_stack.")
        )
        assert all(
            g is None or g.abs().sum() == 0
            for n, g in gradients.items()
            if n.startswith("predictor_")
        )
    finally:
        torch.set_num_threads(previous_threads)
