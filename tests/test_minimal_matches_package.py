"""The one-file version in minimal/place_cells.py computes what the package computes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay
from placecell_research.config import load_experiment_config
from placecell_research.envs.miniworld_wallgap_asym_large import (
    _ALLOWED_REGION_BOUNDS_BY_NAME,
    FULL_ROOM_BOUNDS_BY_NAME,
)
from placecell_research.measures.decoding import fit_ridge, ridge_predict, scores
from placecell_research.measures.single_unit import rectified_track
from placecell_research.objectives import build_objectives, compute_total_loss
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.training import TrainLoopConfig, train_model
from placecell_research.training.optimizer import build_optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "configs" / "thesis" / "baseline.yaml"
EPISODES, STEPS, NUM_ACTIONS = 3, 24, 3


def load_place_cells():
    spec = importlib.util.spec_from_file_location(
        "place_cells", REPO_ROOT / "minimal" / "place_cells.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    place_cells = load_place_cells()
except Exception as error:
    pytest.skip(f"minimal/place_cells.py needs MiniWorld: {error}", allow_module_level=True)


def test_defaults_are_the_baseline_config():
    config = load_experiment_config(BASELINE, [])
    model, training = config.spatial_model, config.spatial_model.training
    regularizers = model.objectives["vicreg_encoder"]
    args = place_cells.parse_args([])
    expected = {
        "seed": config.seed.global_seed,
        "episodes": config.collection.episodes,
        "episode_length": config.collection.episode_length,
        "frames_per_episode": config.vision.max_frames_per_episode,
        "latent_dim": config.vision.latent_dim,
        "vision_epochs": config.vision.epochs,
        "vision_batch_size": config.vision.batch_size,
        "vision_learning_rate": config.vision.learning_rate,
        "encoder_width": model.encoder.layer_sizes[0],
        "encoder_layers": len(model.encoder.layer_sizes),
        "code_dim": training.code_dim,
        "winners": round(training.code_dim * model.sparsifier.k_fraction),
        "predictor_width": model.predictor.layer_sizes[0],
        "predictor_layers": len(model.predictor.layer_sizes),
        "action_embedding_dim": model.predictor.action_embedding_dim,
        "ema_decay": model.teacher_student.ema_decay,
        "variance_weight": regularizers.variance_weight,
        "covariance_weight": regularizers.covariance_weight,
        "minimum_std": regularizers.minimum_std,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "gradient_clip_norm": training.gradient_clip_norm,
        "epochs": training.epochs,
        "batch_size": training.batch_size,
        "measure_episodes": config.representation_collection.max_episodes,
        "null_shuffles": config.measures.null_shuffles,
    }
    assert {name: getattr(args, name) for name in expected} == expected
    assert set(model.encoder.layer_sizes) == {args.encoder_width}
    assert set(model.predictor.layer_sizes) == {args.predictor_width}
    assert config.vision.channels == [32, 64, 128, 256, 512]
    assert (config.vision.type, config.vision.loss_type) == ("autoencoder", "mse")
    assert (training.optimizer, model.sparsifier.type) == ("adam", "kwinners")
    assert config.environment.env_kwargs == {
        "forward_step": place_cells.FORWARD_STEP,
        "turn_step": place_cells.TURN_STEP_DEGREES,
    }
    assert config.collection.action_probabilities == place_cells.ACTION_PROBABILITIES
    assert place_cells.SPAWN_REGIONS == {
        name: _ALLOWED_REGION_BOUNDS_BY_NAME[name] for name in config.collection.spawn_regions
    }
    room_bounds = [room[:4] for room in place_cells.ROOMS.values()]
    assert room_bounds == list(FULL_ROOM_BOUNDS_BY_NAME.values())
    bounds = overlay_bounds(resolve_world_overlay(config.environment.env_id))
    edges = place_cells.arena_edges()
    assert [(edge[0], edge[-1]) for edge in edges] == [
        tuple(np.float32(value) for value in axis) for axis in bounds
    ]


def parameter_pairs(package, model, target_encoder):
    """(package parameter, one-file parameter) for every weight of the baseline model."""
    pairs = []
    stacks = [
        (package.encoder_stack, model["encoder"]),
        (package.teacher_controller.teacher_encoder_stack, target_encoder),
    ]
    for stack, encoder in stacks:
        temporal = stack.encoder_temporal
        pairs += [
            (temporal.input_projection.weight, encoder.input_layer.weight),
            (temporal.input_projection.bias, encoder.input_layer.bias),
            (stack.encoder_head.linear.weight, encoder.head.weight),
            (stack.encoder_head.linear.bias, encoder.head.bias),
        ]
        for layer, lstm in enumerate(temporal.lstm_layers):
            for name, parameter in lstm.named_parameters():
                pairs.append((parameter, getattr(encoder.lstm, name.replace("_l0", f"_l{layer}"))))
    predictor = model["predictor"]
    pairs += [
        (package.action_embedding.weight, predictor.action_embedding.weight),
        (package.predictor_temporal.input_projection.weight, predictor.input_layer.weight),
        (package.predictor_temporal.input_projection.bias, predictor.input_layer.bias),
        (package.predictor_head.linear.weight, predictor.head.weight),
        (package.predictor_head.linear.bias, predictor.head.bias),
    ]
    for layer, gru in enumerate(package.predictor_temporal.gru_layers):
        for name, parameter in gru.named_parameters():
            pairs.append((parameter, getattr(predictor.gru, name.replace("_l0", f"_l{layer}"))))
    return pairs


def rows_in_use(package_parameter, minimal_parameter):
    """The package embeds one padding action that never occurs in WallGap; compare real rows."""
    return package_parameter[: minimal_parameter.shape[0]]


def build_both():
    config = load_experiment_config(BASELINE, [])
    context = ModelBuildContext(
        num_actions=NUM_ACTIONS, observation_dim=64, kinematics_dim=4, total_optimizer_steps=1
    )
    torch.manual_seed(0)
    package = build_place_model(config.spatial_model, context)
    teacher = package.teacher_controller.teacher_encoder_stack
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        for stack in (package.encoder_stack, teacher):
            stack.encoder_head.linear.bias.sub_(1.0)
    args = place_cells.parse_args(["--device", "cpu"])
    model, target_encoder = place_cells.build_model(args, NUM_ACTIONS)
    pairs = parameter_pairs(package, model, target_encoder)
    assert len(pairs) == len(list(model.parameters())) + len(list(target_encoder.parameters()))
    with torch.no_grad():
        for package_parameter, minimal_parameter in pairs:
            minimal_parameter.copy_(rows_in_use(package_parameter, minimal_parameter))
    return config, context, package, args, model, target_encoder, pairs


def batches():
    generator = torch.Generator().manual_seed(1)
    latent = torch.randn(EPISODES, STEPS, 64, generator=generator)
    actions = torch.randint(0, NUM_ACTIONS, (EPISODES, STEPS), generator=generator)
    kinematics = torch.randn(EPISODES, STEPS, 4, generator=generator)
    package_batch = {
        "latent": latent,
        "actions": actions,
        "kinematics": kinematics,
        "valid_steps": torch.ones(EPISODES, STEPS, dtype=torch.bool),
        "position_xy": torch.randn(EPISODES, STEPS, 2, generator=generator),
    }
    minimal_batch = {
        "visual_latent": latent,
        "actions": actions,
        "self_motion": kinematics[..., :2],
    }
    return package_batch, minimal_batch


def test_codes_losses_optimizer_step_and_target_update_match(tmp_path):
    config, context, package, args, model, target_encoder, pairs = build_both()
    package_batch, minimal_batch = batches()
    objectives = build_objectives(package, config.spatial_model)

    package.train()
    bundle = package.forward_sequence(package_batch)
    _, metrics = compute_total_loss(
        objectives.objectives, bundle, package_batch, config.spatial_model
    )
    place_code, predicted_code, target_code = place_cells.forward(
        model, target_encoder, minimal_batch
    )
    assert torch.equal((place_code < 0).sum(-1), torch.full((EPISODES, STEPS), args.winners))
    torch.testing.assert_close(place_code, bundle.get_representation("encoder.place_codes"))
    torch.testing.assert_close(target_code, bundle.get_representation("teacher.place_codes"))
    torch.testing.assert_close(
        predicted_code, bundle.get_representation("predictor.place_codes")[:, 1:]
    )
    losses = place_cells.compute_losses(place_code, predicted_code, target_code, args)
    expected_losses = {
        "prediction": metrics["loss/prediction_cosine"],
        "variance": metrics["vicreg_encoder/variance_loss"],
        "covariance": metrics["vicreg_encoder/covariance_loss"],
        "total": metrics["loss/total"],
    }
    for name, expected in expected_losses.items():
        torch.testing.assert_close(losses[name], expected, msg=name)

    package_optimizer, _ = build_optimizer(
        package, objectives.auxiliary_heads, config.spatial_model.training
    )
    minimal_optimizer = place_cells.build_optimizer(model, args)
    package_decayed = {
        id(parameter)
        for group in package_optimizer.param_groups
        if group["weight_decay"] > 0
        for parameter in group["params"]
    }
    minimal_decayed = {
        id(parameter)
        for group in minimal_optimizer.param_groups
        if group["weight_decay"] > 0
        for parameter in group["params"]
    }
    trainable = [pair for pair in pairs if pair[1].requires_grad]
    assert [id(package_parameter) in package_decayed for package_parameter, _ in trainable] == [
        id(minimal_parameter) in minimal_decayed for _, minimal_parameter in trainable
    ]
    before_step = [minimal_parameter.detach().clone() for _, minimal_parameter in pairs]

    config.spatial_model.training.epochs = 1
    train_model(
        package,
        objectives,
        config.spatial_model,
        [package_batch],
        None,
        TrainLoopConfig(
            training=config.spatial_model.training,
            checkpoint_dir=tmp_path,
            device=torch.device("cpu"),
            build_context=context.to_checkpoint_payload({}),
        ),
    )
    minimal_losses = place_cells.training_step(
        model, target_encoder, minimal_optimizer, minimal_batch, args
    )
    assert minimal_losses["total"] == pytest.approx(float(metrics["loss/total"]), rel=1e-5)
    for (package_parameter, minimal_parameter), before in zip(pairs, before_step, strict=True):
        assert not torch.equal(minimal_parameter, before)
        torch.testing.assert_close(
            minimal_parameter, rows_in_use(package_parameter, minimal_parameter)
        )


def test_measures_match_package():
    rng = np.random.default_rng(3)
    episodes, steps, units = 4, 400, 64
    position = rng.uniform((-24, -30), (24, 36), (episodes, steps, 2)).astype(np.float32)
    lowered = 2.0 * (rng.random((episodes, steps, 1)) < 0.3)
    pre_competition = torch.from_numpy(rng.normal(size=(episodes, steps, units)) - lowered)
    codes = place_cells.competition(pre_competition.float(), 5).numpy()
    codes[..., :4] = 0.0
    assert (codes < 0).any() and (codes > 0).any()
    bounds = overlay_bounds(resolve_world_overlay("MiniWorld-WallGapAsymLarge-v0"))
    valid = np.ones((episodes, steps), dtype=bool)
    track = rectified_track(np.maximum(codes, 0.0), position, valid, bounds, shuffles=30)
    above_null, rate_maps = place_cells.spatial_information_above_null(codes, position, 30)
    expected = np.maximum(
        track["spatial_information_bits"] - track["spatial_information_null_95"], 0
    )
    np.testing.assert_allclose(above_null, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(rate_maps, track["rate_maps"], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("noise", [0.1, 30.0])
def test_ridge_decoder_matches_package(noise):
    rng = np.random.default_rng(4)
    features = {
        split: rng.normal(size=(600, 48)).astype(np.float32)
        for split in ("train", "validation", "test")
    }
    data = {
        split: (x, (x[:, :2] * 3 + noise * rng.normal(size=(600, 2))).astype(np.float32))
        for split, x in features.items()
    }
    decoder = fit_ridge(data)
    expected = scores(ridge_predict(decoder, data["test"][0]), data["test"][1], "position")
    rmse = place_cells.ridge_decoding_rmse(data["train"], data["validation"], data["test"])
    assert rmse == pytest.approx(expected["rmse"], rel=1e-5)
