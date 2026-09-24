"""The one-file version in minimal/place_cells.py computes what the package computes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay
from placecell_research.config import load_experiment_config
from placecell_research.measures.decoding import fit_ridge, ridge_predict, scores
from placecell_research.measures.single_unit import rectified_track
from placecell_research.objectives import build_objectives, compute_total_loss
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.training import TrainLoopConfig, train_model
from placecell_research.training.optimizer import build_optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_CONFIGS = REPO_ROOT / "configs" / "thesis"
EPISODES, STEPS, NUM_ACTIONS = 3, 24, 3
CONDITIONS_NEEDING_OTHER_MODULES = {
    "feedforward_encoder",
    "l1_0.003",
    "l1_0.01",
    "l1_0.03",
    "next_visual_latent_target",
    "reconstruction_target",
    "reconstruction_target_no_weight_decay",
    "retrofitted_competition",
    "competition_added_after_training",
    "museum",
    "museum_untrained",
}
SELF_MOTION_CHANNELS = ["step_displacement", "angular_velocity"]


def load_place_cells():
    spec = importlib.util.spec_from_file_location(
        "place_cells", REPO_ROOT / "minimal" / "place_cells.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if importlib.util.find_spec("miniworld") is None:
    pytest.skip("minimal/place_cells.py needs MiniWorld", allow_module_level=True)
place_cells = load_place_cells()


def load_condition(condition):
    return load_experiment_config(THESIS_CONFIGS / f"{condition}.yaml", [])


def test_every_thesis_condition_is_supported_or_needs_other_modules():
    names = {path.stem for path in THESIS_CONFIGS.glob("*.yaml")}
    assert set(place_cells.THESIS_CONDITIONS).isdisjoint(CONDITIONS_NEEDING_OTHER_MODULES)
    assert set(place_cells.THESIS_CONDITIONS) | CONDITIONS_NEEDING_OTHER_MODULES == names


@pytest.mark.parametrize("condition", sorted(place_cells.THESIS_CONDITIONS))
def test_condition_arguments_are_the_package_config(condition):
    config = load_condition(condition)
    model, training = config.spatial_model, config.spatial_model.training
    regularizers = model.objectives["vicreg_encoder"]
    prediction = model.objectives["prediction_cosine"]
    args = place_cells.parse_args(["--condition", condition])
    expected = {
        "seed": config.seed.global_seed,
        "episodes": config.collection.episodes,
        "episode_length": config.collection.episode_length,
        "objects": config.environment.env_kwargs.get("place_landmark_objects", True),
        "frames_per_episode": config.vision.max_frames_per_episode,
        "latent_dim": config.vision.latent_dim,
        "vision_epochs": config.vision.epochs,
        "vision_batch_size": config.vision.batch_size,
        "vision_learning_rate": config.vision.learning_rate,
        "encoder_cell": model.encoder.family,
        "encoder_width": model.encoder.layer_sizes[0],
        "encoder_layers": len(model.encoder.layer_sizes),
        "code_dim": training.code_dim,
        "winners": max(1, round(training.code_dim * model.sparsifier.k_fraction)),
        "predictor_cell": model.predictor.family,
        "predictor_width": model.predictor.layer_sizes[0],
        "predictor_layers": len(model.predictor.layer_sizes),
        "action_embedding_dim": model.predictor.action_embedding_dim,
        "ema_decay": model.teacher_student.ema_decay,
        "prediction_weight": prediction.weight,
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
    assert (model.sparsifier.type, set(model.objectives)) == (
        "kwinners",
        {"prediction_cosine", "vicreg_encoder"},
    )
    if args.prediction_from == "encoder":
        assert prediction.type == "timescale_alignment" and args.target_offset == 0
        assert prediction.targets == ["encoder.place_codes", "teacher.place_codes"]
        assert (prediction.timescale, prediction.event_gated) == (0, False)
    else:
        assert (prediction.type, prediction.loss_type) == ("prediction_alignment", "cosine")
        assert args.target_offset == prediction.target_offset
    channels = ["action"] if "action" in args.motion_inputs else []
    channels += SELF_MOTION_CHANNELS if "self_motion" in args.motion_inputs else []
    assert model.inputs.predictor_context_channels == channels


def test_baseline_details_are_the_package_config():
    from placecell_research.envs.miniworld_wallgap_asym_large import (
        ALLOWED_REGION_BOUNDS_BY_NAME,
        FULL_ROOM_BOUNDS_BY_NAME,
    )

    config = load_condition("baseline")
    assert config.vision.channels == [32, 64, 128, 256, 512]
    assert (config.vision.type, config.vision.loss_type) == ("autoencoder", "mse")
    assert config.spatial_model.training.optimizer == "adam"
    assert config.environment.env_kwargs == {
        "forward_step": place_cells.FORWARD_STEP,
        "turn_step": place_cells.TURN_STEP_DEGREES,
    }
    assert config.collection.action_probabilities == place_cells.ACTION_PROBABILITIES
    assert list(place_cells.SPAWN_REGIONS.items()) == [
        (name, ALLOWED_REGION_BOUNDS_BY_NAME[name]) for name in config.collection.spawn_regions
    ]
    room_bounds = [room[:4] for room in place_cells.ROOMS.values()]
    assert room_bounds == list(FULL_ROOM_BOUNDS_BY_NAME.values())
    bounds = overlay_bounds(resolve_world_overlay(config.environment.env_id))
    edges = place_cells.arena_edges()
    assert [(edge[0], edge[-1]) for edge in edges] == [
        tuple(np.float32(value) for value in axis) for axis in bounds
    ]


def test_explicit_flags_win_over_smoke_and_condition():
    args = place_cells.parse_args(["--smoke", "--condition", "untrained", "--epochs", "10"])
    assert (args.epochs, args.episodes, args.learning_rate) == (10, 20, 1e-12)
    args = place_cells.parse_args(["--condition", "self_motion_only", "--motion-inputs", "action"])
    assert args.motion_inputs == ["action"]


def recurrent_layers(temporal):
    return temporal.lstm_layers if hasattr(temporal, "lstm_layers") else temporal.gru_layers


def parameter_pairs(package, model, target_encoder):
    """(package parameter, one-file parameter) for every weight of the model."""
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
        for layer, cell in enumerate(recurrent_layers(temporal)):
            for name, parameter in cell.named_parameters():
                pairs.append(
                    (parameter, getattr(encoder.recurrent, name.replace("_l0", f"_l{layer}")))
                )
    predictor = model["predictor"]
    if package.action_embedding is not None:
        pairs.append((package.action_embedding.weight, predictor.action_embedding.weight))
    pairs += [
        (package.predictor_temporal.input_projection.weight, predictor.input_layer.weight),
        (package.predictor_temporal.input_projection.bias, predictor.input_layer.bias),
        (package.predictor_head.linear.weight, predictor.head.weight),
        (package.predictor_head.linear.bias, predictor.head.bias),
    ]
    for layer, cell in enumerate(recurrent_layers(package.predictor_temporal)):
        for name, parameter in cell.named_parameters():
            pairs.append(
                (parameter, getattr(predictor.recurrent, name.replace("_l0", f"_l{layer}")))
            )
    return pairs


def rows_in_use(package_parameter, minimal_parameter):
    """The package embeds one padding action that never occurs in WallGap; compare real rows."""
    return package_parameter[: minimal_parameter.shape[0]]


def make_every_winner_negative(package):
    """Lower both head biases so that a competition that rectified would fail the comparison."""
    teacher = package.teacher_controller.teacher_encoder_stack
    for stack in (package.encoder_stack, teacher):
        stack.encoder_head.linear.bias.sub_(1.0)


def build_both(condition):
    config = load_condition(condition)
    context = ModelBuildContext(
        num_actions=NUM_ACTIONS, observation_dim=64, kinematics_dim=4, total_optimizer_steps=1
    )
    torch.manual_seed(0)
    package = build_place_model(config.spatial_model, context)
    with torch.no_grad():
        for parameter in package.teacher_controller.teacher_encoder_stack.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        make_every_winner_negative(package)
    args = place_cells.parse_args(["--device", "cpu", "--condition", condition])
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


def decayed_parameters(optimizer):
    return {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"] > 0
        for parameter in group["params"]
    }


@pytest.mark.parametrize(
    "condition",
    [
        "baseline",
        "winners_5",
        "no_ema",
        "same_step_with_predictor",
        "same_step_no_predictor",
        "self_motion_only",
        "encoder_gru_predictor_lstm",
    ],
)
def test_codes_losses_optimizer_step_and_target_update_match(condition, tmp_path):
    config, context, package, args, model, target_encoder, pairs = build_both(condition)
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
    package_decayed = decayed_parameters(package_optimizer)
    minimal_decayed = decayed_parameters(minimal_optimizer)
    trainable = [pair for pair in pairs if pair[1].requires_grad]
    assert [id(package_parameter) in package_decayed for package_parameter, _ in trainable] == [
        id(minimal_parameter) in minimal_decayed for _, minimal_parameter in trainable
    ]
    before_step = [parameter.detach().clone() for pair in pairs for parameter in pair]

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
    for checkpoint in tmp_path.glob("*.pt"):
        checkpoint.unlink()
    minimal_losses = place_cells.training_step(
        model, target_encoder, minimal_optimizer, minimal_batch, args
    )
    assert minimal_losses["total"] == pytest.approx(float(metrics["loss/total"]), rel=1e-5)
    after_step = [parameter for pair in pairs for parameter in pair]
    changed = [
        not torch.equal(after, before)
        for after, before in zip(after_step, before_step, strict=True)
    ]
    assert changed[0::2] == changed[1::2]
    assert any(changed)
    for package_parameter, minimal_parameter in pairs:
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
    measures = place_cells.rate_maps_and_spatial_information(codes, position, 30)
    expected = np.maximum(
        track["spatial_information_bits"] - track["spatial_information_null_95"], 0
    )
    np.testing.assert_allclose(
        measures["spatial_information"], track["spatial_information_bits"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        measures["spatial_information_above_null"], expected, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(measures["rate_maps"], track["rate_maps"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(measures["occupancy"], track["occupancy"], rtol=1e-6)


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
