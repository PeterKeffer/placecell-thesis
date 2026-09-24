"""The baseline place-cell model of the thesis, from data collection to measures, in one file.

An agent walks at random through the WallGap arena (MiniWorld). A convolutional autoencoder
compresses each frame to a visual latent. An LSTM encoder and a k-winners competition turn the
latents into a place code, and a GRU predictor, given the action and the self-motion, predicts
the code that an EMA target encoder assigns to the next latent. The script prints the linear
position decoding error, the spatial information above a circular-shift null and the share of
silent units. In --output-dir it saves the rate maps of the 16 most informative units and of all
units, the place code along one test episode, the training curves, and all rate maps as an npz.
--condition switches to one of the simple thesis conditions; explicit flags still win.

    python minimal/place_cells.py --smoke    about a minute on a laptop
    python minimal/place_cells.py            thesis baseline: a GPU, 50 GB disk, 32 GB memory
"""

import argparse
import copy
import functools
import math
import multiprocessing
import os
import random
import sys
from itertools import islice, pairwise
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("PYGLET_SHADOW_WINDOW", "0")
elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLET_HEADLESS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from miniworld.entity import MeshEnt
from miniworld.miniworld import MiniWorldEnv
from torch import nn

# Arguments

SMOKE = dict(episodes=20, episode_length=256, frames_per_episode=32, vision_epochs=2, epochs=3)
SMOKE.update(encoder_width=64, predictor_width=64, batch_size=4, null_shuffles=19)
THESIS_CONDITIONS = {
    "baseline": {},
    **{f"winners_{winners}": dict(winners=winners) for winners in (1, 5, 26, 51, 128)},
    "no_competition": dict(winners=512),
    "no_competition_no_weight_decay": dict(winners=512, weight_decay=0.0),
    "variance_regularizer_off": dict(variance_weight=0.0),
    "covariance_regularizer_off": dict(covariance_weight=0.0),
    "both_regularizers_off": dict(variance_weight=0.0, covariance_weight=0.0),
    "both_regularizers_off_no_weight_decay": dict(
        variance_weight=0.0, covariance_weight=0.0, weight_decay=0.0
    ),
    "weight_decay_0": dict(weight_decay=0.0),
    "weight_decay_1e-6": dict(weight_decay=1e-6),
    "weight_decay_1e-5": dict(weight_decay=1e-5),
    "weight_decay_1e-4": dict(weight_decay=1e-4),
    "no_ema": dict(ema_decay=0.0),
    "no_prediction": dict(prediction_weight=0.0),
    "no_prediction_no_weight_decay": dict(prediction_weight=0.0, weight_decay=0.0),
    "same_step_with_predictor": dict(target_offset=0),
    "same_step_no_predictor": dict(target_offset=0, prediction_from="encoder"),
    "actions_only": dict(motion_inputs=["action"]),
    "self_motion_only": dict(motion_inputs=["self_motion"]),
    "no_motion_input": dict(motion_inputs=[]),
    **{
        f"encoder_{encoder}_predictor_{predictor}": dict(
            encoder_width=encoder, predictor_width=predictor
        )
        for encoder in (256, 512, 1024)
        for predictor in (256, 512, 1024)
        if (encoder, predictor) != (1024, 512)
    },
    "encoder_gru_predictor_gru": dict(encoder_cell="gru"),
    "encoder_gru_predictor_lstm": dict(encoder_cell="gru", predictor_cell="lstm"),
    "encoder_lstm_predictor_lstm": dict(predictor_cell="lstm"),
    "code_128": dict(code_dim=128),
    "code_256": dict(code_dim=256),
    "untrained": dict(epochs=1, learning_rate=1e-12, weight_decay=0.0),
    "objects_removed": dict(objects=False),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("place_cells_run"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=8192)
    parser.add_argument("--episode-length", type=int, default=2048)
    parser.add_argument("--no-objects", dest="objects", action="store_false")
    parser.add_argument("--frames-per-episode", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--vision-epochs", type=int, default=16)
    parser.add_argument("--vision-batch-size", type=int, default=256)
    parser.add_argument("--vision-learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-cell", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--encoder-width", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--winners", type=int, default=10)
    parser.add_argument("--predictor-cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--predictor-width", type=int, default=512)
    parser.add_argument("--predictor-layers", type=int, default=2)
    parser.add_argument("--motion-inputs", nargs="*", choices=["action", "self_motion"])
    parser.set_defaults(motion_inputs=["action", "self_motion"])
    parser.add_argument("--action-embedding-dim", type=int, default=32)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--prediction-from", choices=["predictor", "encoder"], default="predictor")
    parser.add_argument("--target-offset", type=int, choices=[0, 1], default=1)
    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=5.0)
    parser.add_argument("--covariance-weight", type=float, default=1.0)
    parser.add_argument("--minimum-std", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--measure-episodes", type=int, default=512)
    parser.add_argument("--null-shuffles", type=int, default=999)
    parser.add_argument("--condition", choices=THESIS_CONDITIONS, default="baseline")
    parser.add_argument("--smoke", action="store_true", help="shrink everything to about a minute")
    chosen = parser.parse_args(argv)
    presets = {**(SMOKE if chosen.smoke else {}), **THESIS_CONDITIONS[chosen.condition]}
    args = parser.parse_args(argv, namespace=argparse.Namespace(**presets))
    args.device = torch.device(args.device)
    return args


# Environment

ROOMS = {
    "northern_courtyard": (-18.0, 18.0, 6.0, 36.0, "brick_wall", "grass", True, 2.74),
    "central_corridor": (-6.0, 6.0, -6.0, 6.0, "metal_grill", "concrete_tiles", False, 2.74),
    "southern_yard_left": (-24.0, -6.0, -30.0, 4.5, "cinder_blocks", "wood_planks", True, 10.5),
    "southern_yard_right": (6.0, 24.0, -30.0, 4.5, "stucco", "concrete", True, 2.74),
}
SPAWN_REGIONS = {
    "courtyard_north": (-13.5, 13.5, 18.0, 33.0),
    "courtyard_south": (-13.5, 13.5, 9.0, 15.0),
    "corridor_indoor": (-3.0, 3.0, -3.0, 3.0),
    "woodplank_yard": (-21.0, -9.0, -27.0, 1.5),
    "concrete_yard": (9.0, 21.0, -27.0, 1.5),
}
ARENA = ((-24.0, 24.0), (-30.0, 36.0))
LANDMARKS = """
mesh, then height x z direction of each object
tree 3.0 -15 9 0.5  3.5 -14 14 1.2  2.8 -16 19 0.8  3.2 -13 23 2.1  3.0 -15 28 1.7  3.6 -17 33 0.3
tree 2.9 -5 11 2.5  3.3 -8 18 1.9  3.1 -3 25 0.6  2.7 2 13 2.8  3.4 7 22 1.1  3.0 4 31 2.3
cone 0.75 12 8 0  0.9 14 12 0  0.7 13 17 0  0.85 15 21 0  0.8 12 25 0  0.95 14 29 0
cone 0.75 16 32 0  0.9 11 35 0
barrel 1.2 -10 32 0.3  1.2 8 10 1.1  1.2 0 28 2.2
barrier 1.0 5 16 0.7  1.0 -2 24 2.4
office_desk 1.0 -4 3 0.5235987755982988  1.0 3 -2 -0.6283185307179586
duckie 0.5 -2 -4 1.0471975511965976  0.5 4 4 -0.7853981633974483  0.5 -3 2 0.2  0.5 2 -3 1.4
tree_pine 4.0 -21 -8 0.7853981633974483  4.5 -18 -15 -0.5235987755982988  3.8 -22 -21 0.9
tree_pine 4.2 -15 -25 2.1  4.0 -20 -28 1.3  4.3 -17 2 -1.7
barrel 1.3 -9 -5 1.5  1.3 -12 -12 -0.4  1.3 -8 -20 2.6  1.3 -14 -27 0.8
office_chair 1.2 -10 0 2.1  1.2 -19 -10 -2.4  1.2 -11 -18 0.6  1.2 -16 -24 -0.9
barrier 1.1 -13 -3 0.9  1.1 -20 -16 -1.6  1.1 -9 -26 2.5
barrel 1.2 9 -6 0.4  1.2 15 -11 -1.1  1.2 12 -18 1.9  1.2 18 -22 2.7  1.2 10 -27 -2.2
barrier 1.0 8 -3 1.8  1.0 20 -14 -0.5  1.0 14 -25 2.2
medkit 0.6 21 -10 0
"""
BACKDROP = ("building", 40.0, 60.0, 54.0, -math.pi / 2)
FLOOR_OFFSET = {"duckie": -0.07, "medkit": -0.545}
FORWARD_STEP = 0.26
TURN_STEP_DEGREES = 20.0


def landmarks():
    for line in LANDMARKS.strip().splitlines()[1:]:
        mesh, *numbers = line.split()
        for height, x, z, direction in np.array(numbers, dtype=float).reshape(-1, 4):
            yield mesh, height, x, z, direction


class WallGap(MiniWorldEnv):
    def __init__(self, episode_length, objects=True):
        self.objects = objects
        super().__init__(max_episode_steps=episode_length, render_mode="rgb_array")
        self.params.set("forward_step", FORWARD_STEP, FORWARD_STEP, FORWARD_STEP)
        self.params.set("turn_step", TURN_STEP_DEGREES, TURN_STEP_DEGREES, TURN_STEP_DEGREES)

    def _gen_world(self):
        north, corridor, yard_left, yard_right = [
            self.add_rect_room(
                *bounds, wall_tex=wall, floor_tex=floor, no_ceiling=open_sky, wall_height=height
            )
            for *bounds, wall, floor, open_sky, height in ROOMS.values()
        ]
        self.connect_rooms(north, corridor, min_x=-3.0, max_x=3.0)
        self.connect_rooms(corridor, yard_left, min_z=-3.0, max_z=3.0)
        self.connect_rooms(corridor, yard_right, min_z=-3.0, max_z=3.0)
        placed = [*landmarks(), BACKDROP] if self.objects else [BACKDROP]
        for mesh, height, x, z, direction in placed:
            position = np.array([x, FLOOR_OFFSET.get(mesh, 0.0) * height, z])
            self.place_entity(MeshEnt(mesh_name=mesh, height=height), pos=position, dir=direction)
        self.place_agent_in_spawn_region()

    def place_agent_in_spawn_region(self):
        regions = list(SPAWN_REGIONS.values())
        areas = [(max_x - min_x) * (max_z - min_z) for min_x, max_x, min_z, max_z in regions]
        threshold = self.np_random.uniform(0.0, sum(areas))
        min_x, max_x, min_z, max_z = next(
            region
            for region, total in zip(regions, np.cumsum(areas), strict=True)
            if threshold <= total
        )
        while True:
            x, z = self.np_random.uniform(min_x, max_x), self.np_random.uniform(min_z, max_z)
            position = np.array([x, 0.0, z])
            if not self.intersect(self.agent, position, self.agent.radius):
                break
        self.place_agent(pos=position, dir=self.np_random.uniform(-math.pi, math.pi))


# Data collection

ACTION_PROBABILITIES = {"turn_left": 0.3, "turn_right": 0.3, "move_forward": 0.4}
TENDENCY_RANGE = (-2.0, 2.0)
TENDENCY_START_SCALE, TENDENCY_DECAY, TENDENCY_NOISE_SCALE = 0.3, 0.85, 0.4
ACTIONS = MiniWorldEnv.Actions


def ou_actions(seed):
    rng = np.random.default_rng(seed)
    low, high = TENDENCY_RANGE
    turn_left_below = low + (high - low) * ACTION_PROBABILITIES["turn_left"]
    turn_right_above = turn_left_below + (high - low) * ACTION_PROBABILITIES["move_forward"]
    tendency = rng.standard_normal() * TENDENCY_START_SCALE
    while True:
        if tendency < turn_left_below:
            yield ACTIONS.turn_left
        elif tendency > turn_right_above:
            yield ACTIONS.turn_right
        else:
            yield ACTIONS.move_forward
        tendency = tendency * TENDENCY_DECAY + rng.standard_normal() * TENDENCY_NOISE_SCALE


def collect_episode(env, seed, steps):
    frame, _ = env.reset(seed=seed)
    frames, positions, headings, actions = [], [], [], []
    for action in islice(ou_actions(seed), steps):
        frames.append(frame.transpose(2, 0, 1))
        positions.append([env.agent.pos[0], env.agent.pos[2]])
        headings.append(env.agent.dir)
        actions.append(action)
        frame, *_ = env.step(action)
    position = np.array(positions, dtype=np.float32)
    heading = np.array(headings, dtype=np.float32)
    heading = np.arctan2(np.sin(heading), np.cos(heading))
    self_motion = np.zeros((steps, 2), dtype=np.float32)
    self_motion[1:, 0] = np.linalg.norm(np.diff(position, axis=0), axis=-1)
    self_motion[1:, 1] = (np.diff(heading) + np.pi) % (2 * np.pi) - np.pi
    rgb = np.stack(frames)
    return dict(rgb=rgb, actions=np.array(actions), position=position, self_motion=self_motion)


def episode_path(args, episode):
    return args.output_dir / "episodes" / f"{episode:05d}.npz"


@functools.cache
def worker_env(episode_length, objects):
    return WallGap(episode_length, objects)


def collect_and_save(args, episode):
    env = worker_env(args.episode_length, args.objects)
    arrays = collect_episode(env, args.seed + episode, args.episode_length)
    np.savez_compressed(episode_path(args, episode), **arrays)
    if (episode + 1) % 64 == 0 or episode + 1 == args.episodes:
        print(f"collected episode {episode + 1}/{args.episodes}", flush=True)
    return env.wall_segs[:, :, [0, 2]]


def collect_dataset(args):
    episode_path(args, 0).parent.mkdir(parents=True, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        walls = pool.map(functools.partial(collect_and_save, args), range(args.episodes))
    return walls[0]


def split_episodes(episodes, seed):
    order = list(range(episodes))
    random.Random(seed).shuffle(order)
    train_end = round(episodes * 0.6)
    validation_end = train_end + round(episodes * 0.1)
    return {
        "train": np.sort(order[:train_end]),
        "validation": np.sort(order[train_end:validation_end]),
        "test": np.sort(order[validation_end:]),
    }


# Visual autoencoder


class Autoencoder(nn.Module):
    def __init__(self, latent_dim, channels=(32, 64, 128, 256, 512), frame_shape=(3, 60, 80)):
        super().__init__()
        widths = [frame_shape[0], *channels]
        self.encoder = nn.Sequential()
        for width, next_width in pairwise(widths):
            self.encoder.extend([nn.Conv2d(width, next_width, 4, stride=2, padding=1), nn.ReLU()])
        self.encoded_shape = self.encoder(torch.zeros(1, *frame_shape)).shape[1:]
        self.to_latent = nn.Linear(self.encoded_shape.numel(), latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.encoded_shape.numel())
        self.decoder = nn.Sequential()
        for width, next_width in pairwise(widths[::-1]):
            self.decoder.extend(
                [nn.ConvTranspose2d(width, next_width, 4, stride=2, padding=1), nn.ReLU()]
            )
        self.decoder[-1] = nn.Sigmoid()

    def encode(self, frames):
        return self.to_latent(self.encoder(frames).flatten(1))

    def forward(self, frames):
        hidden = self.from_latent(self.encode(frames)).view(-1, *self.encoded_shape)
        return F.interpolate(
            self.decoder(hidden), size=frames.shape[2:], mode="bilinear", align_corners=False
        )


def chunks(array, size=65536):
    return [array[start : start + size] for start in range(0, len(array), size)]


def as_frames(rgb, device):
    return torch.as_tensor(rgb, device=device).float() / 255.0


def reconstruction_error(autoencoder, frames):
    return (autoencoder(frames) - frames).pow(2).sum(dim=(1, 2, 3))


def vision_frames(args, episode_ids):
    frames = []
    for index, episode in enumerate(episode_ids):
        rgb = np.load(episode_path(args, episode))["rgb"]
        steps = np.random.default_rng(index).choice(
            len(rgb), args.frames_per_episode, replace=False
        )
        frames.append(rgb[np.sort(steps)])
    return np.concatenate(frames)


def train_autoencoder(args, train_frames, validation_frames):
    autoencoder = Autoencoder(args.latent_dim).to(args.device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.vision_learning_rate)
    best_loss, best_state = math.inf, None
    for epoch in range(args.vision_epochs):
        autoencoder.train()
        for batch in torch.randperm(len(train_frames)).split(args.vision_batch_size):
            frames = as_frames(train_frames[batch.numpy()], args.device)
            loss = reconstruction_error(autoencoder, frames)
            optimizer.zero_grad(set_to_none=True)
            loss.mean().backward()
            optimizer.step()
        autoencoder.eval()
        with torch.inference_mode():
            validation_loss = sum(
                reconstruction_error(autoencoder, as_frames(frames, args.device)).sum().item()
                for frames in chunks(validation_frames, args.vision_batch_size)
            ) / len(validation_frames)
        print(f"autoencoder epoch {epoch + 1}: validation error {validation_loss:.2f}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, copy.deepcopy(autoencoder.state_dict())
    autoencoder.load_state_dict(best_state)
    return autoencoder.eval()


def encode_dataset(args, autoencoder):
    data = {"visual_latent": [], "actions": [], "self_motion": [], "position": []}
    with torch.inference_mode():
        for episode in range(args.episodes):
            arrays = dict(np.load(episode_path(args, episode)))
            visual_latent = autoencoder.encode(as_frames(arrays["rgb"], args.device))
            arrays["visual_latent"] = visual_latent.cpu().numpy()
            for key, values in data.items():
                values.append(arrays[key])
    return {key: np.stack(values) for key, values in data.items()}


# Place-cell model

RECURRENT_CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


def competition(pre_competition, winners):
    winning = pre_competition.topk(winners, dim=-1).indices
    mask = torch.zeros_like(pre_competition, dtype=torch.bool).scatter_(-1, winning, True)
    return torch.where(mask, pre_competition, torch.zeros_like(pre_competition))


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.winners = args.winners
        width, cell = args.encoder_width, RECURRENT_CELLS[args.encoder_cell]
        self.input_layer = nn.Linear(args.latent_dim, width)
        self.recurrent = cell(width, width, num_layers=args.encoder_layers, batch_first=True)
        self.head = nn.Linear(width, args.code_dim)

    def forward(self, visual_latent):
        hidden_state, _ = self.recurrent(torch.tanh(self.input_layer(visual_latent)))
        return competition(self.head(hidden_state), self.winners)


class Predictor(nn.Module):
    def __init__(self, args, num_actions):
        super().__init__()
        self.motion_inputs = args.motion_inputs
        width, cell = args.predictor_width, RECURRENT_CELLS[args.predictor_cell]
        input_dim = args.code_dim
        if "action" in self.motion_inputs:
            self.action_embedding = nn.Embedding(num_actions, args.action_embedding_dim)
            input_dim += args.action_embedding_dim
        if "self_motion" in self.motion_inputs:
            input_dim += 2
        self.input_layer = nn.Linear(input_dim, width)
        self.recurrent = cell(width, width, num_layers=args.predictor_layers, batch_first=True)
        self.head = nn.Linear(width, args.code_dim)

    def forward(self, place_code, actions, self_motion):
        code_at_step, action_at_step = place_code[:, :-1], actions[:, :-1]
        self_motion_to_next_step = self_motion[:, 1:]
        inputs = [code_at_step]
        if "action" in self.motion_inputs:
            inputs.append(self.action_embedding(action_at_step))
        if "self_motion" in self.motion_inputs:
            inputs.append(self_motion_to_next_step)
        hidden_state, _ = self.recurrent(torch.tanh(self.input_layer(torch.cat(inputs, dim=-1))))
        return self.head(hidden_state)


def build_model(args, num_actions=3):
    model = nn.ModuleDict({"encoder": Encoder(args), "predictor": Predictor(args, num_actions)})
    model = model.to(args.device)
    return model, copy.deepcopy(model["encoder"]).requires_grad_(False)


def forward(model, target_encoder, batch):
    place_code = model["encoder"](batch["visual_latent"])
    predicted_code = model["predictor"](place_code, batch["actions"], batch["self_motion"])
    with torch.no_grad():
        target_code = target_encoder(batch["visual_latent"])
    return place_code, predicted_code, target_code


# Losses


def prediction_loss(prediction, target_code, target_offset):
    target_at_offset = target_code[:, target_offset : target_offset + prediction.shape[1]]
    return 1.0 - F.cosine_similarity(prediction, target_at_offset, dim=-1).mean()


def centered_units(place_code):
    steps = place_code.reshape(-1, place_code.shape[-1])
    return steps - steps.mean(dim=0, keepdim=True)


def variance_regularizer(place_code, minimum_std):
    std = torch.sqrt(centered_units(place_code).var(dim=0) + 1e-4)
    return F.relu(minimum_std - std).mean()


def covariance_regularizer(place_code):
    centered = centered_units(place_code)
    covariance = centered.T @ centered / (len(centered) - 1)
    off_diagonal = covariance - torch.diag(covariance.diagonal())
    return off_diagonal.pow(2).sum() / len(covariance)


def compute_losses(place_code, predicted_code, target_code, args):
    prediction = predicted_code if args.prediction_from == "predictor" else place_code
    losses = {
        "prediction": prediction_loss(prediction, target_code, args.target_offset),
        "variance": variance_regularizer(place_code, args.minimum_std),
        "covariance": covariance_regularizer(place_code),
    }
    regularizers = (
        args.variance_weight * losses["variance"] + args.covariance_weight * losses["covariance"]
    )
    losses["total"] = args.prediction_weight * losses["prediction"] + regularizers
    return losses


# Training


def build_optimizer(model, args):
    no_decay = [module.bias for module in model.modules() if isinstance(module, nn.Linear)]
    no_decay += [module.weight for module in model.modules() if isinstance(module, nn.Embedding)]
    skipped = {id(parameter) for parameter in no_decay}
    decay = [parameter for parameter in model.parameters() if id(parameter) not in skipped]
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.Adam(groups, lr=args.learning_rate)


def training_step(model, target_encoder, optimizer, batch, args):
    losses = compute_losses(*forward(model, target_encoder, batch), args)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
    optimizer.step()
    with torch.no_grad():
        online_parameters = model["encoder"].parameters()
        for target, online in zip(target_encoder.parameters(), online_parameters, strict=True):
            target.mul_(args.ema_decay).add_(online, alpha=1.0 - args.ema_decay)
    return {name: loss.item() for name, loss in losses.items()}


def episode_batch(data, episode_ids, device):
    keys = ("visual_latent", "actions", "self_motion")
    return {key: torch.as_tensor(data[key][episode_ids], device=device) for key in keys}


def train_place_cell_model(args, data, train_ids):
    model, target_encoder = build_model(args)
    optimizer = build_optimizer(model, args)
    print(model)
    history = []
    for epoch in range(args.epochs):
        model.train()
        steps = []
        for batch in torch.randperm(len(train_ids)).split(args.batch_size):
            episodes = episode_batch(data, train_ids[batch.numpy()], args.device)
            steps.append(training_step(model, target_encoder, optimizer, episodes, args))
        history.append({name: np.mean([step[name] for step in steps]) for name in steps[0]})
        losses = ", ".join(f"{name} {value:.4f}" for name, value in history[-1].items())
        print(f"epoch {epoch + 1}/{args.epochs}: {losses}", flush=True)
    return model, history


# Measures

SPATIAL_BINS = 60


def place_codes(model, data, episode_ids, device):
    model.eval()
    with torch.inference_mode():
        latents = [
            torch.as_tensor(data["visual_latent"][ids], device=device)
            for ids in chunks(episode_ids, 8)
        ]
        return torch.cat([model["encoder"](latent).cpu() for latent in latents]).numpy()


def arena_edges():
    edges = []
    for low, high in ARENA:
        padding = (high - low) * 0.04
        edges.append(np.linspace(low - padding, high + padding, SPATIAL_BINS + 1, dtype=np.float32))
    return edges


def spatial_bins(position):
    column, row = (
        np.clip(np.searchsorted(edges, position[..., axis], side="right") - 1, 0, SPATIAL_BINS - 1)
        for axis, edges in enumerate(arena_edges())
    )
    return row * SPATIAL_BINS + column


def smooth(maps, sigma=0.3):
    weights = np.exp(-0.5 * (np.arange(-1, 2) / sigma) ** 2)
    weights /= weights.sum()
    rows = np.pad(maps, [(0, 0)] * (maps.ndim - 2) + [(1, 1), (0, 0)], mode="edge")
    maps = sum(weight * rows[..., k : k + maps.shape[-2], :] for k, weight in enumerate(weights))
    columns = np.pad(maps, [(0, 0)] * (maps.ndim - 1) + [(1, 1)], mode="edge")
    return sum(weight * columns[..., k : k + maps.shape[-1]] for k, weight in enumerate(weights))


def skaggs_information(activity, occupancy):
    rate_maps = activity / np.where(occupancy >= 1e-6, occupancy, np.nan)
    probability = occupancy / occupancy.sum()
    mean_rate = np.nansum(probability * rate_maps, axis=(1, 2))
    active = mean_rate > 1e-10
    normalized = np.clip(rate_maps / np.where(active, mean_rate, 1.0)[:, None, None], 1e-10, None)
    information = np.nansum(probability * normalized * np.log2(normalized), axis=(1, 2))
    return np.where(active, information, 0.0), rate_maps


def rate_maps_and_spatial_information(codes, position, shuffles, seed=0):
    positive = np.maximum(codes, 0.0)
    episodes, steps, units = positive.shape
    bins = SPATIAL_BINS * SPATIAL_BINS
    step_bins = spatial_bins(position)
    visits = np.bincount(step_bins.ravel(), minlength=bins).reshape(SPATIAL_BINS, SPATIAL_BINS)
    occupancy = smooth(visits.astype(np.float64))
    episode, step, unit = np.nonzero(positive)
    values = positive[episode, step, unit]

    def information(shifts):
        shifted_bins = step_bins[episode, (step + shifts[episode]) % steps]
        activity = np.bincount(unit * bins + shifted_bins, values, units * bins)
        activity = smooth(activity.reshape(units, SPATIAL_BINS, SPATIAL_BINS))
        return skaggs_information(activity, occupancy)

    observed, rate_maps = information(np.zeros(episodes, dtype=np.int64))
    rng = np.random.default_rng(seed)
    minimum_shift = max(1, int(steps * 0.05))
    null = []
    for _ in range(shuffles):
        shifts = [rng.integers(minimum_shift, steps - minimum_shift + 1) for _ in range(episodes)]
        null.append(information(np.array(shifts))[0])
    return {
        "rate_maps": rate_maps.astype(np.float32),
        "visits": visits,
        "occupancy": occupancy,
        "spatial_information": observed,
        "spatial_information_above_null": np.maximum(observed - np.percentile(null, 95, 0), 0.0),
    }


def decoding_rows(codes, position, first_step=15):
    rows = codes[:, first_step:].reshape(-1, codes.shape[-1])
    return rows, position[:, first_step:].reshape(-1, 2)


def ridge_decoding_rmse(train, validation, test):
    train_codes, train_position = train
    code_mean = train_codes.mean(0, dtype=np.float64)
    position_mean = train_position.mean(0, dtype=np.float64)
    gram, cross = 0.0, 0.0
    for block, targets in zip(chunks(train_codes), chunks(train_position), strict=True):
        block = block - code_mean
        gram, cross = gram + block.T @ block, cross + block.T @ (targets - position_mean)
    scale = np.sqrt(gram.diagonal() / len(train_codes))
    scale[scale == 0] = 1.0
    gram, cross = gram / np.outer(scale, scale), cross / scale[:, None]

    def rmse(weights, split):
        codes, position = split
        squared = sum(
            (((block - code_mean) @ weights + position_mean - targets) ** 2).sum()
            for block, targets in zip(chunks(codes), chunks(position), strict=True)
        )
        return np.sqrt(squared / position.size)

    candidates = [
        np.linalg.solve(gram + alpha * np.eye(len(gram)), cross) / scale[:, None]
        for alpha in 10.0 ** np.arange(-6, 4)
    ]
    best = min(candidates, key=lambda weights: rmse(weights, validation))
    return rmse(best, test)


# Figures

RATE_MAP_COLORS = colormaps["viridis"].with_extremes(bad="#eee4cc")


def draw_rate_map(axis, rate_map, visits, walls, title, fontsize):
    x_edges, z_edges = arena_edges()
    extent = (x_edges[0], x_edges[-1], z_edges[0], z_edges[-1])
    if rate_map is not None:
        image = np.ma.masked_where(visits == 0, rate_map)
        axis.imshow(image, origin="lower", extent=extent, cmap=RATE_MAP_COLORS, vmin=0)
    axis.add_collection(LineCollection(walls, colors="0.45", linewidths=0.6))
    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
    axis.set_title(title, fontsize=fontsize, pad=2)
    axis.set_axis_off()


def save_rate_map_grid(path, units, columns, tile_inches, fontsize, measures, silent, walls):
    rows = math.ceil(len(units) / columns)
    height = rows * tile_inches * 1.5 + 0.8
    figure = Figure(figsize=(columns * tile_inches, height))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.1 / height, top=1 - 0.7 / height)
    title = "Rate maps by spatial information above the shift null (bits); beige: never visited"
    figure.suptitle(title, y=1 - 0.25 / height, fontsize=11)
    axes = figure.subplots(rows, columns, gridspec_kw=dict(wspace=0.08, hspace=0.3)).flat
    for axis, unit in zip(axes, units, strict=False):
        above_null = measures["spatial_information_above_null"][unit]
        title = f"unit {unit}: silent" if silent[unit] else f"unit {unit}: {above_null:.2f}"
        rate_map = None if silent[unit] else measures["rate_maps"][unit]
        draw_rate_map(axis, rate_map, measures["visits"], walls, title, fontsize)
    for axis in axes[len(units) :]:
        axis.set_axis_off()
    figure.savefig(path, dpi=120)


def save_activity_figure(output_dir, episode, code, position, rate_maps, walls):
    active_units = np.flatnonzero(np.any(code != 0, axis=0))
    peak_row = np.nanargmax(rate_maps.reshape(len(rate_maps), -1), axis=1) // SPATIAL_BINS
    order = active_units[np.argsort(peak_row[active_units], kind="stable")]
    z_edges = arena_edges()[1]
    peak_z = (z_edges[:-1] + z_edges[1:])[peak_row[order]] / 2
    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplot_mosaic(
        [["raster", "floor"], ["position", "floor"]], width_ratios=[4, 1], height_ratios=[3, 1]
    )
    raster = (code[:, order] != 0).T
    extent = (0, len(code), -0.5, len(order) - 0.5)
    axes["raster"].imshow(raster, aspect="auto", origin="lower", extent=extent, cmap="Greys")
    ticks = np.linspace(0, len(order) - 1, 6).round().astype(int)
    axes["raster"].set_yticks(ticks, [f"{peak_z[tick]:.0f}" for tick in ticks])
    axes["raster"].set(title=f"Active units along test episode {episode}")
    axes["raster"].set(ylabel="units, by north-south position of their rate-map peak")
    axes["position"].plot(position[:, 1], color="black", linewidth=0.8)
    axes["position"].set(xlim=(0, len(code)), xlabel="step", ylabel="agent, north-south")
    axes["floor"].add_collection(LineCollection(walls, colors="0.4", linewidths=0.8))
    axes["floor"].scatter(*position.T, c=np.arange(len(position)), s=2, cmap="viridis")
    axes["floor"].set(aspect="equal", title="path, colored by step")
    axes["floor"].set_axis_off()
    figure.savefig(output_dir / "activity_episode.png", dpi=150)


def save_training_curves(output_dir, history):
    figure = Figure(figsize=(12, 3), layout="constrained")
    for axis, name in zip(figure.subplots(1, len(history[0])), history[0], strict=True):
        axis.plot(np.arange(1, len(history) + 1), [epoch[name] for epoch in history])
        axis.set(title=f"{name} loss", xlabel="epoch")
    figure.savefig(output_dir / "training_curves.png", dpi=150)


# Main


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    walls = collect_dataset(args)
    split = split_episodes(args.episodes, args.seed)
    autoencoder = train_autoencoder(
        args, vision_frames(args, split["train"]), vision_frames(args, split["validation"])
    )
    data = encode_dataset(args, autoencoder)
    model, history = train_place_cell_model(args, data, split["train"])
    measured = {name: episode_ids[: args.measure_episodes] for name, episode_ids in split.items()}
    codes = {name: place_codes(model, data, ids, args.device) for name, ids in measured.items()}
    positions = {name: data["position"][ids] for name, ids in measured.items()}
    rows = [decoding_rows(codes[name], positions[name]) for name in ("train", "validation", "test")]
    measures = rate_maps_and_spatial_information(
        codes["test"], positions["test"], args.null_shuffles
    )
    silent = ~np.any(codes["test"] != 0, axis=(0, 1))
    above_null = measures["spatial_information_above_null"]
    print(f"linear decoding RMSE: {ridge_decoding_rmse(*rows):.3f} world units")
    print(f"spatial information above null: {above_null.mean():.3f} bits, mean over all units")
    print(f"silent units: {silent.mean():.1%} of {len(silent)}")
    x_edges, z_edges = arena_edges()
    extras = dict(silent=silent, x_edges=x_edges, z_edges=z_edges, walls=walls)
    np.savez_compressed(args.output_dir / "rate_maps.npz", **measures, **extras)
    order = np.lexsort((-above_null, silent))
    grids = {"rate_maps.png": (order[:16], 4, 2.0, 9), "rate_maps_all.png": (order, 16, 1.0, 6)}
    for name, grid in grids.items():
        save_rate_map_grid(args.output_dir / name, *grid, measures, silent, walls)
    first_test_episode = (measured["test"][0], codes["test"][0], positions["test"][0])
    save_activity_figure(args.output_dir, *first_test_episode, measures["rate_maps"], walls)
    save_training_curves(args.output_dir, history)


if __name__ == "__main__":
    main()
