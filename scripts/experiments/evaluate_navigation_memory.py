"""Evaluate frozen navigation policies on saved starts with exploration/state controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from placecell_research.config import load_downstream_run_config
from placecell_research.downstream.env_builders import (
    build_eval_env_factories,
    configure_downstream_process_threads,
)
from placecell_research.downstream.feature_sources import PlaceCodeFeatureSource
from placecell_research.downstream.sb3_algorithms import import_sb3_runtime
from placecell_research.utils.seeds import seed_everything


def epsilon_action(greedy: int, action_count: int, epsilon: float, rng) -> tuple[int, bool]:
    explore = bool(rng.random() < epsilon)
    return (int(rng.integers(action_count)) if explore else greedy), explore


def reset_representation_before_extract(source: PlaceCodeFeatureSource) -> None:
    extract = source.extract

    def extract_without_history(context, previous_action):
        source.reset()
        return extract(context, previous_action)

    source.extract = extract_without_history


def evaluation_load_objects(model_path, env):
    """Restore spaces without deserializing NumPy-version-specific rollout buffers."""
    with zipfile.ZipFile(model_path) as archive:
        saved = json.loads(archive.read("data"))
    observation = saved["observation_space"]
    action = saved["action_space"]
    if (
        tuple(observation["_shape"]) != env.observation_space.shape
        or np.dtype(observation["dtype"]) != env.observation_space.dtype
        or int(action["n"]) != env.action_space.n
        or int(action["start"]) != env.action_space.start
    ):
        raise ValueError("Saved policy spaces do not match the evaluation environment.")
    return {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "_last_obs": None,
        "_last_original_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": deque(),
    }


def evaluate(args):
    config = load_downstream_run_config(args.config)
    config.training.device = "cpu"
    if config.models.place_model_checkpoint is None:
        config.models.place_model_checkpoint = "best_primary"
    if config.training.algorithm not in {"ppo", "dqn"}:
        raise ValueError("This experiment accepts the saved feedforward PPO/DQN policies only.")
    seed_everything(config.seed)
    configure_downstream_process_threads()
    runtime = import_sb3_runtime()
    envs = build_eval_env_factories(repo_root=args.repo, config=config)
    if len(envs) != 1:
        raise ValueError("Expected a single fixed goal per saved policy.")
    env = envs[0]()
    model = getattr(runtime, config.training.algorithm).load(
        str(args.model), device="cpu", custom_objects=evaluation_load_objects(args.model, env)
    )
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    references = [json.loads(line) for line in args.starts.read_text().splitlines()]
    if len(references) != 30:
        raise ValueError("Expected the original 30 evaluation trajectories.")
    reset_sources = []
    if args.reset_each_step:
        reset_sources = [
            s for s in env.feature_extractor.sources if isinstance(s, PlaceCodeFeatureSource)
        ]
        if len(reset_sources) != 1:
            raise ValueError("Reset treatment requires exactly one learned representation source.")
        reset_representation_before_extract(reset_sources[0])
    adapter_reset = env._adapter.reset
    active_reference = None

    def reset_at_saved_start(seed=None):
        observation = adapter_reset(seed=seed)
        native = env._adapter._env.unwrapped
        xy = np.asarray(active_reference["start_position_xy"], dtype=np.float32)
        heading = float(active_reference["headings"][0])
        native.agent.pos[[0, 2]] = xy
        native.agent.dir = heading
        rgb = env._adapter._normalize_rgb(np.asarray(native.render_obs()))
        return replace(observation, position_xy=xy, heading=heading, modalities={"rgb": rgb})

    env._adapter.reset = reset_at_saved_start
    episodes = []
    try:
        for index, reference in enumerate(references[: args.episodes]):
            active_reference = reference
            obs, info = env.reset(seed=config.seed + 10000 + index)
            np.testing.assert_allclose(
                info["position_xy"], reference["start_position_xy"], atol=1e-6
            )
            np.testing.assert_allclose(
                info["goal_position_xy"], reference["goal_position_xy"], atol=1e-6
            )
            rng = np.random.default_rng(500000 + config.seed * 1000 + index)
            poses = []
            actions = []
            explored = []
            total = 0.0
            changes = []
            for _step in range(config.environment.episode_length):
                action, _ = model.predict(obs, deterministic=True)
                action, random_action = epsilon_action(
                    int(np.asarray(action).item()), env.action_space.n, args.epsilon, rng
                )
                next_obs, reward, terminated, truncated, info = env.step(action)
                if isinstance(obs, np.ndarray) and isinstance(next_obs, np.ndarray):
                    changes.append(
                        float(np.linalg.norm(next_obs.astype(float) - obs.astype(float)))
                    )
                obs = next_obs
                total += float(reward)
                poses.append([*map(float, info["position_xy"]), float(info["heading"])])
                actions.append(action)
                explored.append(random_action)
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "episode": index + 1,
                    "start_xy": reference["start_position_xy"],
                    "start_heading": reference["headings"][0],
                    "goal_xy": reference["goal_position_xy"],
                    "steps": len(actions),
                    "success": bool(info["is_success"]),
                    "return": total,
                    "positions_and_heading": poses,
                    "actions": actions,
                    "exploration_steps": explored,
                    "observation_change_norm": changes,
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".progress.json").write_text(
                json.dumps({"completed_episodes": len(episodes), "episodes": episodes})
            )
            print(
                f"{args.output.stem}: episode {index + 1}/{args.episodes}, "
                f"success={info['is_success']}",
                flush=True,
            )
    finally:
        env.close()
    after = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Saved policy file changed during evaluation.")
    result = {
        "complete": True,
        "training_performed": False,
        "epsilon": args.epsilon,
        "place_model_checkpoint": config.models.place_model_checkpoint,
        "reset_representation_each_step": args.reset_each_step,
        "reset_source_count": len(reset_sources),
        "policy_sha256": before,
        "policy_load_mode": "environment_spaces_without_training_buffers",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "starts_sha256": hashlib.sha256(args.starts.read_bytes()).hexdigest(),
        "config": str(args.config),
        "model": str(args.model),
        "episodes": episodes,
        "success_rate": float(np.mean([x["success"] for x in episodes])),
        "episode_limit": config.environment.episode_length,
        "source_snapshot": str(Path(__file__).resolve().parents[2]),
    }
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    args.output.with_suffix(".progress.json").unlink()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--starts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--reset-each-step", action="store_true")
    args = p.parse_args()
    if not 0 <= args.epsilon <= 1 or not 1 <= args.episodes <= 30:
        p.error("epsilon must be in [0,1] and episodes in [1,30]")
    evaluate(args)
