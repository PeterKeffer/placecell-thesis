"""Success, looping failures and learning speed of trained navigation policies."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from placecell_research.config import load_downstream_run_config
from placecell_research.utils.repo_paths import find_repo_root

LOOP_STEPS = 64
LOOP_TOLERANCE = 1e-5
CURVE_BIN_STEPS = 100_000
SUSTAINED_BINS = 3


def absorbing_tail(success: bool, steps: int, limit: int, poses: list[list[float]]) -> bool:
    """A failure at the step limit whose last 64 poses repeat with period one or two."""
    if success or steps != limit:
        return False
    tail = np.asarray(poses[-LOOP_STEPS:])
    if len(tail) != LOOP_STEPS:
        return False
    period_one = np.allclose(tail, tail[-1], atol=LOOP_TOLERANCE, rtol=0)
    period_two = np.allclose(tail[2:], tail[:-2], atol=LOOP_TOLERANCE, rtol=0)
    return bool(period_one or period_two)


def learning_curve(history: list[dict], total_timesteps: int) -> dict[str, float]:
    """Training success per 100,000 steps and the first step of three bins above 50% and 80%."""
    timestep = np.array([entry["timestep"] for entry in history], dtype=np.int64)
    success = np.array([entry["success_rate"] for entry in history], dtype=np.float64)
    edges = np.arange(0, total_timesteps + 1, CURVE_BIN_STEPS)
    curve = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        take = (timestep > low) & (timestep <= high)
        curve.append(success[take].mean() if take.any() else np.nan)
    curve = np.asarray(curve)

    def sustained(threshold: float) -> float:
        for index in range(len(curve) - SUSTAINED_BINS + 1):
            if np.all(curve[index : index + SUSTAINED_BINS] >= threshold):
                return float(edges[index + 1])
        return float("nan")

    def mean_between(low: int, high: int) -> float:
        take = (timestep > low) & (timestep <= high)
        return float(success[take].mean()) if take.any() else float("nan")

    return {
        "training_success_first_million": mean_between(0, 1_000_000),
        "training_success_last_half_million": mean_between(
            total_timesteps - 500_000, total_timesteps
        ),
        "steps_to_sustained_50_percent": sustained(0.5),
        "steps_to_sustained_80_percent": sustained(0.8),
    }


def _load_objects(model_path: Path, env) -> dict:
    """Restore spaces from the environment instead of the saved training buffers."""
    with zipfile.ZipFile(model_path) as archive:
        saved = json.loads(archive.read("data"))
    observation = saved["observation_space"]
    action = saved["action_space"]
    shape = observation.get("_shape")
    if (
        (shape is not None and tuple(shape) != env.observation_space.shape)
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


def evaluate_with_exploration(
    config_path: Path, model_path: Path, starts: list[dict], epsilon: float
) -> list[dict]:
    """Re-run the saved start poses, taking a uniformly random action with probability epsilon."""
    from placecell_research.downstream.env_builders import (
        build_eval_env_factories,
        configure_downstream_process_threads,
    )
    from placecell_research.downstream.sb3_algorithms import import_sb3_runtime
    from placecell_research.utils.seeds import seed_everything

    config = load_downstream_run_config(config_path)
    config.training.device = "cpu"
    seed_everything(config.seed)
    configure_downstream_process_threads()
    runtime = import_sb3_runtime()
    (factory,) = build_eval_env_factories(repo_root=find_repo_root(config_path), config=config)
    env = factory()
    model = getattr(runtime, config.training.algorithm).load(
        str(model_path), device="cpu", custom_objects=_load_objects(model_path, env)
    )
    adapter_reset = env._adapter.reset
    active = {}

    def reset_at_saved_start(seed=None):
        observation = adapter_reset(seed=seed)
        native = env._adapter._env.unwrapped
        xy = np.asarray(active["start_position_xy"], dtype=np.float32)
        heading = float(active["headings"][0])
        native.agent.pos[[0, 2]] = xy
        native.agent.dir = heading
        rgb = env._adapter._normalize_rgb(np.asarray(native.render_obs()))
        return replace(observation, position_xy=xy, heading=heading, modalities={"rgb": rgb})

    env._adapter.reset = reset_at_saved_start
    episodes = []
    try:
        for index, reference in enumerate(starts):
            active.clear()
            active.update(reference)
            obs, info = env.reset(seed=config.seed + 10000 + index)
            np.testing.assert_allclose(
                info["position_xy"], reference["start_position_xy"], atol=1e-6
            )
            np.testing.assert_allclose(
                info["goal_position_xy"], reference["goal_position_xy"], atol=1e-6
            )
            rng = np.random.default_rng(500000 + config.seed * 1000 + index)
            poses = []
            for _step in range(config.environment.episode_length):
                action, _ = model.predict(obs, deterministic=True)
                action = int(np.asarray(action).item())
                if rng.random() < epsilon:
                    action = int(rng.integers(env.action_space.n))
                obs, _reward, terminated, truncated, info = env.step(action)
                poses.append([*map(float, info["position_xy"]), float(info["heading"])])
                if terminated or truncated:
                    break
            episodes.append({"success": bool(info["is_success"]), "poses": poses})
    finally:
        env.close()
    return episodes


def _final_episodes(results: Path) -> list[dict]:
    path = results / "eval_trajectory_diagnostics/final/eval_trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda row: row["episode"])


def navigation_row(run: Path, *, epsilon: float) -> dict:
    """One policy: greedy and optional epsilon-greedy success on the saved starts, loops, curve."""
    results = run / "results/downstream_train"
    config_path = run / "manifests/resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    metrics = json.loads((results / "metrics.json").read_text())
    limit = int(config["environment"]["episode_length"])
    greedy = _final_episodes(results)
    greedy_success = np.array([episode["success"] for episode in greedy], dtype=bool)
    greedy_loops = sum(
        absorbing_tail(
            episode["success"],
            len(episode["positions_xy"]) - 1,
            limit,
            [
                [*xy, heading]
                for xy, heading in zip(
                    episode["positions_xy"][1:], episode["headings"][1:], strict=True
                )
            ],
        )
        for episode in greedy
    )
    row = {
        "condition": config["name"],
        "seed": int(config["seed"]),
        "run": run.name,
        "algorithm": config["training"]["algorithm"],
        "feature_sources": ";".join(config["observation"]["feature_sources"]),
        "representation_source": config["models"]["place_representation_source"],
        "goal_xy": json.dumps(config["goal_task"]["candidate_positions_xy"]),
        "evaluated_episodes": len(greedy),
        "greedy_success_rate": float(greedy_success.mean()),
        "greedy_failures": int((~greedy_success).sum()),
        "greedy_looping_failures": int(greedy_loops),
        **learning_curve(
            metrics["training_episode_history"], int(config["training"]["total_timesteps"])
        ),
    }
    if epsilon > 0:
        model_path = results / "final_model.zip"
        before = hashlib.sha256(model_path.read_bytes()).hexdigest()
        print(f"[navigation] epsilon {epsilon} on {run.name}", file=sys.stderr, flush=True)
        explored = evaluate_with_exploration(config_path, model_path, greedy, epsilon)
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != before:
            raise RuntimeError("Saved policy file changed during evaluation.")
        success = np.array([episode["success"] for episode in explored], dtype=bool)
        row |= {
            "epsilon": epsilon,
            "epsilon_success_rate": float(success.mean()),
            "epsilon_recovered_vs_greedy": int((success & ~greedy_success).sum()),
            "epsilon_lost_vs_greedy": int((~success & greedy_success).sum()),
            "epsilon_looping_failures": sum(
                absorbing_tail(e["success"], len(e["poses"]), limit, e["poses"]) for e in explored
            ),
        }
    return row


def expand_navigation_runs(paths: list[Path]) -> list[Path]:
    """Run folders as given; a folder of runs gives its newest finished run per config and seed."""
    runs: list[Path] = []
    for path in paths:
        if (path / "results/downstream_train").is_dir():
            runs.append(path)
            continue
        newest: dict[tuple[str, int], Path] = {}
        for run in sorted(path.iterdir()):
            metrics = run / "results/downstream_train/metrics.json"
            config_path = run / "manifests/resolved_config.yaml"
            if not metrics.is_file() or not config_path.is_file():
                continue
            config = yaml.safe_load(config_path.read_text())
            key = (str(config["name"]), int(config["seed"]))
            previous = newest.get(key)
            if (
                previous is None
                or metrics.stat().st_mtime
                > (previous / "results/downstream_train/metrics.json").stat().st_mtime
            ):
                newest[key] = run
        runs.extend(newest[key] for key in sorted(newest))
    return runs


def navigation_rows(runs: list[Path], *, epsilon: float) -> list[dict]:
    return [navigation_row(run, epsilon=epsilon) for run in runs]
