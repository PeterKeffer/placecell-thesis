"""Smoke rollouts for downstream navigation environments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.downstream_schema import DownstreamRunConfig

from .online_env import DownstreamNavigationEnv, RolloutSummary
from .previews import build_reconstruction_source, capture_policy_preview
from .runtime import build_feature_extractor, build_goal_place_code_runtime


def run_downstream_rollout(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    output_dir: Path,
) -> RolloutSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)

    def _make_env() -> DownstreamNavigationEnv:
        goal_place_code_runtime = None
        if "goal_place_code" in config.observation.feature_sources:
            goal_place_code_runtime = build_goal_place_code_runtime(
                artifact_registry=artifact_registry,
                models=config.models,
                goal_code=config.goal_code,
                device=config.training.device,
            )
        feature_extractor = build_feature_extractor(
            artifact_registry=artifact_registry,
            models=config.models,
            observation=config.observation,
            env_id=config.environment.env_id,
            goal_candidate_positions_xy=config.goal_task.candidate_positions_xy,
            goal_rbf_sigma=config.goal_task.rbf_sigma,
            device=config.training.device,
            goal_place_code_dim=(
                None
                if goal_place_code_runtime is None
                else int(goal_place_code_runtime.feature_dim)
            ),
        )
        return DownstreamNavigationEnv(
            environment_config=config.environment,
            goal_task_config=config.goal_task,
            feature_extractor=feature_extractor,
            goal_place_code_runtime=goal_place_code_runtime,
            observation_mode=config.observation.mode,
            seed=int(config.seed),
        )

    env = _make_env()
    preview_bundle = None

    try:
        returns = []
        successes = []
        steps_per_episode = []
        for episode_index in range(int(config.rollout.episodes)):
            observation, info = env.reset(seed=int(config.seed) + episode_index)
            total_return = 0.0
            steps = 0
            terminated = False
            truncated = False
            final_info = dict(info)
            while (
                not terminated
                and not truncated
                and steps < int(config.rollout.max_steps_per_episode)
            ):
                if config.rollout.policy == "forward" and hasattr(env.action_space, "n"):
                    action = 2 if int(env.action_space.n) > 2 else 0
                else:
                    action = env.action_space.sample()
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_return += float(reward)
                steps += 1
            returns.append(total_return)
            successes.append(float(bool(final_info.get("is_success", total_return > 0.0))))
            steps_per_episode.append(float(steps))

        summary = RolloutSummary(
            episodes=int(config.rollout.episodes),
            mean_return=float(np.mean(returns) if returns else 0.0),
            success_rate=float(np.mean(successes) if successes else 0.0),
            mean_steps=float(np.mean(steps_per_episode) if steps_per_episode else 0.0),
        )
        if config.preview.enabled:
            preview_reconstruction_source = build_reconstruction_source(
                artifact_registry=artifact_registry,
                models=config.models,
                device=config.training.device,
            )
            policy_name = str(config.rollout.policy)
            action_count = int(env.action_space.n) if hasattr(env.action_space, "n") else 1
            rollout_random = np.random.default_rng(int(config.seed) + 40_000)
            preview_bundle = capture_policy_preview(
                env_factory=_make_env,
                action_fn=(
                    (lambda _observation: 2 if action_count > 2 else 0)
                    if policy_name == "forward"
                    else (lambda _observation: int(rollout_random.integers(0, action_count)))
                ),
                env_id=config.environment.env_id,
                preview_name="rollout_preview",
                output_dir=output_dir / "previews",
                preview_config=config.preview,
                seed=int(config.seed) + 30_000,
                reconstruction_source=preview_reconstruction_source,
            )
        (output_dir / "rollout_summary.json").write_text(
            json.dumps(
                {
                    "episodes": summary.episodes,
                    "mean_return": summary.mean_return,
                    "success_rate": summary.success_rate,
                    "mean_steps": summary.mean_steps,
                    "preview": None
                    if preview_bundle is None
                    else {
                        "combined_gif_path": str(preview_bundle.combined_gif_path),
                        "rgb_gif_path": None
                        if preview_bundle.rgb_gif_path is None
                        else str(preview_bundle.rgb_gif_path),
                        "reconstruction_gif_path": (
                            None
                            if preview_bundle.reconstruction_gif_path is None
                            else str(preview_bundle.reconstruction_gif_path)
                        ),
                        "trajectory_summary_path": (
                            None
                            if preview_bundle.trajectory_summary_path is None
                            else str(preview_bundle.trajectory_summary_path)
                        ),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (output_dir / "used_hyperparameters.yaml").write_text(
            yaml.safe_dump(
                {
                    "models": asdict(config.models),
                    "goal_task": asdict(config.goal_task),
                    "goal_code": asdict(config.goal_code),
                    "observation": asdict(config.observation),
                    "preview": asdict(config.preview),
                    "rollout": asdict(config.rollout),
                },
                sort_keys=False,
            )
        )
        return summary
    finally:
        env.close()
