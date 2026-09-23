"""Environment construction helpers for downstream training."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.downstream_schema import DownstreamRunConfig
from placecell_research.utils.device import resolve_device

from .online_env import DownstreamNavigationEnv
from .runtime import build_feature_extractor, build_goal_place_code_runtime

_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_TORCH_THREADS_CONFIGURED = False


def configure_downstream_process_threads() -> None:
    for variable_name in _THREAD_LIMIT_ENV_VARS:
        os.environ.setdefault(variable_name, "1")
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _TORCH_THREADS_CONFIGURED = True


def _is_macos_miniworld(config: DownstreamRunConfig) -> bool:
    return sys.platform == "darwin" and str(config.environment.kind).strip().lower() == "miniworld"


def _build_vec_env(
    *,
    config: DownstreamRunConfig,
    env_factories,
    dummy_vec_env_type,
    subproc_vec_env_type,
):
    if len(env_factories) > 1 and _is_macos_miniworld(config):
        print(
            "[downstream] using DummyVecEnv because macOS MiniWorld + SubprocVecEnv "
            "can deadlock inside Apple's Metal shader cache",
            file=sys.stderr,
            flush=True,
        )
        return dummy_vec_env_type(env_factories)
    if len(env_factories) > 1:
        return subproc_vec_env_type(env_factories, start_method="spawn")
    return dummy_vec_env_type(env_factories)


def _resolve_training_device(config: DownstreamRunConfig, *, force_raw_pixels: bool) -> str:
    if force_raw_pixels:
        return "cpu"
    return str(resolve_device(config.training.device))


def build_env_factory(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    observation_name: str,
    seed: int,
    force_raw_pixels: bool = False,
    feature_goal_candidate_positions_xy: list[list[float]] | None = None,
) -> Callable[[], DownstreamNavigationEnv]:
    del observation_name
    artifact_registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    device = _resolve_training_device(config, force_raw_pixels=force_raw_pixels)
    runtime_observation_config = (
        replace(
            config.observation,
            mode="raw_pixels",
            feature_sources=[],
        )
        if force_raw_pixels
        else config.observation
    )
    goal_place_code_runtime = None
    if "goal_place_code" in runtime_observation_config.feature_sources:
        goal_place_code_runtime = build_goal_place_code_runtime(
            artifact_registry=artifact_registry,
            models=config.models,
            goal_code=config.goal_code,
            device=device,
        )

    def _make_env() -> DownstreamNavigationEnv:
        configure_downstream_process_threads()
        feature_extractor = None
        if (
            runtime_observation_config.mode != "raw_pixels"
            or runtime_observation_config.feature_sources
        ):
            feature_extractor = build_feature_extractor(
                artifact_registry=artifact_registry,
                models=config.models,
                observation=runtime_observation_config,
                env_id=config.environment.env_id,
                goal_candidate_positions_xy=(
                    config.goal_task.candidate_positions_xy
                    if feature_goal_candidate_positions_xy is None
                    else feature_goal_candidate_positions_xy
                ),
                goal_rbf_sigma=config.goal_task.rbf_sigma,
                device=device,
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
            observation_mode=runtime_observation_config.mode,
            seed=seed,
        )

    return _make_env


def build_curriculum_applied_env_factory(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    observation_name: str,
    seed: int,
    phase,
) -> Callable[[], DownstreamNavigationEnv]:
    base_factory = build_env_factory(
        repo_root=repo_root,
        config=config,
        observation_name=observation_name,
        seed=seed,
    )

    def _make_env() -> DownstreamNavigationEnv:
        env = base_factory()
        env.apply_curriculum_phase(phase)
        return env

    return _make_env


def build_train_vec_env(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    dummy_vec_env_type,
    subproc_vec_env_type,
    vec_monitor_type,
):
    n_envs = max(1, int(config.training.n_envs))
    use_shared_feature_runtime = config.observation.mode == "feature_vector" and n_envs > 1
    if use_shared_feature_runtime:
        from .shared_feature_vec_env import (
            SharedFeatureVecEnvWrapper,
            build_shared_feature_pipeline,
        )

        artifact_registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
        device = _resolve_training_device(config, force_raw_pixels=False)
        raw_env_factories = [
            build_env_factory(
                repo_root=repo_root,
                config=config,
                observation_name=f"train_{index}",
                seed=int(config.seed) + index,
                force_raw_pixels=True,
            )
            for index in range(n_envs)
        ]
        base_vec_env = _build_vec_env(
            config=config,
            env_factories=raw_env_factories,
            dummy_vec_env_type=dummy_vec_env_type,
            subproc_vec_env_type=subproc_vec_env_type,
        )
        shared_pipeline = build_shared_feature_pipeline(
            artifact_registry=artifact_registry,
            models=config.models,
            observation=config.observation,
            env_id=config.environment.env_id,
            goal_candidate_positions_xy=config.goal_task.candidate_positions_xy,
            goal_rbf_sigma=config.goal_task.rbf_sigma,
            device=device,
        )
        return vec_monitor_type(SharedFeatureVecEnvWrapper(base_vec_env, shared_pipeline))

    env_factories = []
    for index in range(n_envs):
        env_factories.append(
            build_env_factory(
                repo_root=repo_root,
                config=config,
                observation_name=f"train_{index}",
                seed=int(config.seed) + index,
            )
        )
    return vec_monitor_type(
        _build_vec_env(
            config=config,
            env_factories=env_factories,
            dummy_vec_env_type=dummy_vec_env_type,
            subproc_vec_env_type=subproc_vec_env_type,
        )
    )


def can_use_parallel_vector_eval(config: DownstreamRunConfig) -> bool:
    if _is_macos_miniworld(config):
        return False
    if config.eval_candidate_positions_xy:
        return False
    goal_count = len(config.goal_task.candidate_positions_xy)
    has_supported_goal_schedule = goal_count <= 1 or str(config.goal_task.schedule) in {
        "random",
        "cycle",
    }
    return int(config.training.n_envs) > 1 and has_supported_goal_schedule


def build_raw_eval_vec_env(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    dummy_vec_env_type,
    subproc_vec_env_type,
    vec_monitor_type,
    episodes: int,
):
    n_envs = max(1, min(int(config.training.n_envs), int(episodes)))
    env_factories = []
    for index in range(n_envs):
        env_factories.append(
            build_env_factory(
                repo_root=repo_root,
                config=config,
                observation_name=f"eval_{index}",
                seed=int(config.seed) + 10_000 + index,
            )
        )
    base_vec_env = _build_vec_env(
        config=config,
        env_factories=env_factories,
        dummy_vec_env_type=dummy_vec_env_type,
        subproc_vec_env_type=subproc_vec_env_type,
    )
    return vec_monitor_type(base_vec_env)


def build_shared_eval_vec_env(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    dummy_vec_env_type,
    subproc_vec_env_type,
    vec_monitor_type,
    episodes: int,
):
    from .shared_feature_vec_env import (
        SharedFeatureVecEnvWrapper,
        build_shared_feature_pipeline,
    )

    n_envs = max(1, min(int(config.training.n_envs), int(episodes)))
    raw_env_factories = [
        build_env_factory(
            repo_root=repo_root,
            config=config,
            observation_name=f"eval_vector_{index}",
            seed=int(config.seed) + 10_000 + index,
            force_raw_pixels=True,
        )
        for index in range(n_envs)
    ]
    base_vec_env = _build_vec_env(
        config=config,
        env_factories=raw_env_factories,
        dummy_vec_env_type=dummy_vec_env_type,
        subproc_vec_env_type=subproc_vec_env_type,
    )
    artifact_registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    device = _resolve_training_device(config, force_raw_pixels=False)
    shared_pipeline = build_shared_feature_pipeline(
        artifact_registry=artifact_registry,
        models=config.models,
        observation=config.observation,
        env_id=config.environment.env_id,
        goal_candidate_positions_xy=config.goal_task.candidate_positions_xy,
        goal_rbf_sigma=config.goal_task.rbf_sigma,
        device=device,
    )
    return vec_monitor_type(SharedFeatureVecEnvWrapper(base_vec_env, shared_pipeline))


def build_eval_env_factories(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
) -> list[Callable[[], DownstreamNavigationEnv]]:
    goal_positions = (
        list(config.eval_candidate_positions_xy)
        if config.eval_candidate_positions_xy
        else list(config.goal_task.candidate_positions_xy)
    )
    if not goal_positions:
        return [
            build_env_factory(
                repo_root=repo_root,
                config=config,
                observation_name="eval",
                seed=int(config.seed) + 10_000,
            )
        ]

    eval_factories: list[Callable[[], DownstreamNavigationEnv]] = []
    for goal_index, goal_position_xy in enumerate(goal_positions):
        fixed_goal_config = replace(
            config,
            goal_task=replace(
                config.goal_task,
                candidate_positions_xy=[list(goal_position_xy)],
                schedule="fixed",
                change_interval_episodes=1,
                initial_goal_index=0,
            ),
        )
        eval_factories.append(
            build_env_factory(
                repo_root=repo_root,
                config=fixed_goal_config,
                observation_name=f"eval_goal_{goal_index}",
                seed=int(config.seed) + 10_000 + goal_index,
                feature_goal_candidate_positions_xy=list(config.goal_task.candidate_positions_xy),
            )
        )
    return eval_factories


def build_curriculum_eval_factories(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    phase,
) -> list[Callable[[], DownstreamNavigationEnv]]:
    return [
        build_curriculum_applied_env_factory(
            repo_root=repo_root,
            config=config,
            observation_name="eval_curriculum",
            seed=int(config.seed) + 30_000,
            phase=phase,
        )
    ]


def extract_vector_env_envs(vec_env) -> list[DownstreamNavigationEnv]:
    current = vec_env
    for _ in range(8):
        envs = getattr(current, "envs", None)
        if envs is not None:
            return list(envs)
        if hasattr(current, "venv"):
            current = current.venv
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    raise RuntimeError("Could not unwrap vectorized training envs for curriculum application.")


def apply_curriculum_phase_to_vector_env(vec_env, phase) -> None:
    current = vec_env
    for _ in range(8):
        env_method = getattr(current, "env_method", None)
        if callable(env_method):
            env_method("apply_curriculum_phase", phase)
            return
        next_current = getattr(current, "venv", None)
        if next_current is None:
            next_current = getattr(current, "env", None)
        if next_current is None:
            break
        current = next_current
    for env in extract_vector_env_envs(vec_env):
        env.apply_curriculum_phase(phase)
