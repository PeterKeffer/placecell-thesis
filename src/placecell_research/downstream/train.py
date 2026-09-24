"""SB3 downstream training for navigation experiments."""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.downstream_schema import DownstreamRunConfig
from placecell_research.utils.device import resolve_device
from placecell_research.utils.memory_watchdog import read_current_rss_mb as _read_current_rss_mb
from placecell_research.utils.seeds import seed_everything

from . import env_builders as _env_builders
from .evaluation import evaluate_policy as _evaluate_policy
from .final_eval_diagnostics import FinalEvalDiagnostics
from .previews import PreviewScheduler, build_reconstruction_source, capture_policy_preview
from .progress import (
    EVALUATION_PHASE,
    POST_TRAINING_PHASE,
    TRAINING_PHASE,
    report_progress_heartbeat,
    report_progress_phase,
)
from .sb3_algorithms import build_sb3_model, import_sb3_runtime
from .spawn_curriculum import SpawnCurriculumScheduler, format_phase_transition


@dataclass(slots=True)
class DownstreamTrainResult:
    output_dir: Path
    metrics_path: Path
    final_model_path: Path | None
    best_model_path: Path | None
    interrupted_model_path: Path | None
    final_metrics: dict[str, float]
    status: str


@dataclass(slots=True)
class _TerminationState:
    requested: bool = False
    signal_number: int | None = None
    timestep: int | None = None
    checkpoint_saved: bool = False


def _import_sb3():
    return import_sb3_runtime()


def _resolve_device(device: str) -> str:
    return str(resolve_device(device))


def _training_previews_enabled(config: DownstreamRunConfig) -> bool:
    return bool(config.preview.enabled)


def _resolve_downstream_memory_watchdog_limit_mb(config: DownstreamRunConfig) -> int:
    explicit_limit_mb = int(config.training.memory_watchdog_limit_mb)
    if explicit_limit_mb > 0:
        return explicit_limit_mb
    if config.launcher.type != "slurm":
        return 0
    return int(float(config.launcher.memory_gb) * 1024.0 * 0.9)


def _raise_if_downstream_memory_watchdog_exceeded(limit_mb: int) -> None:
    if limit_mb <= 0:
        return
    rss_mb = _read_current_rss_mb()
    if rss_mb is None or rss_mb <= limit_mb:
        return
    raise MemoryError(f"Downstream RSS watchdog exceeded {limit_mb} MB (current={rss_mb:.1f} MB).")


def _default_summary_metrics() -> dict[str, float]:
    return {
        "mean_return": 0.0,
        "success_rate": 0.0,
        "mean_time_to_goal": 0.0,
    }


def _position_xy_from_info(info: object, *, key: str = "position_xy") -> list[float] | None:
    if not isinstance(info, dict) or key not in info:
        return None
    values = np.asarray(info[key], dtype=np.float32).reshape(-1)
    if values.shape[0] < 2:
        return None
    return [float(values[0]), float(values[1])]


def _reset_infos_from_vec_env(vec_env: object) -> list[object]:
    current = vec_env
    for _ in range(8):
        reset_infos = getattr(current, "reset_infos", None)
        if reset_infos is not None:
            return list(reset_infos)
        next_env = getattr(current, "venv", None)
        if next_env is None:
            next_env = getattr(current, "env", None)
        current = next_env
        if current is None:
            break
    return []


@contextmanager
def _install_sigterm_handler(termination_state: _TerminationState):
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame) -> None:
        del frame
        if termination_state.requested:
            return
        termination_state.requested = True
        termination_state.signal_number = int(signum)
        print(
            "[placecell_research] received SIGTERM, stopping downstream training cleanly",
            file=sys.stderr,
            flush=True,
        )

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _is_strictly_better_eval_metrics(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
) -> bool:
    if incumbent is None:
        return True
    comparison_keys = (
        ("success_rate", True),
        ("mean_return", True),
        ("mean_time_to_goal", False),
    )
    for metric_name, larger_is_better in comparison_keys:
        candidate_value = float(candidate.get(metric_name, 0.0))
        incumbent_value = float(incumbent.get(metric_name, 0.0))
        if candidate_value == incumbent_value:
            continue
        if larger_is_better:
            return candidate_value > incumbent_value
        return candidate_value < incumbent_value
    return False


_HEARTBEAT_INTERVAL_SECONDS = 30.0


def train_downstream_agent(
    *,
    repo_root: Path,
    config: DownstreamRunConfig,
    output_dir: Path,
    metric_logger: Callable[[dict[str, Any], int | None], None] | None = None,
    progress_reporter: Callable[[int], None] | None = None,
) -> DownstreamTrainResult:
    seed_everything(int(config.seed))
    _env_builders.configure_downstream_process_threads()
    sb3_runtime = _import_sb3()
    BaseCallback = sb3_runtime.base_callback
    DummyVecEnv = sb3_runtime.dummy_vec_env
    VecMonitor = sb3_runtime.vec_monitor
    SubprocVecEnv = sb3_runtime.subproc_vec_env or DummyVecEnv
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_output_dir = output_dir / "previews"
    device = _resolve_device(config.training.device)
    artifact_registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    train_env = _env_builders.build_train_vec_env(
        repo_root=repo_root,
        config=config,
        dummy_vec_env_type=DummyVecEnv,
        subproc_vec_env_type=SubprocVecEnv,
        vec_monitor_type=VecMonitor,
    )
    vector_eval_env = None
    try:
        preview_env_factory = _env_builders.build_env_factory(
            repo_root=repo_root,
            config=config,
            observation_name="eval",
            seed=int(config.seed) + 10_000,
        )
        canonical_eval_factories = _env_builders.build_eval_env_factories(
            repo_root=repo_root,
            config=config,
        )

        def _build_vector_eval_env():
            vector_eval_episodes = max(
                int(config.training.eval_episodes),
                int(config.training.final_eval_episodes),
            )
            if config.observation.mode == "feature_vector":
                return _env_builders.build_shared_eval_vec_env(
                    repo_root=repo_root,
                    config=config,
                    dummy_vec_env_type=DummyVecEnv,
                    subproc_vec_env_type=SubprocVecEnv,
                    vec_monitor_type=VecMonitor,
                    episodes=vector_eval_episodes,
                )
            if config.observation.mode == "raw_pixels":
                return _env_builders.build_raw_eval_vec_env(
                    repo_root=repo_root,
                    config=config,
                    dummy_vec_env_type=DummyVecEnv,
                    subproc_vec_env_type=SubprocVecEnv,
                    vec_monitor_type=VecMonitor,
                    episodes=vector_eval_episodes,
                )
            return None

        eval_requested = (
            int(config.training.eval_freq) > 0 and int(config.training.eval_episodes) > 0
        )
        final_eval_requested = int(config.training.final_eval_episodes) > 0
        if _env_builders.can_use_parallel_vector_eval(config) and (
            eval_requested or final_eval_requested
        ):
            vector_eval_env = _build_vector_eval_env()
        curriculum_scheduler = None
        curriculum_transition_log: list[dict[str, object]] = []
        current_phase = None
        current_phase_index: int | None = None
        if config.curriculum is not None and config.curriculum.spawn_schedule:
            curriculum_scheduler = SpawnCurriculumScheduler(list(config.curriculum.spawn_schedule))
            current_phase_index = curriculum_scheduler.phase_index(0)
            current_phase = curriculum_scheduler.active_phase(0)
            _env_builders.apply_curriculum_phase_to_vector_env(train_env, current_phase)
            preview_env_factory = _env_builders.build_curriculum_applied_env_factory(
                repo_root=repo_root,
                config=config,
                observation_name="preview_curriculum",
                seed=int(config.seed) + 10_000,
                phase=current_phase,
            )
            banner = format_phase_transition(
                current_phase,
                current_phase_index,
                episode=0,
                timestep=0,
            )
            print(banner, file=sys.stderr)
            curriculum_transition_log.append(
                {
                    "event": "initial",
                    "phase_name": current_phase.name,
                    "phase_index": int(current_phase_index),
                    "episode": 0,
                    "timestep": 0,
                }
            )
        curriculum_eval_factories = (
            canonical_eval_factories
            if current_phase is None
            else _env_builders.build_curriculum_eval_factories(
                repo_root=repo_root,
                config=config,
                phase=current_phase,
            )
        )
        evaluation_history: list[dict[str, object]] = []
        training_episode_history: list[dict[str, object]] = []
        preview_history: list[dict[str, object]] = []
        best_metrics: dict[str, float] | None = None
        best_model_path = output_dir / "best_model.zip"
        final_model_path = output_dir / "final_model.zip"
        interrupted_model_path = output_dir / "interrupted_model.zip"
        preview_scheduler = PreviewScheduler(config.preview.every_n_episodes)
        periodic_previews_enabled = _training_previews_enabled(config)
        completed_episodes = 0
        termination_state = _TerminationState()
        memory_watchdog_limit_mb = _resolve_downstream_memory_watchdog_limit_mb(config)
        preview_reconstruction_source = None
        if periodic_previews_enabled:
            preview_reconstruction_source = build_reconstruction_source(
                artifact_registry=artifact_registry,
                models=config.models,
                device=device,
            )
    except Exception:
        if vector_eval_env is not None:
            vector_eval_env.close()
        train_env.close()
        raise

    def _print_blocking_stage(message: str) -> float:
        print(message, file=sys.stderr, flush=True)
        return time.perf_counter()

    def _print_blocking_stage_done(message: str, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        print(f"{message} ({elapsed:.1f}s)", file=sys.stderr, flush=True)

    def _write_output_files(metrics_payload: dict[str, Any]) -> Path:
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
        (output_dir / "used_hyperparameters.yaml").write_text(
            yaml.safe_dump(
                {
                    "models": asdict(config.models),
                    "goal_task": asdict(config.goal_task),
                    "observation": asdict(config.observation),
                    "training": asdict(config.training),
                    "curriculum": None if config.curriculum is None else asdict(config.curriculum),
                    "preview": asdict(config.preview),
                },
                sort_keys=False,
            )
        )
        return metrics_path

    class _EvalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._next_eval_timestep = max(1, int(config.training.eval_freq))
            self._next_checkpoint_timestep = int(config.training.checkpoint_freq)
            self._start_positions_by_env: dict[int, list[float]] = {}

        def _on_step(self) -> bool:
            nonlocal best_metrics, completed_episodes, current_phase, current_phase_index
            nonlocal curriculum_eval_factories, preview_env_factory
            if termination_state.requested:
                termination_state.timestep = int(self.num_timesteps)
                if not termination_state.checkpoint_saved:
                    self.model.save(str(interrupted_model_path))
                    termination_state.checkpoint_saved = True
                return False
            _raise_if_downstream_memory_watchdog_exceeded(memory_watchdog_limit_mb)
            if self.num_timesteps >= self._next_checkpoint_timestep:
                checkpoint_dir = output_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"model_{self.num_timesteps}_steps.zip"
                self.model.save(str(checkpoint_path))
                receipt_path = checkpoint_path.with_suffix(".json")
                temporary_receipt = receipt_path.with_suffix(".tmp")
                temporary_receipt.write_text(
                    json.dumps(
                        {
                            "requested_timestep": self._next_checkpoint_timestep,
                            "actual_timestep": int(self.num_timesteps),
                            "model_path": str(checkpoint_path),
                        }
                    )
                    + "\n"
                )
                temporary_receipt.replace(receipt_path)
                self._next_checkpoint_timestep = (
                    int(self.num_timesteps) // int(config.training.checkpoint_freq) + 1
                ) * int(config.training.checkpoint_freq)
            dones = self.locals.get("dones")
            infos = self.locals.get("infos")
            reset_infos = _reset_infos_from_vec_env(getattr(self, "training_env", train_env))
            for env_index, reset_info in enumerate(reset_infos):
                if env_index in self._start_positions_by_env:
                    continue
                start_position_xy = _position_xy_from_info(reset_info)
                if start_position_xy is not None:
                    self._start_positions_by_env[env_index] = start_position_xy
            done_flags = None if dones is None else np.asarray(dones, dtype=bool)
            episode_completed_env_indices: list[int] = []
            if done_flags is not None:
                episode_completed_env_indices = [
                    int(env_index)
                    for env_index, done in enumerate(done_flags.tolist())
                    if bool(done)
                ]
                completed_episodes += len(episode_completed_env_indices)
            if episode_completed_env_indices:
                info_list = list(infos) if infos is not None else []
                completed_episode_base = completed_episodes - len(episode_completed_env_indices)
                for episode_offset, env_index in enumerate(episode_completed_env_indices, start=1):
                    info = (
                        info_list[env_index]
                        if env_index < len(info_list) and isinstance(info_list[env_index], dict)
                        else {}
                    )
                    episode_summary = info.get("episode")
                    if not isinstance(episode_summary, dict):
                        continue
                    episode_return = float(episode_summary.get("r", 0.0))
                    episode_length = float(episode_summary.get("l", 0.0))
                    episode_success = float(bool(info.get("is_success", episode_return > 0.0)))
                    completed_episode = completed_episode_base + episode_offset
                    start_position_xy = self._start_positions_by_env.get(env_index)
                    goal_position_xy = _position_xy_from_info(info, key="goal_position_xy")
                    next_start_position_xy = _position_xy_from_info(
                        info,
                        key="next_start_position_xy",
                    )
                    if next_start_position_xy is None and env_index < len(reset_infos):
                        next_start_position_xy = _position_xy_from_info(reset_infos[env_index])
                    training_episode_row = {
                        "completed_episodes": int(completed_episode),
                        "timestep": int(self.num_timesteps),
                        "return": episode_return,
                        "length": episode_length,
                        "success_rate": episode_success,
                        "env_index": int(env_index),
                        "curriculum_phase": None if current_phase is None else current_phase.name,
                        "curriculum_phase_index": current_phase_index,
                    }
                    metric_payload = {
                        "train_episode/completed_episodes": float(completed_episode),
                        "train_episode/return": episode_return,
                        "train_episode/length": episode_length,
                        "train_episode/success_rate": episode_success,
                        "train_episode/env_index": float(env_index),
                        "trainer/step": float(self.num_timesteps),
                        "curriculum/phase_name": (
                            None if current_phase is None else current_phase.name
                        ),
                        "curriculum/phase_index": current_phase_index,
                    }
                    if start_position_xy is not None:
                        training_episode_row["start_position_xy"] = start_position_xy
                        metric_payload["train_episode/start_x"] = float(start_position_xy[0])
                        metric_payload["train_episode/start_y"] = float(start_position_xy[1])
                    if goal_position_xy is not None:
                        training_episode_row["goal_position_xy"] = goal_position_xy
                        metric_payload["train_episode/goal_x"] = float(goal_position_xy[0])
                        metric_payload["train_episode/goal_y"] = float(goal_position_xy[1])
                        if start_position_xy is not None:
                            start_goal_distance = float(
                                np.linalg.norm(
                                    np.asarray(goal_position_xy) - np.asarray(start_position_xy)
                                )
                            )
                            training_episode_row["start_goal_distance"] = start_goal_distance
                            metric_payload["train_episode/start_goal_distance"] = (
                                start_goal_distance
                            )
                    if info.get("goal_distance") is not None:
                        final_goal_distance = float(info["goal_distance"])
                        training_episode_row["final_goal_distance"] = final_goal_distance
                        metric_payload["train_episode/final_goal_distance"] = final_goal_distance
                    if next_start_position_xy is not None:
                        training_episode_row["next_start_position_xy"] = next_start_position_xy
                        self._start_positions_by_env[env_index] = next_start_position_xy
                    else:
                        self._start_positions_by_env.pop(env_index, None)
                    training_episode_history.append(training_episode_row)
                    if metric_logger is not None:
                        metric_logger(metric_payload, None)
            if curriculum_scheduler is not None:
                next_phase_index = curriculum_scheduler.phase_index(completed_episodes)
                if current_phase_index != next_phase_index:
                    current_phase_index = int(next_phase_index)
                    current_phase = curriculum_scheduler.active_phase(completed_episodes)
                    _env_builders.apply_curriculum_phase_to_vector_env(train_env, current_phase)
                    curriculum_eval_factories = _env_builders.build_curriculum_eval_factories(
                        repo_root=repo_root,
                        config=config,
                        phase=current_phase,
                    )
                    preview_env_factory = _env_builders.build_curriculum_applied_env_factory(
                        repo_root=repo_root,
                        config=config,
                        observation_name="preview_curriculum",
                        seed=int(config.seed) + 10_000,
                        phase=current_phase,
                    )
                    banner = format_phase_transition(
                        current_phase,
                        current_phase_index,
                        episode=completed_episodes,
                        timestep=self.num_timesteps,
                    )
                    print(banner, file=sys.stderr)
                    curriculum_transition_log.append(
                        {
                            "event": "transition",
                            "phase_name": current_phase.name,
                            "phase_index": int(current_phase_index),
                            "episode": int(completed_episodes),
                            "timestep": int(self.num_timesteps),
                        }
                    )
                    self.logger.record("curriculum/phase_name", current_phase.name)
                    self.logger.record("curriculum/phase_index", float(current_phase_index))
                    if metric_logger is not None:
                        metric_logger(
                            {
                                "trainer/step": float(self.num_timesteps),
                                "curriculum/phase_name": current_phase.name,
                                "curriculum/phase_index": float(current_phase_index),
                            },
                            int(self.num_timesteps),
                        )
            if periodic_previews_enabled and preview_scheduler.should_capture(completed_episodes):
                preview_name = f"after_episode_{completed_episodes:06d}"
                preview_started_at = _print_blocking_stage(
                    "[downstream] capturing preview "
                    f"{preview_name} at episode={completed_episodes}, timestep={self.num_timesteps}"
                )
                preview_bundle = capture_policy_preview(
                    env_factory=preview_env_factory,
                    action_fn=lambda observation: int(
                        np.asarray(
                            self.model.predict(
                                observation,
                                deterministic=bool(config.preview.deterministic_policy),
                            )[0]
                        ).reshape(-1)[0]
                    ),
                    env_id=config.environment.env_id,
                    preview_name=preview_name,
                    output_dir=preview_output_dir,
                    preview_config=config.preview,
                    seed=int(config.seed) + 20_000 + completed_episodes,
                    reconstruction_source=preview_reconstruction_source,
                )
                preview_history.append(
                    {
                        "completed_episodes": completed_episodes,
                        "curriculum_phase": None if current_phase is None else current_phase.name,
                        "curriculum_phase_index": current_phase_index,
                        "combined_gif_path": str(preview_bundle.combined_gif_path),
                        "rgb_gif_path": (
                            None
                            if preview_bundle.rgb_gif_path is None
                            else str(preview_bundle.rgb_gif_path)
                        ),
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
                    }
                )
                _print_blocking_stage_done(
                    f"[downstream] finished preview {preview_name}",
                    preview_started_at,
                )
            eval_freq = int(config.training.eval_freq)
            if eval_freq <= 0 or int(config.training.eval_episodes) <= 0:
                return True
            if int(self.num_timesteps) < self._next_eval_timestep:
                return True
            while self._next_eval_timestep <= int(self.num_timesteps):
                self._next_eval_timestep += eval_freq
            eval_started_at = _print_blocking_stage(
                "[downstream] running evaluation "
                f"at timestep={self.num_timesteps}, episodes={int(config.training.eval_episodes)}"
            )
            report_progress_phase(
                progress_reporter,
                EVALUATION_PHASE,
                int(self.num_timesteps),
            )
            eval_episodes = int(config.training.eval_episodes)
            deterministic_eval_started_at = _print_blocking_stage(
                "[downstream] running deterministic evaluation "
                f"at timestep={self.num_timesteps}, episodes={eval_episodes}"
            )
            canonical_metrics = _evaluate_policy(
                self.model,
                canonical_eval_factories,
                episodes=eval_episodes,
                max_steps_per_episode=int(config.environment.episode_length),
                vector_env=vector_eval_env,
                progress_label="deterministic evaluation",
                progress_callback=lambda: report_progress_heartbeat(
                    progress_reporter,
                    EVALUATION_PHASE,
                    int(self.num_timesteps),
                ),
            )
            _print_blocking_stage_done(
                "[downstream] finished deterministic evaluation",
                deterministic_eval_started_at,
            )
            stochastic_metrics = None
            if config.training.eval_stochastic:
                stochastic_eval_started_at = _print_blocking_stage(
                    "[downstream] running stochastic evaluation "
                    f"at timestep={self.num_timesteps}, episodes={eval_episodes}"
                )
                stochastic_metrics = _evaluate_policy(
                    self.model,
                    canonical_eval_factories,
                    episodes=eval_episodes,
                    max_steps_per_episode=int(config.environment.episode_length),
                    deterministic=False,
                    vector_env=vector_eval_env,
                    progress_label="stochastic evaluation",
                    progress_callback=lambda: report_progress_heartbeat(
                        progress_reporter,
                        EVALUATION_PHASE,
                        int(self.num_timesteps),
                    ),
                )
                _print_blocking_stage_done(
                    "[downstream] finished stochastic evaluation",
                    stochastic_eval_started_at,
                )
            curriculum_metrics = None
            if curriculum_scheduler is not None:
                curriculum_eval_started_at = _print_blocking_stage(
                    "[downstream] running curriculum evaluation "
                    f"at timestep={self.num_timesteps}, episodes={eval_episodes}"
                )
                curriculum_metrics = _evaluate_policy(
                    self.model,
                    curriculum_eval_factories,
                    episodes=eval_episodes,
                    max_steps_per_episode=int(config.environment.episode_length),
                    progress_label="curriculum evaluation",
                    progress_callback=lambda: report_progress_heartbeat(
                        progress_reporter,
                        EVALUATION_PHASE,
                        int(self.num_timesteps),
                    ),
                )
                _print_blocking_stage_done(
                    "[downstream] finished curriculum evaluation",
                    curriculum_eval_started_at,
                )
            _print_blocking_stage_done(
                f"[downstream] finished evaluation at timestep={self.num_timesteps}",
                eval_started_at,
            )
            evaluation_entry: dict[str, float | int | str | None] = {
                "mean_return": float(canonical_metrics["mean_return"]),
                "success_rate": float(canonical_metrics["success_rate"]),
                "mean_time_to_goal": float(canonical_metrics["mean_time_to_goal"]),
                "timesteps": float(self.num_timesteps),
                "curriculum_phase": None if current_phase is None else current_phase.name,
                "curriculum_phase_index": current_phase_index,
            }
            if stochastic_metrics is not None:
                evaluation_entry.update(
                    {
                        "stochastic_mean_return": float(stochastic_metrics["mean_return"]),
                        "stochastic_success_rate": float(stochastic_metrics["success_rate"]),
                        "stochastic_mean_time_to_goal": float(
                            stochastic_metrics["mean_time_to_goal"]
                        ),
                    }
                )
            if curriculum_metrics is not None:
                evaluation_entry.update(
                    {
                        "curriculum_mean_return": float(curriculum_metrics["mean_return"]),
                        "curriculum_success_rate": float(curriculum_metrics["success_rate"]),
                        "curriculum_mean_time_to_goal": float(
                            curriculum_metrics["mean_time_to_goal"]
                        ),
                    }
                )
            evaluation_history.append(evaluation_entry)
            self.logger.record("eval/mean_return", float(canonical_metrics["mean_return"]))
            self.logger.record("eval/success_rate", float(canonical_metrics["success_rate"]))
            self.logger.record(
                "eval/mean_time_to_goal",
                float(canonical_metrics["mean_time_to_goal"]),
            )
            if stochastic_metrics is not None:
                self.logger.record(
                    "eval_stochastic/mean_return",
                    float(stochastic_metrics["mean_return"]),
                )
                self.logger.record(
                    "eval_stochastic/success_rate",
                    float(stochastic_metrics["success_rate"]),
                )
                self.logger.record(
                    "eval_stochastic/mean_time_to_goal",
                    float(stochastic_metrics["mean_time_to_goal"]),
                )
            if curriculum_metrics is not None:
                self.logger.record(
                    "eval_curriculum/mean_return",
                    float(curriculum_metrics["mean_return"]),
                )
                self.logger.record(
                    "eval_curriculum/success_rate",
                    float(curriculum_metrics["success_rate"]),
                )
                self.logger.record(
                    "eval_curriculum/mean_time_to_goal",
                    float(curriculum_metrics["mean_time_to_goal"]),
                )
            if current_phase is not None:
                self.logger.record("curriculum/phase_name", current_phase.name)
                self.logger.record("curriculum/phase_index", float(current_phase_index))
            if metric_logger is not None:
                metric_payload = {
                    "trainer/step": float(self.num_timesteps),
                    "eval/mean_return": float(canonical_metrics["mean_return"]),
                    "eval/success_rate": float(canonical_metrics["success_rate"]),
                    "eval/mean_time_to_goal": float(canonical_metrics["mean_time_to_goal"]),
                    "curriculum/phase_name": None if current_phase is None else current_phase.name,
                    "curriculum/phase_index": current_phase_index,
                }
                if stochastic_metrics is not None:
                    metric_payload.update(
                        {
                            "eval_stochastic/mean_return": float(stochastic_metrics["mean_return"]),
                            "eval_stochastic/success_rate": float(
                                stochastic_metrics["success_rate"]
                            ),
                            "eval_stochastic/mean_time_to_goal": float(
                                stochastic_metrics["mean_time_to_goal"]
                            ),
                        }
                    )
                if curriculum_metrics is not None:
                    metric_payload.update(
                        {
                            "eval_curriculum/mean_return": float(curriculum_metrics["mean_return"]),
                            "eval_curriculum/success_rate": float(
                                curriculum_metrics["success_rate"]
                            ),
                            "eval_curriculum/mean_time_to_goal": float(
                                curriculum_metrics["mean_time_to_goal"]
                            ),
                        }
                    )
                metric_logger(metric_payload, int(self.num_timesteps))
            if _is_strictly_better_eval_metrics(canonical_metrics, best_metrics):
                best_metrics = dict(canonical_metrics)
                self.model.save(str(best_model_path))
            report_progress_phase(
                progress_reporter,
                TRAINING_PHASE,
                int(self.num_timesteps),
            )
            return True

    try:
        model = build_sb3_model(
            runtime=sb3_runtime,
            config=config,
            train_env=train_env,
            device=device,
        )
    except Exception:
        if vector_eval_env is not None:
            vector_eval_env.close()
        train_env.close()
        raise

    class _HeartbeatCallback(BaseCallback):
        """Reports num_timesteps to the parent so its no-progress watchdog can detect a hang."""

        def __init__(self, reporter, interval_seconds):
            super().__init__()
            self._reporter = reporter
            self._interval_seconds = float(interval_seconds)
            self._last_sent_at: float | None = None

        def _on_step(self) -> bool:
            now = time.monotonic()
            if self._last_sent_at is None or now - self._last_sent_at >= self._interval_seconds:
                try:
                    self._reporter(int(self.num_timesteps))
                except Exception:
                    pass
                self._last_sent_at = now
            return True

    try:
        learn_callbacks = [_EvalCallback()]
        if progress_reporter is not None:
            learn_callbacks.append(
                _HeartbeatCallback(progress_reporter, _HEARTBEAT_INTERVAL_SECONDS)
            )
        learn_callback = learn_callbacks[0] if len(learn_callbacks) == 1 else learn_callbacks
        report_progress_phase(progress_reporter, TRAINING_PHASE, 0)
        with _install_sigterm_handler(termination_state):
            model.learn(
                total_timesteps=int(config.training.total_timesteps),
                callback=learn_callback,
                progress_bar=bool(config.training.progress_bar),
                log_interval=int(config.training.log_interval),
            )
        final_training_timestep = int(
            getattr(model, "num_timesteps", config.training.total_timesteps)
        )
        report_progress_phase(
            progress_reporter,
            POST_TRAINING_PHASE,
            final_training_timestep,
        )
        if termination_state.requested:
            if not termination_state.checkpoint_saved:
                model.save(str(interrupted_model_path))
                termination_state.checkpoint_saved = True
            summary_metrics = dict(best_metrics or _default_summary_metrics())
            metrics_payload = {
                "status": "interrupted",
                "final_training_timestep": final_training_timestep,
                "config_name": config.name,
                "seed": int(config.seed),
                "environment_id": config.environment.env_id,
                "observation_mode": config.observation.mode,
                "feature_sources": list(config.observation.feature_sources),
                "evaluation_history": evaluation_history,
                "training_episode_history": training_episode_history,
                "preview_history": preview_history,
                "curriculum_transitions": curriculum_transition_log,
                "curriculum_config": (
                    None if config.curriculum is None else asdict(config.curriculum)
                ),
                "best_metrics": best_metrics,
                "final_metrics": summary_metrics,
                "final_stochastic_metrics": None,
                "final_curriculum_metrics": None,
                "interrupted_signal": termination_state.signal_number,
                "interrupted_timestep": termination_state.timestep,
                "interrupted_model_path": str(interrupted_model_path),
            }
            metrics_path = _write_output_files(metrics_payload)
            return DownstreamTrainResult(
                output_dir=output_dir,
                metrics_path=metrics_path,
                final_model_path=None,
                best_model_path=best_model_path if best_model_path.exists() else None,
                interrupted_model_path=interrupted_model_path,
                final_metrics=summary_metrics,
                status="interrupted",
            )

        model.save(str(final_model_path))
        final_eval_episodes = int(config.training.final_eval_episodes)
        if final_eval_episodes <= 0:
            final_metrics = dict(best_metrics or _default_summary_metrics())
            metrics_payload = {
                "status": "completed",
                "final_training_timestep": final_training_timestep,
                "config_name": config.name,
                "seed": int(config.seed),
                "environment_id": config.environment.env_id,
                "observation_mode": config.observation.mode,
                "feature_sources": list(config.observation.feature_sources),
                "evaluation_history": evaluation_history,
                "training_episode_history": training_episode_history,
                "preview_history": preview_history,
                "curriculum_transitions": curriculum_transition_log,
                "curriculum_config": (
                    None if config.curriculum is None else asdict(config.curriculum)
                ),
                "best_metrics": best_metrics,
                "final_metrics": final_metrics,
                "final_stochastic_metrics": None,
                "final_curriculum_metrics": None,
            }
            metrics_path = _write_output_files(metrics_payload)
            return DownstreamTrainResult(
                output_dir=output_dir,
                metrics_path=metrics_path,
                final_model_path=final_model_path,
                best_model_path=best_model_path if best_model_path.exists() else None,
                interrupted_model_path=None,
                final_metrics=final_metrics,
                status="completed",
            )
        final_eval_started_at = _print_blocking_stage(
            f"[downstream] running final evaluation episodes={final_eval_episodes}"
        )
        report_progress_phase(
            progress_reporter,
            EVALUATION_PHASE,
            final_training_timestep,
        )
        final_eval_diagnostics = (
            FinalEvalDiagnostics.create(
                model=model,
                output_dir=output_dir,
                env_id=config.environment.env_id,
                num_envs=max(1, int(config.training.n_envs)),
                max_episodes=final_eval_episodes,
            )
            if bool(config.training.final_eval_diagnostics)
            else None
        )
        final_diagnostics_payload: dict[str, Any] = {}
        deterministic_final_eval_started_at = _print_blocking_stage(
            f"[downstream] running final deterministic evaluation episodes={final_eval_episodes}"
        )
        try:
            final_metrics = _evaluate_policy(
                model,
                canonical_eval_factories,
                episodes=final_eval_episodes,
                max_steps_per_episode=int(config.environment.episode_length),
                vector_env=vector_eval_env,
                observers=(
                    [] if final_eval_diagnostics is None else final_eval_diagnostics.observers()
                ),
                progress_label="final deterministic evaluation",
                progress_callback=lambda: report_progress_heartbeat(
                    progress_reporter,
                    EVALUATION_PHASE,
                    final_training_timestep,
                ),
            )
        finally:
            _print_blocking_stage_done(
                "[downstream] finished final deterministic evaluation",
                deterministic_final_eval_started_at,
            )
            if final_eval_diagnostics is not None:
                final_diagnostics_started_at = _print_blocking_stage(
                    "[downstream] closing final evaluation diagnostics"
                )
                final_diagnostics_payload = final_eval_diagnostics.close()
                _print_blocking_stage_done(
                    "[downstream] finished final evaluation diagnostics",
                    final_diagnostics_started_at,
                )
        final_stochastic_metrics = None
        if config.training.final_eval_stochastic:
            stochastic_final_eval_started_at = _print_blocking_stage(
                f"[downstream] running final stochastic evaluation episodes={final_eval_episodes}"
            )
            final_stochastic_metrics = _evaluate_policy(
                model,
                canonical_eval_factories,
                episodes=final_eval_episodes,
                max_steps_per_episode=int(config.environment.episode_length),
                deterministic=False,
                vector_env=vector_eval_env,
                progress_label="final stochastic evaluation",
                progress_callback=lambda: report_progress_heartbeat(
                    progress_reporter,
                    EVALUATION_PHASE,
                    final_training_timestep,
                ),
            )
            _print_blocking_stage_done(
                "[downstream] finished final stochastic evaluation",
                stochastic_final_eval_started_at,
            )
        final_curriculum_metrics = None
        if curriculum_scheduler is not None:
            curriculum_final_eval_started_at = _print_blocking_stage(
                f"[downstream] running final curriculum evaluation episodes={final_eval_episodes}"
            )
            final_curriculum_metrics = _evaluate_policy(
                model,
                curriculum_eval_factories,
                episodes=final_eval_episodes,
                max_steps_per_episode=int(config.environment.episode_length),
                progress_label="final curriculum evaluation",
                progress_callback=lambda: report_progress_heartbeat(
                    progress_reporter,
                    EVALUATION_PHASE,
                    final_training_timestep,
                ),
            )
            _print_blocking_stage_done(
                "[downstream] finished final curriculum evaluation",
                curriculum_final_eval_started_at,
            )
        _print_blocking_stage_done("[downstream] finished final evaluation", final_eval_started_at)
        report_progress_phase(
            progress_reporter,
            POST_TRAINING_PHASE,
            final_training_timestep,
        )
        metrics_payload = {
            "status": "completed",
            "final_training_timestep": final_training_timestep,
            "config_name": config.name,
            "seed": int(config.seed),
            "environment_id": config.environment.env_id,
            "observation_mode": config.observation.mode,
            "feature_sources": list(config.observation.feature_sources),
            "evaluation_history": evaluation_history,
            "training_episode_history": training_episode_history,
            "preview_history": preview_history,
            "curriculum_transitions": curriculum_transition_log,
            "curriculum_config": None if config.curriculum is None else asdict(config.curriculum),
            "best_metrics": best_metrics or final_metrics,
            "final_metrics": final_metrics,
            "final_stochastic_metrics": final_stochastic_metrics,
            "final_curriculum_metrics": final_curriculum_metrics,
            **final_diagnostics_payload,
        }
        metrics_path = _write_output_files(metrics_payload)
        return DownstreamTrainResult(
            output_dir=output_dir,
            metrics_path=metrics_path,
            final_model_path=final_model_path,
            best_model_path=best_model_path if best_model_path.exists() else None,
            interrupted_model_path=None,
            final_metrics=final_metrics,
            status="completed",
        )
    finally:
        if vector_eval_env is not None:
            vector_eval_env.close()
        train_env.close()
