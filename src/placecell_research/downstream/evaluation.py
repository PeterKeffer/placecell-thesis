"""Downstream policy evaluation loops."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class EvaluationObserver:
    def on_reset(
        self,
        *,
        infos: list[dict[str, Any]],
        episode_indices: np.ndarray,
        timestep: int,
    ) -> None:
        return None

    def before_predict(
        self,
        *,
        observation: Any,
        infos: list[dict[str, Any]],
        episode_indices: np.ndarray,
    ) -> None:
        return None

    def after_step(
        self,
        *,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
        episode_indices: np.ndarray,
        timestep: int,
    ) -> None:
        return None


def evaluate_policy(
    model,
    eval_env_factories: list[Callable[[], Any]],
    episodes: int,
    max_steps_per_episode: int,
    *,
    deterministic: bool = True,
    vector_env: Any | None = None,
    observers: Sequence[EvaluationObserver] = (),
    progress_label: str | None = None,
    progress_callback: Callable[[], None] | None = None,
    action_repeat: int = 1,
) -> dict[str, float]:
    if vector_env is not None:
        if int(action_repeat) != 1:
            raise NotImplementedError(
                "action_repeat > 1 is only supported on the single-env evaluation path."
            )
        return _evaluate_vector_policy(
            model,
            vector_env,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            deterministic=deterministic,
            observers=observers,
            progress_label=progress_label,
            progress_callback=progress_callback,
        )
    episode_returns = []
    successes = []
    episode_lengths = []
    eval_envs = [factory() for factory in eval_env_factories]
    try:
        for episode_index in range(int(episodes)):
            eval_env = eval_envs[episode_index % len(eval_envs)]
            total_return, success, episode_length = _collect_evaluation_episode(
                model,
                eval_env,
                max_steps_per_episode=max_steps_per_episode,
                deterministic=deterministic,
                observers=observers,
                episode_index=episode_index,
                action_repeat=action_repeat,
                progress_callback=progress_callback,
            )
            episode_returns.append(total_return)
            successes.append(success)
            episode_lengths.append(episode_length)
    finally:
        for eval_env in eval_envs:
            eval_env.close()
    return _summarize_evaluation_metrics(
        episode_returns=episode_returns,
        successes=successes,
        episode_lengths=episode_lengths,
    )


def _collect_evaluation_episode(
    model,
    eval_env,
    *,
    max_steps_per_episode: int,
    deterministic: bool,
    observers: Sequence[EvaluationObserver],
    episode_index: int,
    action_repeat: int = 1,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[float, float, float]:
    observation, reset_info = eval_env.reset()
    current_info = dict(reset_info)
    episode_indices = np.asarray([int(episode_index)], dtype=np.int64)
    _notify_reset(
        observers,
        infos=[current_info],
        episode_indices=episode_indices,
        timestep=episode_index,
    )
    terminated = False
    truncated = False
    total_return = 0.0
    steps = 0
    final_info: dict[str, Any] = {}
    recurrent_state = None
    episode_start = np.ones((1,), dtype=bool)
    while not terminated and not truncated and steps < max_steps_per_episode:
        if progress_callback is not None:
            progress_callback()
        _notify_before_predict(
            observers,
            observation=observation,
            infos=[current_info],
            episode_indices=episode_indices,
        )
        action, recurrent_state = _predict_action(
            model,
            observation,
            recurrent_state,
            episode_start,
            deterministic=deterministic,
        )
        for _ in range(max(1, int(action_repeat))):
            observation, reward, terminated, truncated, info = eval_env.step(action)
            total_return += float(reward)
            steps += 1
            final_info = dict(info)
            current_info = final_info
            if terminated or truncated or steps >= int(max_steps_per_episode):
                break
        episode_start = np.asarray([False], dtype=bool)
        timed_out = steps >= int(max_steps_per_episode)
        _notify_after_step(
            observers,
            infos=[final_info],
            dones=np.asarray([terminated or truncated or timed_out], dtype=bool),
            episode_indices=episode_indices,
            timestep=episode_index,
        )
    success = float(bool(final_info.get("is_success", total_return > 0.0)))
    return total_return, success, float(steps)


def _evaluate_vector_policy(
    model,
    eval_vec_env,
    episodes: int,
    max_steps_per_episode: int,
    *,
    deterministic: bool,
    observers: Sequence[EvaluationObserver],
    progress_label: str | None,
    progress_callback: Callable[[], None] | None,
) -> dict[str, float]:
    episode_returns = np.zeros((eval_vec_env.num_envs,), dtype=np.float64)
    episode_lengths = np.zeros((eval_vec_env.num_envs,), dtype=np.float64)
    completed_returns: list[float] = []
    successes: list[float] = []
    completed_lengths: list[float] = []
    observation = eval_vec_env.reset()
    recurrent_state = None
    episode_starts = np.ones((eval_vec_env.num_envs,), dtype=bool)
    reset_infos = getattr(eval_vec_env, "reset_infos", None)
    current_infos = (
        [dict(info) if isinstance(info, dict) else {} for info in reset_infos]
        if reset_infos is not None
        else [{} for _ in range(eval_vec_env.num_envs)]
    )
    active_episode_indices = np.arange(eval_vec_env.num_envs, dtype=np.int64)
    next_episode_index = int(eval_vec_env.num_envs)
    if reset_infos is not None:
        _notify_reset(
            observers,
            infos=current_infos,
            episode_indices=active_episode_indices,
            timestep=0,
        )
    vector_steps = 0
    last_progress_report = time.perf_counter()
    while len(completed_returns) < int(episodes):
        if progress_callback is not None:
            progress_callback()
        _notify_before_predict(
            observers,
            observation=observation,
            infos=current_infos,
            episode_indices=active_episode_indices,
        )
        actions, recurrent_state = _predict_action(
            model,
            observation,
            recurrent_state,
            episode_starts,
            deterministic=deterministic,
        )
        observation, rewards, dones, infos = eval_vec_env.step(actions)
        vector_steps += 1
        rewards_array = np.asarray(rewards, dtype=np.float64).reshape(eval_vec_env.num_envs)
        done_flags = np.asarray(dones, dtype=bool).reshape(eval_vec_env.num_envs)
        episode_returns += rewards_array
        episode_lengths += 1.0
        timed_out = episode_lengths >= float(max_steps_per_episode)
        timed_out_without_done = timed_out & ~done_flags
        if np.any(timed_out_without_done):
            timed_out_indices = np.flatnonzero(timed_out_without_done)
            raise RuntimeError(
                "Vectorized evaluation reached max_steps_per_episode without env termination "
                f"or truncation for env indices {timed_out_indices.tolist()}. "
                "The vector env must report done so it can reset before the next episode."
            )
        finished = done_flags
        episode_starts = finished.copy()
        normalized_infos = [dict(info) if isinstance(info, dict) else {} for info in infos]
        _notify_after_step(
            observers,
            infos=normalized_infos,
            dones=finished,
            episode_indices=active_episode_indices,
            timestep=int(len(completed_returns) + np.sum(episode_lengths)),
        )
        next_infos: list[dict[str, Any]] = []
        for env_index, done in enumerate(finished.tolist()):
            info = normalized_infos[env_index] if env_index < len(normalized_infos) else {}
            if not done:
                next_infos.append(dict(info))
                continue
            total_return = float(episode_returns[env_index])
            completed_returns.append(total_return)
            successes.append(float(bool(info.get("is_success", total_return > 0.0))))
            completed_lengths.append(float(episode_lengths[env_index]))
            episode_returns[env_index] = 0.0
            episode_lengths[env_index] = 0.0
            next_info = dict(info)
            if "next_start_position_xy" in next_info:
                next_info["position_xy"] = next_info["next_start_position_xy"]
            next_infos.append(next_info)
            active_episode_indices[env_index] = next_episode_index
            next_episode_index += 1
            if len(completed_returns) >= int(episodes):
                break
        if len(next_infos) < eval_vec_env.num_envs:
            next_infos.extend(current_infos[len(next_infos) :])
        current_infos = next_infos
        if progress_label is not None:
            now = time.perf_counter()
            if now - last_progress_report >= 60.0:
                completed_count = min(len(completed_returns), int(episodes))
                active_max_length = float(np.max(episode_lengths)) if episode_lengths.size else 0.0
                print(
                    "[downstream] "
                    f"{progress_label}: completed={completed_count}/{int(episodes)} "
                    f"vector_steps={vector_steps} active_max_len={active_max_length:.0f}",
                    file=sys.stderr,
                    flush=True,
                )
                last_progress_report = now
    return _summarize_evaluation_metrics(
        episode_returns=completed_returns,
        successes=successes,
        episode_lengths=completed_lengths,
    )


def _predict_action(
    model,
    observation: Any,
    recurrent_state: Any,
    episode_start: np.ndarray,
    *,
    deterministic: bool,
) -> tuple[Any, Any]:
    try:
        return model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
    except TypeError as exc:
        if "state" not in str(exc) and "episode_start" not in str(exc):
            raise
        action, _ = model.predict(observation, deterministic=deterministic)
        return action, None


def _summarize_evaluation_metrics(
    *,
    episode_returns: list[float],
    successes: list[float],
    episode_lengths: list[float],
) -> dict[str, float]:
    return {
        "mean_return": float(np.mean(episode_returns) if episode_returns else 0.0),
        "success_rate": float(np.mean(successes) if successes else 0.0),
        "mean_time_to_goal": float(np.mean(episode_lengths) if episode_lengths else 0.0),
    }


def _notify_reset(
    observers: Sequence[EvaluationObserver],
    *,
    infos: list[dict[str, Any]],
    episode_indices: np.ndarray,
    timestep: int,
) -> None:
    for observer in observers:
        observer.on_reset(infos=infos, episode_indices=episode_indices, timestep=timestep)


def _notify_before_predict(
    observers: Sequence[EvaluationObserver],
    *,
    observation: Any,
    infos: list[dict[str, Any]],
    episode_indices: np.ndarray,
) -> None:
    for observer in observers:
        observer.before_predict(
            observation=observation,
            infos=infos,
            episode_indices=episode_indices,
        )


def _notify_after_step(
    observers: Sequence[EvaluationObserver],
    *,
    infos: list[dict[str, Any]],
    dones: np.ndarray,
    episode_indices: np.ndarray,
    timestep: int,
) -> None:
    for observer in observers:
        observer.after_step(
            infos=infos,
            dones=dones,
            episode_indices=episode_indices,
            timestep=timestep,
        )
