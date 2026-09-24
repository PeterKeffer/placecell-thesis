"""SB3 backend selection for downstream RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from placecell_research.config.downstream_schema import DownstreamRunConfig

AUTO_DEFAULT_UTD_GRADIENT_STEPS = "auto_default_utd"
DEFAULT_DQN_UPDATE_TO_DATA_RATIO = 0.25


class DictNStepReplayBuffer(DictReplayBuffer):
    """n-step return replay buffer for Dict observation spaces."""

    def __init__(self, *args, n_steps: int = 3, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        if self.optimize_memory_usage:
            raise NotImplementedError(
                "DictNStepReplayBuffer doesn't support optimize_memory_usage=True"
            )

    def _get_samples(self, batch_inds, env=None) -> DictReplayBufferSamples:
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)

        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(
            original_timeout_values, np.logical_not(self.dones[last_valid_index])
        )

        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (batch_inds[:, None] + steps) % self.buffer_size

        rewards_seq = self._normalize_reward(self.rewards[indices, env_indices[:, None]], env)
        dones_seq = self.dones[indices, env_indices[:, None]]
        truncated_seq = self.timeouts[indices, env_indices[:, None]]

        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.argmax(axis=1)
        has_done_or_truncated = done_or_truncated.any(axis=1)
        done_idx = np.where(has_done_or_truncated, done_idx, self.n_steps - 1)

        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]
        target_q_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)

        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=1, keepdims=True)

        last_indices = (batch_inds + done_idx) % self.buffer_size
        next_dones = self.dones[last_indices, env_indices][:, None].astype(np.float32)
        next_timeouts = self.timeouts[last_indices, env_indices][:, None].astype(np.float32)
        final_dones = next_dones * (1.0 - next_timeouts)

        self.timeouts[last_valid_index] = original_timeout_values

        obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices] for key, obs in self.observations.items()}, env
        )
        next_obs_ = self._normalize_obs(
            {key: obs[last_indices, env_indices] for key, obs in self.next_observations.items()},
            env,
        )

        return DictReplayBufferSamples(
            observations={key: self.to_torch(obs) for key, obs in obs_.items()},
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={key: self.to_torch(obs) for key, obs in next_obs_.items()},
            dones=self.to_torch(final_dones),
            rewards=self.to_torch(n_step_returns),
            discounts=self.to_torch(target_q_discounts),
        )


@dataclass(slots=True)
class Sb3Runtime:
    ppo: type
    dqn: type
    base_callback: type
    dummy_vec_env: type
    vec_monitor: type
    subproc_vec_env: type


def _observation_is_dict(config: DownstreamRunConfig) -> bool:
    return bool(config.observation.mode == "raw_pixels" and config.observation.feature_sources)


def _resolve_replay_buffer(
    training,
    *,
    observation_is_dict=False,
):
    """Map config to the SB3 replay-buffer class and kwargs for off-policy algorithms."""
    replay_buffer_type = str(training.replay_buffer_type)
    if replay_buffer_type == "uniform":
        return None, None
    if replay_buffer_type == "nstep":
        from stable_baselines3.common.buffers import NStepReplayBuffer

        buffer_class = DictNStepReplayBuffer if observation_is_dict else NStepReplayBuffer
        return buffer_class, {
            "n_steps": int(training.n_step_returns),
            "gamma": float(training.gamma),
        }
    raise ValueError(f"Unknown replay_buffer_type {replay_buffer_type!r}.")


def _resolve_gradient_steps(config: DownstreamRunConfig) -> int:
    configured = config.training.gradient_steps
    if str(configured).strip().lower() != AUTO_DEFAULT_UTD_GRADIENT_STEPS:
        return int(configured)
    train_freq_unit = str(config.training.train_freq_unit).strip().lower()
    if train_freq_unit != "step":
        raise ValueError("gradient_steps=auto_default_utd requires train_freq_unit='step'.")
    transitions_per_training_phase = int(config.training.train_freq) * int(config.training.n_envs)
    return max(1, math.ceil(transitions_per_training_phase * DEFAULT_DQN_UPDATE_TO_DATA_RATIO))


def import_sb3_runtime() -> Sb3Runtime:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Downstream RL training requires stable-baselines3. Install it with "
            '`python -m pip install -e ".[rl]"`.'
        ) from exc
    return Sb3Runtime(
        ppo=PPO,
        dqn=DQN,
        base_callback=BaseCallback,
        dummy_vec_env=DummyVecEnv,
        vec_monitor=VecMonitor,
        subproc_vec_env=SubprocVecEnv,
    )


def _resolve_policy_name(config: DownstreamRunConfig) -> str:
    if _observation_is_dict(config):
        return "MultiInputPolicy"
    return "CnnPolicy" if config.observation.mode == "raw_pixels" else "MlpPolicy"


def _build_policy_kwargs(config: DownstreamRunConfig, *, policy_name: str) -> dict[str, Any] | None:
    if policy_name not in {"CnnPolicy", "MlpPolicy", "MultiInputPolicy"}:
        return None
    policy_kwargs: dict[str, Any] = {}
    if config.training.policy_hidden_sizes:
        policy_kwargs["net_arch"] = list(config.training.policy_hidden_sizes)
    cnn_features_dim = int(config.training.cnn_features_dim)
    if cnn_features_dim:
        if policy_name == "MultiInputPolicy":
            policy_kwargs["features_extractor_kwargs"] = {"cnn_output_dim": cnn_features_dim}
        elif policy_name == "CnnPolicy":
            policy_kwargs["features_extractor_kwargs"] = {"features_dim": cnn_features_dim}
        else:
            raise ValueError(
                f"training.cnn_features_dim={cnn_features_dim} has no effect on {policy_name!r}; "
                "it applies only to raw_pixels observations. Leave it at 0."
            )
    extractor_name = config.training.feature_extractor
    if extractor_name in {"split_position_goal", "paired_grid_code"}:
        if policy_name != "MlpPolicy":
            raise ValueError(
                f"training.feature_extractor={extractor_name!r} only supports MlpPolicy "
                f"(feature_vector observation mode); got {policy_name!r}."
            )
        from .feature_extractors import PairedGridCodeExtractor, SplitPositionGoalExtractor

        if extractor_name == "split_position_goal":
            policy_kwargs["features_extractor_class"] = SplitPositionGoalExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "observation_config": config.observation,
                "embed_dim": int(config.training.feature_extractor_embed_dim),
            }
        else:
            policy_kwargs["features_extractor_class"] = PairedGridCodeExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "observation_config": config.observation,
            }
    return policy_kwargs or None


def build_sb3_model(
    *,
    runtime: Sb3Runtime,
    config: DownstreamRunConfig,
    train_env,
    device: str,
):
    algorithm = str(config.training.algorithm).strip().lower()
    policy_name = _resolve_policy_name(config)
    policy_kwargs = _build_policy_kwargs(config, policy_name=policy_name)
    if algorithm == "ppo":
        return runtime.ppo(
            policy_name,
            train_env,
            learning_rate=float(config.training.learning_rate),
            n_steps=int(config.training.n_steps),
            batch_size=int(config.training.batch_size),
            n_epochs=int(config.training.n_epochs),
            gamma=float(config.training.gamma),
            ent_coef=float(config.training.ent_coef),
            clip_range=float(config.training.clip_range),
            gae_lambda=float(config.training.gae_lambda),
            max_grad_norm=float(config.training.max_grad_norm),
            policy_kwargs=policy_kwargs,
            device=device,
            seed=int(config.seed),
            verbose=1,
            stats_window_size=int(config.training.stats_window_size),
        )
    replay_buffer_class, replay_buffer_kwargs = _resolve_replay_buffer(
        config.training,
        observation_is_dict=_observation_is_dict(config),
    )
    return runtime.dqn(
        policy_name,
        train_env,
        learning_rate=float(config.training.learning_rate),
        buffer_size=int(config.training.buffer_size),
        learning_starts=int(config.training.learning_starts),
        batch_size=int(config.training.batch_size),
        tau=float(config.training.tau),
        gamma=float(config.training.gamma),
        train_freq=(int(config.training.train_freq), str(config.training.train_freq_unit)),
        gradient_steps=_resolve_gradient_steps(config),
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        target_update_interval=int(config.training.target_update_interval),
        exploration_fraction=float(config.training.exploration_fraction),
        exploration_initial_eps=float(config.training.exploration_initial_eps),
        exploration_final_eps=float(config.training.exploration_final_eps),
        max_grad_norm=float(config.training.max_grad_norm),
        policy_kwargs=policy_kwargs,
        device=device,
        seed=int(config.seed),
        verbose=1,
        stats_window_size=int(config.training.stats_window_size),
    )
