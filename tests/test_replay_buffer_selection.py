import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import NStepReplayBuffer

from placecell_research.config.downstream_schema import DownstreamTrainingConfig
from placecell_research.downstream.sb3_algorithms import (
    DictNStepReplayBuffer,
    _resolve_replay_buffer,
)


def test_training_config_has_replay_buffer_fields():
    cfg = DownstreamTrainingConfig(replay_buffer_type="nstep", n_step_returns=3)
    assert cfg.replay_buffer_type == "nstep"
    assert cfg.n_step_returns == 3


def test_replay_buffer_type_defaults_to_uniform():
    cfg = DownstreamTrainingConfig()
    assert cfg.replay_buffer_type == "uniform"
    assert cfg.n_step_returns == 3


def test_uniform_returns_sb3_default():
    cfg = DownstreamTrainingConfig(replay_buffer_type="uniform")
    cls, kwargs = _resolve_replay_buffer(cfg)
    assert cls is None and kwargs is None


def test_nstep_returns_nstep_buffer_with_matching_gamma():
    cfg = DownstreamTrainingConfig(replay_buffer_type="nstep", n_step_returns=3, gamma=0.997)
    cls, kwargs = _resolve_replay_buffer(cfg)
    assert cls is NStepReplayBuffer
    assert kwargs == {"n_steps": 3, "gamma": 0.997}


def test_nstep_with_box_obs_returns_box_nstep_buffer():
    cfg = DownstreamTrainingConfig(replay_buffer_type="nstep", n_step_returns=3, gamma=0.997)
    cls, kwargs = _resolve_replay_buffer(cfg, observation_is_dict=False)
    assert cls is NStepReplayBuffer
    assert kwargs == {"n_steps": 3, "gamma": 0.997}


def test_nstep_with_dict_obs_returns_dict_nstep_buffer():
    cfg = DownstreamTrainingConfig(replay_buffer_type="nstep", n_step_returns=3, gamma=0.997)
    cls, kwargs = _resolve_replay_buffer(cfg, observation_is_dict=True)
    assert cls is DictNStepReplayBuffer
    assert kwargs == {"n_steps": 3, "gamma": 0.997}


def test_dict_nstep_buffer_stores_and_samples_dict_obs():
    dict_space = spaces.Dict(
        {
            "image": spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8),
            "features": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        }
    )
    action_space = spaces.Discrete(3)
    buffer = DictNStepReplayBuffer(64, dict_space, action_space, n_steps=3, gamma=0.99)
    for _step in range(20):
        obs = {k: dict_space[k].sample() for k in dict_space.spaces}
        next_obs = {k: dict_space[k].sample() for k in dict_space.spaces}
        buffer.add(obs, next_obs, np.array([1]), np.array([1.0]), np.array([False]), [{}])
    sample = buffer.sample(8)
    assert set(sample.observations) == {"image", "features"}
    assert sample.observations["image"].shape == (8, 4, 4, 3)
    assert sample.discounts is not None
    assert sample.discounts.shape == (8, 1)
