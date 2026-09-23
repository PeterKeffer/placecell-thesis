"""Gym wrapper for frozen place-cell features."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .frozen_extractor import FrozenRepresentationExtractor


class PlaceCellObservationWrapper(gym.ObservationWrapper):
    """Emit frozen place-code features instead of raw observations."""

    def __init__(self, env: gym.Env, extractor: FrozenRepresentationExtractor):
        super().__init__(env)
        self.extractor = extractor
        self._last_action: int | None = None
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.extractor.feature_dim,),
            dtype=np.float32,
        )
        self._validate_required_inputs()

    def _validate_required_inputs(self) -> None:
        observation_space = getattr(self.env, "observation_space", None)
        if self.extractor.contract.requires_kinematics:
            if not isinstance(observation_space, gym.spaces.Dict):
                raise ValueError(
                    "The model contract requires kinematics, but the wrapped environment does not "
                    "expose a Dict observation space."
                )
            if "kinematics" not in observation_space.spaces:
                raise ValueError(
                    "The model contract requires kinematics, but the wrapped environment "
                    "does not expose an observation['kinematics'] field."
                )

    def _split_observation(self, observation: Any) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(observation, dict):
            if "rgb" in observation:
                rgb = np.asarray(observation["rgb"])
            elif "image" in observation:
                rgb = np.asarray(observation["image"])
            else:
                raise ValueError("Expected observation dict with `rgb` or `image`.")
            kinematics = (
                np.asarray(observation["kinematics"]) if "kinematics" in observation else None
            )
            return rgb, kinematics
        return np.asarray(observation), None

    def observation(self, observation: Any) -> np.ndarray:
        rgb, kinematics = self._split_observation(observation)
        return self.extractor.extract(rgb, previous_action=self._last_action, kinematics=kinematics)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.extractor.reset()
        self._last_action = None
        return self.observation(observation), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = int(action)
        return self.observation(observation), reward, terminated, truncated, info
