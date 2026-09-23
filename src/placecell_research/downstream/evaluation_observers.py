"""Concrete observers for downstream policy evaluation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluation import EvaluationObserver
from .policy_layer_maps import PolicyLayerActivationRecorder
from .trajectory_logging import RGBInputRecorder, TrainingTrajectoryRecorder


@dataclass(slots=True)
class TrajectoryRecorderObserver(EvaluationObserver):
    recorder: TrainingTrajectoryRecorder

    def on_reset(
        self,
        *,
        infos: list[dict[str, Any]],
        episode_indices: np.ndarray,
        timestep: int,
    ) -> None:
        del episode_indices
        self.recorder.observe_step(
            infos=infos,
            dones=np.zeros((len(infos),), dtype=bool),
            timestep=timestep,
        )

    def after_step(
        self,
        *,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
        episode_indices: np.ndarray,
        timestep: int,
    ) -> None:
        del episode_indices
        self.recorder.observe_step(infos=infos, dones=dones, timestep=timestep)


@dataclass(slots=True)
class RGBInputRecorderObserver(EvaluationObserver):
    recorder: RGBInputRecorder

    def on_reset(
        self,
        *,
        infos: list[dict[str, Any]],
        episode_indices: np.ndarray,
        timestep: int,
    ) -> None:
        del timestep
        self.recorder.observe_step(
            infos=infos,
            dones=np.zeros((len(infos),), dtype=bool),
            episode_indices=episode_indices,
        )

    def before_predict(
        self,
        *,
        observation: Any,
        infos: list[dict[str, Any]],
        episode_indices: np.ndarray,
    ) -> None:
        del infos
        self.recorder.observe_observation(
            observation=observation,
            episode_indices=episode_indices,
        )

    def after_step(
        self,
        *,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
        episode_indices: np.ndarray,
        timestep: int,
    ) -> None:
        del timestep
        self.recorder.observe_step(
            infos=infos,
            dones=dones,
            episode_indices=episode_indices,
        )


@dataclass(slots=True)
class PolicyLayerRecorderObserver(EvaluationObserver):
    recorder: PolicyLayerActivationRecorder

    def before_predict(
        self,
        *,
        observation: Any,
        infos: list[dict[str, Any]],
        episode_indices: np.ndarray,
    ) -> None:
        del observation
        self.recorder.prepare_step(infos, episode_indices=episode_indices)
