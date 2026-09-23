"""Goal place-code utilities for downstream RL."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from placecell_research.config.downstream_schema import DownstreamGoalCodeConfig

from .frozen_extractor import FrozenRepresentationExtractor


def normalize_goal_code(code: np.ndarray) -> np.ndarray:
    vector = np.asarray(code, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


@dataclass(slots=True)
class GoalPlaceCodeRuntime:
    """Shared place-code runtime for current-state and goal snapshot encoding."""

    current_extractor: FrozenRepresentationExtractor
    goal_snapshot_extractor: FrozenRepresentationExtractor
    snapshot_heading_radians: float
    normalize_codes: bool
    feature_dim: int = field(init=False)
    _goal_code_by_index: dict[int, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_dim = int(self.current_extractor.feature_dim)

    @classmethod
    def build(
        cls,
        *,
        model_artifact_path: Path,
        vision_encoder_path: Path | None,
        representation_source: str,
        device: str,
        checkpoint_selection: str,
        goal_code_config: DownstreamGoalCodeConfig,
    ) -> GoalPlaceCodeRuntime:
        return cls(
            current_extractor=FrozenRepresentationExtractor(
                model_checkpoint=model_artifact_path,
                vision_encoder=vision_encoder_path,
                device=device,
                representation_source=representation_source,
                checkpoint_selection=checkpoint_selection,
            ),
            goal_snapshot_extractor=FrozenRepresentationExtractor(
                model_checkpoint=model_artifact_path,
                vision_encoder=vision_encoder_path,
                device=device,
                representation_source=representation_source,
                checkpoint_selection=checkpoint_selection,
            ),
            snapshot_heading_radians=math.radians(float(goal_code_config.snapshot_heading_degrees)),
            normalize_codes=bool(goal_code_config.normalize_codes),
        )

    def reset_episode(self) -> None:
        self.current_extractor.reset()

    def _normalize_if_configured(self, code: np.ndarray) -> np.ndarray:
        if not self.normalize_codes:
            return np.asarray(code, dtype=np.float32).reshape(-1)
        return normalize_goal_code(code)

    def current_place_code(
        self,
        *,
        rgb: np.ndarray,
        previous_action: int | None,
        kinematics: np.ndarray | None,
    ) -> np.ndarray:
        code = self.current_extractor.extract(
            rgb=rgb,
            previous_action=previous_action,
            kinematics=kinematics,
        )
        return self._normalize_if_configured(code)

    def goal_place_code(
        self,
        *,
        adapter: Any,
        goal_xy: np.ndarray,
        goal_index: int | None,
    ) -> np.ndarray:
        if goal_index is not None and goal_index in self._goal_code_by_index:
            return self._goal_code_by_index[goal_index].copy()
        code = self._encode_goal_snapshot(adapter=adapter, goal_xy=goal_xy)
        if goal_index is not None:
            self._goal_code_by_index[goal_index] = code.copy()
        return code

    def _goal_snapshot_kinematics(self) -> np.ndarray | None:
        required_width = int(self.goal_snapshot_extractor.contract.required_kinematics_width)
        if required_width <= 0:
            return None
        return np.zeros((required_width,), dtype=np.float32)

    def _encode_goal_snapshot(self, *, adapter: Any, goal_xy: np.ndarray) -> np.ndarray:
        raw_env = getattr(adapter, "_env", None)
        env = getattr(raw_env, "unwrapped", raw_env)
        agent = getattr(env, "agent", None)
        render_observation = getattr(env, "render_obs", None)
        if agent is None or not callable(render_observation):
            raise RuntimeError(
                "goal_place_code requires a MiniWorld environment with direct agent pose "
                "access and render_obs()."
            )
        original_position = np.asarray(
            getattr(agent, "pos", np.zeros(3, dtype=float)),
            dtype=float,
        ).copy()
        original_heading = float(getattr(agent, "dir", 0.0))
        goal_xy_array = np.asarray(goal_xy, dtype=np.float32).reshape(2)
        try:
            agent.pos = np.asarray([goal_xy_array[0], 0.0, goal_xy_array[1]], dtype=float)
            agent.dir = float(self.snapshot_heading_radians)
            rendered_rgb = np.asarray(render_observation())
            self.goal_snapshot_extractor.reset()
            code = self.goal_snapshot_extractor.extract(
                rgb=rendered_rgb,
                previous_action=0,
                kinematics=self._goal_snapshot_kinematics(),
            )
            return self._normalize_if_configured(code)
        finally:
            agent.pos = original_position
            agent.dir = original_heading
