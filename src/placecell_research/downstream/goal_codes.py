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


def compute_goal_code_distance(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    *,
    metric: str,
    normalize_codes: bool,
) -> np.ndarray:
    achieved = np.asarray(achieved_goal, dtype=np.float32)
    desired = np.asarray(desired_goal, dtype=np.float32)
    if achieved.ndim == 1:
        achieved = achieved.reshape(1, -1)
    if desired.ndim == 1:
        desired = desired.reshape(1, -1)
    if achieved.shape != desired.shape:
        raise ValueError(
            "Goal-code distance requires achieved_goal and desired_goal with matching shapes. "
            f"Got {tuple(achieved.shape)} and {tuple(desired.shape)}."
        )
    if normalize_codes:
        achieved = np.stack([normalize_goal_code(row) for row in achieved], axis=0)
        desired = np.stack([normalize_goal_code(row) for row in desired], axis=0)
    metric_name = str(metric).strip().lower()
    if metric_name == "l2":
        return np.linalg.norm(achieved - desired, axis=-1).astype(np.float32, copy=False)
    if metric_name == "cosine":
        achieved_norm = np.linalg.norm(achieved, axis=-1)
        desired_norm = np.linalg.norm(desired, axis=-1)
        norm_product = achieved_norm * desired_norm
        safe_denominator = np.where(norm_product <= 1e-8, 1.0, norm_product)
        similarity = np.sum(achieved * desired, axis=-1) / safe_denominator
        return (1.0 - similarity).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported goal-code distance metric: {metric!r}")


@dataclass(frozen=True, slots=True)
class GoalSuccessMetric:
    """Complete distance-and-threshold contract for one goal transition."""

    distance_metric: str
    normalize_codes: bool
    success_threshold: float

    def __post_init__(self) -> None:
        metric = str(self.distance_metric).strip().lower()
        if metric not in {"l2", "cosine"}:
            raise ValueError(f"Unsupported goal distance metric: {self.distance_metric!r}")
        threshold = float(self.success_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Goal success threshold must be positive and finite.")
        object.__setattr__(self, "distance_metric", metric)
        object.__setattr__(self, "normalize_codes", bool(self.normalize_codes))
        object.__setattr__(self, "success_threshold", threshold)

    @classmethod
    def from_step_info(
        cls,
        info: dict[str, Any] | None,
        *,
        code_space: bool | None = None,
        completed_goal: bool = False,
        default_success_threshold: float | None = None,
    ) -> GoalSuccessMetric:
        """Resolve the metric carried by a goal-conditioned environment step."""

        step_info = info if isinstance(info, dict) else {}
        uses_code_space = (
            "goal_code_success_threshold" in step_info
            if code_space is None
            else bool(code_space)
        )
        if uses_code_space:
            raw_threshold = step_info.get("goal_code_success_threshold", 0.35)
            return cls(
                distance_metric=str(step_info.get("goal_code_distance_metric", "l2")),
                normalize_codes=bool(step_info.get("goal_code_normalize", True)),
                success_threshold=float(0.35 if raw_threshold is None else raw_threshold),
            )

        threshold_key = (
            "completed_goal_reach_radius" if completed_goal else "goal_reach_radius"
        )
        raw_threshold = step_info.get(threshold_key)
        if raw_threshold is None and completed_goal:
            raw_threshold = step_info.get("goal_reach_radius")
        if raw_threshold is None:
            raw_threshold = default_success_threshold
        if raw_threshold is None:
            raise RuntimeError(
                "Goal-coordinate replay requires a goal reach radius on every step."
            )
        return cls(
            distance_metric="l2",
            normalize_codes=False,
            success_threshold=float(raw_threshold),
        )

    def distances(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        return compute_goal_code_distance(
            achieved_goal,
            desired_goal,
            metric=self.distance_metric,
            normalize_codes=self.normalize_codes,
        )

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> bool:
        distances = self.distances(achieved_goal, desired_goal)
        if distances.size != 1:
            raise ValueError("is_success() requires exactly one goal transition.")
        return bool(float(distances[0]) < self.success_threshold)

    def state_dict(self) -> dict[str, Any]:
        return {
            "distance_metric": self.distance_metric,
            "normalize_codes": self.normalize_codes,
            "success_threshold": self.success_threshold,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> GoalSuccessMetric:
        return cls(
            distance_metric=str(payload["distance_metric"]),
            normalize_codes=bool(payload["normalize_codes"]),
            success_threshold=float(payload["success_threshold"]),
        )


@dataclass(slots=True)
class GoalPlaceCodeRuntime:
    """Shared place-code runtime for current-state and goal snapshot encoding."""

    current_extractor: FrozenRepresentationExtractor
    goal_snapshot_extractor: FrozenRepresentationExtractor
    snapshot_heading_radians: float
    normalize_codes: bool
    distance_metric: str
    success_threshold: float
    feature_dim: int = field(init=False)
    _goal_codebook_by_index: dict[int, np.ndarray] = field(init=False, default_factory=dict)

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
            distance_metric=str(goal_code_config.distance_metric),
            success_threshold=float(goal_code_config.success_threshold),
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
        if goal_index is not None and goal_index in self._goal_codebook_by_index:
            return self._goal_codebook_by_index[goal_index].copy()
        code = self._encode_goal_snapshot(adapter=adapter, goal_xy=goal_xy)
        if goal_index is not None:
            self._goal_codebook_by_index[goal_index] = code.copy()
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
