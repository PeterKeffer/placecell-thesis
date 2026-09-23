"""Collection policies and config helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from placecell_research.config.loader import (
    _load_yaml,
    _resolve_defaults,
    apply_overrides,
    stamp_l1_semantics_marker,
)
from placecell_research.config.schema import ContinuousMotionConfig
from placecell_research.envs.base import ContinuousMotion


@runtime_checkable
class Sampler(Protocol):
    """Action source for collection: decides the next action given an observation."""

    def reset(self, adapter: Any, episode_seed: int, episode_index: int) -> None: ...

    def sample(self, observation: Any) -> int | ContinuousMotion: ...


class ContinuousRandomPolicy:
    """Sample continuous motion and keep turning away while wall contact persists."""

    def __init__(self, config: ContinuousMotionConfig) -> None:
        self.config = config

    def reset(self, adapter: Any, episode_seed: int, episode_index: int) -> None:
        del episode_index
        if not callable(getattr(adapter, "step_motion", None)):
            raise ValueError("continuous_random requires an adapter with step_motion support.")
        self.rng = np.random.default_rng(episode_seed)
        self.wall_turn_direction = 0

    def sample(self, observation: Any = None) -> ContinuousMotion:
        distance = float(self.rng.rayleigh(self.config.forward_distance_scale))
        turn = float(self.rng.normal(0.0, self.config.turn_std_radians))
        blocked = observation is not None and observation.info.get("motion_blocked", False)
        if blocked and self.config.wall_turn_radians:
            if self.wall_turn_direction == 0:
                self.wall_turn_direction = 1 if turn >= 0 else -1
            turn += self.wall_turn_direction * self.config.wall_turn_radians
        else:
            self.wall_turn_direction = 0
        return ContinuousMotion(
            forward_distance=distance,
            turn_radians=turn,
        )


class RandomDiscretePolicy:
    """Random policy with OG-style OU smoothing for 3-action MiniWorld."""

    def __init__(
        self,
        num_actions: int,
        seed: int,
        mode: str = "ou_smoothed_random",
        *,
        action_names: list[str] | None = None,
        action_probabilities: dict[str, float] | None = None,
    ) -> None:
        self.num_actions = int(num_actions)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.decay = 0.85
        self.noise_scale = 0.4
        self.tendency = 0.0
        self.action_probabilities = self._resolve_action_probabilities(
            action_names=action_names,
            action_probabilities=action_probabilities or {},
        )
        self.left_threshold, self.right_threshold = self._resolve_thresholds(
            action_names=action_names,
            action_probabilities=action_probabilities or {},
        )

    def _resolve_action_probabilities(
        self,
        *,
        action_names: list[str] | None,
        action_probabilities: dict[str, float],
    ) -> np.ndarray | None:
        if not action_probabilities:
            return None
        if self.mode not in {"ou_smoothed_random", "independent_random"}:
            raise ValueError(
                "Action probabilities are only supported for OU-smoothed or independent "
                "random policies."
            )
        if action_names is None:
            raise ValueError(
                "Action probabilities require named actions, but no action_names were provided."
            )
        if len(action_names) != self.num_actions:
            raise ValueError(
                f"Expected {self.num_actions} action names for probability resolution, "
                f"got {len(action_names)}."
            )
        unknown_action_names = sorted(
            name for name in action_probabilities if name not in action_names
        )
        if unknown_action_names:
            raise ValueError(
                f"Unknown action probability names {unknown_action_names!r}. "
                f"Available actions: {action_names!r}."
            )
        missing_action_names = [name for name in action_names if name not in action_probabilities]
        if missing_action_names:
            raise ValueError(
                "Named action probabilities must cover every action. "
                f"Missing {missing_action_names!r} from {action_names!r}."
            )
        probabilities = np.asarray(
            [float(action_probabilities[name]) for name in action_names], dtype=np.float64
        )
        probability_sum = float(np.sum(probabilities))
        if probability_sum <= 0.0:
            raise ValueError("Resolved action probabilities must sum to a positive value.")
        return probabilities / probability_sum

    def _resolve_thresholds(
        self,
        *,
        action_names: list[str] | None,
        action_probabilities: dict[str, float],
    ) -> tuple[float, float]:
        if not action_probabilities:
            return -2.0 + (4.0 / 3.0), -2.0 + (8.0 / 3.0)
        if self.mode != "ou_smoothed_random":
            return -2.0 + (4.0 / 3.0), -2.0 + (8.0 / 3.0)
        if self.num_actions != 3:
            raise ValueError(
                "Named action probabilities for OU smoothing require exactly three actions."
            )
        assert self.action_probabilities is not None
        ordered_probabilities = self.action_probabilities
        total_range = 4.0
        left_width = float(ordered_probabilities[0]) * total_range
        forward_width = float(ordered_probabilities[2]) * total_range
        left_threshold = -2.0 + left_width
        right_threshold = -2.0 + left_width + forward_width
        return left_threshold, right_threshold

    def reset(
        self,
        adapter: Any = None,
        episode_seed: int | None = None,
        episode_index: int | None = None,
    ) -> None:
        del adapter, episode_seed, episode_index
        if self.mode == "ou_smoothed_random":
            self.tendency = float(self.rng.standard_normal() * 0.3)

    def sample(self, observation: Any = None) -> int:
        del observation
        if self.mode == "uniform_random":
            return int(self.rng.integers(0, self.num_actions))
        if self.mode == "independent_random":
            if self.action_probabilities is None:
                return int(self.rng.integers(0, self.num_actions))
            return int(self.rng.choice(self.num_actions, p=self.action_probabilities))
        if self.num_actions == 3:
            if self.tendency < self.left_threshold:
                action = 0
            elif self.tendency > self.right_threshold:
                action = 1
            else:
                action = 2
        else:
            normalized_tendency = np.clip((self.tendency + 2.0) / 4.0, 0.0, 0.999)
            action = int(normalized_tendency * self.num_actions)
        self.tendency = self.tendency * self.decay + float(
            self.rng.standard_normal() * self.noise_scale
        )
        return action


class MotionBoutPolicy:
    """Persistent translation/rotation bouts with collision-triggered turning."""

    def __init__(self, action_names: list[str], switch_probability: float) -> None:
        required = {"move_forward", "turn_left", "turn_right"}
        if set(action_names) != required:
            raise ValueError("motion_bouts requires move_forward, turn_left and turn_right.")
        self.forward = action_names.index("move_forward")
        self.turns = [action_names.index("turn_left"), action_names.index("turn_right")]
        self.switch_probability = switch_probability

    def reset(self, adapter: Any, episode_seed: int, episode_index: int) -> None:
        del adapter, episode_index
        self.rng = np.random.default_rng(episode_seed)
        self.action = self.forward
        self.previous_position = None

    def sample(self, observation: Any) -> int:
        position = np.asarray(observation.position_xy)
        collision = (
            self.action == self.forward and self.previous_position is not None
            and float(np.linalg.norm(position - self.previous_position)) < 1e-6
        )
        if collision or self.rng.random() < self.switch_probability:
            self.action = (
                int(self.rng.choice(self.turns)) if self.action == self.forward else self.forward
            )
        self.previous_position = position.copy()
        return self.action


@dataclass
class ResumePolicy:
    """Resolved policy snapshot."""

    artifact_reuse: str
    training_resume: str


def load_raw_config_payload(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    """Load config including keys not represented in typed dataclasses."""
    payload = _resolve_defaults(config_path, _load_yaml(config_path))
    return stamp_l1_semantics_marker(apply_overrides(payload, overrides, config_path=config_path))


def resolve_policies(raw_payload: dict[str, Any]) -> ResumePolicy:
    """Resolve reuse and resume policies from raw config."""
    policies = raw_payload.get("policies", {})
    return ResumePolicy(
        artifact_reuse=str(policies.get("artifact_reuse", "error")),
        training_resume=str(policies.get("training_resume", "fresh")),
    )
