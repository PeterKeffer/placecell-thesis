"""Goal-aware online environments for downstream RL."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from placecell_research.config.downstream_schema import (
    DownstreamGoalTaskConfig,
    DownstreamSpawnCurriculumPhaseConfig,
)
from placecell_research.config.schema import EnvironmentConfig
from placecell_research.downstream.feature_sources import FeatureConcatenator, StepContext
from placecell_research.downstream.goal_codes import GoalPlaceCodeRuntime
from placecell_research.downstream.synthetic_grid_cells import GridCodeEncoder
from placecell_research.downstream.synthetic_place_cells import (
    PlaceCodeGoalCodebook,
    SyntheticPlaceCodeEncoder,
)
from placecell_research.envs import assert_environment_supports
from placecell_research.envs.builder import build_environment


@dataclass(slots=True)
class RolloutSummary:
    episodes: int
    mean_return: float
    success_rate: float
    mean_steps: float


class GoalScheduler:
    """Episode-level goal scheduler with fixed, cyclic, or random policies."""

    def __init__(self, config: DownstreamGoalTaskConfig, seed: int) -> None:
        self.positions = [
            np.asarray(position, dtype=np.float32) for position in config.candidate_positions_xy
        ]
        self.schedule = config.schedule
        self.change_interval_episodes = int(config.change_interval_episodes)
        self._random = random.Random(seed)
        self._initial_index = int(config.initial_goal_index) if self.positions else -1
        self._current_index = -1

    @property
    def has_goals(self) -> bool:
        return bool(self.positions)

    @property
    def current_index(self) -> int | None:
        if self._current_index < 0:
            return None
        return int(self._current_index)

    def choose_goal(self, episode_index: int) -> np.ndarray | None:
        if not self.positions:
            return None
        if self.schedule == "fixed":
            self._current_index = max(0, min(len(self.positions) - 1, self._initial_index))
            return self.positions[self._current_index].copy()
        if self._current_index < 0:
            self._current_index = max(0, min(len(self.positions) - 1, self._initial_index))
            return self.positions[self._current_index].copy()
        if episode_index % max(1, self.change_interval_episodes) == 0:
            if self.schedule == "cycle":
                self._current_index = (self._current_index + 1) % len(self.positions)
            elif self.schedule == "random":
                self._current_index = self._random.randrange(len(self.positions))
            else:
                raise ValueError(f"Unsupported goal schedule: {self.schedule}")
        return self.positions[self._current_index].copy()

    def advance_goal(self) -> np.ndarray | None:
        """Choose the next goal without resetting the environment state."""
        if not self.positions:
            return None
        if self._current_index < 0:
            self._current_index = max(0, min(len(self.positions) - 1, self._initial_index))
        if self.schedule == "cycle":
            self._current_index = (self._current_index + 1) % len(self.positions)
        elif self.schedule == "random" and len(self.positions) > 1:
            candidate = self._random.randrange(len(self.positions) - 1)
            if candidate >= self._current_index:
                candidate += 1
            self._current_index = candidate
        elif self.schedule != "fixed":
            raise ValueError(
                f"Goal chaining requires fixed, cycle, or random scheduling, got {self.schedule!r}."
            )
        return self.positions[self._current_index].copy()


def _uses_uniform_random_goal_schedule(schedule: object) -> bool:
    return str(schedule).strip().lower() == "uniform_random"


def _sample_bounds_by_area(
    bounds_by_region: dict[str, tuple[float, float, float, float]],
    random_uniform,
) -> tuple[float, float, float, float]:
    weighted_regions: list[tuple[str, float]] = []
    total_area = 0.0
    for region_name, bounds in bounds_by_region.items():
        min_x, max_x, min_z, max_z = bounds
        area = max(0.0, float(max_x - min_x)) * max(0.0, float(max_z - min_z))
        if area <= 0.0:
            continue
        weighted_regions.append((region_name, area))
        total_area += area
    if not weighted_regions:
        return next(iter(bounds_by_region.values()))
    threshold = float(random_uniform(0.0, total_area))
    cumulative_area = 0.0
    for region_name, area in weighted_regions:
        cumulative_area += area
        if threshold <= cumulative_area:
            return bounds_by_region[region_name]
    return bounds_by_region[weighted_regions[-1][0]]


class OnlineKinematicsTracker:
    """Stepwise predictor kinematics channels for downstream feature extraction."""

    def __init__(self) -> None:
        self._previous_position_xy: np.ndarray | None = None
        self._previous_heading: float | None = None

    def reset(self) -> np.ndarray:
        self._previous_position_xy = None
        self._previous_heading = None
        return np.zeros(4, dtype=np.float32)

    def update(self, position_xy: np.ndarray, heading: float) -> np.ndarray:
        position_xy = np.asarray(position_xy, dtype=np.float32)
        if self._previous_position_xy is None or self._previous_heading is None:
            self._previous_position_xy = position_xy.copy()
            self._previous_heading = float(heading)
            return np.asarray(
                [0.0, 0.0, math.sin(float(heading)), math.cos(float(heading))], dtype=np.float32
            )
        displacement = position_xy - self._previous_position_xy
        step_displacement = float(np.linalg.norm(displacement))
        heading_delta = math.atan2(
            math.sin(float(heading) - self._previous_heading),
            math.cos(float(heading) - self._previous_heading),
        )
        self._previous_position_xy = position_xy.copy()
        self._previous_heading = float(heading)
        return np.asarray(
            [step_displacement, heading_delta, math.sin(float(heading)), math.cos(float(heading))],
            dtype=np.float32,
        )


class DownstreamNavigationEnv(gym.Env[Any, int]):
    """Online MiniWorld environment for either feature-vector or raw-pixel observations."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        *,
        environment_config: EnvironmentConfig,
        goal_task_config: DownstreamGoalTaskConfig,
        feature_extractor: FeatureConcatenator | None,
        goal_place_code_runtime: GoalPlaceCodeRuntime | None = None,
        grid_code_encoder: GridCodeEncoder | None = None,
        synthetic_place_code_encoder: SyntheticPlaceCodeEncoder | None = None,
        goal_codebook: PlaceCodeGoalCodebook | None = None,
        observation_mode: str,
        seed: int,
    ) -> None:
        super().__init__()
        self.environment_config = environment_config
        self.goal_task_config = goal_task_config
        self.goal_scheduler = GoalScheduler(goal_task_config, seed=seed)
        self.feature_extractor = feature_extractor
        self.goal_place_code_runtime = goal_place_code_runtime
        self.grid_code_encoder = grid_code_encoder
        self.synthetic_place_code_encoder = synthetic_place_code_encoder
        self.goal_codebook = goal_codebook
        self._goal_codebook_random = np.random.default_rng(int(seed) + 2_003)
        self.observation_mode = observation_mode
        self.seed = int(seed)
        self._base_spawn_regions = [
            str(region).lower()
            for region in self.environment_config.env_kwargs.get("spawn_regions", ["all"])
        ]
        base_spawn_region_xz = self.environment_config.env_kwargs.get("spawn_region_xz")
        self._base_spawn_region_xz = (
            None
            if base_spawn_region_xz is None
            else tuple(float(value) for value in base_spawn_region_xz)
        )
        self._base_goal_schedule = str(goal_task_config.schedule)
        self._base_goal_change_interval_episodes = int(goal_task_config.change_interval_episodes)
        self._base_goal_index = int(goal_task_config.initial_goal_index)
        self._episode_index = 0
        self._previous_action: int | None = None
        self._kinematics = OnlineKinematicsTracker()
        self._current_goal_xy: np.ndarray | None = None
        self._current_goal_index: int | None = None
        self._current_goal_place_code: np.ndarray | None = None
        self._current_place_code: np.ndarray | None = None
        self._current_goal_grid_code: np.ndarray | None = None
        self._current_grid_code: np.ndarray | None = None
        self._last_rgb_frame_hwc: np.ndarray | None = None
        self._last_adapter_observation = None
        self._curriculum_phase: DownstreamSpawnCurriculumPhaseConfig | None = None
        self._curriculum_spawn_regions: list[str] | None = None
        self._curriculum_spawn_region_xz: tuple[float, float, float, float] | None = None
        self._curriculum_max_spawn_distance: float | None = None
        self._curriculum_goal_tolerance: float | None = None
        self._curriculum_goal_schedule: str | None = None
        self._curriculum_goal_change_interval_episodes: int | None = None
        self._curriculum_goal_index: int | None = None
        self._curriculum_goal_current_index = -1
        self._curriculum_goal_random = random.Random(int(seed) + 1_003)
        self._adapter = self._build_adapter()
        required_modalities = ("rgb",)
        assert_environment_supports(
            self._adapter,
            consumer="downstream online RL",
            required=("position_xy", "heading", "discrete_actions"),
            required_modalities=required_modalities,
        )
        self.action_space = gym.spaces.Discrete(int(self._adapter.action_space.count))
        if self.observation_mode == "feature_vector":
            if self.feature_extractor is None:
                raise ValueError("feature_vector mode requires a feature extractor.")
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(self.feature_extractor.feature_dim),),
                dtype=np.float32,
            )
        elif self.observation_mode == "raw_pixels":
            channels, height, width = self._adapter.modality_shapes["rgb"]
            image_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(height, width, channels),
                dtype=np.uint8,
            )
            if self.feature_extractor is None:
                self.observation_space = image_space
            else:
                self.observation_space = gym.spaces.Dict(
                    {
                        "image": image_space,
                        "features": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(int(self.feature_extractor.feature_dim),),
                            dtype=np.float32,
                        ),
                    }
                )
        else:
            raise ValueError(f"Unsupported downstream observation mode: {self.observation_mode}")
        self._seed_spaces(self.seed)

    def _seed_spaces(self, seed: int) -> None:
        self.action_space.seed(int(seed))
        self.observation_space.seed(int(seed))

    def _seed_internal_rngs(self, seed: int) -> None:
        self.seed = int(seed)
        self.goal_scheduler = GoalScheduler(self.goal_task_config, seed=self.seed)
        self._curriculum_goal_random = random.Random(self.seed + 1_003)
        self._goal_codebook_random = np.random.default_rng(self.seed + 2_003)
        self._curriculum_goal_current_index = -1
        self._seed_spaces(self.seed)

    def _build_adapter(self):
        adapter = build_environment(self.environment_config, seed=self.seed)
        if self.goal_scheduler.has_goals or _uses_uniform_random_goal_schedule(
            self._base_goal_schedule
        ):
            assert_environment_supports(
                adapter,
                consumer="goal-conditioned downstream RL",
                required=("goal_override",),
            )
        return adapter

    def _get_unwrapped_env(self) -> Any:
        raw_env = getattr(self._adapter, "_env", None)
        return getattr(raw_env, "unwrapped", raw_env)

    def _apply_runtime_curriculum_state(self) -> None:
        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return
        if hasattr(unwrapped, "spawn_regions"):
            effective_regions = (
                self._base_spawn_regions
                if self._curriculum_spawn_regions is None
                else self._curriculum_spawn_regions
            )
            unwrapped.spawn_regions = list(effective_regions)
        elif self._curriculum_spawn_regions is not None:
            raise RuntimeError(
                "Curriculum spawn_regions override requires an env with mutable spawn_regions."
            )
        if hasattr(unwrapped, "spawn_region_xz"):
            effective_region_xz = (
                self._base_spawn_region_xz
                if self._curriculum_spawn_region_xz is None
                else self._curriculum_spawn_region_xz
            )
            unwrapped.spawn_region_xz = (
                None if effective_region_xz is None else tuple(effective_region_xz)
            )
        elif self._curriculum_spawn_region_xz is not None:
            raise RuntimeError(
                "Curriculum spawn_region_xz override requires an env with mutable spawn_region_xz."
            )
        if hasattr(unwrapped, "goal_tolerance"):
            unwrapped.goal_tolerance = (
                None
                if self._curriculum_goal_tolerance is None
                else float(self._curriculum_goal_tolerance)
            )
        elif self._curriculum_goal_tolerance is not None:
            raise RuntimeError(
                "Curriculum goal_tolerance override requires an env with mutable goal_tolerance."
            )

    def apply_curriculum_phase(self, phase: DownstreamSpawnCurriculumPhaseConfig | None) -> None:
        self._curriculum_phase = phase
        self._curriculum_spawn_regions = None
        self._curriculum_spawn_region_xz = None
        self._curriculum_max_spawn_distance = None
        self._curriculum_goal_tolerance = None
        self._curriculum_goal_schedule = None
        self._curriculum_goal_change_interval_episodes = None
        self._curriculum_goal_index = None
        self._curriculum_goal_current_index = -1
        if phase is not None:
            self._curriculum_spawn_regions = (
                None
                if not phase.spawn_regions
                else [str(region).lower() for region in phase.spawn_regions]
            )
            if phase.spawn_region_xz is not None:
                values = tuple(float(value) for value in phase.spawn_region_xz)
                if len(values) != 4:
                    raise ValueError("Curriculum spawn_region_xz must contain exactly 4 floats.")
                self._curriculum_spawn_region_xz = values
            self._curriculum_max_spawn_distance = phase.max_spawn_distance
            self._curriculum_goal_tolerance = phase.goal_tolerance
            self._curriculum_goal_schedule = phase.goal_schedule
            self._curriculum_goal_change_interval_episodes = phase.goal_change_interval_episodes
            self._curriculum_goal_index = phase.goal_index

    def _choose_curriculum_candidate_goal(
        self,
        *,
        schedule: str,
        change_interval_episodes: int,
        initial_goal_index: int,
    ) -> np.ndarray | None:
        positions = self.goal_scheduler.positions
        if not positions:
            return None
        clamped_initial_index = max(0, min(len(positions) - 1, int(initial_goal_index)))
        if schedule == "fixed":
            self._curriculum_goal_current_index = clamped_initial_index
            return positions[self._curriculum_goal_current_index].copy()
        if self._curriculum_goal_current_index < 0:
            self._curriculum_goal_current_index = clamped_initial_index
            return positions[self._curriculum_goal_current_index].copy()
        if self._episode_index % max(1, int(change_interval_episodes)) == 0:
            if schedule == "cycle":
                self._curriculum_goal_current_index = (
                    self._curriculum_goal_current_index + 1
                ) % len(positions)
            elif schedule == "random":
                self._curriculum_goal_current_index = self._curriculum_goal_random.randrange(
                    len(positions)
                )
            else:
                raise ValueError(f"Unsupported curriculum goal schedule: {schedule}")
        return positions[self._curriculum_goal_current_index].copy()

    def _choose_curriculum_candidate_goal_with_index(
        self,
        *,
        schedule: str,
        change_interval_episodes: int,
        initial_goal_index: int,
    ) -> tuple[int | None, np.ndarray | None]:
        goal_xy = self._choose_curriculum_candidate_goal(
            schedule=schedule,
            change_interval_episodes=change_interval_episodes,
            initial_goal_index=initial_goal_index,
        )
        if goal_xy is None or self._curriculum_goal_current_index < 0:
            return None, goal_xy
        return int(self._curriculum_goal_current_index), goal_xy

    def _sample_uniform_random_goal_xy(self) -> np.ndarray | None:
        unwrapped = self._get_unwrapped_env()
        bounds_by_name = getattr(unwrapped, "full_room_bounds_by_name", None)
        if not bounds_by_name:
            return self._sample_uniform_goal_via_adapter()
        collision_fn = getattr(unwrapped, "intersect", None)
        agent = getattr(unwrapped, "agent", None)
        goal_object_radius = float(getattr(unwrapped, "goal_object_radius", 0.0))
        if not callable(collision_fn) or agent is None:
            raise RuntimeError(
                "MiniWorld uniform_random goal sampling requires an agent and collision check."
            )
        for _ in range(100):
            min_x, max_x, min_z, max_z = _sample_bounds_by_area(
                bounds_by_name,
                self._curriculum_goal_random.uniform,
            )
            candidate_xy = np.asarray(
                [
                    float(self._curriculum_goal_random.uniform(min_x, max_x)),
                    float(self._curriculum_goal_random.uniform(min_z, max_z)),
                ],
                dtype=np.float32,
            )
            candidate_position = np.asarray([candidate_xy[0], 0.0, candidate_xy[1]], dtype=float)
            try:
                if not collision_fn(agent, candidate_position, goal_object_radius):
                    return candidate_xy
            except Exception as exc:
                raise RuntimeError(
                    "MiniWorld uniform_random goal collision checking failed."
                ) from exc
        raise RuntimeError(
            "MiniWorld uniform_random goal sampling exhausted the collision-free retry budget."
        )

    def _sample_uniform_goal_via_adapter(self) -> np.ndarray | None:
        room_bounds_fn = getattr(self._adapter, "room_bounds_xy", None)
        is_free_fn = getattr(self._adapter, "is_free_position_xy", None)
        if not callable(room_bounds_fn) or not callable(is_free_fn):
            raise RuntimeError(
                "goal_task.schedule='uniform_random' needs either MiniWorld room bounds or an "
                "adapter exposing room_bounds_xy() + is_free_position_xy()."
            )
        regions = {str(i): tuple(b) for i, b in enumerate(room_bounds_fn())}
        if not regions:
            raise RuntimeError("uniform_random goal sampling requires at least one arena region.")
        for _ in range(200):
            min_x, max_x, min_y, max_y = _sample_bounds_by_area(
                regions, self._curriculum_goal_random.uniform
            )
            candidate_xy = np.asarray(
                [
                    float(self._curriculum_goal_random.uniform(min_x, max_x)),
                    float(self._curriculum_goal_random.uniform(min_y, max_y)),
                ],
                dtype=np.float32,
            )
            if is_free_fn(candidate_xy):
                return candidate_xy
        raise RuntimeError(
            "uniform_random goal sampling exhausted the collision-free retry budget."
        )

    def _resolve_goal_for_reset(self) -> tuple[int | None, np.ndarray | None]:
        phase = self._curriculum_phase
        if phase is None:
            if _uses_uniform_random_goal_schedule(self._base_goal_schedule):
                return None, self._sample_uniform_random_goal_xy()
            goal_xy = self.goal_scheduler.choose_goal(self._episode_index)
            return self.goal_scheduler.current_index, goal_xy
        if self._curriculum_goal_index is not None:
            goal_index = int(self._curriculum_goal_index)
            return goal_index, self.goal_scheduler.positions[goal_index].copy()
        if (
            self._curriculum_goal_schedule is None
            and self._curriculum_goal_change_interval_episodes is None
        ):
            goal_xy = self.goal_scheduler.choose_goal(self._episode_index)
            return self.goal_scheduler.current_index, goal_xy
        schedule = (
            self._base_goal_schedule
            if self._curriculum_goal_schedule is None
            else self._curriculum_goal_schedule
        )
        change_interval_episodes = (
            self._base_goal_change_interval_episodes
            if self._curriculum_goal_change_interval_episodes is None
            else int(self._curriculum_goal_change_interval_episodes)
        )
        if _uses_uniform_random_goal_schedule(schedule):
            return None, self._sample_uniform_random_goal_xy()
        return self._choose_curriculum_candidate_goal_with_index(
            schedule=schedule,
            change_interval_episodes=change_interval_episodes,
            initial_goal_index=self._base_goal_index,
        )

    def _spawn_distance_satisfied(self, observation) -> bool:
        if self._curriculum_max_spawn_distance is None or self._current_goal_xy is None:
            return True
        agent_xy = np.asarray(observation.position_xy, dtype=np.float32).reshape(2)
        goal_xy = np.asarray(self._current_goal_xy, dtype=np.float32).reshape(2)
        return float(np.linalg.norm(agent_xy - goal_xy)) <= float(
            self._curriculum_max_spawn_distance
        )

    def _reset_feature_state(self) -> None:
        self._previous_action = None
        self._kinematics.reset()
        self._current_place_code = None
        self._current_grid_code = None
        if self.feature_extractor is not None:
            self.feature_extractor.reset()
        if self.goal_place_code_runtime is not None:
            self.goal_place_code_runtime.reset_episode()

    def _uses_codebook_goal_schedule(self) -> bool:
        return (
            self.goal_codebook is not None
            and str(self.goal_task_config.schedule).strip().lower() == "codebook"
        )

    def _resolve_goal_place_code(self) -> np.ndarray | None:
        if self._current_goal_xy is None:
            return None
        if self.synthetic_place_code_encoder is not None:
            return self.synthetic_place_code_encoder.encode(self._current_goal_xy)
        if self.goal_place_code_runtime is None:
            return None
        return self.goal_place_code_runtime.goal_place_code(
            adapter=self._adapter,
            goal_xy=self._current_goal_xy,
            goal_index=self._current_goal_index,
        )

    def _resolve_goal_grid_code(self) -> np.ndarray | None:
        if self.grid_code_encoder is None or self._current_goal_xy is None:
            return None
        return self.grid_code_encoder.encode(self._current_goal_xy)

    def _render_observation(self, observation) -> np.ndarray:
        rgb = np.asarray(observation.require_modality("rgb"))
        if rgb.shape[0] in {1, 3}:
            self._last_rgb_frame_hwc = np.transpose(rgb, (1, 2, 0)).astype(np.uint8, copy=False)
        else:
            self._last_rgb_frame_hwc = rgb.astype(np.uint8, copy=False)
        kinematics = self._kinematics.update(observation.position_xy, observation.heading)
        current_place_code = None
        if self.goal_place_code_runtime is not None:
            current_place_code = self.goal_place_code_runtime.current_place_code(
                rgb=rgb,
                previous_action=self._previous_action,
                kinematics=kinematics,
            )
        self._current_place_code = None if current_place_code is None else current_place_code.copy()
        if self.synthetic_place_code_encoder is not None:
            self._current_place_code = self.synthetic_place_code_encoder.encode(
                observation.position_xy
            )
        if self.grid_code_encoder is not None:
            self._current_grid_code = self.grid_code_encoder.encode(observation.position_xy)
        raw_image = (
            np.transpose(rgb, (1, 2, 0)).astype(np.uint8, copy=False)
            if rgb.shape[0] in {1, 3}
            else rgb.astype(np.uint8, copy=False)
        )
        context = StepContext(
            rgb=rgb,
            position_xy=np.asarray(observation.position_xy, dtype=np.float32),
            heading=float(observation.heading),
            kinematics=kinematics,
            goal_position_xy=None
            if self._current_goal_xy is None
            else self._current_goal_xy.copy(),
            current_place_code=current_place_code,
            goal_place_code=None
            if self._current_goal_place_code is None
            else self._current_goal_place_code.copy(),
        )
        if self.observation_mode == "raw_pixels":
            if self.feature_extractor is None:
                return raw_image
            return {
                "image": raw_image,
                "features": self.feature_extractor.extract(context, self._previous_action),
            }
        return self.feature_extractor.extract(context, self._previous_action)  # type: ignore[union-attr]

    def _goal_reach_radius(self) -> float | None:
        goal_xy = self._current_goal_xy
        if goal_xy is None:
            return None
        adapter_radius_getter = getattr(self._adapter, "goal_reach_radius", None)
        if callable(adapter_radius_getter):
            return float(adapter_radius_getter())
        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return None
        goal_tolerance = getattr(unwrapped, "goal_tolerance", None)
        agent = getattr(unwrapped, "agent", None)
        agent_radius = 0.0 if agent is None else float(getattr(agent, "radius", 0.0))
        max_forward_step = float(getattr(unwrapped, "max_forward_step", 0.0))
        goal_object_radius = float(getattr(unwrapped, "goal_object_radius", 1.5))
        effective_goal_tolerance = (
            goal_object_radius if goal_tolerance is None else float(goal_tolerance)
        )
        return float(effective_goal_tolerance + agent_radius + 1.1 * max_forward_step)

    def _goal_reward_value(self) -> float:
        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return 1.0
        step_count = float(getattr(unwrapped, "step_count", 0.0))
        max_episode_steps = max(
            1.0,
            float(getattr(unwrapped, "max_episode_steps", self.environment_config.episode_length)),
        )
        return float(1.0 - 0.2 * (step_count / max_episode_steps))

    def _build_info(self, observation, base_info: dict[str, Any]) -> dict[str, Any]:
        info = dict(base_info)
        if self._current_goal_xy is None:
            for stale_goal_key in (
                "goal_position_xy",
                "goal_distance",
                "goal_reach_radius",
                "goal_index",
            ):
                info.pop(stale_goal_key, None)
        info["position_xy"] = [
            float(value) for value in np.asarray(observation.position_xy, dtype=np.float32)
        ]
        info["heading"] = float(observation.heading)
        if self._current_goal_xy is not None:
            goal_xy = np.asarray(self._current_goal_xy, dtype=np.float32)
            info["goal_position_xy"] = [float(value) for value in goal_xy]
            info["goal_distance"] = float(
                np.linalg.norm(np.asarray(observation.position_xy) - goal_xy)
            )
            if self._current_goal_index is not None:
                info["goal_index"] = int(self._current_goal_index)
            goal_reach_radius = self._goal_reach_radius()
            if goal_reach_radius is not None:
                info["goal_reach_radius"] = float(goal_reach_radius)
            info["goal_reward_value"] = float(self._goal_reward_value())
        if self._current_place_code is not None:
            info["current_place_code"] = self._current_place_code.copy()
        if self._current_goal_place_code is not None:
            info["goal_place_code"] = self._current_goal_place_code.copy()
            if self.goal_place_code_runtime is not None:
                info["goal_code_distance_metric"] = str(
                    self.goal_place_code_runtime.distance_metric
                )
                info["goal_code_normalize"] = bool(self.goal_place_code_runtime.normalize_codes)
                info["goal_code_success_threshold"] = float(
                    self.goal_place_code_runtime.success_threshold
                )
            elif self.synthetic_place_code_encoder is not None:
                encoder = self.synthetic_place_code_encoder
                info["goal_code_distance_metric"] = str(encoder.distance_metric)
                info["goal_code_normalize"] = bool(encoder.normalize_codes)
                info["goal_code_success_threshold"] = float(encoder.success_threshold)
        if self._current_grid_code is not None:
            info["current_grid_code"] = self._current_grid_code.copy()
        if self._current_goal_grid_code is not None:
            info["goal_grid_code"] = self._current_goal_grid_code.copy()
            if self.grid_code_encoder is not None:
                info["goal_code_distance_metric"] = str(self.grid_code_encoder.distance_metric)
                info["goal_code_normalize"] = bool(self.grid_code_encoder.normalize_codes)
                info["goal_code_success_threshold"] = float(
                    self.grid_code_encoder.success_threshold
                )
        info["episode_index"] = int(self._episode_index)
        return info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._seed_internal_rngs(int(seed))
        codebook_goal_code: np.ndarray | None = None
        if self._uses_codebook_goal_schedule():
            codebook_goal_code = self.goal_codebook.sample(self._goal_codebook_random)
            self._current_goal_index = None
            self._current_goal_xy = None
        else:
            self._current_goal_index, self._current_goal_xy = self._resolve_goal_for_reset()
        goal_xy = self._current_goal_xy
        if goal_xy is not None and hasattr(self._adapter, "set_goal_position_xy"):
            self._adapter.set_goal_position_xy(goal_xy)
        self._apply_runtime_curriculum_state()
        constrained_spawn = self._curriculum_max_spawn_distance is not None and goal_xy is not None
        reset_within_goal_distance = getattr(
            self._adapter,
            "reset_within_goal_distance",
            None,
        )
        if constrained_spawn and callable(reset_within_goal_distance):
            observation = reset_within_goal_distance(
                goal_xy,
                float(self._curriculum_max_spawn_distance),
                seed=None if seed is None else int(seed),
            )
            if not self._spawn_distance_satisfied(observation):  # pragma: no cover
                raise RuntimeError("Adapter returned a start outside max_spawn_distance.")
        else:
            observation = None
            max_attempts = 256 if constrained_spawn else 1
            for attempt_index in range(max_attempts):
                reset_seed = None if seed is None or attempt_index > 0 else int(seed)
                observation = self._adapter.reset(seed=reset_seed)
                if self._spawn_distance_satisfied(observation):
                    break
            else:
                raise RuntimeError(
                    "Could not sample a start satisfying curriculum max_spawn_distance "
                    f"after {max_attempts} attempts."
                )
        if observation is None:  # pragma: no cover
            raise RuntimeError(
                "DownstreamNavigationEnv.reset() did not receive an observation from the adapter."
            )
        self._reset_feature_state()
        self._current_goal_place_code = (
            np.asarray(codebook_goal_code, dtype=np.float32)
            if codebook_goal_code is not None
            else self._resolve_goal_place_code()
        )
        self._current_goal_grid_code = self._resolve_goal_grid_code()
        rendered = self._render_observation(observation)
        info = self._build_info(observation, observation.info)
        self._last_adapter_observation = observation
        self._episode_index += 1
        return rendered, info

    def step(self, action: int):
        step_result = self._adapter.step(int(action))
        self._previous_action = int(action)
        rendered = self._render_observation(step_result.observation)
        info = self._build_info(step_result.observation, step_result.info)
        self._last_adapter_observation = step_result.observation
        if (step_result.terminated or step_result.truncated) and "is_success" not in info:
            info["is_success"] = float(step_result.reward > 0.0)
        return (
            rendered,
            float(step_result.reward),
            step_result.terminated,
            step_result.truncated,
            info,
        )

    def advance_goal(self) -> dict[str, Any]:
        """Switch the commanded goal while preserving the current agent pose."""
        if self._curriculum_phase is not None or self._uses_codebook_goal_schedule():
            raise RuntimeError("Goal chaining does not support curriculum or codebook goals.")
        goal_xy = self.goal_scheduler.advance_goal()
        if goal_xy is None:
            raise RuntimeError("Goal chaining requires configured candidate goal positions.")
        self._current_goal_index = self.goal_scheduler.current_index
        self._current_goal_xy = goal_xy
        goal_setter = getattr(self._adapter, "set_goal_position_xy", None)
        if not callable(goal_setter):
            raise RuntimeError("Goal chaining requires runtime goal override support.")
        goal_setter(goal_xy)
        self._current_goal_place_code = self._resolve_goal_place_code()
        self._current_goal_grid_code = self._resolve_goal_grid_code()
        if self._last_adapter_observation is None:
            raise RuntimeError("Goal chaining requires an environment reset before advance_goal().")
        return self._build_info(
            self._last_adapter_observation,
            self._last_adapter_observation.info,
        )

    def enable_goal_code_reward_mode(self) -> None:
        terminate_on_goal_setter = getattr(self._adapter, "set_terminate_on_goal", None)
        if callable(terminate_on_goal_setter):
            terminate_on_goal_setter(False)
        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return
        if hasattr(unwrapped, "reward_on_goal"):
            unwrapped.reward_on_goal = False
        if hasattr(unwrapped, "terminate_on_goal"):
            unwrapped.terminate_on_goal = False
        if hasattr(unwrapped, "render_goal_object"):
            unwrapped.render_goal_object = False

    @property
    def goal_place_code_dim(self) -> int:
        if self.synthetic_place_code_encoder is not None:
            return int(self.synthetic_place_code_encoder.feature_dim)
        if self.goal_place_code_runtime is None:
            raise RuntimeError(
                "Goal place-code dimension is unavailable because goal_place_code_runtime is not "
                "set."
            )
        return int(self.goal_place_code_runtime.feature_dim)

    @property
    def goal_grid_code_dim(self) -> int:
        if self.grid_code_encoder is None:
            raise RuntimeError(
                "Goal grid-code dimension is unavailable because grid_code_encoder is not set."
            )
        return int(self.grid_code_encoder.feature_dim)

    def close(self) -> None:
        if hasattr(self._adapter, "close"):
            self._adapter.close()

    @property
    def current_rgb_frame(self) -> np.ndarray:
        if self._last_rgb_frame_hwc is None:
            raise RuntimeError("No RGB frame is available before the first reset().")
        return self._last_rgb_frame_hwc.copy()
