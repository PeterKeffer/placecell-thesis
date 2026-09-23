"""Gymnasium-backed MiniWorld adapter."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from placecell_research.utils.angles import wrap_radians

from .base import (
    ContinuousMotion,
    DiscreteActionSpace,
    EnvironmentCapabilities,
    Observation,
    StepResult,
)

CUSTOM_MINIWORLD_ENV_MODULES = {
    "MiniWorld-WallGapAsymLarge-v0": "placecell_research.envs.miniworld_wallgap_asym_large",
}


def _raise_display_context_error(exc: Exception) -> None:
    message = (
        "MiniWorld could not create a rendering context. "
        "On macOS, run collection from a logged-in desktop session with an available display. "
        "On headless Linux or SLURM, use the EGL setup in scripts/slurm/env_miniworld.sh."
    )
    raise RuntimeError(message) from exc


def _mirror_builtin_miniworld_envs_into_gymnasium() -> None:
    try:
        from gymnasium.envs import registration as gymnasium_registration
        from gymnasium.error import Error as GymnasiumError
        from miniworld import envs as miniworld_envs

        try:
            from miniworld.envs.miniworld_env import MiniWorldEnv
        except ModuleNotFoundError:
            from miniworld.miniworld import MiniWorldEnv  # type: ignore
    except ModuleNotFoundError:
        return

    registry = getattr(gymnasium_registration, "registry", None)
    register = getattr(gymnasium_registration, "register", None)
    if registry is None or register is None:
        return

    for _, obj in vars(miniworld_envs).items():
        if not inspect.isclass(obj) or not issubclass(obj, MiniWorldEnv) or obj is MiniWorldEnv:
            continue
        env_id = f"MiniWorld-{obj.__name__}-v0"
        if env_id in registry:
            continue
        try:
            register(id=env_id, entry_point=f"{miniworld_envs.__name__}:{obj.__name__}")
        except GymnasiumError as exc:
            if "Cannot re-register" not in str(exc):
                raise


def _ensure_miniworld_env_registered(env_id: str) -> None:
    from gymnasium.envs import registration as gymnasium_registration

    registry = getattr(gymnasium_registration, "registry", {})
    if env_id in registry:
        return

    _mirror_builtin_miniworld_envs_into_gymnasium()
    registry = getattr(gymnasium_registration, "registry", {})
    if env_id in registry:
        return

    custom_module = CUSTOM_MINIWORLD_ENV_MODULES.get(env_id)
    if custom_module is not None:
        importlib.import_module(custom_module)
        registry = getattr(gymnasium_registration, "registry", {})
        if env_id in registry:
            return

    available_miniworld_ids = sorted(
        key for key in registry.keys() if str(key).startswith("MiniWorld-")
    )
    raise RuntimeError(
        f"MiniWorld env '{env_id}' is not registered with Gymnasium. "
        f"Available MiniWorld ids: {available_miniworld_ids[:12]}"
    )


def _extract_position_and_heading(environment: Any) -> tuple[np.ndarray, float]:
    agent = getattr(getattr(environment, "unwrapped", environment), "agent", None)
    position_xy = np.zeros(2, dtype=np.float32)
    heading = 0.0
    if agent is not None:
        raw_position = getattr(agent, "pos", None)
        if raw_position is not None:
            coords = np.asarray(raw_position, dtype=np.float32).reshape(-1)
            if coords.size >= 3:
                position_xy = np.asarray([coords[0], coords[2]], dtype=np.float32)
        heading = _wrap_heading_radians(float(getattr(agent, "dir", 0.0)))
    return position_xy, heading


def _wrap_heading_radians(heading: float) -> float:
    return float(wrap_radians(heading))


def _extract_goal_position(environment: Any) -> np.ndarray | None:
    unwrapped = getattr(environment, "unwrapped", environment)
    getter = getattr(unwrapped, "get_goal_position_xy", None)
    if callable(getter):
        goal_xy = np.asarray(getter(), dtype=np.float32).reshape(-1)
        if goal_xy.size >= 2:
            return np.asarray(goal_xy[:2], dtype=np.float32)
    box = getattr(unwrapped, "box", None)
    if box is None:
        return None
    raw_position = getattr(box, "pos", None)
    if raw_position is None:
        return None
    coords = np.asarray(raw_position, dtype=np.float32).reshape(-1)
    if coords.size >= 3:
        return np.asarray([coords[0], coords[2]], dtype=np.float32)
    return None


@dataclass
class MiniWorldAdapter:
    """Gymnasium-compatible MiniWorld adapter."""

    env_id: str
    seed: int = 0
    episode_length: int = 256
    randomize_agent_start: bool | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    _env: Any = field(init=False, repr=False)
    _num_actions: int = field(init=False)
    _modality_shapes: dict[str, tuple[int, ...]] = field(init=False, repr=False)
    _action_space: DiscreteActionSpace = field(init=False, repr=False)
    _capabilities: EnvironmentCapabilities = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._env = None
        try:
            import gymnasium as gym
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("gymnasium is required to build MiniWorld environments") from exc
        from .miniworld_runtime_compat import ensure_miniworld_runtime_compatibility

        try:
            ensure_miniworld_runtime_compatibility()
            try:
                import miniworld  # noqa: F401
            except IndexError as exc:  # pragma: no cover
                _raise_display_context_error(exc)
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("miniworld is required for MiniWorldAdapter") from exc
            _ensure_miniworld_env_registered(self.env_id)

            kwargs = {
                "render_mode": "rgb_array",
                "max_episode_steps": self.episode_length,
                **self.env_kwargs,
            }
            try:
                self._env = gym.make(self.env_id, **kwargs)
            except IndexError as exc:  # pragma: no cover
                _raise_display_context_error(exc)
            unwrapped = getattr(self._env, "unwrapped", self._env)
            if hasattr(self._env, "_max_episode_steps"):
                self._env._max_episode_steps = int(self.episode_length)
            if hasattr(unwrapped, "max_episode_steps"):
                unwrapped.max_episode_steps = int(self.episode_length)
            if self.randomize_agent_start is not None and hasattr(
                unwrapped, "randomize_agent_start"
            ):
                unwrapped.randomize_agent_start = bool(self.randomize_agent_start)
            try:
                observation, _ = self._env.reset(seed=self.seed)
            except IndexError as exc:  # pragma: no cover
                _raise_display_context_error(exc)
            observation_array = np.asarray(observation, dtype=np.uint8)
            if observation_array.ndim != 3:
                raise ValueError(
                    f"Expected MiniWorld RGB observation with 3 dimensions, got "
                    f"{observation_array.shape}."
                )
            if observation_array.shape[0] in {1, 3}:
                chw_shape = tuple(int(value) for value in observation_array.shape)
            else:
                chw_shape = (
                    int(observation_array.shape[2]),
                    int(observation_array.shape[0]),
                    int(observation_array.shape[1]),
                )
            self._modality_shapes = {"rgb": chw_shape}
            self._num_actions = int(self._env.action_space.n)
            self._action_space = DiscreteActionSpace(
                count=self._num_actions,
                names=tuple(self.action_names),
            )
            render_top_view = getattr(unwrapped, "render_top_view", None)
            set_goal_position = getattr(unwrapped, "set_goal_position_xy", None)
            self._capabilities = EnvironmentCapabilities(
                position_xy=True,
                heading=True,
                discrete_actions=True,
                topdown_render=callable(render_top_view),
                goal_override=callable(set_goal_position),
                observation_modalities=("rgb",),
            )
        except Exception:
            self._close_env()
            raise

    def _close_env(self) -> None:
        env = getattr(self, "_env", None)
        if env is None:
            return
        try:
            env.close()
        except Exception:
            pass
        self._env = None

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return self._capabilities

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def modality_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._modality_shapes)

    @property
    def action_names(self) -> list[str]:
        actions = getattr(self._env.unwrapped, "actions", None)
        if actions is None:
            return [str(action_index) for action_index in range(self._num_actions)]
        return [str(actions(action_index).name) for action_index in range(self._num_actions)]

    def _normalize_rgb(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape[0] in {1, 3}:
            return observation.astype(np.uint8, copy=False)
        return np.transpose(observation, (2, 0, 1)).astype(np.uint8, copy=False)

    def reset(self, seed: int | None = None) -> Observation:
        observation, info = self._env.reset(seed=seed)
        rgb = self._normalize_rgb(np.asarray(observation))
        position_xy, heading = _extract_position_and_heading(self._env)
        info_dict = dict(info)
        goal_xy = _extract_goal_position(self._env)
        if goal_xy is not None:
            info_dict["goal_position_xy"] = [float(value) for value in goal_xy]
        return Observation(
            position_xy=position_xy,
            heading=heading,
            info=info_dict,
            modalities={"rgb": rgb},
        )

    def step(self, action: int) -> StepResult:
        observation, reward, terminated, truncated, info = self._env.step(int(action))
        rgb = self._normalize_rgb(np.asarray(observation))
        position_xy, heading = _extract_position_and_heading(self._env)
        info_dict = dict(info)
        goal_xy = _extract_goal_position(self._env)
        if goal_xy is not None:
            info_dict["goal_position_xy"] = [float(value) for value in goal_xy]
        return StepResult(
            observation=Observation(
                position_xy=position_xy,
                heading=heading,
                info=info_dict,
                modalities={"rgb": rgb},
            ),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info_dict,
        )

    def step_motion(self, motion: ContinuousMotion) -> StepResult:
        """Move, then turn, with one render and one environment timestep."""
        if (
            not np.isfinite(motion.forward_distance)
            or motion.forward_distance < 0
            or not np.isfinite(motion.turn_radians)
        ):
            raise ValueError("Motion requires a finite nonnegative distance and finite turn.")
        env = self._env.unwrapped
        segments = max(1, int(np.ceil(motion.forward_distance / (env.agent.radius / 2))))
        distance = motion.forward_distance / segments
        blocked = False
        for _ in range(segments):
            if not env.move_agent(distance, 0.0):
                blocked = True
                break
        env.turn_agent(float(np.degrees(motion.turn_radians)))
        result = self.step(env.actions.done)
        result.observation.info["motion_blocked"] = blocked
        return result

    def set_goal_position_xy(
        self, goal_position_xy: np.ndarray | list[float] | tuple[float, float]
    ) -> None:
        unwrapped = getattr(self._env, "unwrapped", self._env)
        setter = getattr(unwrapped, "set_goal_position_xy", None)
        if not callable(setter):
            raise RuntimeError(
                f"Env '{self.env_id}' does not support setting goal positions at runtime."
            )
        setter(np.asarray(goal_position_xy, dtype=np.float32).reshape(2))

    def room_bounds_xy(self) -> list[tuple[float, float, float, float]]:
        """Per-room (min_x, max_x, min_z, max_z) bounds for waypoint sampling/planning."""
        unwrapped = getattr(self._env, "unwrapped", self._env)
        bounds_by_name = getattr(unwrapped, "full_room_bounds_by_name", None)
        if not bounds_by_name:
            raise RuntimeError(
                f"Env '{self.env_id}' lacks full_room_bounds_by_name for waypoint sampling."
            )
        return [
            (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
            for bounds in bounds_by_name.values()
        ]

    def is_free_position_xy(self, xy: np.ndarray | list[float] | tuple[float, float]) -> bool:
        """Collision-free against static walls, excluding the agent and the goal box."""
        unwrapped = getattr(self._env, "unwrapped", self._env)
        agent = getattr(unwrapped, "agent", None)
        intersect = getattr(unwrapped, "intersect", None)
        if agent is None or not callable(intersect):
            raise RuntimeError(
                f"Env '{self.env_id}' does not support collision queries (needs a reset world)."
            )
        coords = np.asarray(xy, dtype=np.float32).reshape(-1)
        position = np.array([float(coords[0]), 0.0, float(coords[1])], dtype=float)
        radius = float(getattr(agent, "radius", 0.0))
        box = getattr(unwrapped, "box", None)
        raw_entities = getattr(unwrapped, "entities", None)
        entity_list: list[Any] | None = raw_entities if isinstance(raw_entities, list) else None
        removed = False
        if box is not None and entity_list is not None and box in entity_list:
            entity_list.remove(box)
            removed = True
        try:
            collides = intersect(agent, position, radius)
        finally:
            if removed and entity_list is not None:
                entity_list.append(box)
        return not bool(collides)

    def goal_object_radius(self) -> float:
        unwrapped = getattr(self._env, "unwrapped", self._env)
        return float(getattr(unwrapped, "goal_object_radius", 0.0))

    def render_topdown(self) -> np.ndarray | None:
        render_top_view = getattr(self._env.unwrapped, "render_top_view", None)
        if render_top_view is None:
            return None
        frame = render_top_view()
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self._close_env()
