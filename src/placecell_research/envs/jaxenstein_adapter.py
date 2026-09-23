"""Imperative adapter wrapping the functional JAXenstein first-person env."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from placecell_research.utils.angles import wrap_radians

from .base import DiscreteActionSpace, EnvironmentCapabilities, Observation, StepResult

JAXENSTEIN_NAV_ACTION_NAMES = ("turn_left", "turn_right", "move_forward")


def require_reached_goal(info: Mapping[str, Any]) -> Any:
    """Return JAXenstein's explicit success flag or fail on an incompatible upstream API."""
    try:
        return info["reached_goal"]
    except KeyError as exc:
        raise RuntimeError(
            "JAXenstein step info is missing the required 'reached_goal' flag; refusing to "
            "classify generic episode completion as goal success."
        ) from exc


@dataclass
class JaxensteinAdapter:
    """Single-env imperative wrapper over the functional JAXenstein env."""

    env_id: str
    seed: int = 0
    episode_length: int = 256
    randomize_agent_start: bool | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    _jax: Any = field(init=False, repr=False)
    _env: Any = field(init=False, repr=False)
    _state: Any = field(init=False, repr=False)
    _key: Any = field(init=False, repr=False)
    _jit_reset: Any = field(init=False, repr=False)
    _jit_step: Any = field(init=False, repr=False)
    _jit_render: Any = field(init=False, repr=False)
    _goal_env_cache: dict = field(init=False, repr=False)
    _terminate_on_goal: bool = field(init=False, default=True)
    _uniform_spawn_enabled: bool = field(init=False, default=False)
    _modality_shapes: dict[str, tuple[int, ...]] = field(init=False, repr=False)
    _action_space: DiscreteActionSpace = field(init=False, repr=False)
    _capabilities: EnvironmentCapabilities = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import jax
            import jaxenstein  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "jax and jaxenstein are required for JaxensteinAdapter; install the '[jax]' "
                "extra on a Python >=3.11 interpreter."
            ) from exc

        self._jax = jax
        self._terminate_on_goal = bool(self.env_kwargs.get("terminate_on_goal", True))
        self._uniform_spawn_enabled = bool(self.env_kwargs.get("uniform_spawn", False))
        from .jaxenstein_maps import build_jaxenstein_env, supports_goal_override

        self._goal_env_cache = {}
        self._env = build_jaxenstein_env(
            self.env_id,
            episode_horizon=self.episode_length,
        )
        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)
        self._jit_render = jax.jit(self._env.render)
        self._key = jax.random.PRNGKey(int(self.seed))

        observation, state = self._jit_reset(self._key)
        self._state = state
        rgb = self._to_chw(np.asarray(observation))
        self._modality_shapes = {"rgb": rgb.shape}
        self._action_space = DiscreteActionSpace(
            count=len(JAXENSTEIN_NAV_ACTION_NAMES), names=JAXENSTEIN_NAV_ACTION_NAMES
        )
        self._capabilities = EnvironmentCapabilities(
            position_xy=True,
            heading=True,
            discrete_actions=True,
            topdown_render=True,
            goal_override=supports_goal_override(self.env_id),
            observation_modalities=("rgb",),
        )

    @staticmethod
    def _to_chw(observation: np.ndarray) -> np.ndarray:
        if observation.ndim != 3:
            raise ValueError(f"Expected (H, W, 3) RGB from jaxenstein, got {observation.shape}.")
        if observation.shape[0] in {1, 3}:
            return observation.astype(np.uint8, copy=False)
        return np.transpose(observation, (2, 0, 1)).astype(np.uint8, copy=False)

    def _observation(self, observation: Any, state: Any, info: Any = None) -> Observation:
        position_xy = np.asarray(state.pos, dtype=np.float32).reshape(-1)[:2]
        heading = float(wrap_radians(float(np.asarray(state.theta))))
        info_dict = dict(info) if isinstance(info, dict) else {}
        return Observation(
            position_xy=position_xy,
            heading=heading,
            info=info_dict,
            modalities={"rgb": self._to_chw(np.asarray(observation))},
        )

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return self._capabilities

    @property
    def num_actions(self) -> int:
        return self._action_space.count

    @property
    def modality_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._modality_shapes)

    def _uniform_free_spawn(
        self,
        state: Any,
        key: Any,
        *,
        goal_position_xy: np.ndarray | list[float] | None = None,
        max_goal_distance: float | None = None,
    ) -> tuple[Any, Any]:
        """Override JAXenstein's S-tile spawn with a uniform draw over every free tile + heading."""
        jnp = self._jax.numpy
        wall_grid = np.asarray(self._env.maze.wall_grid)
        free_rows, free_cols = np.nonzero(wall_grid == 0)
        if (goal_position_xy is None) != (max_goal_distance is None):
            raise ValueError(
                "goal_position_xy and max_goal_distance must either both be set or both be omitted."
            )
        if goal_position_xy is not None and max_goal_distance is not None:
            safe_center_radius = float(max_goal_distance) - float(np.sqrt(2.0) * 0.25)
            goal_xy = np.asarray(goal_position_xy, dtype=np.float32).reshape(2)
            free_centers = np.column_stack((free_cols + 0.5, free_rows + 0.5))
            eligible = np.linalg.norm(free_centers - goal_xy[None, :], axis=1) <= safe_center_radius
            free_rows = free_rows[eligible]
            free_cols = free_cols[eligible]
            if free_rows.size == 0:
                raise RuntimeError(
                    "No collision-free JAXenstein spawn tile satisfies "
                    f"max_goal_distance={float(max_goal_distance):.3f}."
                )
        tile_key, jitter_key, theta_key = self._jax.random.split(key, 3)
        pick = self._jax.random.randint(tile_key, (), 0, int(free_rows.shape[0]))
        jitter = self._jax.random.uniform(jitter_key, (2,), minval=-0.25, maxval=0.25)
        centre = jnp.stack(
            [jnp.asarray(free_cols)[pick], jnp.asarray(free_rows)[pick]]
        ).astype(jnp.float32) + 0.5
        new_pos = (centre + jitter).astype(state.pos.dtype)
        new_theta = self._jax.random.uniform(
            theta_key, (), minval=-jnp.pi, maxval=jnp.pi
        ).astype(state.theta.dtype)
        state = state.replace(pos=new_pos, theta=new_theta)
        observation = self._jit_render(state)
        return observation, state

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._key = self._jax.random.PRNGKey(int(seed))
        if self._uniform_spawn_enabled:
            self._key, subkey = self._jax.random.split(self._key)
            observation, state = self._jit_reset(subkey)
            self._key, spawn_key = self._jax.random.split(self._key)
            observation, state = self._uniform_free_spawn(state, spawn_key)
        elif self.randomize_agent_start is False:
            subkey = self._jax.random.fold_in(self._key, 0)
            observation, state = self._jit_reset(subkey)
        else:
            self._key, subkey = self._jax.random.split(self._key)
            observation, state = self._jit_reset(subkey)
        self._state = state
        return self._observation(observation, state)

    def reset_within_goal_distance(
        self,
        goal_position_xy: np.ndarray | list[float],
        max_goal_distance: float,
        seed: int | None = None,
    ) -> Observation:
        """Reset directly onto a free tile whose full jitter cell satisfies the distance bound."""
        if float(max_goal_distance) <= 0.0:
            raise ValueError("max_goal_distance must be positive.")
        if seed is not None:
            self._key = self._jax.random.PRNGKey(int(seed))
        self._key, reset_key = self._jax.random.split(self._key)
        _observation, state = self._jit_reset(reset_key)
        self._key, spawn_key = self._jax.random.split(self._key)
        observation, state = self._uniform_free_spawn(
            state,
            spawn_key,
            goal_position_xy=goal_position_xy,
            max_goal_distance=float(max_goal_distance),
        )
        self._state = state
        return self._observation(observation, state)

    def step(self, action: int) -> StepResult:
        observation, state, reward, _done, info = self._jit_step(self._state, int(action))
        result_observation = self._observation(observation, state, info)
        reached_goal = bool(np.asarray(require_reached_goal(info)))
        timed_out = bool(np.asarray(state.t >= self._env.episode_horizon))
        if reached_goal and not self._terminate_on_goal and not timed_out:
            state = state.replace(done=self._jax.numpy.asarray(False))
        self._state = state
        if self._terminate_on_goal:
            terminated = reached_goal
            truncated = timed_out and not reached_goal
        else:
            terminated = False
            truncated = timed_out
        return StepResult(
            observation=result_observation,
            reward=float(np.asarray(reward)),
            terminated=terminated,
            truncated=truncated,
            info=result_observation.info,
        )

    def render_topdown(self) -> np.ndarray | None:
        maze = getattr(self._env, "maze", None)
        wall_grid = getattr(maze, "wall_grid", None)
        if wall_grid is None:
            return None
        grid = np.asarray(wall_grid)
        cell = 12
        rows, cols = grid.shape
        frame = np.full((rows * cell, cols * cell, 3), 235, dtype=np.uint8)
        frame[np.repeat(np.repeat(grid > 0, cell, 0), cell, 1)] = (70, 70, 80)
        position = np.asarray(self._state.pos, dtype=np.float32).reshape(-1)
        center_x, center_y = int(position[0] * cell), int(position[1] * cell)
        radius = 3
        row0, row1 = max(0, center_y - radius), min(frame.shape[0], center_y + radius + 1)
        col0, col1 = max(0, center_x - radius), min(frame.shape[1], center_x + radius + 1)
        frame[row0:row1, col0:col1] = (220, 40, 40)
        return frame

    def _maze_wall_grid(self) -> np.ndarray:
        maze = getattr(self._env, "maze", None)
        grid = getattr(maze, "wall_grid", None)
        if grid is None:
            raise RuntimeError("JAXenstein maze.wall_grid is unavailable for spatial queries.")
        return np.asarray(grid)

    def room_bounds_xy(self) -> list[tuple[float, float, float, float]]:
        """One contiguous tile arena; bounds in tile units (x=col, y=row)."""
        rows, cols = self._maze_wall_grid().shape
        return [(0.0, float(cols), 0.0, float(rows))]

    def is_free_position_xy(self, xy: np.ndarray | list[float] | tuple[float, float]) -> bool:
        """Collision-free iff the tile at (x=col, y=row) is not a wall."""
        grid = self._maze_wall_grid()
        coords = np.asarray(xy, dtype=np.float32).reshape(-1)
        col, row = int(coords[0]), int(coords[1])
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            return bool(grid[row, col] == 0)
        return False

    def get_goal_position_xy(self) -> np.ndarray:
        return np.asarray(self._env.maze.goal_xy, dtype=np.float32).reshape(-1)[:2]

    def goal_reach_radius(self) -> float:
        return float(np.asarray(self._env.params.goal_radius))

    def set_goal_position_xy(self, goal_position_xy: np.ndarray | list[float]) -> None:
        from .jaxenstein_maps import build_jaxenstein_env

        tile = self._snap_to_free_tile(goal_position_xy)
        if tile not in self._goal_env_cache:
            env = build_jaxenstein_env(
                self.env_id,
                goal_tile=tile,
                episode_horizon=self.episode_length,
            )
            self._goal_env_cache[tile] = (
                env,
                self._jax.jit(env.reset),
                self._jax.jit(env.step),
                self._jax.jit(env.render),
            )
        self._env, self._jit_reset, self._jit_step, self._jit_render = self._goal_env_cache[tile]
        _obs, fresh = self._jit_reset(self._jax.random.fold_in(self._key, 0))
        if self._state is not None:
            fresh = fresh.replace(pos=self._state.pos, theta=self._state.theta)
        self._state = fresh

    def _snap_to_free_tile(self, xy: np.ndarray | list[float]) -> tuple[int, int]:
        coords = np.asarray(xy, dtype=np.float32).reshape(-1)
        col0, row0 = int(coords[0]), int(coords[1])
        grid = self._maze_wall_grid()
        for radius in range(6):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = row0 + dr, col0 + dc
                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and not grid[r, c]:
                        return (c, r)
        raise RuntimeError(f"no free tile near goal {tuple(coords)!r}")

    def close(self) -> None:
        self._state = None
