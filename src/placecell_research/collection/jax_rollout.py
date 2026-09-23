"""Vectorized on-device JAXenstein rollout producing per-episode dicts in the pipeline schema."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any

import numpy as np

from placecell_research.envs.jaxenstein_adapter import require_reached_goal


def _uniform_spawn(env: Any, state: Any, keys: Any) -> tuple[Any, Any]:
    """Replace JAXenstein's S-tile spawn with a uniform draw over every free tile + heading."""
    import jax
    import jax.numpy as jnp

    wall_grid = np.asarray(env.maze.wall_grid)
    free_rows, free_cols = np.nonzero(wall_grid == 0)
    split_keys = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
    tile_keys = split_keys[:, 0]
    jitter_keys = split_keys[:, 1]
    theta_keys = split_keys[:, 2]
    pick = jax.vmap(lambda key: jax.random.randint(key, (), 0, int(free_rows.shape[0])))(tile_keys)
    jitter = jax.vmap(lambda key: jax.random.uniform(key, (2,), minval=-0.25, maxval=0.25))(
        jitter_keys
    )
    centres = jnp.stack([jnp.asarray(free_cols)[pick], jnp.asarray(free_rows)[pick]], axis=-1) + 0.5
    new_pos = (centres + jitter).astype(state.pos.dtype)
    new_theta = jax.vmap(lambda key: jax.random.uniform(key, (), minval=-jnp.pi, maxval=jnp.pi))(
        theta_keys
    ).astype(state.theta.dtype)
    state = state.replace(pos=new_pos, theta=new_theta)
    observation = jax.vmap(env.render)(state)
    return observation, state


def _reset_episode_batch(
    env: Any,
    *,
    base_seed: int,
    num_envs: int,
    randomize_agent_start: bool | None,
    uniform_spawn: bool,
) -> tuple[Any, Any]:
    """Reset one state per source seed using the serial adapter's key schedule."""
    import jax
    import jax.numpy as jnp

    episode_seeds = jnp.arange(int(num_envs), dtype=jnp.uint32) + jnp.uint32(base_seed)
    episode_keys = jax.vmap(jax.random.PRNGKey)(episode_seeds)
    if randomize_agent_start is False and not uniform_spawn:
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(episode_keys)
        return jax.vmap(env.reset)(reset_keys)

    reset_splits = jax.vmap(lambda key: jax.random.split(key, 2))(episode_keys)
    continuing_keys = reset_splits[:, 0]
    reset_keys = reset_splits[:, 1]
    observation, state = jax.vmap(env.reset)(reset_keys)
    if not uniform_spawn:
        return observation, state

    spawn_splits = jax.vmap(lambda key: jax.random.split(key, 2))(continuing_keys)
    spawn_keys = spawn_splits[:, 1]
    return _uniform_spawn(env, state, spawn_keys)


def _make_trajectory_scan(
    *,
    env_id: str,
    horizon: int,
    policy: Any,
    terminate_on_goal: bool,
    randomize_agent_start: bool | None,
    uniform_spawn: bool,
) -> Callable:
    """Keep the scan body stable so changing episode seeds reuses its compiled executable."""
    import jax
    import jax.numpy as jnp

    from placecell_research.envs.jaxenstein_maps import build_jaxenstein_env

    env = build_jaxenstein_env(env_id, episode_horizon=int(horizon))

    reset_batch = jax.jit(
        partial(
            _reset_episode_batch,
            env,
            randomize_agent_start=randomize_agent_start,
            uniform_spawn=uniform_spawn,
        ),
        static_argnames=("num_envs",),
    )

    def step_fn(carry, _):
        observation, state, policy_state, done_flags = carry
        action, policy_state = policy.sample(policy_state, observation)
        record = {
            "rgb": observation,
            "pos": state.pos,
            "theta": state.theta,
            "action": action,
            "valid": ~done_flags,
        }
        observation, state, _reward, done, info = jax.vmap(env.step)(state, action)
        reached_goal = require_reached_goal(info)
        timed_out = state.t >= int(horizon)
        step_terminated = (~done_flags) & reached_goal & bool(terminate_on_goal)
        step_truncated = (~done_flags) & timed_out & ~step_terminated
        if not terminate_on_goal:
            continue_after_goal = reached_goal & ~timed_out
            state = state.replace(done=jnp.where(continue_after_goal, False, state.done))
        record["terminated"] = step_terminated
        record["truncated"] = step_truncated
        episode_done = step_terminated | step_truncated
        return (observation, state, policy_state, done_flags | episode_done), record

    def scan_batch(*, num_envs: int, base_seed: int):
        observation, state = reset_batch(base_seed=base_seed, num_envs=num_envs)
        policy_key = jax.random.split(jax.random.PRNGKey(int(base_seed)), 3)[2]
        policy_state = policy.init(policy_key, int(num_envs))
        done_flags = jnp.zeros((int(num_envs),), dtype=bool)

        _carry, trajectory = jax.lax.scan(
            step_fn,
            (observation, state, policy_state, done_flags),
            xs=None,
            length=int(horizon),
        )
        return trajectory

    return scan_batch


def rollout_jaxenstein(
    *,
    env_id: str,
    num_envs: int,
    horizon: int,
    base_seed: int,
    policy: Any,
    terminate_on_goal: bool = True,
    randomize_agent_start: bool | None = None,
    uniform_spawn: bool = True,
    keep_rgb_on_device: bool = False,
) -> list[dict[str, Any]]:
    """Roll out num_envs JAXenstein envs for horizon steps on-device (vmap + scan)."""
    rollout = make_jaxenstein_rollout(
        env_id=env_id,
        policy=policy,
        terminate_on_goal=terminate_on_goal,
        randomize_agent_start=randomize_agent_start,
        uniform_spawn=uniform_spawn,
    )
    return rollout(
        num_envs=num_envs,
        horizon=horizon,
        base_seed=base_seed,
        keep_rgb_on_device=keep_rgb_on_device,
    )


def make_jaxenstein_rollout(
    *,
    env_id: str,
    policy: Any,
    terminate_on_goal: bool = True,
    randomize_agent_start: bool | None = None,
    uniform_spawn: bool = True,
) -> Callable:
    """Create a reusable collector for one environment and policy."""

    @lru_cache(maxsize=2)
    def scan_for_horizon(horizon: int):
        return _make_trajectory_scan(
            env_id=env_id,
            horizon=horizon,
            policy=policy,
            terminate_on_goal=terminate_on_goal,
            randomize_agent_start=randomize_agent_start,
            uniform_spawn=uniform_spawn,
        )

    def rollout(*, num_envs: int, horizon: int, base_seed: int, keep_rgb_on_device: bool = False):
        trajectory = scan_for_horizon(int(horizon))(num_envs=num_envs, base_seed=base_seed)
        return _package_episodes(
            trajectory,
            num_envs=num_envs,
            base_seed=base_seed,
            keep_rgb_on_device=keep_rgb_on_device,
        )

    return rollout


def _package_episodes(trajectory, *, num_envs, base_seed, keep_rgb_on_device):
    from .collector import _compute_kinematics

    if keep_rgb_on_device:
        import jax.numpy as jnp

        rgb = jnp.transpose(trajectory["rgb"], (1, 0, 4, 2, 3)).astype(jnp.uint8)
    else:
        rgb = np.asarray(trajectory["rgb"]).transpose(1, 0, 2, 3, 4)
        rgb = np.ascontiguousarray(rgb.transpose(0, 1, 4, 2, 3)).astype(np.uint8)
    position_xy = np.asarray(trajectory["pos"]).transpose(1, 0, 2).astype(np.float32)
    heading = np.asarray(trajectory["theta"]).transpose(1, 0).astype(np.float32)
    heading = (((heading + np.pi) % (2.0 * np.pi)) - np.pi).astype(np.float32)
    actions = np.asarray(trajectory["action"]).transpose(1, 0).astype(np.int64)
    valid = np.asarray(trajectory["valid"]).transpose(1, 0).astype(bool)
    terminated_steps = np.asarray(trajectory["terminated"]).transpose(1, 0).astype(bool)
    truncated_steps = np.asarray(trajectory["truncated"]).transpose(1, 0).astype(bool)

    episodes: list[dict[str, Any]] = []
    for env_index in range(int(num_envs)):
        length = int(valid[env_index].sum())
        terminated = bool(terminated_steps[env_index].any())
        truncated = bool(truncated_steps[env_index].any())
        episodes.append(
            {
                "observations/rgb": rgb[env_index],
                "position_xy": position_xy[env_index],
                "heading": heading[env_index],
                "kinematics": _compute_kinematics(
                    position_xy[env_index], heading[env_index]
                ).astype(np.float32),
                "actions": actions[env_index],
                "valid_steps": valid[env_index],
                "length": length,
                "terminated": terminated,
                "truncated": truncated,
                "source_seed": int(base_seed) + env_index,
            }
        )
    return episodes
