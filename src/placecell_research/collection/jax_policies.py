"""Vectorized Ornstein-Uhlenbeck-smoothed discrete navigation policy (JAX)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JaxOUPolicy:
    """Vectorized OU-smoothed 3-action policy: turn_left=0, turn_right=1, move_forward=2."""

    decay: float = 0.85
    noise_scale: float = 0.4
    turn_threshold: float = 0.5

    def init(self, key: Any, num_envs: int) -> Any:
        import jax

        key, subkey = jax.random.split(key)
        tendency = jax.random.normal(subkey, (num_envs,)) * 0.3
        return (tendency, key)

    def sample(self, policy_state: Any, observation: Any = None) -> tuple[Any, Any]:
        import jax
        import jax.numpy as jnp

        tendency, key = policy_state
        key, subkey = jax.random.split(key)
        tendency = self.decay * tendency + self.noise_scale * jax.random.normal(
            subkey, tendency.shape
        )
        action = jnp.where(
            tendency < -self.turn_threshold,
            0,
            jnp.where(tendency > self.turn_threshold, 1, 2),
        ).astype(jnp.int32)
        return action, (tendency, key)
