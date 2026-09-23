"""Environment factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import EnvironmentAdapter
from .miniworld_adapter import MiniWorldAdapter

EnvironmentFactory = Callable[[Any, int], EnvironmentAdapter]

_ENVIRONMENT_BUILDERS: dict[str, EnvironmentFactory] = {}


def register_environment_builder(kind: str, factory: EnvironmentFactory) -> None:
    """Register one environment backend under a stable kind string."""

    normalized_kind = str(kind).strip().lower()
    if not normalized_kind:
        raise ValueError("Environment builder kind must be a non-empty string.")
    _ENVIRONMENT_BUILDERS[normalized_kind] = factory


def available_environment_kinds() -> tuple[str, ...]:
    """Return registered environment backend kinds."""

    return tuple(sorted(_ENVIRONMENT_BUILDERS))


def _build_miniworld_environment(config: Any, seed: int) -> EnvironmentAdapter:
    return MiniWorldAdapter(
        env_id=config.env_id,
        seed=seed,
        episode_length=config.episode_length,
        randomize_agent_start=config.randomize_agent_start,
        env_kwargs=config.env_kwargs,
    )


def _build_jaxenstein_environment(config: Any, seed: int) -> EnvironmentAdapter:
    from .jaxenstein_adapter import JaxensteinAdapter

    return JaxensteinAdapter(
        env_id=config.env_id,
        seed=seed,
        episode_length=config.episode_length,
        randomize_agent_start=config.randomize_agent_start,
        env_kwargs=config.env_kwargs,
    )


def build_environment(config: Any, seed: int) -> EnvironmentAdapter:
    """Build the configured environment adapter through the backend registry."""

    kind = str(config.kind).strip().lower()
    factory = _ENVIRONMENT_BUILDERS.get(kind)
    if factory is None:
        raise ValueError(
            f"Unsupported environment kind: {config.kind!r}. "
            f"Registered kinds: {list(available_environment_kinds())!r}."
        )
    return factory(config, seed)


register_environment_builder("miniworld", _build_miniworld_environment)
register_environment_builder("jaxenstein", _build_jaxenstein_environment)
