"""Environment interfaces and capability contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

CapabilityName = Literal[
    "position_xy",
    "heading",
    "discrete_actions",
    "topdown_render",
    "goal_override",
]


@dataclass
class Observation:
    """Normalized environment observation."""

    position_xy: np.ndarray
    heading: float
    info: dict[str, Any]
    modalities: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def modality_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.modalities))

    @property
    def rgb(self) -> np.ndarray | None:
        return self.modalities.get("rgb")

    @property
    def latent(self) -> np.ndarray | None:
        return self.modalities.get("latent")

    def get_modality(self, name: str) -> np.ndarray | None:
        return self.modalities.get(str(name))

    def require_modality(self, name: str) -> np.ndarray:
        value = self.get_modality(name)
        if value is None:
            raise KeyError(
                f"Observation is missing required modality {name!r}. "
                f"Available modalities: {list(self.modality_names)!r}."
            )
        return value


@dataclass
class StepResult:
    """Normalized step output."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class ContinuousMotion:
    """Outgoing forward distance in world units and heading change in radians."""

    forward_distance: float
    turn_radians: float


@dataclass(frozen=True, slots=True)
class DiscreteActionSpace:
    """Stable action-space description for the canonical codepath."""

    count: int
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.count) < 1:
            raise ValueError("DiscreteActionSpace.count must be >= 1.")
        if self.names and len(self.names) != int(self.count):
            raise ValueError(
                "DiscreteActionSpace.names must be empty or match the discrete action count. "
                f"Got {len(self.names)} names for {self.count} actions."
            )


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilities:
    """Explicit capability flags used to validate stage requirements."""

    position_xy: bool = True
    heading: bool = True
    discrete_actions: bool = True
    topdown_render: bool = False
    goal_override: bool = False
    observation_modalities: tuple[str, ...] = ()

    def missing(self, required: tuple[CapabilityName, ...]) -> list[str]:
        return [name for name in required if not bool(getattr(self, name))]

    def missing_modalities(self, required: tuple[str, ...]) -> list[str]:
        available = set(self.observation_modalities)
        return [name for name in required if name not in available]


class EnvironmentAdapter(Protocol):
    """Stable adapter interface used by collection."""

    def reset(self, seed: int | None = None) -> Observation:
        ...

    def step(self, action: int) -> StepResult:
        ...

    @property
    def action_space(self) -> DiscreteActionSpace:
        ...

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        ...

    @property
    def num_actions(self) -> int:
        ...

    @property
    def modality_shapes(self) -> dict[str, tuple[int, ...]]:
        ...

    def render_topdown(self) -> np.ndarray | None:
        ...

    def close(self) -> None:
        ...


def assert_environment_supports(
    adapter: EnvironmentAdapter,
    *,
    consumer: str,
    required: tuple[CapabilityName, ...] = (),
    required_modalities: tuple[str, ...] = (),
) -> None:
    """Fail fast when an environment backend does not satisfy a stage contract."""

    missing_capabilities = adapter.capabilities.missing(required)
    missing_modalities = adapter.capabilities.missing_modalities(required_modalities)
    if not missing_capabilities and not missing_modalities:
        return
    missing_parts: list[str] = []
    if missing_capabilities:
        missing_parts.append(f"capabilities [{', '.join(sorted(missing_capabilities))}]")
    if missing_modalities:
        missing_parts.append(f"modalities [{', '.join(sorted(missing_modalities))}]")
    raise RuntimeError(
        f"{consumer} requires an environment with {' and '.join(missing_parts)}, "
        f"but backend capabilities were {adapter.capabilities!r}."
    )
