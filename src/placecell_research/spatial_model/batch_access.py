"""Canonical accessors for batch tensors and dataset-style aliases."""

from __future__ import annotations

import torch
from torch import Tensor

from placecell_research.datasets.schema import (
    ACTIONS_KEY,
    KINEMATICS_KEY,
    LATENT_KEY,
    POSITION_KEY,
    RGB_KEY,
    VALID_MASK_KEY,
)

OBSERVATION_KEYS = ("latent", LATENT_KEY, "rgb", RGB_KEY)
ACTIONS_KEYS = ("actions", ACTIONS_KEY)
VALID_STEPS_KEYS = ("valid_steps", VALID_MASK_KEY)
KINEMATICS_KEYS = ("kinematics", KINEMATICS_KEY)
POSITION_XY_KEYS = ("position_xy", POSITION_KEY)


def resolve_optional_tensor(batch: dict[str, Tensor], candidates: tuple[str, ...]) -> Tensor | None:
    for candidate in candidates:
        if candidate in batch:
            return batch[candidate]
    return None


def require_tensor(
    batch: dict[str, Tensor], candidates: tuple[str, ...], error_message: str
) -> Tensor:
    value = resolve_optional_tensor(batch, candidates)
    if value is None:
        raise KeyError(error_message)
    return value


def resolve_observations(batch: dict[str, Tensor]) -> Tensor:
    return require_tensor(batch, OBSERVATION_KEYS, "Batch is missing latent/rgb observations.")


def resolve_actions(batch: dict[str, Tensor]) -> Tensor:
    return require_tensor(batch, ACTIONS_KEYS, "Batch is missing actions.")


def resolve_valid_steps(batch: dict[str, Tensor]) -> Tensor:
    valid_steps = resolve_optional_tensor(batch, VALID_STEPS_KEYS)
    if valid_steps is not None:
        return valid_steps.bool()
    return torch.ones_like(resolve_actions(batch), dtype=torch.bool)


def resolve_kinematics(batch: dict[str, Tensor]) -> Tensor | None:
    return resolve_optional_tensor(batch, KINEMATICS_KEYS)


def resolve_position_xy(batch: dict[str, Tensor]) -> Tensor | None:
    return resolve_optional_tensor(batch, POSITION_XY_KEYS)
