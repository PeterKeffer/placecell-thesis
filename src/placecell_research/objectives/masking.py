"""Shared masking helpers for padded sequence objectives."""

from __future__ import annotations

import torch
from torch import Tensor

from placecell_research.spatial_model.batch_access import resolve_actions, resolve_valid_steps
from placecell_research.spatial_model.types import RepresentationBundle


def valid_steps(bundle: RepresentationBundle, batch: dict[str, Tensor]) -> Tensor:
    if "valid_steps" in bundle.masks:
        return bundle.masks["valid_steps"].bool()
    if any(key in batch for key in ("valid_steps", "masks/valid_steps")):
        return resolve_valid_steps(batch)
    return torch.ones_like(resolve_actions(batch), dtype=torch.bool)


def transition_mask(
    bundle: RepresentationBundle,
    batch: dict[str, Tensor],
    horizon: int = 1,
) -> Tensor:
    mask = valid_steps(bundle, batch)
    if horizon <= 0:
        return mask
    return mask[:, horizon:] & mask[:, :-horizon]


def sample_time_mask(mask: Tensor, *, stride: int, offset: int) -> Tensor:
    if offset >= stride:
        raise ValueError(f"objective anchor_offset={offset} must be below anchor_stride={stride}.")
    if stride == 1:
        return mask
    time_indices = torch.arange(mask.shape[1], device=mask.device)
    selected = (time_indices >= offset) & ((time_indices - offset) % stride == 0)
    return mask & selected.unsqueeze(0)


def flatten_valid_steps(tensor: Tensor, mask: Tensor) -> Tensor:
    filtered = tensor[mask]
    feature_dim = tensor.shape[-1]
    return filtered.reshape(-1, feature_dim)


def flatten_valid_pairs(anchor: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    return flatten_valid_steps(anchor, mask), flatten_valid_steps(target, mask)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    weighted = values * expanded_mask.to(values.dtype)
    denominator = expanded_mask.to(values.dtype).sum().clamp_min(1.0)
    return weighted.sum() / denominator
