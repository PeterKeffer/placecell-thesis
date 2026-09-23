"""Tensor helper functions."""

from __future__ import annotations

from torch import Tensor


def masked_mean(
    values: Tensor, mask: Tensor | None, dim: int | tuple[int, ...] | None = None
) -> Tensor:
    """Compute a mean over valid entries only."""
    if mask is None:
        return values.mean(dim=dim)
    mask_f = mask.to(values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    weighted = values * mask_f
    if dim is None:
        denom = mask_f.sum().clamp_min(1.0)
        return weighted.sum() / denom
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return weighted.sum(dim=dim) / denom
