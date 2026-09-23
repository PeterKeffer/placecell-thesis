"""Objective and auxiliary-head registry."""

from .registry import build_objectives, build_objectives_and_heads, compute_total_loss

__all__ = ["build_objectives", "build_objectives_and_heads", "compute_total_loss"]
