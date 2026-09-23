"""Objective and auxiliary-head registry."""

from .registry import build_objectives, compute_total_loss

__all__ = ["build_objectives", "compute_total_loss"]
