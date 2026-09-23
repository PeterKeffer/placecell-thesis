"""Angle utilities."""

from __future__ import annotations

import numpy as np


def wrap_radians(values: np.ndarray | float) -> np.ndarray:
    """Wrap radians to [-pi, pi] without changing represented direction."""
    value_array = np.asarray(values, dtype=np.float32)
    return np.arctan2(np.sin(value_array), np.cos(value_array)).astype(np.float32, copy=False)
