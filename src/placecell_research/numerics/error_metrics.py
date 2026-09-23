"""Version-independent RMSE aggregation shared by position decoders."""

from __future__ import annotations

import numpy as np

RMSE_AGGREGATION = "root_mean_coordinate_mse_v1"


def rmse_from_coordinate_mse(coordinate_mse: np.ndarray) -> np.ndarray:
    """Root after averaging coordinate MSEs; the final axis contains coordinates."""
    return np.sqrt(np.mean(coordinate_mse, axis=-1, dtype=np.float64))


def root_mean_squared_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    """sqrt(mean over samples and coordinates of squared prediction error))."""
    error = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error), dtype=np.float64)))
