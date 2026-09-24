"""Fixed normalization statistics for frozen place-code features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class PlaceCodeStats:
    mean: np.ndarray
    std: np.ndarray
    active_rms: np.ndarray
    sample_count: int
    active_count: np.ndarray


def load_place_code_stats(path: Path | str) -> PlaceCodeStats:
    payload = np.load(Path(path), allow_pickle=False)
    return PlaceCodeStats(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        active_rms=np.asarray(payload["active_rms"], dtype=np.float32),
        sample_count=int(np.asarray(payload["sample_count"]).reshape(-1)[0]),
        active_count=np.asarray(payload["active_count"], dtype=np.int64),
    )
