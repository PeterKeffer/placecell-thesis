"""Seed management."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(slots=True)
class SeedBundle:
    """Reproducibility seed bundle."""

    global_seed: int = 42
    collection_seed: int | None = None
    split_seed: int | None = None
    training_seed: int | None = None

    def resolve(self) -> SeedBundle:
        def _resolved_seed(seed_value: int | None) -> int:
            return self.global_seed if seed_value is None else seed_value

        return SeedBundle(
            global_seed=self.global_seed,
            collection_seed=_resolved_seed(self.collection_seed),
            split_seed=_resolved_seed(self.split_seed),
            training_seed=_resolved_seed(self.training_seed),
        )

    def to_json(self) -> dict[str, int]:
        return asdict(self.resolve())

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and Torch random state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
