"""Training schedules."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from placecell_research.config.schema import SpatialTrainingConfig


def build_scheduler(optimizer: Optimizer, config: SpatialTrainingConfig):
    warmup_epochs = int(config.lr_warmup_epochs)
    if warmup_epochs == 0:
        if config.lr_schedule == "constant":
            return LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
        return CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    total_epochs = max(int(config.epochs), 1)
    decay_epochs = max(total_epochs - warmup_epochs, 1)

    def lr_factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if config.lr_schedule == "constant":
            return 1.0
        progress = min((epoch - warmup_epochs) / decay_epochs, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_factor)
