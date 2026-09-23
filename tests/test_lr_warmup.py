from __future__ import annotations

import math

import pytest
import torch

from placecell_research.config.schema import SpatialTrainingConfig
from placecell_research.training.scheduling import build_scheduler


def _learning_rates(config: SpatialTrainingConfig) -> list[float]:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, config)
    rates = []
    for _ in range(config.epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


def test_warmup_ramps_linearly_then_holds_constant() -> None:
    config = SpatialTrainingConfig(epochs=8, lr_schedule="constant", lr_warmup_epochs=4)
    assert _learning_rates(config) == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_warmup_then_cosine_decays_to_zero_at_the_last_epoch() -> None:
    config = SpatialTrainingConfig(epochs=6, lr_schedule="cosine", lr_warmup_epochs=2)
    rates = _learning_rates(config)
    assert rates[:2] == pytest.approx([0.5, 1.0])
    expected_tail = [0.5 * (1.0 + math.cos(math.pi * step / 4)) for step in range(4)]
    assert rates[2:] == pytest.approx(expected_tail)


def test_zero_warmup_keeps_the_historical_schedulers() -> None:
    constant = SpatialTrainingConfig(epochs=3, lr_schedule="constant")
    assert _learning_rates(constant) == pytest.approx([1.0, 1.0, 1.0])
    cosine = SpatialTrainingConfig(epochs=3, lr_schedule="cosine")
    assert _learning_rates(cosine)[0] == pytest.approx(1.0)
    assert _learning_rates(cosine)[-1] < 0.3
