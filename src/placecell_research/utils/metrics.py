"""Transfer scalar metrics without synchronizing once per value."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

import torch
from torch import Tensor

_MetricKey = TypeVar("_MetricKey")


def _stack_metric_tensors(
    metrics: dict[_MetricKey, float | Tensor],
) -> Iterator[tuple[list[_MetricKey], Tensor]]:
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[_MetricKey, Tensor]]] = {}
    for name, value in metrics.items():
        if isinstance(value, Tensor):
            groups.setdefault((value.device, value.dtype), []).append(
                (name, value.detach().reshape(()))
            )
    for group in groups.values():
        yield [name for name, _ in group], torch.stack([value for _, value in group])


def materialize_metric_values(
    metrics: dict[_MetricKey, float | Tensor],
) -> dict[_MetricKey, float]:
    """Copy each device/dtype group once, without rounding values through another dtype."""
    materialized = {
        name: float(value) for name, value in metrics.items() if not isinstance(value, Tensor)
    }
    for names, values in _stack_metric_tensors(metrics):
        materialized.update(zip(names, map(float, values.cpu().tolist()), strict=False))
    return {name: materialized[name] for name in metrics}


def snapshot_metric_values(metrics: dict[str, float | Tensor]) -> dict[str, float | Tensor]:
    """Copy scalar groups on their existing device so later producer updates cannot alter them."""
    snapshot: dict[str, float | Tensor] = {
        name: float(value) for name, value in metrics.items() if not isinstance(value, Tensor)
    }
    for names, values in _stack_metric_tensors(metrics):
        snapshot.update(zip(names, values.unbind(), strict=False))
    return {name: snapshot[name] for name in metrics}


def sum_metric_batches(batches: list[dict[str, float | Tensor]]) -> dict[str, float]:
    """Transfer buffered scalars together, retaining Python's batch-wise addition order."""
    values = materialize_metric_values(
        {
            (index, name): value
            for index, metrics in enumerate(batches)
            for name, value in metrics.items()
        }
    )
    totals: dict[str, float] = {}
    for (_, name), value in values.items():
        totals[name] = totals.get(name, 0.0) + value
    return totals
