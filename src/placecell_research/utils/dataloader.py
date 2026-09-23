"""DataLoader runtime helpers."""

from __future__ import annotations


def spawned_worker_kwargs(num_workers: int) -> dict[str, str]:
    """Use spawn whenever PyTorch starts DataLoader worker processes."""
    if int(num_workers) <= 0:
        return {}
    return {"multiprocessing_context": "spawn"}
