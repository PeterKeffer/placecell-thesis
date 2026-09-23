"""Parallel width derived from the SLURM allocation, never from the machine."""

from __future__ import annotations

import os

DEFAULT_MAXIMUM_WORKERS = 8


def allocated_cpu_count() -> int:
    """CPUs SLURM granted this task, or 0 when the job is not running under SLURM."""
    value = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if not value.isdigit():
        return 0
    return int(value)


def parallel_read_worker_count(maximum: int = DEFAULT_MAXIMUM_WORKERS) -> int:
    """Reader width for this allocation, reserving one CPU for the consumer."""
    allocated = allocated_cpu_count()
    if allocated <= 1:
        return 0
    return max(0, min(int(maximum), allocated - 1))
