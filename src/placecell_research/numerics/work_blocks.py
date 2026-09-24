"""Splitting independent work into blocks, and how wide the pools that run them may be."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MAX_WORKERS = 8
WORKER_COUNT_ENV_VAR = "PLACECELL_ANALYSIS_WORKERS"

BLAS_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _slurm_allocated_cpu_count() -> int | None:
    """Cores this job was granted, or None when it is not running under SLURM."""
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus_per_task.isdigit():
        return max(1, int(slurm_cpus_per_task))
    return None


def _available_cpu_count() -> int:
    allocated_cpus = _slurm_allocated_cpu_count()
    if allocated_cpus is not None:
        return allocated_cpus
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _blas_thread_count() -> int:
    """Threads numpy/torch may start *underneath* one worker of this pool."""
    limits = [
        int(os.environ[name])
        for name in BLAS_THREAD_ENV_VARS
        if os.environ.get(name, "").strip().isdigit()
    ]
    return max(1, max(limits, default=1))


def worker_limit(env_var: str) -> int:
    """How wide a pool may run."""
    override = os.environ.get(env_var, "").strip()
    if override.isdigit():
        return max(1, int(override))
    allocated_cpus = _slurm_allocated_cpu_count()
    if allocated_cpus is None:
        return max(1, min(DEFAULT_MAX_WORKERS, _available_cpu_count()))
    return max(1, allocated_cpus // _blas_thread_count())


def analysis_worker_count(num_blocks: int) -> int:
    """Threads to use for num_blocks independent blocks of work."""
    return max(1, min(worker_limit(WORKER_COUNT_ENV_VAR), num_blocks))


def process_pool_plan(num_jobs: int) -> tuple[int, int]:
    """Worker processes for num_jobs independent jobs, and the thread limit each one may use."""
    worker_count = max(1, min(worker_limit(WORKER_COUNT_ENV_VAR), num_jobs))
    return worker_count, max(1, _available_cpu_count() // worker_count)


def index_blocks(num_items: int, block_count: int) -> list[range]:
    """Split range(num_items) into up to block_count contiguous, near-equal blocks."""
    edges = [num_items * block // block_count for block in range(block_count + 1)]
    return [
        range(start, stop)
        for start, stop in zip(edges[:-1], edges[1:], strict=False)
        if stop > start
    ]


def run_over_index_blocks(num_items: int, run_block: Callable[[range], None]) -> None:
    """Split range(num_items) into one contiguous block per worker and run the blocks."""
    if num_items <= 0:
        return
    blocks = index_blocks(num_items, analysis_worker_count(num_items))
    if len(blocks) == 1:
        run_block(blocks[0])
        return
    with ThreadPoolExecutor(max_workers=len(blocks)) as pool:
        for future in [pool.submit(run_block, block) for block in blocks]:
            future.result()
