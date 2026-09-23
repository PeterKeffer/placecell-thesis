"""Run the per-source in-training validation metrics on worker processes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context, shared_memory

import numpy as np

from placecell_research.numerics.work_blocks import WORKER_COUNT_ENV_VAR, process_pool_plan

from .source_decode import SourceDecodeSettings, source_decode_metrics

MIN_ELEMENTS_PER_SOURCE_WORKER = 1 << 23


@dataclass(frozen=True)
class SharedArraySpec:
    """Everything a worker needs to map one published array."""

    name: str
    shape: tuple[int, ...]
    dtype: str


class SharedArrays:
    """The shared-memory blocks the source jobs read, owned by the process that published them."""

    def __init__(self) -> None:
        self._blocks: list[shared_memory.SharedMemory] = []

    def publish(self, array: np.ndarray | None) -> SharedArraySpec | None:
        if array is None:
            return None
        contiguous = np.ascontiguousarray(array)
        spec, view = self._allocate(contiguous.shape, contiguous.dtype)
        view[...] = contiguous
        return spec

    def publish_chunks(self, chunks: list[np.ndarray]) -> SharedArraySpec:
        """Concatenate chunks along axis 0 straight into a block, skipping the private copy."""
        rows = sum(int(chunk.shape[0]) for chunk in chunks)
        spec, view = self._allocate((rows, *chunks[0].shape[1:]), chunks[0].dtype)
        np.concatenate(chunks, axis=0, out=view)
        return spec

    def _allocate(self, shape, dtype) -> tuple[SharedArraySpec, np.ndarray]:
        item_count = int(np.prod(shape)) if len(shape) else 1
        item_size = int(np.dtype(dtype).itemsize)
        block = shared_memory.SharedMemory(create=True, size=max(1, item_count * item_size))
        self._blocks.append(block)
        return (
            SharedArraySpec(block.name, tuple(int(size) for size in shape), np.dtype(dtype).str),
            np.ndarray(shape, dtype=dtype, buffer=block.buf),
        )

    def close(self) -> None:
        for block in self._blocks:
            block.close()
            block.unlink()
        self._blocks.clear()

    def __enter__(self) -> SharedArrays:
        return self

    def __exit__(self, *_exception) -> None:
        self.close()


@dataclass(frozen=True)
class SourceDecodeJob:
    """One source's decode: which representation, against which shared trajectory arrays."""

    source: str
    representation: SharedArraySpec
    position: SharedArraySpec
    valid: SharedArraySpec
    heading: SharedArraySpec | None
    kinematics: SharedArraySpec | None
    settings: SourceDecodeSettings


def _map(spec: SharedArraySpec | None):
    if spec is None:
        return None, None
    block = shared_memory.SharedMemory(name=spec.name)
    return block, np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=block.buf)


def _run_job(job: SourceDecodeJob) -> dict[str, float]:
    blocks = []
    try:
        arrays = []
        for spec in (job.representation, job.position, job.valid, job.heading, job.kinematics):
            block, array = _map(spec)
            blocks.append(block)
            arrays.append(array)
        return source_decode_metrics(*arrays, settings=job.settings)
    finally:
        for block in blocks:
            if block is not None:
                block.close()


def _limit_inner_threads(thread_count: int) -> None:
    os.environ[WORKER_COUNT_ENV_VAR] = str(thread_count)


def should_use_worker_pool(source_element_counts: Sequence[int]) -> bool:
    """Whether this set of sources is worth publishing to workers."""
    if len(source_element_counts) < 2:
        return False
    worker_count, _ = process_pool_plan(len(source_element_counts))
    if worker_count < 2:
        return False
    return all(count >= MIN_ELEMENTS_PER_SOURCE_WORKER for count in source_element_counts)


def decode_sources(jobs: list[SourceDecodeJob]) -> dict[str, dict[str, float]]:
    """Metrics per source, one worker process per source."""
    worker_count, inner_thread_count = process_pool_plan(len(jobs))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
        initializer=_limit_inner_threads,
        initargs=(inner_thread_count,),
    ) as pool:
        return dict(zip([job.source for job in jobs], pool.map(_run_job, jobs), strict=False))
