"""Spawned worker supervision for parallel dataset collection."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING, Any

import numpy as np

from placecell_research.collection.safety import configure_start_method
from placecell_research.utils.memory_watchdog import read_process_rss_mb as _read_process_rss_mb

if TYPE_CHECKING:
    from placecell_research.config.schema import CollectionConfig, EnvironmentConfig


@dataclass(slots=True)
class _ParallelCollectionWorker:
    shard_index: int
    seeds: list[int]
    process: Any
    started_at: float
    last_progress_at: float | None = None


@dataclass(slots=True)
class ParallelEpisodeBatch:
    shard_index: int
    seeds: list[int]
    episodes: list[dict[str, np.ndarray | bool | int]]
    num_actions: int


def _format_seed_span(seeds: list[int]) -> str:
    if not seeds:
        return "empty shard"
    if len(seeds) == 1:
        return f"seed {int(seeds[0])}"
    return f"seeds {int(seeds[0])}..{int(seeds[-1])}"


def _terminate_collection_process(process: Any) -> None:
    if process.exitcode is not None:
        process.join(timeout=0.1)
        return
    process_group_id = int(process.pid)
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        process.join(timeout=0.1)
        return
    process.join(timeout=1.0)
    if process.exitcode is not None:
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.join(timeout=1.0)


def _stop_parallel_workers(active_workers: dict[int, _ParallelCollectionWorker]) -> None:
    for worker in list(active_workers.values()):
        _terminate_collection_process(worker.process)
    active_workers.clear()


_PARENT_RESERVED_CPUS = 1

AUTO_COLLECTION_WORKERS = 0


def _allocated_cpu_count() -> int | None:
    """CPUs SLURM allocated to this job, or None when there is no allocation to read."""
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if not allocated_cpus.isdigit():
        return None
    return max(1, int(allocated_cpus))


def resolve_collection_worker_count(collection_config: CollectionConfig) -> int:
    """How many shard workers may run at once."""
    configured_workers = int(collection_config.safety.num_workers)
    if configured_workers > 0:
        return configured_workers
    allocated_cpus = _allocated_cpu_count()
    if allocated_cpus is None:
        return 1
    return max(1, allocated_cpus - _PARENT_RESERVED_CPUS)


def _resolve_collection_shards(
    seeds: list[int],
    collection_config: CollectionConfig,
    max_workers: int,
) -> list[list[int]]:
    if collection_config.safety.worker_max_episodes > 0:
        shard_size = max(1, int(collection_config.safety.worker_max_episodes))
    else:
        target_worker_waves = 2
        shard_size = int(np.ceil(len(seeds) / max_workers / target_worker_waves))
        shard_size = max(8, min(32, shard_size))
    return [seeds[index : index + shard_size] for index in range(0, len(seeds), shard_size)]


def _resolve_collection_memory_watchdog_limit_mb(collection_config: CollectionConfig) -> int:
    explicit_limit_mb = int(collection_config.safety.memory_watchdog_limit_mb)
    if explicit_limit_mb > 0:
        return explicit_limit_mb
    slurm_memory_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_memory_mb is None:
        return 0
    try:
        memory_limit_mb = int(slurm_memory_mb)
    except ValueError:
        return 0
    if memory_limit_mb <= 0:
        return 0
    return int(float(memory_limit_mb) * 0.9)


def _collect_episode_batch_worker(
    result_queue,
    batch_collector: Callable[..., tuple[list[dict[str, np.ndarray | bool | int]], int]],
    environment_config: EnvironmentConfig,
    collection_config: CollectionConfig,
    shard_index: int,
    episode_seeds: list[int],
    startup_delay_seconds: float,
) -> None:
    if mp.parent_process() is not None:
        os.setsid()
    try:

        def _emit_episode_progress(completed: int) -> None:
            result_queue.put(
                {
                    "status": "progress",
                    "shard_index": int(shard_index),
                    "completed": int(completed),
                }
            )

        batch_episodes, batch_num_actions = batch_collector(
            environment_config,
            collection_config,
            episode_seeds,
            startup_delay_seconds,
            on_episode_complete=_emit_episode_progress,
        )
        result_queue.put(
            {
                "status": "ok",
                "shard_index": int(shard_index),
                "episodes": batch_episodes,
                "num_actions": int(batch_num_actions),
            }
        )
    except Exception as exc:  # pragma: no cover
        result_queue.put(
            {
                "status": "error",
                "shard_index": int(shard_index),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _launch_collection_worker(
    context,
    result_queue,
    *,
    batch_collector: Callable[..., tuple[list[dict[str, np.ndarray | bool | int]], int]],
    shard_index: int,
    environment_config: EnvironmentConfig,
    collection_config: CollectionConfig,
    shard: list[int],
    startup_delay_seconds: float,
) -> _ParallelCollectionWorker:
    process = context.Process(
        target=_collect_episode_batch_worker,
        args=(
            result_queue,
            batch_collector,
            environment_config,
            collection_config,
            int(shard_index),
            list(shard),
            float(startup_delay_seconds),
        ),
    )
    process.start()
    return _ParallelCollectionWorker(
        shard_index=int(shard_index),
        seeds=list(shard),
        process=process,
        started_at=time.monotonic(),
    )


_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pin_single_threaded_blas() -> None:
    """Cap BLAS/OpenMP to 1 thread so spawned workers cannot oversubscribe the node."""
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")


def iter_parallel_episode_batches(
    environment_config: EnvironmentConfig,
    collection_config: CollectionConfig,
    seeds: list[int],
    *,
    batch_collector: Callable[..., tuple[list[dict[str, np.ndarray | bool | int]], int]],
    on_episode_complete: Callable[[int], None] | None = None,
) -> Iterator[ParallelEpisodeBatch]:
    _pin_single_threaded_blas()
    context = configure_start_method(collection_config.safety)
    if not seeds:
        raise ValueError("Parallel collection requires at least one episode seed.")

    max_workers = resolve_collection_worker_count(collection_config)
    shards = _resolve_collection_shards(seeds, collection_config, max_workers)
    stagger_seconds = max(0.0, float(collection_config.safety.stagger_worker_start_seconds))
    watchdog_timeout_seconds = float(collection_config.safety.watchdog_timeout_seconds)
    memory_watchdog_limit_mb = _resolve_collection_memory_watchdog_limit_mb(collection_config)
    poll_interval_seconds = min(1.0, max(0.1, watchdog_timeout_seconds / 10.0))
    result_queue = context.Queue()
    active_workers: dict[int, _ParallelCollectionWorker] = {}
    next_shard_index = 0
    try:
        while next_shard_index < len(shards) or active_workers:
            while next_shard_index < len(shards) and len(active_workers) < max_workers:
                shard = list(shards[next_shard_index])
                startup_delay_seconds = (next_shard_index % max_workers) * stagger_seconds
                worker = _launch_collection_worker(
                    context,
                    result_queue,
                    batch_collector=batch_collector,
                    shard_index=next_shard_index,
                    environment_config=environment_config,
                    collection_config=collection_config,
                    shard=shard,
                    startup_delay_seconds=startup_delay_seconds,
                )
                active_workers[worker.shard_index] = worker
                next_shard_index += 1
            try:
                message = result_queue.get(timeout=poll_interval_seconds)
            except Empty:
                message = None
            if message is not None:
                shard_index = int(message["shard_index"])
                worker = active_workers.get(shard_index)
                if worker is None:
                    raise RuntimeError(
                        f"Collection received an unexpected result for shard {shard_index}."
                    )
                if message["status"] == "progress":
                    worker.last_progress_at = time.monotonic()
                    if on_episode_complete is not None:
                        on_episode_complete(int(message["completed"]))
                    continue
                active_workers.pop(shard_index, None)
                worker.process.join(timeout=1.0)
                if worker.process.exitcode is None:
                    _terminate_collection_process(worker.process)
                if message["status"] != "ok":
                    _stop_parallel_workers(active_workers)
                    seed_span = _format_seed_span(worker.seeds)
                    raise RuntimeError(
                        "Parallel collection worker failed for "
                        f"shard {shard_index} ({seed_span}): "
                        f"{message['error_type']}: {message['error_message']}\n"
                        f"{message['traceback']}"
                    )
                yield ParallelEpisodeBatch(
                    shard_index=shard_index,
                    seeds=list(worker.seeds),
                    episodes=list(message["episodes"]),
                    num_actions=int(message["num_actions"]),
                )
            current_time = time.monotonic()
            if memory_watchdog_limit_mb > 0:
                worker_rss_mb: list[tuple[int, float]] = []
                for shard_index, worker in active_workers.items():
                    rss_mb = _read_process_rss_mb(worker.process.pid)
                    if rss_mb is not None:
                        worker_rss_mb.append((int(shard_index), float(rss_mb)))
                total_worker_rss_mb = sum(rss_mb for _, rss_mb in worker_rss_mb)
                if worker_rss_mb and total_worker_rss_mb > memory_watchdog_limit_mb:
                    _stop_parallel_workers(active_workers)
                    rss_summary = ", ".join(
                        f"shard {shard_index}: {rss_mb:.1f} MB"
                        for shard_index, rss_mb in worker_rss_mb
                    )
                    raise MemoryError(
                        "Parallel collection RSS watchdog exceeded "
                        f"{memory_watchdog_limit_mb} MB "
                        f"(total={total_worker_rss_mb:.1f} MB; {rss_summary})."
                    )
            for shard_index, worker in list(active_workers.items()):
                last_progress_at = (
                    worker.started_at
                    if worker.last_progress_at is None
                    else worker.last_progress_at
                )
                if current_time - last_progress_at <= watchdog_timeout_seconds:
                    if worker.process.exitcode not in (None, 0):
                        _stop_parallel_workers(active_workers)
                        raise RuntimeError(
                            "Parallel collection worker exited before reporting a result for "
                            f"shard {shard_index} ({_format_seed_span(worker.seeds)}), "
                            f"exit code {worker.process.exitcode}."
                        )
                    continue
                _terminate_collection_process(worker.process)
                active_workers.pop(shard_index, None)
                _stop_parallel_workers(active_workers)
                raise TimeoutError(
                    "Parallel collection watchdog timed out after "
                    f"{watchdog_timeout_seconds:.1f}s for shard {shard_index} "
                    f"({_format_seed_span(worker.seeds)})."
                )
    finally:
        _stop_parallel_workers(active_workers)
        result_queue.cancel_join_thread()
        result_queue.close()
