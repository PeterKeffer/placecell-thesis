"""The process pool that draws analysis figures."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import matplotlib

from ..numerics.work_blocks import worker_limit

RENDER_WORKER_COUNT_ENV_VAR = "PLACECELL_RENDER_WORKERS"
MIN_FIGURES_PER_RENDER_WORKER = 64


def render_worker_count() -> int:
    """Processes the render pool runs figure work on."""
    return worker_limit(RENDER_WORKER_COUNT_ENV_VAR)


def render_block_count(num_figures: int) -> int:
    """How many blocks to split num_figures renders into."""
    return max(1, min(render_worker_count(), num_figures // MIN_FIGURES_PER_RENDER_WORKER))


def _use_headless_backend() -> None:
    """Spawned workers draw off-screen; the default backend would want a display."""
    matplotlib.use("Agg")


_render_pool: ProcessPoolExecutor | None = None


def render_pool() -> ProcessPoolExecutor:
    """The process pool for figure rendering, started on first use and then kept."""
    global _render_pool
    if _render_pool is None:
        _render_pool = ProcessPoolExecutor(
            max_workers=render_worker_count(),
            mp_context=get_context("spawn"),
            initializer=_use_headless_backend,
        )
    return _render_pool
