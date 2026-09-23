"""Collection safety helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from placecell_research.config.schema import CollectionSafetyConfig, EnvironmentConfig


def _should_skip_env_close_on_slurm(safety: CollectionSafetyConfig) -> bool:
    return bool(os.environ.get("SLURM_JOB_ID")) and bool(safety.skip_env_close_on_slurm)


def _render_topdown_worker(
    environment_config: EnvironmentConfig,
    seed: int,
    safety: CollectionSafetyConfig,
    connection,
) -> None:  # type: ignore[no-untyped-def]
    from placecell_research.envs.builder import build_environment

    adapter = None
    try:
        adapter = build_environment(environment_config, seed=seed)
        connection.send(adapter.render_topdown())
    except Exception:
        connection.send(None)
    finally:
        if adapter is not None and not _should_skip_env_close_on_slurm(safety):
            adapter.close()
        connection.close()


def configure_start_method(safety: CollectionSafetyConfig) -> mp.context.BaseContext:
    """Return the configured multiprocessing context."""
    if safety.multiprocessing_start_method != "spawn":
        raise ValueError("Collection requires multiprocessing_start_method='spawn'.")
    return mp.get_context("spawn")


def render_topdown_with_timeout(
    environment_config: EnvironmentConfig,
    seed: int,
    safety: CollectionSafetyConfig,
) -> np.ndarray | None:
    """Render topdown frame in a short-lived spawn subprocess."""
    context = configure_start_method(safety)
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_render_topdown_worker,
        args=(environment_config, seed, safety, child_connection),
    )
    process.start()
    child_connection.close()
    process.join(timeout=safety.render_timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            kill_method = getattr(process, "kill", None)
            if callable(kill_method):
                kill_method()
                process.join(timeout=1.0)
        parent_connection.close()
        return None
    try:
        if parent_connection.poll():
            return parent_connection.recv()
        return None
    finally:
        parent_connection.close()
