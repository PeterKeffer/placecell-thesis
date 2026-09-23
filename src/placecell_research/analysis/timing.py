"""Small timing helpers for analysis modules."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from time import perf_counter
from typing import Any

DEFER_MODULE_TIMING_LOGS_KEY = "_defer_module_timing_logs"


def record_timing(
    timing_seconds: dict[str, float],
    section_name: str,
    section_started_at: float,
) -> None:
    timing_seconds[section_name] = perf_counter() - section_started_at


def timing_text(
    timing_seconds: dict[str, Any],
    section_names: Iterable[str] | None = None,
) -> str:
    names = tuple(section_names) if section_names is not None else tuple(timing_seconds)
    return " ".join(
        f"{section_name}={float(timing_seconds[section_name]):.2f}s"
        for section_name in names
        if section_name in timing_seconds
    )


def with_deferred_module_timing_logs(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, DEFER_MODULE_TIMING_LOGS_KEY: True}


def log_timing(
    label: str,
    source_name: str,
    split_name: str,
    timing_seconds: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    section_names: Iterable[str] | None = None,
) -> None:
    log_timing_line(
        label,
        (source_name, split_name),
        timing_seconds,
        config=config,
        section_names=section_names,
    )


def log_timing_line(
    label: str,
    context_parts: Iterable[str],
    timing_seconds: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    section_names: Iterable[str] | None = None,
) -> None:
    if config is not None and bool(config.get(DEFER_MODULE_TIMING_LOGS_KEY, False)):
        return
    if not timing_seconds:
        return
    context = " ".join(str(context_part) for context_part in context_parts if context_part)
    print(
        f"[{label}] {context} timings {timing_text(timing_seconds, section_names)}",
        file=sys.stderr,
        flush=True,
    )
