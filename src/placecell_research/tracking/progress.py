"""Reusable live progress reporting with sparse non-TTY logging."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from time import perf_counter
from typing import TextIO


@dataclass(slots=True)
class ProgressUpdate:
    """Progress snapshot for a long-running stage or phase."""

    completed: int
    total: int
    elapsed_seconds: float
    unit_name: str = "items"
    detail: str | None = None


class ConsoleProgressReporter:
    """Render compact live progress without spamming captured logs."""

    def __init__(
        self,
        stage_name: str,
        *,
        stream: TextIO | None = None,
        non_tty_milestones: int = 5,
        non_tty_max_silence_seconds: float = 120.0,
    ) -> None:
        self.stage_name = stage_name
        self.stream = stream or sys.stderr
        self.non_tty_milestones = max(1, int(non_tty_milestones))
        self.non_tty_max_silence_seconds = max(0.0, float(non_tty_max_silence_seconds))
        self._last_rendered_width = 0
        self._last_logged_bucket = -1
        self._last_logged_completed: int | None = None
        self._last_logged_detail: str | None = None
        self._last_logged_elapsed_seconds: float | None = None

    def __call__(self, update: ProgressUpdate) -> None:
        total = max(1, int(update.total))
        completed = max(0, min(int(update.completed), total))
        percent_complete = 100.0 * completed / total
        message = (
            f"[{self.stage_name}] {update.unit_name} {completed}/{total} "
            f"({percent_complete:5.1f}%) elapsed={update.elapsed_seconds:6.1f}s"
        )
        if update.detail:
            message = f"{message} | {update.detail}"

        if self.stream.isatty():
            self._last_rendered_width = max(self._last_rendered_width, len(message))
            self.stream.write("\r" + message.ljust(self._last_rendered_width))
            if completed >= total:
                self.stream.write("\n")
            self.stream.flush()
            return

        bucket = int((self.non_tty_milestones * completed) / total)
        detail_changed = update.detail != self._last_logged_detail
        elapsed_since_last_log = (
            None
            if self._last_logged_elapsed_seconds is None
            else max(0.0, float(update.elapsed_seconds) - self._last_logged_elapsed_seconds)
        )
        silence_timeout_reached = (
            elapsed_since_last_log is not None
            and self.non_tty_max_silence_seconds > 0.0
            and elapsed_since_last_log >= self.non_tty_max_silence_seconds
        )
        boundary_update = completed in {0, 1, total} and completed != self._last_logged_completed
        should_log = (
            boundary_update
            or bucket > self._last_logged_bucket
            or detail_changed
            or silence_timeout_reached
        )
        if not should_log:
            return
        self._last_logged_bucket = max(self._last_logged_bucket, bucket)
        self._last_logged_completed = completed
        self._last_logged_detail = update.detail
        self._last_logged_elapsed_seconds = float(update.elapsed_seconds)
        self.stream.write(message + "\n")
        self.stream.flush()


class ProgressTracker:
    """Small stateful helper for stages that report incremental progress."""

    def __init__(
        self,
        reporter: ConsoleProgressReporter,
        *,
        total: int,
        unit_name: str,
        detail: str | None = None,
    ) -> None:
        self.reporter = reporter
        self.total = max(1, int(total))
        self.unit_name = unit_name
        self.detail = detail
        self.completed = 0
        self.started_at = perf_counter()

    def emit(self, *, detail: str | None = None) -> None:
        self.reporter(
            ProgressUpdate(
                completed=self.completed,
                total=self.total,
                elapsed_seconds=perf_counter() - self.started_at,
                unit_name=self.unit_name,
                detail=self.detail if detail is None else detail,
            )
        )

    def advance(self, amount: int = 1, *, detail: str | None = None) -> None:
        self.completed = min(self.total, self.completed + int(amount))
        self.emit(detail=detail)
