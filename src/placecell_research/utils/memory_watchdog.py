"""Small RSS helpers for long-running HPC processes."""

from __future__ import annotations

import os


def read_process_rss_mb(pid: int) -> float | None:
    try:
        page_size_bytes = os.sysconf("SC_PAGE_SIZE")
        with open(f"/proc/{int(pid)}/statm", encoding="utf-8") as statm_file:
            fields = statm_file.read().split()
    except (OSError, ValueError):
        return None
    if len(fields) < 2:
        return None
    try:
        resident_pages = int(fields[1])
    except ValueError:
        return None
    return resident_pages * float(page_size_bytes) / (1024.0 * 1024.0)


def read_current_rss_mb() -> float | None:
    return read_process_rss_mb(os.getpid())
