"""Time helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return an ISO timestamp with a Z suffix."""
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")
