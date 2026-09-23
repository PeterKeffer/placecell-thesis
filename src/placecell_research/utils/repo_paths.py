"""Helpers for resolving the repository root from config paths."""

from __future__ import annotations

from pathlib import Path


def find_repo_root_from_path(start_path: Path) -> Path:
    """Walk upward from a file or directory until the placecell_research repo root is found."""
    current = start_path.resolve()
    search_start = current if current.is_dir() else current.parent
    fallback_candidate: Path | None = None
    for candidate in [search_start, *search_start.parents]:
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "src" / "placecell_research").exists()
        ):
            return candidate
        if fallback_candidate is None and (candidate / "configs").exists():
            fallback_candidate = candidate
    if fallback_candidate is not None:
        return fallback_candidate
    raise FileNotFoundError(
        f"Could not locate the placecell_research repo root from {start_path}."
    )


def find_repo_root(config_path: Path) -> Path:
    """Walk upward from a config file until the placecell_research repo root is found."""
    return find_repo_root_from_path(config_path)
