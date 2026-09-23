"""Helpers for resolving the repository root from config paths."""

from __future__ import annotations

import os
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


def configs_root() -> Path:
    """configs/ of the code snapshot this process imports, else of the checkout at cwd."""
    snapshot = os.environ.get("PLACECELL_CODE_SNAPSHOT", "").strip()
    if snapshot:
        return Path(snapshot) / "configs"
    return find_repo_root_from_path(Path.cwd()) / "configs"


def resolve_experiment_config_path(study_config_path: Path, reference: str) -> Path:
    """Resolve a study's base experiment from a path or experiment preset name."""
    reference_text = str(reference).strip()
    if not reference_text:
        raise FileNotFoundError("Base experiment reference must not be empty.")

    candidate = Path(reference_text).expanduser()
    repo_root = find_repo_root(study_config_path)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                candidate,
                study_config_path.resolve().parent / candidate,
                repo_root / candidate,
            ]
        )
        preset_reference = (
            candidate
            if candidate.suffix.lower() in {".yaml", ".yml"}
            else Path(f"{reference_text}.yaml")
        )
        candidates.append(repo_root / "configs" / "experiment" / preset_reference)

    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not resolve base experiment {reference_text!r} from {study_config_path}."
    )
