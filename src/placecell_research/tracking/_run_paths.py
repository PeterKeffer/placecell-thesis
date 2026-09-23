"""Shared read helpers and atomic symlink writer for run directories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def iter_run_paths(run_root: Path) -> list[Path]:
    by_id_dir = run_root / "by_id"
    if not by_id_dir.exists():
        return []
    return sorted(path for path in by_id_dir.iterdir() if path.is_dir())


def read_run_manifest(run_path: Path) -> dict[str, Any]:
    manifest_path = run_path / "manifests" / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def write_relative_symlink(link_path: Path, target_path: Path) -> Path:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_target = target_path.resolve()
    relative_target = Path(os.path.relpath(resolved_target, start=link_path.parent.resolve()))
    tmp_path = link_path.with_name(f".{link_path.name}.tmp.{os.getpid()}")
    if tmp_path.is_symlink() or tmp_path.exists():
        tmp_path.unlink()
    tmp_path.symlink_to(relative_target, target_is_directory=resolved_target.is_dir())
    os.replace(tmp_path, link_path)
    return link_path


def link_if_absent(link_path: Path, target_path: Path) -> None:
    """Link link_path to an existing target_path unless link_path is already taken."""
    if not target_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return
    relative_target = Path(os.path.relpath(target_path.resolve(), start=link_path.parent.resolve()))
    link_path.symlink_to(relative_target, target_is_directory=target_path.is_dir())
