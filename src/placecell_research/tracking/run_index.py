"""Variant-slug indexes for run directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placecell_research.artifacts.ids import short_fingerprint
from placecell_research.tracking._run_paths import (
    _iter_run_paths,
    _load_run_manifest,
    _write_relative_symlink,
)

DEFAULT_VARIANT_INDEX_NAME = "by_variant"
DEFAULT_STAGE_INDEX_NAME = "by_stage"
MAX_RUN_INDEX_LINK_NAME_LENGTH = 180
DEFAULT_INDEXED_STAGE_NAME = "pipeline"
ENVIRONMENT_TOKEN_PREFIX = "env-"
UNKNOWN_ENVIRONMENT_NAME = "unknown_environment"


@dataclass(slots=True)
class RunIndexSummary:
    """Summary of one run-index rebuild."""

    run_root: Path
    index_dir: Path
    indexed_count: int
    skipped_count: int
    link_paths: list[Path]
    skipped_run_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_root": str(self.run_root),
            "index_dir": str(self.index_dir),
            "indexed_count": self.indexed_count,
            "skipped_count": self.skipped_count,
            "link_paths": [str(path) for path in self.link_paths],
            "skipped_run_ids": self.skipped_run_ids,
        }


def write_variant_run_link(
    run_root: Path,
    *,
    run_id: str,
    variant_slug: str,
    status: str = "",
    index_name: str = DEFAULT_VARIANT_INDEX_NAME,
    remove_existing_target_links: bool = True,
) -> Path:
    """Write one variant-slug symlink pointing at runs/by_id/<run_id>."""
    return _write_grouped_run_link(
        run_root,
        run_id=run_id,
        variant_slug=variant_slug,
        status=status,
        index_name=index_name,
        group_names=(),
        remove_existing_target_links=remove_existing_target_links,
    )


def write_stage_run_link(
    run_root: Path,
    *,
    run_id: str,
    stage_name: str,
    variant_slug: str,
    status: str = "",
    index_name: str = DEFAULT_STAGE_INDEX_NAME,
    remove_existing_target_links: bool = True,
) -> Path:
    """Write one stage-grouped symlink pointing at runs/by_id/<run_id>."""
    normalized_stage_name = _normalize_link_name(stage_name)
    if not normalized_stage_name:
        raise ValueError("stage_name must not be empty.")
    return _write_grouped_run_link(
        run_root,
        run_id=run_id,
        variant_slug=variant_slug,
        status=status,
        index_name=index_name,
        group_names=(normalized_stage_name,),
        remove_existing_target_links=remove_existing_target_links,
    )


def _write_grouped_run_link(
    run_root: Path,
    *,
    run_id: str,
    variant_slug: str,
    status: str,
    index_name: str,
    group_names: tuple[str, ...],
    remove_existing_target_links: bool,
) -> Path:
    normalized_variant_slug = _normalize_link_name(variant_slug)
    if not normalized_variant_slug:
        raise ValueError("variant_slug must not be empty.")
    normalized_run_id = _normalize_link_name(run_id)
    if not normalized_run_id:
        raise ValueError("run_id must not be empty.")
    run_path = run_root / "by_id" / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    index_dir = run_root / index_name
    environment_name, variant_name = _variant_index_parts(
        normalized_variant_slug,
        run_id=normalized_run_id,
    )
    visible_name = f"{_status_token(status)}__{variant_name}"
    environment_dir = index_dir
    for group_name in group_names:
        environment_dir = environment_dir / _shorten_link_name(group_name)
    environment_dir = environment_dir / _shorten_link_name(environment_name)
    if remove_existing_target_links:
        _remove_existing_links_to_target(index_dir, run_path)
    link_name = _unique_link_name(
        index_dir=environment_dir,
        run_path=run_path,
        variant_slug=visible_name,
        run_id=normalized_run_id,
    )
    return _write_relative_symlink(environment_dir / link_name, run_path)


def manifest_should_be_indexed(manifest: dict[str, Any]) -> bool:
    """Return true when a run manifest belongs in the default variant index."""
    return str(manifest.get("stage_name") or "").strip() == DEFAULT_INDEXED_STAGE_NAME


def manifest_should_be_stage_indexed(manifest: dict[str, Any]) -> bool:
    """Return true when a run manifest belongs in the stage index."""
    stage_name = str(manifest.get("stage_name") or "").strip()
    return bool(stage_name and stage_name != DEFAULT_INDEXED_STAGE_NAME)


def rebuild_variant_run_index(
    run_root: Path,
    *,
    index_name: str = DEFAULT_VARIANT_INDEX_NAME,
) -> RunIndexSummary:
    """Rebuild runs/<index_name>/ from run manifests under runs/by_id/."""
    index_dir = run_root / index_name
    _clear_index_symlinks(index_dir)
    link_paths: list[Path] = []
    skipped_run_ids: list[str] = []

    for run_path in _iter_run_paths(run_root):
        manifest = _load_run_manifest(run_path)
        run_id = str(manifest.get("run_id") or run_path.name).strip()
        variant_slug = str(manifest.get("variant_slug") or "").strip()
        if not variant_slug or not manifest_should_be_indexed(manifest):
            skipped_run_ids.append(run_id)
            continue
        link_paths.append(
            write_variant_run_link(
                run_root,
                run_id=run_id,
                variant_slug=variant_slug,
                status=str(manifest.get("status") or ""),
                index_name=index_name,
                remove_existing_target_links=False,
            )
        )

    return RunIndexSummary(
        run_root=run_root,
        index_dir=index_dir,
        indexed_count=len(link_paths),
        skipped_count=len(skipped_run_ids),
        link_paths=link_paths,
        skipped_run_ids=skipped_run_ids,
    )


def rebuild_stage_run_index(
    run_root: Path,
    *,
    index_name: str = DEFAULT_STAGE_INDEX_NAME,
) -> RunIndexSummary:
    """Rebuild runs/<index_name>/ from non-pipeline run manifests."""
    index_dir = run_root / index_name
    _clear_index_symlinks(index_dir)
    link_paths: list[Path] = []
    skipped_run_ids: list[str] = []

    for run_path in _iter_run_paths(run_root):
        manifest = _load_run_manifest(run_path)
        run_id = str(manifest.get("run_id") or run_path.name).strip()
        stage_name = str(manifest.get("stage_name") or "").strip()
        variant_slug = str(manifest.get("variant_slug") or "").strip()
        if not variant_slug or not manifest_should_be_stage_indexed(manifest):
            skipped_run_ids.append(run_id)
            continue
        link_paths.append(
            write_stage_run_link(
                run_root,
                run_id=run_id,
                stage_name=stage_name,
                variant_slug=variant_slug,
                status=str(manifest.get("status") or ""),
                index_name=index_name,
                remove_existing_target_links=False,
            )
        )

    return RunIndexSummary(
        run_root=run_root,
        index_dir=index_dir,
        indexed_count=len(link_paths),
        skipped_count=len(skipped_run_ids),
        link_paths=link_paths,
        skipped_run_ids=skipped_run_ids,
    )


def _normalize_link_name(value: str) -> str:
    return value.strip().replace("/", "_")


def _variant_index_parts(variant_slug: str, *, run_id: str) -> tuple[str, str]:
    parts = [part for part in variant_slug.split("__") if part]
    first_part = parts[0] if parts else ""
    if first_part.startswith(ENVIRONMENT_TOKEN_PREFIX):
        environment_name = first_part[len(ENVIRONMENT_TOKEN_PREFIX) :] or UNKNOWN_ENVIRONMENT_NAME
        variant_name = "__".join(parts[1:]) or run_id
        return environment_name, variant_name
    return UNKNOWN_ENVIRONMENT_NAME, variant_slug


def _status_token(status: str) -> str:
    normalized_status = status.strip().lower().replace("-", "_")
    if normalized_status in {"completed", "reused", "passed"}:
        return "done"
    if normalized_status in {"failed", "error"}:
        return "fail"
    if normalized_status in {"initialized", "initializing"}:
        return "init"
    if normalized_status in {"running", "submitted", "queued"}:
        return "run"
    if normalized_status in {"interrupted", "cancelled", "canceled", "timeout"}:
        return "stop"
    return "unk"


def _shorten_link_name(name: str, *, max_length: int = MAX_RUN_INDEX_LINK_NAME_LENGTH) -> str:
    if len(name) <= max_length:
        return name
    digest = short_fingerprint(name, length=10)
    keep = max_length - len(digest) - 2
    if keep <= 0:
        return digest[:max_length]
    return f"{name[:keep].rstrip('._-')}__{digest}"


def _unique_link_name(*, index_dir: Path, run_path: Path, variant_slug: str, run_id: str) -> str:
    link_name = _shorten_link_name(variant_slug)
    link_path = index_dir / link_name
    if not _link_points_to(link_path, run_path) and (link_path.exists() or link_path.is_symlink()):
        link_name = _shorten_link_name(f"{variant_slug}__{run_id}")
        link_path = index_dir / link_name
    if not _link_points_to(link_path, run_path) and (link_path.exists() or link_path.is_symlink()):
        raise FileExistsError(f"Run index link already exists for another target: {link_path}")
    return link_name


def _link_points_to(link_path: Path, target_path: Path) -> bool:
    if not link_path.is_symlink():
        return False
    return link_path.resolve() == target_path.resolve()


def _remove_existing_links_to_target(index_dir: Path, target_path: Path) -> None:
    if not index_dir.exists():
        return
    target_resolved = target_path.resolve()
    for path in sorted(index_dir.rglob("*"), reverse=True):
        try:
            if path.is_symlink() and path.resolve() == target_resolved:
                path.unlink(missing_ok=True)
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except (FileNotFoundError, NotADirectoryError):
            continue


def _clear_index_symlinks(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    for path in index_dir.iterdir():
        _clear_index_path(path)


def _clear_index_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if path.is_dir():
        for child_path in path.iterdir():
            _clear_index_path(child_path)
        path.rmdir()
        return
    if path.exists():
        raise FileExistsError(f"Refusing to remove non-symlink run index path: {path}")
