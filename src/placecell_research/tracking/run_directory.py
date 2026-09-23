"""Run directory creation and bookkeeping."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from placecell_research.tracking._run_paths import (
    iter_run_paths,
    read_run_manifest,
    write_relative_symlink,
)
from placecell_research.tracking.naming import RunIdentity, slurm_job_id_from_token
from placecell_research.tracking.run_index import (
    manifest_should_be_indexed,
    manifest_should_be_stage_indexed,
    write_stage_run_link,
    write_variant_run_link,
)


def _merge_payload(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_payload(merged[key], value)
        else:
            merged[key] = value
    return merged


def _slurm_job_id_from_runtime(run_id: str) -> str:
    env_job_id = slurm_job_id_from_token("", env=os.environ)
    if env_job_id:
        return env_job_id
    for token in (
        str(os.environ.get("PLACECELL_RUN_ID", "")).strip(),
        run_id,
    ):
        job_id = slurm_job_id_from_token(token)
        if job_id:
            return job_id
    return ""


def _slurm_job_name_from_runtime() -> str:
    job_name = str(os.environ.get("SLURM_JOB_NAME", "placecell_research")).strip()
    return job_name or "placecell_research"


def _slurm_log_path_from_runtime(run_root: Path, run_id: str) -> Path | None:
    job_id = _slurm_job_id_from_runtime(run_id)
    if not job_id:
        return None
    return run_root / "slurm_logs" / f"{_slurm_job_name_from_runtime()}_{job_id}.out"


def _slurm_log_path_from_manifest(run_root: Path, payload: dict[str, Any]) -> Path | None:
    slurm_payload = payload.get("slurm")
    if not isinstance(slurm_payload, dict):
        return None
    explicit_log_path = str(slurm_payload.get("log_path", "")).strip()
    if explicit_log_path:
        return Path(explicit_log_path)
    job_id = str(slurm_payload.get("job_id", "")).strip()
    if not job_id:
        return None
    job_name = str(slurm_payload.get("job_name", "placecell_research")).strip()
    if not job_name:
        job_name = "placecell_research"
    return run_root / "slurm_logs" / f"{job_name}_{job_id}.out"


@dataclass(slots=True)
class RunSlurmLogLinkSummary:
    """Summary of Slurm-log link repairs under runs/by_id."""

    run_root: Path
    linked_count: int
    skipped_count: int
    link_paths: list[Path]
    skipped_run_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_root": str(self.run_root),
            "linked_count": self.linked_count,
            "skipped_count": self.skipped_count,
            "link_paths": [str(path) for path in self.link_paths],
            "skipped_run_ids": self.skipped_run_ids,
        }


@dataclass(slots=True)
class RunResultShortcutSummary:
    """Summary of root-level result-shortcut repairs under runs/by_id."""

    run_root: Path
    linked_count: int
    skipped_count: int
    link_paths: list[Path]
    skipped_paths: list[Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_root": str(self.run_root),
            "linked_count": self.linked_count,
            "skipped_count": self.skipped_count,
            "link_paths": [str(path) for path in self.link_paths],
            "skipped_paths": [str(path) for path in self.skipped_paths],
        }


def repair_slurm_log_links(run_root: Path) -> RunSlurmLogLinkSummary:
    """Ensure existing Slurm-backed runs expose logs/slurm_log.txt."""
    link_paths: list[Path] = []
    skipped_run_ids: list[str] = []
    for run_path in iter_run_paths(run_root):
        manifest = read_run_manifest(run_path)
        run_id = str(manifest.get("run_id") or run_path.name).strip()
        slurm_log_path = _slurm_log_path_from_manifest(
            run_root,
            manifest,
        ) or _slurm_log_path_from_run_id(run_root, run_id)
        if slurm_log_path is None:
            skipped_run_ids.append(run_id)
            continue
        link_paths.append(
            write_relative_symlink(run_path / "logs" / "slurm_log.txt", slurm_log_path)
        )

    return RunSlurmLogLinkSummary(
        run_root=run_root,
        linked_count=len(link_paths),
        skipped_count=len(skipped_run_ids),
        link_paths=link_paths,
        skipped_run_ids=skipped_run_ids,
    )


def repair_result_shortcut_links(run_root: Path) -> RunResultShortcutSummary:
    """Expose existing direct results/<name> entries at the run root."""
    link_paths: list[Path] = []
    skipped_paths: list[Path] = []
    for run_path in iter_run_paths(run_root):
        results_dir = run_path / "results"
        if not results_dir.exists():
            continue
        for result_path in sorted(results_dir.iterdir()):
            link_path = _root_result_shortcut_path(run_path, result_path)
            if link_path is None:
                continue
            if _shortcut_conflicts_with_real_path(link_path):
                skipped_paths.append(link_path)
                continue
            link_paths.append(write_relative_symlink(link_path, result_path))
        for link_path, target_path in _analysis_result_shortcuts(run_path):
            if _shortcut_conflicts_with_real_path(link_path):
                skipped_paths.append(link_path)
                continue
            link_paths.append(write_relative_symlink(link_path, target_path))

    return RunResultShortcutSummary(
        run_root=run_root,
        linked_count=len(link_paths),
        skipped_count=len(skipped_paths),
        link_paths=link_paths,
        skipped_paths=skipped_paths,
    )


def _slurm_log_path_from_run_id(run_root: Path, run_id: str) -> Path | None:
    job_id = slurm_job_id_from_token(run_id)
    if not job_id:
        return None
    return run_root / "slurm_logs" / f"placecell_research_{job_id}.out"


@dataclass(slots=True)
class RunDirectory:
    """Filesystem API for runs/by_id/<run_id>/...."""

    root: Path
    identity: RunIdentity

    @property
    def path(self) -> Path:
        return self.root / "by_id" / self.identity.run_id

    @property
    def manifests_dir(self) -> Path:
        return self.path / "manifests"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def results_dir(self) -> Path:
        return self.path / "results"

    def create(self) -> None:
        for directory in (self.manifests_dir, self.logs_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._link_slurm_log(_slurm_log_path_from_runtime(self.root, self.identity.run_id))

    def write_yaml(self, relative_path: str, payload: Any) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return path

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def write_comparison_card(self, payload: dict[str, Any]) -> Path:
        return self.write_yaml("manifests/comparison_card.yaml", payload)

    def run_manifest_path(self) -> Path:
        return self.manifests_dir / "run_manifest.json"

    def load_run_manifest(self) -> dict[str, Any]:
        path = self.run_manifest_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_run_manifest(self, payload: dict[str, Any]) -> Path:
        path = self.write_json("manifests/run_manifest.json", payload)
        self._link_slurm_log(
            _slurm_log_path_from_manifest(self.root, payload)
            or _slurm_log_path_from_runtime(self.root, self.identity.run_id)
        )
        if manifest_should_be_indexed(payload):
            write_variant_run_link(
                self.root,
                run_id=self.identity.run_id,
                variant_slug=self.identity.variant_slug,
                status=str(payload.get("status") or ""),
            )
        if manifest_should_be_stage_indexed(payload):
            write_stage_run_link(
                self.root,
                run_id=self.identity.run_id,
                stage_name=str(payload.get("stage_name") or ""),
                variant_slug=self.identity.variant_slug,
                status=str(payload.get("status") or ""),
            )
        return path

    def update_run_manifest(self, updates: dict[str, Any]) -> Path:
        return self.write_run_manifest(_merge_payload(self.load_run_manifest(), updates))

    def _link_slurm_log(self, slurm_log_path: Path | None) -> None:
        if slurm_log_path is None:
            return
        self.write_symlink("logs/slurm_log.txt", slurm_log_path)

    def write_symlink(self, relative_path: str, target: Path) -> Path:
        path = self.path / relative_path
        link_path = write_relative_symlink(path, target)
        self._write_root_result_shortcut(relative_path, link_path)
        return link_path

    def _write_root_result_shortcut(self, relative_path: str, result_path: Path) -> None:
        shortcut_path = _root_result_shortcut_path(self.path, self.path / relative_path)
        if shortcut_path is None or _shortcut_conflicts_with_real_path(shortcut_path):
            return
        write_relative_symlink(shortcut_path, result_path)


_ROOT_RESULT_SHORTCUT_RESERVED_NAMES = {
    "logs",
    "manifests",
    "open_me_first",
    "results",
    "wandb",
}


def _root_result_shortcut_path(run_path: Path, result_path: Path) -> Path | None:
    relative_path = result_path.relative_to(run_path)
    if len(relative_path.parts) != 2 or relative_path.parts[0] != "results":
        return None
    result_name = relative_path.parts[1]
    if result_name.startswith(".") or result_name == "README.md":
        return None
    if result_name in _ROOT_RESULT_SHORTCUT_RESERVED_NAMES:
        return None
    return run_path / result_name


def _shortcut_conflicts_with_real_path(shortcut_path: Path) -> bool:
    return shortcut_path.exists() and not shortcut_path.is_symlink()


def _analysis_result_shortcuts(run_path: Path) -> list[tuple[Path, Path]]:
    shortcuts: list[tuple[Path, Path]] = []
    partial_analysis_dir = run_path / "results" / "partial_analysis"
    final_analysis_dir = run_path / "results" / "analysis_report"
    for analysis_dir in (partial_analysis_dir, final_analysis_dir):
        figures_dir = analysis_dir / "figures"
        if figures_dir.exists() or figures_dir.is_symlink():
            shortcuts.append((run_path / "figures", figures_dir))
    unfinished_workspace = partial_analysis_dir / "unfinished_workspace"
    if unfinished_workspace.exists() or unfinished_workspace.is_symlink():
        shortcuts.append((run_path / "staging_workspace", unfinished_workspace))
    return shortcuts
