"""Human-readable campaign indexes over immutable run and report artifacts."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from placecell_research.tracking.naming import slurm_job_id_from_token
from placecell_research.tracking.run_directory import RunDirectory
from placecell_research.utils.timing import utc_now_iso

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_STATUS_COLUMNS = (
    "group",
    "experiment",
    "arm",
    "seed",
    "job_id",
    "run_id",
    "job_status",
    "registry_status",
    "harvest_status",
    "evidence_status",
    "report_status",
    "analysis_report",
    "evaluation_report",
    "run",
    "slurm_log",
    "updated_at",
)


@dataclass(frozen=True, slots=True)
class CampaignIndexEntry:
    """Explicit identity and targets for one campaign job."""

    campaign: str
    experiment: str
    arm: str
    seed: int
    job_id: str
    run_id: str
    run_path: Path
    group: str = ""
    slurm_log_path: Path | None = None
    analysis_report_path: Path | None = None
    evaluation_report_path: Path | None = None
    report_run_ids: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def leaf_path(self, artifact_root: Path) -> Path:
        parts = [artifact_root, "campaigns", self.campaign]
        if self.group:
            parts.append(self.group)
        parts.extend(
            [
                self.experiment,
                self.arm,
                f"seed_{self.seed}",
                f"job_{self.job_id}",
            ]
        )
        return Path(*parts)


def _validate_component(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not _SAFE_COMPONENT.fullmatch(normalized):
        raise ValueError(
            f"tracking.{field_name} must be one safe path component containing only "
            f"letters, numbers, '.', '_', or '-'; got {value!r}."
        )
    return normalized


def _load_manifest(run_path: Path) -> dict[str, Any]:
    manifest_path = run_path / "manifests" / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _pipeline_run_path(run_directory: RunDirectory, job_id: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for candidate in (run_directory.root / "by_id").glob(f"slurm_{job_id}__*"):
        manifest = _load_manifest(candidate)
        if manifest.get("stage_name") != "pipeline":
            continue
        candidates.append((str(manifest.get("created_at", "")), candidate))
    if not candidates:
        return run_directory.path
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[-1][1]


def _configured_entry(
    raw_config: dict[str, Any],
    run_directory: RunDirectory,
) -> CampaignIndexEntry | None:
    tracking = raw_config.get("tracking", {})
    campaign = str(tracking.get("campaign_name", "") or "").strip()
    experiment = str(tracking.get("experiment_id", "") or "").strip()
    arm = str(tracking.get("experiment_arm", "") or "").strip()
    group = str(tracking.get("campaign_group", "") or "").strip()
    if not any((campaign, experiment, arm, group)):
        return None
    if not all((campaign, experiment, arm)):
        raise ValueError(
            "Campaign indexing requires tracking.campaign_name, tracking.experiment_id, "
            "and tracking.experiment_arm together."
        )
    campaign = _validate_component(campaign, "campaign_name")
    experiment = _validate_component(experiment, "experiment_id")
    arm = _validate_component(arm, "experiment_arm")
    if group:
        group = _validate_component(group, "campaign_group")
    job_id = slurm_job_id_from_token(run_directory.identity.run_id, env=os.environ)
    if not job_id:
        return None
    pipeline_path = _pipeline_run_path(run_directory, job_id)
    pipeline_manifest = _load_manifest(pipeline_path)
    run_id = str(pipeline_manifest.get("run_id") or pipeline_path.name)
    seed_payload = raw_config.get("seed", {})
    seed = int(seed_payload.get("training_seed", seed_payload.get("global_seed", 42)))
    slurm_log_path = pipeline_path / "logs" / "slurm_log.txt"
    if not (slurm_log_path.exists() or slurm_log_path.is_symlink()):
        slurm_log_path = None
    return CampaignIndexEntry(
        campaign=campaign,
        experiment=experiment,
        arm=arm,
        seed=seed,
        job_id=job_id,
        run_id=run_id,
        run_path=pipeline_path,
        group=group,
        slurm_log_path=slurm_log_path,
    )


def _write_symlink(link_path: Path, target_path: Path) -> None:
    target = _preflight_symlink(link_path, target_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        return
    relative_target = Path(os.path.relpath(target, start=link_path.parent.resolve()))
    link_path.symlink_to(relative_target, target_is_directory=target.is_dir())


def _preflight_symlink(link_path: Path, target_path: Path) -> Path:
    target = target_path.resolve(strict=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve(strict=True) == target:
            return target
        raise FileExistsError(
            f"Refusing to replace existing campaign index path: {link_path}"
        )
    return target


def _linked_target(leaf_path: Path, name: str) -> str:
    link_path = leaf_path / name
    if not (link_path.exists() or link_path.is_symlink()):
        return ""
    try:
        return str(link_path.resolve(strict=True))
    except FileNotFoundError:
        return ""


_LOCK_NAME = ".campaign_index.lock"


@contextmanager
def _campaign_lock(campaign_root: Path) -> Iterator[None]:
    """Serialize index writers for one campaign across concurrent jobs."""
    campaign_root.mkdir(parents=True, exist_ok=True)
    with (campaign_root / _LOCK_NAME).open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_text_atomically(path: Path, text: str) -> None:
    """Publish file content by rename, so a concurrent reader never sees a partial file."""
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(text)
    os.replace(temp_path, path)


def _write_entry_metadata(leaf_path: Path, entry: CampaignIndexEntry) -> None:
    metadata_path = leaf_path / "campaign_entry.json"
    identity = _entry_identity(entry)
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    payload = {
        **identity,
        "report_run_ids": {**previous.get("report_run_ids", {}), **entry.report_run_ids},
        "links": {
            name: _linked_target(leaf_path, name)
            for name in ("analysis_report", "evaluation_report", "run", "slurm_log")
        },
        "extra": dict(entry.extra),
        "updated_at": utc_now_iso(),
    }
    _write_text_atomically(metadata_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _entry_identity(entry: CampaignIndexEntry) -> dict[str, object]:
    return {
        "campaign": entry.campaign,
        "group": entry.group,
        "experiment": entry.experiment,
        "arm": entry.arm,
        "seed": entry.seed,
        "job_id": entry.job_id,
        "run_id": entry.run_id,
    }


def _preflight_metadata(leaf_path: Path, entry: CampaignIndexEntry) -> None:
    metadata_path = leaf_path / "campaign_entry.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        for key, value in _entry_identity(entry).items():
            if existing.get(key, value) != value:
                raise ValueError(
                    f"Campaign index identity conflict at {metadata_path}: {key}"
                )


def _tsv_value(value: object) -> str:
    return str(value or "").replace("\t", " ").replace("\n", " ")


def _write_status(campaign_root: Path) -> None:
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(campaign_root.rglob("campaign_entry.json")):
        payload = json.loads(metadata_path.read_text())
        links = payload.get("links", {})
        extra = payload.get("extra", {})
        analysis_report = str(links.get("analysis_report", ""))
        evaluation_report = str(links.get("evaluation_report", ""))
        if analysis_report:
            report_status = "analysis_complete"
        elif evaluation_report:
            report_status = "evaluation_complete"
        else:
            report_status = "run_complete"
        rows.append(
            {
                "group": payload.get("group", ""),
                "experiment": payload.get("experiment", ""),
                "arm": payload.get("arm", ""),
                "seed": payload.get("seed", ""),
                "job_id": payload.get("job_id", ""),
                "run_id": payload.get("run_id", ""),
                "job_status": extra.get("job_status", ""),
                "registry_status": extra.get("registry_status", ""),
                "harvest_status": extra.get("harvest_status", ""),
                "evidence_status": extra.get("evidence_status", ""),
                "report_status": report_status,
                "analysis_report": analysis_report,
                "evaluation_report": evaluation_report,
                "run": links.get("run", ""),
                "slurm_log": links.get("slurm_log", ""),
                "updated_at": payload.get("updated_at", ""),
            }
        )
    lines = ["\t".join(_STATUS_COLUMNS)]
    lines.extend(
        "\t".join(_tsv_value(row[column]) for column in _STATUS_COLUMNS) for row in rows
    )
    _write_text_atomically(campaign_root / "STATUS.tsv", "\n".join(lines) + "\n")


def preflight_campaign_entry(
    artifact_root: Path,
    entry: CampaignIndexEntry,
) -> Path:
    """Validate all targets and conflicts without writing anything."""
    for field_name, value in (
        ("campaign_name", entry.campaign),
        ("experiment_id", entry.experiment),
        ("experiment_arm", entry.arm),
    ):
        _validate_component(value, field_name)
    if entry.group:
        _validate_component(entry.group, "campaign_group")
    if not str(entry.job_id).isdigit():
        raise ValueError(f"Campaign job_id must be numeric, got {entry.job_id!r}.")
    leaf_path = entry.leaf_path(artifact_root)
    targets = {
        "run": entry.run_path,
        "slurm_log": entry.slurm_log_path,
        "analysis_report": entry.analysis_report_path,
        "evaluation_report": entry.evaluation_report_path,
    }
    for link_name, target_path in targets.items():
        if target_path is not None:
            _preflight_symlink(leaf_path / link_name, target_path)
    _preflight_metadata(leaf_path, entry)
    return leaf_path


def materialize_campaign_entry(
    artifact_root: Path,
    entry: CampaignIndexEntry,
) -> Path:
    """Create missing campaign links without replacing any existing path."""
    leaf_path = preflight_campaign_entry(artifact_root, entry)
    leaf_path.mkdir(parents=True, exist_ok=True)
    targets = {
        "run": entry.run_path,
        "slurm_log": entry.slurm_log_path,
        "analysis_report": entry.analysis_report_path,
        "evaluation_report": entry.evaluation_report_path,
    }
    for link_name, target_path in targets.items():
        if target_path is not None:
            _write_symlink(leaf_path / link_name, target_path)
    campaign_root = artifact_root / "campaigns" / entry.campaign
    with _campaign_lock(campaign_root):
        _write_entry_metadata(leaf_path, entry)
        _write_status(campaign_root)
    return leaf_path


def index_published_report(
    *,
    artifact_root: Path,
    raw_config: dict[str, Any],
    run_directory: RunDirectory,
    artifact_type: str,
    report_path: Path,
) -> Path | None:
    """Index one published or reused report when explicit campaign fields are configured."""
    entry = _configured_entry(raw_config, run_directory)
    if entry is None:
        return None
    pipeline_results = entry.run_path / "results"
    analysis_report = None
    evaluation_report = None
    if artifact_type == "analysis_report":
        analysis_report = report_path
        candidate = pipeline_results / "evaluation_report"
        if candidate.exists() or candidate.is_symlink():
            evaluation_report = candidate
    elif artifact_type == "evaluation_report":
        evaluation_report = report_path
        candidate = pipeline_results / "analysis_report"
        if candidate.exists() or candidate.is_symlink():
            analysis_report = candidate
    else:
        raise ValueError(f"Campaign report index does not support {artifact_type!r}.")
    return materialize_campaign_entry(
        artifact_root,
        replace(
            entry,
            report_run_ids={artifact_type: run_directory.identity.run_id},
            analysis_report_path=analysis_report,
            evaluation_report_path=evaluation_report,
        ),
    )
