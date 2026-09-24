"""Run naming and git metadata."""

from __future__ import annotations

import os
import re
import secrets
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from placecell_research.artifacts.ids import short_fingerprint, slugify
from placecell_research.config.diff import (
    ACTIVE_OBJECTIVES_PATH,
    COMPARISON_FIELDS,
    extract_comparison_values,
)
from placecell_research.utils.timing import utc_now

_RUN_DESCRIPTOR_UNSAFE = re.compile(r"[^a-z0-9_.-]+")


def _git(args: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            check=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def _run_descriptor_slug(descriptor: str | None) -> str:
    raw_descriptor = str(descriptor or "").strip().lower()
    if not raw_descriptor:
        return ""
    return _RUN_DESCRIPTOR_UNSAFE.sub("_", raw_descriptor).strip("_.-")


def _run_unique_suffix() -> str:
    return secrets.token_hex(3)


_SLURM_TOKEN_JOB_ID_RE = re.compile(r"slurm_(\d+)")


def slurm_job_id_from_token(token: str, *, env: Mapping[str, str] | None = None) -> str:
    """Extract a numeric SLURM job id from a slurm_<id>__..."""
    if env is not None:
        for env_name in ("SLURM_JOB_ID", "SLURM_JOBID"):
            job_id = str(env.get(env_name, "")).strip()
            if job_id:
                return job_id
    match = _SLURM_TOKEN_JOB_ID_RE.match(token.strip())
    return match.group(1) if match else ""


def _slurm_job_slug() -> str:
    slurm_job_id = slurm_job_id_from_token("", env=os.environ)
    if slurm_job_id:
        return f"slurm_{slugify(slurm_job_id)}"
    configured_run_id = os.environ.get("PLACECELL_RUN_ID", "").strip().lower()
    configured_prefix = configured_run_id.split("__", 1)[0]
    return configured_prefix if configured_prefix.startswith("slurm_") else ""


def make_run_id(
    repo_root: Path,
    *,
    descriptor: str | None = None,
    include_slurm_job: bool = True,
) -> str:
    """Generate a run id from optional human descriptor, UTC time, and short git hash."""
    timestamp = utc_now().strftime("%Y%m%d_%H%M")
    short_commit = _git(["rev-parse", "--short", "HEAD"], repo_root) or "nogit"
    suffix = f"{timestamp}_{short_commit}_{_run_unique_suffix()}"
    safe_descriptor = _run_descriptor_slug(descriptor)
    run_id = f"{safe_descriptor}__{suffix}" if safe_descriptor else suffix
    if not include_slurm_job:
        return run_id
    slurm_prefix = _slurm_job_slug()
    if slurm_prefix:
        return f"{slurm_prefix}__{run_id}"
    return run_id


def capture_git_state(repo_root: Path) -> dict[str, str | bool]:
    """Collect current git metadata."""
    commit = _git(["rev-parse", "HEAD"], repo_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    dirty = bool(_git(["status", "--porcelain"], repo_root))
    git_state: dict[str, str | bool] = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }
    code_snapshot = os.environ.get("PLACECELL_CODE_SNAPSHOT", "").strip()
    if code_snapshot:
        git_state["code_snapshot"] = code_snapshot
        git_state["code_snapshot_hash"] = Path(code_snapshot).name.rpartition("_")[2]
    return git_state


@dataclass(slots=True)
class RunIdentity:
    """Human and machine identifiers for a run."""

    run_id: str
    study_name: str
    variant_name: str
    variant_slug: str
    signature: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "study_name": self.study_name,
            "variant_name": self.variant_name,
            "variant_slug": self.variant_slug,
            "signature": self.signature,
        }


def generate_signature(resolved_config: dict) -> str:
    """Generate a compact scientific signature from a resolved config."""
    environment_id = resolved_config.get("environment", {}).get("env_id", "unknown")
    spatial_model = resolved_config.get("spatial_model", {})
    encoder = spatial_model.get("encoder", {})
    predictor = spatial_model.get("predictor", {})
    sparsifier = spatial_model.get("sparsifier", {})
    code_blocks = spatial_model.get("code_blocks", [])
    objectives = spatial_model.get("objectives", {})
    seed = resolved_config.get("seed", {}).get("global_seed", 42)

    def temporal_signature(module_config: dict[str, object]) -> str:
        family = slugify(str(module_config.get("family", "unknown")))
        layer_sizes = module_config.get("layer_sizes", [])
        if isinstance(layer_sizes, list) and layer_sizes:
            widths = "x".join(str(int(width)) for width in layer_sizes)
            return f"{family}{widths}"
        return f"{family}na"

    def format_number(value: object) -> str:
        text = f"{float(value):g}" if isinstance(value, (int, float)) else str(value)
        return text.replace(".", "p")

    def block_signature(blocks: object) -> str | None:
        if not isinstance(blocks, list) or not blocks:
            return None
        block_parts: list[str] = []
        for block in sorted(
            (block for block in blocks if isinstance(block, dict)),
            key=lambda item: item.get("start", 0),
        ):
            child = block.get("sparsifier", {})
            child_config = child if isinstance(child, dict) else {}
            child_type = slugify(str(child_config.get("type", "none")))
            child_value = child_config.get("k_fraction", child_config.get("temperature", "na"))
            block_parts.append(
                "_".join(
                    [
                        str(block.get("start", "na")),
                        str(block.get("end", "na")),
                        child_type,
                        format_number(child_value),
                    ]
                )
            )
        return "_".join(["blocks", *block_parts]) if block_parts else None

    def sparsifier_signature(sparsifier_config: dict[str, object]) -> str:
        sparsifier_type = slugify(str(sparsifier_config.get("type", "none")))
        return f"{sparsifier_type}_{sparsifier_config.get('temperature', 'na')}"

    parts = [
        slugify(environment_id),
        f"enc_{temporal_signature(encoder)}",
        f"pred_{temporal_signature(predictor)}",
        block_signature(code_blocks) or sparsifier_signature(sparsifier),
        f"seed{seed}",
        short_fingerprint(*sorted(objectives.keys())) if objectives else "noobj",
    ]
    return "__".join(str(part) for part in parts)


def generate_variant_slug(resolved_config: dict, *, fallback_name: str = "baseline") -> str:
    """Generate a readable variant slug from curated comparison fields."""
    comparison_values = extract_comparison_values(resolved_config)
    parts: list[str] = []
    for field in COMPARISON_FIELDS:
        value = comparison_values.get(field.path)
        if value in (None, "", []):
            continue
        parts.append(f"{field.token}-{slugify(str(value))}")
    objective_names = comparison_values.get(ACTIVE_OBJECTIVES_PATH, [])
    if objective_names:
        parts.append(f"obj-{short_fingerprint(*objective_names)}")
    if parts:
        return "__".join(parts)
    return slugify(fallback_name)
