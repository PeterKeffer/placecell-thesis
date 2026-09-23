"""Run-directory setup for downstream RL commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placecell_research.artifacts.ids import slugify
from placecell_research.collection.policies import load_raw_config_payload
from placecell_research.config.loader import _load_yaml, _resolve_defaults
from placecell_research.tracking.naming import RunIdentity, capture_git_state, make_run_id
from placecell_research.tracking.run_directory import RunDirectory
from placecell_research.utils.environment_info import capture_environment_info
from placecell_research.utils.repo_paths import find_repo_root
from placecell_research.utils.timing import utc_now_iso


@dataclass(slots=True)
class DownstreamSession:
    repo_root: Path
    config_path: Path
    raw_payload: dict[str, Any]
    run_directory: RunDirectory
    git_state: dict[str, Any]


def initialize_downstream_session(
    *,
    config_path: Path,
    config_name: str,
    tracking,
    stage_name: str,
    overrides: list[str],
) -> DownstreamSession:
    config_path = config_path.resolve()
    repo_root = find_repo_root(config_path)
    _ = _resolve_defaults(config_path, _load_yaml(config_path))
    raw_payload = load_raw_config_payload(config_path, overrides)
    variant_slug = slugify(tracking.variant_name)
    run_identity = RunIdentity(
        run_id=make_run_id(repo_root, descriptor=variant_slug),
        study_name=tracking.study_name,
        variant_name=tracking.variant_name,
        variant_slug=variant_slug,
        signature=f"{stage_name}__{config_name}",
    )
    run_directory = RunDirectory(repo_root / tracking.run_root, run_identity)
    run_directory.create()
    git_state = capture_git_state(repo_root)
    run_directory.write_yaml("manifests/resolved_config.yaml", raw_payload)
    run_directory.write_run_manifest(
        {
            "run_id": run_identity.run_id,
            "stage_name": stage_name,
            "created_at": utc_now_iso(),
            "variant_name": run_identity.variant_name,
            "variant_slug": run_identity.variant_slug,
            "status": "initialized",
            "git_state": git_state,
            "environment_info": capture_environment_info(),
        }
    )
    run_directory.write_comparison_card(
        {
            "variant_name": run_identity.variant_name,
            "variant_slug": run_identity.variant_slug,
            "signature": run_identity.signature,
        }
    )
    return DownstreamSession(
        repo_root=repo_root,
        config_path=config_path,
        raw_payload=raw_payload,
        run_directory=run_directory,
        git_state=git_state,
    )
