"""Artifact manifests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from placecell_research.utils.timing import utc_now_iso


@dataclass(slots=True)
class CreatedBy:
    """Manifest provenance."""

    run_id: str
    stage_name: str


@dataclass(slots=True)
class ArtifactManifest:
    """Common manifest payload for all first-class artifacts."""

    artifact_id: str
    artifact_type: str
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    created_by: CreatedBy | None = None
    input_artifact_ids: list[str] = field(default_factory=list)
    config_fingerprint: str = ""
    git_commit: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance_amendments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.created_by is None:
            data["created_by"] = None
        return data

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> ArtifactManifest:
        payload = json.loads(path.read_text())
        created_by_raw = payload.get("created_by")
        payload["created_by"] = CreatedBy(**created_by_raw) if created_by_raw else None
        return cls(**payload)
