"""Filesystem-backed artifact registry."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .manifests import ArtifactManifest

_ARTIFACT_LINEAGE_LABELS: dict[str, dict[str, str]] = {
    "split_set": {
        "raw_dataset": "source_dataset",
        "encoded_dataset": "source_dataset",
    },
    "vision_encoder": {
        "raw_dataset": "trained_on_dataset",
        "encoded_dataset": "trained_on_dataset",
        "split_set": "trained_with_split",
    },
    "encoded_dataset": {
        "raw_dataset": "source_dataset",
        "encoded_dataset": "source_dataset",
        "vision_encoder": "encoded_with_vision_encoder",
    },
    "place_model": {
        "raw_dataset": "trained_on_dataset",
        "encoded_dataset": "trained_on_dataset",
        "split_set": "trained_with_split",
        "vision_encoder": "features_from_vision_encoder",
    },
    "representation_set": {
        "place_model": "collected_from_model",
        "raw_dataset": "collected_on_dataset",
        "encoded_dataset": "collected_on_dataset",
        "split_set": "collected_with_split",
    },
    "evaluation_report": {
        "place_model": "evaluated_model",
        "raw_dataset": "evaluated_on_dataset",
        "encoded_dataset": "evaluated_on_dataset",
        "split_set": "evaluated_with_split",
        "representation_set": "evaluated_representations",
    },
    "analysis_report": {
        "place_model": "analyzed_model",
        "raw_dataset": "analyzed_on_dataset",
        "encoded_dataset": "analyzed_on_dataset",
        "split_set": "analyzed_with_split",
        "representation_set": "analyzed_representations",
    },
}

_STAGE_STATUS_METADATA_KEY = "stage_status"
_STAGE_STATUS_REGISTERED = "registered"
_STAGE_STATUS_COMPLETED = "completed"


@dataclass(slots=True)
class RegisteredArtifact:
    """Resolved artifact location."""

    artifact_id: str
    artifact_type: str
    path: Path
    manifest: ArtifactManifest


class ArtifactRegistry:
    """Stable artifact layout rooted at artifacts/."""

    def __init__(self, root: Path):
        self.root = root
        self._manifest_cache: dict[Path, ArtifactManifest] = {}
        self._load_cache: dict[tuple[str, str], RegisteredArtifact] = {}
        self._find_by_id_cache: dict[str, RegisteredArtifact | None] = {}
        self._iter_cache: dict[str, tuple[RegisteredArtifact, ...]] = {}

    def _clear_caches(self) -> None:
        self._manifest_cache.clear()
        self._load_cache.clear()
        self._find_by_id_cache.clear()
        self._iter_cache.clear()

    @staticmethod
    def _normalize_tag(tag: str) -> str:
        normalized_tag = str(tag or "").strip().strip("/")
        if not normalized_tag:
            raise ValueError("Artifact tag must be a non-empty relative path.")
        tag_path = Path(normalized_tag)
        if tag_path.is_absolute() or any(part in {"", ".", ".."} for part in tag_path.parts):
            raise ValueError(f"Artifact tag must stay within artifacts/by_tag, got {tag!r}.")
        return "/".join(tag_path.parts)

    @staticmethod
    def is_tag_reference(artifact_reference: str) -> bool:
        return str(artifact_reference or "").strip().startswith("tag:")

    def tag_path(self, tag: str) -> Path:
        return self.root / "by_tag" / self._normalize_tag(tag)

    def _read_manifest(self, manifest_path: Path) -> ArtifactManifest:
        cached_manifest = self._manifest_cache.get(manifest_path)
        if cached_manifest is not None:
            return cached_manifest
        manifest = ArtifactManifest.read(manifest_path)
        self._manifest_cache[manifest_path] = manifest
        return manifest

    def _type_directory(self, artifact_type: str) -> Path:
        mapping = {
            "raw_dataset": self.root / "datasets" / "raw",
            "encoded_dataset": self.root / "datasets" / "encoded",
            "split_set": self.root / "splits",
            "vision_encoder": self.root / "vision_encoders",
            "place_model": self.root / "place_models",
            "representation_set": self.root / "representation_sets",
            "evaluation_report": self.root / "reports" / "evaluation",
            "analysis_report": self.root / "reports" / "analysis",
            "study_report": self.root / "reports" / "studies",
        }
        if artifact_type not in mapping:
            raise KeyError(f"Unknown artifact type: {artifact_type}")
        return mapping[artifact_type]

    def staging_root(self) -> Path:
        staging_root = self.root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        return staging_root

    @contextmanager
    def temporary_directory(self, *, prefix: str) -> Iterator[Path]:
        with tempfile.TemporaryDirectory(
            prefix=prefix, dir=str(self.staging_root())
        ) as temporary_dir:
            yield Path(temporary_dir)

    def artifact_path(self, artifact_type: str, artifact_id: str) -> Path:
        return self._type_directory(artifact_type) / artifact_id

    def write_relative_symlink(self, link_path: Path, target_path: Path) -> None:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_target = target_path.resolve()
        if link_path.is_symlink():
            try:
                if link_path.resolve(strict=True) == resolved_target:
                    return
            except FileNotFoundError:
                pass
            link_path.unlink()
        elif link_path.exists():
            return
        relative_target = Path(os.path.relpath(resolved_target, start=link_path.parent.resolve()))
        link_path.symlink_to(relative_target, target_is_directory=resolved_target.is_dir())

    def _materialize_lineage_links(self, artifact: RegisteredArtifact) -> None:
        relationship_labels = _ARTIFACT_LINEAGE_LABELS.get(artifact.artifact_type)
        if not relationship_labels:
            return
        upstream_by_type: dict[str, list[RegisteredArtifact]] = {}
        for input_artifact_id in artifact.manifest.input_artifact_ids:
            upstream_artifact = self.find_by_id(str(input_artifact_id))
            if upstream_artifact is None:
                continue
            upstream_by_type.setdefault(
                upstream_artifact.artifact_type,
                [],
            ).append(upstream_artifact)
        upstream_by_relationship: dict[str, list[RegisteredArtifact]] = {}
        for upstream_artifact_type, relationship_label in relationship_labels.items():
            upstream_by_relationship.setdefault(relationship_label, []).extend(
                upstream_by_type.get(upstream_artifact_type, [])
            )
        for relationship_label, upstream_matches in upstream_by_relationship.items():
            if len(upstream_matches) != 1:
                continue
            upstream_artifact = upstream_matches[0]
            link_name = f"{relationship_label}__{upstream_artifact.artifact_id}"
            self.write_relative_symlink(
                artifact.path / link_name,
                upstream_artifact.path,
            )

    def register_directory(self, artifact_type: str, artifact_id: str, source_dir: Path) -> Path:
        destination = self.artifact_path(artifact_type, artifact_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Artifact already exists: {destination}")
        shutil.move(str(source_dir), str(destination))
        manifest = self._read_manifest(destination / "manifest.json")
        if _STAGE_STATUS_METADATA_KEY not in manifest.metadata:
            manifest.metadata[_STAGE_STATUS_METADATA_KEY] = _STAGE_STATUS_REGISTERED
            manifest.write(destination / "manifest.json")
            self._manifest_cache[destination / "manifest.json"] = manifest
        self._materialize_lineage_links(
            RegisteredArtifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                path=destination,
                manifest=manifest,
            )
        )
        self._clear_caches()
        return destination

    def mark_artifact_completed(self, artifact_type: str, artifact_id: str) -> None:
        artifact = self.load(artifact_type, artifact_id)
        manifest = artifact.manifest
        manifest.metadata[_STAGE_STATUS_METADATA_KEY] = _STAGE_STATUS_COMPLETED
        manifest.write(artifact.path / "manifest.json")
        self._clear_caches()

    @staticmethod
    def is_completed(artifact: RegisteredArtifact) -> bool:
        return (
            artifact.manifest.metadata.get(_STAGE_STATUS_METADATA_KEY, _STAGE_STATUS_COMPLETED)
            == _STAGE_STATUS_COMPLETED
        )

    def require_completed(self, artifact: RegisteredArtifact) -> RegisteredArtifact:
        if not self.is_completed(artifact):
            raise ValueError(
                f"{artifact.artifact_type} artifact {artifact.artifact_id} is not marked "
                "completed; "
                "refusing to reuse an unfinished artifact."
            )
        return artifact

    def resolve_completed_reference(
        self,
        artifact_type: str,
        artifact_reference: str,
    ) -> RegisteredArtifact:
        return self.require_completed(self.resolve_reference(artifact_type, artifact_reference))

    def load(self, artifact_type: str, artifact_id: str) -> RegisteredArtifact:
        cache_key = (artifact_type, artifact_id)
        cached_artifact = self._load_cache.get(cache_key)
        if cached_artifact is not None:
            self._materialize_lineage_links(cached_artifact)
            return cached_artifact
        path = self.artifact_path(artifact_type, artifact_id)
        manifest = self._read_manifest(path / "manifest.json")
        artifact = RegisteredArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=path,
            manifest=manifest,
        )
        self._materialize_lineage_links(artifact)
        self._load_cache[cache_key] = artifact
        self._find_by_id_cache.setdefault(artifact_id, artifact)
        return artifact

    def find_by_id(self, artifact_id: str) -> RegisteredArtifact | None:
        if artifact_id in self._find_by_id_cache:
            return self._find_by_id_cache[artifact_id]
        for artifact_type in (
            "raw_dataset",
            "encoded_dataset",
            "split_set",
            "vision_encoder",
            "place_model",
            "representation_set",
            "evaluation_report",
            "analysis_report",
            "study_report",
        ):
            path = self.artifact_path(artifact_type, artifact_id)
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                artifact = self.load(artifact_type, artifact_id)
                self._find_by_id_cache[artifact_id] = artifact
                return artifact
        self._find_by_id_cache[artifact_id] = None
        return None

    def iter_artifacts(self, artifact_type: str) -> Iterator[RegisteredArtifact]:
        cached_artifacts = self._iter_cache.get(artifact_type)
        if cached_artifacts is not None:
            return iter(cached_artifacts)
        artifact_root = self._type_directory(artifact_type)
        if not artifact_root.exists():
            self._iter_cache[artifact_type] = ()
            return iter(())
        artifacts: list[RegisteredArtifact] = []
        for artifact_path in sorted(artifact_root.iterdir()):
            manifest_path = artifact_path / "manifest.json"
            if not artifact_path.is_dir() or not manifest_path.exists():
                continue
            manifest = self._read_manifest(manifest_path)
            artifact = RegisteredArtifact(
                artifact_id=manifest.artifact_id,
                artifact_type=artifact_type,
                path=artifact_path,
                manifest=manifest,
            )
            artifacts.append(artifact)
            self._load_cache[(artifact_type, manifest.artifact_id)] = artifact
            self._find_by_id_cache.setdefault(manifest.artifact_id, artifact)
        cached_tuple = tuple(artifacts)
        self._iter_cache[artifact_type] = cached_tuple
        return iter(cached_tuple)

    def find_matching(
        self,
        artifact_type: str,
        config_fingerprint: str,
        input_artifact_ids: list[str],
        *,
        include_pruned: bool = False,
    ) -> RegisteredArtifact | None:
        normalized_inputs = sorted(
            str(artifact_id) for artifact_id in input_artifact_ids if artifact_id
        )
        matches = [
            artifact
            for artifact in self.iter_artifacts(artifact_type)
            if artifact.manifest.config_fingerprint == config_fingerprint
            and self.is_completed(artifact)
            and (
                include_pruned or not bool(artifact.manifest.metadata.get("payload_pruned", False))
            )
            and sorted(
                str(artifact_id)
                for artifact_id in artifact.manifest.input_artifact_ids
                if artifact_id
            )
            == normalized_inputs
        ]
        if not matches:
            return None
        matches.sort(key=lambda artifact: (artifact.manifest.created_at, artifact.artifact_id))
        return matches[-1]

    def write_tag(self, tag: str, artifact_path: Path) -> Path:
        tag_path = self.tag_path(tag)
        tag_path.parent.mkdir(parents=True, exist_ok=True)
        staged_tag_path = tag_path.with_name(f".{tag_path.name}.tmp-{os.getpid()}")
        staged_tag_path.unlink(missing_ok=True)
        staged_tag_path.symlink_to(artifact_path.resolve())
        try:
            os.replace(staged_tag_path, tag_path)
        except OSError:
            staged_tag_path.unlink(missing_ok=True)
            raise
        return tag_path

    def resolve_tag(self, tag: str, *, expected_type: str | None = None) -> RegisteredArtifact:
        normalized_tag = self._normalize_tag(tag)
        tag_path = self.tag_path(normalized_tag)
        if not tag_path.exists() and not tag_path.is_symlink():
            raise FileNotFoundError(f"Artifact tag not found: {normalized_tag}")
        try:
            resolved_path = tag_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Artifact tag '{normalized_tag}' points to a missing artifact directory."
            ) from exc
        manifest_path = resolved_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Artifact tag '{normalized_tag}' points to {resolved_path}, "
                "but manifest.json is missing."
            )
        manifest = self._read_manifest(manifest_path)
        artifact = self.load(manifest.artifact_type, manifest.artifact_id)
        if expected_type is not None and artifact.artifact_type != expected_type:
            raise ValueError(
                f"Artifact tag '{normalized_tag}' resolved to {artifact.artifact_type} "
                f"({artifact.artifact_id}), expected {expected_type}."
            )
        return artifact

    def resolve_reference(self, artifact_type: str, artifact_reference: str) -> RegisteredArtifact:
        normalized_reference = str(artifact_reference or "").strip()
        if not normalized_reference:
            raise ValueError("Artifact reference must be non-empty.")
        if self.is_tag_reference(normalized_reference):
            return self.resolve_tag(
                normalized_reference.removeprefix("tag:"),
                expected_type=artifact_type,
            )
        return self.load(artifact_type, normalized_reference)

    def describe(self, artifact_type: str, artifact_id: str) -> str:
        registered = self.load(artifact_type, artifact_id)
        return json.dumps(
            {
                "artifact_id": registered.artifact_id,
                "artifact_type": registered.artifact_type,
                "path": str(registered.path),
            },
            indent=2,
        )
