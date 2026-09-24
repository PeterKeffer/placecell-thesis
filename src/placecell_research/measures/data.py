"""Locate and read the stored forward pass and reports of one place model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr

from placecell_research.artifacts.registry import ArtifactRegistry, RegisteredArtifact
from placecell_research.evaluation.representation_store import read_representation_manifest

EPISODES_PER_SPLIT = 512
SPLITS = ("train", "validation", "test")
PLACE_CODE_SOURCE = "encoder.place_codes"


@dataclass
class SplitArrays:
    episode_ids: list[int]
    sources: dict[str, np.ndarray]
    position_xy: np.ndarray
    heading: np.ndarray
    valid_steps: np.ndarray
    latent: np.ndarray | None


def _completed(registry: ArtifactRegistry, artifact_type: str) -> list[RegisteredArtifact]:
    artifacts = [a for a in registry.iter_artifacts(artifact_type) if registry.is_completed(a)]
    return sorted(
        artifacts, key=lambda artifact: (artifact.manifest.created_at, artifact.artifact_id)
    )


def find_representation_set(
    registry: ArtifactRegistry, model_id: str, checkpoint_selection: str, explicit_id: str = ""
) -> RegisteredArtifact:
    """The newest completed representation set of the model that holds all three splits."""
    if explicit_id:
        candidates = [registry.resolve_completed_reference("representation_set", explicit_id)]
    else:
        candidates = _completed(registry, "representation_set")
    matches = []
    for artifact in candidates:
        manifest = read_representation_manifest(artifact.path)
        if (
            manifest["place_model_artifact_id"] == model_id
            and set(SPLITS) <= set(manifest["split_names"])
            and manifest["checkpoint_selection"] == checkpoint_selection
        ):
            matches.append(artifact)
    if not matches:
        raise FileNotFoundError(
            f"No completed representation set of {model_id} with splits {list(SPLITS)} and "
            f"checkpoint '{checkpoint_selection}'. Run pc collect-representations first."
        )
    return matches[-1]


def find_analysis_report(registry: ArtifactRegistry, model_id: str) -> RegisteredArtifact:
    """The newest completed analysis report whose inputs include the model."""
    matches = [
        artifact
        for artifact in _completed(registry, "analysis_report")
        if model_id in artifact.manifest.input_artifact_ids
    ]
    if not matches:
        raise FileNotFoundError(
            f"No completed analysis report of {model_id}. Run pc analyze first."
        )
    return matches[-1]


def representation_sources(directory: Path) -> list[str]:
    return list(read_representation_manifest(directory)["sources"])


def fixed_topk(values: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest signed activations per sample, without fitting or rescaling."""
    if not 1 <= k <= values.shape[-1]:
        raise ValueError("k must be between 1 and the representation width.")
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    indices = tensor.topk(k, dim=-1).indices
    return torch.zeros_like(tensor).scatter(-1, indices, tensor.gather(-1, indices)).numpy()


def load_split(
    directory: Path,
    split_name: str,
    source_names: list[str],
    *,
    read_time_top_k: int = 0,
    limit: int = EPISODES_PER_SPLIT,
) -> SplitArrays:
    """First `limit` sorted episodes of one split, as stored by pc collect-representations."""
    manifest = read_representation_manifest(directory)
    episode_ids = list(manifest["episode_ids"][split_name])
    if episode_ids != sorted(episode_ids):
        raise ValueError(
            f"{split_name} episode ids are not sorted; the first-{limit} subset is ambiguous"
        )
    count = min(limit, len(episode_ids))
    group = zarr.open_group(str(directory / "representations.zarr"), mode="r")[
        f"split_{split_name}"
    ]
    metadata = group["metadata"]
    sources = {}
    for name in source_names:
        values = np.asarray(group["sources"][name][:count], dtype=np.float32)
        if read_time_top_k and name == PLACE_CODE_SOURCE:
            values = fixed_topk(values, read_time_top_k)
        sources[name] = values
    return SplitArrays(
        episode_ids=episode_ids[:count],
        sources=sources,
        position_xy=np.asarray(metadata["position_xy"][:count], dtype=np.float32),
        heading=np.asarray(metadata["heading"][:count], dtype=np.float32),
        valid_steps=np.asarray(metadata["valid_steps"][:count], dtype=bool),
        latent=np.asarray(metadata["latent"][:count], dtype=np.float32)
        if "latent" in metadata
        else None,
    )


def analysis_report_metrics(report: RegisteredArtifact) -> dict:
    return json.loads((report.path / "summary.json").read_text())
