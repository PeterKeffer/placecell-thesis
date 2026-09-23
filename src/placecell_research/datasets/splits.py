"""Reusable split artifacts."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy


def _sorted_unique_indices(values: Sequence[int]) -> list[int]:
    result = sorted({int(value) for value in values})
    for value in result:
        if value < 0:
            raise ValueError("Split indices must be non-negative integers.")
    return result


@dataclass
class SplitIndices:
    """Explicit split lists."""

    split_id: str
    dataset_artifact_id: str
    strategy: str
    seed: int
    train_episode_ids: list[int]
    validation_episode_ids: list[int]
    test_episode_ids: list[int]
    constraints: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "dataset_artifact_id": self.dataset_artifact_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "train_episode_ids": self.train_episode_ids,
            "validation_episode_ids": self.validation_episode_ids,
            "test_episode_ids": self.test_episode_ids,
            "constraints": self.constraints,
        }


def create_split_indices(
    split_id: str,
    dataset_artifact_id: str,
    strategy: str,
    seed: int,
    num_episodes: int,
    constraints: dict[str, object],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    manual_ids: dict[str, Sequence[int]] | None = None,
) -> SplitIndices:
    """Create explicit sorted split indices."""
    all_episode_ids = list(range(int(num_episodes)))
    if strategy == "manual_ids":
        if manual_ids is None:
            raise ValueError("manual_ids strategy requires explicit split lists.")
        train_ids = _sorted_unique_indices(manual_ids.get("train", []))
        validation_ids = _sorted_unique_indices(manual_ids.get("validation", []))
        test_ids = _sorted_unique_indices(manual_ids.get("test", []))
    else:
        if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0.")
        if strategy == "chronological":
            shuffled = all_episode_ids
        elif strategy in {"episode_random", "episode_random_stratified_by_environment"}:
            shuffled = list(all_episode_ids)
            random.Random(seed).shuffle(shuffled)
        else:
            raise ValueError(f"Unsupported split strategy: {strategy}")
        train_end = int(round(len(shuffled) * train_fraction))
        validation_end = train_end + int(round(len(shuffled) * validation_fraction))
        train_ids = _sorted_unique_indices(shuffled[:train_end])
        validation_ids = _sorted_unique_indices(shuffled[train_end:validation_end])
        test_ids = _sorted_unique_indices(shuffled[validation_end:])
    overlap = (
        set(train_ids) & set(validation_ids)
        | set(train_ids) & set(test_ids)
        | set(validation_ids) & set(test_ids)
    )
    if overlap:
        raise ValueError(f"Split lists must be disjoint. Overlap: {sorted(overlap)}")
    return SplitIndices(
        split_id=split_id,
        dataset_artifact_id=dataset_artifact_id,
        strategy=strategy,
        seed=seed,
        train_episode_ids=train_ids,
        validation_episode_ids=validation_ids,
        test_episode_ids=test_ids,
        constraints=constraints,
    )


def build_split_artifact(
    output_dir: Path,
    split_indices: SplitIndices,
    run_id: str,
    config_fingerprint: str,
    git_commit: str,
) -> ArtifactManifest:
    """Write split artifact files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_indices.json").write_text(
        json.dumps(split_indices.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    manifest = ArtifactManifest(
        artifact_id=split_indices.split_id,
        artifact_type="split_set",
        created_by=CreatedBy(run_id=run_id, stage_name="create_split"),
        input_artifact_ids=[split_indices.dataset_artifact_id],
        config_fingerprint=config_fingerprint,
        git_commit=git_commit,
        summary={
            "strategy": split_indices.strategy,
            "train_count": len(split_indices.train_episode_ids),
            "validation_count": len(split_indices.validation_episode_ids),
            "test_count": len(split_indices.test_episode_ids),
        },
    )
    manifest.write(output_dir / "manifest.json")
    return manifest
