"""Episode-held-out decoding of true structural rooms in WallGap environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import AnalysisInput, AnalysisResult
from .episode_holdout_decode import decode_labels_episode_holdout
from .spatial_code_dynamics import wallgap_room_ids


@dataclass(slots=True)
class DecodeStructuralRoomModule:
    """Decode named rooms from a code with entire episodes held out."""

    name: str = "decode_structural_room"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        del output_dir
        env_id = str(analysis_input.metadata.get("env_id", ""))
        try:
            room_ids = wallgap_room_ids(analysis_input.position_xy, env_id=env_id)
        except ValueError as error:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={"decode_skipped": True, "decode_skip_reason": str(error)},
            )

        representation = np.asarray(analysis_input.representation)
        valid = np.asarray(analysis_input.valid_mask, dtype=bool) & (room_ids != "outside")
        episodes, time_steps = valid.shape
        random_seed = int(config.get("decode_random_seed", 0))
        train_fraction = float(config.get("decode_train_fraction", 0.8))
        maximum_samples = int(config.get("structural_room_decode_max_samples", 50_000))
        episode_index = np.broadcast_to(np.arange(episodes)[:, None], (episodes, time_steps))
        flat_valid = valid.reshape(-1)
        result, skip_reason = decode_labels_episode_holdout(
            representation.reshape(episodes * time_steps, -1)[flat_valid],
            room_ids.reshape(-1)[flat_valid],
            episode_index.reshape(-1)[flat_valid],
            train_fraction=train_fraction,
            random_seed=random_seed,
            maximum_samples=maximum_samples,
            max_iter=int(config.get("decode_region_max_iter", 200)),
            class_weight="balanced",
            classifier_random_state=random_seed,
        )
        if result is None:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "decode_skipped": True,
                    "decode_skip_reason": skip_reason,
                },
            )

        return AnalysisResult(
            metrics={
                "structural_room_decode_accuracy": result.accuracy,
                "structural_room_decode_balanced_accuracy": result.balanced_accuracy,
                "structural_room_decode_macro_f1": result.macro_f1,
                "structural_room_decode_majority_chance": result.majority_chance,
                "structural_room_decode_class_count": float(len(np.unique(result.test_labels))),
            },
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={
                "train_episode_count": int(len(result.train_episode_ids)),
                "test_episode_count": int(len(result.test_episode_ids)),
                "train_sample_count": result.train_sample_count,
                "test_sample_count": result.test_sample_count,
                "structural_room_names": np.unique(result.train_labels).tolist(),
            },
        )
