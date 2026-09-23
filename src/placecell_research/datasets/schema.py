"""Dataset schema constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RGB_KEY = "observations/rgb"
LATENT_KEY = "observations/latent"
ACTIONS_KEY = "actions/discrete"
CONTINUOUS_ACTIONS_KEY = "actions/continuous"
POSITION_KEY = "state/position_xy"
HEADING_KEY = "state/heading"
KINEMATICS_KEY = "state/kinematics"
VALID_MASK_KEY = "masks/valid_steps"
LENGTH_KEY = "episode_metadata/length"
TERMINATED_KEY = "episode_metadata/terminated"
TRUNCATED_KEY = "episode_metadata/truncated"
SOURCE_SEED_KEY = "episode_metadata/source_seed"

NUMERIC_ARRAY_KEYS = {
    RGB_KEY,
    LATENT_KEY,
    ACTIONS_KEY,
    CONTINUOUS_ACTIONS_KEY,
    POSITION_KEY,
    HEADING_KEY,
    KINEMATICS_KEY,
    VALID_MASK_KEY,
    LENGTH_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    SOURCE_SEED_KEY,
}


def observation_array_key(modality_name: str) -> str:
    normalized_name = str(modality_name).strip()
    if not normalized_name:
        raise ValueError("Observation modality name must be non-empty.")
    return f"observations/{normalized_name}"


@dataclass
class DatasetSummary:
    """Manifest-only dataset metadata."""

    env_id: str
    num_episodes: int
    episode_length: int
    num_actions: int
    schema_version: int = 1
    dataset_dt: str | None = None
    modalities: list[str] | None = None
    episode_environment_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "num_episodes": self.num_episodes,
            "episode_length": self.episode_length,
            "num_actions": self.num_actions,
            "schema_version": self.schema_version,
            "dataset_dt": self.dataset_dt,
            "modalities": self.modalities or [],
            "episode_environment_ids": self.episode_environment_ids,
        }
