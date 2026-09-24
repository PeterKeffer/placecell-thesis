from __future__ import annotations

from dataclasses import asdict

from placecell_research.config import artifact_match_fingerprint
from placecell_research.config.schema import DatasetReferenceConfig, ExperimentConfig
from placecell_research.stages.encode_dataset import encoded_dataset_stage_fingerprint
from placecell_research.stages.train_vision_encoder import (
    VISION_ARCHITECTURE_FINGERPRINT,
    vision_encoder_stage_fingerprint,
)


def test_vision_encoder_stage_fingerprint_sorts_dataset_ids() -> None:
    config = ExperimentConfig()
    expected = artifact_match_fingerprint(
        {
            "vision": asdict(config.vision),
            "dataset_artifact_ids": ["dataset_a", "dataset_b"],
            "vision_architecture": VISION_ARCHITECTURE_FINGERPRINT,
            "split_artifact_id": "split_1",
            "multi_dataset_split_semantics": "episode_splits_v1",
            "vision_training_seed": config.seed.global_seed,
        }
    )

    assert (
        vision_encoder_stage_fingerprint(
            config,
            dataset_ids=["dataset_b", "dataset_a"],
            split_artifact_id="split_1",
        )
        == expected
    )


def test_encoded_dataset_stage_fingerprint_ignores_transient_dataset_handoff_fields() -> None:
    reuse_check_config = ExperimentConfig(
        dataset=DatasetReferenceConfig(canonicality_policy="latent_canonical")
    )
    encode_time_config = ExperimentConfig(
        dataset=DatasetReferenceConfig(
            artifact_id="raw_xyz",
            artifact_type="raw_dataset",
            canonicality_policy="latent_canonical",
        )
    )

    fingerprint_kwargs = {
        "source_dataset_artifact_id": "raw_xyz",
        "vision_encoder_artifact_id": "vision_1",
    }
    assert encoded_dataset_stage_fingerprint(
        reuse_check_config, **fingerprint_kwargs
    ) == encoded_dataset_stage_fingerprint(encode_time_config, **fingerprint_kwargs)
