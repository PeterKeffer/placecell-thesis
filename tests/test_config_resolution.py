from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from placecell_research.config import (
    load_downstream_run_config,
    load_experiment_config,
    summarize_reuse,
)
from placecell_research.config.schema import ExperimentConfig


def test_default_groups_materialize_under_expected_top_level_keys() -> None:
    config = load_experiment_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "wallgap_asym_submit_stream_recipe.yaml"
    )
    assert config.environment.env_id == "MiniWorld-WallGapAsymLarge-v0"
    assert config.dataset.artifact_type == "raw_dataset"
    assert config.spatial_model.encoder.family == "lstm"
    assert config.evaluation.decode_train_fraction == 0.8
    assert config.evaluation.decode_ridge_alpha == 1e-3
    assert config.evaluation.split_names == ["validation", "test"]
    assert "encoder_place_cells" in config.analysis.targets
    assert config.analysis.dataset_coverage_extra_splits == ["train", "validation", "test"]
    assert config.policies.auto_resume_interrupted is True


def test_explicit_reuse_summary_distinguishes_reuse_from_resume() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "wallgap_asym_submit_stream_recipe.yaml"
    )
    reuse_config = load_experiment_config(
        config_path,
        [
            "reuse.vision_encoder_artifact_id=vision_fixture_encoder",
            "reuse.place_model_artifact_id=place_fixture_model",
        ],
    )
    reuse_summary = summarize_reuse(reuse_config)
    assert reuse_summary["vision_encoder"]["stage_behavior"] == "reuse_existing_artifact"
    assert reuse_summary["place_model"]["stage_behavior"] == "reuse_existing_artifact"

    resume_config = load_experiment_config(
        config_path,
        [
            "reuse.place_model_artifact_id=place_fixture_model",
            "policies.training_resume=weights_only",
        ],
    )
    resume_summary = summarize_reuse(resume_config)
    assert resume_summary["place_model"]["stage_behavior"] == "resume_training"


def test_reuse_summary_reports_direct_recovery_checkpoint() -> None:
    config = ExperimentConfig()
    config.policies.training_resume = "weights_and_optimizer"
    config.reuse.place_model_checkpoint_path = "runs/by_id/example/weights_last.pt"

    summary = summarize_reuse(config)

    assert summary["place_model"]["reference_kind"] == "checkpoint_path"
    assert summary["place_model"]["stage_behavior"] == "resume_training"


def test_tracking_output_tags_materialize_from_overrides() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    config = load_experiment_config(
        config_path,
        [
            "tracking.output_tags.vision_encoder=[vision/wallgap_best]",
            "tracking.output_tags.place_model=[place/wallgap_best,place/wallgap_archive]",
        ],
    )
    assert config.tracking.output_tags.vision_encoder == ["vision/wallgap_best"]
    assert config.tracking.output_tags.place_model == [
        "place/wallgap_best",
        "place/wallgap_archive",
    ]


def test_unknown_keys_are_rejected_at_every_level() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    for override in (
        "grid_stream.enabled=true",
        "spatial_model.dag_nodes=[]",
        "analysis.dataset.artifact_id=auto",
        "spatial_model.encoder.xlstm_variant=block_stack",
    ):
        with pytest.raises(ValidationError, match="Unexpected keyword argument"):
            load_experiment_config(config_path, [override])


def test_unknown_downstream_keys_are_rejected() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "downstream"
        / "wallgap_navigation_ppo.yaml"
    )
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        load_downstream_run_config(config_path, ["training.her_goal_representation=goal_xy"])
