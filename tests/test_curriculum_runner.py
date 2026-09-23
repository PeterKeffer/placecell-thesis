from __future__ import annotations

import json
from pathlib import Path

from placecell_research.config.schema import (
    CurriculumAnalyzeAfter,
    CurriculumConfig,
    CurriculumPhaseConfig,
    CurriculumSourceConfig,
)
from placecell_research.studies.curriculum import CurriculumStageRunners, run_curriculum

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_EXPERIMENT_PATH = REPO_ROOT / "configs/experiment/smoke_museum.yaml"


def test_curriculum_injects_explicit_comparative_inputs() -> None:
    captured_analysis_overrides: list[list[str]] = []
    captured_train_overrides: list[list[str]] = []

    def train_model(_path: Path, overrides: list[str]) -> dict[str, object]:
        captured_train_overrides.append(overrides)
        return {
            "place_model_artifact_id": "place_model_env_b"
            if any("dataset.artifact_id=encoded_env_b" == item for item in overrides)
            else "place_model_env_a",
        }

    def analyze_model(_path: Path, overrides: list[str]) -> dict[str, object]:
        captured_analysis_overrides.append(overrides)
        return {"analysis_report_id": f"report_{len(captured_analysis_overrides)}"}

    def create_split(_path: Path, overrides: list[str]) -> dict[str, object]:
        dataset_override = next(
            item for item in overrides if item.startswith("dataset.artifact_id=")
        )
        dataset_id = dataset_override.split("=", 1)[1]
        return {"splits.artifact_id": f"split_for_{dataset_id}"}

    curriculum = CurriculumConfig(
        name="remap_test",
        base_experiment="smoke_museum",
        phases=[
            CurriculumPhaseConfig(
                name="env_a",
                dataset="encoded_env_a",
                epochs=4,
                resume_policy="fresh",
                analyze_after=CurriculumAnalyzeAfter(
                    datasets=["encoded_env_a"], modules=["decode_xy"]
                ),
            ),
            CurriculumPhaseConfig(
                name="env_b",
                dataset="encoded_env_b",
                epochs=4,
                resume_from="previous",
                resume_policy="weights_only",
                analyze_after=CurriculumAnalyzeAfter(
                    datasets=["encoded_env_a", "encoded_env_b"],
                    modules=["decode_xy"],
                    comparative_modules=["remapping_comparison"],
                ),
            ),
        ],
    )

    run_curriculum(
        curriculum,
        BASE_EXPERIMENT_PATH,
        CurriculumStageRunners(
            train_model=train_model,
            analyze_model=analyze_model,
            create_split=create_split,
        ),
        base_tracking_tags=["local"],
    )

    comparative_overrides = next(
        overrides
        for overrides in captured_analysis_overrides
        if any(
            item.startswith("analysis.comparative.remapping_comparison.inputs=")
            for item in overrides
        )
    )
    assert "analysis.dataset_artifact_type=encoded_dataset" in comparative_overrides
    comparative_inputs_raw = next(
        item.split("=", 1)[1]
        for item in comparative_overrides
        if item.startswith("analysis.comparative.remapping_comparison.inputs=")
    )
    comparative_inputs = json.loads(comparative_inputs_raw)
    assert [item["label"] for item in comparative_inputs] == ["encoded_env_a", "encoded_env_b"]
    assert {item["dataset_artifact_id"] for item in comparative_inputs} == {
        "encoded_env_a",
        "encoded_env_b",
    }
    train_tags_override = next(
        item for item in captured_train_overrides[0] if item.startswith("tracking.tags=")
    )
    assert "study:remap_test" in train_tags_override
    assert "study_mode:curriculum" in train_tags_override
    assert "phase:0_env_a" in train_tags_override
    assert "dataset:encoded_env_a" in train_tags_override
    single_analysis_overrides = next(
        overrides
        for overrides in captured_analysis_overrides
        if "analysis.targets.curriculum_target.enabled=true" in overrides
    )
    assert "analysis.dataset_artifact_type=encoded_dataset" in single_analysis_overrides


def test_curriculum_collects_named_sources_then_reuses_them_for_shared_vision_and_encoding() -> (
    None
):
    captured_collection_overrides: list[list[str]] = []
    captured_vision_overrides: list[list[str]] = []
    captured_encode_overrides: list[list[str]] = []

    def collect_dataset(_path: Path, overrides: list[str]) -> dict[str, object]:
        captured_collection_overrides.append(overrides)
        environment_override = next(
            item for item in overrides if item.startswith("environment.env_id=")
        )
        source_name = (
            "env_a" if environment_override.endswith("MiniWorld-WallGapAsym-v0") else "env_b"
        )
        return {"dataset.artifact_id": f"raw_{source_name}"}

    def train_vision_encoder(_path: Path, overrides: list[str]) -> dict[str, object]:
        captured_vision_overrides.append(overrides)
        return {"vision.artifact_id": "shared_vision_encoder"}

    def encode_dataset(_path: Path, overrides: list[str]) -> dict[str, object]:
        captured_encode_overrides.append(overrides)
        source_override = next(
            item for item in overrides if item.startswith("dataset.artifact_id=")
        )
        source_id = source_override.split("=", 1)[1]
        alias = "encoded_env_a" if source_id == "raw_env_a" else "encoded_env_b"
        return {"dataset.artifact_id": f"{alias}_artifact"}

    def train_model(_path: Path, overrides: list[str]) -> dict[str, object]:
        return {"place_model_artifact_id": "place_model_env_a"}

    def analyze_model(_path: Path, overrides: list[str]) -> dict[str, object]:
        del overrides
        return {"analysis_report_id": "analysis_report"}

    def create_split(_path: Path, overrides: list[str]) -> dict[str, object]:
        dataset_override = next(
            item for item in overrides if item.startswith("dataset.artifact_id=")
        )
        dataset_id = dataset_override.split("=", 1)[1]
        return {"splits.artifact_id": f"split_for_{dataset_id}"}

    curriculum = CurriculumConfig(
        name="shared_ae_sources",
        base_experiment="smoke_museum",
        sources={
            "env_a": CurriculumSourceConfig(
                environment={"env_id": "MiniWorld-WallGapAsym-v0"},
                collection={"episodes": 8},
            ),
            "env_b": CurriculumSourceConfig(
                environment={"env_id": "MiniWorld-WallGapAsymLarge-v0"},
                collection={"episodes": 12},
            ),
        },
        vision_encoder={
            "config": "autoencoder_64",
            "train_on": [
                {"dataset": "env_a", "weight": 1.0},
                {"dataset": "env_b", "weight": 1.0},
            ],
        },
        encoding={
            "encode_each": [
                {"source": "env_a", "alias": "encoded_env_a"},
                {"source": "env_b", "alias": "encoded_env_b"},
            ],
        },
        phases=[
            CurriculumPhaseConfig(
                name="env_a_phase",
                dataset="encoded_env_a",
                epochs=4,
                analyze_after=CurriculumAnalyzeAfter(datasets=["encoded_env_a", "encoded_env_b"]),
            )
        ],
    )

    run_curriculum(
        curriculum,
        BASE_EXPERIMENT_PATH,
        CurriculumStageRunners(
            collect_dataset=collect_dataset,
            train_vision_encoder=train_vision_encoder,
            encode_dataset=encode_dataset,
            train_model=train_model,
            analyze_model=analyze_model,
            create_split=create_split,
        ),
        base_tracking_tags=["local"],
    )

    assert len(captured_collection_overrides) == 2
    assert any("collection.episodes=8" in overrides for overrides in captured_collection_overrides)
    assert any("collection.episodes=12" in overrides for overrides in captured_collection_overrides)

    vision_datasets_override = next(
        item for item in captured_vision_overrides[0] if item.startswith("vision.datasets=")
    )
    assert '"artifact_id": "raw_env_a"' in vision_datasets_override
    assert '"artifact_id": "raw_env_b"' in vision_datasets_override

    assert "dataset.artifact_id=raw_env_a" in captured_encode_overrides[0]
    assert "dataset.artifact_id=raw_env_b" in captured_encode_overrides[1]
    assert "reuse.vision_encoder_artifact_id=shared_vision_encoder" in captured_encode_overrides[0]
    assert "policies.artifact_reuse=reuse_if_config_match" in captured_encode_overrides[0]
    assert "policies.artifact_reuse=reuse_if_config_match" in captured_encode_overrides[1]
    assert not any(
        item.startswith("dataset.output_artifact_id=") for item in captured_encode_overrides[0]
    )
    assert not any(
        item.startswith("dataset.output_artifact_id=") for item in captured_encode_overrides[1]
    )
