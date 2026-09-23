from __future__ import annotations

from pathlib import Path

import pytest

from placecell_research.config import (
    load_experiment_config,
    load_study_config,
    validate_experiment_config,
    validate_study_config,
)
from placecell_research.config.schema import CurriculumConfig, StudyConfig
from placecell_research.utils.repo_paths import resolve_experiment_config_path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIGS = REPO_ROOT / "configs" / "study"


def test_experiment_reference_resolution_is_shared_across_study_entrypoints() -> None:
    study_config_path = STUDY_CONFIGS / "example_grid.yaml"
    expected = (REPO_ROOT / "configs" / "experiment" / "smoke_museum.yaml").resolve()

    references = (
        str(expected),
        "configs/experiment/smoke_museum.yaml",
        "../experiment/smoke_museum.yaml",
        "smoke_museum",
        "smoke_museum.yaml",
    )

    assert {
        resolve_experiment_config_path(study_config_path, reference) for reference in references
    } == {expected}


@pytest.mark.parametrize("name", ["example_grid", "example_curriculum"])
def test_example_study_config_and_its_base_experiment_validate(name: str) -> None:
    path = STUDY_CONFIGS / f"{name}.yaml"
    study = load_study_config(path)
    assert validate_study_config(study) == []
    block = study.sweep if study.sweep is not None else study.curriculum
    base_path = resolve_experiment_config_path(path, block.base_experiment)
    validate_experiment_config(load_experiment_config(base_path))


def test_curriculum_rejects_encoding_keys_it_does_not_run() -> None:
    study = StudyConfig(
        curriculum=CurriculumConfig(
            base_experiment="smoke_museum",
            encoding={"pool": {"alias": "joint", "sources": ["a", "b"]}},
        )
    )
    with pytest.raises(ValueError, match="encode_each"):
        validate_study_config(study)
