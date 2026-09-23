from __future__ import annotations

import sys
import time
import types
import warnings
from pathlib import Path

from placecell_research.stages import run_study
from placecell_research.studies.sweep import SweepRunResult
from placecell_research.tracking import wandb_logger
from placecell_research.tracking.naming import RunIdentity
from placecell_research.tracking.tags import (
    _MAX_TAG_LENGTH,
    curriculum_tags,
    default_wandb_group,
    stage_tags,
    sweep_tags,
)
from placecell_research.tracking.wandb_logger import (
    WandbLogger,
    build_wandb_run_config,
    flatten_wandb_config,
)


def test_flatten_wandb_config_builds_compare_friendly_dotted_keys() -> None:
    flattened = flatten_wandb_config(
        {
            "environment": {"env_id": "MiniWorld-WallGapAsym-v0"},
            "tracking": {"tags": ["local", "debug"]},
            "path": Path("artifacts"),
        }
    )
    assert flattened["environment.env_id"] == "MiniWorld-WallGapAsym-v0"
    assert flattened["tracking.tags"] == ["local", "debug"]
    assert flattened["path"] == "artifacts"

    config = build_wandb_run_config(
        {"seed": {"global_seed": 7}},
        identity=RunIdentity(
            run_id="20260307_test",
            study_name="study_a",
            variant_name="baseline",
            variant_slug="env-wallgap__enc-gru__seed7",
            signature="sig_a",
        ),
        stage_name="train_place_model",
        git_state={"commit": "abc123", "branch": "main", "dirty": False},
        salient_diff={
            "spatial_model.sparsifier.temperature": {
                "base": 1.0,
                "current": 0.3,
            }
        },
    )
    assert config["seed.global_seed"] == 7
    assert config["runtime.run_id"] == "20260307_test"
    assert config["runtime.stage_name"] == "train_place_model"
    assert config["runtime.variant_slug"] == "env-wallgap__enc-gru__seed7"
    assert config["runtime.changed_salient_fields"] == ["spatial_model.sparsifier.temperature"]


def test_wandb_logger_disables_itself_after_runtime_failure(monkeypatch) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(update=lambda payload, allow_val_change=False: None)
            self.summary = {}

        def log(self, payload, step=None) -> None:
            del payload, step
            raise RuntimeError("network dropped")

        def save(self, path, base_path=None, policy=None) -> None:
            del path, base_path, policy

        def finish(self) -> None:
            return

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            del kwargs
            return FakeRun()

        @staticmethod
        def define_metric(name, step_metric=None) -> None:
            del name, step_metric

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    logger = WandbLogger(
        enabled=True,
        project="placecells-model",
        name="wandb_runtime_failure",
        group="debug",
        job_type="train_place_model",
        tags=["test"],
        config={"seed.global_seed": 7},
        mode="offline",
    )
    logger.start()
    assert logger.run is not None

    logger.log({"train/loss": 1.0}, step=0)

    assert logger.run is None
    assert logger.enabled is False
    assert logger.last_error is not None
    assert "network dropped" in logger.last_error
    logger.finish()


def test_wandb_logger_emits_loud_warning(monkeypatch, capsys) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(update=lambda payload, allow_val_change=False: None)
            self.summary = {}

        def log(self, payload, step=None) -> None:
            del payload, step
            raise RuntimeError("api offline")

        def save(self, path, base_path=None, policy=None) -> None:
            del path, base_path, policy

        def finish(self) -> None:
            return

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            del kwargs
            return FakeRun()

        @staticmethod
        def define_metric(name, step_metric=None) -> None:
            del name, step_metric

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    logger = WandbLogger(
        enabled=True,
        project="placecells-model",
        name="wandb_loud_warning",
        group="debug",
        job_type="train_place_model",
        tags=["test"],
        config={"seed.global_seed": 7},
        mode="offline",
    )
    logger.start()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logger.log({"train/loss": 1.0}, step=0)
    stderr = capsys.readouterr().err
    assert "W&B FAILURE" in stderr
    assert any("W&B FAILURE" in str(item.message) for item in caught)


def test_wandb_logger_finish_has_bounded_timeout(monkeypatch, capsys) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(update=lambda payload, allow_val_change=False: None)
            self.summary = {}

        def log(self, payload, step=None) -> None:
            del payload, step

        def save(self, path, base_path=None, policy=None) -> None:
            del path, base_path, policy

        def finish(self) -> None:
            time.sleep(1.0)

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            del kwargs
            return FakeRun()

        @staticmethod
        def define_metric(name, step_metric=None) -> None:
            del name, step_metric

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    monkeypatch.setattr(wandb_logger, "_WANDB_FINISH_TIMEOUT_SECONDS", 0.01, raising=False)
    logger = WandbLogger(
        enabled=True,
        project="placecells-model",
        name="wandb_finish_timeout",
        group="debug",
        job_type="train_place_model",
        tags=["test"],
        config={"seed.global_seed": 7},
        mode="offline",
    )
    logger.start()

    started_at = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logger.finish()
    elapsed = time.perf_counter() - started_at

    stderr = capsys.readouterr().err
    assert elapsed < 0.5
    assert logger.run is None
    assert logger.enabled is False
    assert logger.last_error is not None
    assert "timed out" in logger.last_error
    assert "W&B FAILURE" in stderr
    assert any("timed out" in str(item.message) for item in caught)


def test_structured_wandb_tags_are_compare_friendly() -> None:
    assert stage_tags(
        "train_place_model",
        "MiniWorld-WallGapAsym-v0",
        base_tags=["local"],
        study_name="wallgap",
        variant_slug="wallgap__gru",
        seed=7,
    ) == [
        "local",
        "stage:train_place_model",
        "env:MiniWorld-WallGapAsym-v0",
        "study:wallgap",
        "variant_slug:wallgap__gru",
        "seed:7",
    ]
    assert sweep_tags("sparsifier_search", 3, 42, base_tags=["local"]) == [
        "local",
        "study:sparsifier_search",
        "study_mode:sweep",
        "trial:3",
        "seed:42",
    ]
    assert curriculum_tags("remap_seq", "forest env", 1, "forest_64x64", base_tags=["local"]) == [
        "local",
        "study:remap_seq",
        "study_mode:curriculum",
        "phase:1_forest_env",
        "dataset:forest_64x64",
    ]


def test_long_variant_slug_tags_are_shortened_to_wandb_limit() -> None:
    long_variant_slug = (
        "env-miniworld-wallgapasymlarge-v0__vision-ae64-prediction-only__"
        "place-lstm-gru-kwinners512__seed-42__obj-4c89f8__episodes-1024__steps-1024"
    )

    tags = stage_tags(
        "train_vision_encoder",
        "MiniWorld-WallGapAsymLarge-v0",
        variant_slug=long_variant_slug,
    )
    variant_slug_tag = next(tag for tag in tags if tag.startswith("variant_slug:"))

    assert len(variant_slug_tag) <= _MAX_TAG_LENGTH
    assert variant_slug_tag != f"variant_slug:{long_variant_slug}"
    assert variant_slug_tag.startswith("variant_slug:")
    assert variant_slug_tag.count("_") >= 1


def test_long_default_wandb_group_is_shortened_to_wandb_limit() -> None:
    long_variant_slug = (
        "env-miniworld-wallgapasymlarge-v0__vision-ae64-prediction-only__"
        "place-lstm-gru-kwinners512__seed-42__obj-4c89f8__episodes-1024__steps-1024"
    )

    group = default_wandb_group(None, long_variant_slug)

    assert len(group) <= _MAX_TAG_LENGTH
    assert group.startswith("variant:")


def test_run_study_propagates_wandb_and_study_name_to_child_runs(tmp_path, monkeypatch) -> None:
    captured_overrides: list[list[str]] = []

    class DummyLogger:
        def __init__(self, **kwargs):
            del kwargs

        def start(self) -> None:
            return

        def upload_files(self, paths) -> None:
            del paths

        def log(self, payload) -> None:
            del payload

        def update_config(self, payload) -> None:
            del payload

        def set_summary(self, payload) -> None:
            del payload

        def finish(self) -> None:
            return

    def fake_train_place_model(
        config_path: Path,
        overrides: list[str],
        **kwargs,
    ) -> dict[str, object]:
        del config_path
        captured_overrides.append(list(overrides))
        run_path = tmp_path / "runs" / "by_id" / "fake_train_run"
        run_path.mkdir(parents=True)
        on_runtime_created = kwargs.get("on_runtime_created")
        if on_runtime_created is not None:
            on_runtime_created(
                types.SimpleNamespace(
                    run_directory=types.SimpleNamespace(
                        identity=types.SimpleNamespace(run_id="fake_train_run"),
                        path=run_path,
                    )
                )
            )
        assert (
            tmp_path
            / "runs"
            / "studies"
            / "example_grid"
            / "k_fraction_sparsifier-0_05"
            / "artifact_reuse_policies-reuse_if_config_match"
            / "seed-0"
        ).is_symlink()
        return {
            "place_model_artifact_id": "place_model_fake",
            "validation.xy_decode_rmse": 0.123,
        }

    def fake_run_sweep(sweep_config, experiment_path: Path, runner, **kwargs):
        del sweep_config
        base_tracking_tags = kwargs["base_tracking_tags"]
        study_name = kwargs["study_name"]
        tags = sweep_tags(study_name, 0, 0, base_tags=base_tracking_tags)
        runner(
            experiment_path,
            [f"tracking.tags={tags}", "seed.global_seed=0"],
            {
                "seed": 0,
                "trial_index": 0,
                "spatial_model.sparsifier.k_fraction": 0.05,
                "policies.artifact_reuse": "reuse_if_config_match",
            },
        )
        return SweepRunResult(
            rows=[
                {
                    "place_model_artifact_id": "place_model_fake",
                    "validation.xy_decode_rmse": 0.123,
                }
            ]
        )

    monkeypatch.setattr(run_study, "WandbLogger", DummyLogger)
    monkeypatch.setattr(run_study.train_place_model, "run", fake_train_place_model)
    monkeypatch.setattr(run_study, "run_sweep", fake_run_sweep)

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "study"
        / "example_grid.yaml"
    )
    result = run_study.run(
        config_path,
        [
            f"tracking.run_root={tmp_path / 'runs'}",
            f"tracking.artifact_root={tmp_path / 'artifacts'}",
            "tracking.use_wandb=true",
            "tracking.wandb_mode=offline",
        ],
    )

    assert result["study_report_id"]
    assert captured_overrides
    assert "tracking.use_wandb=true" in captured_overrides[0]
    assert "tracking.wandb_mode=offline" in captured_overrides[0]
    assert "tracking.wandb_failure_mode=warn" in captured_overrides[0]
    assert "tracking.wandb_project=" in captured_overrides[0]
    assert "tracking.study_name=example_grid" in captured_overrides[0]
    assert f"tracking.run_root={tmp_path / 'runs'}" in captured_overrides[0]
    assert f"tracking.artifact_root={tmp_path / 'artifacts'}" in captured_overrides[0]
    tags_override = next(
        override
        for override in captured_overrides[0]
        if override.startswith("tracking.tags=")
    )
    assert "study:example_grid" in tags_override
    assert "study_mode:sweep" in tags_override
    assert "trial:0" in tags_override
    assert "seed:0" in tags_override
