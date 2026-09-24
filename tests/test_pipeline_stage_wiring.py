from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config import (
    artifact_match_fingerprint,
    load_experiment_config,
)
from placecell_research.config.schema import ExperimentConfig, SeedBundleConfig, SplitPolicyConfig
from placecell_research.stages import pipeline
from placecell_research.stages.collect_dataset import raw_dataset_stage_fingerprint
from placecell_research.stages.create_split import split_stage_fingerprint
from placecell_research.stages.encode_dataset import encoded_dataset_stage_fingerprint
from placecell_research.stages.train_vision_encoder import VISION_ARCHITECTURE_FINGERPRINT
from placecell_research.tracking.naming import RunIdentity
from placecell_research.tracking.run_directory import RunDirectory


def _write_artifact(
    artifact_root: Path,
    artifact_type: str,
    artifact_id: str,
    *,
    input_artifact_ids: list[str] | None = None,
    run_id: str | None = None,
) -> Path:
    registry_paths = {
        "raw_dataset": artifact_root / "datasets" / "raw" / artifact_id,
        "encoded_dataset": artifact_root / "datasets" / "encoded" / artifact_id,
        "split_set": artifact_root / "splits" / artifact_id,
        "vision_encoder": artifact_root / "vision_encoders" / artifact_id,
        "place_model": artifact_root / "place_models" / artifact_id,
        "representation_set": artifact_root / "representation_sets" / artifact_id,
        "evaluation_report": artifact_root / "reports" / "evaluation" / artifact_id,
        "analysis_report": artifact_root / "reports" / "analysis" / artifact_id,
    }
    path = registry_paths[artifact_type]
    path.mkdir(parents=True, exist_ok=True)
    ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        created_by=CreatedBy(run_id=run_id or f"{artifact_id}_run", stage_name=artifact_type),
        input_artifact_ids=input_artifact_ids or [],
        config_fingerprint="sha256:test",
        git_commit="test",
    ).write(path / "manifest.json")
    return path


def test_pipeline_injects_stage_outputs_into_later_stage_overrides_and_builds_results_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    captured_overrides: dict[str, list[str]] = {}
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"
    raw_dataset = _write_artifact(
        artifact_root,
        "raw_dataset",
        "raw_fixture_dataset",
        run_id="collect_run",
    )
    _write_artifact(
        artifact_root,
        "split_set",
        "encoded_fixture_split",
        input_artifact_ids=["raw_fixture_dataset"],
        run_id="split_run",
    )
    vision_encoder = _write_artifact(
        artifact_root,
        "vision_encoder",
        "vision_fixture_encoder",
        input_artifact_ids=["raw_fixture_dataset"],
        run_id="vision_run",
    )
    (vision_encoder / "architecture.txt").write_text(
        "VisionEncoder(\n  (encoder): ConvBetaVAE(...)\n)\n"
    )
    _write_artifact(
        artifact_root,
        "encoded_dataset",
        "encoded_fixture_dataset",
        input_artifact_ids=["raw_fixture_dataset", "vision_fixture_encoder"],
        run_id="encode_run",
    )
    place_model = _write_artifact(
        artifact_root,
        "place_model",
        "place_fixture_model",
        input_artifact_ids=[
            "encoded_fixture_dataset",
            "encoded_fixture_split",
            "vision_fixture_encoder",
        ],
        run_id="train_run",
    )
    (place_model / "weights_best_primary.pt").write_text("checkpoint")
    (place_model / "architecture.txt").write_text(
        "CompositePlaceModel(\n  (encoder_stack): EncoderStack(...)\n)\n"
    )
    representation_set = _write_artifact(
        artifact_root, "representation_set", "representation_fixture",
        input_artifact_ids=["place_fixture_model"],
    )
    evaluation_report = _write_artifact(
        artifact_root,
        "evaluation_report",
        "evaluation_fixture_report",
        input_artifact_ids=[
            "place_fixture_model",
            "encoded_fixture_dataset",
            "encoded_fixture_split",
        ],
        run_id="evaluate_run",
    )
    (evaluation_report / "metrics.json").write_text(json.dumps({"rmse": 0.1}))
    (evaluation_report / "metrics.csv").write_text("metric,value\nrmse,0.1\n")
    analysis_report = _write_artifact(
        artifact_root,
        "analysis_report",
        "analysis_fixture_report",
        input_artifact_ids=[
            "place_fixture_model",
            "encoded_fixture_dataset",
            "encoded_fixture_split",
        ],
        run_id="analyze_run",
    )
    (analysis_report / "summary.json").write_text(json.dumps({"decode_rmse": 0.1}))
    (analysis_report / "metrics.csv").write_text("metric,value\ndecode_rmse,0.1\n")
    (analysis_report / "figures").mkdir()
    (analysis_report / "tables").mkdir()
    slurm_log = run_root / "slurm_logs" / "placecell_research_424242.out"
    slurm_log.parent.mkdir(parents=True, exist_ok=True)
    slurm_log.write_text("[slurm] starting pipeline\n")

    for run_id in (
        "collect_run",
        "split_run",
        "vision_run",
        "encode_run",
        "train_run",
        "evaluate_run",
        "analyze_run",
    ):
        run_dir = run_root / "by_id" / run_id / "manifests"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text("{}")
        stage_logs_dir = run_root / "by_id" / run_id / "logs"
        stage_logs_dir.mkdir(parents=True, exist_ok=True)
        (stage_logs_dir / f"{run_id}.log").write_text(f"log for {run_id}\n")
        (run_root / "by_id" / run_id / "results").mkdir(parents=True, exist_ok=True)

    def make_runner(stage_name: str):
        def _runner(_config_path: Path, overrides: list[str]) -> dict[str, object]:
            captured_overrides[stage_name] = list(overrides)
            responses = {
                "collect_dataset": {
                    "dataset.artifact_id": "raw_fixture_dataset",
                    "dataset.artifact_type": "raw_dataset",
                    "stage.run_id": "collect_run",
                    "stage.run_path": str(run_root / "by_id" / "collect_run"),
                },
                "create_split": {
                    "splits.artifact_id": "encoded_fixture_split",
                    "stage.run_id": "split_run",
                    "stage.run_path": str(run_root / "by_id" / "split_run"),
                },
                "train_vision_encoder": {
                    "vision.artifact_id": "vision_fixture_encoder",
                    "stage.run_id": "vision_run",
                    "stage.run_path": str(run_root / "by_id" / "vision_run"),
                },
                "encode_dataset": {
                    "dataset.artifact_id": "encoded_fixture_dataset",
                    "dataset.artifact_type": "encoded_dataset",
                    "stage.run_id": "encode_run",
                    "stage.run_path": str(run_root / "by_id" / "encode_run"),
                },
                "train_place_model": {
                    "place_model_artifact_id": "place_fixture_model",
                    "evaluation.model_artifact_id": "place_fixture_model",
                    "analysis.model_artifact_id": "place_fixture_model",
                    "stage.run_id": "train_run",
                    "stage.run_path": str(run_root / "by_id" / "train_run"),
                },
                "collect_representations": {
                    "reuse.representation_set_artifact_id": "representation_fixture",
                },
                "evaluate_model": {
                    "evaluation_report_id": "evaluation_fixture_report",
                    "stage.run_id": "evaluate_run",
                    "stage.run_path": str(run_root / "by_id" / "evaluate_run"),
                },
                "analyze_model": {
                    "analysis_report_id": "analysis_fixture_report",
                    "analysis_report_path": str(analysis_report),
                    "stage.run_id": "analyze_run",
                    "stage.run_path": str(run_root / "by_id" / "analyze_run"),
                },
            }
            if stage_name == "train_place_model":
                responses[stage_name]["checkpoint_path"] = str(
                    place_model / "weights_best_primary.pt"
                )
            if stage_name == "evaluate_model":
                responses[stage_name]["evaluation_report_path"] = str(evaluation_report)
            return responses.get(stage_name, {})

        return _runner

    monkeypatch.setattr(
        pipeline,
        "_load_stage_runner",
        lambda stage_name: make_runner(stage_name),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("SLURM_JOB_NAME", "placecell_research")
    monkeypatch.setenv("PLACECELL_RUN_ID", "pipeline_probe_run")

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "smoke_wallgap.yaml"
    )
    result = pipeline.run(
        config_path,
        [
            "pipeline.stages=[collect_dataset,create_split,train_vision_encoder,encode_dataset,"
            "train_place_model,collect_representations,evaluate_model,analyze_model]",
            f"tracking.run_root={run_root}",
            f"tracking.artifact_root={artifact_root}",
        ],
    )

    assert "dataset.artifact_id=raw_fixture_dataset" in captured_overrides["create_split"]
    assert "vision.artifact_id=vision_fixture_encoder" in captured_overrides["encode_dataset"]
    assert "dataset.artifact_id=encoded_fixture_dataset" in captured_overrides["train_place_model"]
    assert "splits.artifact_id=encoded_fixture_split" in captured_overrides["train_place_model"]
    assert (
        "evaluation.model_artifact_id=place_fixture_model"
        in captured_overrides["evaluate_model"]
    )
    assert "analysis.model_artifact_id=place_fixture_model" in captured_overrides["analyze_model"]
    for stage in ("evaluate_model", "analyze_model"):
        assert (
            "reuse.representation_set_artifact_id=representation_fixture"
            in captured_overrides[stage]
        )
    assert result["completed_stages"] == [
        "collect_dataset",
        "create_split",
        "train_vision_encoder",
        "encode_dataset",
        "train_place_model",
        "collect_representations",
        "evaluate_model",
        "analyze_model",
    ]
    assert result["pipeline_run_id"] == "pipeline_probe_run"
    assert result["analysis_report_id"] == "analysis_fixture_report"
    pipeline_results = Path(result["pipeline_results_path"])
    assert (pipeline_results / "representation_set").resolve() == representation_set.resolve()
    pipeline_open = Path(result["pipeline_open_path"])
    assert (pipeline_results / "raw_dataset").resolve() == raw_dataset.resolve()
    assert (pipeline_results / "vision_encoder").resolve() == vision_encoder.resolve()
    assert (pipeline_results / "vision_encoder_architecture.txt").resolve() == (
        vision_encoder / "architecture.txt"
    ).resolve()
    assert (pipeline_results / "place_model").resolve() == place_model.resolve()
    assert (pipeline_results / "place_model_architecture.txt").resolve() == (
        place_model / "architecture.txt"
    ).resolve()
    assert (pipeline_results / "analysis_report").resolve() == analysis_report.resolve()
    assert (pipeline_results / "analysis_figures").resolve() == (
        analysis_report / "figures"
    ).resolve()
    assert (pipeline_results / "stages" / "collect_dataset" / "run").resolve() == (
        run_root / "by_id" / "collect_run"
    ).resolve()
    assert (pipeline_results / "stages" / "train_vision_encoder" / "vision_encoder").resolve() == (
        vision_encoder
    ).resolve()
    assert (pipeline_results / "stages" / "analyze_model" / "analysis_report").resolve() == (
        analysis_report
    ).resolve()
    assert (pipeline_results / "best_checkpoint.pt").resolve() == (
        place_model / "weights_best_primary.pt"
    ).resolve()
    assert (pipeline_results / "selected_checkpoint.pt").resolve() == (
        place_model / "weights_best_primary.pt"
    ).resolve()
    assert (
        pipeline_results
        / "stages"
        / "train_place_model"
        / "selected_checkpoint.pt"
    ).resolve() == (place_model / "weights_best_primary.pt").resolve()
    assert (
        pipeline_results
        / "related"
        / "artifacts"
        / "raw_dataset"
        / "raw_fixture_dataset"
    ).exists()
    assert (pipeline_results / "related" / "runs" / "collect_run__raw_dataset").exists()
    assert (pipeline_open / "artifacts" / "model_dataset").resolve() == (
        artifact_root / "datasets" / "encoded" / "encoded_fixture_dataset"
    ).resolve()
    assert (pipeline_open / "artifacts" / "source_raw_dataset").resolve() == raw_dataset.resolve()
    assert (pipeline_open / "artifacts" / "place_model").resolve() == place_model.resolve()
    assert (pipeline_open / "files" / "best_checkpoint.pt").resolve() == (
        place_model / "weights_best_primary.pt"
    ).resolve()
    assert (pipeline_open / "files" / "selected_checkpoint.pt").resolve() == (
        place_model / "weights_best_primary.pt"
    ).resolve()
    assert (pipeline_open / "files" / "analysis_figures").resolve() == (
        analysis_report / "figures"
    ).resolve()
    assert (pipeline_open / "config").resolve() == (
        run_root / "by_id" / result["pipeline_run_id"] / "manifests"
    ).resolve()
    assert (pipeline_open / "logs" / "pipeline").resolve() == (
        run_root / "by_id" / result["pipeline_run_id"] / "logs"
    ).resolve()
    assert (pipeline_open / "logs" / "stages" / "collect_dataset").resolve() == (
        run_root / "by_id" / "collect_run" / "logs"
    ).resolve()
    assert (pipeline_open / "logs" / "slurm_log.txt").resolve() == slurm_log.resolve()
    assert "collect_run.log" in (pipeline_open / "logs" / "all_stage_logs.txt").read_text()
    assert result["slurm_log_path"] == str(slurm_log)
    assert (pipeline_open / "stages").resolve() == (pipeline_results / "stages").resolve()
    assert (pipeline_open / "lineage").resolve() == (pipeline_results / "related").resolve()


def test_last_place_model_checkpoint_is_not_exposed_as_best(tmp_path: Path) -> None:
    run_directory = RunDirectory(
        root=tmp_path / "runs",
        identity=RunIdentity(
            run_id="last_checkpoint_run",
            study_name="test",
            variant_name="test",
            variant_slug="test",
            signature="test",
        ),
    )
    run_directory.create()
    last_checkpoint = tmp_path / "weights_last.pt"
    last_checkpoint.write_text("checkpoint")

    pipeline._link_selected_place_model_checkpoint(
        run_directory,
        last_checkpoint,
        results_directory="results",
    )

    assert (
        run_directory.results_dir / "selected_checkpoint.pt"
    ).resolve() == last_checkpoint.resolve()
    assert not (run_directory.results_dir / "best_checkpoint.pt").is_symlink()


def test_link_related_lineage_skips_self_referential_run_symlink(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"
    _write_artifact(
        artifact_root,
        "raw_dataset",
        "raw_fixture_dataset",
        run_id="self_run",
    )
    run_directory = RunDirectory(
        root=run_root,
        identity=RunIdentity(
            run_id="self_run",
            study_name="test",
            variant_name="test",
            variant_slug="test",
            signature="test",
        ),
    )
    run_directory.create()
    registry = ArtifactRegistry(artifact_root)

    pipeline._link_related_lineage(
        run_directory=run_directory,
        registry=registry,
        run_root=run_root,
        root_artifacts=[registry.load("raw_dataset", "raw_fixture_dataset")],
    )

    assert (
        run_directory.results_dir
        / "related"
        / "artifacts"
        / "raw_dataset"
        / "raw_fixture_dataset"
    ).exists()
    assert not (
        run_directory.results_dir / "related" / "runs" / "self_run__raw_dataset"
    ).exists()


def test_pipeline_does_not_inject_metrics_as_overrides(tmp_path, monkeypatch) -> None:
    captured_overrides: dict[str, list[str]] = {}
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"
    _write_artifact(artifact_root, "raw_dataset", "raw_d", run_id="r1")
    place_model = _write_artifact(artifact_root, "place_model", "pm_1", run_id="r2")
    (place_model / "weights_best_primary.pt").write_text("checkpoint")

    for run_id in ("r1", "r2"):
        run_dir = run_root / "by_id" / run_id / "manifests"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text("{}")

    def make_runner(stage_name: str):
        def _runner(_config_path: Path, overrides: list[str]) -> dict[str, object]:
            captured_overrides[stage_name] = list(overrides)
            if stage_name == "collect_dataset":
                return {
                    "dataset.artifact_id": "raw_d",
                    "dataset.artifact_type": "raw_dataset",
                }
            if stage_name == "train_place_model":
                return {
                    "place_model_artifact_id": "pm_1",
                    "evaluation.model_artifact_id": "pm_1",
                    "analysis.model_artifact_id": "pm_1",
                    "checkpoint_path": str(place_model / "weights_best_primary.pt"),
                    "validation.xy_decode_rmse": 0.042,
                    "validation.total_loss": 1.23,
                }
            return {}

        return _runner

    monkeypatch.setattr(pipeline, "_load_stage_runner", lambda name: make_runner(name))

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "smoke_wallgap.yaml"
    )
    pipeline.run(
        config_path,
        [
            f"tracking.run_root={run_root}",
            f"tracking.artifact_root={artifact_root}",
            "pipeline.stages=[collect_dataset,train_place_model,evaluate_model]",
        ],
    )

    evaluation_overrides = " ".join(captured_overrides.get("evaluate_model", []))
    assert "validation.xy_decode_rmse" not in evaluation_overrides
    assert "validation.total_loss" not in evaluation_overrides
    assert "checkpoint_path" not in evaluation_overrides


def test_pipeline_skips_producer_stages_satisfied_by_explicit_encoded_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    called_stages: list[str] = []
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"
    encoded_dataset = _write_artifact(
        artifact_root,
        "encoded_dataset",
        "encoded_existing",
        input_artifact_ids=["raw_existing", "vision_existing"],
        run_id="encode_existing_run",
    )
    split_set = _write_artifact(
        artifact_root,
        "split_set",
        "split_existing",
        input_artifact_ids=["raw_existing"],
        run_id="split_existing_run",
    )
    place_model = _write_artifact(
        artifact_root,
        "place_model",
        "place_new",
        input_artifact_ids=["encoded_existing", "split_existing"],
        run_id="train_run",
    )
    (place_model / "weights_best_primary.pt").write_text("checkpoint")

    for run_id in ("train_run",):
        run_dir = run_root / "by_id" / run_id / "manifests"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text("{}")

    def make_runner(stage_name: str):
        def _runner(_config_path: Path, _overrides: list[str]) -> dict[str, object]:
            called_stages.append(stage_name)
            if stage_name in {
                "collect_dataset",
                "create_split",
                "train_vision_encoder",
                "encode_dataset",
            }:
                raise AssertionError(f"{stage_name} should be skipped")
            if stage_name == "train_place_model":
                return {
                    "place_model_artifact_id": "place_new",
                    "checkpoint_path": str(place_model / "weights_best_primary.pt"),
                    "stage.run_id": "train_run",
                    "stage.run_path": str(run_root / "by_id" / "train_run"),
                }
            return {}

        return _runner

    monkeypatch.setattr(pipeline, "_load_stage_runner", lambda name: make_runner(name))

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "smoke_wallgap.yaml"
    )
    result = pipeline.run(
        config_path,
        [
            f"tracking.run_root={run_root}",
            f"tracking.artifact_root={artifact_root}",
            "pipeline.stages=[collect_dataset,create_split,train_vision_encoder,encode_dataset,train_place_model]",
            "dataset.artifact_id=encoded_existing",
            "dataset.artifact_type=encoded_dataset",
            "splits.artifact_id=split_existing",
        ],
    )

    assert called_stages == ["train_place_model"]
    assert result["completed_stages"] == ["train_place_model"]
    pipeline_results = Path(result["pipeline_results_path"])
    assert (pipeline_results / "place_model").resolve() == place_model.resolve()
    assert encoded_dataset.exists()
    assert split_set.exists()
    assert (
        pipeline_results / "related" / "artifacts" / "encoded_dataset" / "encoded_existing"
    ).exists()
    assert (pipeline_results / "related" / "artifacts" / "split_set" / "split_existing").exists()


def test_pipeline_auto_reuses_encoded_replacement_for_matching_pruned_raw_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    called_stages: list[str] = []
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "smoke_wallgap.yaml"
    )
    overrides = [
        f"tracking.run_root={run_root}",
        f"tracking.artifact_root={artifact_root}",
        "policies.artifact_reuse=reuse_if_config_match",
        "pipeline.stages=[collect_dataset,create_split,train_vision_encoder,encode_dataset,train_place_model]",
    ]
    config = load_experiment_config(config_path, overrides)
    raw_fingerprint = raw_dataset_stage_fingerprint(config)
    split_fingerprint = split_stage_fingerprint(
        config,
        dataset_artifact_id="raw_existing",
        dataset_artifact_type="raw_dataset",
    )
    vision_fingerprint = artifact_match_fingerprint(
        {
            "vision": asdict(config.vision),
            "dataset_artifact_ids": ["raw_existing"],
            "split_artifact_id": "split_existing",
            "vision_architecture": VISION_ARCHITECTURE_FINGERPRINT,
        }
    )
    encoded_fingerprint = encoded_dataset_stage_fingerprint(
        config,
        source_dataset_artifact_id="raw_existing",
        vision_encoder_artifact_id="vision_existing",
    )
    raw_artifact = _write_artifact(artifact_root, "raw_dataset", "raw_existing")
    ArtifactManifest(
        artifact_id="raw_existing",
        artifact_type="raw_dataset",
        created_by=CreatedBy(run_id="raw_run", stage_name="collect_dataset"),
        config_fingerprint=raw_fingerprint,
        git_commit="test",
        metadata={
            "payload_pruned": True,
            "replaced_by_artifact_id": "encoded_existing",
        },
    ).write(raw_artifact / "manifest.json")
    _write_artifact(
        artifact_root,
        "split_set",
        "split_existing",
        input_artifact_ids=["raw_existing"],
        run_id="split_run",
    )
    ArtifactManifest(
        artifact_id="split_existing",
        artifact_type="split_set",
        created_by=CreatedBy(run_id="split_run", stage_name="create_split"),
        input_artifact_ids=["raw_existing"],
        config_fingerprint=split_fingerprint,
        git_commit="test",
    ).write(artifact_root / "splits" / "split_existing" / "manifest.json")
    _write_artifact(
        artifact_root,
        "vision_encoder",
        "vision_existing",
        input_artifact_ids=["raw_existing", "split_existing"],
        run_id="vision_run",
    )
    ArtifactManifest(
        artifact_id="vision_existing",
        artifact_type="vision_encoder",
        created_by=CreatedBy(run_id="vision_run", stage_name="train_vision_encoder"),
        input_artifact_ids=["raw_existing", "split_existing"],
        config_fingerprint=vision_fingerprint,
        git_commit="test",
    ).write(artifact_root / "vision_encoders" / "vision_existing" / "manifest.json")
    _write_artifact(
        artifact_root,
        "encoded_dataset",
        "encoded_existing",
        input_artifact_ids=["raw_existing", "vision_existing"],
        run_id="encode_run",
    )
    ArtifactManifest(
        artifact_id="encoded_existing",
        artifact_type="encoded_dataset",
        created_by=CreatedBy(run_id="encode_run", stage_name="encode_dataset"),
        input_artifact_ids=["raw_existing", "vision_existing"],
        config_fingerprint=encoded_fingerprint,
        git_commit="test",
    ).write(artifact_root / "datasets" / "encoded" / "encoded_existing" / "manifest.json")
    place_model = _write_artifact(
        artifact_root,
        "place_model",
        "place_new",
        input_artifact_ids=["encoded_existing", "split_existing"],
        run_id="train_run",
    )
    (place_model / "weights_best_primary.pt").write_text("checkpoint")
    run_dir = run_root / "by_id" / "train_run" / "manifests"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text("{}")

    def make_runner(stage_name: str):
        def _runner(_config_path: Path, _overrides: list[str]) -> dict[str, object]:
            called_stages.append(stage_name)
            if stage_name in {
                "collect_dataset",
                "create_split",
                "train_vision_encoder",
                "encode_dataset",
            }:
                raise AssertionError(f"{stage_name} should be skipped")
            if stage_name == "train_place_model":
                return {
                    "place_model_artifact_id": "place_new",
                    "checkpoint_path": str(place_model / "weights_best_primary.pt"),
                    "stage.run_id": "train_run",
                    "stage.run_path": str(run_root / "by_id" / "train_run"),
                }
            return {}

        return _runner

    monkeypatch.setattr(pipeline, "_load_stage_runner", lambda name: make_runner(name))

    result = pipeline.run(config_path, overrides)

    assert called_stages == ["train_place_model"]
    assert result["completed_stages"] == ["train_place_model"]
    assert result["skipped_stages"] == [
        "collect_dataset",
        "create_split",
        "train_vision_encoder",
        "encode_dataset",
    ]


def test_resolve_pinned_dataset_artifact_type_infers_encoded_type_from_registry(
    tmp_path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_artifact(artifact_root, "encoded_dataset", "encoded_pinned")
    registry = ArtifactRegistry(artifact_root)
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_wallgap.yaml"
    )
    overrides = [
        f"tracking.artifact_root={artifact_root}",
        "dataset.artifact_id=encoded_pinned",
    ]

    pipeline._resolve_pinned_dataset_artifact_type(
        config_path=config_path,
        active_overrides=overrides,
        registry=registry,
    )

    assert "dataset.artifact_type=encoded_dataset" in overrides


def test_split_stage_fingerprint_depends_only_on_the_effective_split_seed() -> None:
    def fingerprint(split_seed: int | None, splits_seed: int = 42) -> str:
        config = ExperimentConfig(
            seed=SeedBundleConfig(split_seed=split_seed),
            splits=SplitPolicyConfig(seed=splits_seed),
        )
        return split_stage_fingerprint(
            config, dataset_artifact_id="raw_fixture", dataset_artifact_type="raw_dataset"
        )

    baseline = fingerprint(None)
    assert fingerprint(42) == baseline
    assert fingerprint(42, splits_seed=42) == baseline
    assert fingerprint(7) != baseline
    assert fingerprint(7) == fingerprint(None, splits_seed=7)
