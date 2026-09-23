from __future__ import annotations

from pathlib import Path

import torch

from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.reuse import resolve_artifact_reference_id
from placecell_research.config.schema import SweepConfig, VisionConfig
from placecell_research.downstream.frozen_extractor import FrozenRepresentationExtractor
from placecell_research.studies import sweep as sweep_module
from placecell_research.vision.builder import build_vision_model


def _write_artifact(
    artifact_root: Path,
    artifact_type: str,
    artifact_id: str,
    *,
    input_artifact_ids: list[str] | None = None,
    config_fingerprint: str = "sha256:test",
    metadata: dict[str, object] | None = None,
) -> Path:
    registry_paths = {
        "raw_dataset": artifact_root / "datasets" / "raw" / artifact_id,
        "encoded_dataset": artifact_root / "datasets" / "encoded" / artifact_id,
        "split_set": artifact_root / "splits" / artifact_id,
        "vision_encoder": artifact_root / "vision_encoders" / artifact_id,
        "place_model": artifact_root / "place_models" / artifact_id,
        "evaluation_report": artifact_root / "reports" / "evaluation" / artifact_id,
        "analysis_report": artifact_root / "reports" / "analysis" / artifact_id,
        "study_report": artifact_root / "reports" / "studies" / artifact_id,
    }
    path = registry_paths[artifact_type]
    path.mkdir(parents=True, exist_ok=True)
    ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        created_by=CreatedBy(run_id=f"{artifact_id}_run", stage_name=artifact_type),
        input_artifact_ids=input_artifact_ids or [],
        config_fingerprint=config_fingerprint,
        git_commit="test",
        metadata=metadata or {},
    ).write(path / "manifest.json")
    return path


def test_registry_find_matching_ignores_unfinished_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_artifact(
        artifact_root,
        "vision_encoder",
        "vision_registered",
        input_artifact_ids=["raw_fixture"],
        config_fingerprint="sha256:vision",
        metadata={"stage_status": "registered"},
    )
    registry = ArtifactRegistry(artifact_root)

    assert registry.find_matching("vision_encoder", "sha256:vision", ["raw_fixture"]) is None


def test_explicit_artifact_reuse_rejects_unfinished_artifact(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_artifact(
        artifact_root,
        "vision_encoder",
        "vision_registered",
        metadata={"stage_status": "registered"},
    )
    registry = ArtifactRegistry(artifact_root)

    try:
        resolve_artifact_reference_id(registry, "vision_encoder", "vision_registered")
    except ValueError as exc:
        assert "not marked completed" in str(exc)
    else:
        raise AssertionError("Explicit reuse should reject unfinished artifacts.")


def test_registry_find_matching_prefers_completed_artifact_over_newer_unfinished(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_artifact(
        artifact_root,
        "vision_encoder",
        "vision_completed",
        input_artifact_ids=["raw_fixture"],
        config_fingerprint="sha256:vision",
        metadata={"stage_status": "completed"},
    )
    _write_artifact(
        artifact_root,
        "vision_encoder",
        "vision_registered",
        input_artifact_ids=["raw_fixture"],
        config_fingerprint="sha256:vision",
        metadata={"stage_status": "registered"},
    )
    registry = ArtifactRegistry(artifact_root)

    match = registry.find_matching("vision_encoder", "sha256:vision", ["raw_fixture"])

    assert match is not None
    assert match.artifact_id == "vision_completed"


def test_registry_caches_find_by_id_and_matching(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_artifact(artifact_root, "raw_dataset", "raw_fixture")
    _write_artifact(
        artifact_root,
        "vision_encoder",
        "vision_fixture",
        input_artifact_ids=["raw_fixture"],
        config_fingerprint="sha256:vision",
    )
    registry = ArtifactRegistry(artifact_root)

    original_read = ArtifactManifest.read
    read_count = 0

    def counted_read(path: Path) -> ArtifactManifest:
        nonlocal read_count
        read_count += 1
        return original_read(path)

    monkeypatch.setattr("placecell_research.artifacts.registry.ArtifactManifest.read", counted_read)

    first_found = registry.find_by_id("raw_fixture")
    second_found = registry.find_by_id("raw_fixture")
    assert first_found is not None
    assert second_found is not None
    assert read_count == 1

    first_match = registry.find_matching("vision_encoder", "sha256:vision", ["raw_fixture"])
    second_match = registry.find_matching("vision_encoder", "sha256:vision", ["raw_fixture"])
    assert first_match is not None
    assert second_match is not None
    assert read_count == 2


def test_registry_temporary_directory_uses_hidden_staging_root(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")

    with registry.temporary_directory(prefix="unit_test_") as temporary_dir:
        assert temporary_dir.parent == registry.staging_root()
        assert temporary_dir.parent.name == ".staging"
        assert temporary_dir.exists()

    assert registry.staging_root().exists()


def test_run_sweep_triggers_cleanup_after_each_trial(monkeypatch, tmp_path: Path) -> None:
    cleanup_calls = 0
    observed_overrides: list[list[str]] = []

    def counted_cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(sweep_module, "_cleanup_after_trial", counted_cleanup)

    result = sweep_module.run_sweep(
        SweepConfig(
            method="grid",
            base_experiment="debug",
            parameters={"training.lr": [1e-3, 1e-4]},
            seeds=[0, 1],
        ),
        tmp_path / "experiment.yaml",
        lambda _path, overrides, _row: observed_overrides.append(list(overrides))
        or {"metric": 1.0},
    )

    assert len(result.rows) == 4
    assert cleanup_calls == 4
    assert "seed.global_seed=0" in observed_overrides[0]
    assert "seed.collection_seed=0" in observed_overrides[0]
    assert "seed.split_seed=0" in observed_overrides[0]
    assert "seed.training_seed=0" in observed_overrides[0]
    assert "splits.seed=0" in observed_overrides[0]
    assert "analysis.example_episode_random_seed=0" in observed_overrides[0]
    assert "analysis.umap_random_seed=0" in observed_overrides[0]
    assert "analysis.probing_shuffle_seed=0" in observed_overrides[0]
    assert "analysis.remapping_shuffle_seed=0" in observed_overrides[0]


def test_run_sweep_emits_rows_after_each_grid_trial(tmp_path: Path) -> None:
    emitted_rows: list[dict[str, object]] = []

    result = sweep_module.run_sweep(
        SweepConfig(
            method="grid",
            base_experiment="debug",
            parameters={"training.lr": [1e-3, 1e-4]},
            seeds=[0],
        ),
        tmp_path / "experiment.yaml",
        lambda _path, _overrides, _row: {"stage.run_path": str(tmp_path / "run")},
        on_row=emitted_rows.append,
    )

    assert emitted_rows == result.rows
    assert [row["trial_index"] for row in emitted_rows] == [0, 1]
    assert [row["training.lr"] for row in emitted_rows] == [1e-3, 1e-4]


def test_run_paired_sweep_zips_parameters_in_declared_order(tmp_path: Path) -> None:
    observed_overrides: list[list[str]] = []

    result = sweep_module.run_sweep(
        SweepConfig(
            method="paired",
            base_experiment="debug",
            parameters={
                "spatial_model.encoder.family": ["transformer", "gru"],
                "spatial_model.encoder.layer_sizes": [[592] * 5, [1184] * 3],
                "spatial_model.encoder.xlstm_slstm_at": [None, [3]],
            },
            seeds=[42],
        ),
        tmp_path / "experiment.yaml",
        lambda _path, overrides, _row: observed_overrides.append(list(overrides))
        or {"metric": 1.0},
    )

    assert [row["spatial_model.encoder.family"] for row in result.rows] == [
        "transformer",
        "gru",
    ]
    assert len(observed_overrides) == 2
    assert "spatial_model.encoder.layer_sizes=[592, 592, 592, 592, 592]" in observed_overrides[0]
    assert "spatial_model.encoder.xlstm_slstm_at=null" in observed_overrides[0]
    assert "spatial_model.encoder.xlstm_slstm_at=[3]" in observed_overrides[1]


def test_run_sweep_logs_grid_trial_start_and_finish(tmp_path: Path, capsys) -> None:
    result = sweep_module.run_sweep(
        SweepConfig(
            method="grid",
            base_experiment="debug",
            parameters={"training.lr": [1e-3]},
            seeds=[2],
            objective_metric="metric",
        ),
        tmp_path / "experiment.yaml",
        lambda _path, _overrides, _row: {"metric": 0.75},
    )
    captured = capsys.readouterr()

    assert len(result.rows) == 1
    assert "[sweep] starting 1/1: trial_index=0 seed=2 training.lr=0.001" in captured.err
    assert "[sweep] finished 1/1: trial_index=0 seed=2 metric=0.75" in captured.err


def test_run_sweep_passes_pending_row_to_runner(tmp_path: Path) -> None:
    observed_rows: list[dict[str, object]] = []

    def runner(_path: Path, _overrides: list[str], row: dict[str, object]) -> dict[str, object]:
        observed_rows.append(dict(row))
        return {"metric": 1.0}

    sweep_module.run_sweep(
        SweepConfig(
            method="grid",
            base_experiment="debug",
            parameters={"training.lr": [1e-3]},
            seeds=[2],
        ),
        tmp_path / "experiment.yaml",
        runner,
    )

    assert observed_rows == [
        {
            "training.lr": 1e-3,
            "seed": 2,
            "trial_index": 0,
        }
    ]


def test_frozen_representation_extractor_releases_lazy_vision_state_after_build() -> None:
    vision_config = VisionConfig(type="beta_vae", latent_dim=4, channels=[8, 16])
    vision_model = build_vision_model(vision_config, (3, 16, 16))

    extractor = FrozenRepresentationExtractor.__new__(FrozenRepresentationExtractor)
    extractor.device = torch.device("cpu")
    extractor.vision_encoder = None
    extractor._vision_lazy_state_dict = vision_model.state_dict()
    extractor._vision_lazy_config = vision_config

    built_model = FrozenRepresentationExtractor._ensure_vision_encoder(
        extractor,
        torch.zeros((1, 3, 16, 16), dtype=torch.float32),
    )

    assert built_model is not None
    assert extractor.vision_encoder is built_model
    assert extractor._vision_lazy_state_dict is None
    assert extractor._vision_lazy_config is None


def test_manifest_round_trips_a_provenance_amendment(tmp_path: Path) -> None:
    """A published artifact that later gained a file records where those bytes came from."""
    amendment = {
        "date": "2000-01-01",
        "file": "weights_best_primary.pt",
        "sha1": "c5c1f84dbcd88404f260d3322848f55d8dbeeeb4",
        "size_bytes": 456900743,
        "source_path": "/elsewhere/place_model_test/weights_best_primary.pt",
        "epoch": 95,
        "reason": "restored from the external copy of the same run",
    }
    path = tmp_path / "manifest.json"
    ArtifactManifest(
        artifact_id="place_model_test",
        artifact_type="place_model",
        created_by=CreatedBy(run_id="run", stage_name="train_place_model"),
        provenance_amendments=[amendment],
    ).write(path)
    assert ArtifactManifest.read(path).provenance_amendments == [amendment]
    assert ArtifactManifest.read(tmp_path / "manifest.json").artifact_id == "place_model_test"


def test_manifest_without_amendments_reads_as_an_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"artifact_id": "a", "artifact_type": "place_model"}')
    assert ArtifactManifest.read(path).provenance_amendments == []
