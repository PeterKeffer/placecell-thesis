from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.collection.collector import CollectionResult
from placecell_research.datasets.schema import (
    ACTIONS_KEY,
    HEADING_KEY,
    KINEMATICS_KEY,
    LENGTH_KEY,
    POSITION_KEY,
    RGB_KEY,
    SOURCE_SEED_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    VALID_MASK_KEY,
    DatasetSummary,
)
from placecell_research.datasets.zarr_io import save_dataset_zarr_atomic
from placecell_research.stages import (
    analyze_model,
    collect_dataset,
    create_split,
    encode_dataset,
    evaluate_model,
    train_place_model,
    train_vision_encoder,
)
from placecell_research.tracking import ProgressUpdate


def _make_structured_rgb_frame(
    position_xy: np.ndarray, step_index: int, size: int = 16
) -> np.ndarray:
    y_coords, x_coords = np.meshgrid(
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    red = np.clip((position_xy[0] + 1.5) / 3.0 + 0.25 * x_coords, 0.0, 1.0)
    green = np.clip((position_xy[1] + 1.5) / 3.0 + 0.25 * y_coords, 0.0, 1.0)
    blue = np.full((size, size), step_index / 8.0, dtype=np.float32)
    return np.stack([red, green, blue], axis=0)


def _write_raw_dataset_artifact(artifact_root: Path, artifact_id: str) -> None:
    num_episodes = 8
    time_steps = 5
    image_size = 16
    num_actions = 3

    rgb = np.zeros((num_episodes, time_steps, 3, image_size, image_size), dtype=np.uint8)
    actions = np.zeros((num_episodes, time_steps), dtype=np.int64)
    positions = np.zeros((num_episodes, time_steps, 2), dtype=np.float32)
    headings = np.zeros((num_episodes, time_steps), dtype=np.float32)
    kinematics = np.zeros((num_episodes, time_steps, 2), dtype=np.float32)
    valid_steps = np.ones((num_episodes, time_steps), dtype=bool)

    for episode_index in range(num_episodes):
        base_x = -1.0 + 0.25 * episode_index
        base_y = -0.8 + 0.2 * episode_index
        for step_index in range(time_steps):
            position_xy = np.array(
                [base_x + 0.08 * step_index, base_y + 0.05 * step_index],
                dtype=np.float32,
            )
            positions[episode_index, step_index] = position_xy
            headings[episode_index, step_index] = 0.2 * step_index
            kinematics[episode_index, step_index] = np.array([0.08, 0.05], dtype=np.float32)
            actions[episode_index, step_index] = (episode_index + step_index) % num_actions
            rgb[episode_index, step_index] = (
                _make_structured_rgb_frame(position_xy, step_index) * 255.0
            ).astype(np.uint8)

    arrays = {
        RGB_KEY: rgb,
        ACTIONS_KEY: actions,
        POSITION_KEY: positions,
        HEADING_KEY: headings,
        KINEMATICS_KEY: kinematics,
        VALID_MASK_KEY: valid_steps,
        LENGTH_KEY: np.full((num_episodes,), time_steps, dtype=np.int32),
        TERMINATED_KEY: np.zeros((num_episodes,), dtype=bool),
        TRUNCATED_KEY: np.zeros((num_episodes,), dtype=bool),
        SOURCE_SEED_KEY: np.arange(num_episodes, dtype=np.int64),
    }

    dataset_dir = artifact_root / "datasets" / "raw" / artifact_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_zarr_atomic(
        dataset_dir / "dataset.zarr",
        arrays,
        DatasetSummary(
            env_id="MiniWorld-WallGapAsymLarge-v0",
            num_episodes=num_episodes,
            episode_length=time_steps,
            num_actions=num_actions,
            modalities=["rgb"],
        ),
    )
    ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type="raw_dataset",
        created_by=CreatedBy(run_id="fixture_run", stage_name="test_fixture"),
        config_fingerprint="sha256:test-fixture",
        git_commit="test",
        summary={"num_episodes": num_episodes, "episode_length": time_steps},
    ).write(dataset_dir / "manifest.json")


def _common_overrides(tmp_path: Path) -> list[str]:
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"
    return [
        f"tracking.artifact_root={artifact_root}",
        f"tracking.run_root={run_root}",
        "tracking.use_wandb=false",
        "tracking.wandb_mode=disabled",
        "splits.train_fraction=0.5",
        "splits.validation_fraction=0.25",
        "splits.test_fraction=0.25",
        "splits.constraints.minimum_episodes_per_split=1",
        "vision.epochs=1",
        "vision.batch_size=2",
        "vision.latent_dim=4",
        "vision.channels=[8,16]",
        "spatial_model.inputs.observation_source=latent",
        "spatial_model.encoder.layer_sizes=[8]",
        "spatial_model.predictor.layer_sizes=[8]",
        "spatial_model.predictor.action_embedding_dim=4",
        "spatial_model.training.epochs=1",
        "spatial_model.training.batch_size=2",
        "spatial_model.training.num_workers=0",
        "spatial_model.training.code_dim=8",
        "spatial_model.sparsifier.k_fraction=0.125",
        "spatial_model.teacher_student.mode=none",
        "evaluation.decode_include_shuffle=false",
        "analysis.max_cost_tier=standard",
        "analysis.targets.encoder_place_cells.modules=[dataset_coverage,rate_map_fields,rate_map_reliability,rate_map_bin_consistency,rate_map_split_half,rate_map_episode_correlation,rate_map_coding_purity,rate_map_panel,rate_map_grid,rate_map_extra_reliability_panels,spatial_info,sparsity,decode_xy,episode_dynamics]",
        "analysis.targets.predictor_place_cells.modules=[rate_map_fields,rate_map_reliability,rate_map_bin_consistency,rate_map_split_half,rate_map_episode_correlation,rate_map_coding_purity,rate_map_panel,rate_map_grid,rate_map_extra_reliability_panels,spatial_info,sparsity,decode_xy,episode_dynamics]",
        "analysis.targets.encoder_hidden_state.modules=[rate_map_fields,rate_map_reliability,rate_map_bin_consistency,rate_map_split_half,rate_map_episode_correlation,rate_map_coding_purity,rate_map_panel,rate_map_grid,rate_map_extra_reliability_panels,spatial_info,sparsity,decode_xy,episode_dynamics]",
        "analysis.targets.predictor_hidden_state.modules=[rate_map_fields,rate_map_reliability,rate_map_bin_consistency,rate_map_split_half,rate_map_episode_correlation,rate_map_coding_purity,rate_map_panel,rate_map_grid,rate_map_extra_reliability_panels,spatial_info,sparsity,decode_xy,episode_dynamics]",
    ]


@pytest.mark.filterwarnings("ignore:.*Matplotlib is building the font cache.*")
def test_tiny_stage_end_to_end_runs_train_eval_and_analysis(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("zarr")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    artifact_root = tmp_path / "artifacts"
    raw_artifact_id = "raw_tiny_stage_e2e"
    _write_raw_dataset_artifact(artifact_root, raw_artifact_id)

    overrides = _common_overrides(tmp_path) + [
        f"dataset.artifact_id={raw_artifact_id}",
        "dataset.artifact_type=raw_dataset",
    ]

    vision_result = train_vision_encoder.run(config_path, list(overrides))
    vision_artifact_id = str(vision_result["vision.artifact_id"])
    vision_dir = artifact_root / "vision_encoders" / vision_artifact_id
    vision_manifest_path = artifact_root / "vision_encoders" / vision_artifact_id / "manifest.json"
    vision_manifest_payload = json.loads(vision_manifest_path.read_text())
    vision_training_config = yaml.safe_load((vision_dir / "training_config.yaml").read_text())
    assert vision_manifest_payload["summary"]["input_shape"] == [3, 16, 16]
    assert vision_manifest_payload["summary"]["channels"] == [8, 16]
    assert vision_training_config["channels"] == [8, 16]

    encoded_result = encode_dataset.run(
        config_path,
        list(overrides) + [f"reuse.vision_encoder_artifact_id={vision_artifact_id}"],
    )
    encoded_artifact_id = str(encoded_result["dataset.artifact_id"])

    split_result = create_split.run(
        config_path,
        _common_overrides(tmp_path)
        + [
            f"dataset.artifact_id={encoded_artifact_id}",
            "dataset.artifact_type=encoded_dataset",
        ],
    )
    split_artifact_id = str(split_result["splits.artifact_id"])

    model_result = train_place_model.run(
        config_path,
        _common_overrides(tmp_path)
        + [
            f"dataset.artifact_id={encoded_artifact_id}",
            "dataset.artifact_type=encoded_dataset",
            f"splits.artifact_id={split_artifact_id}",
            "evaluation.evaluate_training_split=true",
        ],
    )
    model_artifact_id = str(model_result["place_model_artifact_id"])

    evaluation_result = evaluate_model.run(
        config_path,
        _common_overrides(tmp_path)
        + [
            f"dataset.artifact_id={encoded_artifact_id}",
            "dataset.artifact_type=encoded_dataset",
            f"splits.artifact_id={split_artifact_id}",
            f"reuse.place_model_artifact_id={model_artifact_id}",
            "evaluation.compute_spatial_info=true",
            "evaluation.spatial_info_top_k=4",
        ],
    )
    analysis_result = analyze_model.run(
        config_path,
        _common_overrides(tmp_path)
        + [
            f"dataset.artifact_id={encoded_artifact_id}",
            "dataset.artifact_type=encoded_dataset",
            f"splits.artifact_id={split_artifact_id}",
            f"reuse.place_model_artifact_id={model_artifact_id}",
        ],
    )

    place_model_dir = artifact_root / "place_models" / model_artifact_id
    encoded_dir = artifact_root / "datasets" / "encoded" / encoded_artifact_id
    evaluation_dir = (
        artifact_root / "reports" / "evaluation" / str(evaluation_result["evaluation_report_id"])
    )
    analysis_dir = (
        artifact_root / "reports" / "analysis" / str(analysis_result["analysis_report_id"])
    )
    run_root = tmp_path / "runs"

    assert (vision_dir / "weights.pt").exists()
    assert (vision_dir / "architecture.txt").exists()
    assert (vision_dir / "resolved_config.yaml").exists()
    assert (vision_dir / "used_hyperparameters.yaml").exists()
    assert list(vision_dir.glob("trained_on_dataset__*"))
    assert (vision_dir / "previews" / "reconstruction_grid.png").exists()
    assert (vision_dir / "previews" / "reconstruction_samples.gif").exists()
    assert len(list((vision_dir / "previews" / "reconstruction_examples").glob("*.gif"))) == 4
    assert (encoded_dir / "dataset.zarr").exists()
    assert (encoded_dir / "resolved_config.yaml").exists()
    assert (encoded_dir / "used_hyperparameters.yaml").exists()
    assert list(encoded_dir.glob("source_dataset__*"))
    assert list(encoded_dir.glob("encoded_with_vision_encoder__*"))
    assert (encoded_dir / "previews" / "reconstruction_grid.png").exists()
    assert (encoded_dir / "previews" / "reconstruction_samples.gif").exists()
    assert len(list((encoded_dir / "previews" / "reconstruction_examples").glob("*.gif"))) == 8
    assert (artifact_root / "splits" / split_artifact_id / "split_indices.json").exists()
    assert list((artifact_root / "splits" / split_artifact_id).glob("source_dataset__*"))
    assert (place_model_dir / "weights_best_primary.pt").exists()
    assert (place_model_dir / "architecture.txt").exists()
    assert (place_model_dir / "resolved_config.yaml").exists()
    assert (place_model_dir / "used_hyperparameters.yaml").exists()
    assert list(place_model_dir.glob("trained_on_dataset__*"))
    assert list(place_model_dir.glob("trained_with_split__*"))
    assert (place_model_dir / "model_contract.json").exists()
    assert (place_model_dir / "active_objectives.json").exists()
    assert (evaluation_dir / "metrics.json").exists()
    assert (evaluation_dir / "resolved_config.yaml").exists()
    assert (evaluation_dir / "used_hyperparameters.yaml").exists()
    assert list(evaluation_dir.glob("evaluated_model__*"))
    assert list(evaluation_dir.glob("evaluated_on_dataset__*"))
    assert list(evaluation_dir.glob("evaluated_with_split__*"))
    assert (analysis_dir / "summary.json").exists()
    assert (analysis_dir / "resolved_config.yaml").exists()
    assert (analysis_dir / "used_hyperparameters.yaml").exists()
    assert list(analysis_dir.glob("analyzed_model__*"))
    assert list(analysis_dir.glob("analyzed_on_dataset__*"))
    assert list(analysis_dir.glob("analyzed_with_split__*"))
    assert (analysis_dir / "figures").is_dir()
    assert (
        analysis_dir / "figures" / "encoder_place_cells" / "dataset_coverage__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_place_cells" / "dataset_coverage__train.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_place_cells" / "dataset_coverage__test.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_place_cells" / "example_episode__validation.gif"
    ).exists()
    assert (
        analysis_dir / "figures" / "predictor_place_cells" / "example_episode__validation.gif"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_hidden_state" / "example_episode__validation.gif"
    ).exists()
    assert (
        analysis_dir / "figures" / "predictor_hidden_state" / "example_episode__validation.gif"
    ).exists()
    assert (
        analysis_dir
        / "figures"
        / "comparative_combined_example_episode"
        / "combined_example_episode__validation.gif"
    ).exists()
    assert (analysis_dir / "tables").is_dir()
    assert (
        analysis_dir / "tables" / "encoder_place_cells" / "rate_map_panel_metrics__validation.csv"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_place_cells" / "rate_map_panel__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "predictor_place_cells" / "rate_map_panel__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_hidden_state" / "rate_map_panel__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "predictor_hidden_state" / "rate_map_panel__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_place_cells" / "rate_map_grid__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "predictor_place_cells" / "rate_map_grid__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "encoder_hidden_state" / "rate_map_grid__validation.png"
    ).exists()
    assert (
        analysis_dir / "figures" / "predictor_hidden_state" / "rate_map_grid__validation.png"
    ).exists()
    assert not list(
        (analysis_dir / "figures").glob("**/rate_maps__encoder.place_codes__validation__unit_*.png")
    )
    assert not any((run_dir / "analysis").exists() for run_dir in (run_root / "by_id").iterdir())
    assert not list(tmp_path.glob("split_*"))
    assert not list(tmp_path.glob("vision_encoder_*"))
    assert not list(tmp_path.glob("encoded_dataset_*"))
    assert not list(tmp_path.glob("place_model_*"))
    for run_dir in (run_root / "by_id").iterdir():
        manifest_path = run_dir / "manifests" / "run_manifest.json"
        manifest_payload = json.loads(manifest_path.read_text())
        assert "git_state" in manifest_payload
        assert "environment_info" in manifest_payload
        assert not (run_dir / "links").exists()
        assert not (run_dir / "manifests" / "git_state.json").exists()
        assert not (run_dir / "manifests" / "environment_info.json").exists()
        assert not (run_dir / "manifests" / "model_contract.json").exists()
        assert not (run_dir / "manifests" / "active_objectives.json").exists()
        assert not (run_dir / "manifests" / "input_routes.json").exists()
        assert not (run_dir / "manifests" / "selection_policy.json").exists()
        assert not (run_dir / "manifests" / "data_lineage.json").exists()
    train_model_logs = list((run_root / "by_id").glob("*/logs/stage_train_model.log"))
    train_vision_logs = list((run_root / "by_id").glob("*/logs/stage_train_vision.log"))
    assert train_model_logs
    assert train_vision_logs
    assert any("=== place_model_architecture ===" in path.read_text() for path in train_model_logs)
    assert any(
        "=== vision_encoder_architecture ===" in path.read_text() for path in train_vision_logs
    )
    assert "encoder.place_codes.decode_rmse" in evaluation_result
    assert "encoder.place_codes.spatial_info_mean_top_k" in evaluation_result
    assert "encoder.place_codes.spatial_info_max" in evaluation_result
    assert "encoder.place_codes.reliability_weighted_information_mean_top_k" in evaluation_result
    assert "encoder.place_codes.coding_purity_score_mean_top_k" in evaluation_result
    assert "encoder.place_codes.place_code_quality" in evaluation_result
    assert "encoder.place_codes.place_code_fraction_place_cells" in evaluation_result
    assert "encoder.place_codes.place_code_field_coverage" in evaluation_result
    assert "train.encoder.place_codes.decode_rmse" in evaluation_result
    assert "validation.encoder.place_codes.decode_rmse" in evaluation_result
    assert "test.encoder.place_codes.decode_rmse" in evaluation_result
    assert "predictor.place_codes.decode_rmse" in evaluation_result
    assert "encoder.hidden_state.decode_rmse" in evaluation_result
    assert "predictor.hidden_state.decode_rmse" in evaluation_result
    assert "encoder_place_cells.decode_xy.decode_rmse" in analysis_result
    assert "encoder_place_cells.spatial_info.mean_spatial_information_bits" in analysis_result
    assert (
        "encoder_place_cells.rate_map_coding_purity.mean_reliability_weighted_information"
        in analysis_result
    )
    assert "encoder_place_cells.rate_map_coding_purity.mean_coding_purity_score" in analysis_result
    assert "encoder_place_cells.population.place_code_quality" in analysis_result
    assert "encoder_place_cells.rate_map_coding_purity.fraction_place_cells" in analysis_result
    assert "encoder_place_cells.rate_map_coding_purity.field_coverage_fraction" in analysis_result
    assert "encoder_place_cells.dataset_coverage.occupied_bins_fraction" in analysis_result
    assert "encoder_place_cells.dataset_coverage_train.occupied_bins_fraction" in analysis_result
    assert "encoder_place_cells.dataset_coverage_test.occupied_bins_fraction" in analysis_result
    assert "encoder_place_cells.episode_dynamics.selected_episode_length" in analysis_result
    assert "predictor_place_cells.decode_xy.decode_rmse" in analysis_result
    assert "encoder_hidden_state.decode_xy.decode_rmse" in analysis_result
    assert "predictor_hidden_state.decode_xy.decode_rmse" in analysis_result

    place_model_hparams = yaml.safe_load(
        (place_model_dir / "used_hyperparameters.yaml").read_text()
    )
    assert place_model_hparams["stage_name"] == "train_place_model"
    assert "spatial_model" in place_model_hparams["selected_config_sections"]
    assert place_model_hparams["stage_context"]["dataset_artifact_id"] == encoded_artifact_id

    evaluation_hparams = yaml.safe_load((evaluation_dir / "used_hyperparameters.yaml").read_text())
    assert evaluation_hparams["stage_name"] == "evaluate_model"
    assert evaluation_hparams["stage_context"]["model_artifact_id"] == model_artifact_id

    analysis_hparams = yaml.safe_load((analysis_dir / "used_hyperparameters.yaml").read_text())
    assert analysis_hparams["stage_name"] == "analyze_model"
    assert "analysis" in analysis_hparams["selected_config_sections"]


def test_train_vision_encoder_uses_raw_dataset_split_for_validation_when_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("zarr")

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    artifact_root = tmp_path / "artifacts"
    raw_artifact_id = "raw_tiny_stage_with_validation"
    _write_raw_dataset_artifact(artifact_root, raw_artifact_id)

    base_overrides = _common_overrides(tmp_path) + [
        f"dataset.artifact_id={raw_artifact_id}",
        "dataset.artifact_type=raw_dataset",
    ]
    split_result = create_split.run(config_path, list(base_overrides))
    split_artifact_id = str(split_result["splits.artifact_id"])

    vision_result = train_vision_encoder.run(
        config_path,
        list(base_overrides) + [f"splits.artifact_id={split_artifact_id}"],
    )
    vision_artifact_id = str(vision_result["vision.artifact_id"])
    vision_dir = artifact_root / "vision_encoders" / vision_artifact_id
    vision_manifest_payload = json.loads((vision_dir / "manifest.json").read_text())

    assert split_artifact_id in vision_manifest_payload["input_artifact_ids"]
    assert vision_manifest_payload["summary"]["validation_enabled"] is True
    assert vision_manifest_payload["summary"]["split_artifact_id"] == split_artifact_id
    assert vision_manifest_payload["summary"]["best_validation_loss"] is not None
    assert list(vision_dir.glob("trained_with_split__*"))


def test_train_vision_encoder_uses_default_validation_split_without_split_artifact(
    tmp_path: Path,
) -> None:
    pytest.importorskip("zarr")

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    artifact_root = tmp_path / "artifacts"
    raw_artifact_id = "raw_tiny_stage_default_validation"
    _write_raw_dataset_artifact(artifact_root, raw_artifact_id)

    vision_result = train_vision_encoder.run(
        config_path,
        _common_overrides(tmp_path)
        + [
            f"dataset.artifact_id={raw_artifact_id}",
            "dataset.artifact_type=raw_dataset",
        ],
    )
    vision_artifact_id = str(vision_result["vision.artifact_id"])
    vision_dir = artifact_root / "vision_encoders" / vision_artifact_id
    vision_manifest_payload = json.loads((vision_dir / "manifest.json").read_text())

    assert vision_manifest_payload["summary"]["validation_enabled"] is True
    assert vision_manifest_payload["summary"]["split_artifact_id"] is None
    assert vision_manifest_payload["summary"]["implicit_split_enabled"] is True
    assert vision_manifest_payload["summary"]["best_validation_loss"] is not None
    assert (vision_dir / "previews" / "validation" / "reconstruction_grid.png").exists()
    assert (vision_dir / "previews" / "validation" / "reconstruction_samples.gif").exists()


@pytest.mark.filterwarnings("ignore:.*Matplotlib is building the font cache.*")
def test_collect_stage_publishes_without_repo_root_temp_leak(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("zarr")

    def fake_collect_raw_dataset(
        environment_config,
        collection_config,
        output_dir,
        collection_seed,
        progress_callback=None,
    ):
        del environment_config, collection_config, collection_seed
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb = np.zeros((1, 2, 3, 8, 8), dtype=np.uint8)
        arrays = {
            RGB_KEY: rgb,
            ACTIONS_KEY: np.zeros((1, 2), dtype=np.int64),
            POSITION_KEY: np.zeros((1, 2, 2), dtype=np.float32),
            HEADING_KEY: np.zeros((1, 2), dtype=np.float32),
            KINEMATICS_KEY: np.zeros((1, 2, 4), dtype=np.float32),
            VALID_MASK_KEY: np.ones((1, 2), dtype=bool),
            LENGTH_KEY: np.asarray([2], dtype=np.int32),
            TERMINATED_KEY: np.asarray([False], dtype=bool),
            TRUNCATED_KEY: np.asarray([False], dtype=bool),
            SOURCE_SEED_KEY: np.asarray([0], dtype=np.int64),
        }
        save_dataset_zarr_atomic(
            output_dir / "dataset.zarr",
            arrays,
            DatasetSummary(
                env_id="MiniWorld-WallGapAsymLarge-v0",
                num_episodes=1,
                episode_length=2,
                num_actions=3,
                modalities=["rgb"],
            ),
        )
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        (preview_dir / "sample_frames.png").write_bytes(b"stub")
        return CollectionResult(
            arrays=arrays,
            summary=DatasetSummary(
                env_id="MiniWorld-WallGapAsymLarge-v0",
                num_episodes=1,
                episode_length=2,
                num_actions=3,
                modalities=["rgb"],
            ),
            preview_dir=preview_dir,
        )

    monkeypatch.setattr(collect_dataset, "collect_raw_dataset", fake_collect_raw_dataset)
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )

    result = collect_dataset.run(config_path, _common_overrides(tmp_path))

    artifact_root = tmp_path / "artifacts"
    raw_artifact_id = str(result["dataset.artifact_id"])
    raw_artifact_dir = artifact_root / "datasets" / "raw" / raw_artifact_id
    assert (raw_artifact_dir / "dataset.zarr").exists()
    assert (raw_artifact_dir / "resolved_config.yaml").exists()
    assert (raw_artifact_dir / "used_hyperparameters.yaml").exists()
    raw_hparams = yaml.safe_load((raw_artifact_dir / "used_hyperparameters.yaml").read_text())
    assert raw_hparams["stage_name"] == "collect_dataset"
    assert "collection" in raw_hparams["selected_config_sections"]
    assert not list(tmp_path.glob("raw_dataset_*"))


def test_collect_stage_emits_collection_progress_updates(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    pytest.importorskip("zarr")

    def fake_collect_raw_dataset(
        environment_config,
        collection_config,
        output_dir,
        collection_seed,
        progress_callback=None,
    ):
        del environment_config, collection_config, collection_seed
        if progress_callback is not None:
            progress_callback(
                ProgressUpdate(
                    completed=0,
                    total=4,
                    elapsed_seconds=0.0,
                    unit_name="episodes",
                )
            )
            progress_callback(
                ProgressUpdate(
                    completed=2,
                    total=4,
                    elapsed_seconds=1.2,
                    unit_name="episodes",
                )
            )
            progress_callback(
                ProgressUpdate(
                    completed=4,
                    total=4,
                    elapsed_seconds=2.5,
                    unit_name="episodes",
                )
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb = np.zeros((1, 2, 3, 8, 8), dtype=np.uint8)
        arrays = {
            RGB_KEY: rgb,
            ACTIONS_KEY: np.zeros((1, 2), dtype=np.int64),
            POSITION_KEY: np.zeros((1, 2, 2), dtype=np.float32),
            HEADING_KEY: np.zeros((1, 2), dtype=np.float32),
            KINEMATICS_KEY: np.zeros((1, 2, 4), dtype=np.float32),
            VALID_MASK_KEY: np.ones((1, 2), dtype=bool),
            LENGTH_KEY: np.asarray([2], dtype=np.int32),
            TERMINATED_KEY: np.asarray([False], dtype=bool),
            TRUNCATED_KEY: np.asarray([False], dtype=bool),
            SOURCE_SEED_KEY: np.asarray([0], dtype=np.int64),
        }
        save_dataset_zarr_atomic(
            output_dir / "dataset.zarr",
            arrays,
            DatasetSummary(
                env_id="MiniWorld-WallGapAsymLarge-v0",
                num_episodes=1,
                episode_length=2,
                num_actions=3,
                modalities=["rgb"],
            ),
        )
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        return CollectionResult(
            arrays=arrays,
            summary=DatasetSummary(
                env_id="MiniWorld-WallGapAsymLarge-v0",
                num_episodes=1,
                episode_length=2,
                num_actions=3,
                modalities=["rgb"],
            ),
            preview_dir=preview_dir,
        )

    monkeypatch.setattr(collect_dataset, "collect_raw_dataset", fake_collect_raw_dataset)
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )

    collect_dataset.run(config_path, _common_overrides(tmp_path))

    captured = capsys.readouterr()
    assert "[collect_dataset] episodes 0/4" in captured.err
    assert "[collect_dataset] episodes 2/4" in captured.err
    assert "[collect_dataset] episodes 4/4 (100.0%)" in captured.err
