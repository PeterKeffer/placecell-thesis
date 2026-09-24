import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from placecell_research.artifacts.manifests import ArtifactManifest
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.schema import ExperimentConfig
from placecell_research.datasets.schema import DatasetSummary
from placecell_research.datasets.zarr_io import save_dataset_zarr_atomic
from placecell_research.evaluation.inference import collect_representations
from placecell_research.stages import analyze_model, evaluate_model
from placecell_research.stages import collect_representations as stage
from placecell_research.tracking.naming import RunIdentity
from placecell_research.tracking.run_directory import RunDirectory


@pytest.mark.parametrize("cached_device", ["cpu", "cuda"])
def test_streamed_stage_can_feed_consumers_without_loading_the_model(
    tmp_path, monkeypatch, cached_device
):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    raw = tmp_path / "raw"
    raw.mkdir()
    arrays = {
        "observations/rgb": np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3, 1, 1),
        "actions/discrete": np.zeros((3, 4), dtype=np.int64),
        "state/position_xy": np.zeros((3, 4, 2), dtype=np.float32),
        "masks/valid_steps": np.ones((3, 4), dtype=bool),
        "episode_metadata/length": np.full(3, 4, dtype=np.int32),
    }
    save_dataset_zarr_atomic(
        raw / "dataset.zarr",
        arrays,
        DatasetSummary(
            env_id="test", num_episodes=3, episode_length=4, num_actions=2, modalities=["rgb"]
        ),
    )
    ArtifactManifest(artifact_id="data", artifact_type="raw_dataset").write(raw / "manifest.json")
    data_path = registry.register_directory("raw_dataset", "data", raw)
    split = tmp_path / "split"
    split.mkdir()
    (split / "split_indices.json").write_text(json.dumps({"test_episode_ids": [0, 1, 2]}))
    ArtifactManifest(
        artifact_id="split", artifact_type="split_set", input_artifact_ids=["data"]
    ).write(split / "manifest.json")
    split_path = registry.register_directory("split_set", "split", split)
    model_path = tmp_path / "model"
    model_path.mkdir()
    ArtifactManifest(
        artifact_id="model", artifact_type="place_model", summary={"observation_source": "rgb"}
    ).write(model_path / "manifest.json")
    registry.register_directory("place_model", "model", model_path)
    for artifact_type, artifact_id in (
        ("raw_dataset", "data"), ("split_set", "split"), ("place_model", "model"),
    ):
        registry.mark_artifact_completed(artifact_type, artifact_id)

    class Model(torch.nn.Module):
        components = SimpleNamespace(
            config=SimpleNamespace(inputs=SimpleNamespace(observation_source="rgb"))
        )

        def forward_sequence(self, batch):
            codes = batch["rgb"][:, :, :, 0, 0].float()
            return SimpleNamespace(get_representation=lambda name: codes)

    model = Model()
    expected, _ = collect_representations(
        model,
        data_path,
        split_path,
        "test",
        ["encoder.place_codes"],
        torch.device("cpu"),
        2,
    )
    config = ExperimentConfig()
    config.reuse.place_model_artifact_id = "model"
    config.dataset.artifact_id = "data"
    config.dataset.artifact_type = "raw_dataset"
    config.splits.artifact_id = "split"
    config.representation_collection.sources = ["encoder.place_codes"]
    config.representation_collection.split_names = ["test"]
    config.representation_collection.batch_size = 2
    run = RunDirectory(tmp_path / "runs", RunIdentity("run", "test", "test", "test", "test"))
    run.create()
    runtime = SimpleNamespace(
        config=config,
        raw_payload=config.to_dict(),
        artifact_registry=registry,
        run_directory=run,
        git_state={},
    )
    monkeypatch.setattr(stage, "initialize_stage_runtime", lambda *args: runtime)
    monkeypatch.setattr(stage, "load_model_checkpoint", lambda *a, **kw: (model, {}))
    result = stage.run(Path("unused.yaml"), [])
    artifact = registry.load("representation_set", result["artifact_id"])
    assert registry.is_completed(artifact)
    assert run.load_run_manifest()["status"] == "completed"
    assert artifact.manifest.input_artifact_ids == ["model", "data", "split"]

    def fail_load(*args, **kwargs):
        raise AssertionError("Cached analysis must not load model weights")

    monkeypatch.setattr(analyze_model, "load_model_checkpoint", fail_load)
    model_cache = {}
    analysis_input = analyze_model._build_analysis_input(
        analyze_model.AnalysisSourceReference(
            "test",
            "encoder.place_codes",
            "model",
            "data",
            "raw_dataset",
            "split",
            "test",
        ),
        registry=registry,
        device=torch.device("cpu"),
        model_cache=model_cache,
        collection_plan=analyze_model.CollectionPlan(
            ("encoder.place_codes",), include_batch_keys=("rgb", "latent")
        ),
        collection_cache={},
        batch_size=2,
        max_episodes=None,
        representation_set_directory=artifact.path,
        checkpoint_selection=config.policies.checkpoint_selection,
        allow_tf32=config.spatial_model.training.allow_tf32,
    )
    np.testing.assert_array_equal(analysis_input.representation, expected["encoder.place_codes"])
    assert model_cache == {}
    np.testing.assert_array_equal(analysis_input.rgb, arrays["observations/rgb"])
    assert analysis_input.latent is None

    manifest_path = artifact.path / "representations.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["device"] = cached_device
    manifest_path.write_text(json.dumps(manifest))
    config.reuse.representation_set_artifact_id = artifact.artifact_id
    config.evaluation.sources = ["encoder.place_codes"]
    config.evaluation.split_names = ["test"]
    config.evaluation.compute_spatial_info = False
    config.evaluation.decode_include_shuffle = False
    runtime.raw_payload = config.to_dict()
    runtime.raw_payload["evaluation"].update(batch_size=16, device="cpu")
    monkeypatch.setattr(evaluate_model, "initialize_stage_runtime", lambda *args: runtime)
    monkeypatch.setattr(evaluate_model, "load_model_checkpoint", fail_load)

    evaluate = evaluate_model.evaluate_representations
    def check_cached_arrays(representations, *args, **kwargs):
        np.testing.assert_array_equal(representations["encoder.place_codes"],
                                      expected["encoder.place_codes"])
        return evaluate(representations, *args, **kwargs)

    monkeypatch.setattr(evaluate_model, "evaluate_representations", check_cached_arrays)
    result = evaluate_model.run(Path("unused.yaml"), [])
    protocol = json.loads((Path(result["evaluation_report_path"]) /
                           "evaluation_protocol.json").read_text())
    assert protocol["evaluation_config"]["batch_size"] == 16
    assert protocol["representation_inference"]["batch_size"] == 2
    assert protocol["representation_inference"]["device"] == cached_device
    assert protocol["representation_set_artifact_id"] == artifact.artifact_id
