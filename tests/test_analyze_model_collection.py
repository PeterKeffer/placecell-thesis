from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from placecell_research.stages import analyze_model as analyze_model_stage
from placecell_research.stages.analyze_model import (
    _analysis_config_with_available_layer_targets,
    _AnalysisSourceReference,
    _build_analysis_input,
    _build_dataset_coverage_analysis_input,
    _collection_plans_by_source_group,
    _CollectionPlan,
    _required_batch_keys,
    _required_comparative_batch_keys,
    _single_work_items_by_source_group,
    _SingleAnalysisWorkItem,
    _source_collection_group_key,
)


def test_build_analysis_input_collects_all_sources_for_shared_dataset_pass(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    model_cache = {"model_a": object()}
    collection_cache = {}
    dataset_path = tmp_path / "dataset"
    split_path = tmp_path / "split"
    dataset_path.mkdir()
    split_path.mkdir()

    class FakeRegistry:
        def load(self, artifact_type: str, artifact_id: str):
            if artifact_type == "encoded_dataset":
                return SimpleNamespace(path=dataset_path)
            if artifact_type == "split_set":
                return SimpleNamespace(path=split_path)
            raise AssertionError(f"unexpected load: {artifact_type} {artifact_id}")

    def fake_collect_representations(
        model,
        dataset_directory: Path,
        split_directory: Path,
        split_name: str,
        source_names: list[str],
        device: torch.device,
        batch_size: int,
        include_batch_keys: list[str] | None = None,
        observation_source=None,
        max_episodes=None,
        progress_callback=None,
        input_batch_cache=None,
    ):
        del model, device, observation_source, progress_callback
        calls.append(
            {
                "dataset_directory": dataset_directory,
                "split_directory": split_directory,
                "split_name": split_name,
                "source_names": list(source_names),
                "batch_size": batch_size,
                "include_batch_keys": list(include_batch_keys or []),
                "max_episodes": max_episodes,
            }
        )
        representations = {
            "encoder.place_codes": np.ones((2, 3, 4), dtype=np.float32),
            "predictor.hidden_state": np.full((2, 3, 5), 2.0, dtype=np.float32),
            "encoder.hidden_state_layer_1": np.full((2, 3, 6), 3.0, dtype=np.float32),
        }
        metadata = {
            "position_xy": np.zeros((2, 3, 2), dtype=np.float32),
            "valid_steps": np.ones((2, 3), dtype=bool),
            "rgb": np.zeros((2, 3, 3, 4, 4), dtype=np.float32),
        }
        return representations, metadata

    monkeypatch.setattr(
        "placecell_research.stages.analyze_model.collect_representations",
        fake_collect_representations,
    )
    monkeypatch.setattr(
        "placecell_research.stages.analyze_model.load_dataset_manifest",
        lambda _path: SimpleNamespace(env_id="MiniWorld-WallGapAsymLarge-v0"),
    )

    encoder_reference = _AnalysisSourceReference(
        label="encoder_place_cells",
        source_name="encoder.place_codes",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="validation",
    )
    predictor_reference = _AnalysisSourceReference(
        label="predictor_hidden_state",
        source_name="predictor.hidden_state",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="validation",
    )
    collection_plan = _CollectionPlan(
        source_names=("encoder.place_codes", "predictor.hidden_state"),
        include_batch_keys=("rgb",),
    )

    encoder_input = _build_analysis_input(
        encoder_reference,
        registry=FakeRegistry(),
        device=torch.device("cpu"),
        model_cache=model_cache,
        collection_plan=collection_plan,
        collection_cache=collection_cache,
        batch_size=8,
        max_episodes=16,
    )
    predictor_input = _build_analysis_input(
        predictor_reference,
        registry=FakeRegistry(),
        device=torch.device("cpu"),
        model_cache=model_cache,
        collection_plan=collection_plan,
        collection_cache=collection_cache,
        batch_size=8,
        max_episodes=16,
    )

    assert len(calls) == 1
    assert calls[0]["source_names"] == ["encoder.place_codes", "predictor.hidden_state"]
    assert calls[0]["include_batch_keys"] == ["rgb"]
    assert encoder_input.representation.shape == (2, 3, 4)
    assert predictor_input.representation.shape == (2, 3, 5)
    assert encoder_input.metadata["env_id"] == "MiniWorld-WallGapAsymLarge-v0"
    assert predictor_input.metadata["env_id"] == "MiniWorld-WallGapAsymLarge-v0"


def test_build_analysis_input_can_request_layerwise_hidden_state_sources(
    monkeypatch, tmp_path: Path
) -> None:
    model_cache = {"model_a": object()}
    collection_cache = {}
    dataset_path = tmp_path / "dataset"
    split_path = tmp_path / "split"
    dataset_path.mkdir()
    split_path.mkdir()

    class FakeRegistry:
        def load(self, artifact_type: str, artifact_id: str):
            if artifact_type == "encoded_dataset":
                return SimpleNamespace(path=dataset_path)
            if artifact_type == "split_set":
                return SimpleNamespace(path=split_path)
            raise AssertionError(f"unexpected load: {artifact_type} {artifact_id}")

    def fake_collect_representations(
        model,
        dataset_directory: Path,
        split_directory: Path,
        split_name: str,
        source_names: list[str],
        device: torch.device,
        batch_size: int,
        include_batch_keys: list[str] | None = None,
        observation_source=None,
        max_episodes=None,
        progress_callback=None,
        input_batch_cache=None,
    ):
        del (
            model,
            dataset_directory,
            split_directory,
            split_name,
            device,
            batch_size,
            include_batch_keys,
            observation_source,
            max_episodes,
            progress_callback,
        )
        assert source_names == ["encoder.hidden_state_layer_1"]
        return (
            {"encoder.hidden_state_layer_1": np.full((2, 3, 6), 7.0, dtype=np.float32)},
            {
                "position_xy": np.zeros((2, 3, 2), dtype=np.float32),
                "valid_steps": np.ones((2, 3), dtype=bool),
            },
        )

    monkeypatch.setattr(
        "placecell_research.stages.analyze_model.collect_representations",
        fake_collect_representations,
    )
    monkeypatch.setattr(
        "placecell_research.stages.analyze_model.load_dataset_manifest",
        lambda _path: SimpleNamespace(env_id="MiniWorld-WallGapAsymLarge-v0"),
    )

    layer_reference = _AnalysisSourceReference(
        label="encoder_hidden_state_layer_1",
        source_name="encoder.hidden_state_layer_1",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="validation",
    )
    analysis_input = _build_analysis_input(
        layer_reference,
        registry=FakeRegistry(),
        device=torch.device("cpu"),
        model_cache=model_cache,
        collection_plan=_CollectionPlan(source_names=("encoder.hidden_state_layer_1",)),
        collection_cache=collection_cache,
        batch_size=8,
        max_episodes=16,
    )

    assert analysis_input.representation.shape == (2, 3, 6)
    assert np.all(analysis_input.representation == 7.0)


def test_dataset_coverage_analysis_input_reads_positions_without_model_inference(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset"
    split_path = tmp_path / "split"
    dataset_path.mkdir()
    split_path.mkdir()
    (split_path / "split_indices.json").write_text('{"train_episode_ids": [2, 0]}')

    positions = np.asarray(
        [
            [[0.0, 0.0], [0.1, 0.0]],
            [[1.0, 1.0], [1.1, 1.0]],
            [[2.0, 2.0], [2.1, 2.0]],
        ],
        dtype=np.float32,
    )
    valid_steps = np.asarray(
        [
            [True, False],
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )
    fake_group = {
        "state": {"position_xy": positions},
        "masks": {"valid_steps": valid_steps},
    }

    class FakeZarr:
        def open(self, path: str, mode: str):
            assert path == str(dataset_path / "dataset.zarr")
            assert mode == "r"
            return fake_group

    class FakeRegistry:
        def load(self, artifact_type: str, artifact_id: str):
            if artifact_type == "encoded_dataset":
                return SimpleNamespace(path=dataset_path)
            if artifact_type == "split_set":
                return SimpleNamespace(path=split_path)
            raise AssertionError(f"unexpected load: {artifact_type} {artifact_id}")

    def fail_model_load(*_args, **_kwargs):
        raise AssertionError("model loaded")

    def fail_collect_representations(*_args, **_kwargs):
        raise AssertionError("inference ran")

    monkeypatch.setattr(analyze_model_stage, "load_model_checkpoint", fail_model_load)
    monkeypatch.setattr(
        analyze_model_stage,
        "collect_representations",
        fail_collect_representations,
    )
    monkeypatch.setattr(analyze_model_stage, "_require_zarr", lambda: (FakeZarr(), None))
    monkeypatch.setattr(
        analyze_model_stage,
        "load_dataset_manifest",
        lambda _path: SimpleNamespace(env_id="MiniWorld-WallGapAsymLarge-v0"),
    )

    reference = _AnalysisSourceReference(
        label="encoder_place_cells",
        source_name="encoder.place_codes",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="train",
    )
    analysis_input = _build_dataset_coverage_analysis_input(
        reference,
        registry=FakeRegistry(),
    )

    np.testing.assert_array_equal(analysis_input.position_xy, positions[[2, 0]])
    np.testing.assert_array_equal(analysis_input.valid_mask, valid_steps[[2, 0]])
    assert analysis_input.representation.shape == (2, 2, 1)
    assert analysis_input.metadata["env_id"] == "MiniWorld-WallGapAsymLarge-v0"


def test_single_source_collection_plans_group_targets_by_shared_inputs(monkeypatch) -> None:
    def fake_required_batch_keys(module_names: list[str]) -> list[str]:
        if "episode_dynamics" in module_names:
            return ["rgb"]
        return []

    monkeypatch.setattr(
        analyze_model_stage,
        "_required_batch_keys",
        fake_required_batch_keys,
    )
    encoder_reference = _AnalysisSourceReference(
        label="encoder_place_cells",
        source_name="encoder.place_codes",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="validation",
    )
    predictor_reference = _AnalysisSourceReference(
        label="predictor_place_cells",
        source_name="predictor.place_codes",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="validation",
    )
    hidden_reference = _AnalysisSourceReference(
        label="encoder_hidden_state_layer_0",
        source_name="encoder.hidden_state_layer_0",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="validation",
    )
    train_reference = _AnalysisSourceReference(
        label="encoder_place_cells",
        source_name="encoder.place_codes",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="train",
    )
    work_items = [
        _SingleAnalysisWorkItem(
            progress_label="encoder_place_cells",
            reference=encoder_reference,
            module_names=[
                "rate_map_fields",
                "rate_map_reliability",
                "rate_map_bin_consistency",
                "rate_map_split_half",
                "rate_map_episode_correlation",
                "rate_map_coding_purity",
                "episode_dynamics",
            ],
        ),
        _SingleAnalysisWorkItem(
            progress_label="predictor_place_cells",
            reference=predictor_reference,
            module_names=[
                "rate_map_fields",
                "rate_map_reliability",
                "rate_map_bin_consistency",
                "rate_map_split_half",
                "rate_map_episode_correlation",
                "rate_map_coding_purity",
            ],
        ),
        _SingleAnalysisWorkItem(
            progress_label="encoder_hidden_state_layer_0",
            reference=hidden_reference,
            module_names=[
                "rate_map_fields",
                "rate_map_reliability",
                "rate_map_bin_consistency",
                "rate_map_split_half",
                "rate_map_episode_correlation",
                "rate_map_coding_purity",
            ],
        ),
        _SingleAnalysisWorkItem(
            progress_label="encoder_place_cells:dataset_coverage:train",
            reference=train_reference,
            module_names=["dataset_coverage"],
        ),
    ]

    plans = _collection_plans_by_source_group(work_items)
    grouped_items = _single_work_items_by_source_group(work_items)

    validation_plan = plans[_source_collection_group_key(encoder_reference)]
    hidden_plan = plans[_source_collection_group_key(hidden_reference)]
    train_plan = plans[_source_collection_group_key(train_reference)]
    assert validation_plan.source_names == ("encoder.place_codes", "predictor.place_codes")
    assert validation_plan.include_batch_keys == ("rgb",)
    assert hidden_plan.source_names == ("encoder.hidden_state_layer_0",)
    assert train_plan.source_names == ("encoder.place_codes",)
    assert train_plan.include_batch_keys == ()
    assert [group_key for group_key, _items in grouped_items] == [
        _source_collection_group_key(encoder_reference),
        _source_collection_group_key(hidden_reference),
        _source_collection_group_key(train_reference),
    ]
    assert [item.progress_label for _group_key, items in grouped_items for item in items] == [
        "encoder_place_cells",
        "predictor_place_cells",
        "encoder_hidden_state_layer_0",
        "encoder_place_cells:dataset_coverage:train",
    ]


def test_required_comparative_batch_keys_reads_combined_episode_requirements() -> None:
    required_keys = _required_comparative_batch_keys({"module": "combined_episode_dynamics"})
    assert required_keys == ["latent", "rgb"]


def test_cognitive_map_geometry_requests_latent_batch_key() -> None:
    assert "latent" in _required_batch_keys(["cognitive_map_geometry"])


def test_build_analysis_input_populates_latent_when_collected(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    split_path = tmp_path / "split"
    dataset_path.mkdir()
    split_path.mkdir()

    class FakeRegistry:
        def load(self, artifact_type: str, artifact_id: str):
            if artifact_type == "encoded_dataset":
                return SimpleNamespace(path=dataset_path)
            if artifact_type == "split_set":
                return SimpleNamespace(path=split_path)
            raise AssertionError(f"unexpected load: {artifact_type} {artifact_id}")

    def fake_collect_representations(*_args, include_batch_keys=None, **_kwargs):
        assert "latent" in list(include_batch_keys or [])
        representations = {"encoder.place_codes": np.ones((2, 3, 4), dtype=np.float32)}
        metadata = {
            "position_xy": np.zeros((2, 3, 2), dtype=np.float32),
            "valid_steps": np.ones((2, 3), dtype=bool),
            "latent": np.ones((2, 3, 5), dtype=np.float32),
        }
        return representations, metadata

    monkeypatch.setattr(
        "placecell_research.stages.analyze_model.collect_representations",
        fake_collect_representations,
    )
    monkeypatch.setattr(
        "placecell_research.stages.analyze_model.load_dataset_manifest",
        lambda _path: SimpleNamespace(env_id="MiniWorld-WallGapAsymLarge-v0"),
    )

    reference = _AnalysisSourceReference(
        label="encoder_place_cells",
        source_name="encoder.place_codes",
        model_artifact_id="model_a",
        dataset_artifact_id="dataset_a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split_a",
        split_name="validation",
    )
    analysis_input = _build_analysis_input(
        reference,
        registry=FakeRegistry(),
        device=torch.device("cpu"),
        model_cache={"model_a": object()},
        collection_plan=_CollectionPlan(
            source_names=("encoder.place_codes",), include_batch_keys=("latent",)
        ),
        collection_cache={},
        batch_size=8,
        max_episodes=16,
    )

    assert analysis_input.latent is not None
    assert analysis_input.latent.shape == (2, 3, 5)


def test_analysis_config_disables_layer_targets_missing_from_model_contract() -> None:
    analysis_config = {
        "targets": {
            "encoder_place_cells": {
                "source": "encoder.place_codes",
                "enabled": True,
                "modules": ["decode_xy"],
            },
            "encoder_hidden_state_layer_0": {
                "source": "encoder.hidden_state_layer_0",
                "enabled": True,
                "modules": ["decode_xy"],
            },
            "encoder_hidden_state_layer_2": {
                "source": "encoder.hidden_state_layer_2",
                "enabled": True,
                "modules": ["decode_xy"],
            },
            "bad_source": {
                "source": "encoder.missing",
                "enabled": True,
                "modules": ["decode_xy"],
            },
        }
    }

    filtered_config, skipped_targets = _analysis_config_with_available_layer_targets(
        analysis_config,
        available_representations={
            "encoder.place_codes",
            "encoder.hidden_state_layer_0",
            "encoder.hidden_state_layer_1",
        },
    )

    assert skipped_targets == [
        {
            "target": "encoder_hidden_state_layer_2",
            "source": "encoder.hidden_state_layer_2",
        }
    ]
    assert filtered_config["targets"]["encoder_place_cells"]["enabled"] is True
    assert filtered_config["targets"]["encoder_hidden_state_layer_0"]["enabled"] is True
    assert filtered_config["targets"]["encoder_hidden_state_layer_2"]["enabled"] is False
    assert filtered_config["targets"]["bad_source"]["enabled"] is True
    assert analysis_config["targets"]["encoder_hidden_state_layer_2"]["enabled"] is True
