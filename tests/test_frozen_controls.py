from __future__ import annotations

import numpy as np
import pytest
import torch

from placecell_research.config.schema import ObjectiveConfig, SpatialModelConfig
from placecell_research.evaluation.frozen_controls import (
    code_organisation_metrics,
    fixed_topk,
    localization_curves,
    perturbed_codes,
)
from placecell_research.evaluation.matched_decode import MatchedPositionDecoder
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model


def test_fixed_mask_selects_signed_values_without_learning_or_rescaling():
    values = np.array([[[-20, -2, 1, 4], [8, 0, -3, 2]]], dtype=np.float32)
    original = values.copy()
    np.testing.assert_array_equal(fixed_topk(values, 2), [[[0, 0, 1, 4], [8, 0, 0, 2]]])
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(fixed_topk(values, 4), values)


def test_real_recurrent_model_reset_and_restore():
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [8]
    config.predictor.layer_sizes = [8]
    config.training.code_dim = 8
    config.objectives = {"prediction": ObjectiveConfig(type="prediction_alignment")}
    model = build_place_model(
        config,
        ModelBuildContext(
            num_actions=3, observation_dim=6, kinematics_dim=2, total_optimizer_steps=4
        ),
    )
    model.eval()
    torch.manual_seed(7)
    batch = {
        "latent": torch.randn(2, 12, 6),
        "actions": torch.zeros(2, 12, dtype=torch.long),
        "kinematics": torch.randn(2, 12, 2),
        "valid_steps": torch.ones(2, 12, dtype=torch.bool),
    }
    common = dict(onset=4, duration=3, source="encoder.hidden_state")
    clean = perturbed_codes(model, batch, **common)
    with torch.inference_mode():
        full = model.forward_sequence(batch).get_representation("encoder.hidden_state")
    torch.testing.assert_close(clean, full)
    reset = perturbed_codes(model, batch, reset=True, **common)
    with torch.inference_mode():
        tail = model.forward_sequence({k: v[:, 4:] for k, v in batch.items()})
    torch.testing.assert_close(reset[:, :4], clean[:, :4])
    torch.testing.assert_close(reset[:, 4:], tail.get_representation("encoder.hidden_state"))
    assert not torch.allclose(reset[:, 4:], clean[:, 4:])
    corrupted = {k: v.clone() for k, v in batch.items()}
    corrupted["latent"][:, 4:7] = 0
    with torch.inference_mode():
        expected = model.forward_sequence(corrupted).get_representation("encoder.hidden_state")
    blackout = perturbed_codes(model, batch, blackout=True, **common)
    torch.testing.assert_close(blackout, expected)
    assert torch.count_nonzero(batch["latent"][:, 4:7]) > 0


def test_tuning_reports_dead_units_and_retains_signed_activity():
    positions = np.array([[[0, 0], [1, 1], [0, 0], [1, 1]]], dtype=np.float32)
    values = np.array([[[1, 0, -1], [0, 2, -1], [1, 0, -1], [0, 2, -1]]], dtype=np.float32)
    metrics, arrays = code_organisation_metrics(
        values, positions, np.ones((1, 4), bool), ((-1, 2), (-1, 2))
    )
    assert metrics["recruited_fraction_abs_gt_1e-4"] == 1
    assert metrics["positive_part_field_unit_fraction"] == pytest.approx(2 / 3)
    assert np.all(
        arrays["positive_part_rate_maps"][2][np.isfinite(arrays["positive_part_rate_maps"][2])] == 0
    )


def test_recovery_decoder_is_frozen_and_errors_are_paired():
    rng = np.random.default_rng(42)
    codes = rng.normal(size=(8, 10, 3))
    positions = codes[..., :2] * 2
    decoder = MatchedPositionDecoder.fit(codes[:4].reshape(-1, 3), positions[:4].reshape(-1, 2))
    decoder.score(codes[4:6].reshape(-1, 3), positions[4:6].reshape(-1, 2), select=True)
    chosen = decoder.selected
    curves = localization_curves(decoder, codes[6:], positions[6:], np.ones((2, 10), bool))
    assert curves["decode_rmse_by_step"].max() < 1e-5
    assert curves["decode_r2_by_step"].min() > 0.999
    perturbed = localization_curves(
        decoder, np.zeros_like(codes[6:]), positions[6:], np.ones((2, 10), bool)
    )
    assert perturbed["episode_localization_error"].mean() > 0.5
    assert decoder.selected == chosen


def test_organisation_runner_matches_manifests_and_rejects_changed_recipe(tmp_path):
    import argparse
    import copy
    import json
    from pathlib import Path
    from runpy import run_path

    import yaml

    from placecell_research.evaluation.representation_store import (
        write_representation_manifest,
        write_representation_set,
    )

    runner = run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/evaluation/frozen_controls.py")
    )
    rng = np.random.default_rng(9)
    recipe = {
        "seed": {"training_seed": 42},
        "dataset": {"artifact_id": "dataset"},
        "splits": {"artifact_id": "split"},
        "spatial_model": {
            "training": {"code_dim": 4},
            "sparsifier": {"type": "kwinners", "k_fraction": 1.0},
        },
    }
    models, stores = [], []
    samples = {
        split: (
            rng.normal(size=(4, 10, 4)).astype(np.float32),
            rng.uniform(-1, 1, size=(4, 10, 2)).astype(np.float32),
        )
        for split in ("train", "validation", "test")
    }
    for name, fraction in [("dense", 1.0), ("sparse", 0.5)]:
        model, store = tmp_path / name, tmp_path / (name + "_store")
        model.mkdir()
        store.mkdir()
        models.append(model)
        stores.append(store)
        (model / "manifest.json").write_text(
            json.dumps({"artifact_id": name, "input_artifact_ids": ["dataset", "split"]})
        )
        configured = copy.deepcopy(recipe)
        configured["spatial_model"]["sparsifier"]["k_fraction"] = fraction
        (model / "used_hyperparameters.yaml").write_text(
            yaml.safe_dump({"selected_config_sections": configured})
        )
        write_representation_manifest(
            store,
            {
                "place_model_artifact_id": name,
                "dataset_artifact_id": "dataset",
                "split_artifact_id": "split",
                "checkpoint_selection": "last",
                "episode_ids": {
                    "train": [0, 1, 2, 3],
                    "validation": [4, 5, 6, 7],
                    "test": [8, 9, 10, 11],
                },
                "device": "cpu",
                "batch_size": 4,
                "allow_tf32": False,
                "torch_version": str(torch.__version__),
            },
        )
        for split, (codes, positions) in samples.items():
            write_representation_set(
                store,
                split_name=split,
                representations={"encoder.place_codes": codes},
                metadata={"position_xy": positions, "valid_steps": np.ones((4, 10), bool)},
            )
    output = tmp_path / "output"
    output.mkdir()
    args = argparse.Namespace(
        dense=stores[0],
        sparse=stores[1],
        dense_model=models[0],
        sparse_model=models[1],
        k=2,
        bounds=[-1, 1, -1, 1],
        output=output,
    )
    result = runner["organisation"](args)
    assert len(result["metrics"]) == 6
    assert (output / "dense_fixed_topk_tuning.npz").is_file()
    configured["spatial_model"]["changed_objective"] = True
    (models[1] / "used_hyperparameters.yaml").write_text(
        yaml.safe_dump({"selected_config_sections": configured})
    )
    with pytest.raises(ValueError, match="differ beyond k_fraction"):
        runner["organisation"](args)


def test_perturbation_runner_reads_checkpoint_dataset_and_saves_recovery(tmp_path):
    import argparse
    import json
    from pathlib import Path
    from runpy import run_path

    import zarr

    runner = run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/evaluation/frozen_controls.py")
    )
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [8]
    config.predictor.layer_sizes = [8]
    config.training.code_dim = 8
    model = build_place_model(
        config,
        ModelBuildContext(
            num_actions=3, observation_dim=6, kinematics_dim=2, total_optimizer_steps=4
        ),
    )
    model_dir, dataset, split, output = [tmp_path / n for n in ("model", "data", "split", "out")]
    for path in (model_dir, dataset, split, output):
        path.mkdir()
    from dataclasses import asdict

    from placecell_research.objectives.registry import build_objectives_and_heads

    built = build_objectives_and_heads(model, config)
    context = ModelBuildContext(
        num_actions=3, observation_dim=6, kinematics_dim=2, total_optimizer_steps=4
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "auxiliary_state_dict": built.auxiliary_heads.state_dict(),
            "build_context": context.to_checkpoint_payload(asdict(config)),
        },
        model_dir / "weights_last.pt",
    )
    for directory, manifest in [
        (model_dir, {"artifact_id": "model", "input_artifact_ids": ["dataset", "split"]}),
        (dataset, {"artifact_id": "dataset"}),
        (split, {"artifact_id": "split"}),
    ]:
        (directory / "manifest.json").write_text(json.dumps(manifest))
    (split / "split_indices.json").write_text(
        json.dumps({"train": [0, 1], "validation": [2, 3], "test": [4, 5]})
    )
    rng = np.random.default_rng(42)
    group = zarr.open_group(str(dataset / "dataset.zarr"), mode="w")
    for key, values in {
        "observations/latent": rng.normal(size=(6, 12, 6)).astype(np.float32),
        "actions/discrete": np.zeros((6, 12), np.int64),
        "state/position_xy": rng.normal(size=(6, 12, 2)).astype(np.float32),
        "state/kinematics": np.zeros((6, 12, 2), np.float32),
        "masks/valid_steps": np.ones((6, 12), bool),
    }.items():
        group.create_dataset(key, data=values)
    result = runner["perturbation"](
        argparse.Namespace(
            model=model_dir,
            dataset=dataset,
            split=split,
            output=output,
            device="cpu",
            max_episodes=2,
            batch_size=2,
            onset=4,
            duration=3,
        )
    )
    assert len(result["checkpoint_sha256"]) == 64
    assert result["episode_ids"]["test"] == [4, 5]
    with np.load(output / "recovery_curves.npz") as curves:
        assert curves["blackout.paired_excess_error"].shape == (2, 12)
        np.testing.assert_array_equal(curves["reset.paired_excess_error"][:, :4], 0)


def test_same_step_alignment_backpropagates_through_predictor():
    from placecell_research.objectives.prediction import PredictionAlignmentObjective

    config = SpatialModelConfig()
    config.encoder.layer_sizes = [8]
    config.predictor.layer_sizes = [8]
    config.training.code_dim = 8
    objective_config = ObjectiveConfig(type="prediction_alignment", target_offset=0)
    config.objectives = {"prediction": objective_config}
    model = build_place_model(
        config,
        ModelBuildContext(
            num_actions=3, observation_dim=6, kinematics_dim=2, total_optimizer_steps=4
        ),
    )
    batch = {
        "latent": torch.randn(2, 6, 6),
        "actions": torch.zeros(2, 6, dtype=torch.long),
        "kinematics": torch.zeros(2, 6, 2),
        "valid_steps": torch.ones(2, 6, dtype=torch.bool),
    }
    result = PredictionAlignmentObjective(name="prediction", config=objective_config).compute(
        model.forward_sequence(batch), batch
    )
    result.loss.backward()
    gradients = [parameter.grad for parameter in model.predictor_temporal.parameters()]
    assert gradients
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
