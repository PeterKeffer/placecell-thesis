from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from placecell_research.config.schema import ExperimentConfig, ObjectiveConfig, SpatialModelConfig
from placecell_research.objectives.registry import build_objectives
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.stages import train_place_model
from placecell_research.stages.train_place_model import evaluate_place_model_online
from placecell_research.training.loop import TrainLoopConfig, train_model


class _SyntheticSequenceDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "latent": torch.randn(4, 8, generator=generator),
            "actions": torch.randint(0, 4, (4,), generator=generator),
            "kinematics": torch.randn(4, 2, generator=generator),
            "valid_steps": torch.tensor([1, 1, 1, 0], dtype=torch.bool),
        }


class _SingleEpisodeSequenceDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "latent": torch.randn(4, 8, generator=generator),
            "actions": torch.randint(0, 4, (4,), generator=generator),
            "kinematics": torch.randn(4, 2, generator=generator),
            "position_xy": torch.randn(4, 2, generator=generator),
            "valid_steps": torch.ones(4, dtype=torch.bool),
        }


class _MultiEpisodeSequenceDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "latent": torch.randn(4, 8, generator=generator),
            "actions": torch.randint(0, 4, (4,), generator=generator),
            "kinematics": torch.randn(4, 2, generator=generator),
            "position_xy": torch.randn(4, 2, generator=generator),
            "valid_steps": torch.ones(4, dtype=torch.bool),
        }


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}


def _build_context(total_optimizer_steps: int) -> ModelBuildContext:
    return ModelBuildContext(
        num_actions=4,
        observation_dim=8,
        kinematics_dim=2,
        total_optimizer_steps=total_optimizer_steps,
    )


def _make_model_and_objectives(loader_length: int):
    config = SpatialModelConfig()
    config.objectives = {
        "prediction_cosine": ObjectiveConfig(type="prediction_alignment", weight=1.0),
    }
    config.training.epochs = 10
    config.training.batch_size = 2
    model = build_place_model(
        config, _build_context(total_optimizer_steps=loader_length * config.training.epochs)
    )
    built_objectives = build_objectives(model, config)
    return config, model, built_objectives


def test_training_loop_runs_evaluation_on_first_interval_and_final_epoch(tmp_path: Path) -> None:
    dataset = _SyntheticSequenceDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=_collate)
    config, model, built_objectives = _make_model_and_objectives(len(loader))
    evaluated_steps: list[int] = []

    def _evaluate_fn(*_args) -> dict[str, float]:
        evaluated_steps.append(len(evaluated_steps))
        metric_value = float(len(evaluated_steps))
        return {
            "validation.total_loss": metric_value,
            "validation.xy_decode_rmse": metric_value,
        }

    result = train_model(
        model=model,
        built_objectives=built_objectives,
        model_config=config,
        train_loader=loader,
        validation_loader=loader,
        loop_config=TrainLoopConfig(
            training=config.training,
            checkpoint_dir=tmp_path,
            device=torch.device("cpu"),
            eval_every_n_epochs=8,
        ),
        evaluate_fn=_evaluate_fn,
    )

    assert len(evaluated_steps) == 3
    assert result.final_metrics["validation.xy_decode_rmse"] == 3.0
    assert result.best_primary_path is not None and result.best_primary_path.exists()


def test_training_loop_saves_independent_best_expert_probe_checkpoint(tmp_path: Path) -> None:
    dataset = _SyntheticSequenceDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=_collate)
    config, model, built_objectives = _make_model_and_objectives(len(loader))
    expert_values = iter((3.0, 2.0, 4.0))
    logged_epochs: list[dict[str, float]] = []
    expert_metric = "validation.expert_probe.experts.place_codes.xy_decode_rmse"

    def _evaluate_fn(*_args) -> dict[str, float]:
        return {
            "validation.total_loss": 1.0,
            "validation.xy_decode_rmse": 1.0,
            expert_metric: next(expert_values),
        }

    result = train_model(
        model=model,
        built_objectives=built_objectives,
        model_config=config,
        train_loader=loader,
        validation_loader=loader,
        loop_config=TrainLoopConfig(
            training=config.training,
            checkpoint_dir=tmp_path,
            device=torch.device("cpu"),
            eval_every_n_epochs=8,
            expert_probe_metric=expert_metric,
            epoch_metrics_callback=lambda metrics, _step: logged_epochs.append(dict(metrics)),
        ),
        evaluate_fn=_evaluate_fn,
    )

    assert result.best_expert_probe_path is not None
    payload = torch.load(result.best_expert_probe_path, weights_only=False)
    assert payload["epoch"] == 7
    assert logged_epochs[0]["checkpoint/expert_probe_saved"] == 1.0
    assert logged_epochs[7]["checkpoint/expert_probe_saved"] == 1.0
    assert logged_epochs[9]["checkpoint/expert_probe_saved"] == 0.0


def test_place_model_online_evaluator_has_explicit_empty_loader_contract() -> None:
    config = ExperimentConfig()
    model_config, model, built_objectives = _make_model_and_objectives(loader_length=1)
    config.spatial_model = model_config

    metrics = evaluate_place_model_online(
        model,
        built_objectives.objectives,
        None,
        config=config,
        device=torch.device("cpu"),
        online_source=config.evaluation.online_decode_source,
    )

    assert metrics == {
        "validation.total_loss": 0.0,
        config.spatial_model.training.selection.primary_metric: 0.0,
    }


def test_online_validation_preserves_per_batch_metrics_and_snapshots_reused_scalars(monkeypatch):
    config = ExperimentConfig()
    model_config, model, objectives = _make_model_and_objectives(loader_length=2)
    config.spatial_model = model_config
    loader = DataLoader(_MultiEpisodeSequenceDataset(), batch_size=2, collate_fn=_collate)
    original_loss = train_place_model.compute_total_loss
    expected = {}
    reused_scalar = torch.tensor(0.0, dtype=torch.float64)
    batch_count = 0

    def capture_loss(*args, **kwargs):
        nonlocal batch_count
        loss, batch_metrics = original_loss(*args, **kwargs)
        batch_count += 1
        reused_scalar.fill_(batch_count + 2.0**-40)
        batch_metrics["reused_scalar"] = reused_scalar
        for name, value in train_place_model.materialize_metric_values(batch_metrics).items():
            expected[name] = expected.get(name, 0.0) + value
        return loss, batch_metrics

    monkeypatch.setattr(train_place_model, "compute_total_loss", capture_loss)
    actual = evaluate_place_model_online(
        model, objectives.objectives, loader, config=config, device=torch.device("cpu"),
        online_source="encoder.place_codes",
    )
    assert batch_count == 2
    for name, total in expected.items():
        assert actual[f"validation.{name.removeprefix('loss/')}"] == total / batch_count


def test_place_model_online_evaluator_skips_decode_for_single_episode() -> None:
    config = ExperimentConfig()
    model_config, model, built_objectives = _make_model_and_objectives(loader_length=1)
    config.spatial_model = model_config
    loader = DataLoader(
        _SingleEpisodeSequenceDataset(),
        batch_size=1,
        collate_fn=_collate,
    )

    metrics = evaluate_place_model_online(
        model,
        built_objectives.objectives,
        loader,
        config=config,
        device=torch.device("cpu"),
        online_source=config.evaluation.online_decode_source,
    )

    assert "validation.total_loss" in metrics
    assert "validation.xy_decode_rmse" not in metrics
    assert "validation.code_fraction_active" in metrics


def test_place_model_online_evaluator_logs_dense_participation_ratio() -> None:
    config = ExperimentConfig()
    model_config, model, built_objectives = _make_model_and_objectives(loader_length=1)
    config.spatial_model = model_config
    loader = DataLoader(
        _MultiEpisodeSequenceDataset(),
        batch_size=4,
        collate_fn=_collate,
    )

    metrics = evaluate_place_model_online(
        model,
        built_objectives.objectives,
        loader,
        config=config,
        device=torch.device("cpu"),
        online_source="encoder.place_codes",
    )

    assert "validation.encoder.pre_sparsifier.participation_ratio" in metrics


def test_place_model_online_evaluator_caps_accumulated_eval_episodes() -> None:
    config = ExperimentConfig()
    model_config, model, built_objectives = _make_model_and_objectives(loader_length=1)
    config.spatial_model = model_config
    config.spatial_model.training.max_validation_episodes = 2
    loader = DataLoader(
        _MultiEpisodeSequenceDataset(),
        batch_size=4,
        collate_fn=_collate,
    )
    seen_batch_sizes: list[int] = []
    original_forward_sequence = model.forward_sequence

    def capturing_forward_sequence(batch):
        seen_batch_sizes.append(int(batch["valid_steps"].shape[0]))
        return original_forward_sequence(batch)

    model.forward_sequence = capturing_forward_sequence

    evaluate_place_model_online(
        model,
        built_objectives.objectives,
        loader,
        config=config,
        device=torch.device("cpu"),
        online_source=config.evaluation.online_decode_source,
    )

    assert seen_batch_sizes == [2]


def test_place_model_online_evaluator_stops_when_episode_cap_is_reached() -> None:
    config = ExperimentConfig()
    model_config, model, built_objectives = _make_model_and_objectives(loader_length=1)
    config.spatial_model = model_config
    config.spatial_model.training.max_validation_episodes = 2
    loader = DataLoader(
        _MultiEpisodeSequenceDataset(),
        batch_size=2,
        collate_fn=_collate,
    )
    seen_batch_sizes: list[int] = []
    original_forward_sequence = model.forward_sequence

    def capturing_forward_sequence(batch):
        seen_batch_sizes.append(int(batch["valid_steps"].shape[0]))
        return original_forward_sequence(batch)

    model.forward_sequence = capturing_forward_sequence

    evaluate_place_model_online(
        model,
        built_objectives.objectives,
        loader,
        config=config,
        device=torch.device("cpu"),
        online_source=config.evaluation.online_decode_source,
    )

    assert seen_batch_sizes == [2]


def _count_decode_calls(monkeypatch) -> list[int]:
    """Wrap source_decode_metrics so tests can count how often a source is actually decoded."""
    calls: list[int] = []
    original = train_place_model.source_decode_metrics

    def counting_source_decode_metrics(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(train_place_model, "source_decode_metrics", counting_source_decode_metrics)
    return calls


def test_online_evaluator_decodes_aliased_sources_once(monkeypatch) -> None:
    """encoder.hidden_state and encoder.backbone_output are the same tensor for an lstm encoder."""
    config = ExperimentConfig()
    model_config, model, built_objectives = _make_model_and_objectives(loader_length=1)
    config.spatial_model = model_config
    loader = DataLoader(_MultiEpisodeSequenceDataset(), batch_size=2, collate_fn=_collate)
    calls = _count_decode_calls(monkeypatch)

    metrics = evaluate_place_model_online(
        model,
        built_objectives.objectives,
        loader,
        config=config,
        device=torch.device("cpu"),
        online_source="encoder.place_codes",
        extra_sources=("encoder.backbone_output", "encoder.hidden_state"),
    )

    assert sum(calls) == 2
    aliased = {
        key.removeprefix("validation.encoder.backbone_output."): value
        for key, value in metrics.items()
        if key.startswith("validation.encoder.backbone_output.")
    }
    reused = {
        key.removeprefix("validation.encoder.hidden_state."): value
        for key, value in metrics.items()
        if key.startswith("validation.encoder.hidden_state.")
    }
    assert aliased and aliased == reused


def test_online_evaluator_still_decodes_every_distinct_source(monkeypatch) -> None:
    config = ExperimentConfig()
    model_config, model, built_objectives = _make_model_and_objectives(loader_length=1)
    config.spatial_model = model_config
    loader = DataLoader(_MultiEpisodeSequenceDataset(), batch_size=2, collate_fn=_collate)
    calls = _count_decode_calls(monkeypatch)

    metrics = evaluate_place_model_online(
        model,
        built_objectives.objectives,
        loader,
        config=config,
        device=torch.device("cpu"),
        online_source="encoder.place_codes",
        extra_sources=("predictor.place_codes", "encoder.pre_sparsifier"),
    )

    assert sum(calls) == 3
    assert "validation.predictor.place_codes.xy_decode_rmse" in metrics
    assert "validation.encoder.pre_sparsifier.xy_decode_rmse" in metrics


def test_duplicate_source_tracking_drops_a_pair_that_stops_matching() -> None:
    duplicates: dict[str, str] = {}
    first = torch.ones(2, 3)
    train_place_model._track_duplicate_sources(
        duplicates, {"a": first, "b": first.clone()}, first_batch=True
    )
    assert duplicates == {"b": "a"}
    train_place_model._track_duplicate_sources(
        duplicates, {"a": torch.ones(2, 3), "b": torch.zeros(2, 3)}, first_batch=False
    )
    assert duplicates == {}
