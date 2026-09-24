from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import TypeAdapter, ValidationError
from torch.utils.data import DataLoader

from placecell_research.config.schema import ObjectiveConfig, SpatialModelConfig
from placecell_research.objectives.registry import build_objectives
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.training.loop import TrainLoopConfig, train_model


def _config() -> SpatialModelConfig:
    config = SpatialModelConfig()
    config.encoder.layer_sizes = [8]
    config.predictor.layer_sizes = [8]
    config.training.code_dim = 8
    config.training.batch_size = 2
    config.training.epochs = 1
    config.objectives = {
        "prediction": ObjectiveConfig(type="prediction_alignment", weight=1.0),
    }
    return config


def _build_model(config: SpatialModelConfig, total_optimizer_steps: int = 4):
    return build_place_model(
        config,
        ModelBuildContext(
            num_actions=4,
            observation_dim=6,
            kinematics_dim=2,
            total_optimizer_steps=total_optimizer_steps,
        ),
    )


def _batch(time_steps: int = 6) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "latent": torch.randn(2, time_steps, 6),
        "actions": torch.randint(0, 4, (2, time_steps)),
        "kinematics": torch.randn(2, time_steps, 2),
        "valid_steps": torch.ones(2, time_steps, dtype=torch.bool),
    }


def _slice_batch(
    batch: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    return {key: value[:, start:end] for key, value in batch.items()}


@pytest.mark.parametrize(
    "representation",
    ["encoder.place_codes", "predictor.place_codes", "teacher.place_codes"],
)
def test_stateful_chunks_match_full_recurrent_forward(representation: str) -> None:
    model = _build_model(_config())
    model.eval()
    batch = _batch()

    full = model.forward_sequence(batch)
    first, state = model.forward_chunk(_slice_batch(batch, 0, 3))
    second, _state = model.forward_chunk(_slice_batch(batch, 3, 6), state)
    chunked = torch.cat(
        (first.get_representation(representation), second.get_representation(representation)),
        dim=1,
    )

    torch.testing.assert_close(chunked, full.get_representation(representation))
    assert second.masks["prediction_valid_steps"].all()


def test_detached_chunk_state_truncates_input_gradient() -> None:
    model = _build_model(_config())
    model.train()
    batch = _batch()
    first_batch = _slice_batch(batch, 0, 3)
    second_batch = _slice_batch(batch, 3, 6)
    first_batch["latent"] = first_batch["latent"].clone().requires_grad_()
    second_batch["latent"] = second_batch["latent"].clone().requires_grad_()

    _first_bundle, state = model.forward_chunk(first_batch)
    detached_state = model.detach_chunk_state(state)
    second_bundle, _state = model.forward_chunk(second_batch, detached_state)
    loss = second_bundle.get_representation("predictor.place_codes").square().mean()
    loss.backward()

    assert first_batch["latent"].grad is None
    assert second_batch["latent"].grad is not None


class _SequenceDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = _batch(time_steps=4)
        return {key: value[index] for key, value in batch.items()}


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}


def test_training_loop_optimizes_each_contiguous_chunk(tmp_path: Path) -> None:
    config = _config()
    config.training.bptt_window = 2
    loader = DataLoader(_SequenceDataset(), batch_size=2, collate_fn=_collate)
    model = _build_model(config, total_optimizer_steps=2)
    objectives = build_objectives(model, config)
    observed_steps: list[int] = []

    train_model(
        model=model,
        built_objectives=objectives,
        model_config=config,
        train_loader=loader,
        validation_loader=None,
        loop_config=TrainLoopConfig(
            training=config.training,
            checkpoint_dir=tmp_path,
            device=torch.device("cpu"),
            sequence_length=4,
            step_metrics_callback=lambda _metrics, step: observed_steps.append(step),
        ),
    )

    assert observed_steps == [0, 1]


def test_stateful_bptt_config_rejects_nonrecurrent_predictor() -> None:
    with pytest.raises(ValidationError, match="predictor.family"):
        TypeAdapter(SpatialModelConfig).validate_python(
            {
                "encoder": {"family": "lstm"},
                "predictor": {"family": "transformer"},
                "training": {"bptt_window": 2},
            }
        )
