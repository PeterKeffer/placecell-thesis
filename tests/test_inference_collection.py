from __future__ import annotations

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from placecell_research.evaluation import inference
from placecell_research.evaluation.inference import (
    InputBatchCache,
    collect_representations,
    iter_representation_batches,
)


class _FakeBundle:
    def __init__(self, representation: torch.Tensor) -> None:
        self.representation = representation

    def get_representation(self, source_name: str) -> torch.Tensor:
        assert source_name == "encoder.place_codes"
        return self.representation


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_training_states: list[bool] = []
        self.forward_inference_mode_states: list[bool] = []

    def forward_sequence(self, batch: dict[str, torch.Tensor]) -> _FakeBundle:
        self.forward_training_states.append(self.training)
        self.forward_inference_mode_states.append(torch.is_inference_mode_enabled())
        batch_shape = batch["valid_steps"].shape[:2]
        representation = torch.ones((*batch_shape, 3), dtype=torch.float32)
        return _FakeBundle(representation)


def test_streaming_is_lazy_and_restores_mode_when_consumer_stops(monkeypatch, tmp_path):
    reads = []
    def batches(*args, **kwargs):
        for index in range(3):
            reads.append(index)
            yield {"valid_steps": torch.ones((2, 4), dtype=torch.bool)}
    monkeypatch.setattr(inference, "iterate_dataset_batches", batches)
    monkeypatch.setattr(inference, "load_split_indices", lambda *a: list(range(6)))
    model = _FakeModel()
    with closing(iter_representation_batches(
        model, tmp_path, tmp_path, "test", ["encoder.place_codes"], torch.device("cpu"), 2,
    )) as stream:
        representations, metadata = next(stream)
        assert reads == [0]
        assert representations["encoder.place_codes"].shape == (2, 4, 3)
        assert metadata["valid_steps"].all()
        assert not torch.is_inference_mode_enabled()
    assert model.training
    assert model.forward_inference_mode_states == [True]


def test_collect_representations_uses_inference_mode_and_restores_training_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_iterator_kwargs: dict[str, object] = {}

    def fake_iterate_dataset_batches(*args, **kwargs):
        del args
        captured_iterator_kwargs.update(kwargs)
        yield {
            "position_xy": torch.zeros((2, 4, 2), dtype=torch.float32),
            "valid_steps": torch.ones((2, 4), dtype=torch.bool),
        }

    monkeypatch.setattr(inference, "load_split_indices", lambda *_args, **_kwargs: np.arange(2))
    monkeypatch.setattr(inference, "iterate_dataset_batches", fake_iterate_dataset_batches)
    model = _FakeModel()
    model.train(True)

    representations, metadata = collect_representations(
        model,
        tmp_path,
        tmp_path,
        "validation",
        ["encoder.place_codes"],
        torch.device("cpu"),
        batch_size=8,
        max_episodes=1,
    )

    assert captured_iterator_kwargs["max_episodes"] == 1
    assert model.forward_training_states == [False]
    assert model.forward_inference_mode_states == [True]
    assert model.training is True
    assert representations["encoder.place_codes"].shape == (2, 4, 3)
    assert metadata["valid_steps"].shape == (2, 4)


def test_collect_representations_collects_requested_latent_into_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_iterate_dataset_batches(*args, **kwargs):
        del args, kwargs
        yield {
            "position_xy": torch.zeros((2, 4, 2), dtype=torch.float32),
            "valid_steps": torch.ones((2, 4), dtype=torch.bool),
            "latent": torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5),
        }

    monkeypatch.setattr(inference, "load_split_indices", lambda *_args, **_kwargs: np.arange(2))
    monkeypatch.setattr(inference, "iterate_dataset_batches", fake_iterate_dataset_batches)

    _representations, metadata = collect_representations(
        _FakeModel(),
        tmp_path,
        tmp_path,
        "validation",
        ["encoder.place_codes"],
        torch.device("cpu"),
        batch_size=8,
        include_batch_keys=["latent"],
        max_episodes=1,
    )

    assert "latent" in metadata
    assert metadata["latent"].shape == (2, 4, 5)


def test_collect_action_representations_without_position_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_iterator_kwargs: dict[str, object] = {}

    def fake_iterate_dataset_batches(*args, **kwargs):
        del args
        captured_iterator_kwargs.update(kwargs)
        yield {
            "actions": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "valid_steps": torch.ones((1, 4), dtype=torch.bool),
        }

    monkeypatch.setattr(inference, "load_split_indices", lambda *_args, **_kwargs: np.arange(1))
    monkeypatch.setattr(inference, "iterate_dataset_batches", fake_iterate_dataset_batches)

    representations, metadata = collect_representations(
        _FakeModel(),
        tmp_path,
        tmp_path,
        "validation",
        ["encoder.place_codes"],
        torch.device("cpu"),
        batch_size=1,
        observation_source="action",
    )

    assert captured_iterator_kwargs["observation_source"] == "action"
    assert representations["encoder.place_codes"].shape == (1, 4, 3)
    assert metadata["actions"].shape == (1, 4)
    assert "position_xy" not in metadata


def test_collect_representations_infers_action_source_from_dag_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_iterator_kwargs: dict[str, object] = {}

    def fake_iterate_dataset_batches(*args, **kwargs):
        del args
        captured_iterator_kwargs.update(kwargs)
        yield {
            "actions": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "valid_steps": torch.ones((1, 4), dtype=torch.bool),
        }

    monkeypatch.setattr(inference, "load_split_indices", lambda *_args, **_kwargs: np.arange(1))
    monkeypatch.setattr(inference, "iterate_dataset_batches", fake_iterate_dataset_batches)
    model = _FakeModel()
    model.root = SimpleNamespace(
        components=SimpleNamespace(
            config=SimpleNamespace(
                inputs=SimpleNamespace(observation_source="action"),
            ),
        ),
    )

    collect_representations(
        model,
        tmp_path,
        tmp_path,
        "test",
        ["encoder.place_codes"],
        torch.device("cpu"),
        batch_size=1,
    )

    assert captured_iterator_kwargs["observation_source"] == "action"


def _latent_batches(read_counter: list[int], *, with_rgb: bool = False):
    def fake_iterate_dataset_batches(*args, **kwargs):
        del args, kwargs
        read_counter.append(1)
        for offset in (0.0, 10.0):
            batch = {
                "latent": torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5) + offset,
                "actions": torch.zeros((2, 4), dtype=torch.long),
                "position_xy": torch.full((2, 4, 2), offset, dtype=torch.float32),
                "valid_steps": torch.ones((2, 4), dtype=torch.bool),
            }
            if with_rgb:
                batch["rgb"] = torch.zeros((2, 4, 3, 2, 2), dtype=torch.float32)
            yield batch

    return fake_iterate_dataset_batches


def test_input_batch_cache_replays_the_dataset_read_across_passes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    read_counter: list[int] = []
    monkeypatch.setattr(inference, "load_split_indices", lambda *_args, **_kwargs: np.arange(4))
    monkeypatch.setattr(inference, "iterate_dataset_batches", _latent_batches(read_counter))
    cache = InputBatchCache()

    def collect():
        return collect_representations(
            _FakeModel(),
            tmp_path,
            tmp_path,
            "test",
            ["encoder.place_codes"],
            torch.device("cpu"),
            batch_size=2,
            include_batch_keys=["latent"],
            observation_source="latent",
            input_batch_cache=cache,
        )

    first_representations, first_metadata = collect()
    second_representations, second_metadata = collect()

    assert read_counter == [1]
    assert cache.resident_bytes() > 0
    for key, expected in first_metadata.items():
        assert np.array_equal(second_metadata[key], expected), key
    assert np.array_equal(
        second_representations["encoder.place_codes"],
        first_representations["encoder.place_codes"],
    )
    cache.clear()
    assert cache.resident_bytes() == 0


def test_input_batch_cache_never_holds_rgb(monkeypatch, tmp_path: Path) -> None:
    read_counter: list[int] = []
    monkeypatch.setattr(inference, "load_split_indices", lambda *_args, **_kwargs: np.arange(4))
    monkeypatch.setattr(
        inference, "iterate_dataset_batches", _latent_batches(read_counter, with_rgb=True)
    )
    cache = InputBatchCache()

    for _pass in range(2):
        collect_representations(
            _FakeModel(),
            tmp_path,
            tmp_path,
            "test",
            ["encoder.place_codes"],
            torch.device("cpu"),
            batch_size=2,
            include_batch_keys=["rgb"],
            observation_source="both",
            input_batch_cache=cache,
        )

    assert read_counter == [1, 1]
    assert cache.resident_bytes() == 0
