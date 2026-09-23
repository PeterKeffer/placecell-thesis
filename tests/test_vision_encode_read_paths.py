"""The read paths behind vision training and encode_dataset stay exact when parallelised."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from placecell_research.datasets.dataset import TrajectoryDataset
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
from placecell_research.datasets.zarr_io import _require_zarr, save_dataset_zarr_atomic
from placecell_research.utils import cpu_budget
from placecell_research.vision.builder import FrameDataset, _RawEpisodeReadDataset

_EPISODES = 9
_STEPS = 12
_FRAME_SHAPE = (3, 8, 8)


def _raw_arrays() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(23)
    valid_steps = np.ones((_EPISODES, _STEPS), dtype=bool)
    valid_steps[3, 5:] = False
    return {
        RGB_KEY: generator.integers(0, 256, (_EPISODES, _STEPS, *_FRAME_SHAPE), dtype=np.uint8),
        ACTIONS_KEY: generator.integers(0, 3, (_EPISODES, _STEPS), dtype=np.int64),
        POSITION_KEY: generator.normal(size=(_EPISODES, _STEPS, 2)).astype(np.float32),
        HEADING_KEY: generator.normal(size=(_EPISODES, _STEPS)).astype(np.float32),
        KINEMATICS_KEY: generator.normal(size=(_EPISODES, _STEPS, 4)).astype(np.float32),
        VALID_MASK_KEY: valid_steps,
        LENGTH_KEY: np.full((_EPISODES,), _STEPS, dtype=np.int64),
        TERMINATED_KEY: generator.integers(0, 2, (_EPISODES,)).astype(bool),
        TRUNCATED_KEY: generator.integers(0, 2, (_EPISODES,)).astype(bool),
        SOURCE_SEED_KEY: np.arange(_EPISODES, dtype=np.int64) * 7,
    }


def _write_raw_dataset(artifact_dir: Path, arrays: dict[str, np.ndarray]) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = DatasetSummary(
        env_id="test-read-paths",
        num_episodes=_EPISODES,
        episode_length=_STEPS,
        num_actions=3,
        modalities=["rgb"],
    )
    save_dataset_zarr_atomic(artifact_dir / "dataset.zarr", arrays, summary)
    (artifact_dir / "manifest_summary.json").write_text(json.dumps(summary.to_dict()))
    return artifact_dir


def _write_legacy_raw_dataset(artifact_dir: Path, arrays: dict[str, np.ndarray]) -> Path:
    """One chunk file per episode for every array."""
    zarr, _ = _require_zarr()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    from placecell_research.datasets.zarr_io import _default_compressor

    group = zarr.open_group(str(artifact_dir / "dataset.zarr"), mode="w")
    compressor = _default_compressor()
    for key, array in arrays.items():
        branch, name = key.rsplit("/", 1)
        chunks = (1,) if array.ndim == 1 else (1, *array.shape[1:])
        group.require_group(branch).create_dataset(
            name, data=array, chunks=chunks, compressor=compressor
        )
    summary = DatasetSummary(
        env_id="test-read-paths",
        num_episodes=_EPISODES,
        episode_length=_STEPS,
        num_actions=3,
        modalities=["rgb"],
    )
    (artifact_dir / "manifest_summary.json").write_text(json.dumps(summary.to_dict()))
    return artifact_dir


def test_allocated_cpu_count_is_zero_without_a_slurm_allocation(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert cpu_budget.allocated_cpu_count() == 0
    assert cpu_budget.parallel_read_worker_count() == 0


@pytest.mark.parametrize(
    ("allocated", "expected"),
    [("1", 0), ("2", 1), ("4", 3), ("16", 8), ("112", 8), ("", 0), ("not-a-number", 0)],
)
def test_parallel_read_worker_count_never_exceeds_the_allocation(
    monkeypatch, allocated: str, expected: int
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", allocated)
    assert cpu_budget.parallel_read_worker_count() == expected
    assert cpu_budget.parallel_read_worker_count() <= max(0, cpu_budget.allocated_cpu_count() - 1)


def test_parallel_read_worker_count_honours_a_smaller_maximum(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert cpu_budget.parallel_read_worker_count(maximum=2) == 2


@pytest.mark.parametrize("frame_cache_mode", ["memory", "disk"])
def test_threaded_frame_cache_fill_is_bitwise_identical_to_serial(
    tmp_path: Path, monkeypatch, frame_cache_mode: str
) -> None:
    first = _write_raw_dataset(tmp_path / "first", _raw_arrays())
    second = _write_raw_dataset(tmp_path / "second", _raw_arrays())

    def build() -> tuple[list, np.ndarray]:
        datasets = [
            TrajectoryDataset(path, include_rgb=True, include_latent=False)
            for path in (first, second)
        ]
        frames = FrameDataset(datasets, max_frames_per_episode=5, frame_cache_mode=frame_cache_mode)
        cached = np.stack([np.asarray(frames[index]) for index in range(len(frames))])
        return frames.frame_index, cached

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    serial_index, serial_frames = build()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    assert cpu_budget.parallel_read_worker_count() == 4
    threaded_index, threaded_frames = build()

    assert threaded_index == serial_index
    np.testing.assert_array_equal(threaded_frames, serial_frames)


def test_frame_cache_holds_the_frames_the_index_names(tmp_path: Path, monkeypatch) -> None:
    """Guards the vectorised copy: every cache row is its own episode/step frame."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    arrays = _raw_arrays()
    artifact_dir = _write_raw_dataset(tmp_path / "raw", arrays)
    dataset = TrajectoryDataset(artifact_dir, include_rgb=True, include_latent=False)
    frames = FrameDataset([dataset], max_frames_per_episode=5, frame_cache_mode="memory")

    assert len(frames) > 0
    for position, (dataset_index, episode_index, step_index) in enumerate(frames.frame_index):
        assert dataset_index == 0
        np.testing.assert_array_equal(
            np.asarray(frames[position]), arrays[RGB_KEY][episode_index, step_index]
        )


def _raw_episode_dicts(artifact_dir: Path, episode_ids: list[int] | None) -> list[dict]:
    dataset = TrajectoryDataset(
        artifact_dir, episode_ids=episode_ids, include_rgb=True, include_latent=False
    )
    reader = _RawEpisodeReadDataset(dataset)
    return [reader[index] for index in range(len(reader))]


@pytest.mark.parametrize("episode_ids", [None, [0, 2, 5, 8]])
def test_whole_array_preload_matches_the_per_episode_reads(
    tmp_path: Path, episode_ids: list[int] | None
) -> None:
    arrays = _raw_arrays()
    current = _write_raw_dataset(tmp_path / "current", arrays)
    legacy = _write_legacy_raw_dataset(tmp_path / "legacy", arrays)

    current_episodes = _raw_episode_dicts(current, episode_ids)
    legacy_episodes = _raw_episode_dicts(legacy, episode_ids)

    expected_count = _EPISODES if episode_ids is None else len(episode_ids)
    assert len(current_episodes) == len(legacy_episodes) == expected_count
    for position, (from_current, from_legacy) in enumerate(
        zip(current_episodes, legacy_episodes, strict=False)
    ):
        episode_id = position if episode_ids is None else episode_ids[position]
        assert from_current.keys() == from_legacy.keys()
        for name in from_current:
            np.testing.assert_array_equal(from_current[name], from_legacy[name], err_msg=name)
        assert int(from_current["source_seed"]) == int(arrays[SOURCE_SEED_KEY][episode_id])
        assert bool(from_current["terminated"]) == bool(arrays[TERMINATED_KEY][episode_id])


def test_whole_array_preload_engages_only_on_the_grouped_layout(tmp_path: Path) -> None:
    arrays = _raw_arrays()
    current = TrajectoryDataset(
        _write_raw_dataset(tmp_path / "current", arrays), include_rgb=True, include_latent=False
    )
    legacy = TrajectoryDataset(
        _write_legacy_raw_dataset(tmp_path / "legacy", arrays),
        include_rgb=True,
        include_latent=False,
    )

    assert set(_RawEpisodeReadDataset(current)._whole_array_reads) == {
        LENGTH_KEY,
        TERMINATED_KEY,
        TRUNCATED_KEY,
        SOURCE_SEED_KEY,
    }
    assert _RawEpisodeReadDataset(legacy)._whole_array_reads == {}


def test_parallel_reads_leave_the_latents_bitwise_identical(tmp_path: Path) -> None:
    """The claim behind defaulting read_workers on: readers move, arithmetic does not."""
    import torch

    from placecell_research.vision.autoencoder import ConvAutoEncoder
    from placecell_research.vision.builder import iter_encoded_episodes_with_model

    artifact_dir = _write_raw_dataset(tmp_path / "raw", _raw_arrays())
    torch.manual_seed(0)
    model = ConvAutoEncoder(latent_dim=4, input_shape=_FRAME_SHAPE, channels=(8, 16))

    def encode(read_workers: int) -> list[dict]:
        dataset = TrajectoryDataset(artifact_dir, include_rgb=True, include_latent=False)
        return list(
            iter_encoded_episodes_with_model(
                model,
                dataset,
                device=torch.device("cpu"),
                read_workers=read_workers,
                episode_batch_size=1,
            )
        )

    serial = encode(0)
    parallel = encode(2)

    assert len(serial) == len(parallel) == _EPISODES
    for from_serial, from_parallel in zip(serial, parallel, strict=False):
        assert from_serial.keys() == from_parallel.keys()
        for name in from_serial:
            np.testing.assert_array_equal(from_parallel[name], from_serial[name], err_msg=name)
