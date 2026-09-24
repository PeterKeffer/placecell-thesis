from __future__ import annotations

import errno
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from placecell_research.datasets import zarr_io
from placecell_research.datasets.batch_iterator import iterate_dataset_batches
from placecell_research.datasets.dataset import TrajectoryDataset
from placecell_research.datasets.schema import (
    ACTIONS_KEY,
    HEADING_KEY,
    KINEMATICS_KEY,
    LATENT_KEY,
    LENGTH_KEY,
    POSITION_KEY,
    RGB_KEY,
    SOURCE_SEED_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    VALID_MASK_KEY,
    DatasetSummary,
)
from placecell_research.datasets.zarr_io import (
    _default_compressor,
    _episodes_per_chunk,
    _raise_if_output_disk_too_small,
    require_zarr,
    save_dataset_zarr_atomic,
    save_dataset_zarr_streaming_atomic,
    validate_dataset_store,
)


def test_raise_if_output_disk_too_small_reports_actionable_message(
    tmp_path: Path, monkeypatch
) -> None:
    class _DiskUsage:
        free = 2 * 1024**3

    monkeypatch.setattr(
        "placecell_research.datasets.zarr_io.shutil.disk_usage", lambda path: _DiskUsage()
    )

    with pytest.raises(RuntimeError, match="tracking.artifact_root"):
        _raise_if_output_disk_too_small(tmp_path / "dataset.zarr", estimated_bytes=5 * 1024**3)


def test_save_dataset_zarr_streaming_atomic_rewrites_no_space_left_error(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakeArray:
        def __setitem__(self, key, value) -> None:
            del key, value
            raise OSError(errno.ENOSPC, "No space left on device")

    class _FakeGroup:
        def require_group(self, name: str):
            del name
            return self

        def create_dataset(self, *args, **kwargs) -> None:
            del args, kwargs

        def __getitem__(self, key: str):
            del key
            return _FakeArray()

    class _FakeZarr:
        @staticmethod
        def open_group(path: str, mode: str):
            del path, mode
            return _FakeGroup()

    class _DiskUsage:
        free = 3 * 1024**3

    monkeypatch.setattr(
        "placecell_research.datasets.zarr_io.require_zarr",
        lambda: (_FakeZarr(), object()),
    )
    monkeypatch.setattr(
        "placecell_research.datasets.zarr_io._default_compressor",
        lambda: object(),
    )
    monkeypatch.setattr(
        "placecell_research.datasets.zarr_io.shutil.disk_usage", lambda path: _DiskUsage()
    )

    summary = DatasetSummary(
        env_id="MiniWorld-WallGapAsymLarge-v0",
        num_episodes=1,
        episode_length=2,
        num_actions=3,
        modalities=["rgb"],
    )
    array_specs = {"observations/rgb": ((1, 2, 3, 4, 4), np.dtype(np.uint8))}
    episode_arrays = [
        (0, {"observations/rgb": np.zeros((2, 3, 4, 4), dtype=np.uint8)}),
    ]

    with pytest.raises(RuntimeError, match="ran out of disk space"):
        save_dataset_zarr_streaming_atomic(
            tmp_path / "dataset.zarr",
            array_specs,
            episode_arrays,
            summary,
        )


def test_validate_dataset_store_skips_missing_zarr_branches(tmp_path: Path, monkeypatch) -> None:
    class _FakeGroup:
        def __contains__(self, key: str) -> bool:
            return key == "episode_metadata"

        def __getitem__(self, key: str):
            if key == "episode_metadata":
                return {}
            raise KeyError(key)

    class _FakeZarr:
        @staticmethod
        def open_group(path: str, mode: str):
            del path, mode
            return _FakeGroup()

    monkeypatch.setattr(
        "placecell_research.datasets.zarr_io.require_zarr",
        lambda: (_FakeZarr(), object()),
    )

    validate_dataset_store(tmp_path / "dataset.zarr")


_EPISODES = 6
_STEPS = 4
_LATENT_DIM = 3


def _episode_metadata_arrays(generator) -> dict[str, np.ndarray]:
    return {
        ACTIONS_KEY: generator.integers(0, 3, (_EPISODES, _STEPS), dtype=np.int64),
        POSITION_KEY: generator.normal(size=(_EPISODES, _STEPS, 2)).astype(np.float32),
        HEADING_KEY: generator.normal(size=(_EPISODES, _STEPS)).astype(np.float32),
        KINEMATICS_KEY: generator.normal(size=(_EPISODES, _STEPS, 4)).astype(np.float32),
        VALID_MASK_KEY: np.ones((_EPISODES, _STEPS), dtype=bool),
        LENGTH_KEY: np.full((_EPISODES,), _STEPS, dtype=np.int64),
        TERMINATED_KEY: np.zeros((_EPISODES,), dtype=bool),
        TRUNCATED_KEY: np.ones((_EPISODES,), dtype=bool),
        SOURCE_SEED_KEY: np.arange(_EPISODES, dtype=np.int64),
    }


def _latent_arrays() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(7)
    return {
        LATENT_KEY: generator.normal(size=(_EPISODES, _STEPS, _LATENT_DIM)).astype(np.float32),
        **_episode_metadata_arrays(generator),
    }


def _rgb_arrays() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(11)
    return {
        RGB_KEY: generator.integers(0, 256, (_EPISODES, _STEPS, 3, 8, 8), dtype=np.uint8),
        **_episode_metadata_arrays(generator),
    }


def _summary(modality: str) -> DatasetSummary:
    return DatasetSummary(
        env_id="test-chunking",
        num_episodes=_EPISODES,
        episode_length=_STEPS,
        num_actions=3,
        modalities=[modality],
    )


def _chunk_shapes(dataset_zarr: Path) -> dict[str, tuple[int, ...]]:
    zarr, _ = require_zarr()
    group = zarr.open_group(str(dataset_zarr), mode="r")
    shapes: dict[str, tuple[int, ...]] = {}
    for branch_name in group.group_keys():
        for array_name in group[branch_name].array_keys():
            key = f"{branch_name}/{array_name}"
            shapes[key] = tuple(int(size) for size in group[key].chunks)
    return shapes


def test_episodes_per_chunk_matches_the_filesystem_stripe_arithmetic() -> None:
    """The real 8,192 x 2,048 encoded dataset geometry."""
    latent_shape = (8192, 2048, 64)
    assert _episodes_per_chunk(
        LATENT_KEY, latent_shape, np.dtype(np.float32).itemsize
    ) == 16

    assert _episodes_per_chunk(
        LENGTH_KEY, (8192,), np.dtype(np.int32).itemsize
    ) == 8192

    assert _episodes_per_chunk(
        RGB_KEY, (8192, 2048, 3, 64, 64), np.dtype(np.uint8).itemsize
    ) == 1
    assert _episodes_per_chunk(
        KINEMATICS_KEY, (8192, 2048, 4), np.dtype(np.float32).itemsize
    ) == 256


def test_latent_store_groups_episodes_into_stripe_sized_chunks(tmp_path: Path) -> None:
    save_dataset_zarr_atomic(tmp_path / "dataset.zarr", _latent_arrays(), _summary("latent"))

    chunks = _chunk_shapes(tmp_path / "dataset.zarr")
    assert chunks[LATENT_KEY] == (_EPISODES, _STEPS, _LATENT_DIM)
    assert chunks[LENGTH_KEY] == (_EPISODES,)
    for key, shape in chunks.items():
        assert shape[0] == _EPISODES, f"{key} should hold every episode in one chunk"


def test_rgb_store_keeps_one_episode_per_chunk(tmp_path: Path) -> None:
    """Pins the lazy/random read path: grouping RGB would fetch N episodes to serve one."""
    save_dataset_zarr_atomic(tmp_path / "dataset.zarr", _rgb_arrays(), _summary("rgb"))

    chunks = _chunk_shapes(tmp_path / "dataset.zarr")
    assert chunks[RGB_KEY] == (1, _STEPS, 3, 8, 8)
    for key in (ACTIONS_KEY, POSITION_KEY, HEADING_KEY, KINEMATICS_KEY, VALID_MASK_KEY):
        assert chunks[key][0] == _EPISODES, f"{key} should group companion episodes"
    for key in (LENGTH_KEY, TERMINATED_KEY, TRUNCATED_KEY, SOURCE_SEED_KEY):
        assert chunks[key] == (_EPISODES,)


def _write_legacy_layout(dataset_zarr: Path, arrays: dict[str, np.ndarray]) -> None:
    """One chunk file per episode, for every array."""
    zarr, _ = require_zarr()
    group = zarr.open_group(str(dataset_zarr), mode="w")
    compressor = _default_compressor()
    for key, array in arrays.items():
        branch, name = key.rsplit("/", 1)
        legacy_chunks = (1,) if array.ndim == 1 else (1, *array.shape[1:])
        group.require_group(branch).create_dataset(
            name, data=array, chunks=legacy_chunks, compressor=compressor
        )


def _read_every_episode(artifact_dir: Path, *, include_rgb: bool) -> list[dict[str, np.ndarray]]:
    dataset = TrajectoryDataset(
        artifact_dir, include_rgb=include_rgb, include_latent=not include_rgb
    )
    return [
        {
            name: value.numpy()
            for name, value in dataset[index].items()
            if value is not None
        }
        for index in range(len(dataset))
    ]


@pytest.mark.parametrize("include_rgb", [False, True])
def test_reader_is_indifferent_to_chunk_layout(tmp_path: Path, include_rgb: bool) -> None:
    """Existing artifacts keep their per-episode layout, so the reader may not assume one."""
    arrays = _rgb_arrays() if include_rgb else _latent_arrays()
    summary = _summary("rgb" if include_rgb else "latent")

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    _write_legacy_layout(legacy_dir / "dataset.zarr", arrays)
    (legacy_dir / "manifest_summary.json").write_text(json.dumps(summary.to_dict()))

    current_dir = tmp_path / "current"
    save_dataset_zarr_atomic(current_dir / "dataset.zarr", arrays, summary)

    legacy_episodes = _read_every_episode(legacy_dir, include_rgb=include_rgb)
    current_episodes = _read_every_episode(current_dir, include_rgb=include_rgb)

    assert len(legacy_episodes) == len(current_episodes) == _EPISODES
    for legacy, current in zip(legacy_episodes, current_episodes, strict=False):
        assert legacy.keys() == current.keys()
        for name in legacy:
            np.testing.assert_array_equal(legacy[name], current[name], err_msg=name)


_SELECTED_EPISODES = [0, 1, 3, 5]


def test_preload_and_batch_sweep_are_indifferent_to_chunk_layout(
    tmp_path: Path, monkeypatch
) -> None:
    """The two paths that actually read an encoded dataset: RAM preload and eval/analysis sweep."""
    monkeypatch.setattr(zarr_io, "TARGET_CHUNK_RAW_BYTES", 2 * _STEPS * _LATENT_DIM * 4)
    arrays = _latent_arrays()
    summary = _summary("latent")

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    _write_legacy_layout(legacy_dir / "dataset.zarr", arrays)
    (legacy_dir / "manifest_summary.json").write_text(json.dumps(summary.to_dict()))

    current_dir = tmp_path / "current"
    save_dataset_zarr_atomic(current_dir / "dataset.zarr", arrays, summary)
    assert _chunk_shapes(current_dir / "dataset.zarr")[LATENT_KEY][0] == 2

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "split_indices.json").write_text(
        json.dumps({"test_episode_ids": _SELECTED_EPISODES})
    )

    def preloaded(directory: Path) -> list[dict[str, np.ndarray]]:
        dataset = TrajectoryDataset(
            directory,
            episode_ids=_SELECTED_EPISODES,
            include_rgb=False,
            include_latent=True,
            preload_to_memory=True,
        )
        return [
            {name: value.numpy() for name, value in dataset[index].items() if value is not None}
            for index in range(len(dataset))
        ]

    def swept(directory: Path) -> list[dict[str, np.ndarray]]:
        return [
            {name: value.numpy() for name, value in batch.items()}
            for batch in iterate_dataset_batches(
                directory,
                split_dir,
                "test",
                batch_size=3,
                device=torch.device("cpu"),
                observation_source="latent",
            )
        ]

    for read_split in (preloaded, swept):
        legacy_read = read_split(legacy_dir)
        current_read = read_split(current_dir)
        assert len(legacy_read) == len(current_read)
        for legacy, current in zip(legacy_read, current_read, strict=False):
            assert legacy.keys() == current.keys()
            for name in legacy:
                np.testing.assert_array_equal(legacy[name], current[name], err_msg=name)

    np.testing.assert_array_equal(
        np.stack([episode["latent"] for episode in preloaded(current_dir)]),
        arrays[LATENT_KEY][_SELECTED_EPISODES],
    )


@pytest.mark.parametrize("include_rgb", [False, True])
def test_streaming_writer_buffers_whole_chunks_without_changing_values(
    tmp_path: Path, include_rgb: bool
) -> None:
    """The grouped chunks are filled one episode at a time, so pin the buffered result."""
    arrays = _rgb_arrays() if include_rgb else _latent_arrays()
    summary = _summary("rgb" if include_rgb else "latent")
    specs = {key: (tuple(array.shape), array.dtype) for key, array in arrays.items()}

    streamed_dir = tmp_path / "streamed"
    streamed_dir.mkdir()
    save_dataset_zarr_streaming_atomic(
        streamed_dir / "dataset.zarr",
        specs,
        (
            (index, {key: array[index] for key, array in arrays.items()})
            for index in range(_EPISODES)
        ),
        summary,
    )

    whole_dir = tmp_path / "whole"
    save_dataset_zarr_atomic(whole_dir / "dataset.zarr", arrays, summary)

    assert _chunk_shapes(streamed_dir / "dataset.zarr") == _chunk_shapes(whole_dir / "dataset.zarr")
    zarr, _ = require_zarr()
    streamed = zarr.open_group(str(streamed_dir / "dataset.zarr"), mode="r")
    for key, expected in arrays.items():
        np.testing.assert_array_equal(np.asarray(streamed[key]), expected, err_msg=key)


def test_streaming_writer_flushes_a_partial_chunk_on_a_gap(tmp_path: Path) -> None:
    """A non-consecutive episode index must not silently land in the wrong slot."""
    arrays = _latent_arrays()
    summary = _summary("latent")
    specs = {key: (tuple(array.shape), array.dtype) for key, array in arrays.items()}
    write_order = [0, 1, 4, 5, 2, 3]

    out_dir = tmp_path / "out_of_order"
    out_dir.mkdir()
    save_dataset_zarr_streaming_atomic(
        out_dir / "dataset.zarr",
        specs,
        ((index, {key: array[index] for key, array in arrays.items()}) for index in write_order),
        summary,
    )

    zarr, _ = require_zarr()
    written = zarr.open_group(str(out_dir / "dataset.zarr"), mode="r")
    for key, expected in arrays.items():
        np.testing.assert_array_equal(np.asarray(written[key]), expected, err_msg=key)
