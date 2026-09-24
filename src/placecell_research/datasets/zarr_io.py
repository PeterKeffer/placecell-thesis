"""Zarr dataset IO with numeric-only array storage."""

from __future__ import annotations

import errno
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schema import ACTIONS_KEY, CONTINUOUS_ACTIONS_KEY, NUMERIC_ARRAY_KEYS, RGB_KEY, DatasetSummary


def _is_supported_array_key(key: str) -> bool:
    if key in NUMERIC_ARRAY_KEYS:
        return True
    if key.startswith("observations/"):
        suffix = key.split("/", 1)[1].strip()
        return bool(suffix)
    return False


def require_zarr() -> Any:
    try:
        import zarr  # type: ignore
        from numcodecs import Blosc  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("zarr and numcodecs are required for dataset IO") from exc
    return zarr, Blosc


def _default_compressor() -> Any:
    _, blosc = require_zarr()
    return blosc(cname="zstd", clevel=3, shuffle=blosc.BITSHUFFLE)


TARGET_CHUNK_RAW_BYTES = 8 * 1024 * 1024


def _episodes_per_chunk(
    key: str,
    shape: tuple[int, ...],
    itemsize: int,
) -> int:
    """How many episodes share one chunk file."""
    if key == RGB_KEY:
        return 1
    per_episode_bytes = max(1, int(np.prod(shape[1:], dtype=np.int64)) * int(itemsize))
    episodes = TARGET_CHUNK_RAW_BYTES // per_episode_bytes
    return int(min(max(1, episodes), max(1, int(shape[0]))))


def _chunks_for_shape(
    key: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> tuple[int, ...] | None:
    if len(shape) == 0:
        return None
    episodes = _episodes_per_chunk(key, shape, np.dtype(dtype).itemsize)
    return (episodes, *[int(size) for size in shape[1:]])


def _ensure_numeric_array(key: str, array: np.ndarray) -> np.ndarray:
    if not _is_supported_array_key(key):
        raise KeyError(f"Unsupported dataset array key: {key}")
    if array.dtype.kind in {"O", "U", "S"}:
        raise TypeError(
            f"Dataset Zarr stores numeric arrays only. Key '{key}' has dtype {array.dtype}."
        )
    return array


def _ensure_numeric_dtype(key: str, dtype: np.dtype[Any]) -> np.dtype[Any]:
    if not _is_supported_array_key(key):
        raise KeyError(f"Unsupported dataset array key: {key}")
    if dtype.kind in {"O", "U", "S"}:
        raise TypeError(f"Dataset Zarr stores numeric arrays only. Key '{key}' has dtype {dtype}.")
    return dtype


def _branch_and_name(key: str) -> tuple[str, str]:
    branch, name = key.rsplit("/", 1)
    return branch, name


def _finalize_dataset_publish(
    temp_path: Path, output_path: Path, manifest_summary: DatasetSummary
) -> None:
    validate_dataset_store(temp_path)
    manifest_path = temp_path.parent / f"{temp_path.name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_summary.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    temp_path.replace(output_path)
    final_manifest = output_path.parent / "manifest_summary.json"
    if final_manifest.exists():
        final_manifest.unlink()
    manifest_path.replace(final_manifest)


def _estimated_storage_bytes_for_arrays(
    arrays: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
) -> int:
    estimated_bytes = 0
    for key, (shape, raw_dtype) in arrays.items():
        dtype = np.dtype(raw_dtype)
        raw_bytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if key == RGB_KEY and dtype == np.dtype(np.uint8):
            estimated_bytes += int(raw_bytes * 0.40)
        elif dtype.kind == "f":
            estimated_bytes += int(raw_bytes * 0.80)
        else:
            estimated_bytes += int(raw_bytes * 0.60)
    return estimated_bytes


def _raise_if_output_disk_too_small(output_path: Path, estimated_bytes: int) -> None:
    if estimated_bytes <= 0:
        return
    free_bytes = shutil.disk_usage(str(output_path.parent)).free
    if free_bytes >= estimated_bytes:
        return
    free_gb = free_bytes / (1024**3)
    estimated_gb = estimated_bytes / (1024**3)
    raise RuntimeError(
        "Insufficient free disk space for dataset write. "
        f"Need roughly {estimated_gb:.1f} GB free at '{output_path.parent}', "
        f"but only {free_gb:.1f} GB is available. "
        "Move tracking.artifact_root to a larger volume or reduce dataset size."
    )


def _rewrite_no_space_left_error(output_path: Path, exc: OSError) -> RuntimeError:
    free_bytes = shutil.disk_usage(str(output_path.parent)).free
    free_gb = free_bytes / (1024**3)
    return RuntimeError(
        f"Dataset write ran out of disk space at '{output_path.parent}' "
        f"while creating '{output_path.name}'. "
        f"Remaining free space was about {free_gb:.1f} GB. "
        "Move tracking.artifact_root to a larger volume or reduce dataset size."
    )


def save_dataset_zarr_atomic(
    output_path: Path,
    arrays: Mapping[str, np.ndarray],
    manifest_summary: DatasetSummary,
) -> None:
    """Write dataset arrays to a temp directory, validate, then atomically publish."""
    zarr, _ = require_zarr()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _raise_if_output_disk_too_small(
        output_path,
        _estimated_storage_bytes_for_arrays(
            {
                key: (tuple(int(size) for size in np.asarray(array).shape), np.asarray(array).dtype)
                for key, array in arrays.items()
            }
        ),
    )
    temp_path = output_path.with_name(f"{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    compressor = _default_compressor()

    try:
        group = zarr.open_group(str(temp_path), mode="w")
        for key, raw_array in arrays.items():
            array = _ensure_numeric_array(key, np.asarray(raw_array))
            branch, name = _branch_and_name(key)
            dataset = group.require_group(branch)
            chunks = _chunks_for_shape(
                key,
                tuple(int(size) for size in array.shape),
                array.dtype,
            )
            dataset.create_dataset(name, data=array, chunks=chunks, compressor=compressor)
        _finalize_dataset_publish(temp_path, output_path, manifest_summary)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise _rewrite_no_space_left_error(output_path, exc) from exc
        raise
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def save_dataset_zarr_streaming_atomic(
    output_path: Path,
    array_specs: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
    episode_arrays: Iterable[tuple[int, Mapping[str, np.ndarray]]],
    manifest_summary: DatasetSummary,
) -> None:
    """Write dataset arrays episode-by-episode to a temp Zarr store, then atomically publish."""
    with DatasetZarrStreamWriter(output_path, array_specs, manifest_summary) as writer:
        for episode_index, arrays in episode_arrays:
            writer.write_episode(episode_index, arrays)
        writer.finalize()


@dataclass
class _PendingChunk:
    """Episodes buffered for one not-yet-complete chunk of one array."""

    start_index: int
    episodes: list[np.ndarray] = field(default_factory=list)

    @property
    def stop_index(self) -> int:
        return self.start_index + len(self.episodes)


class DatasetZarrStreamWriter:
    """Incrementally write one episode at a time to a temp Zarr store."""

    def __init__(
        self,
        output_path: Path,
        array_specs: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
        manifest_summary: DatasetSummary,
    ) -> None:
        self.output_path = Path(output_path)
        self.array_specs = dict(array_specs)
        self.manifest_summary = manifest_summary
        self.temp_path = self.output_path.with_name(
            f"{self.output_path.stem}.tmp.{os.getpid()}{self.output_path.suffix}"
        )
        self._group = None
        self._published = False
        self._episodes_per_chunk: dict[str, int] = {}
        self._pending_chunks: dict[str, _PendingChunk] = {}

    def __enter__(self) -> DatasetZarrStreamWriter:
        return self.open()

    def open(self) -> DatasetZarrStreamWriter:
        if self._group is not None:
            return self
        zarr, _ = require_zarr()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        _raise_if_output_disk_too_small(
            self.output_path,
            _estimated_storage_bytes_for_arrays(self.array_specs),
        )
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        compressor = _default_compressor()
        try:
            self._group = zarr.open_group(str(self.temp_path), mode="w")
            for key, (shape, raw_dtype) in self.array_specs.items():
                dtype = _ensure_numeric_dtype(key, np.dtype(raw_dtype))
                branch, name = _branch_and_name(key)
                dataset = self._group.require_group(branch)
                array_shape = tuple(int(size) for size in shape)
                chunks = _chunks_for_shape(key, array_shape, dtype)
                self._episodes_per_chunk[key] = 1 if chunks is None else int(chunks[0])
                dataset.create_dataset(
                    name,
                    shape=array_shape,
                    dtype=dtype,
                    chunks=chunks,
                    compressor=compressor,
                )
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise _rewrite_no_space_left_error(self.output_path, exc) from exc
            raise
        return self

    def write_episode(
        self,
        episode_index: int,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        if self._group is None:
            raise RuntimeError("DatasetZarrStreamWriter must be opened before writing episodes.")
        try:
            for key, raw_array in arrays.items():
                array = _ensure_numeric_array(key, np.asarray(raw_array))
                if self._episodes_per_chunk.get(key, 1) == 1:
                    self._group[key][int(episode_index)] = array
                else:
                    self._buffer_episode(key, int(episode_index), array)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise _rewrite_no_space_left_error(self.output_path, exc) from exc
            raise

    def _buffer_episode(self, key: str, episode_index: int, array: np.ndarray) -> None:
        """Hold episodes until a whole chunk is ready."""
        pending = self._pending_chunks.get(key)
        if pending is not None and pending.stop_index != episode_index:
            self._flush_pending_chunk(key)
            pending = None
        if pending is None:
            pending = _PendingChunk(start_index=episode_index)
            self._pending_chunks[key] = pending
        pending.episodes.append(array)
        if pending.stop_index % self._episodes_per_chunk[key] == 0:
            self._flush_pending_chunk(key)

    def _flush_pending_chunk(self, key: str) -> None:
        pending = self._pending_chunks.pop(key, None)
        if pending is None:
            return
        self._group[key][pending.start_index : pending.stop_index] = np.stack(pending.episodes)

    def finalize(self) -> None:
        if self._group is None:
            raise RuntimeError("DatasetZarrStreamWriter must be opened before finalize().")
        if self._published:
            return
        try:
            for key in list(self._pending_chunks):
                self._flush_pending_chunk(key)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise _rewrite_no_space_left_error(self.output_path, exc) from exc
            raise
        _finalize_dataset_publish(self.temp_path, self.output_path, self.manifest_summary)
        self._published = True

    def close(self) -> None:
        self._group = None
        self._pending_chunks.clear()
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path, ignore_errors=True)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._group = None
        self._pending_chunks.clear()
        if not self._published and self.temp_path.exists():
            shutil.rmtree(self.temp_path, ignore_errors=True)


def validate_dataset_store(path: Path) -> None:
    """Open and touch every stored array to catch malformed writes."""
    zarr, _ = require_zarr()
    group = zarr.open_group(str(path), mode="r")
    for key in NUMERIC_ARRAY_KEYS:
        branch, name = key.rsplit("/", 1)
        if branch not in group or name not in group[branch]:
            continue
        array = group[key]
        _ = array.shape
        if key == ACTIONS_KEY:
            if array.ndim != 2:
                raise ValueError("actions/discrete must be [N, T].")
        if key == RGB_KEY and array.ndim != 5:
            raise ValueError("observations/rgb must be [N, T, C, H, W].")
        if key == CONTINUOUS_ACTIONS_KEY and (array.ndim != 3 or array.shape[-1] != 2):
            raise ValueError("actions/continuous must be [N, T, 2].")


def load_dataset_manifest(dataset_dir: Path) -> DatasetSummary:
    """Load manifest summary stored beside the dataset."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest_summary.json"
    if not manifest_path.exists() and dataset_dir.name == "dataset.zarr":
        manifest_path = dataset_dir.parent / "manifest_summary.json"
    payload = json.loads(manifest_path.read_text())
    return DatasetSummary(**payload)
