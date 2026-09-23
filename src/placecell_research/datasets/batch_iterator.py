"""Direct batch iteration over canonical dataset artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from placecell_research.utils.angles import wrap_radians

from .staging import stage_dataset_dir
from .zarr_io import _TARGET_CHUNK_RAW_BYTES, _require_zarr

_SPLIT_ALIASES = {
    "train": ["train_episode_ids", "train"],
    "validation": ["validation_episode_ids", "val_episode_ids", "validation"],
    "test": ["test_episode_ids", "test"],
}


class _EpisodeChunkReader:
    """Retain at most one bounded episode chunk between batches."""

    def __init__(self, array: Any) -> None:
        self.array = array
        self._cached_chunk: tuple[int, np.ndarray] | None = None
        self._episodes_per_chunk = int(array.chunks[0])
        chunk_bytes = (
            min(self._episodes_per_chunk, array.shape[0])
            * int(np.prod(array.shape[1:]))
            * np.dtype(array.dtype).itemsize
        )
        self._cache_enabled = (
            self._episodes_per_chunk > 1 and chunk_bytes <= _TARGET_CHUNK_RAW_BYTES
        )

    def read(self, episode_ids: list[int]) -> np.ndarray:
        if not self._cache_enabled:
            return self.array[episode_ids]
        selection = np.asarray(episode_ids)
        if selection.ndim != 1 or selection.dtype.kind not in "iu":
            return self.array[episode_ids]
        selection = np.where(selection < 0, selection + self.array.shape[0], selection)
        if np.any((selection < 0) | (selection >= self.array.shape[0])):
            return self.array[episode_ids]
        output = np.empty(
            (len(selection), *self.array.shape[1:]), dtype=self.array.dtype, order=self.array.order
        )
        chunk_starts = selection // self._episodes_per_chunk * self._episodes_per_chunk
        for start in np.unique(chunk_starts):
            if self._cached_chunk is None or self._cached_chunk[0] != start:
                self._cached_chunk = None
                self._cached_chunk = (
                    int(start), self.array[int(start):int(start) + self._episodes_per_chunk]
                )
            rows = np.flatnonzero(chunk_starts == start)
            output[rows] = self._cached_chunk[1][selection[rows] - start]
        return output


def load_split_indices(split_directory: Path, split_name: str) -> list[int]:
    payload = json.loads((split_directory / "split_indices.json").read_text())
    for key in _SPLIT_ALIASES[split_name]:
        if key in payload:
            return list(payload[key])
    raise KeyError(f"Split '{split_name}' not found in {split_directory / 'split_indices.json'}.")


def available_split_names(split_directory: Path) -> list[str]:
    """Return canonical split names present with at least one episode."""
    payload = json.loads((split_directory / "split_indices.json").read_text())
    available_names: list[str] = []
    for split_name, aliases in _SPLIT_ALIASES.items():
        for key in aliases:
            if key in payload and payload[key]:
                available_names.append(split_name)
                break
    return available_names


def iterate_dataset_batches(
    dataset_directory: Path,
    split_directory: Path,
    split_name: str,
    batch_size: int,
    device: torch.device,
    observation_source: Literal["latent", "rgb", "action", "both"] = "both",
    max_episodes: int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Read batches directly from the canonical dataset Zarr schema."""
    zarr, _ = _require_zarr()
    dataset_group = zarr.open(str(stage_dataset_dir(dataset_directory) / "dataset.zarr"), mode="r")
    episode_ids = load_split_indices(split_directory, split_name)
    if max_episodes is not None and max_episodes > 0:
        episode_ids = episode_ids[:max(0, int(max_episodes))]
    if not episode_ids:
        raise ValueError(
            f"Split '{split_name}' in {split_directory / 'split_indices.json'} "
            "contains no episode ids."
        )

    observation_group = dataset_group.get("observations")
    latent_array = None if observation_group is None else observation_group.get("latent")
    rgb_array = None if observation_group is None else observation_group.get("rgb")
    if observation_source != "action" and latent_array is None and rgb_array is None:
        raise KeyError("Dataset must contain either observations/latent or observations/rgb.")
    if observation_source == "latent" and latent_array is None:
        raise KeyError(
            "Requested latent observations, but dataset does not contain observations/latent."
        )
    if observation_source == "rgb" and rgb_array is None:
        raise KeyError("Requested RGB observations, but dataset does not contain observations/rgb.")

    actions = dataset_group["actions"]["discrete"]
    state_group = dataset_group.get("state")
    positions = None if state_group is None else state_group.get("position_xy")
    headings = None if state_group is None else state_group.get("heading")
    kinematics = None if state_group is None else state_group.get("kinematics")
    valid_steps = dataset_group["masks"]["valid_steps"]

    readers = {
        name: _EpisodeChunkReader(array)
        for name, array in {
            "actions": actions,
            "valid_steps": valid_steps,
            "position_xy": positions,
            "heading": headings,
            "kinematics": kinematics,
            "latent": latent_array if observation_source in {"latent", "both"} else None,
            "rgb": rgb_array if observation_source in {"rgb", "both"} else None,
        }.items()
        if array is not None
    }

    for start in range(0, len(episode_ids), batch_size):
        batch_ids = episode_ids[start:start + batch_size]
        batch = {
            "actions": torch.as_tensor(
                readers["actions"].read(batch_ids), dtype=torch.long, device=device
            ),
            "valid_steps": torch.as_tensor(
                readers["valid_steps"].read(batch_ids), dtype=torch.bool, device=device
            ),
        }
        if positions is not None:
            batch["position_xy"] = torch.as_tensor(
                readers["position_xy"].read(batch_ids), dtype=torch.float32, device=device
            )
        if latent_array is not None and observation_source in {"latent", "both"}:
            batch["latent"] = torch.as_tensor(
                readers["latent"].read(batch_ids), dtype=torch.float32, device=device
            )
        if rgb_array is not None and observation_source in {"rgb", "both"}:
            batch["rgb"] = torch.as_tensor(readers["rgb"].read(batch_ids), device=device)
        if headings is not None:
            batch["heading"] = torch.as_tensor(
                wrap_radians(readers["heading"].read(batch_ids)),
                dtype=torch.float32,
                device=device,
            )
        if kinematics is not None:
            batch["kinematics"] = torch.as_tensor(
                readers["kinematics"].read(batch_ids), dtype=torch.float32, device=device
            )
        yield batch
