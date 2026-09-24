"""Lazy trajectory dataset backed by Zarr."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.schema import ExperimentConfig
from placecell_research.utils.angles import wrap_radians
from placecell_research.utils.dataloader import spawned_worker_kwargs

from .schema import (
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
)
from .staging import stage_dataset_dir
from .zarr_io import load_dataset_manifest, require_zarr


def _read_selected_episodes(array: Any, episode_selection: np.ndarray) -> np.ndarray:
    """Read each selected chunk once, preserving episode order and duplicates."""
    output = np.empty(
        (len(episode_selection), *tuple(int(size) for size in array.shape[1:])),
        dtype=np.dtype(array.dtype),
    )
    if not len(episode_selection):
        return output
    episodes_per_chunk = int(array.chunks[0])
    chunk_ids = episode_selection // episodes_per_chunk
    order = np.argsort(chunk_ids, kind="stable")
    boundaries = np.flatnonzero(np.diff(chunk_ids[order])) + 1
    for rows in np.split(order, boundaries):
        source_start = int(chunk_ids[rows[0]]) * episodes_per_chunk
        source_stop = min(source_start + episodes_per_chunk, int(array.shape[0]))
        chunk = np.asarray(array[source_start:source_stop])
        output[rows] = chunk[episode_selection[rows] - source_start]
    return output


@dataclass
class EpisodeBatch:
    """One episode worth of sequence data."""

    rgb: torch.Tensor | None
    latent: torch.Tensor | None
    actions: torch.Tensor
    position_xy: torch.Tensor | None
    heading: torch.Tensor | None
    kinematics: torch.Tensor | None
    valid_steps: torch.Tensor
    episode_length: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    source_seed: torch.Tensor
    episode_index: torch.Tensor


class TrajectoryDataset(Dataset):
    """Lazily load sequences by episode id."""

    def __init__(
        self,
        dataset_dir: Path,
        episode_ids: list[int] | None = None,
        include_rgb: bool = True,
        include_latent: bool = True,
        preload_to_memory: bool = False,
    ) -> None:
        self.dataset_dir = stage_dataset_dir(dataset_dir)
        self.episode_ids = episode_ids
        self.include_rgb = include_rgb
        self.include_latent = include_latent
        self.preload_to_memory = bool(preload_to_memory)
        self._group = None
        self._preloaded_arrays: dict[str, np.ndarray] | None = None
        self.manifest_summary = load_dataset_manifest(self.dataset_dir)
        if self.preload_to_memory:
            self._preload_arrays()

    def _open(self) -> Any:
        if self._group is None:
            zarr, _ = require_zarr()
            self._group = zarr.open_group(str(self.dataset_dir / "dataset.zarr"), mode="r")
        return self._group

    @property
    def num_actions(self) -> int:
        return int(self.manifest_summary.num_actions)

    def __len__(self) -> int:
        if self._preloaded_arrays is not None and LENGTH_KEY in self._preloaded_arrays:
            return int(self._preloaded_arrays[LENGTH_KEY].shape[0])
        group = self._open()
        total = int(group[LENGTH_KEY].shape[0])
        return total if self.episode_ids is None else len(self.episode_ids)

    def _episode_id(self, index: int) -> int:
        return index if self.episode_ids is None else int(self.episode_ids[index])

    def _read(self, key: str, episode_id: int) -> np.ndarray:
        group = self._open()
        return np.asarray(group[key][episode_id])

    def _read_frame(self, key: str, episode_id: int, step_index: int) -> np.ndarray:
        group = self._open()
        return np.asarray(group[key][episode_id, step_index])

    def _keys_to_preload(self) -> list[str]:
        keys = [
            ACTIONS_KEY,
            POSITION_KEY,
            HEADING_KEY,
            KINEMATICS_KEY,
            VALID_MASK_KEY,
            LENGTH_KEY,
            TERMINATED_KEY,
            TRUNCATED_KEY,
            SOURCE_SEED_KEY,
        ]
        if self.include_rgb:
            keys.append(RGB_KEY)
        if self.include_latent:
            keys.append(LATENT_KEY)
        return keys

    def _preload_arrays(self) -> None:
        group = self._open()
        episode_selection = (
            None if self.episode_ids is None else np.asarray(self.episode_ids, dtype=np.int64)
        )
        preloaded_arrays: dict[str, np.ndarray] = {}
        for key in self._keys_to_preload():
            branch, name = key.rsplit("/", 1)
            if branch not in group or name not in group[branch]:
                continue
            if episode_selection is not None:
                array = _read_selected_episodes(group[key], episode_selection)
            else:
                array = np.asarray(group[key])
            preloaded_arrays[key] = array
        self._preloaded_arrays = preloaded_arrays

    def _read_for_index(self, key: str, index: int) -> np.ndarray:
        if self._preloaded_arrays is not None and key in self._preloaded_arrays:
            return np.asarray(self._preloaded_arrays[key][index])
        return self._read(key, self._episode_id(index))

    def _has_key(self, key: str) -> bool:
        if self._preloaded_arrays is not None and key in self._preloaded_arrays:
            return True
        group = self._open()
        branch, name = key.rsplit("/", 1)
        return branch in group and name in group[branch]

    def has_array(self, key: str) -> bool:
        return self._has_key(key)

    def read_episode_array(self, index: int, key: str) -> np.ndarray:
        return self._read_for_index(key, index)

    def read_rgb_frame(self, index: int, step_index: int) -> np.ndarray:
        if not self._has_key(RGB_KEY):
            raise KeyError(f"Dataset {self.dataset_dir} does not contain {RGB_KEY}.")
        if self._preloaded_arrays is not None and RGB_KEY in self._preloaded_arrays:
            return np.asarray(self._preloaded_arrays[RGB_KEY][index, int(step_index)])
        return self._read_frame(RGB_KEY, self._episode_id(index), int(step_index))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | None]:
        episode_id = self._episode_id(index)
        rgb = (
            self._read_for_index(RGB_KEY, index)
            if self.include_rgb and self._has_key(RGB_KEY)
            else None
        )
        latent = (
            self._read_for_index(LATENT_KEY, index)
            if self.include_latent and self._has_key(LATENT_KEY)
            else None
        )
        sample: dict[str, torch.Tensor | None] = {
            "rgb": torch.from_numpy(rgb) if rgb is not None else None,
            "latent": torch.from_numpy(latent).float() if latent is not None else None,
            "actions": torch.from_numpy(self._read_for_index(ACTIONS_KEY, index)).long(),
            "valid_steps": torch.from_numpy(self._read_for_index(VALID_MASK_KEY, index)).bool(),
            "episode_length": torch.from_numpy(
                np.asarray(self._read_for_index(LENGTH_KEY, index))
            ).long(),
            "terminated": torch.from_numpy(
                np.asarray(self._read_for_index(TERMINATED_KEY, index))
            ).bool(),
            "truncated": torch.from_numpy(
                np.asarray(self._read_for_index(TRUNCATED_KEY, index))
            ).bool(),
            "source_seed": torch.from_numpy(
                np.asarray(self._read_for_index(SOURCE_SEED_KEY, index))
            ).long(),
            "episode_index": torch.tensor(episode_id, dtype=torch.long),
        }
        if self._has_key(POSITION_KEY):
            sample["position_xy"] = torch.from_numpy(
                self._read_for_index(POSITION_KEY, index)
            ).float()
        if self._has_key(HEADING_KEY):
            sample["heading"] = torch.from_numpy(
                wrap_radians(self._read_for_index(HEADING_KEY, index))
            ).float()
        if self._has_key(KINEMATICS_KEY):
            sample["kinematics"] = torch.from_numpy(
                self._read_for_index(KINEMATICS_KEY, index)
            ).float()
        return sample

    def iter_episode_ids(self) -> Iterator[int]:
        for index in range(len(self)):
            yield self._episode_id(index)


@dataclass(frozen=True)
class _WorldSampleContract:
    """The parts of a world's samples that must agree before its episodes share a batch."""

    num_actions: int
    episode_length: int
    observation_shape: tuple[int, ...] | None
    kinematics_shape: tuple[int, ...] | None


def _per_step_shape(value: torch.Tensor | None) -> tuple[int, ...] | None:
    return None if value is None else tuple(int(size) for size in value.shape[1:])


def _world_sample_contract(
    dataset_id: str,
    dataset: TrajectoryDataset,
    *,
    observation_key: str | None,
) -> _WorldSampleContract:
    """Read one world's batch contract off a real sample, the way the collator will see it."""
    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset_id!r} contributes no training episodes.")
    sample = dataset[0]
    observation = None if observation_key is None else sample[observation_key]
    if observation_key is not None and observation is None:
        raise ValueError(
            f"Dataset {dataset_id!r} is missing expected observation source '{observation_key}'."
        )
    return _WorldSampleContract(
        num_actions=dataset.num_actions,
        episode_length=int(sample["valid_steps"].shape[0]),
        observation_shape=_per_step_shape(observation),
        kinematics_shape=_per_step_shape(sample.get("kinematics")),
    )


def _require_one_shared_contract(
    world_ids: list[tuple[str, str]],
    contracts: list[_WorldSampleContract],
) -> None:
    """Refuse worlds whose samples cannot be stacked into one batch."""
    primary_dataset_id = world_ids[0][0]
    primary = contracts[0]

    def shape_text(shape: tuple[int, ...] | None) -> str:
        return "no such field" if shape is None else str(shape)

    for (dataset_id, _), contract in zip(world_ids[1:], contracts[1:], strict=False):
        for field_name, primary_value, world_value in (
            ("num_actions", primary.num_actions, contract.num_actions),
            ("episode_length", primary.episode_length, contract.episode_length),
            (
                "observation shape",
                shape_text(primary.observation_shape),
                shape_text(contract.observation_shape),
            ),
            (
                "kinematics shape",
                shape_text(primary.kinematics_shape),
                shape_text(contract.kinematics_shape),
            ),
        ):
            if primary_value != world_value:
                raise ValueError(
                    f"Multi-world training needs one shared {field_name}: dataset "
                    f"{primary_dataset_id!r} has {primary_value}, "
                    f"{dataset_id!r} has {world_value}."
                )


def collate_episodes(batch: list[dict[str, torch.Tensor | None]]) -> dict[str, torch.Tensor]:
    """Stack a batch of equal-length padded trajectories."""
    collated: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        first_value = batch[0][key]
        if first_value is None:
            continue
        collated[key] = torch.stack(
            [sample[key] for sample in batch if sample[key] is not None], dim=0
        )  # type: ignore[arg-type]
    return collated


def build_training_dataloaders(
    config: ExperimentConfig,
    artifact_root: Path | None = None,
    shuffle_seed: int | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, int]]:
    """Build train/validation dataloaders from explicit dataset and split artifacts."""
    if not config.dataset.artifact_id:
        raise ValueError("train_place_model requires dataset.artifact_id.")
    if not config.splits.artifact_id:
        raise ValueError("train_place_model requires splits.artifact_id.")
    registry = ArtifactRegistry(artifact_root or Path(config.tracking.artifact_root))
    include_latent = config.spatial_model.inputs.observation_source == "latent"
    include_rgb = config.spatial_model.inputs.observation_source == "rgb"
    preload_to_memory = config.dataset.artifact_type == "encoded_dataset"
    selected_ids = config.spatial_model.training.train_episode_ids
    if selected_ids is not None and config.dataset.extra_worlds:
        raise ValueError("train_episode_ids requires a single training world.")

    def _split_datasets(
        dataset_id: str, split_id: str
    ) -> tuple[TrajectoryDataset, TrajectoryDataset]:
        artifact = registry.load(config.dataset.artifact_type, dataset_id)
        payload = json.loads(
            (registry.load("split_set", split_id).path / "split_indices.json").read_text()
        )
        if selected_ids is not None:
            if not selected_ids or len(set(selected_ids)) != len(selected_ids):
                raise ValueError("train_episode_ids must be nonempty and unique.")
            if not set(selected_ids).issubset(payload["train_episode_ids"]):
                raise ValueError("train_episode_ids must belong to the pinned training split.")
            payload["train_episode_ids"] = selected_ids
        return tuple(
            TrajectoryDataset(
                artifact.path,
                episode_ids=[int(value) for value in payload[key]],
                include_rgb=include_rgb,
                include_latent=include_latent,
                preload_to_memory=preload_to_memory,
            )
            for key in ("train_episode_ids", "validation_episode_ids")
        )

    world_ids = [(config.dataset.artifact_id, config.splits.artifact_id)] + [
        (world.dataset_id, world.split_id) for world in config.dataset.extra_worlds
    ]
    world_parts = [_split_datasets(dataset_id, split_id) for dataset_id, split_id in world_ids]
    train_parts = [train for train, _ in world_parts]
    validation_parts = [validation for _, validation in world_parts]
    observation_key = "latent" if include_latent else "rgb" if include_rgb else None
    world_contracts = [
        _world_sample_contract(dataset_id, part, observation_key=observation_key)
        for (dataset_id, _), part in zip(world_ids, train_parts, strict=False)
    ]
    _require_one_shared_contract(world_ids, world_contracts)
    primary_contract = world_contracts[0]
    num_actions = primary_contract.num_actions
    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    validation_dataset = (
        validation_parts[0] if len(validation_parts) == 1 else ConcatDataset(validation_parts)
    )
    validation_episode_limit = int(config.spatial_model.training.max_validation_episodes)
    if 0 < validation_episode_limit < len(validation_dataset):
        validation_dataset = Subset(validation_dataset, range(validation_episode_limit))
    num_loader_workers = max(0, int(config.spatial_model.training.num_workers))
    persistent_workers = num_loader_workers > 0 and preload_to_memory
    pin_memory = torch.cuda.is_available()
    shuffle_generator = (
        None if shuffle_seed is None else torch.Generator().manual_seed(int(shuffle_seed))
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.spatial_model.training.batch_size,
        shuffle=config.spatial_model.training.shuffle_episodes,
        generator=shuffle_generator,
        num_workers=num_loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_episodes,
        **spawned_worker_kwargs(num_loader_workers),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            config.spatial_model.training.validation_batch_size
            or config.spatial_model.training.batch_size
        ),
        shuffle=False,
        num_workers=num_loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_episodes,
        **spawned_worker_kwargs(num_loader_workers),
    )
    observation_shape = primary_contract.observation_shape
    if observation_shape is None:
        observation_dim = num_actions + 1
    else:
        observation_dim = observation_shape[-1] if include_latent else observation_shape[0]
    kinematics_shape = primary_contract.kinematics_shape
    return (
        train_loader,
        validation_loader,
        {
            "num_actions": num_actions,
            "observation_dim": int(observation_dim),
            "kinematics_dim": 0 if kinematics_shape is None else int(kinematics_shape[-1]),
            "episode_length": primary_contract.episode_length,
        },
    )
