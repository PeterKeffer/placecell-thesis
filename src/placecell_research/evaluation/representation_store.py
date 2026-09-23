"""Stored inference arrays shared by evaluation and analysis."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

MANIFEST_NAME = "representations.json"
_STORE_NAME = "representations.zarr"
_SOURCE_GROUP = "sources"
_METADATA_GROUP = "metadata"


@dataclass(frozen=True)
class RepresentationRequest:
    place_model_artifact_id: str
    dataset_artifact_id: str
    dataset_artifact_type: str
    split_artifact_id: str
    checkpoint_selection: str
    device: str
    batch_size: int
    allow_tf32: bool
    torch_version: str
    episode_ids: list[int]


def write_representation_batches(
    directory: Path,
    *,
    split_name: str,
    episode_count: int,
    batches: Iterable[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> None:
    """Write aligned inference batches without materializing a whole split in RAM."""
    directory.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(directory / _STORE_NAME), mode="a")
    split = root.create_group(_split_group_name(split_name), overwrite=True)
    arrays: dict[tuple[str, str], Any] = {}
    offset = 0
    for representations, metadata in batches:
        batch_rows = int(metadata["valid_steps"].shape[0])
        stop = offset + batch_rows
        if batch_rows <= 0 or stop > episode_count:
            raise ValueError("Representation batches exceed the requested episode count.")
        values = {
            (group, name): value
            for group, contents in ((_SOURCE_GROUP, representations), (_METADATA_GROUP, metadata))
            for name, value in contents.items()
        }
        if offset == 0:
            for (group, name), value in values.items():
                arrays[group, name] = split.require_group(group).create_dataset(
                    name,
                    shape=(episode_count, *value.shape[1:]),
                    dtype=value.dtype,
                    chunks=(batch_rows, *value.shape[1:]),
                )
        if values.keys() != arrays.keys():
            raise ValueError("Representation sources or metadata changed between batches.")
        for key, value in values.items():
            array = arrays[key]
            if value.shape != (batch_rows, *array.shape[1:]) or value.dtype != array.dtype:
                raise ValueError(f"Representation batch shape or dtype changed for {key}.")
            array[offset:stop] = value
        offset = stop
    if offset != episode_count or offset == 0:
        raise ValueError(f"Expected {episode_count} representation episodes, received {offset}.")


def _validate_request(directory: Path, split_name: str, request: RepresentationRequest) -> int:
    manifest = read_representation_manifest(directory)
    expected = asdict(request)
    episode_ids = expected.pop("episode_ids")
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        raise ValueError(f"Representation set does not match requested {', '.join(mismatches)}.")
    stored_ids = manifest.get("episode_ids", {}).get(split_name, [])
    count = len(episode_ids)
    if not count or stored_ids[:count] != episode_ids:
        raise ValueError(
            f"Representation set does not contain the requested {split_name} episodes."
        )
    if count < len(stored_ids) and count % request.batch_size:
        raise ValueError("Representation reuse would change the final inference batch size.")
    return count


def _split_group_name(split_name: str) -> str:
    return f"split_{split_name}"


def write_representation_set(
    directory: Path,
    *,
    split_name: str,
    representations: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
) -> None:
    """Append one split's arrays to the store, creating it if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(directory / _STORE_NAME), mode="a")
    split_group = root.require_group(_split_group_name(split_name))
    source_group = split_group.require_group(_SOURCE_GROUP)
    metadata_group = split_group.require_group(_METADATA_GROUP)
    for name, array in representations.items():
        source_group.array(name, np.ascontiguousarray(array), overwrite=True)
    for name, array in metadata.items():
        metadata_group.array(name, np.ascontiguousarray(array), overwrite=True)


def write_representation_manifest(directory: Path, payload: dict[str, Any]) -> None:
    (directory / MANIFEST_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_representation_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no representation manifest at {manifest_path}")
    return json.loads(manifest_path.read_text())


def stored_split_names(directory: Path) -> list[str]:
    root = zarr.open_group(str(directory / _STORE_NAME), mode="r")
    prefix = _split_group_name("")
    return sorted(name[len(prefix) :] for name in root.group_keys() if name.startswith(prefix))


def read_representation_set(
    directory: Path,
    *,
    split_name: str,
    source_names: list[str],
    expected_device: str | None = None,
    require_metadata_keys: list[str] | None = None,
    metadata_keys: list[str] | None = None,
    request: RepresentationRequest | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The same (representations, metadata) pair collect_representations returns."""
    count = None if request is None else _validate_request(directory, split_name, request)
    if expected_device is not None:
        stored_device = str(read_representation_manifest(directory).get("device", ""))
        if stored_device and stored_device != expected_device:
            raise ValueError(
                f"representation set was collected on '{stored_device}' but this stage asked "
                f"for '{expected_device}'; k-winner active sets differ across devices "
                "on a small fraction of steps. Drop expected_device to accept it."
            )
    root = zarr.open_group(str(directory / _STORE_NAME), mode="r")
    group_name = _split_group_name(split_name)
    if group_name not in root:
        raise KeyError(
            f"split '{split_name}' is not in this representation set; it holds "
            f"{stored_split_names(directory)}."
        )
    split_group = root[group_name]
    source_group = split_group[_SOURCE_GROUP]
    available = set(source_group.array_keys())
    missing = [name for name in source_names if name not in available]
    if missing:
        raise KeyError(
            f"representation set has no source(s) {missing} for split '{split_name}'; "
            f"it holds {sorted(available)}."
        )
    selection = slice(None, count)
    representations = {name: np.asarray(source_group[name][selection]) for name in source_names}
    metadata_group = split_group[_METADATA_GROUP]
    available_metadata = set(metadata_group.array_keys())
    if count is not None and any(
        metadata_group[name].shape[0] < count for name in available_metadata
    ):
        raise ValueError("Representation arrays do not match the recorded episode count.")
    missing_metadata = [
        key for key in (require_metadata_keys or []) if key not in available_metadata
    ]
    if missing_metadata:
        raise KeyError(
            f"representation set has no metadata key(s) {missing_metadata} for split "
            f"'{split_name}'; it holds {sorted(available_metadata)}. Collect with those batch keys."
        )
    selected_metadata = (
        available_metadata
        if metadata_keys is None
        else set(metadata_keys) | set(require_metadata_keys or [])
    )
    metadata = {
        name: np.asarray(metadata_group[name][selection])
        for name in metadata_group.array_keys()
        if name in selected_metadata
    }
    if count is not None and any(
        value.shape[0] != count for value in [*representations.values(), *metadata.values()]
    ):
        raise ValueError("Representation arrays do not match the recorded episode count.")
    return representations, metadata


def resolve_representations(
    *,
    artifact_directory: Path | None,
    split_name: str,
    source_names: list[str],
    request: RepresentationRequest | None,
    require_metadata_keys: list[str] | None = None,
    optional_metadata_keys: list[str] | None = None,
    collect: Callable[[], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read a pinned representation set, or run collect when none is pinned."""
    if artifact_directory is None:
        return collect()
    if request is None:
        raise ValueError("Cached representations require an explicit inference request.")
    return read_representation_set(
        artifact_directory,
        split_name=split_name,
        source_names=source_names,
        require_metadata_keys=require_metadata_keys,
        metadata_keys=[
            "valid_steps", "position_xy", "heading", "kinematics", "actions",
            *(optional_metadata_keys or []),
        ],
        request=request,
    )
