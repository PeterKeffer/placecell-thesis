"""Frozen-model inference for evaluation and analysis."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import closing
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from placecell_research.datasets.batch_iterator import iterate_dataset_batches, load_split_indices
from placecell_research.spatial_model.loading import load_place_model_artifact
from placecell_research.tracking.progress import ProgressUpdate

_OPEN_LOOP_REANCHOR_STEPS = 8


def configure_open_loop_rollout(model: object, source_names: tuple[str, ...]) -> None:
    """Enable the open-loop predictor rollout only when a predictor_rollout source is requested."""
    if hasattr(model, "export_open_loop_rollout_steps"):
        model.export_open_loop_rollout_steps = (  # type: ignore[attr-defined]
            _OPEN_LOOP_REANCHOR_STEPS
            if any(str(name).startswith("predictor_rollout.") for name in source_names)
            else None
        )


def load_model_checkpoint(
    model_directory: Path, device: torch.device, *, selection: str = "last"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a place-model artifact and rebuild its model from checkpoint metadata when needed."""
    model, _auxiliary_heads, contract, _payload = load_place_model_artifact(
        model_directory, device, selection=selection
    )
    return model, contract


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _add_elapsed(timing_seconds: dict[str, float], section_name: str, started_at: float) -> None:
    elapsed = perf_counter() - started_at
    timing_seconds[section_name] = timing_seconds.get(section_name, 0.0) + elapsed


def _timing_text(timing_seconds: dict[str, float], section_names: tuple[str, ...]) -> str:
    return " ".join(f"{name}={timing_seconds.get(name, 0.0):.2f}s" for name in section_names)


def _concatenate_and_release(chunks: list[np.ndarray]) -> np.ndarray:
    """Join per-batch chunks into one array, freeing each chunk as it is copied."""
    total_rows = sum(chunk.shape[0] for chunk in chunks)
    joined = np.empty((total_rows, *chunks[0].shape[1:]), dtype=chunks[0].dtype)
    next_row = 0
    chunks.reverse()
    while chunks:
        chunk = chunks.pop()
        joined[next_row : next_row + chunk.shape[0]] = chunk
        next_row += chunk.shape[0]
        del chunk
    return joined


_CACHEABLE_BATCH_KEYS = frozenset(
    {"latent", "actions", "valid_steps", "position_xy", "heading", "kinematics"}
)
_InputBatchCacheKey = tuple[str, str, str, int, int, str]


class InputBatchCache:
    """Dataset-side model inputs, read once and replayed for later passes over the same episodes."""

    def __init__(self) -> None:
        self._batches_by_key: dict[_InputBatchCacheKey, list[dict[str, torch.Tensor]]] = {}

    def resident_bytes(self) -> int:
        """Host memory the recorded batches hold, for the stage's memory accounting."""
        return sum(
            tensor.element_size() * tensor.nelement()
            for batches in self._batches_by_key.values()
            for batch in batches
            for tensor in batch.values()
        )

    def clear(self) -> None:
        self._batches_by_key.clear()

    def iterate(
        self,
        cache_key: _InputBatchCacheKey,
        device: torch.device,
        read_batches: Callable[[], Iterator[dict[str, torch.Tensor]]],
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield the batches for cache_key, recording them on the first pass."""
        recorded_batches = self._batches_by_key.get(cache_key)
        if recorded_batches is not None:
            for batch in recorded_batches:
                yield {name: tensor.to(device=device, copy=True) for name, tensor in batch.items()}
            return

        recording: list[dict[str, torch.Tensor]] | None = []
        for batch in read_batches():
            if recording is not None:
                if _CACHEABLE_BATCH_KEYS.issuperset(batch):
                    recording.append(
                        {name: tensor.detach().cpu() for name, tensor in batch.items()}
                    )
                else:
                    recording = None
            yield batch
        if recording:
            self._batches_by_key[cache_key] = recording


def collect_representations(
    model: torch.nn.Module,
    dataset_directory: Path,
    split_directory: Path,
    split_name: str,
    source_names: list[str],
    device: torch.device,
    batch_size: int,
    include_batch_keys: list[str] | None = None,
    observation_source: Literal["latent", "rgb", "both"] | None = None,
    max_episodes: int | None = None,
    progress_callback: Callable[[ProgressUpdate], None] | None = None,
    input_batch_cache: InputBatchCache | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run inference and collect requested representations and aligned metadata."""
    representation_chunks: dict[str, list[np.ndarray]] = {name: [] for name in source_names}
    metadata_chunks: dict[str, list[np.ndarray]] = {}
    batches = iter_representation_batches(
        model,
        dataset_directory,
        split_directory,
        split_name,
        source_names,
        device,
        batch_size,
        include_batch_keys=include_batch_keys,
        observation_source=observation_source,
        max_episodes=max_episodes,
        progress_callback=progress_callback,
        input_batch_cache=input_batch_cache,
    )
    with closing(batches):
        for representations, metadata in batches:
            for name, value in representations.items():
                representation_chunks[name].append(value)
            for name, value in metadata.items():
                metadata_chunks.setdefault(name, []).append(value)
    return (
        {name: _concatenate_and_release(chunks) for name, chunks in representation_chunks.items()},
        {name: _concatenate_and_release(chunks) for name, chunks in metadata_chunks.items()},
    )


def iter_representation_batches(
    model: torch.nn.Module,
    dataset_directory: Path,
    split_directory: Path,
    split_name: str,
    source_names: list[str],
    device: torch.device,
    batch_size: int,
    include_batch_keys: list[str] | None = None,
    observation_source: Literal["latent", "rgb", "both"] | None = None,
    max_episodes: int | None = None,
    progress_callback: Callable[[ProgressUpdate], None] | None = None,
    input_batch_cache: InputBatchCache | None = None,
) -> Iterator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Yield aligned host arrays; close early consumers to restore model mode."""
    total_started_at = perf_counter()
    timing_seconds: dict[str, float] = {}
    if observation_source is None:
        section_started_at = perf_counter()
        components = getattr(model, "components", None)
        observation_source = getattr(
            getattr(components, "config", None),
            "inputs",
            None,
        )
        observation_source = getattr(observation_source, "observation_source", "both")
        _add_elapsed(timing_seconds, "resolve_observation_source", section_started_at)
    requested_batch_keys = list(include_batch_keys or [])
    episode_ids = load_split_indices(split_directory, split_name)
    if max_episodes is not None and max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]
    total_batches = max(1, int(ceil(len(episode_ids) / max(batch_size, 1))))
    processed_batches = 0
    if progress_callback is not None:
        progress_callback(
            ProgressUpdate(
                completed=0,
                total=total_batches,
                elapsed_seconds=0.0,
                unit_name="batches",
                detail=f"collect {split_name} representations",
            )
        )

    def read_batches() -> Iterator[dict[str, torch.Tensor]]:
        return iterate_dataset_batches(
            dataset_directory,
            split_directory,
            split_name,
            batch_size,
            device,
            observation_source=observation_source,
            max_episodes=max_episodes,
        )

    batch_iterator = (
        read_batches()
        if input_batch_cache is None
        else input_batch_cache.iterate(
            (
                str(dataset_directory),
                str(split_directory),
                split_name,
                int(batch_size),
                -1 if max_episodes is None else int(max_episodes),
                str(observation_source),
            ),
            device,
            read_batches,
        )
    )
    was_training = model.training
    model.eval()
    try:
        while True:
            with torch.inference_mode():
                section_started_at = perf_counter()
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    _add_elapsed(timing_seconds, "batch_read", section_started_at)
                    break
                _sync_if_cuda(device)
                _add_elapsed(timing_seconds, "batch_read", section_started_at)
                section_started_at = perf_counter()
                bundle = model.forward_sequence(batch)
                _sync_if_cuda(device)
                _add_elapsed(timing_seconds, "forward", section_started_at)
                section_started_at = perf_counter()
                representations = {
                    name: bundle.get_representation(name)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                    for name in source_names
                }
                _add_elapsed(timing_seconds, "representations_to_numpy", section_started_at)
                section_started_at = perf_counter()
                metadata = {
                    "valid_steps": batch["valid_steps"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool, copy=False)
                }
                for key in dict.fromkeys(
                    ["position_xy", "heading", "kinematics", "actions", *requested_batch_keys]
                ):
                    if key in batch:
                        metadata[key] = batch[key].detach().cpu().numpy()
                _add_elapsed(timing_seconds, "metadata_to_numpy", section_started_at)
                del bundle, batch
            processed_batches += 1
            yield representations, metadata
            del representations, metadata
            if progress_callback is not None:
                progress_callback(
                    ProgressUpdate(
                        completed=processed_batches,
                        total=total_batches,
                        elapsed_seconds=perf_counter() - total_started_at,
                        unit_name="batches",
                        detail=f"collect {split_name} representations",
                    )
                )
    finally:
        model.train(was_training)
    if not processed_batches:
        raise ValueError(f"No representations were collected for split '{split_name}'.")
    timing_seconds["total"] = perf_counter() - total_started_at
    print(
        "[collect_representations] "
        f"split={split_name} sources={','.join(source_names)} "
        f"device={device} observation_source={observation_source} "
        + _timing_text(
            timing_seconds,
            (
                "total",
                "batch_read",
                "forward",
                "representations_to_numpy",
                "metadata_to_numpy",
            ),
        )
        + f" batches={processed_batches} episodes={len(episode_ids)} batch_size={batch_size}",
        file=sys.stderr,
        flush=True,
    )
