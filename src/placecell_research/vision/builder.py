"""Vision model building, training, and frozen encoding."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from placecell_research.collection.previews import (
    write_reconstruction_grid,
    write_side_by_side_reconstruction_gif,
)
from placecell_research.datasets.dataset import TrajectoryDataset
from placecell_research.datasets.schema import (
    ACTIONS_KEY,
    CONTINUOUS_ACTIONS_KEY,
    HEADING_KEY,
    KINEMATICS_KEY,
    LENGTH_KEY,
    POSITION_KEY,
    RGB_KEY,
    SOURCE_SEED_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    VALID_MASK_KEY,
)
from placecell_research.tracking.progress import ProgressUpdate
from placecell_research.utils.cpu_budget import parallel_read_worker_count
from placecell_research.utils.dataloader import spawned_worker_kwargs
from placecell_research.utils.metrics import materialize_metric_values

from .autoencoder import ConvAutoEncoder, ConvBetaVAE

if TYPE_CHECKING:
    from placecell_research.config.schema import VisionConfig

logger = logging.getLogger(__name__)


class FrameDataset(Dataset[torch.Tensor]):
    """Flatten valid RGB frames across datasets and optionally cache them."""

    def __init__(
        self,
        datasets: list[TrajectoryDataset],
        *,
        max_frames_per_episode: int = 0,
        frame_cache_mode: str = "auto",
        frame_cache_memory_fraction: float = 0.7,
    ) -> None:
        self.datasets = datasets
        self.max_frames_per_episode = max(0, int(max_frames_per_episode))
        self.frame_cache_mode = str(frame_cache_mode or "auto").strip().lower()
        self.frame_cache_memory_fraction = float(frame_cache_memory_fraction)
        self.frame_index: list[tuple[int, int, int]] = []
        self.cached_frames: torch.Tensor | None = None
        self._disk_cache_directory: tempfile.TemporaryDirectory | None = None
        self._disk_cached_frames: np.memmap | None = None
        self._creator_pid = os.getpid()
        self.frame_cache_backend = "none"
        self.frame_cache_uses_shared_memory = False
        self.frame_cache_estimated_bytes = 0
        self.frame_cache_budget_bytes = 0
        episode_keys = [
            (dataset_index, episode_index)
            for dataset_index, dataset in enumerate(datasets)
            if dataset.has_array(VALID_MASK_KEY)
            for episode_index in range(len(dataset))
        ]
        for (dataset_index, episode_index), step_indices in zip(
            episode_keys,
            self._map_episode_reads(self._sampled_step_indices, episode_keys),
            strict=False,
        ):
            self.frame_index.extend(
                (dataset_index, episode_index, int(step_index)) for step_index in step_indices
            )
        self._initialize_frame_cache()

    def _map_episode_reads(self, read: Callable, episode_keys: list) -> list:
        """Run one per-episode read over every episode, in order, at the allocation's width."""
        worker_count = parallel_read_worker_count()
        if worker_count <= 0:
            return [read(key) for key in episode_keys]
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            return list(pool.map(read, episode_keys))

    def _sampled_step_indices(self, episode_key: tuple[int, int]) -> np.ndarray:
        dataset_index, episode_index = episode_key
        valid_steps = (
            self.datasets[dataset_index]
            .read_episode_array(episode_index, VALID_MASK_KEY)
            .astype(bool, copy=False)
        )
        valid_indices = np.flatnonzero(valid_steps)
        if self.max_frames_per_episode <= 0 or len(valid_indices) <= self.max_frames_per_episode:
            return valid_indices
        sampling_seed = dataset_index * 1_000_003 + episode_index
        generator = np.random.default_rng(sampling_seed)
        return np.sort(
            generator.choice(valid_indices, size=self.max_frames_per_episode, replace=False)
        )

    def _reopen_datasets_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._creator_pid:
            return
        for dataset in self.datasets:
            if hasattr(dataset, "_group"):
                dataset._group = None
            if hasattr(dataset, "_open"):
                dataset._open()
        self._creator_pid = current_pid

    @staticmethod
    def _available_memory_bytes() -> int:
        try:
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            return 0
        if available_pages > 0 and page_size > 0:
            return available_pages * page_size
        return 0

    def _read_frame(self, dataset_index: int, episode_index: int, step_index: int) -> np.ndarray:
        frame = self.datasets[dataset_index].read_rgb_frame(episode_index, step_index)
        return np.asarray(frame, dtype=np.uint8)

    def _initialize_frame_cache(self) -> None:
        if not self.frame_index or self.frame_cache_mode == "none":
            return
        first_dataset = self.datasets[self.frame_index[0][0]]
        if not first_dataset.has_array(RGB_KEY):
            return
        if self.frame_cache_mode not in {"auto", "memory", "disk"}:
            raise ValueError(
                f"Unsupported frame_cache_mode={self.frame_cache_mode!r}. Use auto, memory, disk, "
                "or none."
            )
        first_frame = self._read_frame(*self.frame_index[0])
        self.frame_cache_estimated_bytes = len(self.frame_index) * int(first_frame.nbytes)
        available_memory_bytes = self._available_memory_bytes()
        if available_memory_bytes > 0:
            self.frame_cache_budget_bytes = int(
                available_memory_bytes * self.frame_cache_memory_fraction
            )
        if self.frame_cache_mode == "auto":
            if (
                self.frame_cache_budget_bytes > 0
                and self.frame_cache_estimated_bytes <= self.frame_cache_budget_bytes
            ):
                self._cache_frames_in_memory(first_frame)
                return
            self._cache_frames_on_disk(first_frame)
            return
        if self.frame_cache_mode == "memory":
            if (
                self.frame_cache_budget_bytes > 0
                and self.frame_cache_estimated_bytes > self.frame_cache_budget_bytes
            ):
                raise ValueError(
                    "frame_cache_mode=memory exceeds the available memory budget. "
                    f"estimated_bytes={self.frame_cache_estimated_bytes}, "
                    f"budget_bytes={self.frame_cache_budget_bytes}."
                )
            self._cache_frames_in_memory(first_frame)
            return
        self._cache_frames_on_disk(first_frame)

    def _fill_one_episode(
        self,
        storage: np.ndarray | np.memmap,
        dataset_index: int,
        episode_index: int,
        frame_positions: np.ndarray,
        step_indices: np.ndarray,
    ) -> None:
        dataset = self.datasets[dataset_index]
        try:
            episode_rgb = dataset.read_episode_array(episode_index, RGB_KEY)
        except KeyError:
            logger.warning(
                "Dataset %s episode %s has no full RGB episode array; "
                "reading sampled frames individually.",
                dataset_index,
                episode_index,
                exc_info=True,
            )
            for frame_position, step_index in zip(frame_positions, step_indices, strict=False):
                storage[int(frame_position)] = self._read_frame(
                    dataset_index,
                    episode_index,
                    int(step_index),
                )
            return
        storage[frame_positions] = np.asarray(episode_rgb[step_indices], dtype=np.uint8)

    def _fill_cache_storage(self, storage: np.ndarray | np.memmap, first_frame: np.ndarray) -> None:
        """Copy every sampled frame into the cache, one whole-episode read at a time."""
        del first_frame
        frames_by_episode: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for frame_position, (dataset_index, episode_index, step_index) in enumerate(
            self.frame_index
        ):
            frames_by_episode.setdefault((dataset_index, episode_index), []).append(
                (frame_position, step_index)
            )
        episode_reads = [
            (
                dataset_index,
                episode_index,
                np.fromiter((position for position, _ in entries), np.int64, len(entries)),
                np.fromiter((step for _, step in entries), np.int64, len(entries)),
            )
            for (dataset_index, episode_index), entries in frames_by_episode.items()
        ]
        for dataset in self.datasets:
            dataset.has_array(RGB_KEY)
        self._map_episode_reads(lambda read: self._fill_one_episode(storage, *read), episode_reads)

    def _cache_frames_in_memory(self, first_frame: np.ndarray) -> None:
        cached_frames = np.empty((len(self.frame_index), *first_frame.shape), dtype=np.uint8)
        self._fill_cache_storage(cached_frames, first_frame)
        cached_tensor = torch.from_numpy(cached_frames)
        try:
            cached_tensor = cached_tensor.share_memory_()
            self.frame_cache_uses_shared_memory = True
        except RuntimeError:
            self.frame_cache_uses_shared_memory = False
        self.cached_frames = cached_tensor
        self.frame_cache_backend = "memory"

    def _cache_frames_on_disk(self, first_frame: np.ndarray) -> None:
        cache_root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
        self._disk_cache_directory = tempfile.TemporaryDirectory(
            prefix="vision_frame_cache_",
            dir=str(cache_root),
        )
        cache_path = Path(self._disk_cache_directory.name) / "frames.memmap"
        cached_frames = np.memmap(
            cache_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(self.frame_index), *first_frame.shape),
        )
        self._fill_cache_storage(cached_frames, first_frame)
        cached_frames.flush()
        self._disk_cached_frames = cached_frames
        self.frame_cache_backend = "disk"

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.cached_frames is not None:
            return self.cached_frames[index]
        elif self._disk_cached_frames is not None:
            rgb_frame = np.asarray(self._disk_cached_frames[index], dtype=np.uint8)
        else:
            self._reopen_datasets_after_fork()
            dataset_index, episode_index, step_index = self.frame_index[index]
            dataset = self.datasets[dataset_index]
            if not dataset.has_array(RGB_KEY):
                raise ValueError("Vision training requires RGB frames.")
            rgb_frame = dataset.read_rgb_frame(episode_index, step_index)
        return torch.from_numpy(np.asarray(rgb_frame, dtype=np.uint8))


@dataclass
class VisionTrainingResult:
    model: torch.nn.Module
    loss_history: list[float]
    validation_loss_history: list[float]
    preview_dir: Path
    input_shape: tuple[int, int, int]
    best_validation_loss: float | None
    validation_eval_steps: list[int]
    training_summary: dict[str, int | float | str | bool]
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]


MAX_RECONSTRUCTION_PREVIEW_EPISODES = 10
MAX_RECONSTRUCTION_PREVIEW_FRAMES = 64
PSNR_EPSILON = 1e-12


def _frame_batch_to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch.to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    ).div_(255.0)


def build_vision_model(config: VisionConfig, input_shape: tuple[int, int, int]) -> torch.nn.Module:
    """Build the configured vision model."""
    loss_type = config.loss_type
    if config.type == "autoencoder":
        return ConvAutoEncoder(
            latent_dim=config.latent_dim,
            channels=tuple(config.channels),
            input_shape=input_shape,
            loss_type=loss_type,
            loss_l1_weight=config.loss_l1_weight,
            loss_ssim_weight=config.loss_ssim_weight,
            loss_edge_weight=config.loss_edge_weight,
        )
    if loss_type != "mse":
        raise ValueError(
            f"vision.loss_type={loss_type!r} is only supported for type=autoencoder, "
            f"not type={config.type!r}. Leave loss_type at 'mse' for this model."
        )
    if config.type == "beta_vae":
        return ConvBetaVAE(
            latent_dim=config.latent_dim,
            beta=config.beta,
            channels=tuple(config.channels),
            input_shape=input_shape,
        )
    if config.type == "identity":
        return torch.nn.Identity()
    raise ValueError(f"Unsupported vision model type: {config.type}")


def _squared_error_sum(inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    return (reconstruction.detach() - inputs.detach()).pow(2).sum()


def _reconstruction_metric_values(squared_error_sum: float, pixel_count: int) -> dict[str, float]:
    mse_per_pixel = float(squared_error_sum) / max(int(pixel_count), 1)
    rmse_per_pixel = math.sqrt(mse_per_pixel)
    psnr = -10.0 * math.log10(max(mse_per_pixel, PSNR_EPSILON))
    return {
        "mse_per_pixel": mse_per_pixel,
        "rmse_per_pixel": rmse_per_pixel,
        "psnr": psnr,
    }


def _prefix_reconstruction_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _write_loss_curve(
    loss_history: list[float],
    output_path: Path,
    *,
    validation_loss_history: list[float] | None = None,
    validation_eval_steps: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    from ..figure_style import despine

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 3))
    axis.plot(loss_history, label="train")
    if validation_loss_history and validation_eval_steps:
        axis.plot(validation_eval_steps, validation_loss_history, marker="o", label="validation")
    axis.set_title("Training loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    if validation_loss_history:
        axis.legend()
    despine(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _model_supports_reconstruction_previews(model: torch.nn.Module) -> bool:
    return hasattr(model, "decode") and not isinstance(model, torch.nn.Identity)


def _reconstruct_rgb_frames(
    model: torch.nn.Module,
    rgb_frames: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    if rgb_frames.ndim != 4:
        raise ValueError(f"Expected RGB frames shaped [T, C, H, W], got {tuple(rgb_frames.shape)}.")
    was_training = model.training
    model.eval()
    reconstructed_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, rgb_frames.shape[0], batch_size):
            batch = _frame_batch_to_device(rgb_frames[start : start + batch_size], device)
            output = model(batch)
            reconstructed_chunks.append(
                output.reconstruction.detach().cpu().numpy().astype(np.float32, copy=False)
            )
    if was_training:
        model.train()
    return np.concatenate(reconstructed_chunks, axis=0)


def _evaluate_vision_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    was_training = model.training
    total_loss = 0.0
    total_examples = 0
    total_squared_error = 0.0
    total_pixels = 0
    batch_metrics: dict[str, torch.Tensor] = {}
    batch_sizes: list[tuple[int, int]] = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            batch = _frame_batch_to_device(batch, device)
            output = model(batch)
            loss = model.loss(batch, output)
            batch_size = int(batch.shape[0])
            batch_metrics[f"loss/{index}"] = loss.detach()
            batch_metrics[f"squared_error/{index}"] = _squared_error_sum(
                batch, output.reconstruction
            )
            batch_sizes.append((batch_size, int(batch.numel())))
    values = materialize_metric_values(batch_metrics)
    for index, (batch_size, pixel_count) in enumerate(batch_sizes):
        total_loss += values[f"loss/{index}"] * batch_size
        total_examples += batch_size
        total_squared_error += values[f"squared_error/{index}"]
        total_pixels += pixel_count
    if was_training:
        model.train()
    if total_examples <= 0:
        raise ValueError("Validation loader produced no examples.")
    validation_metrics = _reconstruction_metric_values(total_squared_error, total_pixels)
    return total_loss / total_examples, validation_metrics


def write_dataset_reconstruction_previews(
    model: torch.nn.Module,
    datasets: list[TrajectoryDataset],
    preview_dir: Path,
    device: torch.device,
    *,
    max_preview_episodes: int = MAX_RECONSTRUCTION_PREVIEW_EPISODES,
    max_preview_frames: int = MAX_RECONSTRUCTION_PREVIEW_FRAMES,
) -> list[Path]:
    """Write compact reconstruction previews for up to ten example episodes."""
    if not _model_supports_reconstruction_previews(model):
        return []

    preview_dir.mkdir(parents=True, exist_ok=True)
    example_dir = preview_dir / "reconstruction_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    wrote_top_level_sample = False
    episodes_written = 0

    for dataset_index, dataset in enumerate(datasets):
        for episode_index in range(len(dataset)):
            if episodes_written >= max_preview_episodes:
                return written_paths
            sample = dataset[episode_index]
            rgb = sample["rgb"]
            valid_steps = sample["valid_steps"]
            if rgb is None or valid_steps is None:
                continue
            valid_indices = torch.nonzero(valid_steps, as_tuple=False).view(-1)
            if len(valid_indices) == 0:
                continue
            source_frames = rgb[valid_indices].detach().cpu()
            heading = sample.get("heading")
            heading_values = (
                None if heading is None else heading[valid_indices].detach().cpu().numpy()
            )
            reconstructed_frames = _reconstruct_rgb_frames(model, source_frames, device=device)
            if not wrote_top_level_sample:
                write_reconstruction_grid(
                    source_frames.numpy(),
                    reconstructed_frames,
                    preview_dir / "reconstruction_grid.png",
                    max_frames=4,
                )
                sample_gif_path = preview_dir / "reconstruction_samples.gif"
                write_side_by_side_reconstruction_gif(
                    source_frames.numpy(),
                    reconstructed_frames,
                    sample_gif_path,
                    max_frames=max_preview_frames,
                    headings=heading_values,
                )
                written_paths.extend([preview_dir / "reconstruction_grid.png", sample_gif_path])
                wrote_top_level_sample = True

            example_path = (
                example_dir / f"episode_{episodes_written:03d}__dataset_{dataset_index:02d}.gif"
            )
            write_side_by_side_reconstruction_gif(
                source_frames.numpy(),
                reconstructed_frames,
                example_path,
                max_frames=max_preview_frames,
                headings=heading_values,
            )
            written_paths.append(example_path)
            episodes_written += 1
    return written_paths


def train_vision_model(
    config: VisionConfig,
    datasets: list[TrajectoryDataset],
    output_dir: Path,
    device: torch.device,
    validation_datasets: list[TrajectoryDataset] | None = None,
    resume_checkpoint: Path | None = None,
    shuffle_seed: int | None = None,
    step_metrics_callback: Callable[[dict[str, float], int], None] | None = None,
    progress_callback: Callable[[ProgressUpdate], None] | None = None,
    metric_interval_steps: int = 1,
) -> VisionTrainingResult:
    """Train a frozen vision encoder artifact."""
    if metric_interval_steps < 1:
        raise ValueError("metric_interval_steps must be positive.")
    dataset_episode_count = sum(len(dataset) for dataset in datasets)
    frame_dataset_started_at = perf_counter()
    frame_dataset = FrameDataset(
        datasets,
        max_frames_per_episode=config.max_frames_per_episode,
        frame_cache_mode=config.frame_cache_mode,
        frame_cache_memory_fraction=config.frame_cache_memory_fraction,
    )
    frame_dataset_elapsed_seconds = perf_counter() - frame_dataset_started_at
    if len(frame_dataset) == 0:
        raise ValueError("Vision training requires at least one valid RGB frame.")
    print(
        "[vision/frame_dataset] "
        f"datasets={len(datasets)} "
        f"episodes={dataset_episode_count} "
        f"sampled_frames={len(frame_dataset)} "
        f"build_seconds={frame_dataset_elapsed_seconds:.2f}",
        flush=True,
    )
    print(
        "[vision/frame_cache] "
        f"requested_mode={config.frame_cache_mode} "
        f"backend={frame_dataset.frame_cache_backend} "
        f"shared_memory={frame_dataset.frame_cache_uses_shared_memory} "
        f"frames={len(frame_dataset)} "
        f"estimated_bytes={frame_dataset.frame_cache_estimated_bytes} "
        f"budget_bytes={frame_dataset.frame_cache_budget_bytes}",
        flush=True,
    )
    sample_shape = tuple(int(value) for value in frame_dataset[0].shape)
    model = build_vision_model(config, sample_shape)
    if isinstance(model, torch.nn.Identity):
        return VisionTrainingResult(
            model=model,
            loss_history=[],
            validation_loss_history=[],
            preview_dir=output_dir / "previews",
            input_shape=sample_shape,
            best_validation_loss=None,
            validation_eval_steps=[],
            training_summary={
                "dataset_count": len(datasets),
                "dataset_episode_count": dataset_episode_count,
                "sampled_frame_count": len(frame_dataset),
                "frame_dataset_build_seconds": frame_dataset_elapsed_seconds,
                "frame_cache_backend": frame_dataset.frame_cache_backend,
                "frame_cache_shared_memory": frame_dataset.frame_cache_uses_shared_memory,
                "frame_cache_estimated_bytes": frame_dataset.frame_cache_estimated_bytes,
                "frame_cache_budget_bytes": frame_dataset.frame_cache_budget_bytes,
                "data_loader_batch_size": int(config.batch_size),
                "data_loader_batches_per_epoch": 0,
                "training_total_steps": 0,
                "validation_enabled": False,
                "validation_dataset_count": 0,
                "validation_dataset_episode_count": 0,
                "validation_sampled_frame_count": 0,
                "data_loader_num_workers": int(config.data_loader_num_workers),
                "data_loader_pin_memory": bool(
                    config.data_loader_pin_memory and device.type == "cuda"
                ),
                "data_loader_persistent_workers": bool(
                    config.data_loader_persistent_workers
                    and int(config.data_loader_num_workers) > 0
                ),
            },
            train_metrics={},
            validation_metrics={},
        )
    model = model.to(device)
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    data_loader_num_workers = int(config.data_loader_num_workers)
    data_loader_pin_memory = bool(config.data_loader_pin_memory and device.type == "cuda")
    data_loader_persistent_workers = bool(
        config.data_loader_persistent_workers and data_loader_num_workers > 0
    )
    shuffle_generator = (
        None if shuffle_seed is None else torch.Generator().manual_seed(int(shuffle_seed))
    )
    dataloader = DataLoader(
        frame_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
        persistent_workers=data_loader_persistent_workers,
        **spawned_worker_kwargs(data_loader_num_workers),
    )
    validation_dataset_episode_count = sum(len(dataset) for dataset in validation_datasets or [])
    validation_frame_dataset: FrameDataset | None = None
    validation_dataloader: DataLoader | None = None
    if validation_datasets:
        validation_frame_dataset = FrameDataset(
            validation_datasets,
            max_frames_per_episode=config.max_frames_per_episode,
            frame_cache_mode=config.frame_cache_mode,
            frame_cache_memory_fraction=config.frame_cache_memory_fraction,
        )
        if len(validation_frame_dataset) > 0:
            validation_dataloader = DataLoader(
                validation_frame_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=data_loader_num_workers,
                pin_memory=data_loader_pin_memory,
                persistent_workers=data_loader_persistent_workers,
                **spawned_worker_kwargs(data_loader_num_workers),
            )
    steps_per_epoch = max(len(dataloader), 1)
    total_steps = config.epochs * steps_per_epoch
    print(
        "[vision/data_loader] "
        f"batch_size={config.batch_size} "
        f"batches_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps} "
        f"num_workers={data_loader_num_workers} "
        f"pin_memory={data_loader_pin_memory} "
        f"persistent_workers={data_loader_persistent_workers}",
        flush=True,
    )
    loss_history: list[float] = []
    validation_loss_history: list[float] = []
    validation_eval_steps: list[int] = []
    best_validation_loss: float | None = None
    best_validation_metrics: dict[str, float] = {}
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    latest_train_metrics: dict[str, float] = {}
    global_step = 0
    started_at = perf_counter()
    model.train()
    if progress_callback is not None:
        progress_callback(
            ProgressUpdate(
                completed=0,
                total=total_steps,
                elapsed_seconds=0.0,
                unit_name="optimizer_steps",
                detail=f"epoch 0/{config.epochs}",
            )
        )
    for _epoch in range(config.epochs):
        epoch_index = _epoch + 1
        epoch_losses: list[torch.Tensor] = []
        for batch_index, batch in enumerate(dataloader):
            batch = _frame_batch_to_device(batch, device)
            output = model(batch)
            loss = model.loss(batch, output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())
            report_step = (
                step_metrics_callback is not None and global_step % metric_interval_steps == 0
            )
            if report_step or batch_index == len(dataloader) - 1:
                values = materialize_metric_values(
                    {
                        "loss": loss,
                        "squared_error": _squared_error_sum(batch, output.reconstruction),
                        **model.last_loss_components,
                    }
                )
                latest_train_metrics = _reconstruction_metric_values(
                    values["squared_error"], int(batch.numel())
                )
                if report_step:
                    step_metrics_callback(
                        {
                            "train/loss": values["loss"],
                            **{
                                f"train/loss_{name}": values[name]
                                for name in model.last_loss_components
                            },
                            **_prefix_reconstruction_metrics("train", latest_train_metrics),
                        },
                        global_step,
                    )
            global_step += 1
            if progress_callback is not None:
                progress_callback(
                    ProgressUpdate(
                        completed=global_step,
                        total=total_steps,
                        elapsed_seconds=perf_counter() - started_at,
                        unit_name="optimizer_steps",
                        detail=f"epoch {epoch_index}/{config.epochs}",
                    )
                )
        loss_history.extend(torch.stack(epoch_losses).cpu().tolist())
        if validation_dataloader is not None:
            validation_loss, validation_metrics = _evaluate_vision_model(
                model,
                validation_dataloader,
                device=device,
            )
            validation_loss_history.append(validation_loss)
            validation_eval_steps.append(global_step)
            if step_metrics_callback is not None:
                step_metrics_callback(
                    {
                        "validation/loss": validation_loss,
                        **_prefix_reconstruction_metrics("validation", validation_metrics),
                    },
                    global_step,
                )
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_metrics = validation_metrics
                best_model_state_dict = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    _write_loss_curve(
        loss_history,
        preview_dir / "loss_curve.png",
        validation_loss_history=validation_loss_history,
        validation_eval_steps=validation_eval_steps,
    )
    write_dataset_reconstruction_previews(model, datasets, preview_dir, device=device)
    if validation_datasets:
        write_dataset_reconstruction_previews(
            model,
            validation_datasets,
            preview_dir / "validation",
            device=device,
        )
    return VisionTrainingResult(
        model=model,
        loss_history=loss_history,
        validation_loss_history=validation_loss_history,
        preview_dir=preview_dir,
        input_shape=sample_shape,
        best_validation_loss=best_validation_loss,
        validation_eval_steps=validation_eval_steps,
        training_summary={
            "dataset_count": len(datasets),
            "dataset_episode_count": dataset_episode_count,
            "sampled_frame_count": len(frame_dataset),
            "frame_dataset_build_seconds": frame_dataset_elapsed_seconds,
            "frame_cache_backend": frame_dataset.frame_cache_backend,
            "frame_cache_shared_memory": frame_dataset.frame_cache_uses_shared_memory,
            "frame_cache_estimated_bytes": frame_dataset.frame_cache_estimated_bytes,
            "frame_cache_budget_bytes": frame_dataset.frame_cache_budget_bytes,
            "data_loader_batch_size": int(config.batch_size),
            "data_loader_batches_per_epoch": steps_per_epoch,
            "training_total_steps": total_steps,
            "validation_enabled": validation_dataloader is not None,
            "validation_dataset_count": len(validation_datasets or []),
            "validation_dataset_episode_count": validation_dataset_episode_count,
            "validation_sampled_frame_count": (
                len(validation_frame_dataset) if validation_frame_dataset is not None else 0
            ),
            "data_loader_num_workers": data_loader_num_workers,
            "data_loader_pin_memory": data_loader_pin_memory,
            "data_loader_persistent_workers": data_loader_persistent_workers,
            "best_validation_loss": (
                float(best_validation_loss) if best_validation_loss is not None else 0.0
            ),
        },
        train_metrics=latest_train_metrics,
        validation_metrics=best_validation_metrics,
    )


def _latent_dim_for_model(model: torch.nn.Module) -> int:
    if hasattr(model, "fc_mu"):
        return int(model.fc_mu.out_features)
    if hasattr(model, "fc_latent"):
        return int(model.fc_latent.out_features)
    raise ValueError(f"Cannot infer latent dimension for vision model type {type(model).__name__}.")


class _RawEpisodeReadDataset(Dataset):
    """Map-style dataset that reads one raw episode's arrays per index."""

    _EPISODE_FIELDS = (
        ("rgb", RGB_KEY, np.uint8),
        ("valid_steps", VALID_MASK_KEY, bool),
        ("actions", ACTIONS_KEY, np.int64),
        ("position_xy", POSITION_KEY, np.float32),
        ("heading", HEADING_KEY, np.float32),
        ("kinematics", KINEMATICS_KEY, np.float32),
        ("length", LENGTH_KEY, np.int32),
        ("terminated", TERMINATED_KEY, bool),
        ("truncated", TRUNCATED_KEY, bool),
        ("source_seed", SOURCE_SEED_KEY, np.int64),
    )
    _WHOLE_ARRAY_KEYS = (LENGTH_KEY, TERMINATED_KEY, TRUNCATED_KEY, SOURCE_SEED_KEY)

    def __init__(self, dataset: TrajectoryDataset) -> None:
        self._dataset = dataset
        self._whole_array_reads = self._read_whole_array_keys(dataset)

    @classmethod
    def _read_whole_array_keys(cls, dataset: TrajectoryDataset) -> dict[str, np.ndarray]:
        """Read once the arrays whose chunk already spans every episode."""
        open_store = getattr(dataset, "_open", None)
        if open_store is None:
            return {}
        group = open_store()
        episode_ids = getattr(dataset, "episode_ids", None)
        selection = None if episode_ids is None else np.asarray(episode_ids, dtype=np.int64)
        whole_array_reads: dict[str, np.ndarray] = {}
        for key in cls._WHOLE_ARRAY_KEYS:
            branch, name = key.rsplit("/", 1)
            if branch not in group or name not in group[branch]:
                continue
            array = group[key]
            if array.ndim != 1 or int(array.chunks[0]) < int(array.shape[0]):
                continue
            values = np.asarray(array)
            whole_array_reads[key] = values if selection is None else values[selection]
        return whole_array_reads

    def _read(self, index: int, key: str) -> np.ndarray:
        cached = self._whole_array_reads.get(key)
        if cached is not None:
            return cached[index]
        return self._dataset.read_episode_array(index, key)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        episode = {
            name: np.asarray(self._read(index, key)).astype(dtype, copy=False)
            for name, key, dtype in self._EPISODE_FIELDS
        }
        if self._dataset.has_array(CONTINUOUS_ACTIONS_KEY):
            episode[CONTINUOUS_ACTIONS_KEY] = np.asarray(
                self._read(index, CONTINUOUS_ACTIONS_KEY), dtype=np.float32
            )
        return episode


def _raw_episode_list_collate(
    batch: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Identity collate; module-level so it pickles for spawn workers."""
    return batch


def _assemble_encoded_episode(
    raw_episode: dict[str, np.ndarray], latents: np.ndarray
) -> dict[str, np.ndarray]:
    episode = {
        "observations/rgb": raw_episode["rgb"],
        "observations/latent": latents,
        "actions/discrete": raw_episode["actions"],
        "state/position_xy": raw_episode["position_xy"],
        "state/heading": raw_episode["heading"],
        "state/kinematics": raw_episode["kinematics"],
        "masks/valid_steps": raw_episode["valid_steps"],
        "episode_metadata/length": raw_episode["length"],
        "episode_metadata/terminated": raw_episode["terminated"],
        "episode_metadata/truncated": raw_episode["truncated"],
        "episode_metadata/source_seed": raw_episode["source_seed"],
    }
    if CONTINUOUS_ACTIONS_KEY in raw_episode:
        episode[CONTINUOUS_ACTIONS_KEY] = raw_episode[CONTINUOUS_ACTIONS_KEY]
    return episode


def iter_encoded_episodes_with_model(
    model: torch.nn.Module,
    dataset: TrajectoryDataset,
    device: torch.device,
    *,
    read_workers: int = 0,
    episode_batch_size: int = 1,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield encoded dataset episodes one at a time, in dataset order."""
    if isinstance(model, torch.nn.Identity):
        raise ValueError("Identity vision model cannot encode RGB frames into latents.")
    model = model.to(device)
    model.eval()
    latent_dim = _latent_dim_for_model(model)
    read_workers = max(0, int(read_workers))
    episode_batch_size = max(1, int(episode_batch_size))
    loader = DataLoader(
        _RawEpisodeReadDataset(dataset),
        batch_size=episode_batch_size,
        shuffle=False,
        num_workers=read_workers,
        collate_fn=_raw_episode_list_collate,
        **spawned_worker_kwargs(read_workers),
    )
    with torch.inference_mode():
        for raw_episode_batch in loader:
            valid_indices_per_episode = [
                np.flatnonzero(raw_episode["valid_steps"]) for raw_episode in raw_episode_batch
            ]
            frames_to_encode = [
                raw_episode["rgb"][valid_indices]
                for raw_episode, valid_indices in zip(
                    raw_episode_batch, valid_indices_per_episode, strict=False
                )
                if valid_indices.size > 0
            ]
            encoded_frames: np.ndarray | None = None
            if frames_to_encode:
                stacked = torch.from_numpy(np.concatenate(frames_to_encode, axis=0)).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                stacked.div_(255.0)
                encoded_result = model.encode(stacked)
                encoded = encoded_result[0] if isinstance(encoded_result, tuple) else encoded_result
                encoded_frames = encoded.to(torch.float32).cpu().numpy()
            offset = 0
            for raw_episode, valid_indices in zip(
                raw_episode_batch, valid_indices_per_episode, strict=False
            ):
                time_steps = int(raw_episode["rgb"].shape[0])
                latents = np.zeros((time_steps, latent_dim), dtype=np.float32)
                if valid_indices.size > 0:
                    latents[valid_indices] = encoded_frames[offset : offset + valid_indices.size]
                    offset += valid_indices.size
                yield _assemble_encoded_episode(raw_episode, latents)


def encode_dataset_with_model(
    model: torch.nn.Module,
    dataset: TrajectoryDataset,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Encode a raw dataset with a frozen vision model."""
    encoded_episodes = list(iter_encoded_episodes_with_model(model, dataset, device))
    if not encoded_episodes:
        raise ValueError("Encoded datasets require at least one episode.")
    return {
        key: np.stack([episode[key] for episode in encoded_episodes]).astype(
            encoded_episodes[0][key].dtype, copy=False
        )
        for key in encoded_episodes[0]
    }
