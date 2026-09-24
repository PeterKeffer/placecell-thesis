"""Training loop."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from placecell_research.config.schema import (
    PhaseConfig,
    SpatialModelConfig,
    SpatialTrainingConfig,
)
from placecell_research.objectives import compute_total_loss
from placecell_research.objectives.registry import BuiltObjectives, MetricValue
from placecell_research.spatial_model.protocol import PlaceModel
from placecell_research.tracking.progress import ProgressUpdate
from placecell_research.utils.metrics import materialize_metric_values

from .checkpointing import load_checkpoint, save_checkpoint
from .optimizer import build_optimizer, clip_gradients, clip_gradients_by_component
from .scheduling import build_scheduler
from .selection import CheckpointSelector

PREDICTOR_SIDE_SELECTORS: frozenset[str] = frozenset({"predictor", "sparsifier", "embeddings"})


def evaluation_epochs(
    *,
    total_epochs: int,
    every_n_epochs: int,
    schedule: str = "uniform",
) -> frozenset[int]:
    """Zero-based epochs that run in-training validation."""
    if total_epochs <= 0:
        return frozenset()
    if schedule == "front_loaded":
        epochs = set()
        one_based_epoch = 1
        while one_based_epoch <= total_epochs:
            epochs.add(one_based_epoch - 1)
            one_based_epoch *= 2
    elif schedule == "uniform":
        interval = max(1, int(every_n_epochs))
        epochs = {epoch for epoch in range(total_epochs) if (epoch + 1) % interval == 0}
    else:
        raise ValueError(f"Unknown evaluation schedule {schedule!r}.")
    epochs.add(total_epochs - 1)
    return frozenset(epochs)


@dataclass
class TrainLoopConfig:
    training: SpatialTrainingConfig
    checkpoint_dir: Path
    device: torch.device
    eval_every_n_epochs: int = 1
    eval_schedule: str = "uniform"
    sequence_length: int | None = None
    build_context: dict[str, object] | None = None
    resume_checkpoint: Path | None = None
    resume_policy: str = "fresh"
    step_metrics_callback: Callable[[dict[str, MetricValue], int], None] | None = None
    epoch_metrics_callback: Callable[[dict[str, float], int], None] | None = None
    progress_callback: Callable[[ProgressUpdate], None] | None = None


@dataclass
class TrainLoopResult:
    final_metrics: dict[str, float]
    best_primary_path: Path | None
    best_validation_loss_path: Path | None
    last_path: Path | None
    parameter_groups: dict[str, list[nn.Parameter]]


def build_phase_schedule(
    *, epochs: int, phases: list[PhaseConfig], model: PlaceModel
) -> list[PhaseConfig]:
    """The effective ordered phase schedule the loop runs."""
    if phases:
        return phases
    return [PhaseConfig(name="all", epochs=epochs, train=sorted(model.available_selectors()))]


def active_phase_for_epoch(phases: list[PhaseConfig], epoch: int) -> tuple[PhaseConfig, int]:
    """The phase active at epoch and the epoch's 0-based offset within that phase."""
    cumulative = 0
    for phase in phases:
        if epoch < cumulative + phase.epochs:
            return phase, epoch - cumulative
        cumulative += phase.epochs
    return phases[-1], epoch - (cumulative - phases[-1].epochs)


def _metric_value_to_tensor(value: MetricValue, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().reshape(())
    return torch.as_tensor(float(value), device=device, dtype=torch.float32)


def _accumulate_metric_sums(
    metric_sums: dict[str, Tensor],
    metric_counts: dict[str, int],
    metrics: dict[str, MetricValue],
    *,
    device: torch.device,
) -> None:
    for name, value in metrics.items():
        metric_value = _metric_value_to_tensor(value, device)
        metric_sums[name] = metric_sums.get(name, torch.zeros_like(metric_value)) + metric_value
        metric_counts[name] = metric_counts.get(name, 0) + 1


def _materialize_metric_means(
    metric_sums: dict[str, Tensor], metric_counts: dict[str, int]
) -> dict[str, float]:
    """Epoch means, each divided by the number of BATCHES that actually reported that metric."""
    if not metric_sums:
        return {}
    return materialize_metric_values(
        {name: value / max(metric_counts.get(name, 1), 1) for name, value in metric_sums.items()}
    )


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _data_loader_state_dict(loader: object) -> dict[str, object] | None:
    state_dict = getattr(loader, "state_dict", None)
    if callable(state_dict):
        return dict(state_dict())
    generator = getattr(loader, "generator", None)
    if isinstance(generator, torch.Generator):
        return {"generator_state": generator.get_state()}
    return None


def _load_data_loader_state_dict(loader: object, state: dict[str, object]) -> None:
    load_state_dict = getattr(loader, "load_state_dict", None)
    if callable(load_state_dict):
        load_state_dict(state)
        return
    generator = getattr(loader, "generator", None)
    generator_state = state.get("generator_state")
    if isinstance(generator, torch.Generator) and isinstance(generator_state, Tensor):
        generator.set_state(generator_state)


def _optimizer_steps_per_batch(training: SpatialTrainingConfig, sequence_length: int | None) -> int:
    if training.bptt_window == 0:
        return 1
    if sequence_length is None:
        raise ValueError("Stateful BPTT requires TrainLoopConfig.sequence_length.")
    return max(1, math.ceil(sequence_length / training.bptt_window))


def _iter_sequence_chunks(
    batch: dict[str, Tensor],
    window: int,
) -> list[dict[str, Tensor]]:
    if window == 0:
        return [batch]
    valid_steps = batch.get("valid_steps")
    if valid_steps is None or valid_steps.ndim != 2:
        raise ValueError("Stateful BPTT requires batch['valid_steps'] with shape [batch, time].")
    batch_size, time_steps = valid_steps.shape
    chunks: list[dict[str, Tensor]] = []
    for start in range(0, time_steps, window):
        end = min(start + window, time_steps)
        chunks.append(
            {
                key: (
                    value[:, start:end]
                    if isinstance(value, Tensor)
                    and value.ndim >= 2
                    and value.shape[:2] == (batch_size, time_steps)
                    else value
                )
                for key, value in batch.items()
            }
        )
    return chunks


def _predictor_side_is_active(phase_selectors: set[str]) -> bool:
    return bool(phase_selectors & PREDICTOR_SIDE_SELECTORS)


def _selectors_for_optimizer_step(
    phase_selectors: set[str],
    *,
    step_count: int,
    predictor_update_interval: int,
) -> tuple[set[str], bool]:
    predictor_side_is_active = _predictor_side_is_active(phase_selectors)
    predictor_update = predictor_side_is_active and step_count % predictor_update_interval == 0
    if not predictor_side_is_active:
        return phase_selectors, False
    if predictor_update:
        return phase_selectors, True
    return phase_selectors - PREDICTOR_SIDE_SELECTORS, False


def apply_tf32_policy(allow_tf32: bool) -> None:
    """Set the process-global TF32 switches from config."""
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def train_model(
    model: PlaceModel,
    built_objectives: BuiltObjectives,
    model_config: SpatialModelConfig,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    loop_config: TrainLoopConfig,
    evaluate_fn: Callable[[PlaceModel, nn.ModuleDict, list, DataLoader | None], dict[str, float]]
    | None = None,
) -> TrainLoopResult:
    apply_tf32_policy(loop_config.training.allow_tf32)
    device = loop_config.device
    if hasattr(model, "set_auxiliary_heads"):
        model.set_auxiliary_heads(built_objectives.auxiliary_heads)
    model = model.to(device)
    built_objectives.auxiliary_heads.to(device)
    optimizer, parameter_groups = build_optimizer(
        model, built_objectives.auxiliary_heads, loop_config.training
    )
    scheduler = build_scheduler(optimizer, loop_config.training)
    selector = CheckpointSelector(loop_config.training.selection)
    step_count = 0
    start_epoch = 0
    best_primary_path: Path | None = None
    best_validation_loss_path: Path | None = None
    last_path: Path | None = None
    final_metrics: dict[str, float] = {}
    latest_evaluation_metrics: dict[str, float] | None = None

    if loop_config.resume_checkpoint is not None and loop_config.resume_policy != "fresh":
        restore_training_state = loop_config.resume_policy == "weights_and_optimizer"
        checkpoint_state = load_checkpoint(
            loop_config.resume_checkpoint,
            model,
            built_objectives.auxiliary_heads,
            optimizer if restore_training_state else None,
            scheduler if restore_training_state else None,
            restore_random_state=restore_training_state,
            allow_auxiliary_mismatch=not restore_training_state,
        )
        if restore_training_state:
            start_epoch = checkpoint_state.epoch + 1
            step_count = checkpoint_state.step
            final_metrics = dict(checkpoint_state.metrics)
            if checkpoint_state.data_loader_state is not None:
                _load_data_loader_state_dict(train_loader, checkpoint_state.data_loader_state)
            loop_state = checkpoint_state.loop_state or {}
            selector.best_primary = loop_state.get("best_primary")
            selector.best_primary_tie_break = loop_state.get("best_primary_tie_break")
            selector.best_validation_loss = loop_state.get("best_validation_loss")
            primary_candidate = loop_config.checkpoint_dir / "weights_best_primary.pt"
            validation_candidate = loop_config.checkpoint_dir / "weights_best_validation_loss.pt"
            best_primary_path = primary_candidate if primary_candidate.exists() else None
            best_validation_loss_path = (
                validation_candidate if validation_candidate.exists() else None
            )
        elif loop_config.resume_policy == "weights_only":
            step_count = checkpoint_state.step

    progress_step_offset = step_count if loop_config.resume_policy == "weights_only" else 0
    steps_per_batch = _optimizer_steps_per_batch(
        loop_config.training,
        loop_config.sequence_length,
    )
    total_steps = loop_config.training.epochs * max(len(train_loader), 1) * steps_per_batch
    evaluation_epoch_set = evaluation_epochs(
        total_epochs=loop_config.training.epochs,
        every_n_epochs=loop_config.eval_every_n_epochs,
        schedule=loop_config.eval_schedule,
    )
    started_at = perf_counter()
    if loop_config.progress_callback is not None:
        loop_config.progress_callback(
            ProgressUpdate(
                completed=step_count - progress_step_offset,
                total=total_steps,
                elapsed_seconds=perf_counter() - started_at,
                unit_name="optimizer_steps",
                detail=f"epoch {start_epoch}/{loop_config.training.epochs}",
            )
        )

    phase_schedule = build_phase_schedule(
        epochs=loop_config.training.epochs,
        phases=loop_config.training.phases,
        model=model,
    )

    for epoch in range(start_epoch, loop_config.training.epochs):
        model.train(True)
        built_objectives.auxiliary_heads.train(True)
        active_phase, _ = active_phase_for_epoch(phase_schedule, epoch)
        predictor_active = _predictor_side_is_active(set(active_phase.train))
        if hasattr(model, "set_trainable"):
            model.set_trainable(set(active_phase.train))
        active_head_namespaces = {
            namespace
            for namespace in getattr(model, "representation_heads", {})
            if namespace in active_phase.train
        }
        epoch_metric_sums: dict[str, Tensor] = {}
        epoch_metric_counts: dict[str, int] = {}
        for batch in train_loader:
            chunk_state = None
            for batch_chunk in _iter_sequence_chunks(
                batch,
                loop_config.training.bptt_window,
            ):
                step_selectors, predictor_update = _selectors_for_optimizer_step(
                    set(active_phase.train),
                    step_count=step_count,
                    predictor_update_interval=loop_config.training.predictor_update_interval,
                )
                if hasattr(model, "set_trainable"):
                    model.set_trainable(step_selectors)
                optimizer.zero_grad(set_to_none=True)
                batch_on_device = _to_device(batch_chunk, device)
                if loop_config.training.bptt_window > 0:
                    forward_chunk = getattr(model, "forward_chunk", None)
                    detach_chunk_state = getattr(model, "detach_chunk_state", None)
                    if forward_chunk is None or detach_chunk_state is None:
                        raise TypeError(
                            "training.bptt_window requires a model with forward_chunk and "
                            "detach_chunk_state."
                        )
                    bundle, next_chunk_state = forward_chunk(batch_on_device, chunk_state)
                else:
                    bundle = model.forward_sequence(batch_on_device)
                    next_chunk_state = None
                total_loss, metrics = compute_total_loss(
                    built_objectives.objectives,
                    bundle,
                    batch_on_device,
                    model_config,
                )
                total_loss.backward()
                clip_all_parameters = [
                    parameter
                    for group in parameter_groups.values()
                    for parameter in group
                    if parameter.requires_grad
                ]
                clip_isolated_parameters = [
                    parameter
                    for namespace in active_head_namespaces
                    for parameter in model.representation_heads[namespace].parameters()
                    if parameter.requires_grad
                ]
                clip_diagnostics: dict[str, float] = {}
                if loop_config.training.gradient_clip_mode == "component":
                    causal_groups = model.clip_parameter_groups()
                    grad_norm, clip_diagnostics = clip_gradients_by_component(
                        clip_all_parameters,
                        clip_isolated_parameters,
                        {
                            name: [p for p in group if p.requires_grad]
                            for name, group in causal_groups.items()
                        },
                        loop_config.training.gradient_clip_norm,
                    )
                else:
                    grad_norm = clip_gradients(
                        clip_all_parameters,
                        clip_isolated_parameters,
                        loop_config.training.gradient_clip_norm,
                    )
                optimizer.step()
                post_optimizer_step = getattr(model, "post_optimizer_step", None)
                if post_optimizer_step is not None:
                    post_optimizer_step()
                model.update_teacher(step_count)
                if next_chunk_state is not None:
                    chunk_state = detach_chunk_state(next_chunk_state)
                metrics["train/grad_norm"] = (
                    grad_norm.detach().reshape(())
                    if isinstance(grad_norm, Tensor)
                    else float(grad_norm)
                )
                for diagnostic_name, diagnostic_value in clip_diagnostics.items():
                    metrics[f"train/clip_{diagnostic_name}"] = diagnostic_value
                metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
                metrics["train/predictor_active"] = float(predictor_active)
                metrics["train/predictor_update"] = float(predictor_update)
                _accumulate_metric_sums(
                    epoch_metric_sums, epoch_metric_counts, metrics, device=device
                )
                if loop_config.step_metrics_callback is not None:
                    loop_config.step_metrics_callback(metrics, step_count)
                step_count += 1
                if loop_config.progress_callback is not None:
                    loop_config.progress_callback(
                        ProgressUpdate(
                            completed=step_count - progress_step_offset,
                            total=total_steps,
                            elapsed_seconds=perf_counter() - started_at,
                            unit_name="optimizer_steps",
                            detail=f"epoch {epoch + 1}/{loop_config.training.epochs}",
                        )
                    )
        scheduler.step()
        final_metrics = _materialize_metric_means(epoch_metric_sums, epoch_metric_counts)
        final_metrics["epoch"] = float(epoch)
        selection_metrics = dict(final_metrics)
        if evaluate_fn is not None:
            should_run_evaluation = epoch == start_epoch or epoch in evaluation_epoch_set
            if should_run_evaluation:
                validation_metrics = evaluate_fn(
                    model,
                    built_objectives.auxiliary_heads,
                    built_objectives.objectives,
                    validation_loader,
                )
                latest_evaluation_metrics = dict(validation_metrics)
                final_metrics.update(validation_metrics)
                selection_metrics.update(validation_metrics)
            elif latest_evaluation_metrics is not None:
                selection_metrics.update(latest_evaluation_metrics)
        decisions = selector.update(selection_metrics)

        def save_training_checkpoint(
            path: Path,
            checkpoint_epoch: int = epoch,
            checkpoint_step: int = step_count,
            checkpoint_metrics: dict[str, float] = final_metrics,
        ) -> None:
            checkpoint_loop_state: dict[str, object] = {
                "best_primary": selector.best_primary,
                "best_primary_tie_break": selector.best_primary_tie_break,
                "best_validation_loss": selector.best_validation_loss,
            }
            save_checkpoint(
                path,
                model,
                built_objectives.auxiliary_heads,
                optimizer,
                scheduler,
                checkpoint_epoch,
                checkpoint_step,
                checkpoint_metrics,
                build_context=loop_config.build_context,
                loop_state=checkpoint_loop_state,
                data_loader_state=_data_loader_state_dict(train_loader),
            )

        if decisions["save_best_primary"]:
            best_primary_path = loop_config.checkpoint_dir / "weights_best_primary.pt"
            save_training_checkpoint(best_primary_path)
        if decisions["save_best_validation_loss"]:
            best_validation_loss_path = (
                loop_config.checkpoint_dir / "weights_best_validation_loss.pt"
            )
            save_training_checkpoint(best_validation_loss_path)
        if decisions["save_last"]:
            last_path = loop_config.checkpoint_dir / "weights_last.pt"
            save_training_checkpoint(last_path)
        checkpoint_every = loop_config.training.checkpoint_every_n_epochs
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_training_checkpoint(
                loop_config.checkpoint_dir / f"weights_epoch_{epoch + 1:03d}.pt"
            )
        if loop_config.epoch_metrics_callback is not None:
            loop_config.epoch_metrics_callback(final_metrics, step_count)
    return TrainLoopResult(
        final_metrics=final_metrics,
        best_primary_path=best_primary_path,
        best_validation_loss_path=best_validation_loss_path,
        last_path=last_path,
        parameter_groups=parameter_groups,
    )
