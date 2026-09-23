"""Thin objective helpers built on the shared predictor runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from placecell_research.spatial_model.components.input_assemblers import (
    predictor_assembler_uses_belief,
)
from placecell_research.spatial_model.predictor_runtime import (
    assemble_predictor_input,
    assemble_predictor_input_sequence,
    context_from_bundle,
    detach_state,
    future_prediction_from_state,
    predict_code_from_input,
    residual_dynamics_enabled,
    rollout_predictor_sequence,
)
from placecell_research.spatial_model.types import RepresentationBundle

from .masking import masked_mean


def replay_predictor_states(
    model: Any,
    bundle: RepresentationBundle,
) -> tuple[list[Any], list[Any]]:
    """Replay the predictor once to expose clean state snapshots for objectives."""
    encoder_codes = bundle.get_representation("encoder.place_codes")
    _, state_before, state_after = rollout_predictor_sequence(
        model,
        encoder_codes,
        context_from_bundle(model, bundle),
        capture_states=True,
    )
    if state_before is None or state_after is None:
        raise RuntimeError("Predictor replay requested states, but rollout did not capture them.")
    return state_before, state_after


def _can_replay_start_states_fused(model: Any) -> bool:
    family = model.components.config.predictor.family
    return (
        family in {"gru", "lstm", "ssm"}
        and not predictor_assembler_uses_belief(model.predictor_input_assembler)
        and float(model.components.config.predictor.dropout) <= 0.0
        and hasattr(model.predictor_temporal, "forward_sequence_with_layer_outputs")
    )


def _valid_start_steps(start_steps: list[int], time_steps: int) -> list[int]:
    return sorted(
        {int(start_step) for start_step in start_steps if 1 <= int(start_step) < time_steps}
    )


def _fused_gru_start_states(
    model: Any,
    bundle: RepresentationBundle,
    start_steps: list[int],
) -> list[Any]:
    encoder_codes = bundle.get_representation("encoder.place_codes").detach()
    batch_size, time_steps, _ = encoder_codes.shape
    state_before = [None for _ in range(time_steps)]
    valid_start_steps = _valid_start_steps(start_steps, time_steps)
    if not valid_start_steps:
        return state_before

    with torch.no_grad():
        context = context_from_bundle(model, bundle)
        predictor_inputs = assemble_predictor_input_sequence(
            model,
            context,
            encoder_codes=encoder_codes,
            belief_codes=encoder_codes,
        )
        initial_state = model.predictor_temporal.initial_state(batch_size, encoder_codes.device)
        if predictor_inputs.size(1) == 0:
            for start_step in valid_start_steps:
                state_before[start_step] = detach_state(initial_state)
            return state_before
        _, _, layer_outputs = model.predictor_temporal.forward_sequence_with_layer_outputs(
            predictor_inputs,
            initial_state,
        )
        for start_step in valid_start_steps:
            if start_step == 1:
                state_before[start_step] = detach_state(initial_state)
            else:
                state_before[start_step] = detach_state(
                    [layer_output[:, start_step - 2] for layer_output in layer_outputs]
                )
    return state_before


def _segmented_recurrent_start_states(
    model: Any,
    bundle: RepresentationBundle,
    start_steps: list[int],
) -> list[Any]:
    encoder_codes = bundle.get_representation("encoder.place_codes").detach()
    batch_size, time_steps, _ = encoder_codes.shape
    state_before = [None for _ in range(time_steps)]
    valid_start_steps = _valid_start_steps(start_steps, time_steps)
    if not valid_start_steps:
        return state_before

    with torch.no_grad():
        context = context_from_bundle(model, bundle)
        predictor_inputs = assemble_predictor_input_sequence(
            model,
            context,
            encoder_codes=encoder_codes,
            belief_codes=encoder_codes,
        )
        predictor_state = model.predictor_temporal.initial_state(batch_size, encoder_codes.device)
        boundary_step = 1
        for start_step in valid_start_steps:
            if start_step > boundary_step:
                segment = predictor_inputs[:, boundary_step - 1 : start_step - 1]
                if segment.size(1) > 0:
                    (
                        _,
                        predictor_state,
                        _,
                    ) = model.predictor_temporal.forward_sequence_with_layer_outputs(
                        segment, predictor_state
                    )
                boundary_step = start_step
            state_before[start_step] = detach_state(predictor_state)
    return state_before


def replay_predictor_start_states(
    model: Any,
    bundle: RepresentationBundle,
    start_steps: list[int],
) -> list[Any]:
    """Return predictor states before the requested rollout starts."""
    if _can_replay_start_states_fused(model):
        family = model.components.config.predictor.family
        if family == "gru":
            return _fused_gru_start_states(model, bundle, start_steps)
        if family in {"lstm", "ssm"}:
            return _segmented_recurrent_start_states(model, bundle, start_steps)
    state_before, _ = replay_predictor_states(model, bundle)
    return state_before


def bootstrap_future_prediction(
    model: Any,
    bundle: RepresentationBundle,
    step_index: int,
    state_after_step: Any,
) -> Tensor:
    """Recompute the detached one-step-ahead prediction used by TD bootstrap."""
    encoder_codes = bundle.get_representation("encoder.place_codes")
    return future_prediction_from_state(
        model,
        context_from_bundle(model, bundle),
        encoder_code=encoder_codes[:, step_index],
        belief_code=encoder_codes[:, step_index],
        transition_index=step_index,
        predictor_state=state_after_step,
    )


def _prediction_loss(
    predicted_code: Tensor,
    teacher_code: Tensor,
    valid_mask: Tensor,
    prediction_loss_type: str,
) -> Tensor | None:
    if not valid_mask.any():
        return None
    valid_predictions = predicted_code[valid_mask]
    valid_targets = teacher_code[valid_mask]
    if prediction_loss_type == "mse":
        return (valid_predictions - valid_targets).pow(2).mean()
    cosine_similarity = F.cosine_similarity(valid_predictions, valid_targets, dim=-1)
    return (1.0 - cosine_similarity).mean()


def _masked_prediction_loss(
    predicted_code: Tensor,
    teacher_code: Tensor,
    valid_mask: Tensor,
    prediction_loss_type: str,
) -> tuple[Tensor, Tensor]:
    if prediction_loss_type == "mse":
        per_sample_loss = (predicted_code - teacher_code).pow(2).mean(dim=-1)
    else:
        per_sample_loss = 1.0 - F.cosine_similarity(predicted_code, teacher_code, dim=-1)
    return masked_mean(per_sample_loss, valid_mask), valid_mask.any()


def _record_horizon(
    accumulator: dict[int, list[tuple[Tensor, Tensor]]] | None,
    horizon: int,
    step_loss: Tensor,
    valid_count: Tensor,
) -> None:
    """Collect (loss, valid-sample count) per horizon for reporting."""
    if accumulator is None:
        return
    count = valid_count.detach()
    if not bool(count > 0):
        return
    accumulator.setdefault(horizon, []).append((step_loss.detach(), count))


def _combine_rollout_losses(
    losses: list[Tensor],
    loss_horizons: list[int],
    *,
    detach_intermediate: bool,
    step_weight_decay: float,
    horizon_discount_gamma: float | None,
    zero: Tensor,
) -> Tensor:
    if not losses:
        return zero
    if detach_intermediate:
        return losses[-1]
    if horizon_discount_gamma is not None:
        horizons = torch.tensor(loss_horizons, device=losses[0].device, dtype=losses[0].dtype)
        gamma = losses[0].new_tensor(horizon_discount_gamma)
        weights = torch.pow(gamma, horizons - 1.0)
        return (torch.stack(losses) * weights).sum() / weights.sum()
    weighted_losses = [
        loss_value * (step_weight_decay**offset)
        for offset, loss_value in enumerate(losses)
    ]
    return torch.stack(weighted_losses).sum() / len(weighted_losses)


def _combine_masked_rollout_losses(
    losses: list[Tensor],
    valid_offsets: list[Tensor],
    loss_horizons: list[int],
    *,
    detach_intermediate: bool,
    step_weight_decay: float,
    horizon_discount_gamma: float | None,
    zero: Tensor,
) -> Tensor:
    if not losses:
        return zero
    stacked_losses = torch.stack(losses)
    valid_offset_mask = torch.stack(valid_offsets)
    valid_offset_weights = valid_offset_mask.to(stacked_losses.dtype)
    valid_offset_count = valid_offset_weights.sum()
    if detach_intermediate:
        offset_indices = torch.arange(stacked_losses.shape[0], device=stacked_losses.device)
        last_valid_offset = torch.where(
            valid_offset_mask,
            offset_indices,
            torch.full_like(offset_indices, -1),
        ).max()
        return (
            stacked_losses[last_valid_offset.clamp_min(0)]
            * (valid_offset_count > 0).to(stacked_losses.dtype)
            + zero
        )
    if horizon_discount_gamma is not None:
        horizons = torch.tensor(
            loss_horizons,
            device=stacked_losses.device,
            dtype=stacked_losses.dtype,
        )
        gamma = stacked_losses.new_tensor(horizon_discount_gamma)
        horizon_weights = torch.pow(gamma, horizons - 1.0) * valid_offset_weights
        return (
            (stacked_losses * horizon_weights).sum()
            / horizon_weights.sum().clamp_min(torch.finfo(stacked_losses.dtype).eps)
            + zero
        )
    valid_ranks = valid_offset_weights.cumsum(dim=0) - 1.0
    decay = stacked_losses.new_tensor(step_weight_decay)
    step_weights = torch.pow(decay, valid_ranks.clamp_min(0.0)) * valid_offset_weights
    return (
        (stacked_losses * step_weights).sum() / valid_offset_count.clamp_min(1.0)
        + zero
    )


def rollout_loss(
    model: Any,
    bundle: RepresentationBundle,
    start_step: int,
    horizon: int,
    replayed_state_before: list[Any],
    *,
    detach_intermediate: bool,
    step_weight_decay: float,
    prediction_loss_type: str,
    horizon_discount_gamma: float | None = None,
    reset_state: bool = False,
    supervised_horizons: frozenset[int] | None = None,
    horizon_losses: dict[int, list[tuple[Tensor, Tensor]]] | None = None,
) -> Tensor:
    """Roll out the predictor without encoder observations for multistep supervision."""
    encoder_codes = bundle.get_representation("encoder.place_codes")
    teacher_codes = bundle.get_representation("teacher.place_codes")
    valid_steps = bundle.masks["valid_steps"].bool()
    context = context_from_bundle(model, bundle)
    if reset_state:
        predictor_state = model.predictor_temporal.initial_state(
            encoder_codes.shape[0],
            encoder_codes.device,
        )
    else:
        predictor_state = detach_state(replayed_state_before[start_step])
        if predictor_state is None:
            return encoder_codes.sum() * 0.0
    running_code = encoder_codes[:, start_step - 1]
    losses: list[Tensor] = []
    loss_horizons: list[int] = []
    for offset in range(horizon):
        target_step = start_step + offset
        if target_step >= encoder_codes.shape[1]:
            break
        transition_index = target_step - 1
        predictor_input = assemble_predictor_input(
            model,
            context,
            encoder_code=running_code,
            belief_code=running_code,
            transition_index=transition_index,
        )
        _, _, _, predicted_code, predictor_state = predict_code_from_input(
            model,
            predictor_input,
            predictor_state,
            timestep=target_step,
            residual_base=running_code if residual_dynamics_enabled(model) else None,
        )
        running_code = predicted_code.detach() if detach_intermediate else predicted_code
        valid_mask = valid_steps[:, transition_index] & valid_steps[:, target_step]
        if supervised_horizons is None or offset + 1 in supervised_horizons:
            step_loss = _prediction_loss(
                predicted_code,
                teacher_codes[:, target_step],
                valid_mask,
                prediction_loss_type,
            )
            if step_loss is not None:
                losses.append(step_loss)
                loss_horizons.append(offset + 1)
                _record_horizon(
                    horizon_losses, offset + 1, step_loss, valid_mask.sum()
                )
    return _combine_rollout_losses(
        losses,
        loss_horizons,
        detach_intermediate=detach_intermediate,
        step_weight_decay=step_weight_decay,
        horizon_discount_gamma=horizon_discount_gamma,
        zero=encoder_codes.sum() * 0.0,
    )


def persistence_rollout_loss(
    bundle: RepresentationBundle,
    start_steps: list[int],
    horizon: int,
    *,
    detach_intermediate: bool,
    step_weight_decay: float,
    prediction_loss_type: str,
    horizon_discount_gamma: float | None = None,
    supervised_horizons: frozenset[int] | None = None,
    horizon_losses: dict[int, list[tuple[Tensor, Tensor]]] | None = None,
) -> Tensor:
    """Score copying the anchor code over the same rollout windows and targets."""
    encoder_codes = bundle.get_representation("encoder.place_codes")
    teacher_codes = bundle.get_representation("teacher.place_codes")
    valid_steps = bundle.masks["valid_steps"].bool()
    zero = encoder_codes.detach().sum() * 0.0
    start_losses: list[Tensor] = []
    with torch.no_grad():
        for start_step in _valid_start_steps(start_steps, encoder_codes.shape[1]):
            anchor_code = encoder_codes[:, start_step - 1]
            losses: list[Tensor] = []
            valid_offsets: list[Tensor] = []
            loss_horizons: list[int] = []
            for offset in range(min(horizon, encoder_codes.shape[1] - start_step)):
                if supervised_horizons is not None and offset + 1 not in supervised_horizons:
                    continue
                target_step = start_step + offset
                persistence_mask = (
                    valid_steps[:, target_step - 1] & valid_steps[:, target_step]
                )
                step_loss, has_valid_samples = _masked_prediction_loss(
                    anchor_code,
                    teacher_codes[:, target_step],
                    persistence_mask,
                    prediction_loss_type,
                )
                losses.append(step_loss)
                valid_offsets.append(has_valid_samples)
                loss_horizons.append(offset + 1)
                _record_horizon(
                    horizon_losses, offset + 1, step_loss, persistence_mask.sum()
                )
            start_losses.append(
                _combine_masked_rollout_losses(
                    losses,
                    valid_offsets,
                    loss_horizons,
                    detach_intermediate=detach_intermediate,
                    step_weight_decay=step_weight_decay,
                    horizon_discount_gamma=horizon_discount_gamma,
                    zero=zero,
                )
            )
    if not start_losses:
        return zero
    return torch.stack(start_losses).mean()


def _can_batch_reset_rollouts(model: Any) -> bool:
    if model.components.config.predictor.family == "clockwork":
        return False
    if model.predictor_temporal.training and float(model.components.config.predictor.dropout) > 0.0:
        return False
    return not any(
        module.training and float(getattr(module, "boost_strength", 0.0)) > 0.0
        for module in model.predictor_sparsifier.modules()
    )


def reset_rollout_loss(
    model: Any,
    bundle: RepresentationBundle,
    start_steps: list[int],
    horizon: int,
    *,
    detach_intermediate: bool,
    step_weight_decay: float,
    prediction_loss_type: str,
    horizon_discount_gamma: float | None = None,
    supervised_horizons: frozenset[int] | None = None,
    horizon_losses: dict[int, list[tuple[Tensor, Tensor]]] | None = None,
) -> Tensor:
    """Batch strict self-fed rollout windows along the batch dimension."""
    encoder_codes = bundle.get_representation("encoder.place_codes")
    teacher_codes = bundle.get_representation("teacher.place_codes")
    valid_steps = bundle.masks["valid_steps"].bool()
    batch_size, time_steps, _ = encoder_codes.shape
    valid_start_steps = _valid_start_steps(start_steps, time_steps)
    zero = encoder_codes.sum() * 0.0
    if not valid_start_steps:
        return zero

    if not _can_batch_reset_rollouts(model):
        empty_states = [None for _ in range(time_steps)]
        losses = [
            rollout_loss(
                model,
                bundle,
                start_step,
                min(horizon, time_steps - start_step),
                empty_states,
                detach_intermediate=detach_intermediate,
                step_weight_decay=step_weight_decay,
                horizon_discount_gamma=horizon_discount_gamma,
                prediction_loss_type=prediction_loss_type,
                reset_state=True,
                supervised_horizons=supervised_horizons,
                horizon_losses=horizon_losses,
            )
            for start_step in valid_start_steps
        ]
        return torch.stack(losses).mean()

    context = context_from_bundle(model, bundle)
    num_starts = len(valid_start_steps)
    predictor_state = model.predictor_temporal.initial_state(
        batch_size * num_starts,
        encoder_codes.device,
    )
    running_codes = torch.stack(
        [encoder_codes[:, start_step - 1] for start_step in valid_start_steps]
    )
    losses_by_start: list[list[Tensor]] = [[] for _ in valid_start_steps]
    valid_offsets_by_start: list[list[Tensor]] = [[] for _ in valid_start_steps]
    loss_horizons_by_start: list[list[int]] = [[] for _ in valid_start_steps]
    rollout_steps = min(horizon, time_steps - min(valid_start_steps))

    for offset in range(rollout_steps):
        predictor_inputs: list[Tensor] = []
        target_steps: list[int] = []
        for start_index, start_step in enumerate(valid_start_steps):
            target_step = min(start_step + offset, time_steps - 1)
            target_steps.append(target_step)
            transition_index = target_step - 1
            predictor_inputs.append(
                assemble_predictor_input(
                    model,
                    context,
                    encoder_code=running_codes[start_index],
                    belief_code=running_codes[start_index],
                    transition_index=transition_index,
                )
            )
        _, _, _, predicted_codes, predictor_state = predict_code_from_input(
            model,
            torch.cat(predictor_inputs, dim=0),
            predictor_state,
            timestep=offset + 1,
            residual_base=(
                running_codes.flatten(0, 1) if residual_dynamics_enabled(model) else None
            ),
        )
        predicted_codes = predicted_codes.reshape(
            num_starts,
            batch_size,
            predicted_codes.shape[-1],
        )
        running_codes = predicted_codes.detach() if detach_intermediate else predicted_codes

        for start_index, (start_step, target_step) in enumerate(
            zip(valid_start_steps, target_steps, strict=False)
        ):
            if start_step + offset >= time_steps:
                continue
            if supervised_horizons is not None and offset + 1 not in supervised_horizons:
                continue
            transition_index = target_step - 1
            reset_mask = valid_steps[:, transition_index] & valid_steps[:, target_step]
            step_loss, has_valid_samples = _masked_prediction_loss(
                predicted_codes[start_index],
                teacher_codes[:, target_step],
                reset_mask,
                prediction_loss_type,
            )
            losses_by_start[start_index].append(step_loss)
            valid_offsets_by_start[start_index].append(has_valid_samples)
            loss_horizons_by_start[start_index].append(offset + 1)
            _record_horizon(horizon_losses, offset + 1, step_loss, reset_mask.sum())

    start_losses = [
        _combine_masked_rollout_losses(
            losses,
            valid_offsets,
            loss_horizons,
            detach_intermediate=detach_intermediate,
            step_weight_decay=step_weight_decay,
            horizon_discount_gamma=horizon_discount_gamma,
            zero=zero,
        )
        for losses, valid_offsets, loss_horizons in zip(
            losses_by_start,
            valid_offsets_by_start,
            loss_horizons_by_start, strict=False,
        )
    ]
    return torch.stack(start_losses).mean()
