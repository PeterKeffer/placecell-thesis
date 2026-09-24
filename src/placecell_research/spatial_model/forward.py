"""Forward pass orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .auxiliary_heads import run_auxiliary_heads
from .batch_access import (
    resolve_actions,
    resolve_kinematics,
    resolve_observations,
    resolve_position_xy,
    resolve_valid_steps,
)
from .predictor_runtime import (
    context_from_inputs,
    detach_state,
    open_loop_rollout_predictor_sequence,
    rollout_predictor_chunk,
    rollout_predictor_sequence,
)
from .predictor_stack import DetachedPredictorStack
from .representation_contract import representation_view_specs
from .representation_heads import run_representation_heads
from .types import ModuleOutputs, RepresentationBundle

if TYPE_CHECKING:
    from .composite import CompositePlaceModel


@dataclass(slots=True)
class ResolvedBatch:
    observations: Tensor
    actions: Tensor
    valid_steps: Tensor
    kinematics: Tensor | None
    position_xy: Tensor | None


@dataclass(slots=True)
class CorruptionResult:
    corrupted_observations: Tensor
    vision_mask: Tensor
    corruption_regime: Tensor
    noise_mask: Tensor
    blackout_mask: Tensor
    temporal_offset: Tensor | None = None


@dataclass(slots=True)
class TrainingCorruptionPlan:
    noise_mask: Tensor
    blackout_mask: Tensor
    standard_noise: Tensor | None = None


@dataclass(slots=True)
class EncoderTeacherPass:
    clean_observations: Tensor
    corruption: CorruptionResult
    encoder_outputs: ModuleOutputs
    teacher_outputs: ModuleOutputs


@dataclass(slots=True)
class PlaceModelChunkState:
    encoder_state: Any
    teacher_state: Any
    predictor_state: Any
    previous_encoder_code: Tensor
    previous_action: Tensor
    previous_kinematics: Tensor | None
    previous_corruption_regime: Tensor
    previous_temporal_offset: Tensor | None
    previous_valid_step: Tensor

    def detached(self) -> PlaceModelChunkState:
        return PlaceModelChunkState(
            encoder_state=detach_state(self.encoder_state),
            teacher_state=detach_state(self.teacher_state),
            predictor_state=detach_state(self.predictor_state),
            previous_encoder_code=self.previous_encoder_code.detach(),
            previous_action=self.previous_action.detach(),
            previous_kinematics=(
                None if self.previous_kinematics is None else self.previous_kinematics.detach()
            ),
            previous_corruption_regime=self.previous_corruption_regime.detach(),
            previous_temporal_offset=(
                None
                if self.previous_temporal_offset is None
                else self.previous_temporal_offset.detach()
            ),
            previous_valid_step=self.previous_valid_step.detach(),
        )


def resolve_batch(batch: dict[str, Tensor]) -> ResolvedBatch:
    return ResolvedBatch(
        observations=resolve_observations(batch),
        actions=resolve_actions(batch),
        valid_steps=resolve_valid_steps(batch),
        kinematics=resolve_kinematics(batch),
        position_xy=resolve_position_xy(batch),
    )


def _expand_mask(mask: Tensor, reference_tensor: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < reference_tensor.ndim:
        expanded = expanded.unsqueeze(-1)
    return expanded


def _delay_observations(observations: Tensor, delay_steps: int) -> Tensor:
    if delay_steps <= 0:
        return observations
    delayed = torch.zeros_like(observations)
    if delay_steps < observations.shape[1]:
        delayed[:, delay_steps:] = observations[:, :-delay_steps]
    return delayed


def _make_periodic_vision_mask(
    valid_steps: Tensor,
    stride: int,
    random_offset: bool,
    generator: torch.Generator | None = None,
) -> Tensor:
    batch_size, time_steps = valid_steps.shape
    if stride <= 1 or time_steps <= 0:
        return torch.zeros_like(valid_steps, dtype=torch.bool)
    time_indices = torch.arange(time_steps, device=valid_steps.device).unsqueeze(0)
    offsets = (
        torch.randint(
            0,
            stride,
            (batch_size, 1),
            device=valid_steps.device,
            generator=generator,
        )
        if random_offset
        else torch.zeros((batch_size, 1), dtype=torch.long, device=valid_steps.device)
    )
    blackout_mask = ((time_indices - offsets) % stride) != 0
    blackout_mask[:, 0] = False
    return blackout_mask & valid_steps


def _accumulate_random_segments(
    valid_lengths: Tensor,
    floors: Tensor,
    caps: Tensor,
    start_offset: int,
    num_blocks: int,
    eligible: Tensor,
    time_indices: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """OR together num_blocks random [start, start + length) segments per row."""
    batch_size = valid_lengths.shape[0]
    device = valid_lengths.device
    mask = torch.zeros((batch_size, time_indices.shape[1]), dtype=torch.bool, device=device)
    if num_blocks <= 0:
        return mask

    random_draws = torch.rand(
        (num_blocks, 2, batch_size),
        device=device,
        generator=generator,
    )
    length_span = (caps - floors + 1).unsqueeze(0)
    lengths = floors.unsqueeze(0) + torch.floor(random_draws[:, 0] * length_span).long()
    start_span = (valid_lengths.unsqueeze(0) - lengths - start_offset + 1).clamp_min(1)
    starts = start_offset + torch.floor(random_draws[:, 1] * start_span).long()
    expanded_time_indices = time_indices.unsqueeze(0)
    in_block = (expanded_time_indices >= starts.unsqueeze(-1)) & (
        expanded_time_indices < (starts + lengths).unsqueeze(-1)
    )
    return (in_block & eligible.view(1, batch_size, 1)).any(dim=0)


def _sample_random_blackout_segments(
    valid_steps: Tensor,
    blackout_probability: float,
    num_segments: int,
    min_length: int,
    max_length: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    batch_size, time_steps = valid_steps.shape
    if blackout_probability <= 0.0 or time_steps <= 1:
        return torch.zeros_like(valid_steps, dtype=torch.bool)
    if min_length <= 0 or max_length < min_length:
        return torch.zeros_like(valid_steps, dtype=torch.bool)

    apply_blackout = (
        torch.rand(batch_size, device=valid_steps.device, generator=generator)
        < blackout_probability
    )
    valid_lengths = valid_steps.long().sum(dim=1)
    available_lengths = (valid_lengths - 1).clamp_min(0)
    eligible = available_lengths > 0
    minimum_lengths = torch.minimum(
        torch.full_like(available_lengths, min_length),
        available_lengths,
    ).clamp_min(1)
    maximum_lengths = torch.minimum(
        torch.full_like(available_lengths, max_length),
        available_lengths,
    ).clamp_min(1)
    time_indices = torch.arange(time_steps, device=valid_steps.device).unsqueeze(0)
    blackout_mask = _accumulate_random_segments(
        valid_lengths,
        floors=minimum_lengths,
        caps=maximum_lengths,
        start_offset=1,
        num_blocks=num_segments,
        eligible=eligible,
        time_indices=time_indices,
        generator=generator,
    )
    blackout_mask = blackout_mask & apply_blackout.unsqueeze(1) & valid_steps
    blackout_mask[:, 0] = False
    return blackout_mask


def _sample_input_corruption(
    valid_steps: Tensor,
    context_warmup: int,
    noise_num_blocks: int,
    blackout_num_blocks: int,
    noise_min_length: int = 1,
    noise_max_length: int = 10,
    blackout_min_length: int = 1,
    blackout_max_length: int = 0,
    blackout_schedule: str = "random",
    blackout_period_grounded: int = 8,
    blackout_period_masked: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    batch_size, time_steps = valid_steps.shape
    noise_mask = torch.zeros_like(valid_steps, dtype=torch.bool)
    blackout_mask = torch.zeros_like(valid_steps, dtype=torch.bool)
    warmup = max(0, min(context_warmup, time_steps))
    if time_steps <= warmup:
        return noise_mask, blackout_mask

    periodic_blackout = blackout_schedule == "periodic"
    if periodic_blackout:
        period = blackout_period_grounded + blackout_period_masked
        step_indices = torch.arange(time_steps, device=valid_steps.device)
        phase = (step_indices - warmup) % period
        periodic = (phase >= blackout_period_grounded) & (step_indices >= warmup)
        blackout_mask = periodic.unsqueeze(0).expand(batch_size, -1).clone()

    valid_lengths = valid_steps.long().sum(dim=1)
    available_lengths = (valid_lengths - warmup).clamp_min(0)
    if not torch.any(available_lengths > 0):
        return noise_mask, blackout_mask

    time_indices = torch.arange(time_steps, device=valid_steps.device).unsqueeze(0)
    blackout_caps = (
        available_lengths
        if blackout_max_length <= 0
        else torch.minimum(
            available_lengths,
            torch.full_like(available_lengths, blackout_max_length),
        )
    )
    if blackout_num_blocks > 0 and not periodic_blackout:
        blackout_floors = torch.minimum(
            torch.full_like(blackout_caps, max(1, blackout_min_length)),
            blackout_caps,
        ).clamp_min(1)
        blackout_mask = _accumulate_random_segments(
            valid_lengths,
            floors=blackout_floors,
            caps=blackout_caps.clamp_min(1),
            start_offset=warmup,
            num_blocks=blackout_num_blocks,
            eligible=blackout_caps > 0,
            time_indices=time_indices,
            generator=generator,
        )
    blackout_mask[:, :warmup] = False

    if noise_num_blocks > 0:
        noise_caps = torch.minimum(
            available_lengths,
            torch.full_like(available_lengths, noise_max_length),
        )
        noise_floors = torch.minimum(
            torch.full_like(noise_caps, noise_min_length),
            noise_caps,
        ).clamp_min(1)
        noise_mask = _accumulate_random_segments(
            valid_lengths,
            floors=noise_floors,
            caps=noise_caps.clamp_min(1),
            start_offset=warmup,
            num_blocks=noise_num_blocks,
            eligible=noise_caps > 0,
            time_indices=time_indices,
            generator=generator,
        )
    noise_mask[:, :warmup] = False
    noise_mask = noise_mask & ~blackout_mask
    noise_mask = noise_mask & valid_steps
    blackout_mask = blackout_mask & valid_steps
    return noise_mask, blackout_mask


def _compute_temporal_offset(corruption_regime: Tensor, scale: float = 32.0) -> Tensor:
    batch_size, time_steps = corruption_regime.shape
    time_indices = (
        torch.arange(
            time_steps,
            device=corruption_regime.device,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    clean_step_indices = torch.where(
        corruption_regime == 0,
        time_indices,
        torch.zeros_like(time_indices),
    )
    inclusive_last_clean = torch.cummax(clean_step_indices, dim=1).values
    previous_last_clean = torch.cat(
        [
            torch.zeros(batch_size, 1, device=corruption_regime.device, dtype=torch.long),
            inclusive_last_clean[:, :-1],
        ],
        dim=1,
    )
    offset = (time_indices - previous_last_clean).clamp_min(0).to(torch.float32)
    return (offset / float(scale)).unsqueeze(-1)


def _sample_training_corruption_plan(
    model: CompositePlaceModel,
    observations: Tensor,
    valid_steps: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> TrainingCorruptionPlan:
    inputs_config = model.components.config.inputs
    masking_config = inputs_config.visual_masking
    corruption_config = inputs_config.input_corruption
    empty_mask = torch.zeros_like(valid_steps, dtype=torch.bool)
    if corruption_config.enabled:
        noise_is_enabled = corruption_config.noise_num_blocks > 0 and (
            corruption_config.noise_sigma_abs > 0.0 or corruption_config.noise_sigma_rel > 0.0
        )
        noise_mask, blackout_mask = _sample_input_corruption(
            valid_steps,
            context_warmup=corruption_config.context_warmup,
            noise_num_blocks=(corruption_config.noise_num_blocks if noise_is_enabled else 0),
            blackout_num_blocks=corruption_config.blackout_num_blocks,
            noise_min_length=corruption_config.noise_min_length,
            noise_max_length=corruption_config.noise_max_length,
            blackout_min_length=corruption_config.blackout_min_length,
            blackout_max_length=corruption_config.blackout_max_length,
            blackout_schedule=corruption_config.blackout_schedule,
            blackout_period_grounded=corruption_config.blackout_period_grounded,
            blackout_period_masked=corruption_config.blackout_period_masked,
            generator=generator,
        )
        standard_noise = (
            torch.randn(
                observations.shape,
                device=observations.device,
                dtype=observations.dtype,
                generator=generator,
            )
            if noise_is_enabled
            else None
        )
        return TrainingCorruptionPlan(
            noise_mask=noise_mask,
            blackout_mask=blackout_mask,
            standard_noise=standard_noise,
        )

    blackout_mask = _sample_random_blackout_segments(
        valid_steps,
        blackout_probability=masking_config.blackout_probability,
        num_segments=masking_config.blackout_num_segments,
        min_length=masking_config.blackout_min_length,
        max_length=masking_config.blackout_max_length,
        generator=generator,
    ) | _make_periodic_vision_mask(
        valid_steps,
        stride=masking_config.stride,
        random_offset=masking_config.stride_random_offset,
        generator=generator,
    )
    return TrainingCorruptionPlan(noise_mask=empty_mask, blackout_mask=blackout_mask)


def _valid_feature_standard_deviation(
    observations: Tensor,
    valid_steps: Tensor,
) -> Tensor:
    valid_mask = _expand_mask(valid_steps, observations).to(observations.dtype)
    reduction_dims = (0, 1)
    valid_count = valid_mask.sum(dim=reduction_dims, keepdim=True).clamp_min(1.0)
    mean = (observations * valid_mask).sum(dim=reduction_dims, keepdim=True) / valid_count
    variance = ((observations - mean).square() * valid_mask).sum(
        dim=reduction_dims, keepdim=True
    ) / valid_count
    return variance.sqrt().detach()


def _apply_training_corruption(
    model: CompositePlaceModel,
    observations: Tensor,
    valid_steps: Tensor,
) -> CorruptionResult:
    inputs_config = model.components.config.inputs
    masking_config = inputs_config.visual_masking
    corruption_config = inputs_config.input_corruption
    empty_mask = torch.zeros_like(valid_steps, dtype=torch.bool)
    plan = getattr(model, "_training_corruption_plan", None)
    capture_plan = bool(getattr(model, "_capture_training_corruption_plan", False))
    corruption_is_configured = (
        corruption_config.enabled
        or masking_config.blackout_probability > 0.0
        or masking_config.stride > 1
    )
    if (not model.training and plan is None and not capture_plan) or not corruption_is_configured:
        return CorruptionResult(
            corrupted_observations=observations,
            vision_mask=valid_steps.clone(),
            corruption_regime=torch.zeros_like(valid_steps, dtype=torch.long),
            noise_mask=empty_mask,
            blackout_mask=empty_mask,
        )

    if plan is None:
        plan = _sample_training_corruption_plan(
            model,
            observations,
            valid_steps,
            generator=getattr(model, "_training_corruption_generator", None),
        )
    if capture_plan:
        model._captured_training_corruption_plan = plan

    noise_mask = plan.noise_mask
    blackout_mask = plan.blackout_mask
    corrupted = observations
    if corruption_config.enabled:
        if plan.standard_noise is not None:
            if corruption_config.noise_sigma_rel > 0.0:
                noise_scale: Tensor | float = (
                    _valid_feature_standard_deviation(observations, valid_steps)
                    * corruption_config.noise_sigma_rel
                )
            else:
                noise_scale = float(corruption_config.noise_sigma_abs)
            noise_scale_tensor = torch.as_tensor(
                noise_scale,
                device=plan.standard_noise.device,
                dtype=plan.standard_noise.dtype,
            )
            noise = plan.standard_noise * noise_scale_tensor
            expanded_noise_mask = _expand_mask(noise_mask, corrupted)
            if corruption_config.noise_type == "multiplicative":
                corrupted = torch.where(
                    expanded_noise_mask,
                    corrupted * torch.exp(noise - 0.5 * noise_scale_tensor.square()),
                    corrupted,
                )
            else:
                corrupted = torch.where(expanded_noise_mask, corrupted + noise, corrupted)
        blackout_fill = torch.zeros_like(corrupted)
        if corruption_config.use_blackout_token:
            if model.input_corruption_blackout_token is None:
                raise ValueError(
                    "Input corruption blackout token is enabled, but the model has no "
                    "blackout token."
                )
            blackout_fill = model.input_corruption_blackout_token.to(
                device=corrupted.device,
                dtype=corrupted.dtype,
            )
        corrupted = torch.where(_expand_mask(blackout_mask, corrupted), blackout_fill, corrupted)
    else:
        corrupted = torch.where(
            _expand_mask(blackout_mask, corrupted),
            torch.zeros_like(corrupted),
            corrupted,
        )

    vision_mask = (~blackout_mask) & valid_steps
    corruption_regime = torch.where(
        blackout_mask,
        torch.full_like(valid_steps, 2, dtype=torch.long),
        torch.where(
            noise_mask,
            torch.ones_like(valid_steps, dtype=torch.long),
            torch.zeros_like(valid_steps, dtype=torch.long),
        ),
    )
    temporal_offset = (
        _compute_temporal_offset(corruption_regime)
        if corruption_config.enabled and corruption_config.append_temporal_offset
        else None
    )
    return CorruptionResult(
        corrupted_observations=corrupted,
        vision_mask=vision_mask,
        corruption_regime=corruption_regime,
        noise_mask=noise_mask,
        blackout_mask=blackout_mask,
        temporal_offset=temporal_offset,
    )


def _encode_sequence_stateful(
    model: CompositePlaceModel,
    resolved_batch: ResolvedBatch,
    *,
    encoder_state: Any = None,
    teacher_state: Any = None,
) -> tuple[EncoderTeacherPass, Any, Any]:
    clean_observations = resolved_batch.observations.float()
    if model.components.config.inputs.observation_source == "rgb":
        clean_observations = clean_observations / 255.0
    current_clean_observations = clean_observations
    encoder_observations = _delay_observations(
        current_clean_observations,
        model.components.config.inputs.encoder_observation_delay_steps,
    )
    corruption = _apply_training_corruption(
        model,
        encoder_observations,
        resolved_batch.valid_steps,
    )
    encoder_outputs, next_encoder_state = model.encoder_stack.forward_stateful(
        corruption.corrupted_observations,
        actions=resolved_batch.actions,
        kinematics=resolved_batch.kinematics,
        valid_steps=resolved_batch.valid_steps,
        training_regularizer=model.training_regularizer,
        initial_state=encoder_state,
    )
    teacher_input = (
        current_clean_observations
        if model.components.config.teacher_student.teacher_sees_unmasked
        else corruption.corrupted_observations
    )
    teacher_outputs = None
    next_teacher_state = None
    if model.teacher_controller is not None and model.teacher_controller.enabled:
        teacher_full_outputs, next_teacher_state = model.teacher_controller.encode_stateful(
            teacher_input,
            actions=resolved_batch.actions,
            kinematics=resolved_batch.kinematics,
            valid_steps=resolved_batch.valid_steps,
            initial_state=teacher_state,
        )
        if teacher_full_outputs is not None:
            teacher_outputs = ModuleOutputs(
                place_codes=teacher_full_outputs.place_codes,
                pre_sparsifier=teacher_full_outputs.pre_sparsifier,
                backbone_output=teacher_full_outputs.backbone_output,
                state_readout=teacher_full_outputs.state_readout,
                auxiliary={
                    name: value
                    for name, value in teacher_full_outputs.auxiliary.items()
                    if name.startswith("adaptive_state_")
                },
            )
    if teacher_outputs is None:
        encoder_pre_sparsifier = encoder_outputs.pre_sparsifier
        teacher_outputs = ModuleOutputs(
            place_codes=encoder_outputs.place_codes.detach(),
            pre_sparsifier=(
                None if encoder_pre_sparsifier is None else encoder_pre_sparsifier.detach()
            ),
            backbone_output=(
                None
                if encoder_outputs.backbone_output is None
                else encoder_outputs.backbone_output.detach()
            ),
            state_readout=(
                None
                if encoder_outputs.state_readout is None
                else encoder_outputs.state_readout.detach()
            ),
        )
    return (
        EncoderTeacherPass(
            clean_observations=current_clean_observations,
            corruption=corruption,
            encoder_outputs=encoder_outputs,
            teacher_outputs=teacher_outputs,
        ),
        next_encoder_state,
        next_teacher_state,
    )


def _encode_sequence(
    model: CompositePlaceModel,
    resolved_batch: ResolvedBatch,
) -> EncoderTeacherPass:
    encoder_teacher_pass, _encoder_state, _teacher_state = _encode_sequence_stateful(
        model,
        resolved_batch,
    )
    return encoder_teacher_pass


def _rollout_predictor(
    model: CompositePlaceModel,
    resolved_batch: ResolvedBatch,
    encoder_outputs: ModuleOutputs,
    corruption_regime: Tensor,
    temporal_offset: Tensor | None,
) -> ModuleOutputs:
    predictor_input_codes = _prediction_gradient_view(model, encoder_outputs.place_codes)
    predictor_outputs, _, _ = rollout_predictor_sequence(
        model,
        predictor_input_codes,
        context_from_inputs(
            model,
            actions=resolved_batch.actions,
            kinematics_tensor=resolved_batch.kinematics,
            corruption_regime=corruption_regime,
            temporal_offset=temporal_offset,
        ),
        regularizer=model.training_regularizer,
        detach_intermediate_predictions=bool(
            model.components.config.rollout.detach_intermediate_predictions
        ),
    )
    return predictor_outputs


def _slow_operator_stack(model: CompositePlaceModel) -> DetachedPredictorStack:
    """The EMA predictor as a PredictorStack, so it runs the identical rollout code."""
    controller = model.teacher_controller
    if controller is None or not controller.predictor_enabled:
        raise RuntimeError("The slow operator needs an attached predictor teacher.")
    return controller.teacher_predictor_stack().bind(model.components)


def _rollout_slow_operator(
    model: CompositePlaceModel,
    resolved_batch: ResolvedBatch,
    encoder_teacher_pass: EncoderTeacherPass,
) -> ModuleOutputs | None:
    """Roll the EMA predictor one step per timestep."""
    controller = model.teacher_controller
    if controller is None or not controller.predictor_enabled:
        return None
    controller.ensure_predictor_teacher_synced()
    online_side = model.components.config.teacher_student.slow_operator_side == "online"
    controller.set_predictor_teacher_backward_mode(online_side and torch.is_grad_enabled())
    if online_side:
        source_codes = encoder_teacher_pass.encoder_outputs.place_codes
        if source_codes is None:
            raise RuntimeError(
                "slow_operator_side='online' needs encoder.place_codes to differentiate through."
            )
    else:
        teacher_codes = encoder_teacher_pass.teacher_outputs.place_codes
        if teacher_codes is None:
            raise RuntimeError(
                "teacher_student.predictor_ema needs teacher.place_codes; the encoder teacher "
                "produced none."
            )
        source_codes = teacher_codes.detach()
    slow_operator = _slow_operator_stack(model)

    def _run() -> ModuleOutputs:
        outputs, _states, _after = rollout_predictor_sequence(
            slow_operator,
            source_codes,
            context_from_inputs(
                slow_operator,
                actions=resolved_batch.actions,
                kinematics_tensor=resolved_batch.kinematics,
                corruption_regime=encoder_teacher_pass.corruption.corruption_regime,
                temporal_offset=encoder_teacher_pass.corruption.temporal_offset,
            ),
            regularizer=None,
            detach_intermediate_predictions=False,
        )
        return outputs

    if online_side:
        return _run()
    with torch.no_grad():
        return _run()


def _project_for_slow_operator(
    model: CompositePlaceModel,
    encoder_outputs: ModuleOutputs,
) -> Tensor | None:
    """Trainable online-side projection that restores the BYOL asymmetry."""
    projection = model.slow_operator_projection
    if projection is None:
        return None
    if encoder_outputs.place_codes is None:
        raise RuntimeError("slow_operator_projection_hidden requires encoder.place_codes.")
    return projection(encoder_outputs.place_codes)


def _prediction_gradient_view(model: CompositePlaceModel, encoder_codes: Tensor) -> Tensor:
    scale = float(model.components.config.encoder_gradient_scale)
    if scale == 1.0 or not encoder_codes.requires_grad:
        return encoder_codes
    return encoder_codes.detach() + scale * (encoder_codes - encoder_codes.detach())


def _open_loop_rollout(
    model: CompositePlaceModel,
    resolved_batch: ResolvedBatch,
    encoder_outputs: ModuleOutputs,
    corruption_regime: Tensor,
    temporal_offset: Tensor | None,
    reanchor_steps: int,
) -> ModuleOutputs:
    return open_loop_rollout_predictor_sequence(
        model,
        encoder_outputs.place_codes,
        context_from_inputs(
            model,
            actions=resolved_batch.actions,
            kinematics_tensor=resolved_batch.kinematics,
            corruption_regime=corruption_regime,
            temporal_offset=temporal_offset,
        ),
        reanchor_steps=reanchor_steps,
    )


def _bundle_representations(
    model: CompositePlaceModel,
    resolved_batch: ResolvedBatch,
    encoder_teacher_pass: EncoderTeacherPass,
    predictor_outputs: ModuleOutputs,
    open_loop_outputs: ModuleOutputs | None = None,
) -> RepresentationBundle:
    actions = resolved_batch.actions
    empty_placeholder = torch.empty(0, device=actions.device)
    valid_steps = resolved_batch.valid_steps
    noise_mask = encoder_teacher_pass.corruption.noise_mask & valid_steps
    blackout_mask = encoder_teacher_pass.corruption.blackout_mask & valid_steps
    corruption_mask = noise_mask | blackout_mask
    valid_count = valid_steps.sum().clamp_min(1).to(torch.float32)
    corruption_fractions = {
        "noise": noise_mask.sum().to(torch.float32) / valid_count,
        "blackout": blackout_mask.sum().to(torch.float32) / valid_count,
        "any": corruption_mask.sum().to(torch.float32) / valid_count,
    }
    external_observations = resolved_batch.observations.detach()
    bundle = RepresentationBundle(
        modules={
            "encoder": encoder_teacher_pass.encoder_outputs,
            "predictor": predictor_outputs,
            "teacher": encoder_teacher_pass.teacher_outputs,
            "observation": ModuleOutputs(backbone_output=external_observations),
        },
        masks={
            "valid_steps": valid_steps,
            "vision_mask": encoder_teacher_pass.corruption.vision_mask,
            "corruption_regime": encoder_teacher_pass.corruption.corruption_regime,
            "noise_mask": noise_mask,
            "blackout_mask": blackout_mask,
            "corruption_mask": corruption_mask,
        },
        inputs={
            "actions": actions,
            "kinematics": (
                resolved_batch.kinematics
                if resolved_batch.kinematics is not None
                else empty_placeholder
            ),
            "position_xy": (
                resolved_batch.position_xy
                if resolved_batch.position_xy is not None
                else empty_placeholder
            ),
            "temporal_offset": (
                encoder_teacher_pass.corruption.temporal_offset
                if encoder_teacher_pass.corruption.temporal_offset is not None
                else empty_placeholder
            ),
        },
        metadata={
            "teacher_student_mode": model.components.config.teacher_student.mode,
            "predictor_input_mode": model.components.config.inputs.predictor_input_mode,
            "prediction_bootstrap": {
                "active": bool(
                    model.training and model.components.config.prediction_bootstrap.enabled
                ),
                "gamma": float(model.components.config.prediction_bootstrap.gamma),
                "loss_type": model.components.config.prediction_bootstrap.loss_type,
            },
            "rollout_detach_intermediate_predictions": bool(
                model.components.config.rollout.detach_intermediate_predictions
            ),
            "corruption_fractions": {
                name: value.detach() for name, value in corruption_fractions.items()
            },
        },
        views=representation_view_specs(model.components.config),
    )
    if open_loop_outputs is not None:
        bundle.modules["predictor_rollout"] = open_loop_outputs
    if model.inverse_dynamics_head is not None:
        encoder_codes = encoder_teacher_pass.encoder_outputs.place_codes
        if (
            encoder_codes is not None
            and resolved_batch.kinematics is not None
            and encoder_codes.shape[1] > 1
        ):
            inverse_dynamics_inputs = torch.cat(
                [encoder_codes[:, :-1], encoder_codes[:, 1:]],
                dim=-1,
            )
            bundle.auxiliary_outputs["inverse_dynamics.prediction"] = model.inverse_dynamics_head(
                inverse_dynamics_inputs
            )
    return bundle


def forward_sequence(model: CompositePlaceModel, batch: dict[str, Tensor]) -> RepresentationBundle:
    """Run the full sequence model using the module's train/eval mode."""
    if model.training_regularizer is not None:
        model.training_regularizer.reset_sequence()
    resolved_batch = resolve_batch(batch)
    encoder_teacher_pass = _encode_sequence(model, resolved_batch)
    return forward_from_encoded(model, batch, resolved_batch, encoder_teacher_pass)


def forward_from_encoded(
    model: CompositePlaceModel,
    batch: dict[str, Tensor],
    resolved_batch: ResolvedBatch,
    encoder_teacher_pass: EncoderTeacherPass,
) -> RepresentationBundle:
    """Run predictors and heads on already computed student and teacher encodings."""
    predictor_outputs = _rollout_predictor(
        model,
        resolved_batch,
        encoder_teacher_pass.encoder_outputs,
        encoder_teacher_pass.corruption.corruption_regime,
        encoder_teacher_pass.corruption.temporal_offset,
    )
    open_loop_outputs = None
    open_loop_steps = getattr(model, "export_open_loop_rollout_steps", None)
    if open_loop_steps:
        open_loop_outputs = _open_loop_rollout(
            model,
            resolved_batch,
            encoder_teacher_pass.encoder_outputs,
            encoder_teacher_pass.corruption.corruption_regime,
            encoder_teacher_pass.corruption.temporal_offset,
            int(open_loop_steps),
        )
    slow_operator_outputs = _rollout_slow_operator(model, resolved_batch, encoder_teacher_pass)
    projected_codes = _project_for_slow_operator(model, encoder_teacher_pass.encoder_outputs)
    if projected_codes is not None:
        encoder_teacher_pass.encoder_outputs.auxiliary["slow_operator_projection"] = projected_codes
    bundle = _bundle_representations(
        model,
        resolved_batch,
        encoder_teacher_pass,
        predictor_outputs,
        open_loop_outputs,
    )
    if slow_operator_outputs is not None:
        bundle.modules["teacher_predictor"] = slow_operator_outputs
    if len(model.representation_heads) > 0:
        run_representation_heads(model.representation_heads, bundle, batch)
    if len(model.auxiliary_heads) > 0:
        run_auxiliary_heads(model.auxiliary_heads, bundle, batch)
    return bundle


def _prepend_sequence(previous: Tensor | None, current: Tensor | None) -> Tensor | None:
    if previous is None:
        return current
    if current is None:
        raise ValueError("A carried chunk tensor is missing from the current batch.")
    return torch.cat((previous, current), dim=1)


def _validate_chunk_forward_runtime(model: CompositePlaceModel) -> None:
    if model.training_regularizer is not None and model.training_regularizer.has_active_site(""):
        raise ValueError("Stateful BPTT does not support an active sequence regularizer.")
    if len(model.representation_heads) > 0:
        raise ValueError("Stateful BPTT does not support representation heads.")
    if len(model.auxiliary_heads) > 0:
        raise ValueError("Stateful BPTT does not support objectives with auxiliary heads.")
    if model.export_open_loop_rollout_steps:
        raise ValueError("Stateful BPTT does not support analysis-time predictor rollouts.")


def forward_chunk(
    model: CompositePlaceModel,
    batch: dict[str, Tensor],
    state: PlaceModelChunkState | None = None,
    *,
    new_episode: bool = False,
) -> tuple[RepresentationBundle, PlaceModelChunkState]:
    """Carry state across chunks; new episodes omit the cross-boundary transition."""
    _validate_chunk_forward_runtime(model)
    resolved_batch = resolve_batch(batch)
    encoder_teacher_pass, next_encoder_state, next_teacher_state = _encode_sequence_stateful(
        model,
        resolved_batch,
        encoder_state=None if state is None else state.encoder_state,
        teacher_state=None if state is None else state.teacher_state,
    )
    encoder_codes = encoder_teacher_pass.encoder_outputs.place_codes
    if encoder_codes is None:
        raise RuntimeError("Stateful BPTT requires encoder.place_codes.")

    corruption = encoder_teacher_pass.corruption
    actions = resolved_batch.actions
    kinematics = resolved_batch.kinematics
    corruption_regime = corruption.corruption_regime
    temporal_offset = corruption.temporal_offset
    contiguous = state is not None and not new_episode
    if contiguous:
        actions = _prepend_sequence(state.previous_action, actions)
        kinematics = _prepend_sequence(state.previous_kinematics, kinematics)
        corruption_regime = _prepend_sequence(
            state.previous_corruption_regime,
            corruption_regime,
        )
        temporal_offset = _prepend_sequence(state.previous_temporal_offset, temporal_offset)
    predictor_context = context_from_inputs(
        model,
        actions=actions,
        kinematics_tensor=kinematics,
        corruption_regime=corruption_regime,
        temporal_offset=temporal_offset,
    )
    predictor_outputs, next_predictor_state = rollout_predictor_chunk(
        model,
        _prediction_gradient_view(model, encoder_codes),
        predictor_context,
        initial_state=None if state is None else state.predictor_state,
        previous_encoder_code=state.previous_encoder_code if contiguous else None,
    )
    bundle = _bundle_representations(
        model,
        resolved_batch,
        encoder_teacher_pass,
        predictor_outputs,
    )
    if contiguous:
        boundary_valid = state.previous_valid_step & resolved_batch.valid_steps[:, 0]
        internal_valid = resolved_batch.valid_steps[:, :-1] & resolved_batch.valid_steps[:, 1:]
        bundle.masks["prediction_valid_steps"] = torch.cat(
            (boundary_valid.unsqueeze(1), internal_valid),
            dim=1,
        )

    next_state = PlaceModelChunkState(
        encoder_state=next_encoder_state,
        teacher_state=next_teacher_state,
        predictor_state=next_predictor_state,
        previous_encoder_code=encoder_codes[:, -1],
        previous_action=resolved_batch.actions[:, -1:],
        previous_kinematics=(
            None if resolved_batch.kinematics is None else resolved_batch.kinematics[:, -1:]
        ),
        previous_corruption_regime=corruption.corruption_regime[:, -1:],
        previous_temporal_offset=(
            None if corruption.temporal_offset is None else corruption.temporal_offset[:, -1:]
        ),
        previous_valid_step=resolved_batch.valid_steps[:, -1],
    )
    return bundle, next_state
