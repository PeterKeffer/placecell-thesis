"""Shared predictor runtime helpers for forward passes and training-time replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import torch
from torch import Tensor

from .components.input_assemblers import predictor_assembler_uses_belief
from .components.temporal import temporal_backend_capabilities
from .predictor_context import (
    predictor_kinematics_channels,
    select_predictor_kinematics,
    uses_action_context,
    uses_implicit_transition_shift,
)
from .predictor_stack import PredictorStack
from .types import ModuleOutputs, RepresentationBundle


@dataclass(slots=True)
class PredictorRuntimeContext:
    embedded_actions: Tensor | None
    kinematics: Tensor | None
    kinematics_transition_mask: Tensor | None
    corruption_regime: Tensor
    temporal_offset: Tensor | None


def detach_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach().clone()
    if isinstance(state, tuple):
        detached_items = [detach_state(item) for item in state]
        if hasattr(state, "_fields"):
            return type(state)(*detached_items)
        return tuple(detached_items)
    if isinstance(state, list):
        return [detach_state(item) for item in state]
    return deepcopy(state)


def context_from_inputs(
    model: PredictorStack,
    *,
    actions: Tensor,
    kinematics_tensor: Tensor | None,
    corruption_regime: Tensor,
    temporal_offset: Tensor | None,
) -> PredictorRuntimeContext:
    context_channels = model.components.config.inputs.predictor_context_channels
    selected_kinematics_channels = predictor_kinematics_channels(context_channels)
    kinematics = select_predictor_kinematics(kinematics_tensor, context_channels)
    velocity_scale = float(model.components.config.inputs.kinematics_velocity_scale)
    if kinematics is not None and velocity_scale != 1.0:
        velocity_gain = torch.tensor(
            [
                velocity_scale
                if channel in ("speed", "step_displacement", "angular_velocity")
                else 1.0
                for channel in selected_kinematics_channels
            ],
            device=kinematics.device,
            dtype=kinematics.dtype,
        )
        kinematics = kinematics * velocity_gain
    kinematics_transition_mask = None
    if kinematics is not None:
        kinematics_transition_mask = torch.tensor(
            [
                uses_implicit_transition_shift(channel)
                for channel in selected_kinematics_channels
            ],
            device=kinematics.device,
            dtype=torch.bool,
        )
    embedded_actions = None
    if uses_action_context(context_channels):
        if model.action_embedding is None:
            raise ValueError(
                "Predictor action context is enabled, but the model was built "
                "without an action embedding."
            )
        embedded_actions = model.action_embedding(actions.long())
    return PredictorRuntimeContext(
        embedded_actions=embedded_actions,
        kinematics=kinematics,
        kinematics_transition_mask=kinematics_transition_mask,
        corruption_regime=corruption_regime,
        temporal_offset=temporal_offset,
    )


def context_from_bundle(
    model: PredictorStack,
    bundle: RepresentationBundle,
) -> PredictorRuntimeContext:
    kinematics_tensor = bundle.inputs.get("kinematics")
    if kinematics_tensor is not None and kinematics_tensor.numel() == 0:
        kinematics_tensor = None
    temporal_offset = bundle.inputs.get("temporal_offset")
    if temporal_offset is not None and temporal_offset.numel() == 0:
        temporal_offset = None
    return context_from_inputs(
        model,
        actions=bundle.inputs["actions"],
        kinematics_tensor=kinematics_tensor,
        corruption_regime=bundle.masks["corruption_regime"],
        temporal_offset=temporal_offset,
    )


def predictor_runtime_contract(model: PredictorStack) -> str:
    active_rollout_modes = {
        objective.rollout_state_mode
        for objective in model.components.config.objectives.values()
        if objective.type == "multistep_rollout" and objective.weight > 0.0
    }
    if "reset" in active_rollout_modes:
        return "reset_anchored_self_fed_training_with_teacher_forced_diagnostic_pass"
    if "replay" in active_rollout_modes:
        return "replay_warm_started_self_fed_training_with_teacher_forced_diagnostic_pass"
    if model.components.config.predictor.family == "transformer":
        return "full_sequence_causal_training_with_exact_bounded_step_rollout"
    return "stepwise_recurrent_rollout"


def _kinematics_for_transition(
    context: PredictorRuntimeContext,
    transition_index: int,
) -> Tensor | None:
    if context.kinematics is None:
        return None
    source_state = context.kinematics[:, transition_index]
    arriving_transition = context.kinematics[:, transition_index + 1]
    if context.kinematics_transition_mask is None:
        return arriving_transition
    return torch.where(
        context.kinematics_transition_mask.unsqueeze(0),
        arriving_transition,
        source_state,
    )


def _kinematics_for_transition_sequence(context: PredictorRuntimeContext) -> Tensor | None:
    if context.kinematics is None:
        return None
    source_states = context.kinematics[:, :-1]
    arriving_transitions = context.kinematics[:, 1:]
    if context.kinematics_transition_mask is None:
        return arriving_transitions
    return torch.where(
        context.kinematics_transition_mask.view(1, 1, -1),
        arriving_transitions,
        source_states,
    )


def assemble_predictor_input(
    model: PredictorStack,
    context: PredictorRuntimeContext,
    *,
    encoder_code: Tensor,
    belief_code: Tensor,
    transition_index: int,
) -> Tensor:
    previous_action_embedding = (
        None if context.embedded_actions is None else context.embedded_actions[:, transition_index]
    )
    previous_kinematics = _kinematics_for_transition(context, transition_index)
    previous_temporal_offset = (
        None if context.temporal_offset is None else context.temporal_offset[:, transition_index]
    )
    previous_corruption_regime = context.corruption_regime[:, transition_index]
    corruption_info = {
        "noise_level": previous_corruption_regime.float().unsqueeze(-1),
        "is_blackout": (previous_corruption_regime == 2).float().unsqueeze(-1),
    }
    return model.predictor_input_assembler.assemble(
        encoder_code=encoder_code,
        belief_code=belief_code,
        action_embedding=previous_action_embedding,
        kinematics=previous_kinematics,
        temporal_offset=previous_temporal_offset,
        corruption_info=corruption_info,
    )


def assemble_predictor_input_sequence(
    model: PredictorStack,
    context: PredictorRuntimeContext,
    *,
    encoder_codes: Tensor,
    belief_codes: Tensor | None = None,
) -> Tensor:
    batch_size, time_steps, _ = encoder_codes.shape
    if time_steps <= 1:
        return encoder_codes.new_zeros(
            batch_size,
            0,
            model.predictor_input_assembler.output_dim,
        )
    previous_encoder_codes = encoder_codes[:, :-1]
    previous_belief_codes = (
        previous_encoder_codes if belief_codes is None else belief_codes[:, :-1]
    )
    previous_action_embeddings = (
        None if context.embedded_actions is None else context.embedded_actions[:, :-1]
    )
    previous_kinematics = _kinematics_for_transition_sequence(context)
    previous_temporal_offsets = (
        None if context.temporal_offset is None else context.temporal_offset[:, :-1]
    )
    previous_corruption_regime = context.corruption_regime[:, :-1]
    corruption_info = {
        "noise_level": previous_corruption_regime.float().unsqueeze(-1),
        "is_blackout": (previous_corruption_regime == 2).float().unsqueeze(-1),
    }
    return model.predictor_input_assembler.assemble(
        encoder_code=previous_encoder_codes,
        belief_code=previous_belief_codes,
        action_embedding=previous_action_embeddings,
        kinematics=previous_kinematics,
        temporal_offset=previous_temporal_offsets,
        corruption_info=corruption_info,
    )


def residual_dynamics_enabled(model: PredictorStack) -> bool:
    """True when the predictor head outputs a delta added to the fed code (see schema comment)."""
    return bool(model.components.config.predictor_residual_dynamics)


def predict_code_from_input(
    model: PredictorStack,
    predictor_input: Tensor,
    predictor_state: Any,
    *,
    regularizer: Any = None,
    timestep: int | None = None,
    residual_base: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
    if regularizer is not None:
        predictor_input = regularizer.apply("predictor.input", predictor_input)
    transition_state = None
    wrapped = isinstance(predictor_state, PredictorTransitionState)
    if wrapped:
        transition_state = predictor_state.transition_state
        predictor_state = predictor_state.temporal_state
    if model.components.config.predictor.family == "clockwork":
        if timestep is None:
            raise ValueError("clockwork predictor forward_step requires an absolute timestep.")
        hidden_t, next_state = model.predictor_temporal.forward_step(
            predictor_input,
            predictor_state,
            t=timestep,
        )
    else:
        hidden_t, next_state = model.predictor_temporal.forward_step(
            predictor_input,
            predictor_state,
        )
    if regularizer is not None:
        hidden_t = regularizer.apply("predictor.hidden", hidden_t)
    logits_t = model.predictor_head.project(hidden_t)
    transition_binder = getattr(model, "predictor_transition_binder", None)
    if transition_binder is not None:
        logits_t = logits_t + transition_binder.read_step(predictor_input, transition_state)
        next_state = PredictorTransitionState(next_state, transition_state)
    if regularizer is not None:
        logits_t = regularizer.apply("predictor.logits", logits_t)
    activated_t = model.predictor_head.activate(logits_t)
    if residual_base is not None:
        activated_t = activated_t + residual_base
    if regularizer is not None:
        activated_t = regularizer.apply("predictor.pre_sparsifier", activated_t)
    codes_t = model.predictor_sparsifier(activated_t)
    if regularizer is not None:
        codes_t = regularizer.apply("predictor.output", codes_t)
    return hidden_t, logits_t, activated_t, codes_t, next_state


def _sparsifier_auxiliary(sparsifier: Any) -> dict[str, Tensor]:
    """One sparsifier's own diagnostics, as the encoder stack publishes its own."""
    return dict(getattr(sparsifier, "last_auxiliary_outputs", {}))


class PredictorTransitionState(NamedTuple):
    temporal_state: Any
    transition_state: Tensor | None


def _hidden_layers_from_predictor_state(state: Any) -> list[Tensor]:
    if isinstance(state, PredictorTransitionState):
        state = state.temporal_state
    if state is None:
        return []
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and all(isinstance(item, list) for item in state)
    ):
        hidden_layers, _cell_layers = state
        return list(hidden_layers)
    if isinstance(state, list) and all(isinstance(item, Tensor) for item in state):
        return list(state)
    if isinstance(state, list) and all(
        isinstance(getattr(item, "hidden_state", None), Tensor) for item in state
    ):
        return [item.hidden_state for item in state]
    return []


def _run_fused_predictor_sequence(
    model: PredictorStack,
    encoder_codes: Tensor,
    context: PredictorRuntimeContext,
    *,
    regularizer: Any = None,
    initial_state: Any = None,
    previous_encoder_code: Tensor | None = None,
) -> tuple[ModuleOutputs, Any]:
    batch_size, time_steps, code_dim = encoder_codes.shape
    predictor_hidden_dim = model.components.config.predictor.output_size
    dtype = encoder_codes.dtype
    device = encoder_codes.device
    predictor_hidden_tensor = torch.zeros(
        batch_size, time_steps, predictor_hidden_dim, device=device, dtype=dtype
    )
    predictor_logits_tensor = torch.zeros(
        batch_size, time_steps, code_dim, device=device, dtype=dtype
    )
    predictor_pre_sparsifier_tensor = torch.zeros(
        batch_size, time_steps, code_dim, device=device, dtype=dtype
    )
    predictor_codes_tensor = torch.zeros(
        batch_size, time_steps, code_dim, device=device, dtype=dtype
    )
    if previous_encoder_code is None:
        predictor_codes_tensor[:, 0] = encoder_codes[:, 0]
        predictor_input_codes = encoder_codes
        output_start = 1
    else:
        predictor_input_codes = torch.cat(
            (previous_encoder_code.unsqueeze(1), encoder_codes),
            dim=1,
        )
        output_start = 0
    predictor_inputs = assemble_predictor_input_sequence(
        model,
        context,
        encoder_codes=predictor_input_codes,
        belief_codes=predictor_input_codes,
    )
    if predictor_inputs.size(1) == 0:
        next_state = initial_state
        if next_state is None:
            next_state = model.predictor_temporal.initial_state(batch_size, device)
            if getattr(model, "predictor_transition_binder", None) is not None:
                next_state = PredictorTransitionState(next_state, None)
        return (
            ModuleOutputs(
                place_codes=predictor_codes_tensor,
                place_logits=predictor_logits_tensor,
                pre_sparsifier=predictor_pre_sparsifier_tensor,
                hidden_state=predictor_hidden_tensor,
                auxiliary={},
            ),
            next_state,
        )
    if regularizer is not None:
        predictor_inputs = regularizer.apply("predictor.input", predictor_inputs)
    temporal_initial_state = (
        initial_state.temporal_state
        if isinstance(initial_state, PredictorTransitionState)
        else initial_state
    )
    if hasattr(model.predictor_temporal, "forward_sequence_with_layer_outputs"):
        hidden_sequence, next_state, layer_outputs = (
            model.predictor_temporal.forward_sequence_with_layer_outputs(predictor_inputs)
            if temporal_initial_state is None
            else model.predictor_temporal.forward_sequence_with_layer_outputs(
                predictor_inputs,
                temporal_initial_state,
            )
        )
    else:
        hidden_sequence, next_state = model.predictor_temporal.forward_sequence(
            predictor_inputs,
            temporal_initial_state,
        )
        layer_outputs = []
    if regularizer is not None:
        hidden_sequence = regularizer.apply("predictor.hidden", hidden_sequence)
    logits_sequence = model.predictor_head.project(hidden_sequence)
    transition_binder = getattr(model, "predictor_transition_binder", None)
    transition_state = None
    auxiliary_extra: dict[str, Tensor] = {}
    if transition_binder is not None:
        if isinstance(initial_state, PredictorTransitionState):
            temporal_seed = initial_state.temporal_state
            transition_state = initial_state.transition_state
        else:
            temporal_seed = initial_state
        del temporal_seed
        target_codes = encoder_codes[:, output_start:]
        read_contribution, transition_state = transition_binder.forward_sequence(
            predictor_inputs, target_codes, transition_state
        )
        logits_sequence = logits_sequence + read_contribution
        auxiliary_extra = dict(transition_binder.last_auxiliary_outputs)
    if regularizer is not None:
        logits_sequence = regularizer.apply("predictor.logits", logits_sequence)
    activated_sequence = model.predictor_head.activate(logits_sequence)
    if residual_dynamics_enabled(model):
        activated_sequence = activated_sequence + predictor_input_codes[:, :-1]
    if regularizer is not None:
        activated_sequence = regularizer.apply("predictor.pre_sparsifier", activated_sequence)
    codes_sequence = model.predictor_sparsifier(activated_sequence)
    auxiliary_extra.update(_sparsifier_auxiliary(model.predictor_sparsifier))
    if regularizer is not None:
        codes_sequence = regularizer.apply("predictor.output", codes_sequence)
    predictor_hidden_tensor[:, output_start:] = hidden_sequence
    predictor_logits_tensor[:, output_start:] = logits_sequence
    predictor_pre_sparsifier_tensor[:, output_start:] = activated_sequence
    predictor_codes_tensor[:, output_start:] = codes_sequence
    auxiliary = dict(auxiliary_extra)
    if transition_binder is not None:
        next_state = PredictorTransitionState(next_state, transition_state)
    for layer_index, layer_output in enumerate(layer_outputs):
        layer_tensor = torch.zeros(
            batch_size,
            time_steps,
            layer_output.shape[-1],
            device=device,
            dtype=dtype,
        )
        layer_tensor[:, output_start:] = layer_output
        auxiliary[f"hidden_state_layer_{layer_index}"] = layer_tensor
    return (
        ModuleOutputs(
            place_codes=predictor_codes_tensor,
            place_logits=predictor_logits_tensor,
            pre_sparsifier=predictor_pre_sparsifier_tensor,
            hidden_state=predictor_hidden_tensor,
            auxiliary=auxiliary,
        ),
        next_state,
    )


class PredictorFeedback(Protocol):
    """What the predictor eats at each step, and what the step's output does to that choice."""

    def seed_code(self) -> Tensor:
        """The code published at index 0, before the first transition."""

    def step_codes(self, timestep: int) -> tuple[Tensor, Tensor]:
        """(fed code, belief code) for the transition into timestep."""

    def observe(self, timestep: int, pre_sparsifier: Tensor, codes: Tensor) -> None:
        """Take the step's outputs, so the next step_codes can use them."""

    def target_code(self, timestep: int) -> Tensor | None:
        """The OBSERVED code at timestep when the policy has one (teacher forcing), else None."""


class TeacherForcedFeedback:
    """The predictor eats the OBSERVED code at t-1."""

    def __init__(
        self,
        encoder_codes: Tensor,
        *,
        belief_codes: Tensor | None = None,
        detach_intermediate_predictions: bool = False,
    ) -> None:
        self._encoder_codes = encoder_codes
        self._grid_belief_codes = belief_codes
        self._detach_intermediate_predictions = detach_intermediate_predictions
        self._running_belief = encoder_codes[:, 0]

    def seed_code(self) -> Tensor:
        return self._encoder_codes[:, 0]

    def step_codes(self, timestep: int) -> tuple[Tensor, Tensor]:
        fed_code = self._encoder_codes[:, timestep - 1]
        if self._grid_belief_codes is not None:
            return fed_code, self._grid_belief_codes[:, timestep - 1]
        return fed_code, self._running_belief

    def observe(self, timestep: int, pre_sparsifier: Tensor, codes: Tensor) -> None:
        if self._grid_belief_codes is None:
            self._running_belief = (
                codes.detach() if self._detach_intermediate_predictions else codes
            )

    def target_code(self, timestep: int) -> Tensor | None:
        return self._encoder_codes[:, timestep]


class OpenLoopFeedback:
    """Free-run between anchors: the predictor eats its own belief, re-anchored every k steps."""

    def __init__(
        self,
        encoder_codes: Tensor,
        *,
        reanchor_steps: int,
        detach_intermediate_predictions: bool = True,
    ) -> None:
        self._encoder_codes = encoder_codes
        self._reanchor_steps = max(1, int(reanchor_steps))
        self._detach_intermediate_predictions = detach_intermediate_predictions
        self._running_belief = encoder_codes[:, 0]

    def seed_code(self) -> Tensor:
        return self._encoder_codes[:, 0]

    def step_codes(self, timestep: int) -> tuple[Tensor, Tensor]:
        if (timestep - 1) % self._reanchor_steps == 0:
            self._running_belief = self._encoder_codes[:, timestep - 1]
        fed_code = self._running_belief
        return fed_code, fed_code

    def observe(self, timestep: int, pre_sparsifier: Tensor, codes: Tensor) -> None:
        self._running_belief = (
            codes.detach() if self._detach_intermediate_predictions else codes
        )


class ComparatorFeedback:
    """I1 closed loop: the comparator's posterior replaces the encoder code as the fed code."""

    def __init__(
        self,
        comparator: Any,
        encoder_pre: Tensor,
        *,
        corruption_regime: Tensor | None,
        attractor_binding: Any = None,
        encoder_hidden: Tensor | None = None,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self._comparator = comparator
        self._encoder_pre = encoder_pre
        self._corruption_regime = corruption_regime
        self._attractor_binding = attractor_binding
        self._encoder_hidden = encoder_hidden
        self._posterior_pre_steps: list[Tensor] = [encoder_pre[:, 0]]
        self._posterior_code_steps: list[Tensor] = [comparator.sparsifier(encoder_pre[:, 0])]
        self._innovation_steps: list[Tensor] = [torch.zeros_like(encoder_pre[:, 0])]
        self._gate_steps: list[Tensor] = [torch.ones_like(encoder_pre[:, 0])]
        self._binding_pool: Tensor | None = None
        self._binding_pool_steps: list[Tensor] = []
        if attractor_binding is not None:
            if encoder_hidden is None:
                raise RuntimeError("attractor_binding needs the encoder hidden-state sequence.")
            self._binding_pool = attractor_binding.initial_state(batch_size, device)
            self._binding_pool_steps.append(self._binding_pool)

    def seed_code(self) -> Tensor:
        return self._posterior_code_steps[0]

    def step_codes(self, timestep: int) -> tuple[Tensor, Tensor]:
        fed_code = self._posterior_code_steps[timestep - 1]
        return fed_code, fed_code

    def observe(self, timestep: int, pre_sparsifier: Tensor, codes: Tensor) -> None:
        expectation = pre_sparsifier.detach()
        evidence_t = self._encoder_pre[:, timestep]
        dark_t = (
            None
            if self._corruption_regime is None
            else (self._corruption_regime[:, timestep] == 2)
        )
        if self._attractor_binding is not None:
            binding_delta, self._binding_pool = self._attractor_binding.step(
                self._binding_pool,
                self._encoder_hidden[:, timestep],
                evidence_t,
                expectation,
                freeze=dark_t,
            )
            evidence_t = evidence_t + binding_delta
            self._binding_pool_steps.append(self._binding_pool)
        innovation_t = evidence_t - expectation
        gate_t = self._comparator.gate(innovation_t, dark_t)
        posterior_pre_t = expectation + gate_t * innovation_t
        self._posterior_pre_steps.append(posterior_pre_t)
        self._posterior_code_steps.append(self._comparator.sparsifier(posterior_pre_t))
        self._innovation_steps.append(innovation_t)
        self._gate_steps.append(gate_t.detach().expand_as(innovation_t))

    def comparator_outputs(self) -> ModuleOutputs:
        auxiliary = _sparsifier_auxiliary(self._comparator.sparsifier)
        auxiliary.update({
            "innovation": torch.stack(self._innovation_steps, dim=1),
            "gate": torch.stack(self._gate_steps, dim=1),
        })
        if self._binding_pool_steps:
            auxiliary["binding_pool"] = torch.stack(self._binding_pool_steps, dim=1)
        return ModuleOutputs(
            place_codes=torch.stack(self._posterior_code_steps, dim=1),
            pre_sparsifier=torch.stack(self._posterior_pre_steps, dim=1),
            auxiliary=auxiliary,
        )


def _run_stepwise_predictor_sequence(
    model: PredictorStack,
    feedback: PredictorFeedback,
    context: PredictorRuntimeContext,
    *,
    time_steps: int,
    regularizer: Any = None,
    capture_states: bool = False,
) -> tuple[ModuleOutputs, list[Any] | None, list[Any] | None]:
    """The one place recurrent state is threaded step by step."""
    seed_code = feedback.seed_code()
    batch_size, code_dim = seed_code.shape
    predictor_hidden_dim = model.components.config.predictor.output_size
    dtype = seed_code.dtype
    device = seed_code.device

    predictor_hidden_tensor = torch.zeros(
        batch_size, time_steps, predictor_hidden_dim, device=device, dtype=dtype
    )
    predictor_logits_tensor = torch.zeros(
        batch_size, time_steps, code_dim, device=device, dtype=dtype
    )
    predictor_pre_sparsifier_tensor = torch.zeros(
        batch_size, time_steps, code_dim, device=device, dtype=dtype
    )
    predictor_codes_tensor = torch.zeros(
        batch_size, time_steps, code_dim, device=device, dtype=dtype
    )
    predictor_codes_tensor[:, 0] = seed_code

    state_before = [None for _ in range(time_steps)] if capture_states else None
    state_after = [None for _ in range(time_steps)] if capture_states else None
    predictor_state = model.predictor_temporal.initial_state(batch_size, device)
    transition_binder = getattr(model, "predictor_transition_binder", None)
    if transition_binder is not None:
        predictor_state = PredictorTransitionState(predictor_state, None)
    predictor_hidden_layers = [
        torch.zeros(batch_size, time_steps, layer_state.shape[-1], device=device, dtype=dtype)
        for layer_state in _hidden_layers_from_predictor_state(predictor_state)
    ]
    residual = residual_dynamics_enabled(model)
    target_code_of = getattr(feedback, "target_code", None)
    for timestep in range(1, time_steps):
        fed_code, belief_code = feedback.step_codes(timestep)
        if capture_states and state_before is not None:
            state_before[timestep] = detach_state(predictor_state)
        predictor_input = assemble_predictor_input(
            model,
            context,
            encoder_code=fed_code,
            belief_code=belief_code,
            transition_index=timestep - 1,
        )
        hidden_t, logits_t, pre_sparsifier_t, codes_t, predictor_state = predict_code_from_input(
            model,
            predictor_input,
            predictor_state,
            regularizer=regularizer,
            timestep=timestep,
            residual_base=fed_code if residual else None,
        )
        if transition_binder is not None and target_code_of is not None:
            observed = target_code_of(timestep)
            if observed is not None:
                _read, table = transition_binder.forward_step(
                    predictor_input, observed, predictor_state.transition_state
                )
                predictor_state = PredictorTransitionState(predictor_state.temporal_state, table)
        if capture_states and state_after is not None:
            state_after[timestep] = detach_state(predictor_state)
        predictor_hidden_tensor[:, timestep] = hidden_t
        for layer_index, layer_hidden in enumerate(
            _hidden_layers_from_predictor_state(predictor_state)
        ):
            predictor_hidden_layers[layer_index][:, timestep] = layer_hidden
        predictor_logits_tensor[:, timestep] = logits_t
        predictor_pre_sparsifier_tensor[:, timestep] = pre_sparsifier_t
        predictor_codes_tensor[:, timestep] = codes_t
        feedback.observe(timestep, pre_sparsifier=pre_sparsifier_t, codes=codes_t)

    return (
        ModuleOutputs(
            place_codes=predictor_codes_tensor,
            place_logits=predictor_logits_tensor,
            pre_sparsifier=predictor_pre_sparsifier_tensor,
            hidden_state=predictor_hidden_tensor,
            auxiliary={
                f"hidden_state_layer_{layer_index}": layer_tensor
                for layer_index, layer_tensor in enumerate(predictor_hidden_layers)
            }
            | _sparsifier_auxiliary(model.predictor_sparsifier),
        ),
        state_before,
        state_after,
    )


def rollout_predictor_chunk(
    model: PredictorStack,
    encoder_codes: Tensor,
    context: PredictorRuntimeContext,
    *,
    initial_state: Any = None,
    previous_encoder_code: Tensor | None = None,
) -> tuple[ModuleOutputs, Any]:
    """Run one contiguous recurrent chunk and return its final predictor state."""
    return _run_fused_predictor_sequence(
        model,
        encoder_codes,
        context,
        initial_state=initial_state,
        previous_encoder_code=previous_encoder_code,
    )


def rollout_predictor_sequence(
    model: PredictorStack,
    encoder_codes: Tensor,
    context: PredictorRuntimeContext,
    *,
    regularizer: Any = None,
    detach_intermediate_predictions: bool = False,
    capture_states: bool = False,
    belief_codes: Tensor | None = None,
) -> tuple[ModuleOutputs, list[Any] | None, list[Any] | None]:
    family = model.components.config.predictor.family
    regularization_requires_stepwise = (
        regularizer is not None and regularizer.predictor_requires_stepwise()
    )
    can_fuse_recurrent = (
        temporal_backend_capabilities(
            family, model.components.config.predictor.fla_variant
        ).supports_fused
        and not predictor_assembler_uses_belief(model.predictor_input_assembler)
        and not regularization_requires_stepwise
        and hasattr(model.predictor_temporal, "forward_sequence")
    )
    if (
        not capture_states
        and not regularization_requires_stepwise
        and (family == "transformer" or can_fuse_recurrent)
    ):
        fused_outputs, _final_state = _run_fused_predictor_sequence(
            model,
            encoder_codes,
            context,
            regularizer=regularizer,
        )
        return (
            fused_outputs,
            None,
            None,
        )
    return _run_stepwise_predictor_sequence(
        model,
        TeacherForcedFeedback(
            encoder_codes,
            belief_codes=belief_codes,
            detach_intermediate_predictions=detach_intermediate_predictions,
        ),
        context,
        time_steps=encoder_codes.shape[1],
        regularizer=regularizer,
        capture_states=capture_states,
    )


def closed_loop_rollout_predictor_sequence(
    model: PredictorStack,
    encoder_codes: Tensor,
    encoder_pre: Tensor,
    comparator: Any,
    context: PredictorRuntimeContext,
    *,
    regularizer: Any = None,
    attractor_binding: Any = None,
    encoder_hidden: Tensor | None = None,
) -> tuple[ModuleOutputs, ModuleOutputs]:
    """I1 closed-loop fusion: the posterior replaces the encoder code as the predictor's input."""
    feedback = ComparatorFeedback(
        comparator,
        encoder_pre,
        corruption_regime=context.corruption_regime,
        attractor_binding=attractor_binding,
        encoder_hidden=encoder_hidden,
        batch_size=encoder_codes.shape[0],
        device=encoder_codes.device,
    )
    predictor_outputs, _before, _after = _run_stepwise_predictor_sequence(
        model,
        feedback,
        context,
        time_steps=encoder_codes.shape[1],
        regularizer=regularizer,
    )
    return predictor_outputs, feedback.comparator_outputs()


def open_loop_rollout_predictor_sequence(
    model: PredictorStack,
    encoder_codes: Tensor,
    context: PredictorRuntimeContext,
    *,
    reanchor_steps: int,
    detach_intermediate_predictions: bool = True,
) -> ModuleOutputs:
    """Roll the predictor open-loop, re-anchoring to the encoder every reanchor_steps."""
    outputs, _before, _after = _run_stepwise_predictor_sequence(
        model,
        OpenLoopFeedback(
            encoder_codes,
            reanchor_steps=reanchor_steps,
            detach_intermediate_predictions=detach_intermediate_predictions,
        ),
        context,
        time_steps=encoder_codes.shape[1],
    )
    return outputs


def future_prediction_from_state(
    model: PredictorStack,
    context: PredictorRuntimeContext,
    *,
    encoder_code: Tensor,
    belief_code: Tensor,
    transition_index: int,
    predictor_state: Any,
) -> Tensor:
    predictor_input = assemble_predictor_input(
        model,
        context,
        encoder_code=encoder_code,
        belief_code=belief_code,
        transition_index=transition_index,
    )
    _, _, _, future_prediction, _ = predict_code_from_input(
        model,
        predictor_input,
        detach_state(predictor_state),
        timestep=transition_index + 1,
        residual_base=encoder_code if residual_dynamics_enabled(model) else None,
    )
    return future_prediction.detach()
