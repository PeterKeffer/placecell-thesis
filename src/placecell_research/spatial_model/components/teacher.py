"""Teacher-student controller."""

from __future__ import annotations

import math
import weakref
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from placecell_research.config.schema import TeacherStudentConfig

from ..predictor_context import select_kinematics, uses_action_context
from ..predictor_stack import PredictorModules
from ..types import ModuleOutputs

if TYPE_CHECKING:
    from ..regularization import SequenceRegularizer


_SLOW_LEAK_SCAN_CHUNK_SIZE = 64


def _ema_decay(config: TeacherStudentConfig, current_step: int | None, total_steps: int) -> float:
    if config.ema_schedule == "constant" or current_step is None:
        return config.ema_decay
    progress = min(1.0, max(0.0, current_step / max(total_steps, 1)))
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return config.ema_decay + (config.ema_decay_end - config.ema_decay) * cosine


def _uniform_alpha(alpha_value: Tensor) -> Tensor | None:
    flat_alpha = alpha_value.reshape(-1)
    if flat_alpha.numel() == 0:
        return None
    first_alpha = flat_alpha[0]
    if bool(torch.all(flat_alpha == first_alpha)):
        return first_alpha
    return None


def _apply_slow_leak_scan_scalar(
    activations: Tensor,
    alpha_value: Tensor,
    first_state: Tensor,
) -> tuple[Tensor, Tensor]:
    outputs = [first_state.unsqueeze(1)]
    previous_state = first_state
    if activations.shape[1] == 1:
        return outputs[0], previous_state

    chunk_size = min(_SLOW_LEAK_SCAN_CHUNK_SIZE, activations.shape[1] - 1)
    positions = torch.arange(chunk_size, device=activations.device)
    exponents = positions[:, None] - positions[None, :]
    lower_triangle = exponents >= 0
    weights = torch.where(
        lower_triangle,
        alpha_value.pow(exponents.clamp_min(0)),
        torch.zeros((), device=activations.device, dtype=activations.dtype),
    )
    carry_powers = alpha_value.pow(
        torch.arange(1, chunk_size + 1, device=activations.device, dtype=activations.dtype)
    )

    for start in range(1, activations.shape[1], chunk_size):
        block = activations[:, start : start + chunk_size]
        length = block.shape[1]
        block_weights = weights[:length, :length]
        block_carry_powers = carry_powers[:length].view(1, length, 1)
        block_contribution = torch.einsum("bsc,ts->btc", block, block_weights) * (1.0 - alpha_value)
        states = block_carry_powers * previous_state.unsqueeze(1) + block_contribution
        outputs.append(states)
        previous_state = states[:, -1]
    return torch.cat(outputs, dim=1), previous_state


def _apply_slow_leak_stepwise(
    activations: Tensor,
    alpha_value: Tensor,
    first_state: Tensor,
) -> tuple[Tensor, Tensor]:
    outputs = [first_state.unsqueeze(1)]
    previous_state = first_state
    for time_index in range(1, activations.shape[1]):
        activation = activations[:, time_index]
        previous_state = alpha_value * previous_state + (1.0 - alpha_value) * activation
        outputs.append(previous_state.unsqueeze(1))
    return torch.cat(outputs, dim=1), previous_state


def _apply_union_leak_stepwise(
    activations: Tensor,
    alpha_value: Tensor,
    first_state: Tensor,
) -> tuple[Tensor, Tensor]:
    """Decaying-union (leaky-max) temporal pooling: state_t = max(alpha * state_{t-1}, x_t)."""
    outputs = [first_state.unsqueeze(1)]
    previous_state = first_state
    for time_index in range(1, activations.shape[1]):
        previous_state = torch.maximum(alpha_value * previous_state, activations[:, time_index])
        outputs.append(previous_state.unsqueeze(1))
    return torch.cat(outputs, dim=1), previous_state


def _apply_slow_leak(
    activations: Tensor,
    alpha: float | Tensor,
    initial_state: Tensor | None = None,
    initial_state_mask: Tensor | None = None,
    mode: str = "ema",
) -> tuple[Tensor, Tensor | None]:
    if not isinstance(alpha, Tensor):
        if float(alpha) <= 0.0:
            return activations, None
        alpha_value = torch.as_tensor(
            float(alpha), device=activations.device, dtype=activations.dtype
        )
    else:
        alpha_value = alpha.to(device=activations.device, dtype=activations.dtype)
        if bool(torch.all(alpha_value <= 0.0)):
            return activations, None
    if activations.ndim != 3:
        raise ValueError(
            "Slow leak expects activations shaped [batch, time, code_dim], "
            f"got {tuple(activations.shape)}."
        )
    if activations.shape[1] == 0:
        return activations, initial_state
    first_activation = activations[:, 0]
    if mode == "union":
        if initial_state is None:
            union_first = first_activation
        else:
            candidate = torch.maximum(alpha_value * initial_state, first_activation)
            if initial_state_mask is None:
                union_first = candidate
            else:
                union_mask = initial_state_mask.to(
                    device=first_activation.device, dtype=torch.bool
                ).view(-1, 1)
                union_first = torch.where(union_mask, candidate, first_activation)
        return _apply_union_leak_stepwise(activations, alpha_value, union_first)
    if initial_state is None:
        first_state = first_activation
    else:
        candidate = alpha_value * initial_state + (1.0 - alpha_value) * first_activation
        if initial_state_mask is None:
            first_state = candidate
        else:
            mask = initial_state_mask.to(
                device=first_activation.device,
                dtype=torch.bool,
            ).view(-1, 1)
            first_state = torch.where(mask, candidate, first_activation)
    if alpha_value.numel() == 1:
        return _apply_slow_leak_scan_scalar(activations, alpha_value.reshape(()), first_state)

    active_alpha = alpha_value > 0.0
    uniform_active_alpha = _uniform_alpha(alpha_value[active_alpha])
    if uniform_active_alpha is not None:
        if bool(torch.all(active_alpha)):
            return _apply_slow_leak_scan_scalar(activations, uniform_active_alpha, first_state)
        leaked = activations.clone()
        active_outputs, active_state = _apply_slow_leak_scan_scalar(
            activations[:, :, active_alpha],
            uniform_active_alpha,
            first_state[:, active_alpha],
        )
        leaked[:, :, active_alpha] = active_outputs
        final_state = activations[:, -1].clone()
        final_state[:, active_alpha] = active_state
        return leaked, final_state

    return _apply_slow_leak_stepwise(activations, alpha_value, first_state)


class EncoderStackState(NamedTuple):
    """Carry for an encoder stack that owns state OUTSIDE its temporal core."""

    temporal_state: Any
    chart_code_state: Tensor | None
    sensory_state: Tensor | None = None
    slow_leak_state: Tensor | None = None
    slow_leak_initialized: Tensor | None = None


class PostHeadRMSNorm(nn.RMSNorm):
    """RMSNorm on the sparsifier's input, distinguishable from every other RMSNorm in the model."""


class EncoderStack(nn.Module):
    """Full encoder stack copied into the EMA teacher."""

    def __init__(
        self,
        observation_encoder: nn.Module,
        encoder_temporal: nn.Module,
        encoder_head: nn.Module,
        sparsifier: nn.Module,
        gradient_checkpointing: bool = False,
        slow_leak_alpha: float | Tensor = 0.0,
        slow_leak_mode: str = "ema",
        context_channels: list[str] | None = None,
        action_embedding: nn.Embedding | None = None,
        encoder_code_norm: nn.Module | None = None,
        encoder_post_head_norm: nn.Module | None = None,
        expert_heads: nn.ModuleList | None = None,
        gated_mixture: nn.Module | None = None,
        precision_heads: nn.ModuleList | None = None,
        stream_temporals: nn.ModuleList | None = None,
        stream_heads: nn.ModuleList | None = None,
        stream_names: list[str] | None = None,
        environment_conditioner: nn.Module | None = None,
        readout_mode: str = "mixed",
        export_state_readout: bool = False,
        require_explicit_state_readout: bool = False,
        chart_code_binder: nn.Module | None = None,
        sensory_binder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.observation_encoder = observation_encoder
        self.sensory_binder = sensory_binder
        self.encoder_temporal = encoder_temporal
        self.encoder_code_norm = encoder_code_norm or nn.Identity()
        self.encoder_post_head_norm = encoder_post_head_norm or nn.Identity()
        self.encoder_head = encoder_head
        self.expert_heads = expert_heads
        self.gated_mixture = gated_mixture
        self.precision_heads = precision_heads
        self.stream_temporals = stream_temporals
        self.stream_heads = stream_heads
        self.stream_names = list(stream_names or [])
        self.environment_conditioner = environment_conditioner
        self.readout_mode = str(readout_mode)
        self.export_state_readout = bool(export_state_readout)
        self.require_explicit_state_readout = bool(require_explicit_state_readout)
        self.chart_code_binder = chart_code_binder
        self.sparsifier = sparsifier
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if isinstance(slow_leak_alpha, Tensor):
            self.slow_leak_alpha = 0.0
            self.register_buffer("_slow_leak_alpha_tensor", slow_leak_alpha.float())
            self.slow_leak_active = bool(torch.any(slow_leak_alpha > 0.0))
        else:
            self.slow_leak_alpha = float(slow_leak_alpha)
            self.register_buffer("_slow_leak_alpha_tensor", None)
            self.slow_leak_active = self.slow_leak_alpha > 0.0
        self.slow_leak_mode = str(slow_leak_mode)
        self.context_channels = list(context_channels or [])
        self.action_embedding = action_embedding

    def append_context(
        self,
        features: Tensor,
        actions: Tensor | None,
        kinematics: Tensor | None,
        external_context: Tensor | None = None,
    ) -> Tensor:
        if not self.context_channels and external_context is None:
            return features
        context_tensors: list[Tensor] = [features]
        if external_context is not None:
            context_tensors.append(
                external_context.to(device=features.device, dtype=features.dtype)
            )
        if uses_action_context(self.context_channels):
            if self.action_embedding is None:
                raise ValueError(
                    "Encoder action context is enabled, but the model was built without "
                    "an encoder action embedding."
                )
            if actions is None:
                raise ValueError(
                    "Encoder action context requires actions, but the batch did not provide them."
                )
            context_tensors.append(self.action_embedding(actions.long()).to(dtype=features.dtype))
        selected_kinematics = select_kinematics(
            kinematics,
            self.context_channels,
            context_label="Encoder context channels",
        )
        if selected_kinematics is not None:
            context_tensors.append(
                selected_kinematics.to(device=features.device, dtype=features.dtype)
            )
        return torch.cat(context_tensors, dim=-1)

    @property
    def owns_stack_state(self) -> bool:
        """True when this stack carries state the temporal core does not own."""
        return (
            self.chart_code_binder is not None
            or self.sensory_binder is not None
            or self.slow_leak_active
        )

    def initial_state(self) -> EncoderStackState | None:
        """The cold carry to hand forward_stateful at the start of an episode."""
        return EncoderStackState(None, None) if self.owns_stack_state else None

    def apply_slow_leak(
        self,
        activations: Tensor,
        initial_state: Tensor | None = None,
        initial_state_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        return _apply_slow_leak(
            activations,
            self._slow_leak_alpha_tensor
            if self._slow_leak_alpha_tensor is not None
            else self.slow_leak_alpha,
            initial_state=initial_state,
            initial_state_mask=initial_state_mask,
            mode=self.slow_leak_mode,
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        kinematics: Tensor | None = None,
        valid_steps: Tensor | None = None,
        training_regularizer: SequenceRegularizer | None = None,
    ) -> ModuleOutputs:
        outputs, _state = self.forward_stateful(
            observations,
            actions=actions,
            kinematics=kinematics,
            valid_steps=valid_steps,
            training_regularizer=training_regularizer,
        )
        return outputs

    def _sensory_chart_block(
        self,
        features: Tensor,
        scan: Any,
        start: int,
        stop: int,
        temporal_state: Any,
        memory: Tensor | None,
        valid_steps: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Any, list[dict[str, Tensor]], Tensor | None, Tensor]:
        """stop - start steps of the canonical scan: read, trunk step, write."""
        hidden_pieces: list[Tensor] = []
        readout_pieces: list[Tensor] = []
        layer_pieces: list[dict[str, Tensor]] = []
        read_pieces: list[Tensor] = []
        for step in range(start, stop):
            if step == scan.reset_before_step:
                memory = None
            recall, read = self.sensory_binder.read(scan, step, memory)
            step_hidden, step_readout, temporal_state, step_layers = (
                _forward_temporal_with_hidden_layers(
                    self.encoder_temporal,
                    features[:, step : step + 1] + recall.unsqueeze(1),
                    initial_state=temporal_state,
                    valid_steps=None if valid_steps is None else valid_steps[:, step : step + 1],
                    export_state_readout=self.export_state_readout,
                    require_explicit_state_readout=self.require_explicit_state_readout,
                )
            )
            memory = self.sensory_binder.write(scan, step, step_hidden[:, -1], memory)
            hidden_pieces.append(step_hidden)
            if step_readout is not None:
                readout_pieces.append(step_readout)
            layer_pieces.append(step_layers)
            read_pieces.append(read.detach())
        return (
            torch.cat(hidden_pieces, dim=1),
            torch.cat(readout_pieces, dim=1) if readout_pieces else None,
            temporal_state,
            layer_pieces,
            memory,
            torch.stack(read_pieces, dim=1),
        )

    def _forward_temporal_with_sensory_chart(
        self,
        observation_features: Tensor,
        features: Tensor,
        temporal_initial_state: Any,
        memory: Tensor | None,
        valid_steps: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Any, dict[str, Tensor], Tensor]:
        scan = self.sensory_binder.prepare(observation_features, valid_steps)
        block_size = self.sensory_binder.checkpoint_chunk or features.shape[1]
        hidden_blocks: list[Tensor] = []
        readout_blocks: list[Tensor] = []
        layer_pieces: list[dict[str, Tensor]] = []
        read_blocks: list[Tensor] = []
        for start in range(0, features.shape[1], block_size):
            stop = min(start + block_size, features.shape[1])
            arguments = (features, scan, start, stop, temporal_initial_state, memory, valid_steps)
            checkpointed = (
                self.sensory_binder.checkpoint_chunk > 0
                and torch.is_grad_enabled()
                and stop - start > 1
            )
            (
                hidden,
                readout,
                temporal_initial_state,
                block_layers,
                memory,
                reads,
            ) = (
                checkpoint(self._sensory_chart_block, *arguments, use_reentrant=False)
                if checkpointed
                else self._sensory_chart_block(*arguments)
            )
            hidden_blocks.append(hidden)
            if readout is not None:
                readout_blocks.append(readout)
            layer_pieces.extend(block_layers)
            read_blocks.append(reads)
        hidden_states = torch.cat(hidden_blocks, dim=1)
        reads = torch.cat(read_blocks, dim=1)
        self.sensory_binder.publish(scan, reads, memory)
        return (
            hidden_states,
            torch.cat(readout_blocks, dim=1) if readout_blocks else None,
            temporal_initial_state,
            _merge_stepped_layer_outputs(layer_pieces),
            memory,
        )

    def forward_stateful(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        kinematics: Tensor | None = None,
        valid_steps: Tensor | None = None,
        training_regularizer: SequenceRegularizer | None = None,
        initial_state: Any = None,
        external_context: Tensor | None = None,
    ) -> tuple[ModuleOutputs, Any]:
        """Run the encoder over a chunk of steps and return its outputs and its next carry."""
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            observation_features = checkpoint(
                self.observation_encoder, observations, use_reentrant=False
            )
        else:
            observation_features = self.observation_encoder(observations)
        if training_regularizer is not None:
            observation_features = training_regularizer.apply("vision.output", observation_features)
        features = self.append_context(
            observation_features, actions, kinematics, external_context=external_context
        )
        temporal_initial_state = initial_state
        chart_code_state: Tensor | None = None
        sensory_state: Tensor | None = None
        slow_leak_state: Tensor | None = None
        slow_leak_initialized: Tensor | None = None
        owns_outer_state = self.owns_stack_state
        if owns_outer_state:
            if initial_state is None:
                temporal_initial_state = None
            else:
                temporal_initial_state = initial_state.temporal_state
                chart_code_state = initial_state.chart_code_state
                sensory_state = initial_state.sensory_state
                slow_leak_state = initial_state.slow_leak_state
                slow_leak_initialized = initial_state.slow_leak_initialized
        if self.sensory_binder is not None:
            (
                backbone_output,
                state_readout,
                next_state,
                hidden_state_layers,
                sensory_state,
            ) = self._forward_temporal_with_sensory_chart(
                observation_features,
                features,
                temporal_initial_state,
                sensory_state,
                valid_steps,
            )
        else:
            (
                backbone_output,
                state_readout,
                next_state,
                hidden_state_layers,
            ) = _forward_temporal_with_hidden_layers(
                self.encoder_temporal,
                features,
                initial_state=temporal_initial_state,
                valid_steps=valid_steps,
                export_state_readout=self.export_state_readout,
                require_explicit_state_readout=self.require_explicit_state_readout,
            )
        hidden_states = state_readout if self.readout_mode == "state" else backbone_output
        if hidden_states is None:
            raise RuntimeError("encoder_readout='state' requires an exported state readout.")
        if training_regularizer is not None:
            hidden_states = training_regularizer.apply("encoder.hidden", hidden_states)
        environment_code: Tensor | None = None
        if self.environment_conditioner is not None:
            hidden_states, environment_code = self.environment_conditioner(
                hidden_states,
                valid_steps=valid_steps,
            )
        base_logits = self.encoder_code_norm(self.encoder_head.project(hidden_states))
        if self.chart_code_binder is not None:
            read_contribution, chart_code_state = self.chart_code_binder.forward_sequence(
                hidden_states,
                base_logits,
                chart_code_state,
            )
            base_logits = base_logits + read_contribution
        if self.stream_temporals is not None:
            per_expert_logits = [base_logits]
            for stream_name, stream_temporal, stream_head in zip(
                self.stream_names, self.stream_temporals, self.stream_heads, strict=False
            ):
                if stream_name == "kinematics":
                    if kinematics is None:
                        raise ValueError("a 'kinematics' stream expert requires batch kinematics.")
                    stream_input = kinematics
                else:
                    stream_input = observation_features
                (
                    stream_backbone_output,
                    stream_state_readout,
                    _stream_state,
                    _stream_layers,
                ) = _forward_temporal_with_hidden_layers(
                    stream_temporal,
                    stream_input,
                    valid_steps=valid_steps,
                    export_state_readout=self.export_state_readout,
                    require_explicit_state_readout=self.require_explicit_state_readout,
                )
                stream_hidden = (
                    stream_state_readout if self.readout_mode == "state" else stream_backbone_output
                )
                if stream_hidden is None:
                    raise RuntimeError(
                        "A stream temporal did not export its required state readout."
                    )
                per_expert_logits.append(self.encoder_code_norm(stream_head.project(stream_hidden)))
        elif self.expert_heads:
            per_expert_logits = [base_logits] + [
                self.encoder_code_norm(head.project(hidden_states)) for head in self.expert_heads
            ]
        else:
            per_expert_logits = None
        gate_responsibilities: Tensor | None = None
        if per_expert_logits is None:
            logits = base_logits
        elif self.precision_heads is not None:
            precisions = [
                torch.nn.functional.softplus(head(expert_logits))
                for head, expert_logits in zip(
                    self.precision_heads, per_expert_logits, strict=False
                )
            ]
            logits = sum(
                precision * expert_logits
                for precision, expert_logits in zip(precisions, per_expert_logits, strict=False)
            )
        else:
            logits = torch.stack(per_expert_logits, dim=0).sum(dim=0)
        if self.gated_mixture is not None:
            expert_codes = []
            for expert_logits in per_expert_logits:
                activated_expert = self.encoder_head.activate(expert_logits)
                activated_expert, _ = self.apply_slow_leak(activated_expert)
                activated_expert = self.encoder_post_head_norm(activated_expert)
                expert_codes.append(self.sparsifier(activated_expert))
            place_codes, gate_responsibilities = self.gated_mixture(hidden_states, expert_codes)
            pre_sparsifier = None
            if training_regularizer is not None:
                place_codes = training_regularizer.apply("encoder.output", place_codes)
        else:
            if training_regularizer is not None:
                logits = training_regularizer.apply("encoder.logits", logits)
            pre_sparsifier = self.encoder_head.activate(logits)
            pre_sparsifier, slow_leak_state = self.apply_slow_leak(
                pre_sparsifier,
                initial_state=slow_leak_state,
                initial_state_mask=slow_leak_initialized,
            )
            if slow_leak_state is not None:
                slow_leak_initialized = torch.ones(
                    slow_leak_state.shape[0], dtype=torch.bool, device=slow_leak_state.device
                )
            if training_regularizer is not None:
                pre_sparsifier = training_regularizer.apply(
                    "encoder.pre_sparsifier", pre_sparsifier
                )
            pre_sparsifier = self.encoder_post_head_norm(pre_sparsifier)
            place_codes = self.sparsifier(pre_sparsifier)
            if training_regularizer is not None:
                place_codes = training_regularizer.apply("encoder.output", place_codes)
        auxiliary = dict(hidden_state_layers)
        if per_expert_logits is not None:
            for expert_index, expert_logits in enumerate(per_expert_logits):
                auxiliary[f"expert_logits_{expert_index}"] = expert_logits
        if gate_responsibilities is not None:
            auxiliary["gate_responsibilities"] = gate_responsibilities
        if environment_code is not None:
            auxiliary["environment_code"] = environment_code
        auxiliary.update(getattr(self.sparsifier, "last_auxiliary_outputs", {}))
        if self.chart_code_binder is not None:
            auxiliary.update(self.chart_code_binder.last_auxiliary_outputs)
        if self.sensory_binder is not None:
            auxiliary.update(self.sensory_binder.last_auxiliary_outputs)
        if owns_outer_state:
            next_state = EncoderStackState(
                next_state,
                chart_code_state,
                sensory_state,
                slow_leak_state,
                slow_leak_initialized,
            )
        return (
            ModuleOutputs(
                place_codes=place_codes,
                place_logits=logits,
                pre_sparsifier=pre_sparsifier,
                hidden_state=hidden_states,
                backbone_output=backbone_output,
                state_readout=state_readout,
                auxiliary=auxiliary,
            ),
            next_state,
        )


def _merge_stepped_layer_outputs(pieces: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Rejoin what a stepped scan returned one step at a time."""
    merged = dict(pieces[-1])
    for key in merged:
        if key.startswith("hidden_state_layer_"):
            merged[key] = torch.cat([piece[key] for piece in pieces], dim=1)
    return merged


def _forward_temporal_with_hidden_layers(
    temporal_module: nn.Module,
    features: Tensor,
    *,
    initial_state: Any = None,
    valid_steps: Tensor | None = None,
    export_state_readout: bool = False,
    require_explicit_state_readout: bool = False,
) -> tuple[Tensor, Tensor | None, Any, dict[str, Tensor]]:
    state_readout: Tensor | None = None
    if export_state_readout and hasattr(
        temporal_module,
        "forward_sequence_with_state_readout_and_layer_outputs",
    ):
        (
            outputs,
            state_readout,
            next_state,
            layer_outputs,
        ) = temporal_module.forward_sequence_with_state_readout_and_layer_outputs(
            features,
            initial_state,
        )
    elif export_state_readout and hasattr(temporal_module, "forward_sequence_with_state_readout"):
        outputs, state_readout, next_state = temporal_module.forward_sequence_with_state_readout(
            features,
            initial_state,
        )
        layer_outputs = []
    elif export_state_readout and require_explicit_state_readout:
        raise TypeError(
            f"{type(temporal_module).__name__} must implement "
            "forward_sequence_with_state_readout() for encoder_readout='state'."
        )
    elif hasattr(temporal_module, "forward_sequence_with_layer_outputs"):
        outputs, next_state, layer_outputs = temporal_module.forward_sequence_with_layer_outputs(
            features,
            initial_state,
        )
        if export_state_readout:
            state_readout = outputs
    else:
        outputs, next_state = temporal_module.forward_sequence(
            features,
            initial_state,
        )
        layer_outputs = []
        if export_state_readout:
            state_readout = outputs
    hidden_state_layers = {
        f"hidden_state_layer_{layer_index}": layer_output
        for layer_index, layer_output in enumerate(layer_outputs)
    }
    hidden_state_layers.update(getattr(temporal_module, "last_auxiliary_outputs", {}))
    return outputs, state_readout, next_state, hidden_state_layers


class TeacherStudentController(nn.Module):
    """Keeps a full-stack EMA teacher in eval mode."""

    def __init__(
        self,
        encoder_stack: EncoderStack,
        config: TeacherStudentConfig,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.total_steps = total_steps
        self._online_encoder_stack_ref = weakref.ref(encoder_stack)
        self.teacher_encoder_stack = deepcopy(encoder_stack) if config.mode != "none" else None
        self.teacher_predictor_modules: nn.ModuleDict | None = None
        self._online_predictor_refs: dict[str, weakref.ReferenceType[nn.Module]] = {}
        self._shared_frozen_parameter_ids: set[int] = set()
        if self.teacher_encoder_stack is not None:
            if getattr(encoder_stack.observation_encoder, "_placecell_frozen", False):
                self.teacher_encoder_stack.observation_encoder = encoder_stack.observation_encoder
                self._shared_frozen_parameter_ids = {
                    id(parameter) for parameter in encoder_stack.observation_encoder.parameters()
                }
            self.teacher_encoder_stack.eval()
            for parameter in self.teacher_encoder_stack.parameters():
                parameter.requires_grad_(False)

    def attach_predictor(self, online_modules: PredictorModules) -> None:
        """Deep-copy the predictor stack into a second EMA teacher: the SLOW OPERATOR."""
        if not self.config.predictor_ema:
            return
        if self.teacher_encoder_stack is None:
            raise RuntimeError(
                "predictor_ema requires an encoder teacher; the slow operator reads "
                "teacher.place_codes."
            )
        present = online_modules.present()
        self.teacher_predictor_modules = nn.ModuleDict(
            {name: deepcopy(module) for name, module in present.items()}
        )
        self._online_predictor_refs = {
            name: weakref.ref(module) for name, module in present.items()
        }
        self.teacher_predictor_modules.eval()
        for parameter in self.teacher_predictor_modules.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "_predictor_teacher_synced", torch.zeros((), dtype=torch.bool), persistent=True
        )

    @property
    def enabled(self) -> bool:
        return self.teacher_encoder_stack is not None

    @property
    def predictor_enabled(self) -> bool:
        return self.teacher_predictor_modules is not None

    def teacher_predictor_stack(self) -> PredictorModules:
        """The slow operator's modules, typed."""
        modules = self.teacher_predictor_modules
        if modules is None:
            raise RuntimeError(
                "No slow operator is attached; teacher_student.predictor_ema is off."
            )
        missing = [name for name in self._online_predictor_refs if name not in modules]
        if missing:
            raise KeyError(
                f"Slow-operator module(s) {sorted(missing)} are missing from the EMA teacher, "
                f"which holds {sorted(modules)}. Rebuild the model rather than running the online "
                "predictor under the slow operator's name."
            )
        return PredictorModules(
            predictor_input_assembler=modules["predictor_input_assembler"],
            predictor_temporal=modules["predictor_temporal"],
            predictor_head=modules["predictor_head"],
            predictor_sparsifier=modules["predictor_sparsifier"],
            predictor_transition_binder=(
                modules["predictor_transition_binder"]
                if "predictor_transition_binder" in modules
                else None
            ),
            action_embedding=(
                modules["action_embedding"] if "action_embedding" in modules else None
            ),
        )

    @property
    def online_encoder_stack(self) -> EncoderStack:
        encoder_stack = self._online_encoder_stack_ref()
        if encoder_stack is None:
            raise RuntimeError("Online encoder stack was released before the teacher controller.")
        return encoder_stack

    def update(self, current_step: int | None = None) -> None:
        if self.teacher_encoder_stack is None:
            return
        decay = _ema_decay(self.config, current_step, self.total_steps)
        with torch.no_grad():
            teacher_parameters = []
            online_parameters = []
            for teacher_parameter, online_parameter in zip(
                self.teacher_encoder_stack.parameters(),
                self.online_encoder_stack.parameters(),
                strict=False,
            ):
                if (
                    id(teacher_parameter) in self._shared_frozen_parameter_ids
                    or id(online_parameter) in self._shared_frozen_parameter_ids
                ):
                    continue
                teacher_parameters.append(teacher_parameter.detach())
                online_parameters.append(online_parameter.detach())
            if teacher_parameters:
                torch._foreach_mul_(teacher_parameters, decay)
                torch._foreach_add_(
                    teacher_parameters,
                    online_parameters,
                    alpha=1.0 - decay,
                )
            self._copy_encoder_routing_state()
        self.teacher_encoder_stack.eval()
        self._update_predictor_teacher()

    @torch.no_grad()
    def _copy_encoder_routing_state(self) -> None:
        """Keep explicitly supported routing state aligned between online and EMA encoders."""
        if self.teacher_encoder_stack is None:
            return
        online_buffers = dict(self.online_encoder_stack.named_buffers())
        teacher_buffers = dict(self.teacher_encoder_stack.named_buffers())
        synchronized_suffixes = ("balance_bias", "_k_anneal_step")
        for name, online_buffer in online_buffers.items():
            if not name.endswith(synchronized_suffixes):
                continue
            teacher_buffer = teacher_buffers.get(name)
            if teacher_buffer is None or teacher_buffer.shape != online_buffer.shape:
                raise RuntimeError(
                    f"EMA teacher routing buffer does not match online buffer {name!r}."
                )
            teacher_buffer.copy_(online_buffer)

    def set_predictor_teacher_backward_mode(self, needs_backward: bool) -> None:
        modules = self.teacher_predictor_modules
        if modules is None:
            return
        if not needs_backward:
            modules.eval()
            return
        modules.train()
        for module in modules.modules():
            rate = getattr(module, "p", None) if isinstance(module, nn.Dropout) else None
            if rate:
                raise RuntimeError(
                    "slow_operator_side='online' needs the slow operator in train mode for cuDNN "
                    f"RNN backward, but it contains dropout p={rate}, which would resample every "
                    "call and make the 'immobile' operator stochastic. Set predictor.dropout=0."
                )
            if isinstance(module, nn.Dropout):
                module.eval()
            recurrent_dropout = getattr(module, "dropout", None)
            if isinstance(recurrent_dropout, float) and recurrent_dropout > 0.0:
                raise RuntimeError(
                    "slow_operator_side='online' needs the slow operator in train mode for cuDNN "
                    f"RNN backward, but {type(module).__name__} has dropout={recurrent_dropout}. "
                    "Set predictor.dropout=0 for this arm."
                )

    def ensure_predictor_teacher_synced(self) -> None:
        """Adopt the online operator BEFORE the slow operator is first used."""
        if self.teacher_predictor_modules is None or bool(self._predictor_teacher_synced):
            return
        with torch.no_grad():
            for name, teacher_module in self.teacher_predictor_modules.items():
                online_module = self._online_predictor_refs[name]()
                if online_module is None:
                    raise RuntimeError(
                        f"Online predictor module '{name}' was released before its EMA teacher."
                    )
                teacher_module.load_state_dict(online_module.state_dict())
            self._predictor_teacher_synced.fill_(True)
        self.teacher_predictor_modules.eval()

    def _update_predictor_teacher(self) -> None:
        """EMA the slow operator on its OWN decay, independent of the encoder teacher's."""
        teacher_modules = self.teacher_predictor_modules
        if teacher_modules is None:
            return
        self.ensure_predictor_teacher_synced()
        decay = self.config.resolved_predictor_ema_decay
        with torch.no_grad():
            for name, teacher_module in teacher_modules.items():
                online_module = self._online_predictor_refs[name]()
                if online_module is None:
                    raise RuntimeError(
                        f"Online predictor module '{name}' was released before its EMA teacher."
                    )
                teacher_parameters = [
                    parameter.detach() for parameter in teacher_module.parameters()
                ]
                online_parameters = [parameter.detach() for parameter in online_module.parameters()]
                if len(teacher_parameters) != len(online_parameters):
                    raise RuntimeError(
                        f"Slow-operator module '{name}' has {len(teacher_parameters)} parameters "
                        f"but its online counterpart has {len(online_parameters)}."
                    )
                if teacher_parameters:
                    torch._foreach_mul_(teacher_parameters, decay)
                    torch._foreach_add_(
                        teacher_parameters,
                        online_parameters,
                        alpha=1.0 - decay,
                    )
                pullback = self.config.predictor_ema_pullback
                if pullback > 0.0 and teacher_parameters:
                    torch._foreach_mul_(online_parameters, 1.0 - pullback)
                    torch._foreach_add_(
                        online_parameters,
                        teacher_parameters,
                        alpha=pullback,
                    )
                self._lag_buffers(name, teacher_module, online_module, decay)
        teacher_modules.eval()

    def _lag_buffers(
        self,
        name: str,
        teacher_module: nn.Module,
        online_module: nn.Module,
        decay: float,
    ) -> None:
        """Lag the operator's BUFFERS too, not just its parameters."""
        teacher_buffers = dict(teacher_module.named_buffers())
        online_buffers = dict(online_module.named_buffers())
        if teacher_buffers.keys() != online_buffers.keys():
            raise RuntimeError(
                f"Slow-operator module '{name}' has buffers {sorted(teacher_buffers)} but its "
                f"online counterpart has {sorted(online_buffers)}."
            )
        for buffer_name, teacher_buffer in teacher_buffers.items():
            online_buffer = online_buffers[buffer_name]
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(decay).add_(online_buffer, alpha=1.0 - decay)
            else:
                teacher_buffer.copy_(online_buffer)

    def encode(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        kinematics: Tensor | None = None,
        valid_steps: Tensor | None = None,
    ) -> ModuleOutputs | None:
        outputs, _state = self.encode_stateful(
            observations,
            actions=actions,
            kinematics=kinematics,
            valid_steps=valid_steps,
        )
        return outputs

    def encode_stateful(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        kinematics: Tensor | None = None,
        valid_steps: Tensor | None = None,
        initial_state: Any = None,
        external_context: Tensor | None = None,
    ) -> tuple[ModuleOutputs | None, Any]:
        if self.teacher_encoder_stack is None:
            return None, None
        with torch.no_grad():
            self.teacher_encoder_stack.eval()
            return self.teacher_encoder_stack.forward_stateful(
                observations,
                actions=actions,
                kinematics=kinematics,
                valid_steps=valid_steps,
                initial_state=initial_state,
                external_context=external_context,
            )
