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
        readout_mode: str = "mixed",
        export_state_readout: bool = False,
    ) -> None:
        super().__init__()
        self.observation_encoder = observation_encoder
        self.encoder_temporal = encoder_temporal
        self.encoder_code_norm = encoder_code_norm or nn.Identity()
        self.encoder_post_head_norm = encoder_post_head_norm or nn.Identity()
        self.encoder_head = encoder_head
        self.readout_mode = str(readout_mode)
        self.export_state_readout = bool(export_state_readout)
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
    ) -> Tensor:
        if not self.context_channels:
            return features
        context_tensors: list[Tensor] = [features]
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
        return self.slow_leak_active

    def initial_state(self) -> EncoderStackState | None:
        """The cold carry to hand forward_stateful at the start of an episode."""
        return EncoderStackState(None) if self.owns_stack_state else None

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

    def forward_stateful(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        kinematics: Tensor | None = None,
        valid_steps: Tensor | None = None,
        training_regularizer: SequenceRegularizer | None = None,
        initial_state: Any = None,
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
        features = self.append_context(observation_features, actions, kinematics)
        temporal_initial_state = initial_state
        slow_leak_state: Tensor | None = None
        slow_leak_initialized: Tensor | None = None
        owns_outer_state = self.owns_stack_state
        if owns_outer_state:
            if initial_state is None:
                temporal_initial_state = None
            else:
                temporal_initial_state = initial_state.temporal_state
                slow_leak_state = initial_state.slow_leak_state
                slow_leak_initialized = initial_state.slow_leak_initialized
        (
            backbone_output,
            state_readout,
            next_state,
            hidden_state_layers,
        ) = _forward_temporal_with_hidden_layers(
            self.encoder_temporal,
            features,
            initial_state=temporal_initial_state,
            export_state_readout=self.export_state_readout,
        )
        hidden_states = state_readout if self.readout_mode == "state" else backbone_output
        if hidden_states is None:
            raise RuntimeError("encoder_readout='state' requires an exported state readout.")
        if training_regularizer is not None:
            hidden_states = training_regularizer.apply("encoder.hidden", hidden_states)
        logits = self.encoder_code_norm(self.encoder_head.project(hidden_states))
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
            pre_sparsifier = training_regularizer.apply("encoder.pre_sparsifier", pre_sparsifier)
        pre_sparsifier = self.encoder_post_head_norm(pre_sparsifier)
        place_codes = self.sparsifier(pre_sparsifier)
        if training_regularizer is not None:
            place_codes = training_regularizer.apply("encoder.output", place_codes)
        auxiliary = dict(hidden_state_layers)
        auxiliary.update(getattr(self.sparsifier, "last_auxiliary_outputs", {}))
        if owns_outer_state:
            next_state = EncoderStackState(next_state, slow_leak_state, slow_leak_initialized)
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


def _forward_temporal_with_hidden_layers(
    temporal_module: nn.Module,
    features: Tensor,
    *,
    initial_state: Any = None,
    export_state_readout: bool = False,
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
            )
