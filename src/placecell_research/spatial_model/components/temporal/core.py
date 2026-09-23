"""Shared machinery every projected temporal adapter builds on."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _apply_temporal_dropout(
    layer_output: Tensor,
    *,
    layer_index: int,
    num_layers: int,
    dropout: float,
    training: bool,
) -> Tensor:
    if layer_index < num_layers - 1 and dropout > 0.0:
        return torch.nn.functional.dropout(layer_output, p=dropout, training=training)
    return layer_output


class AdaptiveStateGate(nn.Module):
    """Learn one event-sensitive recurrent-state update rate per sample and timestep."""

    def __init__(
        self,
        hidden_size: int,
        minimum_update_rate: float,
        initial_open_probability: float,
    ) -> None:
        super().__init__()
        self.minimum_update_rate = float(minimum_update_rate)
        self.logit = nn.Linear(3 * int(hidden_size), 1)
        initial_logit = math.log(
            float(initial_open_probability) / (1.0 - float(initial_open_probability))
        )
        nn.init.zeros_(self.logit.weight)
        nn.init.constant_(self.logit.bias, initial_logit)

    def forward(
        self,
        previous_state: Tensor,
        candidate_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        gate_features = torch.cat(
            (
                previous_state,
                candidate_state,
                torch.abs(candidate_state - previous_state),
            ),
            dim=-1,
        )
        open_probability = torch.sigmoid(self.logit(gate_features))
        update_rate = self.minimum_update_rate + (
            1.0 - self.minimum_update_rate
        ) * open_probability
        updated_state = (
            (1.0 - update_rate) * previous_state + update_rate * candidate_state
        )
        return updated_state, update_rate.squeeze(-1), open_probability.squeeze(-1)


class ProjectedTemporalBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        recurrent_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.num_layers = len(self.layer_sizes)
        self.dropout = float(dropout)
        self.recurrent_activation: nn.Module = (
            recurrent_activation if recurrent_activation is not None else nn.Identity()
        )
        self._state_leak: float = 1.0
        self.adaptive_state_gates: nn.ModuleList | None = None
        self.last_auxiliary_outputs: dict[str, Tensor] = {}
        self.last_diagnostics: dict[str, Tensor] = {}
        self.register_buffer("encoder_input_mask", None)
        self._free_partition_recurrent_input_mask_names: list[str] = []

    def _configure_adaptive_state_gates(
        self,
        minimum_update_rate: float | None,
        initial_open_probability: float,
    ) -> None:
        if minimum_update_rate is None:
            return
        if self._state_leak < 1.0:
            raise ValueError(
                "Adaptive state gating and state_tau>1 are mutually exclusive recurrent-state "
                "update rules."
            )
        self.adaptive_state_gates = nn.ModuleList(
            AdaptiveStateGate(
                hidden_size,
                minimum_update_rate,
                initial_open_probability,
            )
            for hidden_size in self.layer_sizes
        )

    @property
    def uses_adaptive_state_gate(self) -> bool:
        return self.adaptive_state_gates is not None

    def _adaptive_state_update(
        self,
        layer_index: int,
        previous_state: Tensor,
        candidate_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.adaptive_state_gates is None:
            raise RuntimeError("Adaptive state update requested while the gate is disabled.")
        return self.adaptive_state_gates[layer_index](previous_state, candidate_state)

    def _record_adaptive_state_diagnostics(
        self,
        update_rates: list[Tensor],
        open_probabilities: list[Tensor],
    ) -> None:
        if not update_rates:
            self.last_auxiliary_outputs = {}
            self.last_diagnostics = {}
            return
        self.last_auxiliary_outputs = {
            "adaptive_state_update_rate": torch.stack(update_rates, dim=-1)
            .unsqueeze(1),
            "adaptive_state_gate_open_probability": torch.stack(
                open_probabilities,
                dim=-1,
            )
            .unsqueeze(1),
        }
        self.last_diagnostics = {
            name: value.detach() for name, value in self.last_auxiliary_outputs.items()
        }

    def _project_inputs(self, inputs: Tensor) -> Tensor:
        self._enforce_free_partition_recurrent_mask()
        if (
            self.encoder_input_mask is not None
        ):
            projected = torch.nn.functional.linear(
                inputs,
                self.input_projection.weight * self.encoder_input_mask,
                self.input_projection.bias,
            )
        else:
            projected = self.input_projection(inputs)
        return torch.tanh(projected)

    def _recurrent_modules(self) -> list[nn.Module]:
        recurrent_types = (
            nn.RNN,
            nn.GRU,
            nn.LSTM,
            nn.RNNCell,
            nn.GRUCell,
            nn.LSTMCell,
        )
        recurrent_modules = [
            module for module in self.modules() if isinstance(module, recurrent_types)
        ]
        if not recurrent_modules:
            raise TypeError("Free partition requires an RNN, GRU, or LSTM recurrent layer.")
        return recurrent_modules

    def _enforce_free_partition_recurrent_mask(self) -> None:
        if not self._free_partition_recurrent_input_mask_names:
            return
        with torch.no_grad():
            for recurrent, mask_name in zip(
                self._recurrent_modules(),
                self._free_partition_recurrent_input_mask_names, strict=False,
            ):
                weight_ih = getattr(
                    recurrent,
                    "weight_ih_l0",
                    getattr(recurrent, "weight_ih", None),
                )
                if weight_ih is None:
                    raise TypeError(
                        f"{type(recurrent).__name__} has no input-weight matrix to partition."
                    )
                weight_ih.mul_(getattr(self, mask_name))

    def set_free_partition(self, free_partition_fraction: float, code_dim: int) -> None:
        """Zero the encoder-code input columns for the last free_partition_fraction of units."""
        fraction = float(free_partition_fraction)
        if fraction <= 0.0 or int(code_dim) <= 0:
            return
        out_features, in_features = self.input_projection.weight.shape
        num_free = int(round(fraction * out_features))
        if num_free <= 0:
            return
        mask = torch.ones(out_features, in_features, device=self.input_projection.weight.device)
        mask[out_features - num_free :, : int(code_dim)] = (
            0.0
        )
        self.encoder_input_mask = mask
        recurrent_modules = self._recurrent_modules()
        if int(recurrent_modules[0].input_size) != out_features:  # type: ignore[attr-defined]
            raise ValueError(
                "Free partition requires the first recurrent layer's input width to match the "
                "input-projection width."
            )
        if self._free_partition_recurrent_input_mask_names:
            raise RuntimeError("Free partition may be configured only once per temporal module.")
        for layer_index, recurrent in enumerate(recurrent_modules):
            hidden_size = int(recurrent.hidden_size)  # type: ignore[attr-defined]
            input_size = int(recurrent.input_size)  # type: ignore[attr-defined]
            output_free_units = int(round(fraction * hidden_size))
            input_free_units = int(round(fraction * input_size))
            weight_ih = getattr(
                recurrent,
                "weight_ih_l0",
                getattr(recurrent, "weight_ih", None),
            )
            if weight_ih is None:
                raise TypeError(
                    f"{type(recurrent).__name__} has no input-weight matrix to partition."
                )
            recurrent_mask = torch.ones_like(weight_ih)
            gate_count = recurrent_mask.shape[0] // hidden_size
            first_free_output = hidden_size - output_free_units
            first_free_input = input_size - input_free_units
            for gate_index in range(gate_count):
                gate_start = gate_index * hidden_size
                recurrent_mask[
                    gate_start + first_free_output : gate_start + hidden_size,
                    :first_free_input,
                ] = 0.0
            mask_name = f"_free_partition_recurrent_input_mask_{layer_index}"
            self.register_buffer(mask_name, recurrent_mask, persistent=False)
            self._free_partition_recurrent_input_mask_names.append(mask_name)
            weight_ih.register_hook(
                lambda gradient, name=mask_name: gradient * getattr(self, name)
            )
        self._enforce_free_partition_recurrent_mask()

    def _apply_dropout(self, layer_output: Tensor, layer_index: int) -> Tensor:
        return _apply_temporal_dropout(
            layer_output,
            layer_index=layer_index,
            num_layers=self.num_layers,
            dropout=self.dropout,
            training=self.training,
        )

    def apply_forget_gate_bias(self, forget_bias: float) -> None:
        if forget_bias == 0.0:
            return
        with torch.no_grad():
            for module in self.modules():
                if not isinstance(module, (nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell)):
                    continue
                gate = slice(module.hidden_size, 2 * module.hidden_size)
                for suffix in ("_l0", ""):
                    bias_ih = getattr(module, f"bias_ih{suffix}", None)
                    bias_hh = getattr(module, f"bias_hh{suffix}", None)
                    if bias_ih is None or bias_hh is None:
                        continue
                    bias_ih[gate] = forget_bias
                    bias_hh[gate] = 0.0

    def apply_chrono_init(self, t_max: int) -> None:
        """Chrono initialization (Tallec & Ollivier 2018, arXiv:1804.11188) for LSTM gates."""
        with torch.no_grad():
            for module in self.modules():
                if not isinstance(module, (nn.LSTM, nn.LSTMCell)):
                    continue
                hidden_size = module.hidden_size
                input_gate = slice(0, hidden_size)
                forget_gate = slice(hidden_size, 2 * hidden_size)
                for suffix in ("_l0", ""):
                    bias_ih = getattr(module, f"bias_ih{suffix}", None)
                    bias_hh = getattr(module, f"bias_hh{suffix}", None)
                    if bias_ih is None or bias_hh is None:
                        continue
                    timescales = torch.empty(hidden_size, device=bias_ih.device).uniform_(
                        1.0, float(t_max - 1)
                    )
                    log_timescales = torch.log(timescales)
                    bias_ih[forget_gate] = log_timescales
                    bias_ih[input_gate] = -log_timescales
                    bias_hh[input_gate] = 0.0
                    bias_hh[forget_gate] = 0.0
