"""MTRNN temporal module (Multiple Timescale Recurrent Neural Network)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .core import ProjectedTemporalBase


class MTRNNTemporal(ProjectedTemporalBase):
    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        time_constants: list[float] | None = None,
    ) -> None:
        super().__init__(input_dim, layer_sizes, dropout)
        if len(self.layer_sizes) != 1:
            raise ValueError(
                "MTRNNTemporal is single-layer only; got "
                f"layer_sizes={layer_sizes} ({len(self.layer_sizes)} layers)."
            )
        hidden_size = int(self.layer_sizes[-1])

        if not time_constants:
            raise ValueError("MTRNNTemporal requires a non-empty list of time constants.")
        num_groups = len(time_constants)
        if hidden_size % num_groups != 0:
            raise ValueError(
                f"MTRNNTemporal hidden size {hidden_size} is not divisible by the number of "
                f"time-constant groups {num_groups}."
            )
        time_constant_values = [float(tau) for tau in time_constants]
        if any(tau < 1.0 for tau in time_constant_values):
            raise ValueError(
                "MTRNNTemporal time constants must all be >= 1.0 (tau < 1 gives a negative "
                f"retention weight and diverges); got {time_constant_values}."
            )

        group_width = hidden_size // num_groups
        per_unit_tau = torch.tensor(time_constant_values).repeat_interleave(group_width)
        self.register_buffer("time_constant", per_unit_tau)

        self.input_to_membrane = nn.Linear(input_dim, hidden_size)
        self.recurrent_to_membrane = nn.Linear(hidden_size, hidden_size, bias=False)

        self.hidden_size = hidden_size

    def initial_state(self, batch_size: int, device: torch.device) -> list[Tensor]:
        """Zero MEMBRANE potential u of shape (batch, H) (length-1 list)."""
        return [torch.zeros(batch_size, self.hidden_size, device=device)]

    def forward_step(
        self, x_t: Tensor, state: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        if state is None:
            state = self.initial_state(x_t.shape[0], x_t.device)
        membrane = state[0]
        previous_rate = torch.tanh(membrane)
        drive = self.input_to_membrane(x_t) + self.recurrent_to_membrane(previous_rate)
        leak = 1.0 / self.time_constant
        next_membrane = (1.0 - leak) * membrane + leak * drive
        next_rate = torch.tanh(next_membrane)
        return next_rate, [next_membrane]

    def forward_sequence_with_layer_outputs(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        state = initial_state
        outputs: list[Tensor] = []
        for timestep in range(inputs.shape[1]):
            rate, state = self.forward_step(inputs[:, timestep], state)
            outputs.append(rate.unsqueeze(1))
        sequence_output = torch.cat(outputs, dim=1)
        return sequence_output, state, [sequence_output]

    def forward_sequence(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        outputs, state, _ = self.forward_sequence_with_layer_outputs(inputs, initial_state)
        return outputs, state
