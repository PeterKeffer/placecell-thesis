"""Clockwork RNN temporal module (Koutník et al. 2014)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .core import ProjectedTemporalBase


class ClockworkTemporal(ProjectedTemporalBase):
    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        clock_periods: list[int] | None = None,
    ) -> None:
        super().__init__(input_dim, layer_sizes, dropout)
        if len(self.layer_sizes) > 1:
            raise ValueError(
                "ClockworkTemporal is single-layer only; got "
                f"layer_sizes={self.layer_sizes} (len > 1)."
            )
        if not clock_periods:
            raise ValueError("ClockworkTemporal requires a non-empty clock_periods list.")

        hidden_size = int(self.layer_sizes[-1])
        group_count = len(clock_periods)
        if hidden_size % group_count != 0:
            raise ValueError(
                f"Hidden size {hidden_size} is not divisible by the number of clock modules "
                f"{group_count} (clock_periods={clock_periods})."
            )

        self.hidden_size = hidden_size
        self.clock_periods = [int(period) for period in clock_periods]
        self.group_count = group_count
        self.module_width = hidden_size // group_count

        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.input_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size)

        period_of_unit = torch.repeat_interleave(
            torch.arange(group_count), self.module_width
        )
        destination = period_of_unit.unsqueeze(1)
        source = period_of_unit.unsqueeze(0)
        recurrent_mask = source >= destination
        self.register_buffer("recurrent_mask", recurrent_mask)

    def initial_state(self, batch_size: int, device: torch.device) -> list[Tensor]:
        """[hidden, timestep]."""
        return [
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
        ]

    def _masked_recurrent_weight(self) -> Tensor:
        return self.recurrent.weight * self.recurrent_mask.to(self.recurrent.weight.dtype)

    def forward_step(
        self,
        x_t: Tensor,
        state: list[Tensor] | None = None,
        t: int | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """One Clockwork step."""
        if state is None:
            state = self.initial_state(x_t.shape[0], x_t.device)
        hidden_prev, previous_timestep = state[0], state[1]
        timestep = (
            previous_timestep + 1
            if t is None
            else torch.full_like(previous_timestep, int(t))
        )

        projected = self._project_inputs(x_t)
        preactivation = self.input_to_hidden(projected) + torch.nn.functional.linear(
            hidden_prev, self._masked_recurrent_weight(), self.recurrent.bias
        )
        candidate = torch.tanh(preactivation)

        tick = torch.zeros(
            timestep.shape[0], self.hidden_size, dtype=torch.bool, device=hidden_prev.device
        )
        for module_index, period in enumerate(self.clock_periods):
            start = module_index * self.module_width
            tick[:, start : start + self.module_width] = (
                (timestep % period == 0).unsqueeze(1)
            )
        hidden_new = torch.where(tick, candidate, hidden_prev)
        return hidden_new, [hidden_new, timestep]

    def forward_sequence_with_layer_outputs(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        state = initial_state
        outputs: list[Tensor] = []
        for offset in range(inputs.shape[1]):
            output_t, state = self.forward_step(inputs[:, offset], state)
            outputs.append(output_t.unsqueeze(1))
        sequence_output = torch.cat(outputs, dim=1)
        return sequence_output, state, [sequence_output]

    def forward_sequence(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        outputs, state, _ = self.forward_sequence_with_layer_outputs(inputs, initial_state)
        return outputs, state
