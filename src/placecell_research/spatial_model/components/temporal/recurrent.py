"""RNN / GRU / LSTM adapters, including the stepwise GRU variants."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .core import ProjectedTemporalBase


class GRUSequenceTemporal(ProjectedTemporalBase):
    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        recurrent_activation: nn.Module | None = None,
        state_tau: float = 1.0,
        adaptive_state_alpha: float | None = None,
        adaptive_state_gate_initial_open_probability: float = 0.01,
    ) -> None:
        super().__init__(input_dim, layer_sizes, dropout, recurrent_activation)
        self.input_projection = nn.Linear(input_dim, self.layer_sizes[0])
        self.gru_layers = nn.ModuleList()
        previous_dim = self.layer_sizes[0]
        for hidden_size in self.layer_sizes:
            self.gru_layers.append(
                nn.GRU(previous_dim, hidden_size, num_layers=1, batch_first=True)
            )
            previous_dim = hidden_size
        self.hidden_size = int(self.layer_sizes[-1])
        self._state_leak = 1.0 / float(state_tau)
        self._configure_adaptive_state_gates(
            adaptive_state_alpha,
            adaptive_state_gate_initial_open_probability,
        )

    def initial_state(self, batch_size: int, device: torch.device) -> list[Tensor]:
        return [
            torch.zeros(batch_size, hidden_size, device=device) for hidden_size in self.layer_sizes
        ]

    def _forward_step_with_layer_outputs(
        self,
        x_t: Tensor,
        state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        layer_output = self._project_inputs(x_t).unsqueeze(1)
        next_states: list[Tensor] = []
        layer_outputs: list[Tensor] = []
        update_rates: list[Tensor] = []
        open_probabilities: list[Tensor] = []
        for layer_index, gru_layer in enumerate(self.gru_layers):
            prev_state = None if state is None else state[layer_index]
            layer_initial_state = None if prev_state is None else prev_state.unsqueeze(0)
            layer_output, hidden = gru_layer(layer_output, layer_initial_state)
            layer_output = self.recurrent_activation(layer_output)
            hidden = self.recurrent_activation(hidden)
            new_state = hidden.squeeze(0)
            if self._state_leak < 1.0:
                if prev_state is None:
                    prev_state = torch.zeros_like(new_state)
                new_state = (1.0 - self._state_leak) * prev_state + self._state_leak * new_state
                layer_output = new_state.unsqueeze(1)
            elif self.uses_adaptive_state_gate:
                if prev_state is None:
                    prev_state = torch.zeros_like(new_state)
                new_state, update_rate, open_probability = self._adaptive_state_update(
                    layer_index,
                    prev_state,
                    new_state,
                )
                layer_output = new_state.unsqueeze(1)
                update_rates.append(update_rate)
                open_probabilities.append(open_probability)
            next_states.append(new_state)
            layer_outputs.append(layer_output.squeeze(1))
            layer_output = self._apply_dropout(layer_output, layer_index)
        self._record_adaptive_state_diagnostics(update_rates, open_probabilities)
        return layer_output.squeeze(1), next_states, layer_outputs

    def forward_sequence_with_layer_outputs(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        if (
            not isinstance(self.recurrent_activation, nn.Identity)
            or self._state_leak < 1.0
            or self.uses_adaptive_state_gate
        ):
            state = initial_state
            outputs: list[Tensor] = []
            layer_outputs_by_layer: list[list[Tensor]] = [[] for _ in range(self.num_layers)]
            update_rates: list[Tensor] = []
            open_probabilities: list[Tensor] = []
            for timestep in range(inputs.shape[1]):
                output_t, state, step_layer_outputs = self._forward_step_with_layer_outputs(
                    inputs[:, timestep],
                    state,
                )
                outputs.append(output_t.unsqueeze(1))
                for layer_index, layer_output_t in enumerate(step_layer_outputs):
                    layer_outputs_by_layer[layer_index].append(layer_output_t.unsqueeze(1))
                if self.uses_adaptive_state_gate:
                    update_rates.append(self.last_auxiliary_outputs["adaptive_state_update_rate"])
                    open_probabilities.append(
                        self.last_auxiliary_outputs["adaptive_state_gate_open_probability"]
                    )
            if self.uses_adaptive_state_gate:
                self.last_auxiliary_outputs = {
                    "adaptive_state_update_rate": torch.cat(update_rates, dim=1),
                    "adaptive_state_gate_open_probability": torch.cat(
                        open_probabilities,
                        dim=1,
                    ),
                }
                self.last_diagnostics = {
                    name: value.detach() for name, value in self.last_auxiliary_outputs.items()
                }
            return (
                torch.cat(outputs, dim=1),
                state,
                [torch.cat(layer_outputs, dim=1) for layer_outputs in layer_outputs_by_layer],
            )

        self.last_auxiliary_outputs = {}
        self.last_diagnostics = {}
        layer_output = self._project_inputs(inputs)
        next_states: list[Tensor] = []
        layer_outputs: list[Tensor] = []
        for layer_index, gru_layer in enumerate(self.gru_layers):
            layer_initial_state = None
            if initial_state is not None:
                layer_initial_state = initial_state[layer_index].unsqueeze(0)
            layer_output, hidden = gru_layer(layer_output, layer_initial_state)
            next_states.append(hidden.squeeze(0))
            layer_outputs.append(layer_output)
            layer_output = self._apply_dropout(layer_output, layer_index)
        return layer_output, next_states, layer_outputs

    def forward_sequence(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        outputs, hidden, _ = self.forward_sequence_with_layer_outputs(inputs, initial_state)
        return outputs, hidden

    def forward_step(
        self,
        x_t: Tensor,
        state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        outputs, next_state, _ = self._forward_step_with_layer_outputs(
            x_t,
            state,
        )
        return outputs, next_state


class RNNSequenceTemporal(ProjectedTemporalBase):
    """Vanilla (Elman) tanh RNN backbone: the ungated control for the gated GRU/LSTM cores."""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim, layer_sizes, dropout)
        self.input_projection = nn.Linear(input_dim, self.layer_sizes[0])
        self.rnn_layers = nn.ModuleList()
        previous_dim = self.layer_sizes[0]
        for hidden_size in self.layer_sizes:
            self.rnn_layers.append(
                nn.RNN(
                    previous_dim,
                    hidden_size,
                    num_layers=1,
                    batch_first=True,
                    nonlinearity="tanh",
                )
            )
            previous_dim = hidden_size
        self.hidden_size = int(self.layer_sizes[-1])

    def initial_state(self, batch_size: int, device: torch.device) -> list[Tensor]:
        return [
            torch.zeros(batch_size, hidden_size, device=device) for hidden_size in self.layer_sizes
        ]

    def forward_sequence_with_layer_outputs(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        layer_output = self._project_inputs(inputs)
        next_states: list[Tensor] = []
        layer_outputs: list[Tensor] = []
        for layer_index, rnn_layer in enumerate(self.rnn_layers):
            layer_initial_state = None
            if initial_state is not None:
                layer_initial_state = initial_state[layer_index].unsqueeze(0)
            layer_output, hidden = rnn_layer(layer_output, layer_initial_state)
            next_states.append(hidden.squeeze(0))
            layer_outputs.append(layer_output)
            layer_output = self._apply_dropout(layer_output, layer_index)
        return layer_output, next_states, layer_outputs

    def forward_sequence(
        self,
        inputs: Tensor,
        initial_state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        outputs, hidden, _ = self.forward_sequence_with_layer_outputs(inputs, initial_state)
        return outputs, hidden

    def forward_step(
        self,
        x_t: Tensor,
        state: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        layer_output = self._project_inputs(x_t).unsqueeze(1)
        next_states: list[Tensor] = []
        for layer_index, rnn_layer in enumerate(self.rnn_layers):
            prev_state = None if state is None else state[layer_index]
            layer_initial_state = None if prev_state is None else prev_state.unsqueeze(0)
            layer_output, hidden = rnn_layer(layer_output, layer_initial_state)
            next_states.append(hidden.squeeze(0))
            layer_output = self._apply_dropout(layer_output, layer_index)
        return layer_output.squeeze(1), next_states


class LSTMSequenceTemporal(ProjectedTemporalBase):
    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.0,
        recurrent_activation: nn.Module | None = None,
        adaptive_state_alpha: float | None = None,
        adaptive_state_gate_initial_open_probability: float = 0.01,
    ) -> None:
        super().__init__(input_dim, layer_sizes, dropout, recurrent_activation)
        self.input_projection = nn.Linear(input_dim, self.layer_sizes[0])
        self.lstm_layers = nn.ModuleList()
        previous_dim = self.layer_sizes[0]
        for hidden_size in self.layer_sizes:
            self.lstm_layers.append(
                nn.LSTM(previous_dim, hidden_size, num_layers=1, batch_first=True)
            )
            previous_dim = hidden_size
        self.hidden_size = int(self.layer_sizes[-1])
        self._configure_adaptive_state_gates(
            adaptive_state_alpha,
            adaptive_state_gate_initial_open_probability,
        )

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> tuple[list[Tensor], list[Tensor]]:
        hidden = [
            torch.zeros(batch_size, hidden_size, device=device) for hidden_size in self.layer_sizes
        ]
        cell = [
            torch.zeros(batch_size, hidden_size, device=device) for hidden_size in self.layer_sizes
        ]
        return hidden, cell

    def forward_sequence_with_layer_outputs(
        self,
        inputs: Tensor,
        initial_state: tuple[list[Tensor], list[Tensor]] | None = None,
    ) -> tuple[Tensor, tuple[list[Tensor], list[Tensor]], list[Tensor]]:
        if (
            not isinstance(self.recurrent_activation, nn.Identity)
            or self._state_leak < 1.0
            or self.uses_adaptive_state_gate
        ):
            state = initial_state
            outputs: list[Tensor] = []
            layer_outputs_by_layer: list[list[Tensor]] = [[] for _ in range(self.num_layers)]
            update_rates: list[Tensor] = []
            open_probabilities: list[Tensor] = []
            for timestep in range(inputs.shape[1]):
                output_t, state, step_layer_outputs = self._forward_step_with_layer_outputs(
                    inputs[:, timestep],
                    state,
                )
                outputs.append(output_t.unsqueeze(1))
                for layer_index, layer_output_t in enumerate(step_layer_outputs):
                    layer_outputs_by_layer[layer_index].append(layer_output_t.unsqueeze(1))
                if self.uses_adaptive_state_gate:
                    update_rates.append(self.last_auxiliary_outputs["adaptive_state_update_rate"])
                    open_probabilities.append(
                        self.last_auxiliary_outputs["adaptive_state_gate_open_probability"]
                    )
            if self.uses_adaptive_state_gate:
                self.last_auxiliary_outputs = {
                    "adaptive_state_update_rate": torch.cat(update_rates, dim=1),
                    "adaptive_state_gate_open_probability": torch.cat(
                        open_probabilities,
                        dim=1,
                    ),
                }
                self.last_diagnostics = {
                    name: value.detach() for name, value in self.last_auxiliary_outputs.items()
                }
            return (
                torch.cat(outputs, dim=1),
                state,
                [torch.cat(layer_outputs, dim=1) for layer_outputs in layer_outputs_by_layer],
            )

        self.last_auxiliary_outputs = {}
        self.last_diagnostics = {}
        layer_output = self._project_inputs(inputs)
        next_hidden: list[Tensor] = []
        next_cell: list[Tensor] = []
        layer_outputs: list[Tensor] = []
        for layer_index, lstm_layer in enumerate(self.lstm_layers):
            layer_initial_state = None
            if initial_state is not None:
                hidden_0, cell_0 = initial_state
                layer_initial_state = (
                    hidden_0[layer_index].unsqueeze(0),
                    cell_0[layer_index].unsqueeze(0),
                )
            layer_output, (hidden, cell) = lstm_layer(layer_output, layer_initial_state)
            next_hidden.append(hidden.squeeze(0))
            next_cell.append(cell.squeeze(0))
            layer_outputs.append(layer_output)
            layer_output = self._apply_dropout(layer_output, layer_index)
        return layer_output, (next_hidden, next_cell), layer_outputs

    def forward_sequence(
        self,
        inputs: Tensor,
        initial_state: tuple[list[Tensor], list[Tensor]] | None = None,
    ) -> tuple[Tensor, tuple[list[Tensor], list[Tensor]]]:
        outputs, state, _ = self.forward_sequence_with_layer_outputs(inputs, initial_state)
        return outputs, state

    def _forward_step_with_layer_outputs(
        self,
        x_t: Tensor,
        state: tuple[list[Tensor], list[Tensor]] | None = None,
    ) -> tuple[Tensor, tuple[list[Tensor], list[Tensor]], list[Tensor]]:
        layer_output = self._project_inputs(x_t).unsqueeze(1)
        next_hidden: list[Tensor] = []
        next_cell: list[Tensor] = []
        layer_outputs: list[Tensor] = []
        update_rates: list[Tensor] = []
        open_probabilities: list[Tensor] = []
        for layer_index, lstm_layer in enumerate(self.lstm_layers):
            layer_initial_state = None
            previous_hidden = None
            previous_cell = None
            if state is not None:
                hidden_0, cell_0 = state
                previous_hidden = hidden_0[layer_index]
                previous_cell = cell_0[layer_index]
                layer_initial_state = (
                    previous_hidden.unsqueeze(0),
                    previous_cell.unsqueeze(0),
                )
            layer_output, (hidden, cell) = lstm_layer(layer_output, layer_initial_state)
            layer_output = self.recurrent_activation(layer_output)
            hidden = self.recurrent_activation(hidden)
            new_hidden = hidden.squeeze(0)
            new_cell = cell.squeeze(0)
            if self.uses_adaptive_state_gate:
                if previous_hidden is None:
                    previous_hidden = torch.zeros_like(new_hidden)
                    previous_cell = torch.zeros_like(new_cell)
                if previous_cell is None:
                    raise RuntimeError("LSTM hidden state was provided without its cell state.")
                new_hidden, update_rate, open_probability = self._adaptive_state_update(
                    layer_index,
                    previous_hidden,
                    new_hidden,
                )
                expanded_update_rate = update_rate.unsqueeze(-1)
                new_cell = (
                    1.0 - expanded_update_rate
                ) * previous_cell + expanded_update_rate * new_cell
                layer_output = new_hidden.unsqueeze(1)
                update_rates.append(update_rate)
                open_probabilities.append(open_probability)
            next_hidden.append(new_hidden)
            next_cell.append(new_cell)
            layer_outputs.append(layer_output.squeeze(1))
            layer_output = self._apply_dropout(layer_output, layer_index)
        self._record_adaptive_state_diagnostics(update_rates, open_probabilities)
        return layer_output.squeeze(1), (next_hidden, next_cell), layer_outputs

    def forward_step(
        self,
        x_t: Tensor,
        state: tuple[list[Tensor], list[Tensor]] | None = None,
    ) -> tuple[Tensor, tuple[list[Tensor], list[Tensor]]]:
        outputs, next_state, _ = self._forward_step_with_layer_outputs(
            x_t,
            state,
        )
        return outputs, next_state
