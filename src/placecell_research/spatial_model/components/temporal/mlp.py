"""Feedforward temporal family: no recurrent state at all."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLPTemporal(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_size in layer_sizes:
            layers.extend([nn.Linear(current_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_size
        self.network = nn.Sequential(*layers)
        self.hidden_size = int(layer_sizes[-1])

    def initial_state(self, batch_size: int, device: torch.device) -> None:
        return None

    def forward_step(self, x_t: Tensor, state: None = None) -> tuple[Tensor, None]:
        return self.network(x_t), None

    def forward_sequence(self, inputs: Tensor, initial_state: None = None) -> tuple[Tensor, None]:
        outputs = self.network(inputs)
        return outputs, None
