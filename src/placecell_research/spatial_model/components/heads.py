"""Projection heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported head activation: {name}")


class SparseLinear(nn.Linear):
    """Linear with fixed random sparse input connectivity (Numenta-style sparse weights)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_sparsity: float,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        k = max(1, round(in_features * weight_sparsity))
        kept = torch.rand(out_features, in_features).topk(k, dim=1).indices
        mask = torch.zeros(out_features, in_features)
        mask.scatter_(1, kept, 1.0)
        self.register_buffer("weight_mask", mask)
        with torch.no_grad():
            self.weight.mul_(mask)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight * self.weight_mask, self.bias)


class CodeHead(nn.Module):
    """Project hidden states into code space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str,
        normalize: bool,
        weight_sparsity: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear: nn.Linear = (
            SparseLinear(input_dim, output_dim, weight_sparsity)
            if weight_sparsity < 1.0
            else nn.Linear(input_dim, output_dim)
        )
        self.activation = build_activation(activation)
        self.normalize = normalize
        self.output_dim = output_dim

    def project(self, hidden_states: Tensor) -> Tensor:
        return self.linear(hidden_states)

    def activate(self, logits: Tensor) -> Tensor:
        activated = self.activation(logits)
        if self.normalize:
            activated = F.normalize(activated, dim=-1)
        return activated

    def forward_from_logits(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        return self.activate(logits), logits

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.project(hidden_states)
        return self.forward_from_logits(logits)
