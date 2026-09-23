"""Predictor input routing."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn


class PredictorInputAssembler(Protocol):
    uses_belief: bool

    @property
    def output_dim(self) -> int: ...

    def assemble(
        self,
        encoder_code: Tensor,
        belief_code: Tensor | None,
        action_embedding: Tensor | None,
        kinematics: Tensor | None,
        temporal_offset: Tensor | None,
        corruption_info: dict[str, Tensor] | None,
    ) -> Tensor: ...


def _concat_channels(*channels: Tensor | None) -> Tensor:
    values = [channel for channel in channels if channel is not None]
    if not values:
        raise ValueError("At least one channel is required.")
    return torch.cat(values, dim=-1)


def predictor_assembler_uses_belief(assembler: object) -> bool:
    return bool(getattr(assembler, "uses_belief", True))


class EncoderInputAssembler(nn.Module):
    uses_belief: bool = False

    def __init__(
        self, code_dim: int, action_dim: int, kinematics_dim: int, temporal_offset_dim: int = 0
    ) -> None:
        super().__init__()
        self.output_dim = code_dim + action_dim + kinematics_dim + temporal_offset_dim

    def assemble(
        self,
        encoder_code: Tensor,
        belief_code: Tensor | None,
        action_embedding: Tensor | None,
        kinematics: Tensor | None,
        temporal_offset: Tensor | None,
        corruption_info: dict[str, Tensor] | None,
    ) -> Tensor:
        return _concat_channels(encoder_code, action_embedding, kinematics, temporal_offset)


class BeliefInputAssembler(EncoderInputAssembler):
    uses_belief: bool = True

    def assemble(
        self,
        encoder_code: Tensor,
        belief_code: Tensor | None,
        action_embedding: Tensor | None,
        kinematics: Tensor | None,
        temporal_offset: Tensor | None,
        corruption_info: dict[str, Tensor] | None,
    ) -> Tensor:
        if belief_code is None:
            raise ValueError("BeliefInputAssembler requires a belief_code.")
        return _concat_channels(belief_code, action_embedding, kinematics, temporal_offset)


class DualInputAssembler(nn.Module):
    uses_belief: bool = True

    def __init__(
        self, code_dim: int, action_dim: int, kinematics_dim: int, temporal_offset_dim: int = 0
    ) -> None:
        super().__init__()
        self.merge = nn.Linear(code_dim * 2 + 2, code_dim)
        self.output_dim = code_dim + action_dim + kinematics_dim + temporal_offset_dim

    def assemble(
        self,
        encoder_code: Tensor,
        belief_code: Tensor | None,
        action_embedding: Tensor | None,
        kinematics: Tensor | None,
        temporal_offset: Tensor | None,
        corruption_info: dict[str, Tensor] | None,
    ) -> Tensor:
        if belief_code is None:
            raise ValueError("DualInputAssembler requires a belief_code.")
        noise_level = (
            corruption_info["noise_level"]
            if corruption_info
            else torch.zeros_like(encoder_code[:, :1])
        )
        is_blackout = (
            corruption_info["is_blackout"]
            if corruption_info
            else torch.zeros_like(encoder_code[:, :1])
        )
        merged = self.merge(
            torch.cat([encoder_code, belief_code, noise_level, is_blackout], dim=-1)
        )
        return _concat_channels(merged, action_embedding, kinematics, temporal_offset)


class ConditionalInputAssembler(EncoderInputAssembler):
    uses_belief: bool = True

    def assemble(
        self,
        encoder_code: Tensor,
        belief_code: Tensor | None,
        action_embedding: Tensor | None,
        kinematics: Tensor | None,
        temporal_offset: Tensor | None,
        corruption_info: dict[str, Tensor] | None,
    ) -> Tensor:
        if belief_code is None:
            raise ValueError("ConditionalInputAssembler requires a belief_code.")
        is_blackout = (
            corruption_info["is_blackout"]
            if corruption_info
            else torch.zeros_like(encoder_code[:, :1])
        )
        source_code = torch.where(is_blackout > 0, belief_code, encoder_code)
        return _concat_channels(source_code, action_embedding, kinematics, temporal_offset)


class GatedInputAssembler(nn.Module):
    uses_belief: bool = True

    def __init__(
        self, code_dim: int, action_dim: int, kinematics_dim: int, temporal_offset_dim: int = 0
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(code_dim * 2 + 2, code_dim)
        self.output_dim = code_dim + action_dim + kinematics_dim + temporal_offset_dim

    def assemble(
        self,
        encoder_code: Tensor,
        belief_code: Tensor | None,
        action_embedding: Tensor | None,
        kinematics: Tensor | None,
        temporal_offset: Tensor | None,
        corruption_info: dict[str, Tensor] | None,
    ) -> Tensor:
        if belief_code is None:
            raise ValueError("GatedInputAssembler requires a belief_code.")
        noise_level = (
            corruption_info["noise_level"]
            if corruption_info
            else torch.zeros_like(encoder_code[:, :1])
        )
        is_blackout = (
            corruption_info["is_blackout"]
            if corruption_info
            else torch.zeros_like(encoder_code[:, :1])
        )
        gate_logits = self.gate(
            torch.cat([encoder_code, belief_code, noise_level, is_blackout], dim=-1)
        )
        gate = torch.sigmoid(gate_logits)
        mixed = gate * encoder_code + (1.0 - gate) * belief_code
        return _concat_channels(mixed, action_embedding, kinematics, temporal_offset)


PREDICTOR_INPUT_ASSEMBLER_BUILDERS: dict[str, type[nn.Module]] = {
    "encoder": EncoderInputAssembler,
    "belief": BeliefInputAssembler,
    "dual": DualInputAssembler,
    "conditional": ConditionalInputAssembler,
    "gated": GatedInputAssembler,
}
