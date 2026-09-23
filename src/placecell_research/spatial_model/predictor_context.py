"""Canonical predictor context channel contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from placecell_research.config.context_channels import (
    ACTION_CONTEXT_CHANNEL,
    DELTA_KINEMATICS_CHANNELS,
    KINEMATICS_CHANNEL_TO_INDEX,
    TRANSITION_KINEMATICS_CHANNELS,
    kinematics_channels,
    parse_kinematics_channel,
    required_kinematics_width,
)


def uses_action_context(channels: Sequence[str]) -> bool:
    return ACTION_CONTEXT_CHANNEL in channels


def predictor_kinematics_channels(channels: Sequence[str]) -> list[str]:
    return kinematics_channels(channels)


def kinematics_dim(channels: Sequence[str]) -> int:
    return len(kinematics_channels(channels))


def predictor_kinematics_dim(channels: Sequence[str]) -> int:
    return kinematics_dim(channels)


def uses_implicit_transition_shift(channel: str) -> bool:
    parsed = parse_kinematics_channel(channel)
    if parsed is None:
        return False
    base_channel, explicit_offset = parsed
    return explicit_offset is None and base_channel in TRANSITION_KINEMATICS_CHANNELS


def _shift_temporal_channel(values: Tensor, *, offset: int, fill_boundary: bool) -> Tensor:
    if offset == 0:
        return values
    if values.ndim < 2:
        raise ValueError(
            "Temporally indexed kinematics require a time dimension before the channel dimension."
        )
    shifted = torch.zeros_like(values)
    if values.shape[-2] == 0:
        return shifted
    if offset == -1:
        source_index = [slice(None)] * values.ndim
        target_index = [slice(None)] * values.ndim
        source_index[-2] = slice(None, -1)
        target_index[-2] = slice(1, None)
        shifted[tuple(target_index)] = values[tuple(source_index)]
        if fill_boundary:
            first_index = [slice(None)] * values.ndim
            first_index[-2] = 0
            shifted[tuple(first_index)] = values[tuple(first_index)]
        return shifted
    if offset == 1:
        source_index = [slice(None)] * values.ndim
        target_index = [slice(None)] * values.ndim
        source_index[-2] = slice(1, None)
        target_index[-2] = slice(None, -1)
        shifted[tuple(target_index)] = values[tuple(source_index)]
        if fill_boundary:
            last_index = [slice(None)] * values.ndim
            last_index[-2] = -1
            shifted[tuple(last_index)] = values[tuple(last_index)]
        return shifted
    raise ValueError(f"Unsupported temporal kinematics offset {offset}.")


def select_kinematics(
    kinematics: Tensor | None,
    channels: Sequence[str],
    *,
    context_label: str,
) -> Tensor | None:
    selected_channels = kinematics_channels(channels)
    if not selected_channels:
        return None
    if kinematics is None:
        raise ValueError(
            f"{context_label} {selected_channels} require kinematics, but the batch did not "
            "provide them."
        )
    required_width = required_kinematics_width(selected_channels)
    if kinematics.shape[-1] < required_width:
        raise ValueError(
            f"{context_label} {selected_channels} require at least {required_width} kinematics "
            "values, "
            f"but received width {kinematics.shape[-1]}."
        )
    selected_values: list[Tensor] = []
    for channel in selected_channels:
        parsed = parse_kinematics_channel(channel)
        if parsed is None:
            continue
        base_channel, explicit_offset = parsed
        channel_index = KINEMATICS_CHANNEL_TO_INDEX[base_channel]
        value = kinematics[..., channel_index : channel_index + 1]
        if explicit_offset is not None:
            value = _shift_temporal_channel(
                value,
                offset=explicit_offset,
                fill_boundary=base_channel not in DELTA_KINEMATICS_CHANNELS,
            )
        selected_values.append(value)
    if not selected_values:
        return None
    return torch.cat(selected_values, dim=-1)


def select_predictor_kinematics(
    kinematics: Tensor | None, channels: Sequence[str]
) -> Tensor | None:
    return select_kinematics(
        kinematics,
        channels,
        context_label="Predictor context channels",
    )
