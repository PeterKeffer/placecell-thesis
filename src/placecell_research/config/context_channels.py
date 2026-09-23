"""Context channel names and temporal indexing semantics."""

from __future__ import annotations

from collections.abc import Sequence

ACTION_CONTEXT_CHANNEL = "action"
KINEMATICS_CHANNEL_TO_INDEX: dict[str, int] = {
    "speed": 0,
    "step_displacement": 0,
    "angular_velocity": 1,
    "sin_heading": 2,
    "cos_heading": 3,
}
TRANSITION_KINEMATICS_CHANNELS = {"speed", "step_displacement", "angular_velocity"}
DELTA_KINEMATICS_CHANNELS = TRANSITION_KINEMATICS_CHANNELS

PREFERRED_TEMPORAL_SUFFIXES: dict[str, int] = {
    "_prev": -1,
    "_curr": 0,
    "_next": 1,
}
TEMPORAL_SUFFIX_ALIASES: dict[str, int] = {
    "_tm1": -1,
    "_t": 0,
    "_tp1": 1,
    "tm1": -1,
    "t": 0,
    "tp1": 1,
}
TEMPORAL_SUFFIX_OFFSETS: tuple[tuple[str, int], ...] = tuple(
    {
        **PREFERRED_TEMPORAL_SUFFIXES,
        **TEMPORAL_SUFFIX_ALIASES,
    }.items()
)


def parse_kinematics_channel(channel: str) -> tuple[str, int | None] | None:
    """Return (base_channel, temporal_offset) for a kinematics channel."""
    if channel in KINEMATICS_CHANNEL_TO_INDEX:
        return channel, None
    for suffix, offset in TEMPORAL_SUFFIX_OFFSETS:
        if not channel.endswith(suffix):
            continue
        base_channel = channel[: -len(suffix)]
        if base_channel in KINEMATICS_CHANNEL_TO_INDEX:
            return base_channel, offset
    return None


def is_context_channel(channel: str) -> bool:
    return channel == ACTION_CONTEXT_CHANNEL or parse_kinematics_channel(channel) is not None


def kinematics_channels(channels: Sequence[str]) -> list[str]:
    return [channel for channel in channels if parse_kinematics_channel(channel) is not None]


def required_kinematics_width(channels: Sequence[str]) -> int:
    parsed_channels = [
        parsed for channel in channels if (parsed := parse_kinematics_channel(channel)) is not None
    ]
    if not parsed_channels:
        return 0
    return (
        max(KINEMATICS_CHANNEL_TO_INDEX[base_channel] for base_channel, _offset in parsed_channels)
        + 1
    )


def allowed_context_channel_hint() -> str:
    base_channels = ", ".join([ACTION_CONTEXT_CHANNEL, *KINEMATICS_CHANNEL_TO_INDEX])
    suffixes = ", ".join(PREFERRED_TEMPORAL_SUFFIXES)
    aliases = ", ".join(TEMPORAL_SUFFIX_ALIASES)
    return (
        f"use {base_channels}; kinematics channels may add preferred suffixes "
        f"{suffixes} or aliases {aliases}"
    )
