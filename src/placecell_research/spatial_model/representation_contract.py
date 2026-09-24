"""Canonical spatial-model representation names and shapes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .predictor_context import uses_action_context

if TYPE_CHECKING:
    from placecell_research.config.schema import SpatialModelConfig


MODULE_OUTPUT_FIELDS: tuple[str, ...] = (
    "place_codes",
    "place_logits",
    "pre_sparsifier",
    "hidden_state",
    "backbone_output",
    "state_readout",
)


def representation_name(module_name: str, field_name: str) -> str:
    return f"{module_name}.{field_name}"


def equal_code_block_bounds(width: int, count: int) -> list[tuple[int, int]]:
    """Tile [0, width) into count contiguous floor-division blocks."""
    return [(index * width // count, (index + 1) * width // count) for index in range(count)]


def representation_view_specs(config: SpatialModelConfig) -> dict[str, tuple[str, int, int]]:
    """Map each configured view name to (source_name, start, end)."""
    specs: dict[str, tuple[str, int, int]] = {}
    for block in sorted(config.code_blocks, key=lambda item: item.start):
        for module_name in ("encoder", "predictor"):
            view_name = representation_name(module_name, block.name)
            if view_name in specs:
                raise ValueError(f"duplicate representation view name: {view_name!r}.")
            specs[view_name] = (
                representation_name(module_name, "place_codes"),
                int(block.start),
                int(block.end),
            )
    for view in config.representation_views:
        if view.name in specs:
            raise ValueError(f"duplicate representation view name: {view.name!r}.")
        specs[view.name] = (view.source, int(view.start), int(view.end))
    return specs


LAYERWISE_HIDDEN_STATE_FAMILIES = frozenset(
    {"gru", "gru_softplus", "gru_relu", "lstm", "lstm_softplus", "lstm_relu"}
)


def _layerwise_hidden_state_shapes(
    module_name: str, layer_sizes: list[int], family: str
) -> dict[str, list[object]]:
    if len(layer_sizes) <= 1 or family not in LAYERWISE_HIDDEN_STATE_FAMILIES:
        return {}
    return {
        representation_name(module_name, f"hidden_state_layer_{layer_index}"): [
            "B",
            "T",
            hidden_size,
        ]
        for layer_index, hidden_size in enumerate(layer_sizes)
    }


def base_representation_shapes(config: SpatialModelConfig) -> dict[str, list[object]]:
    shapes = {
        "encoder.place_codes": ["B", "T", config.training.code_dim],
        "predictor.place_codes": ["B", "T", config.training.code_dim],
        "teacher.place_codes": ["B", "T", config.training.code_dim],
        "encoder.hidden_state": ["B", "T", config.encoder.output_size],
        "encoder.backbone_output": ["B", "T", config.encoder.output_size],
        "teacher.backbone_output": ["B", "T", config.encoder.output_size],
        "predictor.hidden_state": ["B", "T", config.predictor.output_size],
        "encoder.place_logits": ["B", "T", config.training.code_dim],
        "predictor.place_logits": ["B", "T", config.training.code_dim],
        "predictor.pre_sparsifier": ["B", "T", config.training.code_dim],
        "encoder.pre_sparsifier": ["B", "T", config.training.code_dim],
        "teacher.pre_sparsifier": ["B", "T", config.training.code_dim],
    }
    if config.teacher_student.predictor_ema:
        for output_name in ("place_codes", "place_logits", "pre_sparsifier"):
            shapes[representation_name("teacher_predictor", output_name)] = [
                "B",
                "T",
                config.training.code_dim,
            ]
        shapes["teacher_predictor.hidden_state"] = ["B", "T", config.predictor.output_size]
    if config.slow_operator_projection_hidden > 0:
        shapes[representation_name("encoder", "slow_operator_projection")] = [
            "B",
            "T",
            config.training.code_dim,
        ]
    exports_encoder_state = config.exports_state_readout()
    if exports_encoder_state:
        shapes["encoder.state_readout"] = ["B", "T", config.encoder.output_size]
        shapes["teacher.state_readout"] = ["B", "T", config.encoder.output_size]
    if config.encoder.adaptive_state_alpha is not None:
        adaptive_layer_count = len(config.encoder.layer_sizes)
        shapes["encoder.adaptive_state_gate_open_probability"] = [
            "B",
            "T",
            adaptive_layer_count,
        ]
        shapes["encoder.adaptive_state_update_rate"] = [
            "B",
            "T",
            adaptive_layer_count,
        ]
        if config.teacher_student.mode != "none":
            shapes["teacher.adaptive_state_gate_open_probability"] = [
                "B",
                "T",
                adaptive_layer_count,
            ]
            shapes["teacher.adaptive_state_update_rate"] = [
                "B",
                "T",
                adaptive_layer_count,
            ]
    for module_name in ("encoder", "predictor"):
        module_config = getattr(config, module_name)
        shapes.update(
            _layerwise_hidden_state_shapes(
                module_name, module_config.layer_sizes, module_config.family
            )
        )
    for head_config in config.cell_type_heads:
        if head_config.predictor_hidden_dim:
            shapes[representation_name(head_config.name, "place_logits")] = [
                "B",
                "T",
                config.training.code_dim,
            ]
        for output_name in ("place_codes", "pre_sparsifier"):
            shapes[representation_name(head_config.name, output_name)] = [
                "B",
                "T",
                int(head_config.code_dim),
            ]
    for view_name, (_source, start, end) in representation_view_specs(config).items():
        shapes[view_name] = ["B", "T", int(end) - int(start)]
    return shapes


def base_representation_names(config: SpatialModelConfig) -> list[str]:
    return sorted(base_representation_shapes(config))


def auxiliary_representation_names(auxiliary_heads: Iterable[object]) -> list[str]:
    names: list[str] = []
    for head in auxiliary_heads:
        target_module = getattr(head, "target_module", "predictor")
        for output_name in head.output_names():
            names.append(representation_name(target_module, output_name))
    return names


def auxiliary_representation_shapes(
    auxiliary_heads: Iterable[object],
    default_output_dim: int,
) -> dict[str, list[object]]:
    shapes: dict[str, list[object]] = {}
    for head in auxiliary_heads:
        target_module = getattr(head, "target_module", "predictor")
        output_dim = int(getattr(head, "output_dim", default_output_dim))
        for output_name in head.output_names():
            shapes[representation_name(target_module, output_name)] = ["B", "T", output_dim]
    return shapes


def predictor_input_channels(config: SpatialModelConfig) -> list[str]:
    channels = ["place_code", *config.inputs.predictor_context_channels]
    if config.inputs.input_corruption.append_temporal_offset:
        channels.append("temporal_offset")
    return channels


def encoder_input_channels(config: SpatialModelConfig) -> list[str]:
    return ["observation", *config.inputs.encoder_context_channels]


def encoder_uses_action_context(config: SpatialModelConfig) -> bool:
    return uses_action_context(config.inputs.encoder_context_channels)


def predictor_uses_action_context(config: SpatialModelConfig) -> bool:
    return uses_action_context(config.inputs.predictor_context_channels)
