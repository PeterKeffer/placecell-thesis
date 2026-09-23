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
    return [
        (index * width // count, (index + 1) * width // count)
        for index in range(count)
    ]


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


def _supports_layerwise_hidden_states(
    family: str,
    *,
    for_predictor: bool,
    xlstm_variant: str | None = None,
    xlstm_backend: str | None = None,
) -> bool:
    recurrent_families = {
        "wyss",
        "leaky_hierarchy",
        "gru",
        "gru_softplus",
        "gru_relu",
        "lstm",
        "lstm_softplus",
        "lstm_relu",
        "ssm",
        "ema_ssm",
    }
    if family in recurrent_families:
        return True
    if family in {"mamba", "fla"}:
        return True
    if family == "transformer":
        return True
    if family == "xlstm":
        return xlstm_variant in {"simple_mlstm", "simple_slstm", "block_stack"} and (
            xlstm_backend != "nxai"
        )
    return False


def _layerwise_hidden_state_names(
    module_name: str,
    layer_sizes: list[int],
    *,
    family: str,
    for_predictor: bool,
    xlstm_variant: str | None = None,
    xlstm_backend: str | None = None,
) -> list[str]:
    if len(layer_sizes) <= 1 or not _supports_layerwise_hidden_states(
        family,
        for_predictor=for_predictor,
        xlstm_variant=xlstm_variant,
        xlstm_backend=xlstm_backend,
    ):
        return []
    return [
        representation_name(module_name, f"hidden_state_layer_{layer_index}")
        for layer_index, _layer_size in enumerate(layer_sizes)
    ]


def _layerwise_hidden_state_shapes(
    module_name: str,
    layer_sizes: list[int],
    *,
    family: str,
    for_predictor: bool,
    xlstm_variant: str | None = None,
    xlstm_backend: str | None = None,
) -> dict[str, list[object]]:
    if len(layer_sizes) <= 1 or not _supports_layerwise_hidden_states(
        family,
        for_predictor=for_predictor,
        xlstm_variant=xlstm_variant,
        xlstm_backend=xlstm_backend,
    ):
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
    }
    routed_after_sparsification = (
        config.encoder_experts.count > 1
        and config.encoder_experts.combiner in {"mixture", "attention"}
    )
    if not routed_after_sparsification:
        shapes["encoder.pre_sparsifier"] = ["B", "T", config.training.code_dim]
        shapes["teacher.pre_sparsifier"] = ["B", "T", config.training.code_dim]
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
    expert_count = config.encoder_experts.count
    for expert_index in range(expert_count if expert_count > 1 else 0):
        shapes[representation_name("encoder", f"expert_logits_{expert_index}")] = [
            "B",
            "T",
            config.training.code_dim,
        ]
    mixture_like = {"mixture", "gated_product", "attention"}
    if expert_count > 1 and config.encoder_experts.combiner in mixture_like:
        shapes[representation_name("encoder", "gate_responsibilities")] = ["B", "T", expert_count]
    if config.environment_code.enabled:
        shapes[representation_name("encoder", "environment_code")] = [
            "B",
            "T",
            config.environment_code.z_dim,
        ]
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
    predictor_expert_count = config.predictor_experts.count
    for expert_index in range(predictor_expert_count if predictor_expert_count > 1 else 0):
        shapes[representation_name("predictor", f"expert_logits_{expert_index}")] = [
            "B",
            "T",
            config.training.code_dim,
        ]
    if config.masked_predictor.enabled:
        shapes.update(
            {
                "jepa_predictor.place_codes": ["B", "T", config.training.code_dim],
                "jepa_predictor.place_logits": ["B", "T", config.training.code_dim],
                "jepa_predictor.pre_sparsifier": ["B", "T", config.training.code_dim],
                "jepa_predictor.hidden_state": ["B", "T", config.masked_predictor.output_size],
            }
        )
    if config.comparator.enabled:
        shapes.update(
            {
                "comparator.place_codes": ["B", "T", config.training.code_dim],
                "comparator.pre_sparsifier": ["B", "T", config.training.code_dim],
                "comparator.innovation": ["B", "T", config.training.code_dim],
            }
        )
        if config.attractor_binding.enabled:
            shapes["comparator.binding_pool"] = [
                "B",
                "T",
                config.attractor_binding.num_components,
            ]
    if config.grid_stream.enabled:
        hidden_dim = config.grid_stream.hidden_dim
        shapes.update(
            {
                "grid.hidden_state": ["B", "T", hidden_dim],
                "grid.integrator_state": ["B", "T", hidden_dim],
                "grid.pred": ["B", "T", config.training.code_dim],
                "grid.target": ["B", "T", config.training.code_dim],
            }
        )
    grid_arm_family = config.enabled_emergent_grid_arm()
    if grid_arm_family == "path_coloring":
        arm = config.path_coloring_arm
        shapes.update(
            {
                "grid_arm.hidden_state": ["B", "T", arm.hidden_dim],
                "grid_arm.pred": ["B", "T", config.training.code_dim],
                "grid_arm.prior_hidden_state": ["B", "T", arm.hidden_dim],
                "grid_arm.action_context": ["B", "T", arm.action_context_dim],
            }
        )
    elif grid_arm_family == "path_coloring_fusion":
        arm = config.path_coloring_fusion_arm
        shapes.update(
            {
                "grid_arm.hidden_state": ["B", "T", arm.hidden_dim],
                "grid_arm.pred": ["B", "T", config.training.code_dim],
                "grid_arm.prior_hidden_state": ["B", "T", arm.hidden_dim],
                "grid_arm.posterior_pred": ["B", "T", config.training.code_dim],
                "grid_arm.action_context": ["B", "T", arm.action_context_dim],
                "grid_arm.fusion_gate": ["B", "T", 1],
            }
        )
    elif grid_arm_family == "wang":
        arm = config.wang_grid_arm
        shapes.update(
            {
                "grid_arm.hidden_state": ["B", "T", arm.recurrent_only_dim],
                "grid_arm.pred": ["B", "T", config.training.code_dim],
                "grid_arm.input_driven_state": ["B", "T", arm.input_driven_dim],
                "grid_arm.full_state": [
                    "B",
                    "T",
                    arm.input_driven_dim + arm.recurrent_only_dim,
                ],
            }
        )
    shapes.update(
        _layerwise_hidden_state_shapes(
            "encoder",
            config.encoder.layer_sizes,
            family=config.encoder.family,
            for_predictor=False,
            xlstm_variant=config.encoder.xlstm_variant,
            xlstm_backend=config.encoder.xlstm_backend,
        )
    )
    shapes.update(
        _layerwise_hidden_state_shapes(
            "predictor",
            config.predictor.layer_sizes,
            family=config.predictor.family,
            for_predictor=True,
            xlstm_variant=config.predictor.xlstm_variant,
            xlstm_backend=config.predictor.xlstm_backend,
        )
    )
    if config.masked_predictor.enabled:
        shapes.update(
            _layerwise_hidden_state_shapes(
                "jepa_predictor",
                config.masked_predictor.layer_sizes,
                family="transformer",
                for_predictor=False,
            )
        )
    for head_config in config.cell_type_heads:
        if head_config.predictor_hidden_dim:
            shapes[representation_name(head_config.name, "place_logits")] = [
                "B", "T", config.training.code_dim,
            ]
        for output_name in ("place_codes", "pre_sparsifier"):
            shapes[representation_name(head_config.name, output_name)] = [
                "B",
                "T",
                int(head_config.code_dim),
            ]
    if config.predictive_context.enabled:
        context = config.predictive_context
        for name in ("pred", "target", "innovation"):
            shapes[f"predictive_context.{name}"] = list(shapes[context.source])
        for name in ("hidden_state", "prior_mean", "prior_std", "posterior_mean", "posterior_std"):
            shapes[f"predictive_context.{name}"] = ["B", "T", context.hidden_dim]
        shapes["predictive_context.observation_mask"] = ["B", "T", 1]
    if config.motion.enabled:
        target = config.motion.target_source
        if target == "observation.backbone_output":
            shapes[target] = ["B", "T", None]
        if target not in shapes or target.startswith("motion."):
            raise ValueError(f"Unavailable motion target: {target}")
        for name in ("pred", "posterior_pred", "target"):
            shapes[f"motion.{name}"] = list(shapes[target])
        for name in ("hidden_state", "prior_hidden_state"):
            width = (
                config.motion.readout_dim
                if config.motion.dynamics in {"lstm", "persistent_lstm"}
                else config.motion.hidden_dim
            )
            shapes[f"motion.{name}"] = ["B", "T", width]
        if config.motion.dynamics == "predict_correct_gru":
            for name in ("place_codes", "pre_sparsifier"):
                shapes[f"motion.{name}"] = ["B", "T", config.motion.readout_dim]
            for name in ("correction", "zero_correction"):
                shapes[f"motion.{name}"] = ["B", "T", config.motion.hidden_dim]
        if config.motion.dynamics != "gated_sigmoid":
            shapes["motion.recurrent_state"] = ["B", "T", config.motion.hidden_dim]
        shapes["motion.anchor_mask"] = ["B", "T", 1]
        shapes["motion.activity"] = ["B", "T", config.motion.hidden_dim]
        if config.motion.fatigue is not None and config.motion.fatigue.strength > 0:
            shapes["motion.fatigue"] = ["B", "T", config.motion.hidden_dim]
        if config.motion.dynamics == "dale_rnn":
            driven = int(config.motion.hidden_dim * config.motion.dale.input_driven_fraction)
            shapes["motion.input_driven_state"] = ["B", "T", driven]
            shapes["motion.recurrent_only_state"] = ["B", "T", config.motion.hidden_dim - driven]
    for view_name, (_source, start, end) in representation_view_specs(config).items():
        shapes[view_name] = ["B", "T", int(end) - int(start)]
    return shapes


def base_representation_names(config: SpatialModelConfig) -> list[str]:
    names = list(base_representation_shapes(config))
    names.extend(
        _layerwise_hidden_state_names(
            "encoder",
            config.encoder.layer_sizes,
            family=config.encoder.family,
            for_predictor=False,
            xlstm_variant=config.encoder.xlstm_variant,
            xlstm_backend=config.encoder.xlstm_backend,
        )
    )
    names.extend(
        _layerwise_hidden_state_names(
            "predictor",
            config.predictor.layer_sizes,
            family=config.predictor.family,
            for_predictor=True,
            xlstm_variant=config.predictor.xlstm_variant,
            xlstm_backend=config.predictor.xlstm_backend,
        )
    )
    if config.masked_predictor.enabled:
        names.extend(
            _layerwise_hidden_state_names(
                "jepa_predictor",
                config.masked_predictor.layer_sizes,
                family="transformer",
                for_predictor=False,
            )
        )
    return sorted(set(names))


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
