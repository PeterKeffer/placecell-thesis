"""Spatial model builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from placecell_research.config.schema import SpatialModelConfig, TemporalFamilyConfig

from .components.backbones import (
    ActionEmbeddingBackbone,
    IdentityBackbone,
    MLPBackbone,
    ResNetBackbone,
    SimpleCNNBackbone,
    ViTBackbone,
)
from .components.heads import CodeHead
from .components.input_assemblers import PREDICTOR_INPUT_ASSEMBLER_BUILDERS
from .components.sparsifiers import (
    BlockwiseSparsifier,
    EntmaxSparsifier,
    GroupedKWinnersSparsifier,
    IdentitySparsifier,
    KWinnersSparsifier,
    LateralInhibitionSparsifier,
    SoftWTASparsifier,
    SparsemaxSparsifier,
)
from .components.teacher import EncoderStack, PostHeadRMSNorm, TeacherStudentController
from .components.temporal.capabilities import temporal_backend_capabilities
from .components.temporal.clockwork import ClockworkTemporal
from .components.temporal.mlp import MLPTemporal
from .components.temporal.mtrnn import MTRNNTemporal
from .components.temporal.recurrent import (
    GRUSequenceTemporal,
    LSTMSequenceTemporal,
    RNNSequenceTemporal,
)
from .composite import CompositePlaceModel, PlaceModelComponents
from .predictor_context import (
    kinematics_dim,
    predictor_kinematics_dim,
    required_kinematics_width,
    uses_action_context,
)
from .predictor_stack import PredictorModules
from .regularization import SequenceRegularizer
from .representation_contract import equal_code_block_bounds
from .representation_heads import build_representation_heads


@dataclass(frozen=True, slots=True)
class ModelBuildContext:
    num_actions: int
    observation_dim: int
    kinematics_dim: int
    total_optimizer_steps: int
    optimizer_steps_per_epoch: int = 1

    def to_checkpoint_payload(self, spatial_model_config: dict[str, object]) -> dict[str, object]:
        return {
            "spatial_model_config": spatial_model_config,
            "num_actions": self.num_actions,
            "observation_dim": self.observation_dim,
            "kinematics_dim": self.kinematics_dim,
            "total_optimizer_steps": self.total_optimizer_steps,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
        }


BackboneBuilder = Callable[[TemporalFamilyConfig, int], nn.Module]
TemporalBuilder = Callable[[TemporalFamilyConfig, int], nn.Module]
SparsifierBuilder = Callable[[object, int, int], nn.Module]
ModelArchitectureBuilder = Callable[[SpatialModelConfig, ModelBuildContext], nn.Module]


def _validate_grouped_single_layer(
    layer_sizes: list[int],
    group_count: int,
    family: str,
    component_name: str,
) -> None:
    """Single-layer + width-divisibility guard for the grouped multi-timescale cores."""
    if len(layer_sizes) != 1:
        raise ValueError(
            f"spatial_model.{component_name}.layer_sizes must be single-layer "
            f"when family='{family}'; "
            f"got {layer_sizes}."
        )
    hidden_size = int(layer_sizes[-1])
    if group_count > 0 and hidden_size % group_count != 0:
        raise ValueError(
            f"spatial_model.{component_name}.layer_sizes width {hidden_size} must be divisible "
            f"by the {family} group count {group_count}."
        )


def validate_component_compatibility(
    config: SpatialModelConfig,
    *,
    build_context: ModelBuildContext,
) -> None:
    """Fail fast on incompatible model-component combinations."""
    if build_context.observation_dim <= 0:
        raise ValueError(f"observation_dim must be positive, got {build_context.observation_dim}.")
    if build_context.kinematics_dim < 0:
        raise ValueError(
            f"kinematics_dim must be non-negative, got {build_context.kinematics_dim}."
        )
    exports_encoder_state = config.exports_state_readout()
    if exports_encoder_state and config.encoder.family == "mlp":
        raise ValueError(
            "spatial_model state_readout requires an encoder with an explicit "
            f"recurrent state; family={config.encoder.family!r} does not export one."
        )
    observation_source = config.inputs.observation_source
    encoder_backbone_type = config.encoder.backbone_type
    if observation_source == "latent" and encoder_backbone_type not in {"identity", "mlp"}:
        raise ValueError(
            "spatial_model.inputs.observation_source=latent only supports "
            "spatial_model.encoder.backbone_type in {'identity', 'mlp'}."
        )
    if observation_source == "rgb" and encoder_backbone_type not in {"cnn", "vit"}:
        raise ValueError(
            "spatial_model.inputs.observation_source=rgb requires "
            "spatial_model.encoder.backbone_type in {'cnn', 'vit'}."
        )
    if observation_source == "action" and encoder_backbone_type != "identity":
        raise ValueError(
            "spatial_model.inputs.observation_source=action requires "
            "spatial_model.encoder.backbone_type='identity'; discrete actions are embedded "
            "by the action backbone before the temporal encoder."
        )
    if (
        observation_source == "action"
        and build_context.observation_dim != build_context.num_actions + 1
    ):
        raise ValueError(
            "spatial_model.inputs.observation_source=action requires observation_dim="
            "num_actions + 1 so the external action feature includes the padding token."
        )
    if observation_source == "action" and uses_action_context(
        config.inputs.encoder_context_channels
    ):
        raise ValueError(
            "spatial_model.inputs.observation_source=action must not also include 'action' in "
            "spatial_model.inputs.encoder_context_channels."
        )
    visual_masking = config.inputs.visual_masking
    if observation_source == "action" and (
        visual_masking.blackout_probability > 0.0 or visual_masking.stride > 1
    ):
        raise ValueError(
            "spatial_model.inputs.observation_source=action does not support visual_masking; "
            "zero is a valid discrete action, not a masking token."
        )
    if observation_source == "action" and config.inputs.input_corruption.enabled:
        raise ValueError(
            "spatial_model.inputs.observation_source=action does not support input_corruption; "
            "action masking requires a distinct discrete mask token."
        )
    required_kinematics = required_kinematics_width(config.inputs.predictor_context_channels)
    if build_context.kinematics_dim < required_kinematics:
        raise ValueError(
            "spatial_model.inputs.predictor_context_channels requires "
            f"{required_kinematics} kinematics values, but the dataset exposes "
            f"{build_context.kinematics_dim}."
        )
    required_encoder_kinematics = required_kinematics_width(config.inputs.encoder_context_channels)
    if build_context.kinematics_dim < required_encoder_kinematics:
        raise ValueError(
            "spatial_model.inputs.encoder_context_channels requires "
            f"{required_encoder_kinematics} kinematics values, but the dataset exposes "
            f"{build_context.kinematics_dim}."
        )
    if config.inputs.input_corruption.use_blackout_token and observation_source != "latent":
        raise ValueError(
            "spatial_model.inputs.input_corruption.use_blackout_token requires "
            "spatial_model.inputs.observation_source='latent'."
        )
    if config.inverse_dynamics.enabled and build_context.kinematics_dim <= 0:
        raise ValueError(
            "spatial_model.inverse_dynamics.enabled requires dataset kinematics_dim > 0."
        )
    if config.encoder.family == "clockwork":
        _validate_grouped_single_layer(
            config.encoder.layer_sizes,
            len(config.encoder.clockwork_periods),
            "clockwork",
            "encoder",
        )
    if config.encoder.family == "mtrnn":
        _validate_grouped_single_layer(
            config.encoder.layer_sizes,
            len(config.encoder.mtrnn_time_constants),
            "mtrnn",
            "encoder",
        )
    if config.predictor.family == "clockwork":
        _validate_grouped_single_layer(
            config.predictor.layer_sizes,
            len(config.predictor.clockwork_periods),
            "clockwork",
            "predictor",
        )
    if config.predictor.family == "mtrnn":
        _validate_grouped_single_layer(
            config.predictor.layer_sizes,
            len(config.predictor.mtrnn_time_constants),
            "mtrnn",
            "predictor",
        )
    predictor_capabilities = temporal_backend_capabilities(
        config.predictor.family, config.predictor.fla_variant
    )
    if not predictor_capabilities.supports_stepwise:
        reason = predictor_capabilities.stepwise_unsupported_message()
        if config.inputs.predictor_input_mode != "encoder":
            raise ValueError(
                "spatial_model.inputs.predictor_input_mode must be 'encoder' for this predictor: "
                "belief-feeding modes replay the predictor stepwise. " + reason
            )
        rollout_objectives = sorted(
            name
            for name, objective in config.objectives.items()
            if objective.type == "multistep_rollout"
        )
        if rollout_objectives:
            raise ValueError(
                f"replay objective(s) {rollout_objectives} replay the predictor stepwise. "
                + reason
            )


def _build_identity_backbone(config: TemporalFamilyConfig, observation_dim: int) -> nn.Module:
    del config
    return IdentityBackbone(observation_dim)


def _build_mlp_backbone(config: TemporalFamilyConfig, observation_dim: int) -> nn.Module:
    return MLPBackbone(observation_dim, config.input_size, 1, 0.0)


LATENT_BACKBONE_BUILDERS: dict[str, BackboneBuilder] = {
    "identity": _build_identity_backbone,
    "mlp": _build_mlp_backbone,
}

RGB_BACKBONE_BUILDERS: dict[str, BackboneBuilder] = {
    "simple": lambda config, observation_dim: SimpleCNNBackbone(
        input_channels=observation_dim,
        channels=config.cnn_channels,
        output_dim=config.input_size,
        pool_size=config.cnn_pool_size,
    ),
    "resnet18": lambda config, observation_dim: ResNetBackbone(
        output_dim=config.input_size,
        pretrained=config.cnn_pretrained,
        input_channels=observation_dim,
    ),
}


def _build_vit_backbone(config: TemporalFamilyConfig, observation_dim: int) -> nn.Module:
    return ViTBackbone(
        input_channels=observation_dim,
        output_dim=config.input_size,
        patch_size=config.vit_patch_size,
        depth=config.vit_depth,
        num_heads=config.vit_num_heads,
        mlp_ratio=config.vit_mlp_ratio,
        dropout=config.dropout,
    )


def _freeze_backbone(backbone: nn.Module) -> nn.Module:
    backbone.requires_grad_(False)
    backbone.eval()
    original_train = backbone.train

    def train_frozen(mode: bool = True) -> nn.Module:
        del mode
        original_train(False)
        return backbone

    backbone.train = train_frozen  # type: ignore[method-assign]
    backbone._placecell_frozen = True  # type: ignore[attr-defined]
    return backbone


def build_backbone(
    config: TemporalFamilyConfig,
    observation_dim: int,
    observation_source: str,
) -> nn.Module:
    if observation_source == "latent":
        builder = LATENT_BACKBONE_BUILDERS.get(config.backbone_type)
        if builder is None:
            raise ValueError(f"Unsupported latent backbone_type: {config.backbone_type}")
        return builder(config, observation_dim)

    if config.backbone_type == "vit":
        builder = _build_vit_backbone
    elif config.backbone_type == "cnn":
        builder = RGB_BACKBONE_BUILDERS.get(config.cnn_type)
        if builder is None:
            raise ValueError(f"Unsupported encoder.cnn_type: {config.cnn_type}")
    else:
        raise ValueError(
            "RGB observation_source requires encoder.backbone_type in {'cnn', 'vit'}, "
            f"got {config.backbone_type}."
        )
    backbone = builder(config, observation_dim)
    if config.cnn_freeze:
        return _freeze_backbone(backbone)
    return backbone


SHARED_TEMPORAL_BUILDERS: dict[str, TemporalBuilder] = {
    "mlp": lambda config, input_dim: MLPTemporal(input_dim, config.layer_sizes, config.dropout),
}

RECURRENT_TEMPORAL_BUILDERS: dict[str, TemporalBuilder] = {
    "rnn": lambda config, input_dim: RNNSequenceTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
    ),
    "gru": lambda config, input_dim: GRUSequenceTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        state_tau=config.state_tau,
        adaptive_state_alpha=config.adaptive_state_alpha,
        adaptive_state_gate_initial_open_probability=(
            config.adaptive_state_gate_initial_open_probability
        ),
    ),
    "gru_softplus": lambda config, input_dim: GRUSequenceTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        recurrent_activation=nn.Softplus(),
        state_tau=config.state_tau,
        adaptive_state_alpha=config.adaptive_state_alpha,
        adaptive_state_gate_initial_open_probability=(
            config.adaptive_state_gate_initial_open_probability
        ),
    ),
    "gru_relu": lambda config, input_dim: GRUSequenceTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        recurrent_activation=nn.ReLU(),
        state_tau=config.state_tau,
        adaptive_state_alpha=config.adaptive_state_alpha,
        adaptive_state_gate_initial_open_probability=(
            config.adaptive_state_gate_initial_open_probability
        ),
    ),
    "lstm": lambda config, input_dim: LSTMSequenceTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        adaptive_state_alpha=config.adaptive_state_alpha,
        adaptive_state_gate_initial_open_probability=(
            config.adaptive_state_gate_initial_open_probability
        ),
    ),
    "lstm_softplus": lambda config, input_dim: LSTMSequenceTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        recurrent_activation=nn.Softplus(),
        adaptive_state_alpha=config.adaptive_state_alpha,
        adaptive_state_gate_initial_open_probability=(
            config.adaptive_state_gate_initial_open_probability
        ),
    ),
    "lstm_relu": lambda config, input_dim: LSTMSequenceTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        recurrent_activation=nn.ReLU(),
        adaptive_state_alpha=config.adaptive_state_alpha,
        adaptive_state_gate_initial_open_probability=(
            config.adaptive_state_gate_initial_open_probability
        ),
    ),
    "clockwork": lambda config, input_dim: ClockworkTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        clock_periods=config.clockwork_periods,
    ),
    "mtrnn": lambda config, input_dim: MTRNNTemporal(
        input_dim,
        config.layer_sizes,
        dropout=config.dropout,
        time_constants=config.mtrnn_time_constants,
    ),
}

ENCODER_TEMPORAL_BUILDERS: dict[str, TemporalBuilder] = {
    **SHARED_TEMPORAL_BUILDERS,
    **RECURRENT_TEMPORAL_BUILDERS,
}

PREDICTOR_TEMPORAL_BUILDERS: dict[str, TemporalBuilder] = {
    **SHARED_TEMPORAL_BUILDERS,
    **RECURRENT_TEMPORAL_BUILDERS,
}


def _build_temporal(
    builders: dict[str, TemporalBuilder],
    config: TemporalFamilyConfig,
    input_dim: int,
) -> nn.Module:
    builder = builders.get(config.family)
    if builder is None:
        raise ValueError(f"Unsupported temporal family: {config.family}")
    module = builder(config, input_dim)
    apply_forget_gate_bias = getattr(module, "apply_forget_gate_bias", None)
    if apply_forget_gate_bias is not None:
        apply_forget_gate_bias(config.forget_bias)
    apply_chrono_init = getattr(module, "apply_chrono_init", None)
    if apply_chrono_init is not None and config.chrono_init:
        apply_chrono_init(config.chrono_t_max)
    return module


def build_encoder_temporal(
    config: TemporalFamilyConfig,
    input_dim: int,
) -> nn.Module:
    return _build_temporal(ENCODER_TEMPORAL_BUILDERS, config, input_dim)


def build_predictor_temporal(
    config: TemporalFamilyConfig,
    input_dim: int,
) -> nn.Module:
    return _build_temporal(PREDICTOR_TEMPORAL_BUILDERS, config, input_dim)


SPARSIFIER_BUILDERS: dict[str, SparsifierBuilder] = {
    "none": lambda config, num_units, steps_per_epoch: IdentitySparsifier(),
    "kwinners": lambda config, num_units, steps_per_epoch: KWinnersSparsifier(
        config.k_fraction,
        num_units=num_units,
        boost_strength=config.kwinners_boost_strength,
        boost_update_rate=config.kwinners_boost_update_rate,
        boost_anneal_steps=config.kwinners_boost_anneal_steps,
        balance_bias_rate=config.kwinners_balance_bias_rate,
        balance_bias_strategy=config.kwinners_balance_bias_strategy,
        balance_liveness_rate_ratio=config.kwinners_balance_liveness_rate_ratio,
        balance_liveness_patience_steps=max(
            1,
            round(config.kwinners_balance_liveness_patience_epochs * steps_per_epoch),
        ),
        selection_noise_scale=config.kwinners_selection_noise_scale,
        selection_noise_anneal_steps=config.kwinners_selection_noise_anneal_steps,
        selection_noise_final_scale=config.kwinners_selection_noise_final_scale,
        balance_bias_clamp=config.kwinners_balance_bias_clamp,
        balance_bias_leak=config.kwinners_balance_bias_leak,
        k_anneal_start=config.kwinners_k_anneal_start,
        k_anneal_steps=config.kwinners_k_anneal_steps,
        binarize=config.kwinners_binarize,
    ),
    "grouped_kwinners": lambda config, num_units, steps_per_epoch: GroupedKWinnersSparsifier(
        config.k_fraction,
        num_units=num_units,
        num_groups=config.grouped_num_groups,
        group_bias_rate=config.grouped_group_bias_rate,
        group_bias_clamp=config.kwinners_balance_bias_clamp,
        group_bias_leak=config.kwinners_balance_bias_leak,
    ),
    "lateral_inhibition": lambda config, num_units, steps_per_epoch: LateralInhibitionSparsifier(
        config.k_fraction,
        config.lateral_inhibition_strength,
        config.lateral_inhibition_rectify,
    ),
    "sparsemax": lambda config, num_units, steps_per_epoch: SparsemaxSparsifier(config.temperature),
    "soft_wta": lambda config, num_units, steps_per_epoch: SoftWTASparsifier(
        config.soft_wta_beta,
        config.soft_wta_learnable_beta,
        config.soft_wta_target_sparsity,
    ),
    "entmax": lambda config, num_units, steps_per_epoch: EntmaxSparsifier(config.entmax_alpha),
}


def build_sparsifier(
    config,
    num_units: int,
    *,
    optimizer_steps_per_epoch: int = 1,
) -> nn.Module:
    builder = SPARSIFIER_BUILDERS.get(config.type)
    if builder is None:
        raise ValueError(f"Unsupported sparsifier: {config.type}")
    return builder(config, num_units, optimizer_steps_per_epoch)


def _build_encoder_sparsifier(
    config: SpatialModelConfig,
    optimizer_steps_per_epoch: int,
) -> nn.Module:
    if config.code_blocks:
        blocks = [
            (
                block.start,
                block.end,
                build_sparsifier(
                    block.sparsifier,
                    block.end - block.start,
                    optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                ),
            )
            for block in sorted(config.code_blocks, key=lambda item: item.start)
        ]
        return BlockwiseSparsifier(blocks, width=config.training.code_dim)
    if config.num_code_blocks > 1:
        width = config.training.code_dim
        blocks = [
            (
                start,
                end,
                build_sparsifier(
                    config.sparsifier,
                    end - start,
                    optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                ),
            )
            for start, end in equal_code_block_bounds(width, config.num_code_blocks)
        ]
        return BlockwiseSparsifier(blocks, width=width)
    return build_sparsifier(
        config.sparsifier,
        config.training.code_dim,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )


def _encoder_slow_leak_alpha(config: SpatialModelConfig) -> float | torch.Tensor:
    if not config.code_blocks:
        return config.encoder.slow_leak_alpha

    code_dim = int(config.training.code_dim)
    alpha = torch.full(
        (code_dim,),
        float(config.encoder.slow_leak_alpha),
        dtype=torch.float32,
    )
    assigned = torch.zeros(code_dim, dtype=torch.bool)
    for block in sorted(config.code_blocks, key=lambda item: item.start):
        if block.end <= block.start or block.end > code_dim:
            raise ValueError(
                "spatial_model.code_blocks entries must satisfy "
                f"0 <= start < end <= code_dim; got start={block.start}, "
                f"end={block.end}, code_dim={code_dim}."
            )
        if bool(assigned[block.start : block.end].any()):
            raise ValueError("spatial_model.code_blocks must not overlap.")
        if block.slow_leak_alpha is not None:
            alpha[block.start : block.end] = float(block.slow_leak_alpha)
        assigned[block.start : block.end] = True
    return alpha


def build_predictor_input_assembler(
    mode: str,
    code_dim: int,
    action_dim: int,
    kinematics_dim: int,
    temporal_offset_dim: int,
) -> nn.Module:
    builder = PREDICTOR_INPUT_ASSEMBLER_BUILDERS.get(mode)
    if builder is None:
        raise ValueError(f"Unsupported predictor_input_mode: {mode}")
    return builder(code_dim, action_dim, kinematics_dim, temporal_offset_dim)


def _prepare_encoder_state_readout(temporal: nn.Module, readout_mode: str) -> nn.Module:
    if readout_mode != "state":
        return temporal
    freeze_shortcuts = getattr(temporal, "freeze_terminal_shortcut_parameters", None)
    if callable(freeze_shortcuts):
        freeze_shortcuts()
    return temporal


def _build_composite_place_model(
    config: SpatialModelConfig,
    build_context: ModelBuildContext,
) -> CompositePlaceModel:
    validate_component_compatibility(config, build_context=build_context)
    observation_encoder = (
        ActionEmbeddingBackbone(
            build_context.num_actions,
            config.encoder.action_embedding_dim,
        )
        if config.inputs.observation_source == "action"
        else build_backbone(
            config.encoder,
            build_context.observation_dim,
            config.inputs.observation_source,
        )
    )
    use_encoder_action_context = uses_action_context(config.inputs.encoder_context_channels)
    encoder_context_dim = (
        (config.encoder.action_embedding_dim if use_encoder_action_context else 0)
        + kinematics_dim(config.inputs.encoder_context_channels)
        + int(config.top_down_context_dim)
    )
    encoder_temporal = _prepare_encoder_state_readout(
        build_encoder_temporal(
            config.encoder,
            observation_encoder.output_dim + encoder_context_dim,
        ),
        config.encoder_readout,
    )
    encoder_head = CodeHead(
        input_dim=config.encoder.output_size,
        output_dim=config.training.code_dim,
        activation=config.encoder.head_activation,
        normalize=config.encoder.normalize_codes,
        weight_sparsity=config.encoder.head_weight_sparsity,
    )
    encoder_expert_heads = (
        nn.ModuleList(
            CodeHead(
                input_dim=config.encoder.output_size,
                output_dim=config.training.code_dim,
                activation=config.encoder.head_activation,
                normalize=config.encoder.normalize_codes,
                weight_sparsity=config.encoder.head_weight_sparsity,
            )
            for _ in range(config.encoder_experts.count - 1)
        )
        if config.encoder_experts.count > 1 and not config.encoder_experts.streams
        else None
    )
    encoder_stream_names = list(config.encoder_experts.streams[1:])
    encoder_stream_temporals: nn.ModuleList | None = None
    encoder_stream_heads: nn.ModuleList | None = None
    if encoder_stream_names:
        encoder_stream_temporals = nn.ModuleList()
        encoder_stream_heads = nn.ModuleList()
        for stream_name in encoder_stream_names:
            stream_input_dim = (
                observation_encoder.output_dim
                if stream_name == "vision"
                else build_context.kinematics_dim
            )
            encoder_stream_temporals.append(
                _prepare_encoder_state_readout(
                    build_encoder_temporal(config.encoder, stream_input_dim),
                    config.encoder_readout,
                )
            )
            encoder_stream_heads.append(
                CodeHead(
                    input_dim=config.encoder.output_size,
                    output_dim=config.training.code_dim,
                    activation=config.encoder.head_activation,
                    normalize=config.encoder.normalize_codes,
                    weight_sparsity=config.encoder.head_weight_sparsity,
                )
            )
    encoder_precision_heads = (
        nn.ModuleList(
            nn.Linear(config.training.code_dim, 1) for _ in range(config.encoder_experts.count)
        )
        if (
            config.encoder_experts.count > 1
            and config.encoder_experts.combiner == "product"
            and config.encoder_experts.precision_weighted
        )
        else None
    )
    encoder_code_norm = {
        "none": nn.Identity,
        "layernorm": lambda: nn.LayerNorm(config.training.code_dim),
        "rmsnorm_fixed": lambda: nn.RMSNorm(
            config.training.code_dim,
            elementwise_affine=False,
        ),
    }[config.encoder_pre_head_norm]()
    encoder_post_head_norm = {
        "none": nn.Identity,
        "rmsnorm_fixed": lambda: PostHeadRMSNorm(
            config.training.code_dim,
            elementwise_affine=False,
        ),
        "rmsnorm": lambda: PostHeadRMSNorm(config.training.code_dim),
    }[config.encoder_post_head_norm]()
    shared_sparsifier = _build_encoder_sparsifier(
        config,
        build_context.optimizer_steps_per_epoch,
    )
    encoder_action_embedding = (
        nn.Embedding(build_context.num_actions + 1, config.encoder.action_embedding_dim)
        if use_encoder_action_context
        else None
    )
    exports_encoder_state = config.exports_state_readout()
    encoder_stack = EncoderStack(
        observation_encoder,
        encoder_temporal,
        encoder_head,
        shared_sparsifier,
        gradient_checkpointing=config.encoder.cnn_gradient_checkpointing,
        slow_leak_alpha=_encoder_slow_leak_alpha(config),
        slow_leak_mode=config.encoder.slow_leak_mode,
        context_channels=config.inputs.encoder_context_channels,
        action_embedding=encoder_action_embedding,
        encoder_code_norm=encoder_code_norm,
        encoder_post_head_norm=encoder_post_head_norm,
        expert_heads=encoder_expert_heads,
        precision_heads=encoder_precision_heads,
        stream_temporals=encoder_stream_temporals,
        stream_heads=encoder_stream_heads,
        stream_names=encoder_stream_names,
        readout_mode=config.encoder_readout,
        export_state_readout=exports_encoder_state,
    )
    teacher_controller = TeacherStudentController(
        encoder_stack,
        config.teacher_student,
        build_context.total_optimizer_steps,
    )
    training_regularizer = SequenceRegularizer(config.regularization)
    use_action_context = uses_action_context(config.inputs.predictor_context_channels)
    selected_kinematics_dim = predictor_kinematics_dim(config.inputs.predictor_context_channels)
    temporal_offset_dim = 1 if config.inputs.input_corruption.append_temporal_offset else 0
    action_embedding = (
        nn.Embedding(build_context.num_actions + 1, config.predictor.action_embedding_dim)
        if use_action_context
        else None
    )
    inverse_dynamics_head = (
        nn.Sequential(
            nn.Linear(config.training.code_dim * 2, config.inverse_dynamics.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.inverse_dynamics.hidden_dim, build_context.kinematics_dim),
        )
        if config.inverse_dynamics.enabled
        else None
    )
    input_corruption_blackout_token = (
        nn.Parameter(nn.init.normal_(torch.empty(1, 1, build_context.observation_dim), std=0.02))
        if config.inputs.input_corruption.use_blackout_token
        else None
    )
    predictor_input_assembler = build_predictor_input_assembler(
        config.inputs.predictor_input_mode,
        code_dim=config.training.code_dim,
        action_dim=config.predictor.action_embedding_dim if use_action_context else 0,
        kinematics_dim=selected_kinematics_dim,
        temporal_offset_dim=temporal_offset_dim,
    )
    predictor_temporal = build_predictor_temporal(
        config.predictor,
        predictor_input_assembler.output_dim,
    )
    if config.predictor_free_partition_fraction > 0.0:
        if not hasattr(predictor_temporal, "set_free_partition"):
            raise ValueError(
                "predictor_free_partition_fraction > 0 requires a projected recurrent predictor "
                f"family (rnn/gru/lstm); family '{config.predictor.family}' has no free partition."
            )
        predictor_temporal.set_free_partition(
            config.predictor_free_partition_fraction, config.training.code_dim
        )

    def _new_predictor_head() -> CodeHead:
        return CodeHead(
            input_dim=config.predictor.output_size,
            output_dim=config.training.code_dim,
            activation=config.predictor.head_activation,
            normalize=config.predictor.normalize_codes,
            weight_sparsity=config.predictor.head_weight_sparsity,
        )

    predictor_head = _new_predictor_head()
    predictor_sparsifier = build_sparsifier(
        config.predictor_sparsifier,
        config.training.code_dim,
        optimizer_steps_per_epoch=build_context.optimizer_steps_per_epoch,
    )
    slow_operator_projection = (
        nn.Sequential(
            nn.Linear(config.training.code_dim, config.slow_operator_projection_hidden),
            nn.ReLU(),
            nn.Linear(config.slow_operator_projection_hidden, config.training.code_dim),
        )
        if config.slow_operator_projection_hidden > 0
        else None
    )
    teacher_controller.attach_predictor(
        PredictorModules(
            predictor_input_assembler=predictor_input_assembler,
            predictor_temporal=predictor_temporal,
            predictor_head=predictor_head,
            predictor_sparsifier=predictor_sparsifier,
            action_embedding=action_embedding,
        )
    )
    components = PlaceModelComponents(
        observation_encoder=observation_encoder,
        encoder_stack=encoder_stack,
        predictor_input_assembler=predictor_input_assembler,
        predictor_temporal=predictor_temporal,
        predictor_head=predictor_head,
        predictor_sparsifier=predictor_sparsifier,
        masked_predictor=None,
        teacher_controller=teacher_controller,
        action_embedding=action_embedding,
        inverse_dynamics_head=inverse_dynamics_head,
        input_corruption_blackout_token=input_corruption_blackout_token,
        training_regularizer=training_regularizer,
        slow_operator_projection=slow_operator_projection,
        attractor_binding=None,
        config=config,
        num_actions=build_context.num_actions,
        observation_dim=build_context.observation_dim,
        kinematics_dim=selected_kinematics_dim,
    )
    model = CompositePlaceModel(components)
    model.set_representation_heads(
        build_representation_heads(
            config,
            num_actions=build_context.num_actions,
            optimizer_steps_per_epoch=build_context.optimizer_steps_per_epoch,
        )
    )
    return model


MODEL_ARCHITECTURES: dict[str, ModelArchitectureBuilder] = {
    "composite": _build_composite_place_model,
}


def build_place_model(
    config: SpatialModelConfig,
    build_context: ModelBuildContext,
) -> nn.Module:
    """Build a place model from the configured architecture registry."""
    removed = config.removed_feature_settings()
    if removed:
        raise ValueError(f"These settings are not supported in this repository: {removed}.")
    builder = MODEL_ARCHITECTURES.get(config.architecture)
    if builder is None:
        raise ValueError(f"Unsupported spatial_model.architecture: {config.architecture}")
    return builder(config, build_context)
