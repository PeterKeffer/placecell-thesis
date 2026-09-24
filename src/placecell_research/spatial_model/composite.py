"""Composite place model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from placecell_research.config.schema import SpatialModelConfig

from .components.teacher import EncoderStack, TeacherStudentController
from .forward import PlaceModelChunkState, forward_chunk, forward_sequence
from .protocol import PlaceModel
from .representation_contract import (
    auxiliary_representation_names,
    auxiliary_representation_shapes,
    base_representation_names,
    base_representation_shapes,
    encoder_input_channels,
    encoder_uses_action_context,
    predictor_input_channels,
    predictor_uses_action_context,
)
from .types import RepresentationBundle


@dataclass
class PlaceModelComponents:
    observation_encoder: nn.Module
    encoder_stack: EncoderStack
    predictor_input_assembler: nn.Module
    predictor_temporal: nn.Module
    predictor_head: nn.Module
    predictor_sparsifier: nn.Module
    teacher_controller: TeacherStudentController | None
    action_embedding: nn.Embedding | None
    inverse_dynamics_head: nn.Module | None
    input_corruption_blackout_token: nn.Parameter | None
    training_regularizer: nn.Module | None
    slow_operator_projection: nn.Module | None
    config: SpatialModelConfig
    num_actions: int
    observation_dim: int
    kinematics_dim: int


class CompositePlaceModel(nn.Module, PlaceModel):
    """Standard place model implementation."""

    def __init__(self, components: PlaceModelComponents) -> None:
        super().__init__()
        self.components = components
        self.encoder_stack = components.encoder_stack
        self.predictor_input_assembler = components.predictor_input_assembler
        self.predictor_temporal = components.predictor_temporal
        self.predictor_head = components.predictor_head
        self.predictor_sparsifier = components.predictor_sparsifier
        self.action_embedding = components.action_embedding
        self.inverse_dynamics_head = components.inverse_dynamics_head
        self.input_corruption_blackout_token = components.input_corruption_blackout_token
        self.training_regularizer = components.training_regularizer
        self.slow_operator_projection = components.slow_operator_projection
        self.teacher_controller = components.teacher_controller
        self.auxiliary_heads = nn.ModuleDict()
        self.representation_heads = nn.ModuleDict()
        self.export_open_loop_rollout_steps: int | None = None

    def train(self, mode: bool = True) -> CompositePlaceModel:
        super().train(mode)
        if self.teacher_controller is not None and self.teacher_controller.enabled:
            self.teacher_controller.teacher_encoder_stack.eval()
        return self

    def forward_sequence(self, batch: dict[str, Tensor]) -> RepresentationBundle:
        return forward_sequence(self, batch)

    def forward_chunk(
        self,
        batch: dict[str, Tensor],
        state: PlaceModelChunkState | None = None,
        *,
        new_episode: bool = False,
    ) -> tuple[RepresentationBundle, PlaceModelChunkState]:
        return forward_chunk(self, batch, state, new_episode=new_episode)

    @staticmethod
    def detach_chunk_state(state: PlaceModelChunkState) -> PlaceModelChunkState:
        return state.detached()

    def set_auxiliary_heads(self, auxiliary_heads: nn.ModuleDict) -> None:
        self.auxiliary_heads = auxiliary_heads

    def set_representation_heads(self, representation_heads: nn.ModuleDict) -> None:
        self.representation_heads = representation_heads

    def post_optimizer_step(self) -> None:
        """Dispatch local post-update rules owned by representation heads."""
        for head in self.representation_heads.values():
            post_step = getattr(head, "post_optimizer_step", None)
            if post_step is not None and any(
                parameter.requires_grad for parameter in head.parameters()
            ):
                post_step()

    def parameters_by_group(self) -> dict[str, list[nn.Parameter]]:
        binding_parameters = list(self.encoder_stack.encoder_head.parameters())
        binding_parameter_ids = {id(parameter) for parameter in binding_parameters}
        encoder_parameters = [
            parameter
            for parameter in self.encoder_stack.parameters()
            if id(parameter) not in binding_parameter_ids
        ]
        if self.input_corruption_blackout_token is not None:
            encoder_parameters.append(self.input_corruption_blackout_token)
        if self.slow_operator_projection is not None:
            encoder_parameters.extend(self.slow_operator_projection.parameters())
        embedding_parameters = list(self.predictor_input_assembler.parameters())
        if self.action_embedding is not None:
            embedding_parameters.extend(self.action_embedding.parameters())
        head_parameters = [
            parameter
            for head in self.representation_heads.values()
            for parameter in head.parameters()
        ]
        groups = {
            "encoder": encoder_parameters,
            "encoder_binding": binding_parameters,
            "predictor": (
                list(self.predictor_temporal.parameters()) + list(self.predictor_head.parameters())
            ),
            "sparsifier": list(self.predictor_sparsifier.parameters()),
            "embeddings": embedding_parameters,
            "auxiliary": list(self.auxiliary_heads.parameters())
            + (
                list(self.inverse_dynamics_head.parameters())
                if self.inverse_dynamics_head is not None
                else []
            ),
        }
        if head_parameters:
            groups["representation_heads"] = head_parameters
        return groups

    def _selector_module_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Map each phase-schedule MODULE selector to its parameters."""
        predictor_parameters = list(self.predictor_temporal.parameters()) + list(
            self.predictor_head.parameters()
        )
        encoder_parameters = list(self.encoder_stack.parameters())
        if self.input_corruption_blackout_token is not None:
            encoder_parameters.append(self.input_corruption_blackout_token)
        if self.slow_operator_projection is not None:
            encoder_parameters.extend(self.slow_operator_projection.parameters())
        selectors: dict[str, list[nn.Parameter]] = {
            "encoder": encoder_parameters,
            "encoder_head": list(self.encoder_stack.encoder_head.parameters()),
            "predictor": predictor_parameters,
            "sparsifier": list(self.predictor_sparsifier.parameters()),
        }
        embedding_parameters = list(self.predictor_input_assembler.parameters())
        if self.action_embedding is not None:
            embedding_parameters.extend(self.action_embedding.parameters())
        selectors["embeddings"] = embedding_parameters
        if self.inverse_dynamics_head is not None:
            selectors["inverse_dynamics"] = list(self.inverse_dynamics_head.parameters())
        for namespace, head in self.representation_heads.items():
            selectors[namespace] = list(head.parameters())
        return selectors

    PREDICTOR_CLIP_SELECTORS = ("predictor", "sparsifier", "embeddings")

    def clip_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Partition parameters into the two CAUSAL groups for component-mode gradient clipping."""
        selectors = self._selector_module_parameters()
        predictor_parameters: list[nn.Parameter] = []
        for name in self.PREDICTOR_CLIP_SELECTORS:
            predictor_parameters.extend(selectors.get(name, []))
        return {
            "encoder": list(selectors.get("encoder", [])),
            "predictor": predictor_parameters,
        }

    def available_selectors(self) -> set[str]:
        """The module selectors present for this model (the valid train: entries for a phase)."""
        return set(self._selector_module_parameters())

    def _resolve_selectors(self, selectors: set[str]) -> set[int]:
        """Resolve module selectors to the set of parameter ids they cover."""
        selector_parameters = self._selector_module_parameters()
        unknown = sorted(selectors - selector_parameters.keys())
        if unknown:
            raise ValueError(
                f"Unknown training-phase selector(s) {unknown}; valid selectors are "
                f"{sorted(selector_parameters)}."
            )
        return {
            id(parameter) for selector in selectors for parameter in selector_parameters[selector]
        }

    def _permanently_frozen_parameter_ids(self) -> set[int]:
        """Parameters owned by a build-frozen module such as cnn_freeze."""
        return {
            id(parameter)
            for module in self.modules()
            if getattr(module, "_placecell_frozen", False)
            for parameter in module.parameters()
        }

    def set_trainable(self, selectors: set[str]) -> None:
        trainable_parameter_ids = self._resolve_selectors(selectors)
        permanently_frozen_parameter_ids = self._permanently_frozen_parameter_ids()
        seen_parameter_ids: set[int] = set()
        managed_parameter_lists = list(self.parameters_by_group().values())
        for group_parameters in managed_parameter_lists:
            for parameter in group_parameters:
                if id(parameter) in seen_parameter_ids:
                    continue
                seen_parameter_ids.add(id(parameter))
                parameter.requires_grad_(
                    id(parameter) in trainable_parameter_ids
                    and id(parameter) not in permanently_frozen_parameter_ids
                )
        for head in self.auxiliary_heads.values():
            train_head = head.target_module in selectors
            for parameter in head.parameters():
                parameter.requires_grad_(train_head)

    def update_teacher(self, current_step: int | None = None) -> None:
        if self.teacher_controller is not None:
            self.teacher_controller.update(current_step)

    def architecture_summary(self) -> str:
        return str(self)

    def model_contract(self) -> dict:
        config = self.components.config
        available_representations = base_representation_names(config)
        active_auxiliary_heads = sorted(self.auxiliary_heads.keys())
        available_representations.extend(
            auxiliary_representation_names(self.auxiliary_heads.values())
        )
        tensor_shapes = base_representation_shapes(config)
        if "observation.backbone_output" not in available_representations:
            available_representations.append("observation.backbone_output")
        tensor_shapes["observation.backbone_output"] = ["B", "T", self.components.observation_dim]
        tensor_shapes.update(
            auxiliary_representation_shapes(
                self.auxiliary_heads.values(),
                default_output_dim=config.training.code_dim,
            )
        )
        return {
            "available_representations": available_representations,
            "tensor_shapes": tensor_shapes,
            "teacher_student_mode": config.teacher_student.mode,
            "encoder_input_channels": encoder_input_channels(config),
            "observation_dim": self.components.observation_dim,
            "kinematics_dim": self.components.kinematics_dim,
            "num_actions": self.components.num_actions,
            "encoder_uses_action_context": encoder_uses_action_context(config),
            "predictor_input_mode": config.inputs.predictor_input_mode,
            "prediction_bootstrap_enabled": config.prediction_bootstrap.enabled,
            "inverse_dynamics_enabled": config.inverse_dynamics.enabled,
            "rollout_detach_intermediate_predictions": (
                config.rollout.detach_intermediate_predictions
            ),
            "predictor_input_channels": predictor_input_channels(config),
            "predictor_uses_action_context": predictor_uses_action_context(config),
            "training_regularization_targets": (
                sorted(self.training_regularizer.active_targets())
                if self.training_regularizer is not None
                and hasattr(self.training_regularizer, "active_targets")
                else []
            ),
            "sparsifier": config.sparsifier.type,
            "predictor_sparsifier": config.predictor_sparsifier.type,
            "checkpoint_selection_metric": config.training.selection.primary_metric,
            "code_dim": config.training.code_dim,
            "encoder_slow_leak_alpha": config.encoder.slow_leak_alpha,
            "encoder_family": config.encoder.family,
            "encoder_adaptive_state_alpha": config.encoder.adaptive_state_alpha,
            "encoder_adaptive_state_gate_initial_open_probability": (
                config.encoder.adaptive_state_gate_initial_open_probability
            ),
            "encoder_readout": config.encoder_readout,
            "encoder_observation_delay_steps": config.inputs.encoder_observation_delay_steps,
            "predictor_family": config.predictor.family,
            "predictor_adaptive_state_alpha": config.predictor.adaptive_state_alpha,
            "predictor_adaptive_state_gate_initial_open_probability": (
                config.predictor.adaptive_state_gate_initial_open_probability
            ),
            "active_auxiliary_heads": active_auxiliary_heads,
        }
