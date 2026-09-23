"""Objective and auxiliary head registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from placecell_research.config.schema import ObjectiveConfig, SpatialModelConfig
from placecell_research.spatial_model.auxiliary_heads import ReconstructionHead
from placecell_research.spatial_model.protocol import PlaceModel
from placecell_research.spatial_model.types import RepresentationBundle

from .base import ConfiguredObjective, ObjectiveModule
from .intrinsic import compute_intrinsic_losses
from .prediction import PredictionAlignmentObjective
from .reconstruction import LatentReconstructionObjective
from .sparsity import L1CapacityObjective, L1SparsityObjective, NormalizedL1CapacityObjective
from .timescale_alignment import TimescaleAlignmentObjective
from .vicreg import VICRegObjective

MetricValue = float | Tensor

AuxiliaryHeadBuilder = Callable[[str, ObjectiveConfig, dict[str, Any]], nn.Module]


@dataclass(frozen=True)
class ObjectiveRegistration:
    objective_class: type[ConfiguredObjective]
    auxiliary_head_builder: AuxiliaryHeadBuilder | None = None
    binds_model: bool = False

    def build_auxiliary_head(
        self,
        objective_name: str,
        objective_config: ObjectiveConfig,
        model_contract: dict[str, Any],
    ) -> nn.Module | None:
        if self.auxiliary_head_builder is None:
            return None
        return self.auxiliary_head_builder(objective_name, objective_config, model_contract)

    def build_objective(
        self,
        objective_name: str,
        objective_config: ObjectiveConfig,
        model: PlaceModel,
    ) -> ObjectiveModule:
        kwargs: dict[str, Any] = {"name": objective_name, "config": objective_config}
        if self.binds_model:
            kwargs["model"] = model
        return self.objective_class(**kwargs)


def _representation_dim(model_contract: dict[str, Any], representation_name: str) -> int:
    tensor_shapes = model_contract.get("tensor_shapes", {})
    shape = tensor_shapes.get(representation_name)
    if not isinstance(shape, list) or not shape:
        raise KeyError(
            f"Model contract is missing tensor shape information for '{representation_name}'."
        )
    output_dim = shape[-1]
    if not isinstance(output_dim, int):
        raise ValueError(
            f"Model contract reported a non-integer output dimension for "
            f"'{representation_name}': {shape!r}"
        )
    return output_dim


def _auxiliary_target_module(objective_config: ObjectiveConfig) -> str:
    head_config = objective_config.auxiliary_head
    if head_config is not None:
        return head_config.target_module
    if objective_config.targets:
        return objective_config.targets[0].split(".", 1)[0]
    return "predictor"


def _build_reconstruction_head(
    objective_name: str,
    objective_config: ObjectiveConfig,
    model_contract: dict[str, Any],
) -> nn.Module:
    head_config = objective_config.auxiliary_head
    source = (
        objective_config.targets[0] if objective_config.targets else "predictor.place_codes"
    )
    return ReconstructionHead(
        objective_name,
        source=source,
        input_dim=_representation_dim(model_contract, source),
        output_dim=model_contract["observation_dim"],
        hidden_dim=head_config.head_hidden_dim if head_config is not None else None,
        target_module=_auxiliary_target_module(objective_config),
    )


OBJECTIVE_REGISTRY: dict[str, ObjectiveRegistration] = {
    "timescale_alignment": ObjectiveRegistration(TimescaleAlignmentObjective),
    "prediction_alignment": ObjectiveRegistration(
        PredictionAlignmentObjective,
        binds_model=True,
    ),
    "latent_reconstruction": ObjectiveRegistration(
        LatentReconstructionObjective,
        auxiliary_head_builder=_build_reconstruction_head,
    ),
    "vicreg": ObjectiveRegistration(VICRegObjective),
    "l1_sparsity": ObjectiveRegistration(L1SparsityObjective),
    "l1_capacity": ObjectiveRegistration(L1CapacityObjective),
    "normalized_l1_capacity": ObjectiveRegistration(NormalizedL1CapacityObjective),
}


@dataclass
class BuiltObjectives:
    objectives: list[ObjectiveModule]
    auxiliary_heads: nn.ModuleDict


def build_objectives(model: PlaceModel, model_config: SpatialModelConfig) -> BuiltObjectives:
    model_contract = model.model_contract()
    available_representations = set(model_contract["available_representations"])
    auxiliary_heads: dict[str, nn.Module] = {}
    objectives: list[ObjectiveModule] = []
    for objective_name, objective_config in model_config.objectives.items():
        registration = OBJECTIVE_REGISTRY.get(objective_config.type)
        if registration is None:
            raise KeyError(f"Unknown objective type: {objective_config.type}")
        auxiliary_head = registration.build_auxiliary_head(
            objective_name,
            objective_config,
            model_contract,
        )
        if auxiliary_head is not None:
            target_module = getattr(auxiliary_head, "target_module", None)
            if target_module is not None:
                known_modules = {
                    representation.split(".", 1)[0] for representation in available_representations
                }
                if target_module not in known_modules:
                    raise ValueError(
                        f"objective {objective_name!r} has auxiliary_head.target_module="
                        f"{target_module!r}, which is not a module this model produces. Valid "
                        f"modules: {sorted(known_modules)}. To keep a head frozen forever, name a "
                        "real module that is never a trainable selector (e.g. 'teacher')."
                    )
            auxiliary_heads[objective_name] = auxiliary_head
        objectives.append(registration.build_objective(objective_name, objective_config, model))
    available_auxiliary_outputs = {
        output_name
        for head in auxiliary_heads.values()
        for output_name in head.output_names()
    }
    for objective in objectives:
        objective.validate_configuration(available_representations, available_auxiliary_outputs)
        objective.validate_against_model(model_config)
    return BuiltObjectives(objectives=objectives, auxiliary_heads=nn.ModuleDict(auxiliary_heads))


def build_objectives_and_heads(
    model: PlaceModel,
    model_config: SpatialModelConfig,
) -> BuiltObjectives:
    """Final-spec name for the combined objective and auxiliary-head registry."""
    return build_objectives(model, model_config)


def _assert_all_finite(loss_terms: list[tuple[str, Tensor]]) -> None:
    """Single host sync over all loss terms; on failure, name the first non-finite one."""
    if not loss_terms:
        return
    stacked = torch.stack([value.detach().reshape(()) for _, value in loss_terms])
    finite_mask = torch.isfinite(stacked)
    if finite_mask.all():
        return
    for (name, _), is_finite in zip(loss_terms, finite_mask.tolist(), strict=False):
        if not is_finite:
            raise FloatingPointError(f"{name} produced a non-finite loss.")


def compute_total_loss(
    objectives: list[ObjectiveModule],
    bundle: RepresentationBundle,
    batch: dict[str, Tensor],
    model_config: SpatialModelConfig,
) -> tuple[Tensor, dict[str, MetricValue]]:
    device = bundle.infer_device()
    total_loss = torch.zeros((), device=device)
    metrics: dict[str, MetricValue] = {}
    loss_terms: list[tuple[str, Tensor]] = []

    def _queue_tensor_metric(name: str, value: Tensor) -> None:
        metrics[name] = value.detach().reshape(())

    for objective in objectives:
        result = objective.compute(bundle, batch)
        weight = model_config.objectives[objective.name].weight
        weighted_loss = weight * result.loss
        loss_terms.append((objective.name, result.loss))
        loss_terms.append((f"{objective.name}_weighted", weighted_loss))
        total_loss = total_loss + weighted_loss
        _queue_tensor_metric(f"loss/{objective.name}", result.loss)
        _queue_tensor_metric(f"loss/{objective.name}_weighted", weighted_loss)
        for key, value in result.metrics.items():
            if isinstance(value, Tensor):
                _queue_tensor_metric(f"{objective.name}/{key}", value)
            else:
                metrics[f"{objective.name}/{key}"] = float(value)
    intrinsic_loss, intrinsic_metrics = compute_intrinsic_losses(bundle, batch, model_config)
    loss_terms.append(("intrinsic", intrinsic_loss))
    total_loss = total_loss + intrinsic_loss
    for key, value in intrinsic_metrics.items():
        _queue_tensor_metric(key, value)
    corruption_fractions = bundle.metadata.get("corruption_fractions")
    if isinstance(corruption_fractions, dict):
        for fraction_name in ("noise", "blackout", "any"):
            realized_fraction = corruption_fractions.get(fraction_name)
            if isinstance(realized_fraction, Tensor):
                _queue_tensor_metric(
                    f"corruption/{fraction_name}_fraction",
                    realized_fraction.float(),
                )
    else:
        corruption_regime = bundle.masks.get("corruption_regime")
        corruption_valid_steps = bundle.masks.get("valid_steps")
        if corruption_regime is not None and corruption_valid_steps is not None:
            corruption_valid_steps = corruption_valid_steps.bool()
            valid_count = corruption_valid_steps.sum().clamp_min(1)
            for fraction_name, selected_steps in (
                ("noise", corruption_regime.eq(1)),
                ("blackout", corruption_regime.eq(2)),
                ("any", corruption_regime.gt(0)),
            ):
                realized_fraction = (
                    selected_steps & corruption_valid_steps
                ).sum() / valid_count
                _queue_tensor_metric(
                    f"corruption/{fraction_name}_fraction",
                    realized_fraction.float(),
                )
    for module_name, module_outputs in bundle.modules.items():
        for auxiliary_name, value in module_outputs.auxiliary.items():
            if (
                auxiliary_name.startswith(
                    (
                        "kwinners.",
                        "grouped_kwinners.",
                        "chart.",
                        "chart_code.",
                        "chart_sensory.",
                        "chart_transition.",
                    )
                )
                and value.numel() == 1
            ):
                _queue_tensor_metric(f"{module_name}/{auxiliary_name}", value)
    _assert_all_finite(loss_terms)
    _queue_tensor_metric("loss/total", total_loss)
    return total_loss, metrics
