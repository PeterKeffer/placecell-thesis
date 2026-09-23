"""Checkpoint and artifact loading for trained place models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch import nn

from placecell_research.config import materialize_dataclass
from placecell_research.config.schema import SpatialModelConfig
from placecell_research.objectives.registry import build_objectives

from .builder import ModelBuildContext, build_place_model

CheckpointSelection = Literal["auto", "best", "last"]


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _artifact_declared_selection(model_directory: Path) -> str:
    """The checkpoint policy the artifact's OWN run recorded, or 'best' when it says nothing."""
    config_path = model_directory / "resolved_config.yaml"
    if not config_path.exists():
        return "best"
    try:
        payload = yaml.safe_load(config_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return "best"
    if not isinstance(payload, dict):
        return "best"
    policies = payload.get("policies")
    if not isinstance(policies, dict):
        return "best"
    declared = policies.get("checkpoint_selection") or "best"
    return "last" if declared == "last" else "best"


def select_place_model_checkpoint(
    model_directory: Path,
    *,
    selection: CheckpointSelection = "last",
) -> Path:
    """Resolve one artifact checkpoint according to the configured selection policy."""
    if selection not in {"auto", "best", "last"}:
        raise ValueError(
            "checkpoint selection must be 'auto', 'best', or 'last', "
            f"got {selection!r}."
        )
    if selection == "auto":
        selection = _artifact_declared_selection(model_directory)
    names = (
        ["weights_last.pt", "weights_best_validation_loss.pt", "weights_best_primary.pt"]
        if selection == "last"
        else ["weights_best_primary.pt", "weights_best_validation_loss.pt", "weights_last.pt"]
    )
    checkpoint_candidates = [model_directory / name for name in names]
    checkpoint_path = next(
        (candidate for candidate in checkpoint_candidates if candidate.exists()),
        None,
    )
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_directory}.")
    return checkpoint_path


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.ModuleDict, dict[str, Any]]:
    """Rebuild a trained model and its auxiliary heads from a checkpoint payload."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    build_context = payload.get("build_context")
    if not isinstance(build_context, dict):
        raise RuntimeError(
            "Checkpoint is missing build_context and cannot be reconstructed into a place model."
    )
    spatial_model_config = materialize_dataclass(
        SpatialModelConfig, build_context["spatial_model_config"]
    )
    total_optimizer_steps = int(build_context.get("total_optimizer_steps", 1))
    training_epochs = max(int(spatial_model_config.training.epochs), 1)
    optimizer_steps_per_epoch = int(
        build_context.get(
            "optimizer_steps_per_epoch",
            max(1, (total_optimizer_steps + training_epochs - 1) // training_epochs),
        )
    )
    model_build_context = ModelBuildContext(
        num_actions=int(build_context["num_actions"]),
        observation_dim=int(build_context["observation_dim"]),
        kinematics_dim=int(build_context["kinematics_dim"]),
        total_optimizer_steps=total_optimizer_steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )
    model = build_place_model(
        config=spatial_model_config,
        build_context=model_build_context,
    )
    built = build_objectives(model, spatial_model_config)
    if hasattr(model, "set_auxiliary_heads"):
        model.set_auxiliary_heads(built.auxiliary_heads)
    model.load_state_dict(payload["model_state_dict"])
    built.auxiliary_heads.load_state_dict(payload["auxiliary_state_dict"])
    _freeze(model.to(device))
    _freeze(built.auxiliary_heads.to(device))
    return model, built.auxiliary_heads, payload


def load_place_model_artifact(
    model_directory: Path,
    device: torch.device,
    *,
    selection: str = "last",
) -> tuple[nn.Module, nn.ModuleDict, dict[str, Any], dict[str, Any]]:
    """Load the selected checkpoint and its JSON contract from a place-model artifact."""
    checkpoint_path = select_place_model_checkpoint(model_directory, selection=selection)
    model, auxiliary_heads, payload = load_model_from_checkpoint(checkpoint_path, device)
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    print(f"[load_place_model] {checkpoint_path.name} epoch={epoch} from {model_directory.name}")
    contract_path = model_directory / "model_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    return model, auxiliary_heads, contract, payload
