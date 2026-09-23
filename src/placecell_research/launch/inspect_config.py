"""Resolved-config inspection without launching a stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config import (
    load_experiment_config,
    summarize_reuse,
    validate_experiment_config,
)
from placecell_research.datasets.dataset import build_training_dataloaders
from placecell_research.objectives import build_objectives
from placecell_research.spatial_model.builder import ModelBuildContext, build_place_model
from placecell_research.spatial_model.contract import build_model_contract
from placecell_research.tracking.naming import generate_signature
from placecell_research.utils.repo_paths import find_repo_root


def inspect_experiment_config(
    config_path: Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Return a compact inspection payload for a resolved experiment config."""
    config = load_experiment_config(config_path, overrides or [])
    warnings = validate_experiment_config(config)
    config_dict = config.to_dict()
    repo_root = find_repo_root(config_path)
    artifact_registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    payload: dict[str, Any] = {
        "resolved_config": config_dict,
        "signature": generate_signature(config_dict),
        "active_objectives": sorted(config.spatial_model.objectives.keys()),
        "reuse": summarize_reuse(config, artifact_registry=artifact_registry),
        "policies": config_dict["policies"],
        "warnings": warnings,
    }
    try:
        if payload["reuse"]["place_model"]["stage_behavior"] == "reuse_existing_artifact":
            place_model_artifact = artifact_registry.load(
                "place_model",
                payload["reuse"]["place_model"]["artifact_id"],
            )
            contract = place_model_artifact.path / "model_contract.json"
            if contract.exists():
                payload["model_contract"] = json.loads(contract.read_text())
                payload["parameter_count"] = payload["model_contract"]["total_parameters"]
                payload["available_representations"] = payload["model_contract"][
                    "available_representations"
                ]
                return payload
        train_loader, _validation_loader, dataset_metadata = build_training_dataloaders(
            config,
            artifact_root=repo_root / config.tracking.artifact_root,
        )
        training_window = int(config.spatial_model.training.bptt_window)
        sequence_length = int(dataset_metadata["episode_length"])
        chunks_per_batch = (
            1
            if training_window == 0
            else max(1, (sequence_length + training_window - 1) // training_window)
        )
        optimizer_steps_per_epoch = max(len(train_loader), 1) * chunks_per_batch
        total_optimizer_steps = max(
            1,
            config.spatial_model.training.epochs * optimizer_steps_per_epoch,
        )
        build_context = ModelBuildContext(
            num_actions=int(dataset_metadata["num_actions"]),
            observation_dim=int(dataset_metadata["observation_dim"]),
            kinematics_dim=int(dataset_metadata["kinematics_dim"]),
            total_optimizer_steps=total_optimizer_steps,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        )
        model = build_place_model(
            config=config.spatial_model,
            build_context=build_context,
        )
        built = build_objectives(model, config.spatial_model)
        if hasattr(model, "set_auxiliary_heads"):
            model.set_auxiliary_heads(built.auxiliary_heads)
        contract = build_model_contract(model, sorted(config.spatial_model.objectives.keys()))
        payload["model_contract"] = contract
        payload["parameter_count"] = contract["total_parameters"]
        payload["available_representations"] = contract["available_representations"]
    except Exception as exc:
        payload["model_build_warning"] = str(exc)
    return payload
