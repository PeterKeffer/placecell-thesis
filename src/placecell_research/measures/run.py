"""Compute every thesis measure of one place model and write it as one CSV row."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config import load_experiment_config
from placecell_research.evaluation.frozen_controls import fixed_topk
from placecell_research.evaluation.inference import collect_representations, load_model_checkpoint
from placecell_research.evaluation.representation_store import read_representation_manifest
from placecell_research.utils.repo_paths import find_repo_root

from .data import (
    PLACE_CODE_SOURCE,
    SPLITS,
    analysis_report_metrics,
    find_analysis_report,
    find_representation_set,
    load_split,
    representation_sources,
)
from .decoding import decode, stack_features, supported_rows
from .similarity import similarity_half_distances
from .single_unit import single_unit_measures
from .table import full_analysis_row, single_unit_row, write_rows
from .traversal import traversal_measures, traversal_summary

INPUT_FEATURES = {"visual_latent": 1, "latent_stack_16": 16}


def _decode_split_features(directory: Path, source: str | None, frames: int, top_k: int):
    features, targets, row_counts = {}, {}, []
    for split in SPLITS:
        arrays = load_split(directory, split, [source] if source else [], read_time_top_k=top_k)
        values = arrays.sources[source] if source else arrays.latent
        features[split], targets[split], counts = supported_rows(
            stack_features(values, frames), arrays.position_xy, arrays.heading, arrays.valid_steps
        )
        if split == "test":
            row_counts = counts
    return features, targets, row_counts


def decoding_row(directory: Path, top_k: int, *, include_inputs: bool) -> dict[str, float]:
    row = {}
    feature_sets = [(source, source, 1) for source in representation_sources(directory)]
    if include_inputs:
        feature_sets += [(name, None, frames) for name, frames in INPUT_FEATURES.items()]
    for label, source, frames in feature_sets:
        print(f"[measures] decoding {label}", file=sys.stderr, flush=True)
        features, targets, row_counts = _decode_split_features(directory, source, frames, top_k)
        scores = decode(features, targets, row_counts)
        row |= {f"decode_{label}_{key}": value for key, value in scores.items()}
    return row


def full_test_codes(registry: ArtifactRegistry, directory: Path, checkpoint: str, top_k: int):
    """encoder.place_codes of every test episode, from a CPU forward pass."""
    manifest = read_representation_manifest(directory)
    model_artifact = registry.load("place_model", manifest["place_model_artifact_id"])
    dataset = registry.load(manifest["dataset_artifact_type"], manifest["dataset_artifact_id"])
    split = registry.load("split_set", manifest["split_artifact_id"])
    device = torch.device("cpu")
    model, _ = load_model_checkpoint(model_artifact.path, device, selection=checkpoint)
    representations, metadata = collect_representations(
        model, dataset.path, split.path, "test", [PLACE_CODE_SOURCE], device, batch_size=8
    )
    codes = representations[PLACE_CODE_SOURCE]
    if top_k:
        codes = fixed_topk(codes, top_k)
    return codes, metadata


def measure_model(config_path: Path, overrides: list[str], *, include_inputs: bool = False) -> Path:
    """Write <output_dir>/<condition>__seed<seed>__<model>.csv and a per-unit table beside it."""
    config = load_experiment_config(config_path, overrides)
    settings = config.measures
    repo_root = find_repo_root(config_path.resolve())
    registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    reference = config.reuse.place_model_artifact_id.strip()
    if not reference:
        raise ValueError("Set reuse.place_model_artifact_id to the model to measure (id or tag:).")
    model = registry.resolve_completed_reference("place_model", reference)
    recipe = yaml.safe_load((model.path / "used_hyperparameters.yaml").read_text())
    training_seed = int(recipe["selected_config_sections"]["seed"]["training_seed"])
    trained_as = yaml.safe_load((model.path / "resolved_config.yaml").read_text())["name"]
    if trained_as != config.name and not settings.read_time_top_k:
        raise ValueError(
            f"{model.artifact_id} was trained with config '{trained_as}', not '{config.name}'. "
            "Pass the model of this condition in reuse.place_model_artifact_id."
        )
    checkpoint = config.policies.checkpoint_selection
    representation_set = find_representation_set(
        registry, model.artifact_id, checkpoint, config.reuse.representation_set_artifact_id
    )
    report = find_analysis_report(registry, model.artifact_id)
    env_id = config.environment.env_id
    bounds = overlay_bounds(resolve_world_overlay(env_id))
    top_k = settings.read_time_top_k
    directory = representation_set.path

    test = load_split(directory, "test", [PLACE_CODE_SOURCE], read_time_top_k=top_k)
    codes = test.sources[PLACE_CODE_SOURCE]
    print(f"[measures] single units on {codes.shape}", file=sys.stderr, flush=True)
    per_unit, population = single_unit_measures(
        codes,
        test.position_xy,
        test.heading,
        test.valid_steps,
        bounds,
        env_id=env_id,
        null_shuffles=settings.null_shuffles,
    )
    unit_row, unit_arrays = single_unit_row(per_unit, population["visited_bin_count"])
    row = {
        "condition": config.name,
        "training_seed": training_seed,
        "model": model.artifact_id,
        "model_trained_as": trained_as,
        "representation_set": representation_set.artifact_id,
        "analysis_report": report.artifact_id,
        "checkpoint": checkpoint,
        "test_episodes": len(test.episode_ids),
        **{f"measures_{key}": value for key, value in asdict(settings).items()},
        **population,
        **unit_row,
    }
    row |= decoding_row(directory, top_k, include_inputs=include_inputs)
    similarity_inputs = {"place_code": codes}
    if test.latent is not None:
        similarity_inputs["visual_latent"] = test.latent
    print("[measures] similarity", file=sys.stderr, flush=True)
    row |= similarity_half_distances(
        similarity_inputs, test.position_xy, test.heading, test.valid_steps
    )
    print("[measures] traversals over all test episodes", file=sys.stderr, flush=True)
    full_codes, metadata = full_test_codes(registry, directory, checkpoint, top_k)
    traversal = traversal_measures(
        full_codes,
        metadata["position_xy"],
        metadata["heading"],
        metadata["valid_steps"].astype(bool),
        env_id=env_id,
        shifts=settings.traversal_shifts,
    )
    row["traversal_test_episodes"] = int(full_codes.shape[0])
    row |= traversal_summary(traversal)
    full_row, missing = full_analysis_row(analysis_report_metrics(report))
    if missing:
        print(f"[measures] not in {report.artifact_id}: {missing}", file=sys.stderr, flush=True)
    row |= full_row

    output_dir = repo_root / settings.output_dir
    stem = f"{config.name}__seed{training_seed}__{model.artifact_id}"
    write_rows(output_dir / f"{stem}.csv", [row])
    unit_columns = {key: value for key, value in unit_arrays.items() if np.ndim(value) == 1}
    unit_columns |= {
        f"traversal_{key}": value for key, value in traversal.items() if np.ndim(value) == 1
    }
    write_rows(
        output_dir / f"{stem}_units.csv",
        [
            {"unit": unit, **{key: float(value[unit]) for key, value in unit_columns.items()}}
            for unit in range(codes.shape[-1])
        ],
    )
    return output_dir / f"{stem}.csv"
