#!/usr/bin/env python3
"""Evaluate frozen code organisation or recurrent reset/latent-blackout recovery."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from placecell_research.datasets.batch_iterator import iterate_dataset_batches, load_split_indices
from placecell_research.evaluation.frozen_controls import (
    code_organisation_metrics,
    fixed_topk,
    localization_curves,
    perturbed_codes,
)
from placecell_research.evaluation.inference import load_model_checkpoint
from placecell_research.evaluation.matched_decode import MatchedPositionDecoder
from placecell_research.evaluation.representation_store import (
    read_representation_manifest,
    read_representation_set,
)
from placecell_research.numerics.error_metrics import RMSE_AGGREGATION
from placecell_research.spatial_model.loading import select_place_model_checkpoint

SOURCE = "encoder.place_codes"


def organisation(args: argparse.Namespace) -> dict:
    manifests = [read_representation_manifest(path) for path in (args.dense, args.sparse)]
    recipes = []
    model_manifests = []
    for directory, manifest in zip((args.dense_model, args.sparse_model), manifests, strict=False):
        model_manifest = json.loads((directory / "manifest.json").read_text())
        if model_manifest["artifact_id"] != manifest["place_model_artifact_id"]:
            raise ValueError("Representation set and supplied model artifact differ.")
        model_manifests.append(model_manifest)
        recipe = yaml.safe_load((directory / "used_hyperparameters.yaml").read_text())
        recipes.append(recipe["selected_config_sections"])
    for key in ("seed", "dataset", "splits"):
        if recipes[0][key] != recipes[1][key]:
            raise ValueError(f"Dense/sparse training differs in {key}.")
    if model_manifests[0]["input_artifact_ids"] != model_manifests[1]["input_artifact_ids"]:
        raise ValueError("Dense/sparse training input artifacts differ.")
    dense_model, sparse_model = (recipe["spatial_model"] for recipe in recipes)
    dense_sparsifier = dict(dense_model["sparsifier"])
    sparse_sparsifier = dict(sparse_model["sparsifier"])
    dense_fraction = dense_sparsifier.pop("k_fraction")
    sparse_fraction = sparse_sparsifier.pop("k_fraction")
    if (
        dense_sparsifier != sparse_sparsifier
        or dense_fraction != 1.0
        or dense_sparsifier.get("type") != "kwinners"
    ):
        raise ValueError("The matched dense control must differ only in k_fraction=1.")
    if round(sparse_fraction * sparse_model["training"]["code_dim"]) != args.k:
        raise ValueError("The fixed mask k must match the jointly trained sparse model.")
    normalized_dense = copy.deepcopy(dense_model)
    normalized_dense["sparsifier"]["k_fraction"] = sparse_fraction
    if normalized_dense != sparse_model:
        raise ValueError("Dense/sparse training recipes differ beyond k_fraction.")
    for key in (
        "dataset_artifact_id",
        "split_artifact_id",
        "checkpoint_selection",
        "episode_ids",
        "device",
        "batch_size",
        "allow_tf32",
        "torch_version",
    ):
        if key not in manifests[0] or manifests[0][key] != manifests[1].get(key):
            raise ValueError(f"Unmatched representation manifests: {key}.")
    if manifests[0]["place_model_artifact_id"] == manifests[1]["place_model_artifact_id"]:
        raise ValueError("Dense and sparse training conditions require distinct models.")
    decoders = {}
    results = {}
    for split in ("train", "validation", "test"):
        dense, metadata = read_representation_set(
            args.dense, split_name=split, source_names=[SOURCE]
        )
        sparse, sparse_metadata = read_representation_set(
            args.sparse, split_name=split, source_names=[SOURCE]
        )
        for key in ("position_xy", "valid_steps"):
            np.testing.assert_array_equal(metadata[key], sparse_metadata[key])
        valid = metadata["valid_steps"].astype(bool)
        positions = metadata["position_xy"]
        variants = {
            "dense_trained": dense[SOURCE],
            "dense_fixed_topk": fixed_topk(dense[SOURCE], args.k),
            "joint_sparse": sparse[SOURCE],
        }
        for name, values in variants.items():
            if values.shape != dense[SOURCE].shape:
                raise ValueError("Code dimensions and sampling must match across conditions.")
            rows, targets = values[valid], positions[valid]
            if split == "train":
                decoders[name] = MatchedPositionDecoder.fit(rows, targets)
            else:
                metrics = decoders[name].score(rows, targets, select=split == "validation")
                if split == "test":
                    tuning, arrays = code_organisation_metrics(
                        values,
                        positions,
                        valid,
                        ((args.bounds[0], args.bounds[1]), (args.bounds[2], args.bounds[3])),
                    )
                    metrics.update(tuning)
                    np.savez_compressed(args.output / f"{name}_tuning.npz", **arrays)
                results[f"{split}.{name}"] = metrics
    for name in decoders:
        directory = args.output / name
        directory.mkdir()
        metrics = {
            f"{split}.{SOURCE}.{key}": value
            for split in ("validation", "test")
            for key, value in results[f"{split}.{name}"].items()
        }
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {
        "metrics": results,
        "input_manifests": manifests,
        "model_manifests": model_manifests,
        "k": args.k,
        "protocol": "signed decode; positive-part tuning; no significance classification",
        "recipe_match": "Checked training recipes, seeds and input configurations.",
    }


def perturbation(args: argparse.Namespace) -> dict:
    manifests = [
        json.loads((directory / "manifest.json").read_text())
        for directory in (args.model, args.dataset, args.split)
    ]
    if not {item["artifact_id"] for item in manifests[1:]}.issubset(
        manifests[0]["input_artifact_ids"]
    ):
        raise ValueError("Perturbation dataset and split must match the model's training inputs.")
    model, contract = load_model_checkpoint(args.model, torch.device(args.device), selection="last")
    decoder = None
    results = {}
    episode_ids = {}
    for split in ("train", "validation", "test"):
        episode_ids[split] = load_split_indices(args.split, split)[: args.max_episodes]
        clean, targets, validity = [], [], []
        interventions = {name: [] for name in ("reset", "blackout", "reset_blackout")}
        for batch in iterate_dataset_batches(
            args.dataset,
            args.split,
            split,
            args.batch_size,
            torch.device(args.device),
            observation_source="latent",
            max_episodes=args.max_episodes,
        ):
            common = dict(onset=args.onset, duration=args.duration)
            clean.append(perturbed_codes(model, batch, **common).cpu().numpy())
            targets.append(batch["position_xy"].cpu().numpy())
            validity.append(batch["valid_steps"].cpu().numpy())
            if split == "test":
                for name in interventions:
                    interventions[name].append(
                        perturbed_codes(
                            model,
                            batch,
                            reset="reset" in name,
                            blackout="blackout" in name,
                            **common,
                        )
                        .cpu()
                        .numpy()
                    )
        codes, positions, valid = map(np.concatenate, (clean, targets, validity))
        if split == "train":
            decoder = MatchedPositionDecoder.fit(codes[valid], positions[valid])
        else:
            results[split] = decoder.score(
                codes[valid], positions[valid], select=split == "validation"
            )
        if split == "test":
            curves = {}
            for name, values in {
                "clean": codes,
                **{key: np.concatenate(value) for key, value in interventions.items()},
            }.items():
                result = localization_curves(decoder, values, positions, valid)
                curves.update({f"{name}.{key}": value for key, value in result.items()})
                scores = decoder.score(values[valid], positions[valid], select=False)
                directory = args.output / name
                directory.mkdir()
                (directory / "metrics.json").write_text(
                    json.dumps(
                        {f"test.{SOURCE}.{key}": value for key, value in scores.items()}, indent=2
                    )
                    + "\n"
                )
            for name in interventions:
                curves[f"{name}.paired_excess_error"] = (
                    curves[f"{name}.episode_localization_error"]
                    - curves["clean.episode_localization_error"]
                )
            np.savez_compressed(
                args.output / "recovery_curves.npz",
                **curves,
                position_xy=positions,
                valid_steps=valid,
            )
    checkpoint = select_place_model_checkpoint(args.model, selection="last")
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    return {
        "clean_decode": results,
        "model_contract": contract,
        "checkpoint_sha256": digest,
        "input_manifests": manifests,
        "episode_ids": episode_ids,
        "onset": args.onset,
        "duration": args.duration,
        "blackout": "zero latent only; actions and kinematics retained; observations restored",
        "reset": "all carried encoder, teacher and predictor state at onset",
        "rmse": "mean of per-axis RMSE, matching MatchedPositionDecoder",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="protocol", required=True)
    org = sub.add_parser("organisation")
    org.add_argument("--dense", type=Path, required=True, help="Dense model representation_set")
    org.add_argument("--sparse", type=Path, required=True, help="Sparse model representation_set")
    org.add_argument("--dense-model", type=Path, required=True)
    org.add_argument("--sparse-model", type=Path, required=True)
    org.add_argument("--k", type=int, default=10)
    org.add_argument(
        "--bounds", type=float, nargs=4, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX")
    )
    perturb = sub.add_parser("perturbation")
    perturb.add_argument("--model", type=Path, required=True)
    perturb.add_argument("--dataset", type=Path, required=True)
    perturb.add_argument("--split", type=Path, required=True)
    perturb.add_argument("--onset", type=int, default=256)
    perturb.add_argument("--duration", type=int, default=32)
    perturb.add_argument("--max-episodes", type=int, default=64)
    perturb.add_argument("--batch-size", type=int, default=8)
    perturb.add_argument("--device", default="cpu")
    for command in (org, perturb):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.protocol == "perturbation" and args.max_episodes < 2:
        parser.error("--max-episodes must be at least two.")
    args.output.mkdir(parents=True, exist_ok=False)
    report = organisation(args) if args.protocol == "organisation" else perturbation(args)
    report["arguments"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    report["rmse_aggregation"] = RMSE_AGGREGATION
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
