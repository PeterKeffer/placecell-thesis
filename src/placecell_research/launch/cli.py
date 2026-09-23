"""Typer CLI for canonical stage runners."""

from __future__ import annotations

import importlib
import multiprocessing as mp
from pathlib import Path

import typer
import yaml

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.utils.repo_paths import find_repo_root

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)


def configure_process_runtime() -> None:
    mp.set_start_method("spawn", force=True)


@app.callback()
def _configure_cli_process_runtime() -> None:
    configure_process_runtime()


def _stage_module(name: str):
    return importlib.import_module(f"placecell_research.stages.{name}")


def submit_cli_entrypoint(*args, **kwargs):
    from placecell_research.launch.submit import submit_cli_entrypoint as _submit_cli_entrypoint

    return _submit_cli_entrypoint(*args, **kwargs)


def load_downstream_run_config(*args, **kwargs):
    from placecell_research.config import load_downstream_run_config as _load_downstream_run_config

    return _load_downstream_run_config(*args, **kwargs)


def validate_downstream_run_config(*args, **kwargs):
    from placecell_research.config import (
        validate_downstream_run_config as _validate_downstream_run_config,
    )

    return _validate_downstream_run_config(*args, **kwargs)


def initialize_downstream_session(**kwargs):
    from placecell_research.downstream.session import (
        initialize_downstream_session as _initialize_downstream_session,
    )

    return _initialize_downstream_session(**kwargs)


def run_downstream_rollout(**kwargs):
    from placecell_research.downstream.rollout import (
        run_downstream_rollout as _run_downstream_rollout,
    )

    return _run_downstream_rollout(**kwargs)


def train_downstream_agent(**kwargs):
    from placecell_research.downstream.train import (
        train_downstream_agent as _train_downstream_agent,
    )

    return _train_downstream_agent(**kwargs)


def managed_stage_run(**kwargs):
    from placecell_research.tracking import managed_stage_run as _managed_stage_run

    return _managed_stage_run(**kwargs)


def stage_tags(*args, **kwargs):
    from placecell_research.tracking import stage_tags as _stage_tags

    return _stage_tags(*args, **kwargs)


def _normalize_overrides(overrides: list[str] | None) -> list[str]:
    return list(overrides or [])


def _resolve_cli_direct_reference(
    config_path: Path,
    overrides: list[str],
    artifact_type: str,
    artifact_reference: str | None,
) -> str:
    from placecell_research.config import load_experiment_config

    normalized_reference = str(artifact_reference or "").strip()
    if not normalized_reference:
        return ""
    if not ArtifactRegistry.is_tag_reference(normalized_reference):
        return normalized_reference
    config = load_experiment_config(config_path, overrides)
    repo_root = find_repo_root(config_path.resolve())
    registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    return registry.resolve_completed_reference(artifact_type, normalized_reference).artifact_id


def _resolve_cli_dataset_reference(
    config_path: Path,
    overrides: list[str],
    dataset_reference: str | None,
    dataset_type: str | None,
) -> tuple[str, str]:
    from placecell_research.config import load_experiment_config

    normalized_reference = str(dataset_reference or "").strip()
    normalized_type = str(dataset_type or "").strip()
    if not normalized_reference:
        return "", normalized_type
    if not ArtifactRegistry.is_tag_reference(normalized_reference):
        return normalized_reference, normalized_type
    config = load_experiment_config(config_path, overrides)
    repo_root = find_repo_root(config_path.resolve())
    registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    if normalized_type:
        resolved_artifact = registry.resolve_completed_reference(
            normalized_type, normalized_reference
        )
        return resolved_artifact.artifact_id, normalized_type
    resolved_artifact = registry.require_completed(
        registry.resolve_tag(normalized_reference.removeprefix("tag:"))
    )
    if resolved_artifact.artifact_type not in {"raw_dataset", "encoded_dataset"}:
        raise ValueError(
            "Dataset tags must resolve to raw_dataset or encoded_dataset, got "
            f"{resolved_artifact.artifact_type}."
        )
    return resolved_artifact.artifact_id, resolved_artifact.artifact_type


def _append_override_if_value(overrides: list[str], key: str, value: str | None) -> list[str]:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return overrides
    return [*overrides, f"{key}={normalized_value}"]


def _resolve_cli_split_reference(
    config_path: Path,
    overrides: list[str],
    dataset_artifact_id: str,
    split_reference: str | None,
) -> str:
    from placecell_research.config import load_experiment_config

    normalized_reference = str(split_reference or "").strip()
    if not normalized_reference and dataset_artifact_id:
        normalized_reference = "auto"
    if normalized_reference and normalized_reference != "auto":
        return _resolve_cli_direct_reference(
            config_path, overrides, "split_set", normalized_reference
        )
    if normalized_reference != "auto":
        return ""
    if not dataset_artifact_id:
        raise ValueError(
            "`--split auto` requires a dataset reference so the CLI can resolve a compatible split."
        )
    config = load_experiment_config(config_path, overrides)
    repo_root = find_repo_root(config_path.resolve())
    registry = ArtifactRegistry(repo_root / config.tracking.artifact_root)
    matching_splits = [
        artifact
        for artifact in registry.iter_artifacts("split_set")
        if dataset_artifact_id in artifact.manifest.input_artifact_ids
    ]
    if not matching_splits:
        raise ValueError(
            f"No split_set artifact found for dataset {dataset_artifact_id}. "
            "Pass --split explicitly or run create-split first."
        )
    matching_splits.sort(key=lambda artifact: (artifact.manifest.created_at, artifact.artifact_id))
    return matching_splits[-1].artifact_id


def _append_output_tag_override(
    overrides: list[str], artifact_type: str, tag_output: list[str] | None
) -> list[str]:
    normalized_tags = [str(tag).strip() for tag in list(tag_output or []) if str(tag).strip()]
    if not normalized_tags:
        return overrides
    serialized_tags = yaml.safe_dump(normalized_tags, default_flow_style=True).strip()
    return [*overrides, f"tracking.output_tags.{artifact_type}={serialized_tags}"]


def _append_force_recompute_override(overrides: list[str], force_recompute: bool) -> list[str]:
    if not force_recompute:
        return overrides
    return [*overrides, "policies.artifact_reuse=force_recompute"]


def _append_pipeline_cli_overrides(
    config_path: Path,
    overrides: list[str],
    dataset: str | None,
    dataset_type: str | None,
    split: str | None,
    vision_reuse: str | None,
    vision_tag_output: list[str] | None,
    place_reuse: str | None,
    place_tag_output: list[str] | None,
    force_recompute: bool,
) -> list[str]:
    resolved_overrides = _append_force_recompute_override(list(overrides), force_recompute)
    dataset_reference, resolved_dataset_type = _resolve_cli_dataset_reference(
        config_path,
        resolved_overrides,
        dataset,
        dataset_type,
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides, "dataset.artifact_id", dataset_reference
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides, "dataset.artifact_type", resolved_dataset_type
    )
    split_reference = _resolve_cli_split_reference(
        config_path, resolved_overrides, dataset_reference, split
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides, "splits.artifact_id", split_reference
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides,
        "reuse.vision_encoder_artifact_id",
        vision_reuse,
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides,
        "reuse.place_model_artifact_id",
        place_reuse,
    )
    resolved_overrides = _append_output_tag_override(
        resolved_overrides, "vision_encoder", vision_tag_output
    )
    return _append_output_tag_override(resolved_overrides, "place_model", place_tag_output)


def _append_train_vision_cli_overrides(
    config_path: Path,
    overrides: list[str],
    dataset: str | None,
    reuse: str | None,
    tag_output: list[str] | None,
    force_recompute: bool,
) -> list[str]:
    resolved_overrides = _append_force_recompute_override(list(overrides), force_recompute)
    dataset_reference, _resolved_dataset_type = _resolve_cli_dataset_reference(
        config_path,
        resolved_overrides,
        dataset,
        "raw_dataset",
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides, "dataset.artifact_id", dataset_reference
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides,
        "reuse.vision_encoder_artifact_id",
        reuse,
    )
    return _append_output_tag_override(resolved_overrides, "vision_encoder", tag_output)


def _append_train_place_cli_overrides(
    config_path: Path,
    overrides: list[str],
    dataset: str | None,
    dataset_type: str | None,
    split: str | None,
    reuse: str | None,
    tag_output: list[str] | None,
    force_recompute: bool,
) -> list[str]:
    resolved_overrides = _append_force_recompute_override(list(overrides), force_recompute)
    dataset_reference, resolved_dataset_type = _resolve_cli_dataset_reference(
        config_path,
        resolved_overrides,
        dataset,
        dataset_type,
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides, "dataset.artifact_id", dataset_reference
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides, "dataset.artifact_type", resolved_dataset_type
    )
    split_reference = _resolve_cli_split_reference(
        config_path, resolved_overrides, dataset_reference, split
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides, "splits.artifact_id", split_reference
    )
    resolved_overrides = _append_override_if_value(
        resolved_overrides,
        "reuse.place_model_artifact_id",
        reuse,
    )
    return _append_output_tag_override(resolved_overrides, "place_model", tag_output)


def _echo_stage_result(result: dict[str, object] | None) -> None:
    from placecell_research.tracking import render_stage_result_summary

    if result:
        typer.echo(render_stage_result_summary(result))


def _register_command_groups() -> None:
    """Attach every command group onto the root app at import time."""
    from placecell_research.launch.commands import downstream, measures, pipeline, remote

    pipeline.register(app)
    downstream.register(app)
    measures.register(app)
    remote.register(app)


_register_command_groups()


def main() -> None:
    configure_process_runtime()
    app()


if __name__ == "__main__":
    main()
