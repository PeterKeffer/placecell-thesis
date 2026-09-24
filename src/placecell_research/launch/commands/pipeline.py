"""Core pipeline stage commands for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.utils.repo_paths import find_repo_root

from . import ConfigOption, OverrideOption, echo_stage_result


def _artifact_registry(config_path: Path, overrides: list[str]) -> ArtifactRegistry:
    from placecell_research.config import load_experiment_config

    config = load_experiment_config(config_path, overrides)
    repo_root = find_repo_root(config_path.resolve())
    return ArtifactRegistry(repo_root / config.tracking.artifact_root)


def _resolve_direct_reference(
    config_path: Path,
    overrides: list[str],
    artifact_type: str,
    artifact_reference: str | None,
) -> str:
    normalized_reference = str(artifact_reference or "").strip()
    if not normalized_reference:
        return ""
    if not ArtifactRegistry.is_tag_reference(normalized_reference):
        return normalized_reference
    registry = _artifact_registry(config_path, overrides)
    return registry.resolve_completed_reference(artifact_type, normalized_reference).artifact_id


def _resolve_dataset_reference(
    config_path: Path,
    overrides: list[str],
    dataset_reference: str | None,
    dataset_type: str | None,
) -> tuple[str, str]:
    normalized_reference = str(dataset_reference or "").strip()
    normalized_type = str(dataset_type or "").strip()
    if not normalized_reference:
        return "", normalized_type
    if not ArtifactRegistry.is_tag_reference(normalized_reference):
        return normalized_reference, normalized_type
    registry = _artifact_registry(config_path, overrides)
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


def _resolve_split_reference(
    config_path: Path,
    overrides: list[str],
    dataset_artifact_id: str,
    split_reference: str | None,
) -> str:
    normalized_reference = str(split_reference or "").strip()
    if not normalized_reference and dataset_artifact_id:
        normalized_reference = "auto"
    if normalized_reference and normalized_reference != "auto":
        return _resolve_direct_reference(config_path, overrides, "split_set", normalized_reference)
    if normalized_reference != "auto":
        return ""
    if not dataset_artifact_id:
        raise ValueError(
            "`--split auto` requires a dataset reference so the CLI can resolve a compatible split."
        )
    registry = _artifact_registry(config_path, overrides)
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


def _with_value(overrides: list[str], key: str, value: str | None) -> list[str]:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return overrides
    return [*overrides, f"{key}={normalized_value}"]


def _with_output_tags(
    overrides: list[str], artifact_type: str, tag_output: list[str] | None
) -> list[str]:
    normalized_tags = [str(tag).strip() for tag in list(tag_output or []) if str(tag).strip()]
    if not normalized_tags:
        return overrides
    serialized_tags = yaml.safe_dump(normalized_tags, default_flow_style=True).strip()
    return [*overrides, f"tracking.output_tags.{artifact_type}={serialized_tags}"]


def _with_force_recompute(overrides: list[str], force_recompute: bool) -> list[str]:
    if not force_recompute:
        return overrides
    return [*overrides, "policies.artifact_reuse=force_recompute"]


def _resolve_auto_dataset(
    config_path: Path, overrides: list[str], dataset: str | None, split: str | None
) -> tuple[list[str], str | None, str | None]:
    """`--dataset auto`: pin the finished data chain that matches this config."""
    if str(dataset or "").strip() != "auto":
        return overrides, dataset, split
    from placecell_research.stages.pipeline import find_reusable_data_overrides

    found = find_reusable_data_overrides(config_path, overrides)
    return [*overrides, *found], None, None if split in (None, "auto") else split


def _with_dataset_and_split(
    config_path: Path,
    overrides: list[str],
    dataset: str | None,
    dataset_type: str | None,
    split: str | None,
) -> list[str]:
    overrides, dataset, split = _resolve_auto_dataset(config_path, overrides, dataset, split)
    dataset_reference, resolved_dataset_type = _resolve_dataset_reference(
        config_path, overrides, dataset, dataset_type
    )
    overrides = _with_value(overrides, "dataset.artifact_id", dataset_reference)
    overrides = _with_value(overrides, "dataset.artifact_type", resolved_dataset_type)
    split_reference = _resolve_split_reference(config_path, overrides, dataset_reference, split)
    return _with_value(overrides, "splits.artifact_id", split_reference)


def pipeline_overrides(
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
    """Overrides that the pipeline command's reference and tag options stand for."""
    resolved = _with_force_recompute(list(overrides), force_recompute)
    resolved = _with_dataset_and_split(config_path, resolved, dataset, dataset_type, split)
    resolved = _with_value(resolved, "reuse.vision_encoder_artifact_id", vision_reuse)
    resolved = _with_value(resolved, "reuse.place_model_artifact_id", place_reuse)
    resolved = _with_output_tags(resolved, "vision_encoder", vision_tag_output)
    return _with_output_tags(resolved, "place_model", place_tag_output)


def train_vision_overrides(
    config_path: Path,
    overrides: list[str],
    dataset: str | None,
    reuse: str | None,
    tag_output: list[str] | None,
    force_recompute: bool,
) -> list[str]:
    """Overrides that the train-vision command's reference and tag options stand for."""
    resolved = _with_force_recompute(list(overrides), force_recompute)
    dataset_reference, _ = _resolve_dataset_reference(
        config_path, resolved, dataset, "raw_dataset"
    )
    resolved = _with_value(resolved, "dataset.artifact_id", dataset_reference)
    resolved = _with_value(resolved, "reuse.vision_encoder_artifact_id", reuse)
    return _with_output_tags(resolved, "vision_encoder", tag_output)


def train_place_overrides(
    config_path: Path,
    overrides: list[str],
    dataset: str | None,
    dataset_type: str | None,
    split: str | None,
    reuse: str | None,
    tag_output: list[str] | None,
    force_recompute: bool,
) -> list[str]:
    """Overrides that the train-model command's reference and tag options stand for."""
    resolved = _with_force_recompute(list(overrides), force_recompute)
    resolved = _with_dataset_and_split(config_path, resolved, dataset, dataset_type, split)
    resolved = _with_value(resolved, "reuse.place_model_artifact_id", reuse)
    return _with_output_tags(resolved, "place_model", tag_output)


def register(app: typer.Typer) -> None:
    @app.command("inspect-config")
    def inspect_config(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Print resolved config, signature, active objectives, parameter count, and warnings."""
        from placecell_research.launch.inspect_config import inspect_experiment_config

        inspection = inspect_experiment_config(config, override or [])
        typer.echo(yaml.safe_dump(inspection, sort_keys=False))

    @app.command("pipeline")
    def pipeline_command(
        config: ConfigOption,
        override: OverrideOption = None,
        force_recompute: bool = typer.Option(
            False,
            "--force-recompute",
            help="Force fresh artifacts instead of reusing exact-match artifacts.",
        ),
        dataset: str | None = typer.Option(
            None,
            "--dataset",
            help="Use this dataset artifact for downstream stages (tag:... or auto for the "
            "finished data chain that matches this config).",
        ),
        dataset_type: str | None = typer.Option(
            None,
            "--dataset-type",
            help="Override dataset.artifact_type when using --dataset, for example "
                 "encoded_dataset.",
        ),
        split: str | None = typer.Option(
            None,
            "--split",
            help="Use this split artifact for downstream stages. Tags are accepted as tag:..., or "
                 "use auto. Defaults to auto when --dataset is set.",
        ),
        vision_reuse: str | None = typer.Option(
            None,
            "--vision-reuse",
            help="Reuse or resume from this vision encoder artifact reference.",
        ),
        vision_tag_output: list[str] | None = typer.Option(
            None,
            "--vision-tag",
            "--vision-tag-output",
            help="Assign one or more tags to the resulting or reused vision encoder artifact.",
        ),
        place_reuse: str | None = typer.Option(
            None,
            "--place-reuse",
            help="Reuse or resume from this place-model artifact reference.",
        ),
        place_tag_output: list[str] | None = typer.Option(
            None,
            "--place-tag",
            "--place-tag-output",
            help="Assign one or more tags to the resulting or reused place model artifact.",
        ),
    ) -> None:
        """Run the configured stages in order, reusing artifacts whose inputs and settings match."""
        from placecell_research.stages import pipeline

        echo_stage_result(
            pipeline.run(
                config,
                pipeline_overrides(
                    config,
                    override or [],
                    dataset,
                    dataset_type,
                    split,
                    vision_reuse,
                    vision_tag_output,
                    place_reuse,
                    place_tag_output,
                    force_recompute,
                ),
                progress_hook=lambda stage_name, status: typer.echo(
                    f"[pipeline] {status}: {stage_name}"
                ),
            )
        )

    @app.command("collect")
    def collect_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Collect a raw dataset of random-walk episodes."""
        from placecell_research.stages import collect_dataset

        echo_stage_result(
            collect_dataset.run(config, override or [])
        )

    @app.command("create-split")
    def split_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Split a dataset into train, validation and test episodes."""
        from placecell_research.stages import create_split

        echo_stage_result(
            create_split.run(config, override or [])
        )

    @app.command("train-vision")
    def train_vision_command(
        config: ConfigOption,
        override: OverrideOption = None,
        force_recompute: bool = typer.Option(
            False,
            "--force-recompute",
            help="Force fresh artifacts instead of reusing exact-match artifacts.",
        ),
        dataset: str | None = typer.Option(
            None,
            "--dataset",
            help="Use this raw dataset artifact for vision training. Tags are accepted as tag:...",
        ),
        reuse: str | None = typer.Option(
            None,
            "--vision-reuse",
            help="Reuse or resume from this vision encoder artifact reference.",
        ),
        tag_output: list[str] | None = typer.Option(
            None,
            "--vision-tag",
            "--vision-tag-output",
            help="Assign one or more tags to the resulting or reused vision encoder artifact.",
        ),
    ) -> None:
        """Train the visual encoder."""
        from placecell_research.stages import train_vision_encoder

        echo_stage_result(
            train_vision_encoder.run(
                config,
                train_vision_overrides(
                    config,
                    override or [],
                    dataset,
                    reuse,
                    tag_output,
                    force_recompute,
                ),
            )
        )

    @app.command("encode-dataset")
    def encode_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Encode a raw dataset with a trained visual encoder."""
        from placecell_research.stages import encode_dataset

        echo_stage_result(
            encode_dataset.run(config, override or [])
        )

    @app.command("train-model")
    def train_place_command(
        config: ConfigOption,
        override: OverrideOption = None,
        force_recompute: bool = typer.Option(
            False,
            "--force-recompute",
            help="Force fresh artifacts instead of reusing exact-match artifacts.",
        ),
        dataset: str | None = typer.Option(
            None,
            "--dataset",
            help="Use this dataset artifact for place-model training (tag:... or auto for the "
            "finished data chain that matches this config).",
        ),
        dataset_type: str | None = typer.Option(
            None,
            "--dataset-type",
            help="Override dataset.artifact_type when using --dataset, for example "
                 "encoded_dataset.",
        ),
        split: str | None = typer.Option(
            None,
            "--split",
            help="Use this split artifact for place-model training. Tags are accepted as tag:..., "
                 "or use auto. Defaults to auto when --dataset is set.",
        ),
        reuse: str | None = typer.Option(
            None,
            "--place-reuse",
            help="Reuse or resume from this place-model artifact reference.",
        ),
        tag_output: list[str] | None = typer.Option(
            None,
            "--place-tag",
            "--place-tag-output",
            help="Assign one or more tags to the resulting or reused place model artifact.",
        ),
    ) -> None:
        """Train the place-cell model."""
        from placecell_research.stages import train_place_model

        echo_stage_result(
            train_place_model.run(
                config,
                train_place_overrides(
                    config,
                    override or [],
                    dataset,
                    dataset_type,
                    split,
                    reuse,
                    tag_output,
                    force_recompute,
                ),
            )
        )

    @app.command("collect-representations")
    def collect_representations_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Store the forward pass of a trained model over the configured splits."""
        from placecell_research.stages import collect_representations

        echo_stage_result(
            collect_representations.run(
                config, override or []
            )
        )

    @app.command("evaluate")
    def evaluate_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Decoding and spatial information of a trained model on each evaluation split."""
        from placecell_research.stages import evaluate_model

        echo_stage_result(
            evaluate_model.run(config, override or [])
        )

    @app.command("analyze")
    def analyze_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Run the configured analysis modules on a trained model."""
        from placecell_research.stages import analyze_model

        echo_stage_result(
            analyze_model.run(config, override or [])
        )
