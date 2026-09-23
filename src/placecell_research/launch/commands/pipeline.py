"""Core pipeline stage commands for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml


def register(app: typer.Typer) -> None:
    from placecell_research.launch import cli

    @app.command("inspect-config")
    def inspect_config(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        """Print resolved config, signature, active objectives, parameter count, and warnings."""
        from placecell_research.launch.inspect_config import inspect_experiment_config

        inspection = inspect_experiment_config(config, cli._normalize_overrides(override))
        typer.echo(yaml.safe_dump(inspection, sort_keys=False))

    @app.command("pipeline")
    def pipeline_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
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
        pipeline_module = cli._stage_module("pipeline")
        cli._echo_stage_result(
            pipeline_module.run(
                config,
                cli._append_pipeline_cli_overrides(
                    config,
                    cli._normalize_overrides(override),
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
    @app.command("collect-dataset", hidden=True)
    def collect_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        cli._echo_stage_result(
            cli._stage_module("collect_dataset").run(config, cli._normalize_overrides(override))
        )

    @app.command("create-split")
    def split_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        cli._echo_stage_result(
            cli._stage_module("create_split").run(config, cli._normalize_overrides(override))
        )

    @app.command("train-vision")
    @app.command("train-vision-encoder", hidden=True)
    def train_vision_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
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
        cli._echo_stage_result(
            cli._stage_module("train_vision_encoder").run(
                config,
                cli._append_train_vision_cli_overrides(
                    config,
                    cli._normalize_overrides(override),
                    dataset,
                    reuse,
                    tag_output,
                    force_recompute,
                ),
            )
        )

    @app.command("encode-dataset")
    def encode_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        cli._echo_stage_result(
            cli._stage_module("encode_dataset").run(config, cli._normalize_overrides(override))
        )

    @app.command("train-model")
    @app.command("train-place-model", hidden=True)
    def train_place_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
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
        cli._echo_stage_result(
            cli._stage_module("train_place_model").run(
                config,
                cli._append_train_place_cli_overrides(
                    config,
                    cli._normalize_overrides(override),
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
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        cli._echo_stage_result(
            cli._stage_module("collect_representations").run(
                config, cli._normalize_overrides(override)
            )
        )

    @app.command("evaluate")
    @app.command("evaluate-model", hidden=True)
    def evaluate_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        cli._echo_stage_result(
            cli._stage_module("evaluate_model").run(config, cli._normalize_overrides(override))
        )

    @app.command("analyze")
    @app.command("analyze-model", hidden=True)
    def analyze_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        cli._echo_stage_result(
            cli._stage_module("analyze_model").run(config, cli._normalize_overrides(override))
        )
