"""Study (sweep / curriculum) commands for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    from placecell_research.launch import cli

    @app.command("sweep")
    @app.command("curriculum")
    def study_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        cli._echo_stage_result(
            cli._stage_module("run_study").run(config, cli._normalize_overrides(override))
        )
