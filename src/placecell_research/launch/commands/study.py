"""Study (sweep / curriculum) commands for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from . import echo_stage_result


def register(app: typer.Typer) -> None:
    @app.command("sweep")
    @app.command("curriculum")
    def study_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
    ) -> None:
        """Run the sweep or curriculum of a study config."""
        from placecell_research.stages import run_study

        echo_stage_result(run_study.run(config, override or []))
