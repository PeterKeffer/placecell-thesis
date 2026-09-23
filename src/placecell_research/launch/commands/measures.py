"""Thesis measure commands for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    from placecell_research.launch import cli

    @app.command("measures")
    def measures_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
        inputs: bool = typer.Option(
            False,
            "--inputs",
            help="Also decode the visual latent and the stack of 16 latents.",
        ),
    ) -> None:
        """Thesis measures of the model in reuse.place_model_artifact_id (id or tag:<name>)."""
        from placecell_research.measures.run import measure_model

        path = measure_model(config, cli._normalize_overrides(override), include_inputs=inputs)
        typer.echo(str(path))

    @app.command("summarize")
    def summarize_command(
        tables: list[Path] = typer.Argument(..., help="CSV files written by pc measures."),
        output: Path = typer.Option(..., "--output", help="CSV of mean and SD per condition."),
    ) -> None:
        """Mean and sample SD of every measure over the runs of each condition."""
        from placecell_research.measures.table import summarize, write_rows

        rows = summarize([path for path in tables if not path.name.endswith("_units.csv")])
        write_rows(output, rows)
        typer.echo(f"{len(rows)} conditions -> {output}")

    @app.command("navigation-measures")
    def navigation_measures_command(
        runs: list[Path] = typer.Argument(..., help="Run directories of pc downstream-train."),
        output: Path = typer.Option(..., "--output", help="CSV with one row per policy."),
        epsilon: float = typer.Option(
            0.0,
            "--epsilon",
            help="Also re-run the saved starts with this chance of a random action (thesis: 0.05).",
        ),
    ) -> None:
        """Success, looping failures and learning speed of trained navigation policies."""
        from placecell_research.measures.navigation import navigation_rows
        from placecell_research.measures.table import write_rows

        rows = navigation_rows(runs, epsilon=epsilon)
        write_rows(output, rows)
        typer.echo(f"{len(rows)} policies -> {output}")
