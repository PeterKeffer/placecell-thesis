"""SLURM submission command for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    from placecell_research.launch import cli

    @app.command("submit")
    def submit_command(
        config: Path = typer.Option(..., "--config", "-c"),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
        entrypoint: str = typer.Option(
            "auto",
            "--entrypoint",
            "--command",
            help=(
                "CLI command to submit on SLURM, for example pipeline, analyze, "
                "or downstream-train."
            ),
        ),
        force_recompute: bool = typer.Option(
            False,
            "--force-recompute",
            help="Force fresh artifacts instead of reusing exact-match artifacts.",
        ),
        slurm_dependency: str | None = typer.Option(
            None,
            "--slurm-dependency",
            help="Pass a SLURM dependency string to sbatch, for example afterany:12345.",
        ),
        slurm_job_name: str = typer.Option(
            "placecell_research",
            "--slurm-job-name",
            help="Operational SLURM job name. This does not alter the experiment config.",
        ),
    ) -> None:
        typer.echo(
            cli.submit_cli_entrypoint(
                config,
                cli._normalize_overrides(override),
                entrypoint=entrypoint,
                force_recompute=force_recompute,
                slurm_dependency=slurm_dependency,
                slurm_job_name=slurm_job_name,
            )
        )
