"""Study (sweep / curriculum) commands for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from . import ConfigOption, OverrideOption, echo_stage_result


def _run_study(config: Path, override: list[str] | None) -> None:
    from placecell_research.stages import run_study

    echo_stage_result(run_study.run(config, override or []))


def register(app: typer.Typer) -> None:
    @app.command("sweep")
    def sweep_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Train every variant of a sweep config over its base experiment."""
        _run_study(config, override)

    @app.command("curriculum")
    def curriculum_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Collect the sources of a curriculum config and train its phases in order."""
        _run_study(config, override)
