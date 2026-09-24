"""Typer command groups for the pc CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from placecell_research.launch.user_settings import RemoteSettings

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="YAML config to run (see configs/).")
]
OverrideOption = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="Config override key=value (repeatable)."),
]


def remote_settings_or_exit(flag_hint: str = "", **values: Any) -> RemoteSettings:
    """Resolve the remote settings, or print one line that names what is missing and exit."""
    from placecell_research.launch.remote_run import (
        MissingRemoteSettingsError,
        resolve_remote_settings,
    )

    try:
        return resolve_remote_settings(**values)
    except MissingRemoteSettingsError as error:
        typer.echo(f"Error: {error}{flag_hint}.", err=True)
        raise typer.Exit(1) from None


def echo_stage_result(result: dict[str, Any] | None) -> None:
    from placecell_research.tracking import render_stage_result_summary

    if result:
        typer.echo(render_stage_result_summary(result))
