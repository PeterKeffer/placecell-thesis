"""Typer command groups for the pc CLI."""

from __future__ import annotations

from typing import Any

import typer


def echo_stage_result(result: dict[str, Any] | None) -> None:
    from placecell_research.tracking import render_stage_result_summary

    if result:
        typer.echo(render_stage_result_summary(result))
