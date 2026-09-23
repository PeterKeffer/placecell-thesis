"""Typer CLI for canonical stage runners."""

from __future__ import annotations

import multiprocessing as mp

import typer

from placecell_research.launch.commands import (
    doctor,
    downstream,
    measures,
    pipeline,
    remote,
    reproduce,
    study,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)


@app.callback()
def configure_process_runtime() -> None:
    mp.set_start_method("spawn", force=True)


for command_group in (pipeline, downstream, measures, remote, study, reproduce, doctor):
    command_group.register(app)


def main() -> None:
    configure_process_runtime()
    app()


if __name__ == "__main__":
    main()
