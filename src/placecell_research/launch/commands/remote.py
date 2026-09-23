"""SLURM submission and remote cluster commands for the pc CLI."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

REMOTE_OPTION_HELP = "Defaults come from the user file (pc doctor prints its path)."


def register(app: typer.Typer) -> None:
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
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Write the batch script and print its path; do not submit."
        ),
    ) -> None:
        """Write a hardened SLURM script for one command and submit it with sbatch."""
        from placecell_research.launch.submit import submit_cli_entrypoint

        typer.echo(
            submit_cli_entrypoint(
                config,
                override or [],
                entrypoint=entrypoint,
                force_recompute=force_recompute,
                slurm_dependency=slurm_dependency,
                slurm_job_name=slurm_job_name,
                dry_run=dry_run,
            )
        )

    @app.command("hpc")
    def hpc_command(
        config: Path = typer.Option(..., "--config", "-c"),
        entrypoint: str = typer.Option("auto", "--entrypoint", "--command"),
        remote_host: str | None = typer.Option(None, "--remote-host", help=REMOTE_OPTION_HELP),
        remote_repo_root: str | None = typer.Option(None, "--remote-repo-root"),
        remote_setup: str | None = typer.Option(None, "--remote-setup"),
        remote_python: str | None = typer.Option(None, "--remote-python"),
        ssh_option: list[str] | None = typer.Option(None, "--ssh-option"),
        sync_repo: bool = typer.Option(
            True, "--sync/--no-sync", help="Rsync the local repo to the remote checkout first."
        ),
        snapshot_code: bool = typer.Option(
            True,
            "--snapshot/--no-snapshot",
            help="Freeze src/, configs/ and scripts/slurm/ for this job so a later sync cannot "
            "change its code.",
        ),
        slurm_dependency: str | None = typer.Option(None, "--slurm-dependency"),
        stream_logs: bool = typer.Option(
            True,
            "--stream/--no-stream",
            help="Tail the remote log until the job ends. --no-stream prints the job id only.",
        ),
        cancel_on_interrupt: bool = typer.Option(
            True,
            "--cancel-on-interrupt/--detach-on-interrupt",
            help="Cancel the remote job when you stop streaming with Ctrl+C.",
        ),
        override: list[str] | None = typer.Option(None, "--override", "-o"),
        force_recompute: bool = typer.Option(False, "--force-recompute"),
    ) -> None:
        """Sync the repo to the cluster over SSH, submit one job there, and stream its log."""
        from placecell_research.launch.remote_run import (
            resolve_remote_settings,
            run_remote_slurm_job,
        )

        settings = resolve_remote_settings(
            remote_host=remote_host,
            remote_repo_root=remote_repo_root,
            remote_setup=remote_setup,
            remote_python=remote_python,
            ssh_options=ssh_option or [],
        )
        result = run_remote_slurm_job(
            config,
            override or [],
            settings,
            entrypoint=entrypoint,
            sync_repo=sync_repo,
            force_recompute=force_recompute,
            snapshot_code=snapshot_code,
            slurm_dependency=slurm_dependency,
            stream_logs=stream_logs,
            cancel_on_interrupt=cancel_on_interrupt,
        )
        summary = yaml.safe_dump(
            {
                "remote_host": result.remote_host,
                "job_id": result.job_id,
                "final_state": result.final_state,
                "slurm_log_path": result.slurm_log_path,
                "code_snapshot_path": result.code_snapshot_path,
            },
            sort_keys=False,
        )
        if stream_logs:
            typer.echo(summary)
            return
        typer.echo(summary, err=True)
        typer.echo(result.job_id)

    @app.command("hpc-logs")
    def hpc_logs_command(
        job_id: str | None = typer.Option(None, "--job-id", help="Job to attach to."),
        latest: bool = typer.Option(False, "--latest", help="Attach to the newest active job."),
        run_root: Path = typer.Option(
            Path("runs"), "--run-root", help="Remote run root relative to the checkout."
        ),
        remote_host: str | None = typer.Option(None, "--remote-host", help=REMOTE_OPTION_HELP),
        remote_repo_root: str | None = typer.Option(None, "--remote-repo-root"),
        ssh_option: list[str] | None = typer.Option(None, "--ssh-option"),
        cancel_on_interrupt: bool = typer.Option(
            False, "--cancel-on-interrupt/--detach-on-interrupt"
        ),
    ) -> None:
        """Stream the log of a running remote job without submitting anything."""
        from placecell_research.launch.remote_run import (
            attach_to_remote_slurm_job,
            list_remote_active_slurm_jobs,
            resolve_remote_settings,
        )

        settings = resolve_remote_settings(
            remote_host=remote_host,
            remote_repo_root=remote_repo_root,
            ssh_options=ssh_option or [],
        )
        selected = job_id
        if selected is None:
            active = list_remote_active_slurm_jobs(settings)
            if not active:
                raise typer.BadParameter("No active placecell_research job; pass --job-id.")
            if len(active) > 1 and not latest:
                listed = ", ".join(f"{job.job_id} ({job.state})" for job in active)
                raise typer.BadParameter(
                    f"Several active jobs: {listed}. Pass --job-id or --latest."
                )
            selected = active[0].job_id
        result = attach_to_remote_slurm_job(
            settings, job_id=selected, run_root=run_root, cancel_on_interrupt=cancel_on_interrupt
        )
        typer.echo(yaml.safe_dump({"job_id": result.job_id, "final_state": result.final_state}))

    @app.command("remote-sync")
    def remote_sync_command(
        remote_host: str | None = typer.Option(None, "--remote-host", help=REMOTE_OPTION_HELP),
        remote_repo_root: str | None = typer.Option(None, "--remote-repo-root"),
        ssh_option: list[str] | None = typer.Option(None, "--ssh-option"),
        exclude: list[str] | None = typer.Option(None, "--exclude"),
        delete: bool = typer.Option(
            False, "--delete", help="Delete remote files that no longer exist locally."
        ),
    ) -> None:
        """Rsync this checkout to the remote checkout (outputs and caches excluded)."""
        from placecell_research.launch.remote_run import (
            resolve_remote_settings,
            sync_repo_to_remote,
        )
        from placecell_research.utils.repo_paths import find_repo_root_from_path

        settings = resolve_remote_settings(
            remote_host=remote_host,
            remote_repo_root=remote_repo_root,
            ssh_options=ssh_option or [],
        )
        output = sync_repo_to_remote(
            local_repo_root=find_repo_root_from_path(Path.cwd()),
            settings=settings,
            delete=delete,
            extra_excludes=exclude or [],
            progress_callback=typer.echo,
        )
        if output:
            typer.echo(output)
