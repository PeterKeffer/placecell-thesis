"""The pc reproduce command: every thesis run as one ordered plan, locally or as a SLURM chain."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer
import yaml

from placecell_research.launch.commands import remote_settings_or_exit


def _split(text: str | None) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def register(app: typer.Typer) -> None:
    @app.command("reproduce")
    def reproduce_command(
        profile: str = typer.Option(
            "local",
            "--profile",
            help="local (this machine, in order), slurm (any SLURM cluster), hpc3 (the lab "
            "cluster), or any other configs/launcher/<name>.yaml.",
        ),
        only: str | None = typer.Option(
            None,
            "--only",
            help="Comma-separated conditions (file names in configs/thesis) or navigation "
            "configs, or 'navigation'. Upstream runs they need are added.",
        ),
        seeds: str | None = typer.Option(
            None, "--seeds", help="Comma-separated training seeds to keep, e.g. 42 or 42,1,2."
        ),
        smoke: bool = typer.Option(
            False, "--smoke", help="Shrink every run to minutes; outputs go to smoke/."
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Print the plan (and batch scripts on SLURM); run nothing."
        ),
        navigation: bool = typer.Option(
            True,
            "--navigation/--no-navigation",
            help="Include the navigation agents whenever the baseline model is in the plan.",
        ),
        override: list[str] | None = typer.Option(
            None, "--override", "-o", help="launcher.* overrides for every job."
        ),
        remote: bool = typer.Option(
            False,
            "--remote",
            help="Sync this checkout to the cluster (remote.* in the user file) and run the "
            "same command on its login node.",
        ),
        sync_repo: bool = typer.Option(True, "--sync/--no-sync", help="With --remote."),
    ) -> None:
        """Reproduce the thesis: data, encoders, every condition and seed, measures, navigation."""
        from placecell_research.reproduce.execute import run_locally, submit_to_slurm
        from placecell_research.reproduce.plan import (
            build_plan,
            load_reproduction_settings,
            output_roots,
        )
        from placecell_research.utils.repo_paths import configs_root, find_repo_root

        launcher_overrides = list(override or [])
        invalid = [item for item in launcher_overrides if not item.startswith("launcher.")]
        if invalid:
            raise typer.BadParameter(f"Only launcher.* overrides are accepted here: {invalid}")
        config_root = configs_root()
        profile_path = config_root / "launcher" / f"{profile}.yaml"
        if not profile_path.is_file():
            available = sorted(path.stem for path in (config_root / "launcher").glob("*.yaml"))
            raise typer.BadParameter(f"No profile {profile!r}; available: {available}")
        profile_type = (yaml.safe_load(profile_path.read_text()) or {}).get("type", "local")
        if remote:
            if profile_type != "slurm":
                raise typer.BadParameter("--remote needs a SLURM profile such as hpc3 or slurm.")
            arguments = ["reproduce", "--profile", profile]
            arguments += ["--only", only] if only else []
            arguments += ["--seeds", seeds] if seeds else []
            arguments += ["--smoke"] if smoke else []
            arguments += ["--dry-run"] if dry_run else []
            arguments += [] if navigation else ["--no-navigation"]
            for item in launcher_overrides:
                arguments += ["-o", item]
            raise typer.Exit(_run_remotely(config_root, arguments, sync_repo))
        steps = build_plan(
            config_root,
            only=_split(only) or None,
            seeds=[int(seed) for seed in _split(seeds)] or None,
            smoke=smoke,
            include_navigation=navigation,
        )
        repo_root = find_repo_root(config_root / "reproduce.yaml")
        roots = output_roots(smoke, load_reproduction_settings(config_root))
        state_dir = repo_root / roots.run_root / "reproduce"
        counts = Counter(step.kind for step in steps)
        typer.echo(
            f"[reproduce] profile={profile} ({profile_type}) smoke={smoke} "
            f"{len(steps)} jobs: " + ", ".join(f"{kind} {count}" for kind, count in counts.items())
        )
        echo = typer.echo
        if profile_type == "slurm":
            status = submit_to_slurm(
                steps,
                profile=profile,
                repo_root=repo_root,
                state_dir=state_dir,
                launcher_overrides=launcher_overrides,
                dry_run=dry_run,
                echo=echo,
            )
        else:
            status = run_locally(
                steps, repo_root=repo_root, state_dir=state_dir, dry_run=dry_run, echo=echo
            )
        raise typer.Exit(status)


def _run_remotely(config_root: Path, arguments: list[str], sync_repo: bool) -> int:
    from placecell_research.launch.remote_run import run_remote_cli

    return run_remote_cli(
        config_root / "thesis" / "baseline.yaml",
        arguments,
        remote_settings_or_exit(),
        sync_repo=sync_repo,
    )
