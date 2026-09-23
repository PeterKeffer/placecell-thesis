"""Run the reproduction plan in order on this machine, or submit it as a SLURM dependency chain."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from placecell_research.config import load_downstream_run_config, load_experiment_config
from placecell_research.launch.remote_entrypoints import default_launcher_overrides
from placecell_research.launch.submit import (
    parse_sbatch_job_id,
    render_slurm_script,
    run_sbatch,
    write_slurm_script,
)

from .plan import Step

ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
DRY_RUN_FIRST_JOB_ID = 9000001
JOB_KIND_KEYS = {
    "partition": "partition",
    "exclude_nodes": "exclude_nodes",
    "gpus": "gpus",
    "gpu_type": "gpu_type",
    "cpus_per_task": "cpus_per_task",
    "memory_gb": "memory_gb",
    "time_hours": "time_hours",
    "omp_num_threads": "threading_safety.omp_num_threads",
}


def cli_command(arguments: tuple[str, ...], python: str) -> str:
    return shlex.join([python, "-u", "-m", "placecell_research.launch.cli", *arguments])


def marker_path(state_dir: Path, step: Step) -> Path:
    return state_dir / "done" / step.name


def _is_done(state_dir: Path, step: Step) -> bool:
    return step.resumable and marker_path(state_dir, step).is_file()


def run_locally(
    steps: list[Step],
    *,
    repo_root: Path,
    state_dir: Path,
    dry_run: bool,
    echo: Callable[[str], None],
) -> int:
    """Run every unfinished step in order; stop at the first failure (rerun to resume)."""
    for index, step in enumerate(steps, start=1):
        label = f"[reproduce {index}/{len(steps)}] {step.name}"
        if _is_done(state_dir, step):
            echo(f"{label}: done, skipped")
            continue
        commands = [cli_command(arguments, sys.executable) for arguments in step.commands()]
        if dry_run:
            echo(f"{label} ({step.kind})")
            for command in commands:
                echo(f"  {command}")
            continue
        echo(f"{label}: running")
        for command in commands:
            completed = subprocess.run(shlex.split(command), cwd=repo_root, check=False)
            if completed.returncode != 0:
                echo(
                    f"{label}: failed with exit code {completed.returncode}. Fix the cause and "
                    "rerun the same pc reproduce command; finished steps are skipped."
                )
                return int(completed.returncode)
        if step.resumable:
            marker = marker_path(state_dir, step)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
    return 0


@dataclass(slots=True)
class _SubmittedJob:
    job_id: str
    lane: int
    gpu: bool


def _load_config(step: Step, overrides: list[str]):
    loader = load_downstream_run_config if step.downstream else load_experiment_config
    return loader(step.config, overrides)


def _step_overrides(step: Step) -> list[str]:
    arguments = list(step.arguments)
    return [arguments[index + 1] for index, value in enumerate(arguments) if value == "-o"]


def resolve_launcher(step: Step, profile: str, launcher_overrides: list[str]):
    """Launcher of one job: profile, then user file, then CLI, then this job kind's resources."""
    base = default_launcher_overrides(
        [f"launcher={profile}", *_step_overrides(step), *launcher_overrides], profile
    )
    launcher = _load_config(step, base).launcher
    resources = launcher.job_resources.get(step.kind)
    if resources is None:
        return launcher
    kind_overrides = [
        f"launcher.{target}={json.dumps(value) if isinstance(value, list) else value}"
        for key, target in JOB_KIND_KEYS.items()
        if (value := getattr(resources, key)) is not None
    ]
    return _load_config(step, [*base, *kind_overrides]).launcher


def _active_job_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}
    states = {}
    for line in completed.stdout.splitlines():
        job_id, _, state = line.partition("|")
        if state.strip() in ACTIVE_STATES:
            states[job_id.strip()] = state.strip()
    return states


def _dependency(ok_ids: list[str], any_ids: list[str]) -> str | None:
    parts = []
    if ok_ids:
        parts.append("afterok:" + ":".join(ok_ids))
    if any_ids:
        parts.append("afterany:" + ":".join(any_ids))
    return ",".join(parts) or None


def submit_to_slurm(
    steps: list[Step],
    *,
    profile: str,
    repo_root: Path,
    state_dir: Path,
    launcher_overrides: list[str],
    dry_run: bool,
    echo: Callable[[str], None],
) -> int:
    """Submit unfinished steps with afterok dependencies and a lane cap on concurrent jobs."""
    state_file = state_dir / "slurm_jobs.json"
    previous = json.loads(state_file.read_text()) if state_file.is_file() else {}
    active = _active_job_states([entry["job_id"] for entry in previous.values()])
    jobs: dict[str, _SubmittedJob] = {
        name: _SubmittedJob(entry["job_id"], int(entry["lane"]), bool(entry["gpu"]))
        for name, entry in previous.items()
        if entry["job_id"] in active
    }
    lane_tails: dict[tuple[bool, int], str] = {}
    lane_sizes: dict[tuple[bool, int], int] = {}
    for job in jobs.values():
        lane_tails[(job.gpu, job.lane)] = job.job_id
        lane_sizes[(job.gpu, job.lane)] = lane_sizes.get((job.gpu, job.lane), 0) + 1
    script_dir = state_dir / ("dry_run_scripts" if dry_run else "slurm_scripts")
    log_dir = state_dir.parent / "slurm_logs"
    by_name = {step.name: step for step in steps}
    next_dry_run_id = DRY_RUN_FIRST_JOB_ID
    submitted = skipped = reused = 0
    for step in steps:
        if _is_done(state_dir, step):
            skipped += 1
            continue
        if step.name in jobs:
            reused += 1
            echo(f"[reproduce] {step.name}: already queued as job {jobs[step.name].job_id}")
            continue
        launcher = resolve_launcher(step, profile, launcher_overrides)
        uses_gpu = int(launcher.gpus) > 0
        lane_count = (
            launcher.max_concurrent_gpu_jobs if uses_gpu else launcher.max_concurrent_cpu_jobs
        )
        dependency_names = [
            name for name in step.dependencies if not _is_done(state_dir, by_name[name])
        ]
        missing = [name for name in dependency_names if name not in jobs]
        if missing:
            raise RuntimeError(f"{step.name} depends on {missing}, which were not submitted.")
        tolerant_ids = {
            lane_tails[(jobs[name].gpu, jobs[name].lane)]
            for name in step.tolerant_dependencies
            if name in jobs
        }
        lane = _choose_lane(jobs, dependency_names, uses_gpu, lane_count, lane_tails, lane_sizes)
        ok_ids = [jobs[name].job_id for name in dependency_names]
        any_ids = sorted(tolerant_ids, key=int)
        lane_tail = lane_tails.get((uses_gpu, lane))
        if lane_tail and lane_tail not in ok_ids and lane_tail not in any_ids:
            any_ids.append(lane_tail)
        commands = [cli_command(arguments, "python") for arguments in step.commands()]
        if step.resumable:
            marker = marker_path(state_dir, step)
            commands.append(shlex.join(["mkdir", "-p", str(marker.parent)]))
            commands.append(shlex.join(["touch", str(marker)]))
        dependency = _dependency(ok_ids, any_ids)
        job_name = f"pc.{step.name}"[:120]
        script_text = render_slurm_script(
            launcher,
            environment_kind=step.environment_kind,
            commands=commands,
            repo_root=repo_root,
            job_name=job_name,
            output_path=log_dir / "%x_%j.out",
            fallback_run_id=f"reproduce_{step.name}",
            dependency=dependency,
            provenance_lines=[f'echo "[placecell_research] reproduce step {step.name}"'],
        )
        script_path = write_slurm_script(script_dir, script_text, stem=step.name)
        if dry_run:
            job_id = str(next_dry_run_id)
            next_dry_run_id += 1
            header = [line for line in script_text.splitlines() if line.startswith("#SBATCH")]
            echo(f"[reproduce] job {job_id} {step.name} lane={'gpu' if uses_gpu else 'cpu'}{lane}")
            for line in header:
                echo(f"  {line}")
            for command in commands:
                echo(f"  $ {command}")
        else:
            job_id = parse_sbatch_job_id(run_sbatch(script_path))
            echo(f"[reproduce] submitted {step.name} as job {job_id} ({dependency or 'no deps'})")
        jobs[step.name] = _SubmittedJob(job_id, lane, uses_gpu)
        lane_tails[(uses_gpu, lane)] = job_id
        lane_sizes[(uses_gpu, lane)] = lane_sizes.get((uses_gpu, lane), 0) + 1
        submitted += 1
        if not dry_run:
            _write_state(state_file, jobs)
    echo(
        f"[reproduce] {submitted} {'planned' if dry_run else 'submitted'}, {reused} already "
        f"queued, {skipped} done. Scripts: {script_dir}. Logs: {log_dir}."
    )
    return 0


def _choose_lane(
    jobs: dict[str, _SubmittedJob],
    dependency_names: list[str],
    uses_gpu: bool,
    lane_count: int,
    lane_tails: dict[tuple[bool, int], str],
    lane_sizes: dict[tuple[bool, int], int],
) -> int:
    for name in dependency_names:
        job = jobs[name]
        is_tail = lane_tails.get((uses_gpu, job.lane)) == job.job_id
        if job.gpu == uses_gpu and job.lane < lane_count and is_tail:
            return job.lane
    return min(range(lane_count), key=lambda lane: (lane_sizes.get((uses_gpu, lane), 0), lane))


def _write_state(state_file: Path, jobs: dict[str, _SubmittedJob]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {"job_id": job.job_id, "lane": job.lane, "gpu": job.gpu}
        for name, job in jobs.items()
    }
    state_file.write_text(json.dumps(payload, indent=1, sort_keys=True))
