"""Remote SLURM launch and log streaming over SSH."""

from __future__ import annotations

import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from rich.console import Console

from placecell_research.utils.repo_paths import find_repo_root

from .code_snapshot import (
    SNAPSHOT_DIR_NAME,
    SNAPSHOT_ENV_VAR,
    CodeSnapshot,
    build_code_snapshot_script,
    parse_code_snapshot_output,
    retention_cutoff,
    rewrite_config_path_for_snapshot,
    snapshot_timestamp,
)
from .remote_entrypoints import resolve_remote_cli_entrypoint
from .submit import normalize_slurm_dependency, normalize_slurm_job_name, parse_sbatch_job_id
from .user_settings import RemoteSettings, load_user_settings, user_config_path

ACTIVE_SLURM_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "STAGE_OUT",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_HOLD",
    "SPECIAL_EXIT",
    "SIGNALING",
}
TERMINAL_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_RSYNC_EXCLUDES = (
    "/.git/",
    f"/{SNAPSHOT_DIR_NAME}/",
    "/runs/",
    "/artifacts/",
    "/smoke/",
    "/measures/",
    "/navigation/",
    "/wandb/",
    "/.venv*/",
    "/.cache/",
    "/.pytest_cache/",
    "/.ruff_cache/",
    "__pycache__/",
    ".DS_Store",
)


@dataclass(slots=True)
class RemoteRunResult:
    remote_host: str
    job_id: str
    final_state: str
    slurm_log_path: str
    submitted: bool = True
    code_snapshot_path: str = ""
    code_snapshot_hash: str = ""


@dataclass(slots=True)
class RemoteActiveJob:
    job_id: str
    state: str
    job_name: str


@dataclass(slots=True)
class _ActiveRemoteJob:
    remote_host: str
    ssh_options: tuple[str, ...]
    job_id: str
    tail_process: subprocess.Popen[str] | None = None
    cancelled: bool = False


_CURRENT_REMOTE_JOB: _ActiveRemoteJob | None = None
_CONSOLE = Console(stderr=True, markup=False, highlight=False, soft_wrap=True)


def _run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _spawn_subprocess(command: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(command, text=True)


def _ssh_command(remote_host: str, ssh_options: Iterable[str], remote_script: str) -> list[str]:
    remote_command = " ".join(
        ["bash", "--noprofile", "--norc", "-lc", shlex.quote(remote_script)]
    )
    return ["ssh", *ssh_options, remote_host, remote_command]


def _rsync_command(
    *,
    local_repo_root: Path,
    remote_host: str,
    remote_repo_root: PurePosixPath,
    ssh_options: Iterable[str],
    excludes: Iterable[str],
    delete: bool,
) -> list[str]:
    ssh_parts = ["ssh", *ssh_options]
    command = ["rsync", "-az", "-e", " ".join(shlex.quote(part) for part in ssh_parts)]
    if delete:
        command.append("--delete")
    for pattern in excludes:
        command.extend(["--exclude", str(pattern)])
    command.extend(
        [
            f"{str(local_repo_root.resolve()).rstrip('/')}/",
            f"{remote_host}:{str(remote_repo_root).rstrip('/')}/",
        ]
    )
    return command


def _remote_shell_block(*lines: str) -> str:
    return "\n".join(["set -euo pipefail", *lines])


def _checked(completed: subprocess.CompletedProcess[str], action: str) -> str:
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{action} failed with exit code {completed.returncode}:\n{output.strip()}"
        )
    return output


def resolve_remote_settings(
    *,
    remote_host: str | None = None,
    remote_repo_root: str | None = None,
    remote_setup: str | None = None,
    remote_python: str | None = None,
    ssh_options: list[str] | None = None,
) -> RemoteSettings:
    """Command-line values over PLACECELL_REMOTE_* variables over the user file."""
    configured = load_user_settings().remote
    resolved = RemoteSettings(
        host=remote_host or configured.host,
        repo_root=remote_repo_root or configured.repo_root,
        setup=remote_setup if remote_setup is not None else configured.setup,
        python=remote_python or configured.python,
        ssh_options=tuple(ssh_options) if ssh_options else configured.ssh_options,
    )
    missing = [name for name in ("host", "repo_root") if not getattr(resolved, name)]
    if missing:
        raise ValueError(
            f"Missing remote {', '.join(missing)}. Set remote.host and remote.repo_root in "
            f"{user_config_path()}, export PLACECELL_REMOTE_HOST / PLACECELL_REMOTE_REPO_ROOT, "
            "or pass --remote-host / --remote-repo-root."
        )
    return resolved


def _remote_path_for(local_path: Path, local_repo_root: Path, remote_root: PurePosixPath):
    try:
        relative = local_path.resolve().relative_to(local_repo_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{local_path} is outside the repository {local_repo_root}, so it has no remote copy."
        ) from exc
    return remote_root.joinpath(*relative.parts)


def sync_repo_to_remote(
    *,
    local_repo_root: Path,
    settings: RemoteSettings,
    delete: bool = False,
    extra_excludes: list[str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Mirror the local repo to the remote checkout with the default excludes."""
    remote_root = PurePosixPath(settings.repo_root)
    excludes = [*DEFAULT_RSYNC_EXCLUDES, *(extra_excludes or [])]
    if progress_callback is not None:
        progress_callback(f"[remote-sync] {local_repo_root} -> {settings.host}:{remote_root}")
    _checked(
        _run_subprocess(
            _ssh_command(
                settings.host,
                settings.ssh_options,
                _remote_shell_block(f"mkdir -p {shlex.quote(str(remote_root))}"),
            )
        ),
        f"Preparing {settings.host}:{remote_root}",
    )
    output = _checked(
        _run_subprocess(
            _rsync_command(
                local_repo_root=local_repo_root,
                remote_host=settings.host,
                remote_repo_root=remote_root,
                ssh_options=settings.ssh_options,
                excludes=excludes,
                delete=delete,
            )
        ),
        f"rsync to {settings.host}:{remote_root}",
    )
    return output.strip()


def create_remote_code_snapshot(settings: RemoteSettings) -> CodeSnapshot:
    """Freeze the remote checkout's importable code and prune stale snapshots."""
    script = build_code_snapshot_script(
        remote_repo_root=PurePosixPath(settings.repo_root),
        timestamp=snapshot_timestamp(),
        cutoff=retention_cutoff(),
    )
    output = _checked(
        _run_subprocess(_ssh_command(settings.host, settings.ssh_options, script)),
        f"Remote code snapshot on {settings.host}",
    )
    return parse_code_snapshot_output(output)


def _remote_environment_lines(settings: RemoteSettings, code_snapshot: CodeSnapshot | None):
    lines = [f"cd {shlex.quote(settings.repo_root)}"]
    if settings.setup:
        lines.append(settings.setup)
    if code_snapshot is None:
        lines.append("export PYTHONPATH=src${PYTHONPATH:+:${PYTHONPATH}}")
    else:
        lines.append(
            f"export PYTHONPATH={shlex.quote(code_snapshot.python_path)}"
            "${PYTHONPATH:+:${PYTHONPATH}}"
        )
        lines.append(f"export {SNAPSHOT_ENV_VAR}={shlex.quote(code_snapshot.path)}")
    return lines


def format_remote_cli_command(
    settings: RemoteSettings,
    arguments: list[str],
    code_snapshot: CodeSnapshot | None,
) -> str:
    """Remote shell script that runs `pc <arguments>` in the remote checkout."""
    command = " ".join(
        [
            shlex.quote(settings.python),
            "-m",
            "placecell_research.launch.cli",
            *(shlex.quote(argument) for argument in arguments),
        ]
    )
    return _remote_shell_block(*_remote_environment_lines(settings, code_snapshot), command)


def _remote_job_status(settings: RemoteSettings, job_id: str) -> str:
    remote_script = _remote_shell_block(
        f"state=$(squeue -h -j {shlex.quote(job_id)} -o %T 2>/dev/null | head -n 1 || true)",
        'if [ -n "${state}" ]; then',
        '  printf "%s\\n" "${state}"',
        "  exit 0",
        "fi",
        "if command -v sacct >/dev/null 2>&1; then",
        f"  state=$(sacct -j {shlex.quote(job_id)} --format=State --noheader 2>/dev/null"
        " | head -n 1 | awk '{print $1}' || true)",
        '  if [ -n "${state}" ]; then',
        '    printf "%s\\n" "${state}"',
        "    exit 0",
        "  fi",
        "fi",
        'printf "%s\\n" "UNKNOWN"',
    )
    completed = _run_subprocess(_ssh_command(settings.host, settings.ssh_options, remote_script))
    lines = [
        line.strip()
        for line in ((completed.stdout or "") + "\n" + (completed.stderr or "")).splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else "UNKNOWN"


def list_remote_active_slurm_jobs(
    settings: RemoteSettings, *, job_name: str = "placecell_research"
) -> list[RemoteActiveJob]:
    """Active jobs of the logged-in remote user with this job name; fails closed."""
    remote_script = _remote_shell_block(
        f'squeue -h -u "${{USER}}" -n {shlex.quote(job_name)} -o "%i|%T|%j"'
    )
    completed = _run_subprocess(_ssh_command(settings.host, settings.ssh_options, remote_script))
    if completed.returncode != 0:
        raise RuntimeError(
            f"Remote job lookup on {settings.host} failed with exit code {completed.returncode}. "
            "An unanswered scheduler query is not proof that no job is queued; fix the "
            f"connection and retry.\n{(completed.stdout or '')}{completed.stderr or ''}".strip()
        )
    jobs = []
    for line in (completed.stdout or "").splitlines():
        parts = [part.strip() for part in line.split("|", maxsplit=2)]
        if len(parts) == 3 and parts[0]:
            jobs.append(RemoteActiveJob(parts[0], parts[1] or "UNKNOWN", parts[2] or job_name))
    jobs.sort(key=lambda job: int(job.job_id) if job.job_id.isdigit() else -1, reverse=True)
    return jobs


def _terminate_tail_process(tail_process: subprocess.Popen[str] | None) -> None:
    if tail_process is None or tail_process.poll() is not None:
        return
    tail_process.terminate()
    try:
        tail_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tail_process.kill()
        tail_process.wait(timeout=5)


def _cancel_active_remote_job() -> None:
    if _CURRENT_REMOTE_JOB is None or _CURRENT_REMOTE_JOB.cancelled:
        return
    _CURRENT_REMOTE_JOB.cancelled = True
    _CONSOLE.log(f"[remote-run] cancelling remote SLURM job {_CURRENT_REMOTE_JOB.job_id}")
    _terminate_tail_process(_CURRENT_REMOTE_JOB.tail_process)
    remote_script = _remote_shell_block(
        f"scancel {shlex.quote(_CURRENT_REMOTE_JOB.job_id)} >/dev/null 2>&1 || true"
    )
    job = _CURRENT_REMOTE_JOB
    _run_subprocess(_ssh_command(job.remote_host, job.ssh_options, remote_script))


def _install_signal_handlers(cancel_on_interrupt: bool) -> dict[int, object]:
    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {handled: signal.getsignal(handled) for handled in handled_signals}

    def _handle_interruption(signum: int, _frame) -> None:
        if cancel_on_interrupt:
            _cancel_active_remote_job()
        raise SystemExit(128 + signum)

    for handled in handled_signals:
        signal.signal(handled, _handle_interruption)
    return previous_handlers


def stream_remote_job(
    settings: RemoteSettings,
    *,
    job_id: str,
    slurm_log_path: PurePosixPath,
    cancel_on_interrupt: bool,
) -> RemoteRunResult:
    """Tail the job log until the job leaves the active states."""
    global _CURRENT_REMOTE_JOB
    _CONSOLE.log(f"[remote-run] streaming {slurm_log_path}")
    action = "cancel" if cancel_on_interrupt else "leave running"
    _CONSOLE.log(f"[remote-run] Ctrl+C will {action} remote job {job_id}")
    previous_handlers = _install_signal_handlers(cancel_on_interrupt)
    tail_script = _remote_shell_block(
        f"while [ ! -f {shlex.quote(str(slurm_log_path))} ]; do sleep 1; done",
        f"tail -n +1 -F {shlex.quote(str(slurm_log_path))}",
    )
    tail_process: subprocess.Popen[str] | None = None
    final_state = "UNKNOWN"
    _CURRENT_REMOTE_JOB = _ActiveRemoteJob(settings.host, settings.ssh_options, job_id)
    try:
        tail_process = _spawn_subprocess(
            _ssh_command(settings.host, settings.ssh_options, tail_script)
        )
        _CURRENT_REMOTE_JOB.tail_process = tail_process
        while True:
            time.sleep(TERMINAL_POLL_INTERVAL_SECONDS)
            final_state = _remote_job_status(settings, job_id)
            if final_state not in ACTIVE_SLURM_STATES:
                break
        time.sleep(1.0)
    except KeyboardInterrupt:
        if cancel_on_interrupt:
            _cancel_active_remote_job()
        raise
    finally:
        _terminate_tail_process(tail_process)
        for handled, handler in previous_handlers.items():
            signal.signal(handled, handler)
        _CURRENT_REMOTE_JOB = None
    _CONSOLE.log(f"[remote-run] job {job_id} finished with state: {final_state}")
    return RemoteRunResult(settings.host, job_id, final_state, str(slurm_log_path))


def prepare_remote_checkout(
    config_path: Path,
    settings: RemoteSettings,
    *,
    sync_repo: bool,
    snapshot_code: bool,
) -> tuple[Path, CodeSnapshot | None]:
    """Sync the checkout and freeze a code snapshot; returns (local repo root, snapshot)."""
    local_repo_root = find_repo_root(config_path)
    if sync_repo:
        output = sync_repo_to_remote(local_repo_root=local_repo_root, settings=settings)
        if output:
            _CONSOLE.print(output)
    if not snapshot_code:
        _CONSOLE.log(
            "[remote-run] --no-snapshot: the job imports the live remote checkout, and a "
            "sync while it runs can change its code."
        )
        return local_repo_root, None
    code_snapshot = create_remote_code_snapshot(settings)
    _CONSOLE.log(f"[remote-run] code snapshot {code_snapshot.describe()}")
    return local_repo_root, code_snapshot


def submit_remote_slurm_job(
    config_path: str | Path,
    overrides: list[str] | None,
    settings: RemoteSettings,
    *,
    entrypoint: str = "auto",
    sync_repo: bool = True,
    force_recompute: bool = False,
    slurm_dependency: str | None = None,
    slurm_job_name: str = "placecell_research",
    snapshot_code: bool = True,
) -> RemoteRunResult:
    """Submit one job on the remote login node with `pc submit`, without streaming."""
    resolved_config_path = Path(config_path).resolve()
    normalized_dependency = normalize_slurm_dependency(slurm_dependency)
    normalized_job_name = normalize_slurm_job_name(slurm_job_name)
    resolved_entrypoint = resolve_remote_cli_entrypoint(
        resolved_config_path, entrypoint, overrides, force_recompute=force_recompute
    )
    local_repo_root, code_snapshot = prepare_remote_checkout(
        resolved_config_path, settings, sync_repo=sync_repo, snapshot_code=snapshot_code
    )
    remote_root = PurePosixPath(settings.repo_root)
    remote_config_path = _remote_path_for(resolved_config_path, local_repo_root, remote_root)
    if code_snapshot is not None:
        remote_config_path = rewrite_config_path_for_snapshot(
            remote_config_path=remote_config_path,
            remote_repo_root=remote_root,
            snapshot_path=code_snapshot.path,
        )
    arguments = [
        "submit",
        "--entrypoint",
        resolved_entrypoint.cli_command,
        "--config",
        str(remote_config_path),
        "--slurm-job-name",
        normalized_job_name,
    ]
    if normalized_dependency is not None:
        arguments.extend(["--slurm-dependency", normalized_dependency])
    for override in resolved_entrypoint.overrides:
        arguments.extend(["-o", override])
    remote_script = format_remote_cli_command(settings, arguments, code_snapshot)
    output = _checked(
        _run_subprocess(_ssh_command(settings.host, settings.ssh_options, remote_script)),
        f"Remote submit on {settings.host}",
    )
    job_id = parse_sbatch_job_id(output)
    run_root = resolved_entrypoint.config.tracking.run_root
    remote_run_root = (
        PurePosixPath(str(run_root))
        if Path(run_root).is_absolute()
        else remote_root.joinpath(*Path(run_root).parts)
    )
    _CONSOLE.log(f"[remote-run] submitted {job_id} on {settings.host}")
    return RemoteRunResult(
        remote_host=settings.host,
        job_id=job_id,
        final_state="SUBMITTED",
        slurm_log_path=str(remote_run_root / "slurm_logs" / f"{normalized_job_name}_{job_id}.out"),
        code_snapshot_path="" if code_snapshot is None else code_snapshot.path,
        code_snapshot_hash="" if code_snapshot is None else code_snapshot.content_hash,
    )


def run_remote_slurm_job(
    config_path: str | Path,
    overrides: list[str] | None,
    settings: RemoteSettings,
    *,
    entrypoint: str = "auto",
    sync_repo: bool = True,
    force_recompute: bool = False,
    snapshot_code: bool = True,
    slurm_dependency: str | None = None,
    stream_logs: bool = True,
    cancel_on_interrupt: bool = True,
) -> RemoteRunResult:
    """Submit on the remote cluster over SSH and stream the SLURM log."""
    submitted = submit_remote_slurm_job(
        config_path,
        overrides,
        settings,
        entrypoint=entrypoint,
        sync_repo=sync_repo,
        force_recompute=force_recompute,
        slurm_dependency=slurm_dependency,
        snapshot_code=snapshot_code,
    )
    if not stream_logs:
        return submitted
    streamed = stream_remote_job(
        settings,
        job_id=submitted.job_id,
        slurm_log_path=PurePosixPath(submitted.slurm_log_path),
        cancel_on_interrupt=cancel_on_interrupt,
    )
    return replace(
        streamed,
        code_snapshot_path=submitted.code_snapshot_path,
        code_snapshot_hash=submitted.code_snapshot_hash,
    )


def attach_to_remote_slurm_job(
    settings: RemoteSettings,
    *,
    job_id: str,
    run_root: Path = Path("runs"),
    job_name: str = "placecell_research",
    cancel_on_interrupt: bool = False,
) -> RemoteRunResult:
    """Stream the log of an existing remote job."""
    remote_root = PurePosixPath(settings.repo_root)
    log_path = remote_root.joinpath(*run_root.parts) / "slurm_logs" / f"{job_name}_{job_id}.out"
    result = stream_remote_job(
        settings, job_id=job_id, slurm_log_path=log_path, cancel_on_interrupt=cancel_on_interrupt
    )
    result.submitted = False
    return result


def run_remote_cli(
    config_path: Path,
    arguments: list[str],
    settings: RemoteSettings,
    *,
    sync_repo: bool = True,
    snapshot_code: bool = True,
) -> int:
    """Sync, snapshot, then run `pc <arguments>` on the remote login node and echo its output."""
    _, code_snapshot = prepare_remote_checkout(
        config_path, settings, sync_repo=sync_repo, snapshot_code=snapshot_code
    )
    remote_script = format_remote_cli_command(settings, list(arguments), code_snapshot)
    completed = subprocess.run(
        _ssh_command(settings.host, settings.ssh_options, remote_script), check=False
    )
    return int(completed.returncode)
