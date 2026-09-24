"""SLURM submission helpers."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

from placecell_research.tracking.naming import generate_variant_slug, make_run_id
from placecell_research.utils.repo_paths import find_repo_root, resolve_experiment_config_path

from .code_snapshot import SNAPSHOT_ENV_VAR
from .remote_entrypoints import resolve_remote_cli_entrypoint

SLURM_ENVIRONMENT_SCRIPTS = {
    "miniworld": "env_miniworld.sh",
    "jaxenstein": "env_jaxenstein.sh",
    "common": "env_common.sh",
}
SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


_SLURM_DEPENDENCY_PATTERN = re.compile(r"^[A-Za-z0-9_:+,?.-]+$")
_SLURM_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _format_wallclock_hours(time_hours: int) -> str:
    safe_hours = max(1, int(time_hours))
    return f"{safe_hours:02d}:00:00"


def normalize_slurm_dependency(slurm_dependency: str | None) -> str | None:
    normalized = str(slurm_dependency or "").strip()
    if not normalized:
        return None
    if _SLURM_DEPENDENCY_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "SLURM dependency may contain only letters, numbers, '_', ':', '+', ',', '?', '.', "
            "and '-'."
        )
    return normalized


def normalize_slurm_job_name(job_name: str) -> str:
    normalized = str(job_name or "").strip()
    if not normalized or _SLURM_JOB_NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError("SLURM job name must contain only letters, numbers, '_', '.', and '-'.")
    return normalized


def parse_sbatch_job_id(submit_output: str) -> str:
    match = SBATCH_JOB_ID_PATTERN.search(submit_output)
    if match is None:
        raise RuntimeError(
            "Could not parse a SLURM job id from sbatch output:\n"
            f"{submit_output.strip() or '<empty output>'}"
        )
    return match.group(1)


def raise_if_gpu_request_may_land_on_a_partial_gpu(launcher) -> None:
    """Refuse a GPU request that the scheduler may place on a MIG slice."""
    partial_types = {str(name).strip().lower() for name in launcher.mig_gpu_types if name}
    if launcher.type != "slurm" or int(launcher.gpus) <= 0 or not partial_types:
        return
    gpu_type = str(launcher.gpu_type or "").strip().lower()
    if gpu_type and gpu_type not in partial_types:
        return
    raise ValueError(
        f"This cluster has partial GPU slices (launcher.mig_gpu_types={launcher.mig_gpu_types}), "
        f"and this job requests gpu_type={launcher.gpu_type!r}. MIG slices have no graphics "
        "API, so MiniWorld rendering dies on them, and too little memory for the place model. "
        "Set launcher.gpu_type to a full GPU type."
    )


def resolve_environment_kind(config, config_path: Path) -> str:
    """environment.kind of an experiment, or of the base experiment of a sweep or curriculum."""
    direct = str(getattr(getattr(config, "environment", None), "kind", "") or "").strip().lower()
    if direct:
        return direct
    for block_name in ("curriculum", "sweep"):
        block = getattr(config, block_name, None)
        base = str(getattr(block, "base_experiment", "") or "").strip() if block else ""
        if base:
            from placecell_research.config import load_experiment_config

            base_path = resolve_experiment_config_path(config_path, base)
            return str(load_experiment_config(base_path, []).environment.kind).strip().lower()
    raise ValueError(f"{config_path} names neither environment.kind nor a base experiment.")


def _conda_shell_script(prefix: Path) -> Path | None:
    if not (prefix / "conda-meta").is_dir():
        return None
    bases = []
    conda_executable = os.environ.get("CONDA_EXE", "")
    if conda_executable:
        bases.append(Path(conda_executable).resolve().parents[1])
    if prefix.parent.name == "envs":
        bases.append(prefix.parent.parent)
    bases.append(prefix)
    for base in bases:
        script = base / "etc" / "profile.d" / "conda.sh"
        if script.is_file():
            return script
    return None


def default_environment_activation(prefix: Path | None = None) -> str:
    """Shell line that activates the Python environment running this process."""
    resolved_prefix = Path(prefix or sys.prefix)
    conda_script = _conda_shell_script(resolved_prefix)
    if conda_script is not None:
        return (
            f"source {shlex.quote(str(conda_script))} && "
            f"conda activate {shlex.quote(str(resolved_prefix))}"
        )
    activate_script = resolved_prefix / "bin" / "activate"
    if activate_script.is_file():
        return f"source {shlex.quote(str(activate_script))}"
    return f'export PATH={shlex.quote(str(resolved_prefix / "bin"))}:"${{PATH}}"'


def _threading_exports(launcher) -> list[str]:
    safety = launcher.threading_safety
    exports = [
        f"export OMP_NUM_THREADS={int(safety.omp_num_threads)}",
        f"export NUMBA_NUM_THREADS={int(safety.omp_num_threads)}",
        f"export MKL_NUM_THREADS={int(safety.mkl_num_threads)}",
        f"export OPENBLAS_NUM_THREADS={int(safety.openblas_num_threads)}",
        f"export TORCH_NUM_THREADS={int(safety.torch_num_threads)}",
    ]
    analysis_workers = int(getattr(launcher, "analysis_workers", 0))
    if analysis_workers > 0:
        exports.append(f"export PLACECELL_ANALYSIS_WORKERS={analysis_workers}")
    return exports


def _render_launcher_command(
    launcher_command: str,
    experiment_config_path: Path,
    overrides: list[str],
) -> str:
    quoted_parts = [launcher_command, "--config", shlex.quote(str(experiment_config_path))]
    for override in overrides:
        quoted_parts.extend(["-o", shlex.quote(str(override))])
    return " ".join(quoted_parts)


def _render_launcher_payload(commands: list[str]) -> str:
    body = [f"exec {commands[0]}"] if len(commands) == 1 else list(commands)
    return "\n".join(
        [
            "set -euo pipefail",
            'if [[ "${PLACECELL_REQUESTED_GPUS:-0}" -gt 0 ]]; then',
            '  timeout_seconds="${PLACECELL_CUDA_SMOKE_TIMEOUT_SECONDS:-60}"',
            '  echo "[placecell_research] running CUDA smoke check"',
            "  if command -v timeout >/dev/null 2>&1; then",
            '    timeout "${timeout_seconds}s" python -u -m placecell_research.utils.cuda_smoke',
            "  else",
            "    python -u -m placecell_research.utils.cuda_smoke",
            "  fi",
            "fi",
            *body,
        ]
    )


def _render_printf_lines(payload: str) -> list[str]:
    lines = payload.rstrip().splitlines() or [""]
    return [f"printf '%s\\n' {shlex.quote(line)}" for line in lines]


def _render_launch_provenance(
    resolved_config_path: Path,
    entrypoint: str,
    overrides: list[str],
    config,
    fallback_run_id: str,
) -> list[str]:
    """Compact launch banner in the SLURM script header."""
    override_lines = [f"  - {override}" for override in overrides] if overrides else ["  (none)"]
    return [
        'echo "[placecell_research] launch provenance"',
        f'echo "[placecell_research]   entrypoint: {entrypoint}"',
        f'echo "[placecell_research]   config:     {resolved_config_path}"',
        f'echo "[placecell_research]   variant:    {config.name}"',
        "printf '%s\\n' \"[placecell_research]   run_id:     ${PLACECELL_RUN_ID:-<fallback>}\"",
        f'echo "[placecell_research]   fallback_run_id: {fallback_run_id}"',
        'echo "[placecell_research]   overrides:"',
        *_render_printf_lines("\n".join(override_lines)),
    ]


def _render_sbatch_header(
    launcher,
    *,
    job_name: str,
    output_path: Path,
    dependency: str | None,
) -> list[str]:
    header = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={normalize_slurm_job_name(job_name)}",
        f"#SBATCH --partition={launcher.partition}",
    ]
    for option, value in (
        ("account", launcher.account),
        ("qos", launcher.qos),
        ("constraint", launcher.constraint),
    ):
        if str(value).strip():
            header.append(f"#SBATCH --{option}={str(value).strip()}")
    header.extend(
        [
            f"#SBATCH --cpus-per-task={int(launcher.cpus_per_task)}",
            f"#SBATCH --mem={int(launcher.memory_gb)}G",
            f"#SBATCH --time={_format_wallclock_hours(launcher.time_hours)}",
            f"#SBATCH --output={output_path}",
            "#SBATCH --signal=B:TERM@120",
        ]
    )
    normalized_dependency = normalize_slurm_dependency(dependency)
    if normalized_dependency is not None:
        header.append(f"#SBATCH --dependency={normalized_dependency}")
        header.append("#SBATCH --kill-on-invalid-dep=yes")
    excluded_nodes = [str(node).strip() for node in launcher.exclude_nodes if str(node).strip()]
    if excluded_nodes:
        header.append(f"#SBATCH --exclude={','.join(excluded_nodes)}")
    if int(launcher.gpus) > 0:
        gpu_count = int(launcher.gpus)
        gpu_type = str(launcher.gpu_type or "").strip()
        gpu_request = f"gpu:{gpu_type}:{gpu_count}" if gpu_type else f"gpu:{gpu_count}"
        header.append(f"#SBATCH --gres={gpu_request}")
    return header


def _render_environment_lines(launcher, environment_kind: str) -> list[str]:
    if environment_kind not in SLURM_ENVIRONMENT_SCRIPTS:
        raise ValueError(f"No SLURM environment script for environment.kind={environment_kind!r}.")
    activation = str(launcher.env_setup).strip() or default_environment_activation()
    lines = [
        f'PLACECELL_SLURM_SCRIPT_ROOT="${{{SNAPSHOT_ENV_VAR}:-.}}/scripts/slurm"',
        'source "${PLACECELL_SLURM_SCRIPT_ROOT}/env_common.sh"',
        "set +u",
        activation,
        "set -u",
    ]
    site_env_script = str(launcher.site_env_script).strip()
    if site_env_script:
        lines.append(f'source "${{PLACECELL_SLURM_SCRIPT_ROOT}}/{site_env_script}"')
    if environment_kind != "common":
        script_name = SLURM_ENVIRONMENT_SCRIPTS[environment_kind]
        lines.append(f'source "${{PLACECELL_SLURM_SCRIPT_ROOT}}/{script_name}"')
    return lines


def _render_snapshot_exports() -> list[str]:
    snapshot = os.environ.get(SNAPSHOT_ENV_VAR, "").strip()
    if not snapshot:
        return []
    return [
        f"export {SNAPSHOT_ENV_VAR}={shlex.quote(snapshot)}",
        f"export PYTHONPATH={shlex.quote(snapshot + '/src')}" + "${PYTHONPATH:+:${PYTHONPATH}}",
    ]


def render_slurm_script(
    launcher,
    *,
    environment_kind: str,
    commands: list[str],
    repo_root: Path,
    job_name: str,
    output_path: Path,
    fallback_run_id: str,
    dependency: str | None = None,
    provenance_lines: list[str] | None = None,
) -> str:
    """The hardened batch script: resources, per-job environment, TERM forwarding, cleanup."""
    raise_if_gpu_request_may_land_on_a_partial_gpu(launcher)
    if not commands:
        raise ValueError("A SLURM job needs at least one command.")
    launcher_command_string = shlex.quote(_render_launcher_payload(commands))
    exports = [
        f"export {name}={shlex.quote(str(value))}" for name, value in launcher.exports.items()
    ]
    script_lines = [
        *_render_sbatch_header(
            launcher, job_name=job_name, output_path=output_path, dependency=dependency
        ),
        "",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(repo_root))}",
        *_render_snapshot_exports(),
        *_threading_exports(launcher),
        f"export PLACECELL_REQUESTED_GPUS={int(launcher.gpus)}",
        f"export PLACECELL_ENVIRONMENT_KIND={environment_kind}",
        *exports,
        'export PLACECELL_DATASET_STAGE_DIR="${PLACECELL_DATASET_STAGE_DIR:-/dev/shm}"',
        f"export PLACECELL_FALLBACK_RUN_ID={shlex.quote(fallback_run_id)}",
        'PLACECELL_EFFECTIVE_JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"',
        'if [[ -n "${PLACECELL_EFFECTIVE_JOB_ID}" ]]; then',
        (
            '  export PLACECELL_RUN_ID="slurm_${PLACECELL_EFFECTIVE_JOB_ID}'
            '__${PLACECELL_FALLBACK_RUN_ID}"'
        ),
        "else",
        '  export PLACECELL_RUN_ID="${PLACECELL_FALLBACK_RUN_ID}_manual_$$"',
        "fi",
        'echo "[placecell_research] run_id=${PLACECELL_RUN_ID}"',
        *_render_environment_lines(launcher, environment_kind),
        *(provenance_lines or []),
        "",
        'launcher_pid=""',
        'launcher_pgid=""',
        "",
        "diagnose_job_state() {",
        '  echo "[placecell_research] post-job diagnostics"',
        '  if [[ -n "${launcher_pgid:-}" ]]; then',
        '    echo "[placecell_research] launcher process group"',
        (
            "    ps -o pid,ppid,pgid,sid,stat,etime,%cpu,%mem,command -g "
            '"${launcher_pgid}" 2>/dev/null || true'
        ),
        "  fi",
        '  echo "[placecell_research] matching user processes"',
        "  if command -v pgrep >/dev/null 2>&1; then",
        (
            '    pgrep -a -u "${USER:-$(id -un)}" -f '
            "'placecell_research|python|miniworld|wandb|stable_baselines|gymnasium' "
            "2>/dev/null || true"
        ),
        "  fi",
        "  if command -v nvidia-smi >/dev/null 2>&1; then",
        "    if command -v timeout >/dev/null 2>&1; then",
        (
            "      timeout 10s nvidia-smi --query-gpu=name,driver_version,pstate "
            "--format=csv,noheader 2>/dev/null || true"
        ),
        "    else",
        (
            "      nvidia-smi --query-gpu=name,driver_version,pstate "
            "--format=csv,noheader 2>/dev/null || true"
        ),
        "    fi",
        "  fi",
        "  local current_pgid",
        "  current_pgid=\"$(ps -o pgid= $$ | tr -d ' ')\"",
        (
            "  ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,command -g "
            '"${current_pgid}" 2>/dev/null || true'
        ),
        "}",
        "launcher_tree_alive() {",
        '  if [[ -n "${launcher_pgid:-}" ]]; then',
        '    kill -0 -- "-${launcher_pgid}" 2>/dev/null && return 0',
        "    if command -v pgrep >/dev/null 2>&1; then",
        '      pgrep -s "${launcher_pgid}" >/dev/null 2>&1 && return 0',
        "    fi",
        "  fi",
        '  if [[ -n "${launcher_pid:-}" ]]; then',
        '    kill -0 "${launcher_pid}" 2>/dev/null && return 0',
        "  fi",
        "  return 1",
        "}",
        "terminate_launcher_tree() {",
        '  local signal_name="$1"',
        '  if [[ -n "${launcher_pgid:-}" ]]; then',
        "    if command -v pkill >/dev/null 2>&1; then",
        '      pkill "-${signal_name}" -s "${launcher_pgid}" 2>/dev/null || true',
        "    fi",
        '    kill "-${signal_name}" -- "-${launcher_pgid}" 2>/dev/null || true',
        "    return",
        "  fi",
        '  if [[ -n "${launcher_pid:-}" ]]; then',
        '    kill "-${signal_name}" "${launcher_pid}" 2>/dev/null || true',
        "  fi",
        "}",
        "wait_for_launcher_tree() {",
        '  local timeout_seconds="$1"',
        "  local deadline=$((SECONDS + timeout_seconds))",
        "  while launcher_tree_alive; do",
        "    if (( SECONDS >= deadline )); then",
        "      return 1",
        "    fi",
        "    sleep 1",
        "  done",
        "  return 0",
        "}",
        "shutdown_launcher() {",
        "  trap - TERM INT",
        '  echo "[placecell_research] forwarding SIGTERM to launcher tree" >&2',
        "  terminate_launcher_tree TERM",
        '  if ! wait_for_launcher_tree "${PLACECELL_TERM_GRACE_SECONDS:-90}"; then',
        '    echo "[placecell_research] launcher tree still alive; sending SIGKILL" >&2',
        "    terminate_launcher_tree KILL",
        "    wait_for_launcher_tree 10 || true",
        "  fi",
        '  wait "${launcher_pid}" 2>/dev/null || true',
        "  exit 143",
        "}",
        "cleanup_on_exit() {",
        "  local status=$?",
        "  trap - EXIT",
        "  if launcher_tree_alive; then",
        '    echo "[placecell_research] cleanup found live launcher tree; killing it" >&2',
        "    terminate_launcher_tree TERM",
        "    sleep 5",
        "    terminate_launcher_tree KILL",
        "    wait_for_launcher_tree 10 || true",
        "  fi",
        "  placecell_cleanup_hpc_env || true",
        "  diagnose_job_state || true",
        '  exit "${status}"',
        "}",
        "trap shutdown_launcher TERM INT",
        "trap cleanup_on_exit EXIT",
        "",
        f"setsid bash --noprofile --norc -c {launcher_command_string} &",
        "launcher_pid=$!",
        'launcher_pgid="${launcher_pid}"',
        "set +e",
        'wait "${launcher_pid}"',
        "launcher_status=$?",
        "set -e",
        'exit "${launcher_status}"',
        "",
    ]
    return "\n".join(script_lines)


def write_slurm_script(script_dir: Path, script_text: str, *, stem: str = "") -> Path:
    script_dir.mkdir(parents=True, exist_ok=True)
    name = stem or f"generated_submit_{uuid.uuid4().hex[:12]}"
    script_path = script_dir / f"{name}.sh"
    script_path.write_text(script_text)
    script_path.chmod(0o755)
    return script_path


def run_sbatch(script_path: Path) -> str:
    completed = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or completed.stderr.strip()


def submit_cli_entrypoint(
    config_path: str | Path,
    overrides: list[str] | None = None,
    *,
    entrypoint: str = "pipeline",
    launcher_command: str | None = None,
    force_recompute: bool = False,
    slurm_dependency: str | None = None,
    slurm_job_name: str = "placecell_research",
    dry_run: bool = False,
) -> str:
    """Generate and optionally submit a thin but hardened SLURM script."""
    resolved_config_path = Path(config_path).resolve()
    resolved_entrypoint = resolve_remote_cli_entrypoint(
        resolved_config_path,
        entrypoint,
        overrides,
        force_recompute=force_recompute,
    )
    repo_root = find_repo_root(resolved_config_path)
    config = resolved_entrypoint.config
    environment_kind = resolve_environment_kind(config, resolved_config_path)
    fallback_run_id = make_run_id(
        repo_root,
        descriptor=generate_variant_slug(
            config.to_dict(),
            fallback_name=config.name,
        ),
        include_slurm_job=False,
    )
    run_root = repo_root / config.tracking.run_root
    slurm_log_dir = run_root / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_launcher_command = (
        launcher_command
        or f"python -u -m placecell_research.launch.cli {resolved_entrypoint.cli_command}"
    )
    launcher_line = _render_launcher_command(
        resolved_launcher_command,
        resolved_config_path,
        resolved_entrypoint.overrides,
    )
    script_text = render_slurm_script(
        config.launcher,
        environment_kind=environment_kind,
        commands=[launcher_line],
        repo_root=repo_root,
        job_name=slurm_job_name,
        output_path=slurm_log_dir / "%x_%j.out",
        fallback_run_id=fallback_run_id,
        dependency=slurm_dependency,
        provenance_lines=_render_launch_provenance(
            resolved_config_path,
            resolved_entrypoint.cli_command,
            resolved_entrypoint.overrides,
            config,
            fallback_run_id,
        ),
    )
    script_path = write_slurm_script(run_root / "slurm_scripts", script_text)
    if config.launcher.type != "slurm" or dry_run:
        return str(script_path)
    return run_sbatch(script_path)
