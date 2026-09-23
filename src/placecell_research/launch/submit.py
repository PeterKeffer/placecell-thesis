"""SLURM submission helpers."""

from __future__ import annotations

import re
import shlex
import subprocess
import uuid
from pathlib import Path

from placecell_research.tracking.naming import generate_variant_slug, make_run_id
from placecell_research.utils.repo_paths import find_repo_root

from .remote_entrypoints import resolve_remote_cli_entrypoint

_SLURM_ENVIRONMENT_SCRIPTS = {
    "miniworld": "env_miniworld.sh",
    "jaxenstein": "env_jaxenstein.sh",
}


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
        raise ValueError(
            "SLURM job name must contain only letters, numbers, '_', '.', and '-'."
        )
    return normalized


def _threading_exports(config) -> list[str]:
    safety = config.launcher.threading_safety
    exports = [
        f"export OMP_NUM_THREADS={int(safety.omp_num_threads)}",
        f"export NUMBA_NUM_THREADS={int(safety.omp_num_threads)}",
        f"export MKL_NUM_THREADS={int(safety.mkl_num_threads)}",
        f"export OPENBLAS_NUM_THREADS={int(safety.openblas_num_threads)}",
        f"export TORCH_NUM_THREADS={int(safety.torch_num_threads)}",
    ]
    analysis_workers = int(getattr(config.launcher, "analysis_workers", 0))
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


def _render_launcher_payload(launcher_line: str) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            "if [[ \"${PLACECELL_REQUESTED_GPUS:-0}\" -gt 0 ]]; then",
            "  timeout_seconds=\"${PLACECELL_CUDA_SMOKE_TIMEOUT_SECONDS:-60}\"",
            "  echo \"[placecell_research] running CUDA smoke check\"",
            "  if command -v timeout >/dev/null 2>&1; then",
            "    timeout \"${timeout_seconds}s\" python -u -m placecell_research.utils.cuda_smoke",
            "  else",
            "    python -u -m placecell_research.utils.cuda_smoke",
            "  fi",
            "fi",
            f"exec {launcher_line}",
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
    override_lines = (
        [f"  - {override}" for override in overrides] if overrides else ["  (none)"]
    )
    return [
        'echo "[placecell_research] launch provenance"',
        f'echo "[placecell_research]   entrypoint: {entrypoint}"',
        f'echo "[placecell_research]   config:     {resolved_config_path}"',
        f'echo "[placecell_research]   variant:    {config.tracking.variant_name}"',
        'printf \'%s\\n\' "[placecell_research]   run_id:     ${PLACECELL_RUN_ID:-<fallback>}"',
        f'echo "[placecell_research]   fallback_run_id: {fallback_run_id}"',
        'echo "[placecell_research]   overrides:"',
        *_render_printf_lines("\n".join(override_lines)),
    ]


def submit_cli_entrypoint(
    config_path: str | Path,
    overrides: list[str] | None = None,
    *,
    entrypoint: str = "pipeline",
    launcher_command: str | None = None,
    force_recompute: bool = False,
    slurm_dependency: str | None = None,
    slurm_job_name: str = "placecell_research",
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
    environment_kind = str(config.environment.kind).strip().lower()
    if environment_kind not in _SLURM_ENVIRONMENT_SCRIPTS:
        raise ValueError(f"No SLURM environment script for environment.kind={environment_kind!r}.")
    fallback_run_id = make_run_id(
        repo_root,
        descriptor=generate_variant_slug(
            config.to_dict(),
            fallback_name=config.tracking.variant_name,
        ),
        include_slurm_job=False,
    )
    normalized_slurm_dependency = normalize_slurm_dependency(slurm_dependency)

    slurm_log_dir = repo_root / config.tracking.run_root / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    slurm_output_path = slurm_log_dir / "%x_%j.out"
    script_dir = repo_root / config.tracking.run_root / "slurm_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"generated_submit_{uuid.uuid4().hex[:12]}.sh"

    normalized_slurm_job_name = normalize_slurm_job_name(slurm_job_name)
    sbatch_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={normalized_slurm_job_name}",
        f"#SBATCH --partition={config.launcher.partition}",
        f"#SBATCH --cpus-per-task={int(config.launcher.cpus_per_task)}",
        f"#SBATCH --mem={int(config.launcher.memory_gb)}G",
        f"#SBATCH --time={_format_wallclock_hours(config.launcher.time_hours)}",
        f"#SBATCH --output={slurm_output_path}",
        "#SBATCH --signal=B:TERM@120",
    ]
    if normalized_slurm_dependency is not None:
        sbatch_lines.append(f"#SBATCH --dependency={normalized_slurm_dependency}")
    excluded_nodes = [
        str(node).strip()
        for node in config.launcher.exclude_nodes
        if str(node).strip()
    ]
    if excluded_nodes:
        sbatch_lines.append(f"#SBATCH --exclude={','.join(excluded_nodes)}")
    if int(config.launcher.gpus) > 0:
        gpu_count = int(config.launcher.gpus)
        gpu_type = str(config.launcher.gpu_type or "").strip()
        gpu_request = f"gpu:{gpu_type}:{gpu_count}" if gpu_type else f"gpu:{gpu_count}"
        sbatch_lines.append(f"#SBATCH --gres={gpu_request}")

    resolved_launcher_command = (
        launcher_command
        or f"python -u -m placecell_research.launch.cli {resolved_entrypoint.cli_command}"
    )
    launcher_line = _render_launcher_command(
        resolved_launcher_command,
        resolved_config_path,
        resolved_entrypoint.overrides,
    )
    launcher_command_string = shlex.quote(_render_launcher_payload(launcher_line))
    environment_setup = str(config.launcher.env_setup).strip()
    script_lines = [
        *sbatch_lines,
        "",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(repo_root))}",
        *([environment_setup] if environment_setup else []),
        *(_threading_exports(config)),
        f"export PLACECELL_REQUESTED_GPUS={int(config.launcher.gpus)}",
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
        f'source scripts/slurm/{_SLURM_ENVIRONMENT_SCRIPTS[environment_kind]}',
        *_render_launch_provenance(
            resolved_config_path,
            resolved_entrypoint.cli_command,
            resolved_entrypoint.overrides,
            config,
            fallback_run_id,
        ),
        "",
        "launcher_pid=\"\"",
        "launcher_pgid=\"\"",
        "",
        "diagnose_job_state() {",
        "  echo \"[placecell_research] post-job diagnostics\"",
        "  if [[ -n \"${launcher_pgid:-}\" ]]; then",
        "    echo \"[placecell_research] launcher process group\"",
        (
            "    ps -o pid,ppid,pgid,sid,stat,etime,%cpu,%mem,command -g "
            "\"${launcher_pgid}\" 2>/dev/null || true"
        ),
        "  fi",
        "  echo \"[placecell_research] matching user processes\"",
        "  if command -v pgrep >/dev/null 2>&1; then",
        (
            "    pgrep -a -u \"${USER:-$(id -un)}\" -f "
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
            "\"${current_pgid}\" 2>/dev/null || true"
        ),
        "}",
        "launcher_tree_alive() {",
        "  if [[ -n \"${launcher_pgid:-}\" ]]; then",
        "    kill -0 -- \"-${launcher_pgid}\" 2>/dev/null && return 0",
        "    if command -v pgrep >/dev/null 2>&1; then",
        "      pgrep -s \"${launcher_pgid}\" >/dev/null 2>&1 && return 0",
        "    fi",
        "  fi",
        "  if [[ -n \"${launcher_pid:-}\" ]]; then",
        "    kill -0 \"${launcher_pid}\" 2>/dev/null && return 0",
        "  fi",
        "  return 1",
        "}",
        "terminate_launcher_tree() {",
        "  local signal_name=\"$1\"",
        "  if [[ -n \"${launcher_pgid:-}\" ]]; then",
        "    if command -v pkill >/dev/null 2>&1; then",
        "      pkill \"-${signal_name}\" -s \"${launcher_pgid}\" 2>/dev/null || true",
        "    fi",
        "    kill \"-${signal_name}\" -- \"-${launcher_pgid}\" 2>/dev/null || true",
        "    return",
        "  fi",
        "  if [[ -n \"${launcher_pid:-}\" ]]; then",
        "    kill \"-${signal_name}\" \"${launcher_pid}\" 2>/dev/null || true",
        "  fi",
        "}",
        "wait_for_launcher_tree() {",
        "  local timeout_seconds=\"$1\"",
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
        "  echo \"[placecell_research] forwarding SIGTERM to launcher tree\" >&2",
        "  terminate_launcher_tree TERM",
        "  if ! wait_for_launcher_tree \"${PLACECELL_TERM_GRACE_SECONDS:-90}\"; then",
        "    echo \"[placecell_research] launcher tree still alive; sending SIGKILL\" >&2",
        "    terminate_launcher_tree KILL",
        "    wait_for_launcher_tree 10 || true",
        "  fi",
        "  wait \"${launcher_pid}\" 2>/dev/null || true",
        "  exit 143",
        "}",
        "cleanup_on_exit() {",
        "  local status=$?",
        "  trap - EXIT",
        "  if launcher_tree_alive; then",
        "    echo \"[placecell_research] cleanup found live launcher tree; killing it\" >&2",
        "    terminate_launcher_tree TERM",
        "    sleep 5",
        "    terminate_launcher_tree KILL",
        "    wait_for_launcher_tree 10 || true",
        "  fi",
        "  placecell_cleanup_hpc_env || true",
        "  diagnose_job_state || true",
        "  exit \"${status}\"",
        "}",
        "trap shutdown_launcher TERM INT",
        "trap cleanup_on_exit EXIT",
        "",
        f"setsid bash --noprofile --norc -c {launcher_command_string} &",
        "launcher_pid=$!",
        "launcher_pgid=\"${launcher_pid}\"",
        "set +e",
        "wait \"${launcher_pid}\"",
        "launcher_status=$?",
        "set -e",
        "exit \"${launcher_status}\"",
        "",
    ]
    script_path.write_text("\n".join(script_lines))
    script_path.chmod(0o755)

    if config.launcher.type != "slurm":
        return str(script_path)

    completed = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or completed.stderr.strip()
