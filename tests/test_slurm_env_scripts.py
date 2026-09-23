from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_SCRIPTS = REPO_ROOT / "scripts" / "slurm"
SHELL_SCRIPTS = sorted([*SLURM_SCRIPTS.rglob("*.sh"), REPO_ROOT / "scripts" / "setup_env.sh"])


def _bash(script: str, cwd: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(cwd),
        "SLURM_JOB_ID": "4242",
        **environment,
    }
    return subprocess.run(
        ["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_are_valid_bash(script: Path) -> None:
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_common_env_uses_a_short_tmpdir_link_is_idempotent_and_cleans_up(tmp_path: Path) -> None:
    completed = _bash(
        f'source {SLURM_SCRIPTS}/env_common.sh\nfirst="${{PLACECELL_TMP_ROOT}}"\n'
        f"source {SLURM_SCRIPTS}/env_common.sh\n"
        '[[ "${first}" == "${PLACECELL_TMP_ROOT}" ]] || exit 3\n'
        'echo "root=${PLACECELL_TMP_ROOT} link=${PLACECELL_TMP_LINK} tmp=${TMPDIR}"\n'
        "placecell_cleanup_hpc_env\n"
        '[[ ! -e "${PLACECELL_TMP_ROOT}" && ! -e "${PLACECELL_TMP_LINK}" ]] || exit 4\n',
        tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    fields = dict(item.split("=", 1) for item in completed.stdout.split("\n")[-2].split())
    assert fields["root"].startswith(str(tmp_path / "runs" / "tmp" / "placecell_4242_"))
    assert fields["tmp"] == fields["link"]
    assert len(fields["link"]) < 40


def test_miniworld_gpu_env_links_egl_first_and_checks_textures(tmp_path: Path) -> None:
    egl_library = tmp_path / "lib64" / "libEGL.so.1"
    egl_library.parent.mkdir()
    egl_library.write_text("")
    completed = _bash(
        f"source {SLURM_SCRIPTS}/env_miniworld.sh\n"
        'echo "LD=${LD_LIBRARY_PATH}"\necho "VENDOR=${__EGL_VENDOR_LIBRARY_FILENAMES}"\n'
        "placecell_cleanup_hpc_env\n",
        tmp_path,
        PLACECELL_REQUESTED_GPUS="1",
        PLACECELL_EGL_LIBRARY=str(egl_library),
        PLACECELL_EXTRA_LIBRARY_PATH="/opt/glu/lib",
    )
    assert completed.returncode == 0, completed.stderr
    assert "MiniWorld textures OK" in completed.stdout
    library_path = next(line for line in completed.stdout.splitlines() if line.startswith("LD="))
    entries = library_path.removeprefix("LD=").split(":")
    assert entries[0].endswith("/egl_link") and entries[1] == str(egl_library.parent)
    assert entries[2] == "/opt/glu/lib"
    assert "VENDOR=/usr/share/glvnd/egl_vendor.d/10_nvidia.json" in completed.stdout


def test_miniworld_cpu_env_fails_closed_without_software_mesa(tmp_path: Path) -> None:
    completed = _bash(
        f"source {SLURM_SCRIPTS}/env_miniworld.sh", tmp_path, PLACECELL_REQUESTED_GPUS="0"
    )
    assert completed.returncode == 1
    assert "needs software Mesa; set PLACECELL_MESA_PREFIX" in completed.stderr


def test_miniworld_gpu_env_fails_closed_without_libegl(tmp_path: Path) -> None:
    completed = _bash(
        f"source {SLURM_SCRIPTS}/env_miniworld.sh",
        tmp_path,
        PLACECELL_REQUESTED_GPUS="1",
        PLACECELL_EGL_LIBRARY=str(tmp_path / "missing" / "libEGL.so.1"),
    )
    assert completed.returncode == 1
    assert "libEGL.so.1 not found" in completed.stderr


def test_lab_site_script_only_touches_miniworld_gpu_jobs(tmp_path: Path) -> None:
    probe = f'source {SLURM_SCRIPTS}/site/hpc3.sh\necho "EGL=${{PLACECELL_EGL_LIBRARY:-unset}}"'
    jax_job = _bash(
        f"set -euo pipefail\n{probe}",
        tmp_path,
        PLACECELL_ENVIRONMENT_KIND="jaxenstein",
        PLACECELL_REQUESTED_GPUS="1",
    )
    assert jax_job.returncode == 0 and "EGL=unset" in jax_job.stdout
    miniworld_job = _bash(
        f"set -euo pipefail\n{probe}",
        tmp_path,
        PLACECELL_ENVIRONMENT_KIND="miniworld",
        PLACECELL_REQUESTED_GPUS="1",
        PLACECELL_EXTRA_LIBRARY_PATH="/opt/glu/lib",
    )
    assert miniworld_job.returncode == 0, miniworld_job.stderr
    assert "EGL=/usr/lib64/libEGL.so.1" in miniworld_job.stdout
    assert "libGLU from /opt/glu/lib" in miniworld_job.stdout


def test_setup_script_dry_run_prints_every_step_and_runs_none(tmp_path: Path) -> None:
    prefix = tmp_path / "miniforge"
    completed = _bash(
        f"bash {REPO_ROOT / 'scripts' / 'setup_env.sh'} --dry-run --prefix {prefix}",
        tmp_path,
        PATH="/usr/bin:/bin",
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "installing Miniforge" in output and "Miniforge3-" in output
    for fragment in ("create -y -p", "pip install torch", "pip install -e", "doctor"):
        assert fragment in output
    assert not prefix.exists()
