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


def _stub_spack_and_conda(directory: Path) -> Path:
    directory.mkdir()
    (directory / "spack").write_text(
        '#!/bin/bash\nif [[ "$1" == location ]]; then echo /spack/mesa-glu; exit; fi\n'
        'echo "stub spack $*"\n'
    )
    (directory / "conda").write_text(
        '#!/bin/bash\nif [[ "$1" == info ]]; then echo /spack/miniconda3; exit; fi\n'
        'echo "conda 4.10.3"\n'
    )
    for stub in directory.iterdir():
        stub.chmod(0o755)
    return directory


def test_lab_site_script_loads_spack_tools_and_keeps_the_callers_shell_options(
    tmp_path: Path,
) -> None:
    stubs = _stub_spack_and_conda(tmp_path / "stubs")
    completed = _bash(
        f"set -euo pipefail\nsource {SLURM_SCRIPTS}/site/hpc3.sh\n"
        'echo "options=$-:$(set -o | grep -c "pipefail.*on")"\n'
        'echo "GLU=${PLACECELL_EXTRA_LIBRARY_PATH}"\n',
        tmp_path,
        PATH=f"{stubs}:/usr/bin:/bin",
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    for spec in ("miniconda3@4.10.3", "git@2.31.1", "mesa-glu@9.0.1"):
        assert f"stub spack load {spec}" in output
    assert "GLU=/spack/mesa-glu/lib" in output
    assert "proxy http://rhn-proxy.rz.uos.de:3128" in output
    options = next(line for line in output.splitlines() if line.startswith("options="))
    assert "e" in options and "u" in options and options.endswith(":1")


def test_setup_script_with_a_site_takes_conda_from_it_and_never_installs_miniforge(
    tmp_path: Path,
) -> None:
    stubs = _stub_spack_and_conda(tmp_path / "stubs")
    prefix = tmp_path / "share" / "placecell"
    completed = _bash(
        f"bash {REPO_ROOT / 'scripts' / 'setup_env.sh'} --site hpc3 --dry-run --prefix {prefix}",
        tmp_path,
        PATH=f"{stubs}:/usr/bin:/bin",
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "stub spack load miniconda3@4.10.3" in output
    assert f"using {stubs}/conda (conda 4.10.3)" in output
    assert f"export TMPDIR={prefix}/setup_tmp" in output
    assert f"rm -rf {prefix}/setup_tmp" in output
    assert f"create -y -p {prefix}/envs/placecell --override-channels -c conda-forge" in output
    assert "Miniforge" not in output
    assert f'source "{SLURM_SCRIPTS}/site/hpc3.sh" && source' in output
    assert not prefix.exists()


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
