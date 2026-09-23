from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from placecell_research.launch import submit as submit_module
from placecell_research.launch.submit import default_environment_activation, submit_cli_entrypoint

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "experiment" / "smoke_wallgap.yaml"
MUSEUM_CONFIG = REPO_ROOT / "configs" / "experiment" / "smoke_museum.yaml"
NAVIGATION_CONFIG = REPO_ROOT / "configs" / "thesis" / "navigation" / "ppo_place_code_north.yaml"


class _SubprocessProxy:
    def __init__(self, run) -> None:
        self.run = run


def _script(tmp_path: Path, overrides: list[str], **kwargs) -> str:
    path = submit_cli_entrypoint(
        kwargs.pop("config", SMOKE_CONFIG),
        [f"tracking.run_root={tmp_path / 'runs'}", *overrides],
        dry_run=True,
        **kwargs,
    )
    return Path(path).read_text()


def test_generic_profile_writes_a_hardened_script(tmp_path: Path) -> None:
    text = _script(
        tmp_path,
        [
            "launcher.partition=gpu_long",
            "launcher.cpus_per_task=6",
            "launcher.memory_gb=24",
            "launcher.time_hours=12",
            "seed.global_seed=123",
        ],
    )
    for line in (
        "#SBATCH --partition=gpu_long",
        "#SBATCH --job-name=placecell_research",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=24G",
        "#SBATCH --time=12:00:00",
        "#SBATCH --signal=B:TERM@120",
        "export OMP_NUM_THREADS=1",
        "export PLACECELL_REQUESTED_GPUS=1",
        "export PLACECELL_ENVIRONMENT_KIND=miniworld",
        'PLACECELL_SLURM_SCRIPT_ROOT="${PLACECELL_CODE_SNAPSHOT:-.}/scripts/slurm"',
        'source "${PLACECELL_SLURM_SCRIPT_ROOT}/env_miniworld.sh"',
        "trap shutdown_launcher TERM INT",
        "trap cleanup_on_exit EXIT",
        "python -u -m placecell_research.launch.cli pipeline",
        "-o seed.global_seed=123",
    ):
        assert line in text
    for absent in ("--account", "--qos", "--constraint", "--dependency", "--exclude", "site/"):
        assert absent not in text
    assert text.index('"${PLACECELL_SLURM_SCRIPT_ROOT}/env_common.sh"') < text.index(
        default_environment_activation()
    )
    assert text.index(default_environment_activation()) < text.index("env_miniworld.sh")
    assert text.index("trap cleanup_on_exit EXIT") < text.index("setsid bash --noprofile --norc")


def test_lab_profile_pins_a_full_gpu_excludes_klab7_and_sources_the_site_script(
    tmp_path: Path,
) -> None:
    text = _script(tmp_path, ["launcher=hpc3"])
    assert "#SBATCH --partition=klab-gpu" in text
    assert "#SBATCH --exclude=klab-7" in text
    assert "#SBATCH --gres=gpu:H100.80gb:1" in text
    assert "#SBATCH --mem=200G" in text
    assert 'source "${PLACECELL_SLURM_SCRIPT_ROOT}/site/hpc3.sh"' in text
    assert text.index("site/hpc3.sh") < text.index("env_miniworld.sh")


@pytest.mark.parametrize("gpu_type", ["null", "H100.10gb", "h100.10gb"])
def test_lab_profile_refuses_gpu_requests_that_may_land_on_a_mig_slice(
    tmp_path: Path, gpu_type: str
) -> None:
    with pytest.raises(ValueError, match="MIG"):
        _script(tmp_path, ["launcher=hpc3", f"launcher.gpu_type={gpu_type}"])


def test_lab_profile_allows_cpu_jobs_without_a_gpu_type(tmp_path: Path) -> None:
    text = _script(tmp_path, ["launcher=hpc3", "launcher.gpus=0", "launcher.gpu_type=null"])
    assert "--gres" not in text


def test_user_file_fills_account_qos_and_exports_and_the_command_line_wins(
    tmp_path: Path, monkeypatch
) -> None:
    user_file = tmp_path / "user.yaml"
    user_file.write_text(
        "launcher:\n  account: 12345\n  qos: normal\n  env_setup: source /opt/env/bin/activate\n"
        "exports:\n  PLACECELL_MESA_PREFIX: /opt/mesa\n"
    )
    monkeypatch.setenv("PLACECELL_USER_CONFIG", str(user_file))
    text = _script(tmp_path, ["launcher.qos=long"])
    assert "#SBATCH --account=12345" in text
    assert "#SBATCH --qos=long" in text
    assert "source /opt/env/bin/activate" in text
    assert "export PLACECELL_MESA_PREFIX=/opt/mesa" in text


def test_environment_variables_override_the_user_file(tmp_path: Path, monkeypatch) -> None:
    user_file = tmp_path / "user.yaml"
    user_file.write_text("launcher:\n  account: from_file\n")
    monkeypatch.setenv("PLACECELL_USER_CONFIG", str(user_file))
    monkeypatch.setenv("PLACECELL_SLURM_ACCOUNT", "from_environment")
    assert "#SBATCH --account=from_environment" in _script(tmp_path, [])


def test_dependency_adds_kill_on_invalid_dependency(tmp_path: Path) -> None:
    text = _script(tmp_path, [], slurm_dependency="afterok:12:13,afterany:14")
    assert "#SBATCH --dependency=afterok:12:13,afterany:14" in text
    assert "#SBATCH --kill-on-invalid-dep=yes" in text


def test_unsafe_job_names_and_dependencies_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SLURM job name"):
        _script(tmp_path, [], slurm_job_name="unsafe name\n#SBATCH --exclusive")
    with pytest.raises(ValueError, match="SLURM dependency"):
        _script(tmp_path, [], slurm_dependency="afterok:1 --exclusive")


def test_museum_and_navigation_configs_pick_their_environment_scripts(tmp_path: Path) -> None:
    assert "env_jaxenstein.sh" in _script(tmp_path, [], config=MUSEUM_CONFIG)
    navigation = _script(
        tmp_path,
        ["models.place_model_artifact_id=place_model_demo"],
        config=NAVIGATION_CONFIG,
        entrypoint="downstream-train",
    )
    assert "python -u -m placecell_research.launch.cli downstream-train" in navigation
    assert "env_miniworld.sh" in navigation


def test_snapshot_exports_only_when_the_submitting_process_runs_from_one(
    tmp_path: Path, monkeypatch
) -> None:
    assert "export PLACECELL_CODE_SNAPSHOT" not in _script(tmp_path, [])
    snapshot = str(REPO_ROOT)
    monkeypatch.setenv("PLACECELL_CODE_SNAPSHOT", snapshot)
    text = _script(tmp_path, [])
    assert f"export PLACECELL_CODE_SNAPSHOT={snapshot}" in text
    assert f"export PYTHONPATH={snapshot}/src" in text


def test_env_script_resolution_prefers_the_snapshot_and_falls_back_to_the_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "scripts" / "slurm").mkdir(parents=True)
    (checkout / "scripts" / "slurm" / "env_miniworld.sh").write_text("ORIGIN=live_checkout\n")
    snapshot = tmp_path / ".code_snapshots" / "20260831T000000Z_ab12cd34"
    (snapshot / "scripts" / "slurm").mkdir(parents=True)
    (snapshot / "scripts" / "slurm" / "env_miniworld.sh").write_text("ORIGIN=snapshot\n")
    source_lines = (
        'PLACECELL_SLURM_SCRIPT_ROOT="${PLACECELL_CODE_SNAPSHOT:-.}/scripts/slurm"\n'
        'source "${PLACECELL_SLURM_SCRIPT_ROOT}/env_miniworld.sh"\n'
        'echo "${ORIGIN}"'
    )

    def origin(code_snapshot: str | None) -> str:
        environment = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
        if code_snapshot is not None:
            environment["PLACECELL_CODE_SNAPSHOT"] = code_snapshot
        completed = subprocess.run(
            ["bash", "-c", f"cd {checkout}\n{source_lines}"],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        return completed.stdout.strip()

    assert origin(str(snapshot)) == "snapshot"
    assert origin(None) == "live_checkout"


def test_analysis_workers_are_exported_only_when_configured(tmp_path: Path) -> None:
    assert "PLACECELL_ANALYSIS_WORKERS" not in _script(tmp_path, [])
    assert "export PLACECELL_ANALYSIS_WORKERS=8" in _script(
        tmp_path, ["launcher.analysis_workers=8"]
    )


def test_sbatch_runs_only_for_a_real_slurm_submission(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command[0] != "sbatch":
            return real_run(command, **kwargs)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Submitted batch job 42", stderr="")

    monkeypatch.setattr(submit_module, "subprocess", _SubprocessProxy(fake_run))
    overrides = [f"tracking.run_root={tmp_path / 'runs'}"]
    assert submit_cli_entrypoint(SMOKE_CONFIG, overrides) == "Submitted batch job 42"
    assert len(calls) == 1
    calls.clear()
    submit_cli_entrypoint(SMOKE_CONFIG, overrides, dry_run=True)
    submit_cli_entrypoint(SMOKE_CONFIG, [*overrides, "launcher.type=local"])
    assert calls == []


def test_default_activation_uses_conda_for_a_conda_prefix(tmp_path: Path) -> None:
    base = tmp_path / "miniforge3"
    environment = base / "envs" / "placecell"
    (environment / "conda-meta").mkdir(parents=True)
    (base / "etc" / "profile.d").mkdir(parents=True)
    (base / "etc" / "profile.d" / "conda.sh").write_text("")
    line = default_environment_activation(environment)
    assert line == f"source {base}/etc/profile.d/conda.sh && conda activate {environment}"
    virtual_environment = tmp_path / "venv"
    (virtual_environment / "bin").mkdir(parents=True)
    (virtual_environment / "bin" / "activate").write_text("")
    assert default_environment_activation(virtual_environment) == (
        f"source {virtual_environment}/bin/activate"
    )
