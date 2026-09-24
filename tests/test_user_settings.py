from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from placecell_research.launch.commands import remote, reproduce
from placecell_research.launch.remote_entrypoints import default_launcher_overrides
from placecell_research.launch.remote_run import resolve_remote_settings
from placecell_research.launch.user_settings import load_user_settings, user_launcher_overrides


def _write(tmp_path: Path, monkeypatch, text: str) -> Path:
    path = tmp_path / "user.yaml"
    path.write_text(text)
    monkeypatch.setenv("PLACECELL_USER_CONFIG", str(path))
    return path


def test_missing_user_file_means_no_personal_values() -> None:
    settings = load_user_settings()
    assert settings.launcher == {} and settings.exports == {}
    assert settings.remote.host == "" and settings.remote.python == "python"
    assert user_launcher_overrides(settings) == []


def test_user_file_values_become_launcher_overrides(tmp_path: Path, monkeypatch) -> None:
    _write(
        tmp_path,
        monkeypatch,
        "launcher:\n  account: lab\n  qos: ''\nexports:\n  PLACECELL_MESA_PREFIX: /opt/mesa\n",
    )
    assert user_launcher_overrides() == [
        "launcher.account=lab",
        "launcher.exports.PLACECELL_MESA_PREFIX=/opt/mesa",
    ]


def test_environment_variables_win_over_the_user_file(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "remote:\n  host: file-host\n  repo_root: /file/root\n")
    monkeypatch.setenv("PLACECELL_REMOTE_HOST", "env-host")
    monkeypatch.setenv("PLACECELL_MESA_PREFIX", "/env/mesa")
    settings = load_user_settings()
    assert settings.remote.host == "env-host"
    assert settings.remote.repo_root == "/file/root"
    assert settings.exports == {"PLACECELL_MESA_PREFIX": "/env/mesa"}


@pytest.mark.parametrize(
    "text",
    ["cluster:\n  host: x\n", "launcher:\n  partitoin: gpu\n", "remote:\n  hots: x\n"],
)
def test_unknown_keys_fail_loudly(tmp_path: Path, monkeypatch, text: str) -> None:
    _write(tmp_path, monkeypatch, text)
    with pytest.raises(ValueError, match="unknown"):
        load_user_settings()


def test_override_order_is_profile_then_user_file_then_command_line(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path, monkeypatch, "launcher:\n  partition: user_partition\n")
    assert default_launcher_overrides(["launcher.partition=cli", "seed=1"], "slurm") == [
        "launcher=slurm",
        "launcher.partition=user_partition",
        "launcher.partition=cli",
        "seed=1",
    ]
    assert default_launcher_overrides(["seed=1", "launcher=hpc3"], "slurm")[:2] == [
        "launcher=hpc3",
        "launcher.partition=user_partition",
    ]


def test_remote_settings_need_a_host_and_a_checkout(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="remote host and repo_root"):
        resolve_remote_settings()
    _write(tmp_path, monkeypatch, "remote:\n  host: login\n  repo_root: /data/checkout\n")
    settings = resolve_remote_settings(remote_python="/env/bin/python")
    assert (settings.host, settings.repo_root, settings.python) == (
        "login",
        "/data/checkout",
        "/env/bin/python",
    )


@pytest.mark.parametrize(
    ("arguments", "has_remote_flags"),
    [
        (["reproduce", "--profile", "hpc3", "--remote"], False),
        (["hpc", "--config", "configs/thesis/baseline.yaml"], True),
        (["hpc-logs"], True),
        (["remote-sync"], True),
    ],
)
def test_remote_commands_without_settings_print_one_error_line(
    arguments: list[str], has_remote_flags: bool
) -> None:
    app = typer.Typer()
    remote.register(app)
    reproduce.register(app)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1
    assert result.output.count("\n") == 1
    assert str(load_user_settings().path) in result.output
    assert "PLACECELL_REMOTE_HOST" in result.output
    assert ("--remote-host" in result.output) == has_remote_flags
