"""Personal launcher and remote values from one user file, overridden by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

USER_CONFIG_ENV_VAR = "PLACECELL_USER_CONFIG"
DEFAULT_USER_CONFIG_PATH = "~/.config/placecell/user.yaml"
LAUNCHER_ENVIRONMENT_VARIABLES = {
    "account": "PLACECELL_SLURM_ACCOUNT",
    "qos": "PLACECELL_SLURM_QOS",
    "partition": "PLACECELL_SLURM_PARTITION",
    "env_setup": "PLACECELL_ENV_SETUP",
}
FORWARDED_EXPORTS = ("PLACECELL_MESA_PREFIX", "PLACECELL_EGL_LIBRARY")
REMOTE_ENVIRONMENT_VARIABLES = {
    "host": "PLACECELL_REMOTE_HOST",
    "repo_root": "PLACECELL_REMOTE_REPO_ROOT",
    "setup": "PLACECELL_REMOTE_SETUP",
    "python": "PLACECELL_REMOTE_PYTHON",
}
SECTIONS = {"launcher", "exports", "remote"}


@dataclass(frozen=True, slots=True)
class RemoteSettings:
    host: str = ""
    repo_root: str = ""
    setup: str = ""
    python: str = "python"
    ssh_options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UserSettings:
    path: Path
    launcher: dict[str, str] = field(default_factory=dict)
    exports: dict[str, str] = field(default_factory=dict)
    remote: RemoteSettings = field(default_factory=RemoteSettings)


def user_config_path() -> Path:
    return Path(os.environ.get(USER_CONFIG_ENV_VAR, "") or DEFAULT_USER_CONFIG_PATH).expanduser()


def _mapping(payload: dict, section: str, path: Path) -> dict:
    value = payload.get(section) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: `{section}` must be a mapping.")
    return value


def load_user_settings() -> UserSettings:
    """Read the user file (if any), then let PLACECELL_* environment variables win."""
    path = user_config_path()
    payload = yaml.safe_load(path.read_text()) if path.is_file() else {}
    payload = payload or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    unknown = sorted(set(payload) - SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown sections {unknown}; expected {sorted(SECTIONS)}.")
    launcher_payload = _mapping(payload, "launcher", path)
    unknown_launcher = sorted(set(launcher_payload) - set(LAUNCHER_ENVIRONMENT_VARIABLES))
    if unknown_launcher:
        raise ValueError(
            f"{path}: unknown launcher keys {unknown_launcher}; "
            f"expected {sorted(LAUNCHER_ENVIRONMENT_VARIABLES)}."
        )
    launcher = {key: str(value) for key, value in launcher_payload.items() if value}
    for key, variable in LAUNCHER_ENVIRONMENT_VARIABLES.items():
        if os.environ.get(variable):
            launcher[key] = os.environ[variable]
    exports = {
        str(key): str(value)
        for key, value in _mapping(payload, "exports", path).items()
        if value not in (None, "")
    }
    for variable in FORWARDED_EXPORTS:
        if os.environ.get(variable):
            exports[variable] = os.environ[variable]
    remote_payload = _mapping(payload, "remote", path)
    unknown_remote = sorted(set(remote_payload) - {*REMOTE_ENVIRONMENT_VARIABLES, "ssh_options"})
    if unknown_remote:
        raise ValueError(f"{path}: unknown remote keys {unknown_remote}.")
    remote_values = {
        key: str(remote_payload.get(key) or "") for key in REMOTE_ENVIRONMENT_VARIABLES
    }
    for key, variable in REMOTE_ENVIRONMENT_VARIABLES.items():
        if os.environ.get(variable):
            remote_values[key] = os.environ[variable]
    ssh_options = remote_payload.get("ssh_options") or []
    if not isinstance(ssh_options, list):
        raise ValueError(f"{path}: `remote.ssh_options` must be a list.")
    remote = RemoteSettings(
        host=remote_values["host"],
        repo_root=remote_values["repo_root"],
        setup=remote_values["setup"],
        python=remote_values["python"] or "python",
        ssh_options=tuple(str(option) for option in ssh_options),
    )
    return UserSettings(path=path, launcher=launcher, exports=exports, remote=remote)


def user_launcher_overrides(settings: UserSettings | None = None) -> list[str]:
    """The user's launcher values as `-o launcher.*` overrides."""
    resolved = settings or load_user_settings()
    overrides = [f"launcher.{key}={value}" for key, value in resolved.launcher.items()]
    overrides.extend(f"launcher.exports.{key}={value}" for key, value in resolved.exports.items())
    return overrides
