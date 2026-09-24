"""The pc doctor command: check Python, devices, rendering, write access and the user file."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import typer

RENDER_PROBE = """
from placecell_research.envs.miniworld_adapter import MiniWorldAdapter
adapter = MiniWorldAdapter(env_id="MiniWorld-WallGapAsymLarge-v0", seed=0, episode_length=8)
frame = adapter.reset(seed=0).rgb
adapter.close()
print("WallGap frame", tuple(frame.shape))
"""
JAXENSTEIN_PROBE = """
from placecell_research.envs.jaxenstein_adapter import JaxensteinAdapter
adapter = JaxensteinAdapter(env_id="museum-gallery", seed=0, episode_length=8, env_kwargs={})
frame = adapter.reset(seed=0).rgb
adapter.close()
print("museum frame", tuple(frame.shape))
"""


class _Report:
    def __init__(self) -> None:
        self.failures = 0

    def line(self, status: str, name: str, detail: str) -> None:
        if status == "FAIL":
            self.failures += 1
        typer.echo(f"{status:<4}  {name:<14} {detail}")


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", None) or getattr(module, "version", "installed"))


def _probe(code: str, timeout_seconds: int) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_seconds} s"
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return completed.returncode == 0, output[-1] if output else f"exit {completed.returncode}"


def _check_python(report: _Report) -> None:
    import placecell_research

    version = platform.python_version()
    status = "OK" if sys.version_info >= (3, 11) else "FAIL"
    report.line(status, "python", f"{version} at {sys.executable}")
    report.line("OK", "package", str(Path(placecell_research.__file__).parent))
    snapshot = os.environ.get("PLACECELL_CODE_SNAPSHOT", "")
    if snapshot:
        report.line("OK", "snapshot", snapshot)


def _check_torch(report: _Report) -> None:
    try:
        import torch
    except Exception as exc:
        report.line("FAIL", "torch", f"not importable: {exc}")
        return
    devices = ["cpu"]
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        devices.append(f"cuda x{len(names)} ({', '.join(names)})")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    from placecell_research.utils.device import resolve_device

    device = resolve_device("auto")
    value = float((torch.ones(4, 4, device=device) @ torch.ones(4, 4, device=device)).sum())
    status = "OK" if value == 64.0 else "FAIL"
    detail = f"{torch.__version__}; devices {', '.join(devices)}; auto={device}"
    report.line(status, "torch", detail)


class _StartupErrors(logging.Handler):
    """Keeps JAX start-up errors as one line each instead of printing their tracebacks."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        error = record.exc_info[1] if record.exc_info else None
        text = str(error) if error is not None else record.getMessage()
        self.messages.append(" ".join(text.split())[:200])


def _jax_devices() -> tuple[Any, list[Any], list[str]]:
    """Import jax and list its devices; on a node without a GPU the CUDA plugin fails loudly."""
    jax_logger = logging.getLogger("jax")
    errors = _StartupErrors()
    propagate = jax_logger.propagate
    jax_logger.addHandler(errors)
    jax_logger.propagate = False
    try:
        import jax

        return jax, jax.devices(), errors.messages
    finally:
        jax_logger.removeHandler(errors)
        jax_logger.propagate = propagate


def _check_jax(report: _Report, render: bool) -> None:
    try:
        jax, devices, startup_errors = _jax_devices()
    except ImportError:
        report.line("WARN", "jax", "not installed (needed for the museum; install the jax extra)")
        return
    except Exception as exc:
        report.line("FAIL", "jax", f"{type(exc).__name__}: {exc}")
        return
    import torch

    if any(device.platform == "gpu" for device in devices):
        report.line("OK", "jax", f"{jax.__version__}; devices {devices}")
    elif not torch.cuda.is_available():
        report.line("OK", "jax", f"{jax.__version__}; no GPU visible, so jax runs on CPU")
    else:
        reason = startup_errors[0] if startup_errors else "no CUDA plugin installed"
        report.line("WARN", "jax", f"{jax.__version__} on CPU although a GPU is visible: {reason}")
    if _module_version("jaxenstein") is None:
        report.line("FAIL", "jaxenstein", "not importable")
        return
    if render:
        ok, detail = _probe(JAXENSTEIN_PROBE, 300)
        report.line("OK" if ok else "FAIL", "museum", detail)


def _check_miniworld(report: _Report, render: bool) -> None:
    version = _module_version("gymnasium")
    report.line("OK" if version else "FAIL", "gymnasium", version or "not importable")
    pyglet_version = _module_version("pyglet")
    report.line("OK" if pyglet_version else "FAIL", "pyglet", pyglet_version or "not importable")
    if sys.platform.startswith("linux"):
        egl = [
            path
            for path in ("/usr/lib64/libEGL.so.1", "/usr/lib/x86_64-linux-gnu/libEGL.so.1")
            if Path(path).is_file()
        ]
        detail = egl[0] if egl else "no system libEGL.so.1; set PLACECELL_EGL_LIBRARY"
        report.line("OK" if egl else "WARN", "egl", detail)
        vendor = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if not Path(vendor).is_file():
            report.line("WARN", "egl vendor", f"{vendor} missing (normal on login/CPU nodes)")
    if not render:
        report.line("WARN", "miniworld", "rendering not checked (--no-render)")
        return
    ok, detail = _probe(RENDER_PROBE, 180)
    hint = ""
    if not ok and sys.platform == "darwin":
        hint = "; on macOS keep the display awake (caffeinate -d -i)"
    elif not ok:
        hint = "; on a cluster run pc doctor inside a GPU job, where EGL is set up"
    report.line("OK" if ok else "FAIL", "miniworld", detail + hint)


def _check_optional(report: _Report) -> None:
    version = _module_version("stable_baselines3")
    report.line("OK" if version else "WARN", "sb3", version or "missing (navigation needs rl)")


def _check_write_access(report: _Report) -> None:
    from placecell_research.utils.repo_paths import find_repo_root_from_path

    try:
        repo_root = find_repo_root_from_path(Path.cwd())
    except FileNotFoundError:
        report.line("FAIL", "repo", f"run pc doctor inside the repository, not {Path.cwd()}")
        return
    for folder in ("artifacts", "runs"):
        target = repo_root / folder
        try:
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target):
                pass
            report.line("OK", folder, f"writable: {target}")
        except OSError as exc:
            report.line("FAIL", folder, f"not writable: {target} ({exc})")
    free_gb = shutil.disk_usage(repo_root).free / 1e9
    report.line("OK" if free_gb > 50 else "WARN", "disk", f"{free_gb:.0f} GB free at {repo_root}")
    try:
        with tempfile.NamedTemporaryFile():
            pass
        report.line("OK", "tmp", f"writable: {tempfile.gettempdir()}")
    except OSError as exc:
        report.line("FAIL", "tmp", str(exc))


def _check_user_settings(report: _Report) -> None:
    from placecell_research.launch.user_settings import load_user_settings, user_config_path

    try:
        settings = load_user_settings()
    except ValueError as exc:
        report.line("FAIL", "user file", str(exc))
        return
    exists = user_config_path().is_file()
    report.line("OK", "user file", f"{settings.path}" + ("" if exists else " (not created yet)"))
    for key, value in settings.launcher.items():
        report.line("OK", f"launcher.{key}", value)
    if settings.remote.host:
        report.line("OK", "remote", f"{settings.remote.host}:{settings.remote.repo_root}")
    scheduler = shutil.which("sbatch")
    report.line("OK", "slurm", scheduler or "no sbatch here (fine on a PC)")


def register(app: typer.Typer) -> None:
    @app.command("doctor")
    def doctor_command(
        render: bool = typer.Option(
            True, "--render/--no-render", help="Render one MiniWorld and one museum frame."
        ),
    ) -> None:
        """Check that this machine can run the pipeline; exit 1 if anything essential fails."""
        report = _Report()
        _check_python(report)
        _check_torch(report)
        _check_jax(report, render)
        _check_miniworld(report, render)
        _check_optional(report)
        _check_write_access(report)
        _check_user_settings(report)
        summary = f"{report.failures} check(s) failed" if report.failures else "all checks passed"
        typer.echo(f"doctor: {summary}")
        raise typer.Exit(1 if report.failures else 0)
