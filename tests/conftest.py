from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from placecell_research.config.schema import AnalysisConfig, ExperimentConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every figure at the test boundary."""
    yield
    plt.close("all")


THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@pytest.fixture(autouse=True)
def _restore_thread_limit_environment():
    """Restore the BLAS/OpenMP thread-limit variables at the test boundary."""
    saved = {name: os.environ.get(name) for name in THREAD_LIMIT_ENV_VARS}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


USER_SETTING_ENV_VARS = (
    "PLACECELL_SLURM_ACCOUNT",
    "PLACECELL_SLURM_QOS",
    "PLACECELL_SLURM_PARTITION",
    "PLACECELL_ENV_SETUP",
    "PLACECELL_MESA_PREFIX",
    "PLACECELL_EGL_LIBRARY",
    "PLACECELL_REMOTE_HOST",
    "PLACECELL_REMOTE_REPO_ROOT",
    "PLACECELL_REMOTE_SETUP",
    "PLACECELL_REMOTE_PYTHON",
    "PLACECELL_CODE_SNAPSHOT",
)


@pytest.fixture(autouse=True)
def _isolate_user_settings(monkeypatch, tmp_path_factory):
    """Keep the developer's own user file and PLACECELL_* variables out of every test."""
    user_file = tmp_path_factory.mktemp("user_settings") / "user.yaml"
    monkeypatch.setenv("PLACECELL_USER_CONFIG", str(user_file))
    for name in USER_SETTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def analysis_settings():
    """Build the analysis section the analyze stage hands to modules: the schema plus overrides."""

    def build(**overrides) -> dict:
        return ExperimentConfig(analysis=AnalysisConfig(**overrides)).to_dict()["analysis"]

    return build
