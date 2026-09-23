"""PlaceCell Research package."""

from __future__ import annotations

import os
from typing import Any

_CODE_SNAPSHOT_ROOT = os.environ.get("PLACECELL_CODE_SNAPSHOT", "").strip()
if _CODE_SNAPSHOT_ROOT:
    _EXPECTED_PACKAGE_PREFIX = os.path.join(_CODE_SNAPSHOT_ROOT, "src", "")
    if not os.path.abspath(__file__).startswith(_EXPECTED_PACKAGE_PREFIX):
        raise ImportError(
            f"PLACECELL_CODE_SNAPSHOT={_CODE_SNAPSHOT_ROOT} but placecell_research was "
            f"imported from {os.path.abspath(__file__)}. The job would run code the snapshot "
            "does not pin. Check that PYTHONPATH still leads with the snapshot's src/."
        )


def __getattr__(name: str) -> Any:
    if name == "ExperimentConfig":
        from .config.schema import ExperimentConfig

        return ExperimentConfig
    if name == "apply_publication_style":
        from .figure_style import apply_publication_style

        return apply_publication_style
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ExperimentConfig", "apply_publication_style"]
