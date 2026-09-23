"""PlaceCell Research package."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ExperimentConfig":
        from .config.schema import ExperimentConfig

        return ExperimentConfig
    if name == "apply_publication_style":
        from .figure_style import apply_publication_style

        return apply_publication_style
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ExperimentConfig", "apply_publication_style"]
