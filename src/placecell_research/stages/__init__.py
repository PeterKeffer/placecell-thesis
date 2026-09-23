"""Lazy stage-module access."""

from __future__ import annotations

import importlib

__all__ = [
    "analyze_model",
    "collect_dataset",
    "create_split",
    "encode_dataset",
    "evaluate_model",
    "pipeline",
    "train_place_model",
    "train_vision_encoder",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
