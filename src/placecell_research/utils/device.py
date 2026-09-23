"""Device helpers."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve auto, cpu, cuda, or mps to a real torch device."""
    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)
