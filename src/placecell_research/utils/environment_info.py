"""Runtime environment capture for reproducibility manifests."""

from __future__ import annotations

import platform
import sys

import torch


def capture_environment_info() -> dict[str, object]:
    """Collect reproducibility-relevant runtime metadata."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }
