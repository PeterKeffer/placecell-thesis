"""Tiny CUDA runtime smoke probe for SLURM jobs."""

from __future__ import annotations

import traceback
from typing import Any


def run_cuda_smoke_check(torch_module: Any | None = None, *, matrix_size: int = 128) -> int:
    """Initialize CUDA and run one small tensor op."""
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            traceback.print_exc()
            return 91

    print(
        "[placecell_research] CUDA smoke: "
        f"torch={torch_module.__version__} "
        f"built_cuda={torch_module.version.cuda} "
        f"cudnn={torch_module.backends.cudnn.version()}",
        flush=True,
    )
    if not torch_module.cuda.is_available():
        print("[placecell_research] CUDA smoke failed: no CUDA visible", flush=True)
        return 92
    try:
        torch_module.cuda.init()
        device_index = 0
        print(
            "[placecell_research] CUDA smoke device: "
            f"{torch_module.cuda.get_device_name(device_index)} "
            f"capability={torch_module.cuda.get_device_capability(device_index)} "
            f"count={torch_module.cuda.device_count()}",
            flush=True,
        )
        x = torch_module.ones((int(matrix_size), int(matrix_size)), device="cuda")
        checksum = (x @ x).sum().item()
        if float(checksum) <= 0.0:
            print("[placecell_research] CUDA smoke failed: invalid tensor checksum", flush=True)
            return 93
    except Exception:
        traceback.print_exc()
        return 94
    print("[placecell_research] CUDA smoke: SMOKE OK", flush=True)
    return 0


def main() -> None:
    raise SystemExit(run_cuda_smoke_check())


if __name__ == "__main__":
    main()
