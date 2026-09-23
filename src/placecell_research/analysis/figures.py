"""Figure helpers for analysis reports."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from ..figure_style import apply_publication_style, despine, save_figure

apply_publication_style()

__all__ = [
    "apply_publication_style",
    "despine",
    "save_figure",
    "save_histogram",
    "figure_to_rgb_array",
    "save_gif",
]


def save_histogram(path: Path, values: np.ndarray, title: str, xlabel: str) -> Path:
    """Save a simple histogram figure."""
    apply_publication_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=min(30, max(5, len(values) // 2)), color="#3d6ba8", alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_to_rgb_array(figure: plt.Figure, *, close: bool = True) -> np.ndarray:
    """Render a matplotlib figure to an RGB uint8 array."""
    figure.canvas.draw()
    rgba_frame = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
    if rgba_frame.ndim == 1:
        width, height = figure.canvas.get_width_height()
        expected_size = height * width * 4
        if rgba_frame.size != expected_size:
            raise ValueError(
                "Figure canvas returned a flat RGBA buffer whose size does not match the canvas "
                "dimensions. "
                f"buffer_size={rgba_frame.size}, width={width}, height={height}"
            )
        rgba_frame = rgba_frame.reshape(height, width, 4)
    elif rgba_frame.ndim != 3 or rgba_frame.shape[-1] != 4:
        raise ValueError(f"Unexpected figure canvas buffer shape: {rgba_frame.shape}")
    frame = rgba_frame[..., :3].copy()
    if close:
        plt.close(figure)
    return frame


def save_gif(path: Path, frames: list[np.ndarray], duration: float = 0.12) -> Path:
    """Write an animated GIF from RGB frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=duration)
    return path
