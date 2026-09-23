"""Uniform, publication-friendly matplotlib defaults and small figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    """Set uniform, publication-friendly defaults once for every figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def despine(axis: plt.Axes) -> None:
    """Hide the top and right spines (modern look for line/scatter plots)."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int = 300,
    vector: bool = False,
    bbox_inches: str | None = "tight",
) -> Path:
    """Save figure as PNG, plus a vector PDF sibling when vector=True."""
    apply_publication_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if vector:
        figure.savefig(path.with_suffix(".pdf"), bbox_inches=bbox_inches)
    return path
