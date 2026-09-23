"""Analysis-side cache wrapper over the place-cell quality kernels."""

from __future__ import annotations

import numpy as np

from ..numerics.place_cell_quality import compute_available_confound_scores
from .base import AnalysisInput


def get_or_compute_confound_scores(analysis_input: AnalysisInput) -> dict[str, np.ndarray]:
    """Return cached per-unit confound scores for one analysis source."""
    return analysis_input.get_cached_metric(
        ("confound_scores",),
        lambda: compute_available_confound_scores(
            analysis_input.representation,
            analysis_input.position_xy,
            analysis_input.valid_mask,
            kinematics=analysis_input.kinematics,
            heading=analysis_input.heading,
        ),
    )
