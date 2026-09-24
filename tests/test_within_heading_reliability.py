"""Within-heading reliability: does conditioning on heading recover reliability?"""

from __future__ import annotations

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.within_heading_reliability import (
    WithinHeadingReliabilityModule,
    compute_within_heading_reliability,
)

_QUADRANT_CENTERS = (np.arange(4) + 0.5) * (np.pi / 2.0)


def _single_bin_episodes(num_episodes: int = 40):
    """Episodes that all sit in one spatial bin, heading cycling through quadrants."""
    quadrant_per_episode = np.arange(num_episodes) % 4
    heading = _QUADRANT_CENTERS[quadrant_per_episode].reshape(num_episodes, 1)
    position_xy = np.zeros((num_episodes, 1, 2), dtype=np.float64)
    valid_mask = np.ones((num_episodes, 1), dtype=bool)
    return quadrant_per_episode, heading, position_xy, valid_mask


def _make(representation_per_unit, heading, position_xy, valid_mask):
    representation = np.stack(representation_per_unit, axis=-1).astype(np.float64)
    return compute_within_heading_reliability(
        representation=representation,
        position_xy=position_xy,
        heading=heading,
        valid_mask=valid_mask,
        num_bins_x=1,
        num_bins_y=1,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        threshold_fraction=0.3,
        num_heading_bins=4,
        min_steps_per_cell=5,
    )


def test_conjunctive_unit_has_large_reliability_gain():
    quadrant, heading, position_xy, valid_mask = _single_bin_episodes()
    conjunctive = (quadrant == 0).astype(np.float64).reshape(-1, 1)
    result = _make([conjunctive], heading, position_xy, valid_mask)
    assert np.isclose(result.pooled[0], 0.25, atol=0.02)
    assert np.isclose(result.within_heading[0], 1.0, atol=0.02)
    assert result.gain[0] > 0.5


def test_omnidirectional_unit_has_no_reliability_gain():
    _quadrant, heading, position_xy, valid_mask = _single_bin_episodes()
    omnidirectional = np.ones((heading.shape[0], 1), dtype=np.float64)
    result = _make([omnidirectional], heading, position_xy, valid_mask)
    assert np.isclose(result.pooled[0], 1.0, atol=0.02)
    assert np.isclose(result.within_heading[0], 1.0, atol=0.02)
    assert abs(result.gain[0]) < 0.05


def test_silent_quadrant_is_not_spuriously_reliable():
    quadrant, heading, position_xy, valid_mask = _single_bin_episodes()
    conjunctive = (quadrant == 0).astype(np.float64).reshape(-1, 1)
    result = _make([conjunctive], heading, position_xy, valid_mask)
    assert result.pooled[0] < 0.4


def test_dead_unit_is_not_assessable():
    _quadrant, heading, position_xy, valid_mask = _single_bin_episodes()
    dead = np.zeros((heading.shape[0], 1), dtype=np.float64)
    result = _make([dead], heading, position_xy, valid_mask)
    assert np.isnan(result.pooled[0])
    assert np.isnan(result.within_heading[0])
    assert np.isnan(result.gain[0])


def test_population_separates_conjunctive_from_omnidirectional():
    quadrant, heading, position_xy, valid_mask = _single_bin_episodes()
    conjunctive = (quadrant == 0).astype(np.float64).reshape(-1, 1)
    omnidirectional = np.ones((heading.shape[0], 1), dtype=np.float64)
    result = _make([conjunctive, omnidirectional], heading, position_xy, valid_mask)
    assert result.gain[0] > result.gain[1]
    assert np.nanmean(result.within_heading) > np.nanmean(result.pooled)


def test_module_run_reports_metrics_and_table(tmp_path, analysis_settings):
    quadrant, heading, _position, valid_mask = _single_bin_episodes()
    num_episodes = heading.shape[0]
    conjunctive = (quadrant == 0).astype(np.float64)
    omnidirectional = np.ones(num_episodes, dtype=np.float64)
    representation = np.stack([conjunctive, omnidirectional], axis=-1).reshape(num_episodes, 1, 2)
    corner = np.where(np.arange(num_episodes) % 2 == 0, -0.5, 0.5).astype(np.float64)
    position_xy = np.stack([corner, corner], axis=-1).reshape(num_episodes, 1, 2)

    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=position_xy,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="test",
    )
    result = WithinHeadingReliabilityModule().run(analysis_input, tmp_path, analysis_settings(
        num_bins=1,
    ))

    assert set(result.metrics) >= {
        "mean_pooled_reliability",
        "mean_within_heading_reliability",
        "mean_within_heading_reliability_gain",
        "fraction_conjunctive",
    }
    assert (
        result.metrics["mean_within_heading_reliability"]
        > result.metrics["mean_pooled_reliability"]
    )
    assert np.isclose(result.metrics["fraction_conjunctive"], 0.5, atol=1e-6)
    assert result.tables["per_unit_metrics"].exists()
