"""Online directionality / within-heading eval metrics."""

from __future__ import annotations

import numpy as np

from placecell_research.evaluation.metrics import (
    circular_true_run_length,
    compute_heading_tuning_shape_eval_metrics,
    count_circular_true_runs,
)
from placecell_research.stages.train_place_model import _compute_directionality_eval_metrics


def _uniform_position_and_heading(num_steps: int = 720):
    heading = np.linspace(-np.pi, np.pi, num_steps, endpoint=False).reshape(num_steps, 1)
    x = np.linspace(-1.0, 1.0, num_steps, endpoint=False)
    y = np.sin(np.linspace(0.0, 6.0 * np.pi, num_steps, endpoint=False))
    position_xy = np.stack([x, y], axis=-1).reshape(num_steps, 1, 2)
    valid = np.ones((num_steps, 1), dtype=bool)
    return position_xy, valid, heading


def _omni_and_conjunctive_field(seed: int = 0):
    rng = np.random.default_rng(seed)
    num_episodes = 800
    position_xy = rng.uniform(-1.0, 1.0, size=(num_episodes, 1, 2))
    heading = rng.uniform(-np.pi, np.pi, size=(num_episodes, 1))
    valid = np.ones((num_episodes, 1), dtype=bool)
    in_field = np.linalg.norm(position_xy.reshape(num_episodes, 2), axis=1) < 0.6
    quadrant_zero = (heading.reshape(num_episodes) % (2 * np.pi)) < (np.pi / 2)
    omnidirectional = in_field.astype(np.float64)
    conjunctive = (in_field & quadrant_zero).astype(np.float64)
    representation = np.stack([omnidirectional, conjunctive], axis=-1).reshape(num_episodes, 1, 2)
    return representation, position_xy, valid, heading


def test_circular_true_runs_treat_all_true_as_one_broad_peak():
    mask = np.ones(36, dtype=bool)

    assert count_circular_true_runs(mask) == 1
    assert circular_true_run_length(mask, 0) == 36


def test_directionality_eval_metrics_distinguish_single_peak_from_180_degree_tuning():
    position_xy, valid, heading = _uniform_position_and_heading()
    flat_heading = heading.reshape(-1)
    single_peak = np.exp(4.0 * np.cos(flat_heading)).reshape(-1, 1)
    bidirectional_peak = np.exp(4.0 * np.cos(2.0 * flat_heading)).reshape(-1, 1)
    representation = np.stack([single_peak, bidirectional_peak], axis=-1)

    metrics = _compute_directionality_eval_metrics(
        representation,
        position_xy,
        valid,
        heading,
        num_bins_x=3,
        num_bins_y=3,
        field_threshold_fraction=0.2,
    )

    assert metrics["validation.heading_tuning_assessable_units"] == 2.0
    assert metrics["validation.heading_tuning_first_harmonic_median_r"] > 0.35
    assert metrics["validation.heading_tuning_second_harmonic_median_r"] > 0.6
    assert metrics["validation.heading_tuning_single_peak_fraction"] == 0.5
    assert metrics["validation.heading_tuning_opposite_peak_fraction"] == 0.5
    assert metrics["validation.heading_tuning_median_half_width_deg"] < 100.0


def test_heading_tuning_reports_readable_activation_corridor_width():
    _position_xy, valid, heading = _uniform_position_and_heading()
    flat_heading = heading.reshape(-1)
    narrow_peak = np.exp(6.0 * np.cos(flat_heading)).reshape(-1, 1, 1)
    broad_peak = np.exp(0.8 * np.cos(flat_heading)).reshape(-1, 1, 1)

    narrow_metrics = compute_heading_tuning_shape_eval_metrics(
        representation_array=narrow_peak,
        valid_array=valid,
        heading_array=heading,
        min_occupancy_per_bin=1,
    )
    broad_metrics = compute_heading_tuning_shape_eval_metrics(
        representation_array=broad_peak,
        valid_array=valid,
        heading_array=heading,
        min_occupancy_per_bin=1,
    )

    assert "validation.heading_activation_corridor_median_deg" in narrow_metrics
    assert (
        narrow_metrics["validation.heading_activation_corridor_median_deg"]
        < broad_metrics["validation.heading_activation_corridor_median_deg"]
    )
    assert 0.0 < narrow_metrics["validation.heading_activation_corridor_median_deg"] <= 360.0


def test_directionality_eval_metrics_show_denominator_and_distribution():
    representation, position_xy, valid, heading = _omni_and_conjunctive_field()
    inactive_unit = np.zeros((*representation.shape[:2], 1), dtype=representation.dtype)
    representation = np.concatenate([representation, inactive_unit], axis=-1)

    metrics = _compute_directionality_eval_metrics(
        representation,
        position_xy,
        valid,
        heading,
        num_bins_x=3,
        num_bins_y=3,
        field_threshold_fraction=0.2,
    )

    assert metrics["validation.directionality_total_units"] == 3.0
    assert metrics["validation.directionality_assessable_units"] == 2.0
    assert metrics["validation.directionality_assessable_fraction"] == 2.0 / 3.0
    assert "validation.directionality_mean_r" in metrics
    assert 0.0 < metrics["validation.fraction_directional"] < 1.0


def test_directionality_eval_metrics_report_place_field_omnidirectionality_directly():
    representation, position_xy, valid, heading = _omni_and_conjunctive_field()

    metrics = _compute_directionality_eval_metrics(
        representation,
        position_xy,
        valid,
        heading,
        num_bins_x=3,
        num_bins_y=3,
        field_threshold_fraction=0.2,
    )

    assert (
        metrics["validation.place_field_omnidirectionality_median"]
        == 1.0 - metrics["validation.directionality_median_r"]
    )
    assert (
        metrics["validation.place_field_omnidirectionality_mean"]
        == 1.0 - metrics["validation.directionality_mean_r"]
    )
    assert (
        metrics["validation.place_field_fraction_omnidirectional"]
        == metrics["validation.fraction_omnidirectional"]
    )
    assert (
        metrics["validation.place_field_fraction_directional"]
        == metrics["validation.fraction_directional"]
    )


def test_directionality_eval_metrics_emitted_and_finite():
    representation, position_xy, valid, heading = _omni_and_conjunctive_field()
    metrics = _compute_directionality_eval_metrics(
        representation,
        position_xy,
        valid,
        heading,
        num_bins_x=3,
        num_bins_y=3,
        field_threshold_fraction=0.2,
    )
    assert "validation.directionality_median_r" in metrics
    assert "validation.within_heading_reliability_gain" in metrics
    assert metrics["validation.directionality_assessable_units"] > 0
    assert all(np.isfinite(v) for v in metrics.values())
    assert 0.0 <= metrics["validation.directionality_median_r"] <= 1.0
    assert metrics["validation.within_heading_reliability_gain"] > 0.0


def test_directionality_eval_metrics_empty_without_heading():
    representation, position_xy, valid, _heading = _omni_and_conjunctive_field()
    assert (
        _compute_directionality_eval_metrics(
            representation,
            position_xy,
            valid,
            None,
            num_bins_x=3,
            num_bins_y=3,
            field_threshold_fraction=0.2,
        )
        == {}
    )
