"""Head-direction tuning analysis: heading selectivity plus spatial independence."""

from __future__ import annotations

import numpy as np

from placecell_research.analysis import head_direction_tuning
from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.head_direction_tuning import HeadDirectionTuningModule
from placecell_research.analysis.registry import ANALYSIS_MODULES


def test_head_direction_tuning_is_registered_for_config_use() -> None:
    assert "head_direction_tuning" in ANALYSIS_MODULES
    assert ANALYSIS_MODULES["head_direction_tuning"]().name == "head_direction_tuning"


def _build_input() -> AnalysisInput:
    """Three units over a complete position x heading grid."""
    heading_centers = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    positions: list[tuple[float, float]] = []
    headings: list[float] = []
    head_direction: list[float] = []
    conjunctive: list[float] = []
    place: list[float] = []
    preferred_heading = np.pi / 2.0
    repeats = 4

    for x in range(4):
        for y in range(4):
            is_field_position = x == 1 and y == 2
            for heading in heading_centers:
                tuning = float(np.exp(3.0 * np.cos(float(heading) - preferred_heading)))
                for _ in range(repeats):
                    positions.append((float(x), float(y)))
                    headings.append(float(heading))
                    head_direction.append(tuning)
                    conjunctive.append(tuning if is_field_position else 0.0)
                    place.append(1.0 if is_field_position else 0.0)

    steps = len(positions)
    representation = np.stack([head_direction, conjunctive, place], axis=1).reshape(1, steps, 3)
    position_xy = np.asarray(positions, dtype=np.float32).reshape(1, steps, 2)
    heading = np.asarray(headings, dtype=np.float32).reshape(1, steps)
    return AnalysisInput(
        representation=representation.astype(np.float32),
        position_xy=position_xy,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )


def _config() -> dict[str, float | int]:
    return {
        "head_direction_num_bins": 16,
        "head_direction_num_bins_x": 4,
        "head_direction_num_bins_y": 4,
        "head_direction_min_occupancy_per_bin": 2,
        "head_direction_min_heading_bins": 8,
        "head_direction_min_active_spatial_bins": 4,
        "head_direction_vector_length_threshold": 0.45,
        "head_direction_spatial_coverage_threshold": 0.45,
        "head_direction_position_invariance_threshold": 0.75,
        "head_direction_hd_cell_score_threshold": 0.1,
    }


def test_head_direction_tuning_separates_hd_conjunctive_and_place_units(tmp_path) -> None:
    result = HeadDirectionTuningModule().run(_build_input(), tmp_path, _config())

    vector_lengths = result.per_unit_metrics["heading_vector_length"]
    spatial_coverage = result.per_unit_metrics["spatial_coverage_fraction"]
    hd_candidates = result.per_unit_metrics["is_hd_candidate"]
    conjunctive_candidates = result.per_unit_metrics["is_conjunctive_candidate"]
    hd_cell_scores = result.per_unit_metrics["hd_cell_score"]

    assert vector_lengths[0] > 0.75
    assert spatial_coverage[0] > 0.95
    assert hd_cell_scores[0] > 0.75
    assert hd_cell_scores[1] < 0.1
    assert hd_cell_scores[2] < 0.1
    assert hd_candidates.tolist() == [1.0, 0.0, 0.0]
    assert conjunctive_candidates.tolist() == [0.0, 1.0, 0.0]


def test_head_direction_tuning_reports_aggregate_metrics(tmp_path) -> None:
    result = HeadDirectionTuningModule().run(_build_input(), tmp_path, _config())

    assert np.isclose(result.metrics["fraction_hd_candidate"], 1.0 / 3.0)
    assert np.isclose(result.metrics["fraction_conjunctive_candidate"], 1.0 / 3.0)
    assert result.metadata["head_direction_assessable_unit_count"] == 3


def test_head_direction_tuning_writes_per_unit_csv(tmp_path) -> None:
    result = HeadDirectionTuningModule().run(_build_input(), tmp_path, _config())

    table_path = result.tables["per_unit_metrics"]
    lines = table_path.read_text().splitlines()
    assert lines[0].startswith(
        "unit_index,heading_vector_length,heading_vector_length_excess,"
        "heading_vector_length_null_p,heading_vector_length_significant,preferred_heading_rad"
    )
    assert len(lines) == 4


def test_head_direction_tuning_renders_preferred_direction_rose(tmp_path) -> None:
    result = HeadDirectionTuningModule().run(_build_input(), tmp_path, _config())

    assert "preferred_direction_rose" in result.figures
    assert result.figures["preferred_direction_rose"].exists()


def test_head_direction_tuning_marks_unassessable_units_when_heading_missing(tmp_path) -> None:
    analysis_input = _build_input()
    analysis_input.heading = None

    result = HeadDirectionTuningModule().run(analysis_input, tmp_path, _config())

    assert np.all(np.isnan(result.per_unit_metrics["heading_vector_length"]))
    assert result.metadata["head_direction_assessable_unit_count"] == 0
    assert "preferred_direction_rose" not in result.figures


CORNERS = np.asarray([(0.0, 0.0), (0.0, 3.0), (3.0, 0.0), (3.0, 3.0)], dtype=np.float32)


def _null_config() -> dict[str, float | int]:
    return {
        **_config(),
        "head_direction_num_bins_x": 2,
        "head_direction_num_bins_y": 2,
        "head_direction_null_shuffles": 100,
    }


def _random_walk_input(
    activations, *, heading_step: float, seed: int, num_episodes: int = 4, num_steps: int = 500
) -> AnalysisInput:
    """Interleaved visits to four corners under an aperiodic heading random walk."""
    rng = np.random.default_rng(seed)
    headings = np.zeros((num_episodes, num_steps), dtype=np.float32)
    positions = np.zeros((num_episodes, num_steps, 2), dtype=np.float32)
    for episode in range(num_episodes):
        heading = rng.uniform(0.0, 2.0 * np.pi)
        for step in range(num_steps):
            heading += rng.normal(0.0, heading_step)
            headings[episode, step] = heading
            positions[episode, step] = CORNERS[step % 4]
    at_corner = np.tile(np.arange(num_steps) % 4 == 0, (num_episodes, 1))
    return AnalysisInput(
        representation=np.stack(activations(headings, at_corner), axis=2).astype(np.float32),
        position_xy=positions,
        heading=headings,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((num_episodes, num_steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )


def test_hd_null_flags_heading_tuned_not_place_units(tmp_path) -> None:
    """Heading tuning that repeats across interleaved visits survives the circular-shift null."""
    analysis_input = _random_walk_input(
        lambda headings, at_corner: [
            np.exp(3.0 * np.cos(headings - np.pi / 2.0)),
            np.where(at_corner, np.exp(3.0 * np.cos(headings - np.pi / 2.0)), 0.0),
            at_corner.astype(np.float32),
        ],
        heading_step=0.9,
        seed=11,
    )

    result = HeadDirectionTuningModule().run(analysis_input, tmp_path, _null_config())

    significant = result.per_unit_metrics["heading_vector_length_significant"]
    excess = result.per_unit_metrics["heading_vector_length_excess"]
    assert significant.tolist() == [1.0, 1.0, 0.0]
    assert excess[0] > 0.5
    assert result.per_unit_metrics["is_hd_candidate_significant"].tolist() == [1.0, 0.0, 0.0]
    assert "fraction_heading_selective_significant" in result.metrics
    assert "fraction_hd_candidate_significant" in result.metrics


def test_hd_null_does_not_flag_a_temporal_bout(tmp_path) -> None:
    """A bout of activity is not heading selectivity, however slowly the heading drifts."""
    analysis_input = _random_walk_input(
        lambda headings, at_corner: [
            np.where(
                np.tile(
                    (np.arange(headings.shape[1]) >= 120) & (np.arange(headings.shape[1]) < 220),
                    (headings.shape[0], 1),
                ),
                1.0,
                0.0,
            ),
            np.ones_like(headings),
        ],
        heading_step=0.08,
        seed=5,
    )

    result = HeadDirectionTuningModule().run(analysis_input, tmp_path, _null_config())

    null_p = result.per_unit_metrics["heading_vector_length_null_p"]
    assert result.per_unit_metrics["assessable"][0] == 1.0
    assert null_p[0] > 0.5
    assert result.per_unit_metrics["heading_vector_length_significant"].tolist() == [0.0, 0.0]


def test_hd_null_draws_every_configured_shuffle(tmp_path) -> None:
    """The null count is calibrated: p must be (1 + exceedances) / (1 + shuffles), nothing else."""
    num_shuffles = 40
    config = {**_config(), "head_direction_null_shuffles": num_shuffles}

    result = HeadDirectionTuningModule().run(_build_input(), tmp_path, config)

    p_values = result.per_unit_metrics["heading_vector_length_null_p"]
    finite_p_values = p_values[np.isfinite(p_values)]
    assert finite_p_values.size == 3
    exceedance_counts = finite_p_values.astype(np.float64) * (num_shuffles + 1) - 1.0
    assert np.allclose(exceedance_counts, np.round(exceedance_counts), atol=1e-4)
    assert np.all(exceedance_counts <= num_shuffles + 1e-4)


def test_hd_null_is_invariant_to_the_shuffle_batch_size(tmp_path, monkeypatch) -> None:
    """The batched null must reproduce the one-shuffle-at-a-time permutation stream exactly."""
    analysis_input = _build_input()
    batched = HeadDirectionTuningModule().run(analysis_input, tmp_path / "batched", _config())
    monkeypatch.setattr(head_direction_tuning, "_NULL_SCRATCH_BYTES", 1)
    assert head_direction_tuning._null_shuffle_batch(len(analysis_input.heading.ravel()), 100) == 1

    one_at_a_time = HeadDirectionTuningModule().run(analysis_input, tmp_path / "single", _config())

    for name, values in batched.per_unit_metrics.items():
        assert np.array_equal(values, one_at_a_time.per_unit_metrics[name], equal_nan=True), name


def test_hd_vector_length_requires_circular_coverage(tmp_path) -> None:
    heading_centers = np.linspace(0.0, np.pi / 2.0, 16, endpoint=False, dtype=np.float32)
    positions: list[tuple[float, float]] = []
    headings: list[float] = []
    for x in range(4):
        for y in range(4):
            for heading in heading_centers:
                for _ in range(4):
                    positions.append((float(x), float(y)))
                    headings.append(float(heading))
    steps = len(positions)
    analysis_input = AnalysisInput(
        representation=np.ones((1, steps, 1), dtype=np.float32),
        position_xy=np.asarray(positions, dtype=np.float32).reshape(1, steps, 2),
        heading=np.asarray(headings, dtype=np.float32).reshape(1, steps),
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )

    result = HeadDirectionTuningModule().run(analysis_input, tmp_path, _config())

    assert result.metadata["head_direction_assessable_unit_count"] == 0
    assert not np.isfinite(result.per_unit_metrics["heading_vector_length"]).any()


def test_shifted_heading_labels_are_a_per_episode_roll() -> None:
    """The null must roll inside episodes and move no heading label between them."""
    episode_lengths = np.asarray([7, 5], dtype=np.int64)
    heading_bins = np.arange(12) % 6
    offsets = [np.asarray([2, 3], dtype=np.int64), np.asarray([5, 1], dtype=np.int64)]

    shifted = head_direction_tuning._shifted_heading_labels(
        heading_bins,
        step_layout=head_direction_tuning._circular_shift_step_layout(
            episode_lengths, np.arange(heading_bins.size)
        ),
        offsets=offsets,
    )

    assert shifted.shape == (2, 12)
    for row, offset in zip(shifted, offsets, strict=False):
        np.testing.assert_array_equal(row[:7], np.roll(heading_bins[:7], -int(offset[0])))
        np.testing.assert_array_equal(row[7:], np.roll(heading_bins[7:], -int(offset[1])))
        np.testing.assert_array_equal(np.sort(row[:7]), np.sort(heading_bins[:7]))
