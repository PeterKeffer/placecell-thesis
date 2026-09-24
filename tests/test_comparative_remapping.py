from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from placecell_research.analysis import remapping
from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.registry import run_comparative_modules
from placecell_research.analysis.remapping_metrics import _occupancy_similarity
from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay
from placecell_research.numerics.rate_map_kernels import RateMapComputation


def _analysis_input(
    *,
    label: str,
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray,
    env_id: str | None = None,
) -> AnalysisInput:
    metadata = {"env_id": env_id} if env_id is not None else {}
    return AnalysisInput(
        label=label,
        representation=representation,
        position_xy=position_xy,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.place_codes",
        split_name="test",
        metadata=metadata,
    )


def test_remapping_comparison_runs_with_comparative_block(tmp_path) -> None:
    rng = np.random.default_rng(1)
    first = rng.normal(size=(4, 12, 5)).astype(np.float32)
    second = first + 0.01 * rng.normal(size=(4, 12, 5)).astype(np.float32)
    positions = rng.normal(size=(4, 12, 2)).astype(np.float32)
    valid_mask = np.ones((4, 12), dtype=bool)
    analysis_inputs = [
        _analysis_input(
            label="phase_a",
            representation=first,
            position_xy=positions,
            valid_mask=valid_mask,
        ),
        _analysis_input(
            label="phase_b",
            representation=second,
            position_xy=positions,
            valid_mask=valid_mask,
        ),
    ]
    results = run_comparative_modules(
        analysis_inputs,
        ["phase_a", "phase_b"],
        tmp_path,
        {
            "comparative": {
                "remapping_across_runs": {
                    "module": "remapping_comparison",
                    "enabled": True,
                    "num_bins_x": 20,
                    "num_bins_y": 20,
                    "smoothing_sigma": 0.0,
                    "remapping_shuffle_seed": 0,
                    "min_occupancy": 1.0e-6,
                    "inputs": [],
                }
            },
            "max_cost_tier": "heavy",
        },
    )
    assert "remapping_across_runs" in results
    assert "mean_pairwise_correlation" in results["remapping_across_runs"].metrics


def test_pairwise_map_correlation_ignores_jointly_unvisited_bins() -> None:
    first = np.array([[[1.0, np.nan, 2.0], [3.0, np.nan, 4.0]]], dtype=np.float32)
    second = np.array([[[1.0, np.nan, 2.0], [3.0, np.nan, 4.0]]], dtype=np.float32)

    correlations, mean_correlation = remapping.pairwise_map_correlation(first, second)

    assert correlations == pytest.approx(np.array([1.0], dtype=np.float32))
    assert mean_correlation == pytest.approx(1.0)


def test_pairwise_map_correlation_rejects_unit_count_mismatch() -> None:
    first = np.zeros((2, 2, 2), dtype=np.float32)
    second = np.zeros((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="same number of units"):
        remapping.pairwise_map_correlation(first, second)


def test_remapping_uses_world_overlay_bounds_for_matching_environments(
    monkeypatch,
    tmp_path,
) -> None:
    captured_bounds: list[tuple[tuple[float, float], tuple[float, float]] | None] = []
    env_id = "MiniWorld-WallGapAsymLarge-v0"
    expected_bounds = overlay_bounds(resolve_world_overlay(env_id))

    def fake_rate_maps(source, *, bounds, num_bins_x, num_bins_y, smoothing_sigma, min_occupancy):
        del source, smoothing_sigma, min_occupancy
        captured_bounds.append(bounds)
        return RateMapComputation(
            rate_maps=np.zeros((1, num_bins_y, num_bins_x), dtype=np.float32),
            occupancy=np.ones((num_bins_y, num_bins_x), dtype=np.float32),
            raw_occupancy=np.ones((num_bins_y, num_bins_x), dtype=np.float32),
            reliability_maps=None,
            visited_episode_counts=None,
            bounds=bounds,
        )

    monkeypatch.setattr(remapping, "get_or_compute_rate_maps", fake_rate_maps)
    representation = np.zeros((1, 2, 1), dtype=np.float32)
    positions = np.zeros((1, 2, 2), dtype=np.float32)
    valid_mask = np.ones((1, 2), dtype=bool)

    remapping.RemappingComparisonModule().run(
        [
            _analysis_input(
                label="phase_a",
                representation=representation,
                position_xy=positions,
                valid_mask=valid_mask,
                env_id=env_id,
            ),
            _analysis_input(
                label="phase_b",
                representation=representation,
                position_xy=positions,
                valid_mask=valid_mask,
                env_id=env_id,
            ),
        ],
        ["phase_a", "phase_b"],
        tmp_path,
        {
            "num_bins_x": 4,
            "num_bins_y": 4,
            "smoothing_sigma": 0.0,
            "remapping_shuffle_seed": 0,
            "min_occupancy": 1.0e-6,
        },
    )

    assert captured_bounds == [expected_bounds, expected_bounds]


def test_remapping_normalizes_known_environments_with_different_bounds(
    monkeypatch,
    tmp_path,
) -> None:
    captured_bounds: list[tuple[tuple[float, float], tuple[float, float]] | None] = []
    captured_positions: list[np.ndarray] = []

    def fake_rate_maps(source, *, bounds, num_bins_x, num_bins_y, smoothing_sigma, min_occupancy):
        del smoothing_sigma, min_occupancy
        captured_bounds.append(bounds)
        captured_positions.append(source.position_xy.copy())
        return RateMapComputation(
            rate_maps=np.zeros((1, num_bins_y, num_bins_x), dtype=np.float32),
            occupancy=np.ones((num_bins_y, num_bins_x), dtype=np.float32),
            raw_occupancy=np.ones((num_bins_y, num_bins_x), dtype=np.float32),
            reliability_maps=None,
            visited_episode_counts=None,
            bounds=bounds,
        )

    monkeypatch.setattr(remapping, "get_or_compute_rate_maps", fake_rate_maps)
    representation = np.zeros((1, 2, 1), dtype=np.float32)
    positions = np.array([[[0.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
    valid_mask = np.ones((1, 2), dtype=bool)

    remapping.RemappingComparisonModule().run(
        [
            _analysis_input(
                label="small",
                representation=representation,
                position_xy=positions,
                valid_mask=valid_mask,
                env_id="museum-gallery",
            ),
            _analysis_input(
                label="large",
                representation=representation,
                position_xy=positions,
                valid_mask=valid_mask,
                env_id="MiniWorld-WallGapAsymLarge-v0",
            ),
        ],
        ["small", "large"],
        tmp_path,
        {
            "num_bins_x": 4,
            "num_bins_y": 4,
            "smoothing_sigma": 0.0,
            "remapping_shuffle_seed": 0,
            "min_occupancy": 1.0e-6,
        },
    )

    assert captured_bounds == [remapping.NORMALIZED_WORLD_BOUNDS, remapping.NORMALIZED_WORLD_BOUNDS]
    assert not np.array_equal(captured_positions[0], positions)
    assert not np.array_equal(captured_positions[1], positions)


def test_remapping_emits_neuroscience_pair_diagnostics(monkeypatch, tmp_path) -> None:
    first_maps = np.array(
        [
            [[1.0, 2.0], [np.nan, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    second_maps = np.array(
        [
            [[1.0, 2.0], [np.nan, 4.0]],
            [[0.0, 1.0], [0.0, 2.0]],
        ],
        dtype=np.float32,
    )
    occupancy = np.array([[3.0, 2.0], [0.0, 1.0]], dtype=np.float32)

    def fake_rate_maps(source, *, bounds, num_bins_x, num_bins_y, smoothing_sigma, min_occupancy):
        del bounds, num_bins_x, num_bins_y, smoothing_sigma, min_occupancy
        rate_maps = first_maps if source.label == "phase_a" else second_maps
        return RateMapComputation(
            rate_maps=rate_maps,
            occupancy=occupancy,
            raw_occupancy=occupancy,
            reliability_maps=None,
            visited_episode_counts=None,
            bounds=((0.0, 1.0), (0.0, 1.0)),
        )

    monkeypatch.setattr(remapping, "get_or_compute_rate_maps", fake_rate_maps)
    representation = np.zeros((1, 2, 2), dtype=np.float32)
    positions = np.array([[[0.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
    valid_mask = np.ones((1, 2), dtype=bool)

    result = remapping.RemappingComparisonModule().run(
        [
            _analysis_input(
                label="phase_a",
                representation=representation,
                position_xy=positions,
                valid_mask=valid_mask,
            ),
            _analysis_input(
                label="phase_b",
                representation=representation,
                position_xy=positions,
                valid_mask=valid_mask,
            ),
        ],
        ["phase_a", "phase_b"],
        tmp_path,
        {
            "num_bins_x": 2,
            "num_bins_y": 2,
            "smoothing_sigma": 0.0,
            "min_occupancy": 1.0e-6,
            "place_field_threshold_fraction": 0.5,
            "active_peak_rate_threshold": 1.0e-6,
            "remapping_shuffle_iterations": 4,
            "remapping_shuffle_seed": 3,
        },
    )

    pair_name = "phase_a__phase_b"
    np.testing.assert_allclose(
        result.per_unit_metrics[f"{pair_name}__unit_correlation"],
        np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert result.metrics[f"{pair_name}__unit_correlation_median"] == pytest.approx(0.5)
    assert result.metrics[f"{pair_name}__unit_correlation_fraction_gt_0_5"] == pytest.approx(0.5)
    assert result.metrics[f"{pair_name}__unit_correlation_fraction_lt_0_2"] == pytest.approx(0.5)
    assert result.metrics[f"{pair_name}__active_both_units"] == pytest.approx(1.0)
    assert result.metrics[f"{pair_name}__active_b_only_units"] == pytest.approx(1.0)
    assert result.metrics[f"{pair_name}__active_jaccard"] == pytest.approx(0.5)
    assert result.metrics[f"{pair_name}__mean_population_vector_correlation"] == pytest.approx(1.0)
    assert result.metrics[f"{pair_name}__occupancy_correlation"] == pytest.approx(1.0)
    assert np.isfinite(result.metrics[f"{pair_name}__null_mean_unit_correlation"])
    assert result.metrics["diagonal_minus_off_diagonal_mean_correlation"] == pytest.approx(0.25)

    assert result.per_unit_metrics[f"{pair_name}__field_count_a"].shape == (2,)
    assert result.per_unit_metrics[f"{pair_name}__field_count_b"].shape == (2,)
    assert np.isfinite(result.per_unit_metrics[f"{pair_name}__field_center_shift"][0])
    assert f"{pair_name}__unit_correlation_histogram" in result.figures
    assert f"{pair_name}__population_vector_correlation" in result.figures
    assert result.figures[f"{pair_name}__unit_correlation_histogram"].exists()
    assert result.tables[f"{pair_name}__per_unit_metrics"].exists()
    assert result.tables[f"{pair_name}__null_distribution"].exists()
    assert result.tables["pair_summary"].exists()


def test_shared_visited_fraction_reads_occupancy_before_smoothing() -> None:
    """Two environments that never share a bin must not look half-overlapping."""
    from scipy.ndimage import gaussian_filter

    from placecell_research.analysis.remapping_metrics import _occupancy_similarity

    def _computation(row: int, column: int) -> RateMapComputation:
        raw_occupancy = np.zeros((9, 9), dtype=np.float32)
        raw_occupancy[row, column] = 40.0
        return RateMapComputation(
            rate_maps=np.zeros((1, 9, 9), dtype=np.float32),
            occupancy=gaussian_filter(raw_occupancy, sigma=1.0),
            raw_occupancy=raw_occupancy,
            reliability_maps=None,
            visited_episode_counts=None,
            bounds=((0.0, 9.0), (0.0, 9.0)),
        )

    first = _computation(2, 2)
    second = _computation(6, 6)

    assert np.count_nonzero((first.occupancy > 0.0) & (second.occupancy > 0.0)) > 0

    _correlation, shared_visited_fraction = _occupancy_similarity(first, second)
    assert shared_visited_fraction == 0.0


def test_shared_visited_fraction_reads_step_counts_before_smoothing() -> None:
    """Two phases that never entered the same bin share no bins, whatever the smoothing does."""
    raw_a = np.zeros((9, 9), dtype=np.float32)
    raw_a[2, 2] = 50.0
    raw_b = np.zeros((9, 9), dtype=np.float32)
    raw_b[6, 6] = 50.0
    smoothed_a = gaussian_filter(raw_a, sigma=2.0).astype(np.float32)
    smoothed_b = gaussian_filter(raw_b, sigma=2.0).astype(np.float32)

    assert float(np.mean((smoothed_a > 0.0) & (smoothed_b > 0.0))) > 0.0

    def result(rate_map_source: np.ndarray, raw: np.ndarray) -> RateMapComputation:
        return RateMapComputation(
            rate_maps=rate_map_source[None, :, :],
            occupancy=rate_map_source,
            raw_occupancy=raw,
            reliability_maps=None,
            visited_episode_counts=None,
            bounds=((0.0, 9.0), (0.0, 9.0)),
        )

    _, shared_visited_fraction = _occupancy_similarity(
        result(smoothed_a, raw_a),
        result(smoothed_b, raw_b),
    )

    assert shared_visited_fraction == 0.0
