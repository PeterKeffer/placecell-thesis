"""Single-unit measures on rectified codes of the first 512 test episodes."""

from __future__ import annotations

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.field_stability import field_map_metrics
from placecell_research.analysis.occupancy import get_or_compute_episode_bin_statistics
from placecell_research.analysis.reliability_splits import compute_field_traversal_reliability
from placecell_research.analysis.shift_nulls import circular_shift_spatial_information_null
from placecell_research.numerics.bin_maps import bin_center_grids
from placecell_research.numerics.place_cell_quality import benjamini_hochberg
from placecell_research.numerics.rate_map_kernels import (
    compute_place_field_mask,
    compute_rate_maps,
    compute_spatial_bin_assignments,
    skaggs_spatial_information,
)
from placecell_research.numerics.split_half import compute_split_half_rate_map_correlations

NUM_BINS = 60
SMOOTHING_SIGMA = 0.3
MIN_OCCUPANCY = 1.0e-6
FIELD_THRESHOLD_FRACTION = 0.2
NULL_SEED = 0
FDR_ALPHA = 0.05
SPLIT_HALF_RANDOM_SPLITS = 20
SPLIT_HALF_SEED = 0
SPLIT_HALF_MIN_EPISODES_PER_HALF = 2
TRAVERSAL_THRESHOLD_FRACTION = 0.3
MINIMUM_TRAVERSALS = 5

Bounds = tuple[tuple[float, float], tuple[float, float]]


def null_p_values(null_matrix: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p, finite draw count) for the circular-shift null."""
    finite = np.isfinite(null_matrix)
    draw_counts = finite.sum(axis=0).astype(np.int32)
    exceed = np.sum(finite & (null_matrix >= observed[None, :]), axis=0)
    p_values = np.full(observed.shape[0], np.nan, dtype=np.float32)
    assessable = np.isfinite(observed) & (draw_counts > 0)
    p_values[assessable] = ((1.0 + exceed[assessable]) / (1.0 + draw_counts[assessable])).astype(
        np.float32
    )
    return p_values, draw_counts


def rectified_track(
    rectified: np.ndarray, position_xy: np.ndarray, valid: np.ndarray, bounds: Bounds, shuffles: int
) -> dict[str, np.ndarray]:
    """Skaggs information of positive-part rate maps against its circular-shift null."""
    shared = dict(
        num_bins_x=NUM_BINS,
        num_bins_y=NUM_BINS,
        smoothing_sigma=SMOOTHING_SIGMA,
        min_occupancy=MIN_OCCUPANCY,
    )
    computation = compute_rate_maps(rectified, position_xy, valid, bounds=bounds, **shared)
    observed = np.asarray(
        skaggs_spatial_information(computation.rate_maps, computation.occupancy), dtype=np.float32
    )
    null_matrix = circular_shift_spatial_information_null(
        rectified,
        position_xy,
        valid,
        bounds=computation.bounds,
        num_shuffles=shuffles,
        rng_seed=NULL_SEED,
        unit_mask=np.isfinite(observed),
        **shared,
    )
    p_values, draw_counts = null_p_values(null_matrix, observed)
    null_95 = np.full(observed.shape, np.nan, dtype=np.float32)
    finite_draws = np.isfinite(null_matrix)
    for unit in np.flatnonzero(draw_counts > 0):
        null_95[unit] = float(np.percentile(null_matrix[finite_draws[:, unit], unit], 95.0))
    return {
        "rate_maps": computation.rate_maps.astype(np.float32),
        "occupancy": computation.occupancy,
        "spatial_information_bits": observed,
        "spatial_information_null_p": p_values,
        "spatial_information_null_95": null_95,
        "fdr_pass": benjamini_hochberg(p_values, alpha=FDR_ALPHA),
    }


def field_anatomy(rate_maps: np.ndarray, bounds: Bounds) -> dict[str, np.ndarray]:
    """Field mask, connected-component count and field area of every rate map."""
    x_grid, y_grid = bin_center_grids(bounds, num_bins_x=NUM_BINS, num_bins_y=NUM_BINS)
    metrics = field_map_metrics(
        rate_maps, FIELD_THRESHOLD_FRACTION, x_center_grid=x_grid, y_center_grid=y_grid
    )
    masks = np.zeros(rate_maps.shape, dtype=bool)
    counts = np.zeros(rate_maps.shape[0], dtype=np.int32)
    for unit in range(rate_maps.shape[0]):
        masks[unit], counts[unit], _ = compute_place_field_mask(
            rate_maps[unit], FIELD_THRESHOLD_FRACTION
        )
    return {
        "field_masks": masks,
        "field_component_count": counts,
        "field_area_bins": metrics.field_areas.astype(np.float32),
    }


def activity_and_traversal(
    analysis_input: AnalysisInput,
    rate_maps: np.ndarray,
    occupancy: np.ndarray,
    field_masks: np.ndarray,
    bounds: Bounds,
) -> dict[str, np.ndarray]:
    """Lifetime and spatial activity fractions and the share of field traversals with a response."""
    positive = analysis_input.representation
    flat = positive[analysis_input.valid_mask]
    mean = flat.mean(0, dtype=np.float64)
    second = np.einsum("ij,ij->j", flat, flat, dtype=np.float64) / len(flat)
    lifetime = np.divide(mean**2, second, out=np.full_like(mean, np.nan), where=second > 0)
    statistics = get_or_compute_episode_bin_statistics(
        analysis_input, num_bins_x=NUM_BINS, num_bins_y=NUM_BINS, bounds=bounds
    )
    hit, _, directional, _ = compute_field_traversal_reliability(
        statistics,
        field_masks,
        threshold_fraction=TRAVERSAL_THRESHOLD_FRACTION,
        minimum_traversals=MINIMUM_TRAVERSALS,
        flat_headings=analysis_input.heading[analysis_input.valid_mask],
    )
    visited = occupancy > 0
    maps = rate_maps[:, visited].astype(np.float64)
    weights = occupancy[visited] / occupancy[visited].sum()
    first = maps @ weights
    square = maps**2 @ weights
    spatial = np.divide(first * first, square, out=np.full_like(first, np.nan), where=square > 0)
    return {
        "lifetime_activity_fraction": lifetime,
        "spatial_activity_fraction": spatial,
        "traversal_hit_rate": hit,
        "directional_traversal_hit_rate": directional,
    }


def explained_variance(
    map_activity: np.ndarray,
    map_bins: np.ndarray,
    test_activity: np.ndarray,
    test_bins: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    """Per unit: 1 - SS_residual / SS_total of the rate map's prediction on the test steps."""
    occupancy = np.bincount(map_bins, minlength=bin_count).astype(np.float64)
    score = np.zeros(test_activity.shape[1])
    for unit in range(test_activity.shape[1]):
        target = test_activity[:, unit].astype(np.float64)
        total = ((target - target.mean()) ** 2).sum()
        if total == 0:
            continue
        source = map_activity[:, unit].astype(np.float64)
        rate_map = np.bincount(map_bins, weights=source, minlength=bin_count) / np.maximum(
            occupancy, 1.0
        )
        rate_map[occupancy == 0] = source.mean()
        score[unit] = 1.0 - ((target - rate_map[test_bins]) ** 2).sum() / total
    return score


def held_out_variance_explained(
    rectified: np.ndarray, position_xy: np.ndarray, valid: np.ndarray, bounds: Bounds
) -> np.ndarray:
    """Variance explained by position, maps from even episodes scored on odd ones and back."""
    episode_index = np.broadcast_to(np.arange(valid.shape[0])[:, None], valid.shape)[valid]
    bins, _, _, _ = compute_spatial_bin_assignments(
        position_xy[valid], num_bins_x=NUM_BINS, num_bins_y=NUM_BINS, bounds=bounds
    )
    activity = rectified[valid]
    even = episode_index % 2 == 0
    count = NUM_BINS * NUM_BINS
    return 0.5 * (
        explained_variance(activity[even], bins[even], activity[~even], bins[~even], count)
        + explained_variance(activity[~even], bins[~even], activity[even], bins[even], count)
    )


def visited_bins(position_xy: np.ndarray, valid: np.ndarray, bounds: Bounds) -> np.ndarray:
    bins, _, _, _ = compute_spatial_bin_assignments(
        position_xy[valid], num_bins_x=NUM_BINS, num_bins_y=NUM_BINS, bounds=bounds
    )
    counts = np.bincount(bins, minlength=NUM_BINS * NUM_BINS).reshape(NUM_BINS, NUM_BINS)
    return counts > 0


def single_unit_measures(
    codes: np.ndarray,
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    bounds: Bounds,
    *,
    env_id: str,
    null_shuffles: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Per-unit arrays and population counts for one model's codes [episodes, steps, units]."""
    rectified = np.maximum(codes, 0.0)
    track = rectified_track(rectified, position_xy, valid, bounds, null_shuffles)
    anatomy = field_anatomy(track["rate_maps"], bounds)
    positive_input = AnalysisInput(
        representation=rectified,
        position_xy=position_xy,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=valid,
        source_name="encoder_place_codes",
        label="encoder.place_codes",
        split_name="test",
        metadata={"env_id": env_id},
    )
    activity = activity_and_traversal(
        positive_input, track["rate_maps"], track["occupancy"], anatomy["field_masks"], bounds
    )
    split_half = compute_split_half_rate_map_correlations(
        codes,
        position_xy,
        valid,
        NUM_BINS,
        NUM_BINS,
        smoothing_sigma=SMOOTHING_SIGMA,
        min_occupancy=MIN_OCCUPANCY,
        bounds=bounds,
        num_random_splits=SPLIT_HALF_RANDOM_SPLITS,
        rng_seed=SPLIT_HALF_SEED,
        minimum_episodes_per_half=SPLIT_HALF_MIN_EPISODES_PER_HALF,
    ).astype(np.float32)
    visited = visited_bins(position_xy, valid, bounds)
    flat_codes = codes[valid]
    per_unit = {
        "active_step_count": np.count_nonzero(flat_codes != 0.0, axis=0).astype(np.int64),
        "spatial_information_bits": track["spatial_information_bits"],
        "spatial_information_null_95": track["spatial_information_null_95"],
        "spatial_information_null_p": track["spatial_information_null_p"],
        "fdr_pass": track["fdr_pass"],
        "split_half_correlation": split_half,
        "field_component_count": anatomy["field_component_count"],
        "field_area_bins": anatomy["field_area_bins"],
        "variance_explained_held_out": held_out_variance_explained(
            rectified, position_xy, valid, bounds
        ),
        **activity,
    }
    population = {
        "visited_bin_count": int(visited.sum()),
        "units_per_location_median": float(
            np.median(anatomy["field_masks"].sum(axis=0)[visited].astype(float))
        ),
        "valid_steps": int(valid.sum()),
    }
    return per_unit, population
