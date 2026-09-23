"""Rate-map computations kept separate from panel rendering."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from placecell_research.config.schema import analysis_config_default

from ..numerics.occupancy import EpisodeBinStatistics
from ..numerics.place_cell_quality import (
    batched_spatial_coherence,
    benjamini_hochberg,
    coding_purity_score,
    field_coverage_fraction,
    fraction_place_cells,
    place_cell_pass_mask,
    reliability_weighted_information,
    reliability_weighted_information_excess,
    resolve_place_cell_gate_thresholds,
)
from ..numerics.rate_map_kernels import (
    PlaceMetricRateMapPreparation,
    RateMapComputation,
    compute_place_field_mask,
    prepare_place_metric_rate_maps,
    skaggs_spatial_information,
)
from .base import AnalysisInput
from .helpers import get_or_compute_rate_maps
from .occupancy import get_or_compute_episode_bin_statistics
from .place_cell_quality import get_or_compute_confound_scores
from .reliability_splits import (
    compute_field_traversal_reliability,
    compute_reliability_lift_maps,
    compute_reliability_maps,
    compute_revisit_activity_metrics,
)
from .shift_nulls import circular_shift_spatial_information_null
from .timing import record_timing


@dataclass(frozen=True, slots=True)
class RateMapMetricSettings:
    """The knobs the bundle reads out of the analysis config, resolved once."""

    num_bins_x: int
    num_bins_y: int
    smoothing_sigma: float
    min_occupancy: float
    threshold_fraction: float
    threshold_quantile: float
    field_threshold_fraction: float
    per_bin_cv_min_episodes: int
    split_half_agreement_min_episodes_per_half: int
    use_absolute_activations: bool
    bin_consistency_active_bin_peak_fraction: float
    bin_consistency_active_episode_threshold_fraction: float
    split_half_num_random_splits: int
    split_half_random_split_seed: int
    place_cell_gate_minimum_split_half: float
    place_cell_gate_minimum_coherence: float
    place_cell_gate_maximum_confound: float
    field_traversal_minimum_traversals: int
    field_traversal_heading_sectors: int
    field_core_threshold_fraction: float
    negative_tolerance: float
    max_negative_bin_fraction: float
    max_negative_peak_fraction: float
    null_num_shuffles: int
    null_seed: int


@dataclass(frozen=True, slots=True)
class ReliabilityMaps:
    """Per-bin reliability, consistency and split-half maps with their visit support."""

    thresholded_maps: np.ndarray
    thresholded_lift_maps: np.ndarray
    thresholded_visit_counts: np.ndarray
    quantile_maps: np.ndarray
    quantile_lift_maps: np.ndarray
    quantile_visit_counts: np.ndarray
    bin_consistency_maps: np.ndarray
    bin_coefficient_of_variation_maps: np.ndarray
    bin_consistency_visit_counts: np.ndarray
    split_half_agreement_maps: np.ndarray
    split_half_agreement_support_counts: np.ndarray
    split_half_rate_map_correlation: np.ndarray
    episode_rate_map_correlation: np.ndarray


@dataclass(frozen=True, slots=True)
class PlaceMetricScores:
    """Per-unit spatial-coding scores and the place-cell gates they feed."""

    prepared_maps: PlaceMetricRateMapPreparation
    rate_maps: np.ndarray
    supported: np.ndarray
    spatial_information_bits: np.ndarray
    spatial_coherence: np.ndarray
    max_available_confound: np.ndarray
    reliability_weighted_information: np.ndarray
    reliability_weighted_information_excess: np.ndarray
    coding_purity: np.ndarray
    passes_gates: np.ndarray
    gates_assessable: np.ndarray
    replicates_split_half: np.ndarray
    gate_summary: dict[str, float]
    field_coverage_fraction: float


@dataclass(frozen=True, slots=True)
class SpatialInformationNull:
    """What the episode-preserving circular-shift null says about each unit's Skaggs bits."""

    null_p: np.ndarray
    null_95: np.ndarray
    significant: np.ndarray
    num_shuffles: int
    fraction_significant: float
    fraction_place_cells_strict: float


@dataclass(frozen=True, slots=True)
class PlaceFieldSummaries:
    """Per-unit place-field masks and the metric means taken inside them."""

    masks: np.ndarray
    counts: np.ndarray
    areas: np.ndarray
    reliability: np.ndarray
    reliability_lift: np.ndarray
    traversal_reliability: np.ndarray
    traversal_counts: np.ndarray
    traversal_reliability_directional: np.ndarray
    traversal_directional_counts: np.ndarray
    core_traversal_reliability: np.ndarray
    core_traversal_counts: np.ndarray
    quantile_reliability: np.ndarray
    bin_consistency: np.ndarray
    split_half_agreement: np.ndarray
    reliability_supported_fraction: np.ndarray
    quantile_reliability_supported_fraction: np.ndarray
    bin_consistency_supported_fraction: np.ndarray
    split_half_agreement_supported_fraction: np.ndarray


@dataclass(frozen=True, slots=True)
class UnitMapSummaries:
    """Whole-map mean/max reductions, one pair per per-bin metric and unit."""

    mean_reliability: np.ndarray
    max_reliability: np.ndarray
    mean_reliability_lift: np.ndarray
    max_reliability_lift: np.ndarray
    mean_quantile_reliability: np.ndarray
    max_quantile_reliability: np.ndarray
    mean_quantile_reliability_lift: np.ndarray
    max_quantile_reliability_lift: np.ndarray
    mean_bin_consistency: np.ndarray
    max_bin_consistency: np.ndarray
    mean_split_half_agreement: np.ndarray
    max_split_half_agreement: np.ndarray
    mean_bin_coefficient_of_variation: np.ndarray


@dataclass(frozen=True, slots=True)
class SignedMapDiagnostics:
    """How much of each unit's rate-map mass is inhibitory, measured on the un-clipped maps."""

    peak_magnitude: np.ndarray
    peak_abs_activation: np.ndarray
    negative_fraction: np.ndarray
    median_negative_fraction: float
    fraction_units_majority_negative: float


@dataclass(frozen=True, slots=True)
class RateMapMetricBundle:
    """Everything one analysis target's rate-map pass produced, grouped by question."""

    settings: RateMapMetricSettings
    rate_map_result: RateMapComputation
    reliability: ReliabilityMaps
    place_metrics: PlaceMetricScores
    spatial_information_null: SpatialInformationNull
    fields: PlaceFieldSummaries
    summaries: UnitMapSummaries
    signed_diagnostics: SignedMapDiagnostics
    ranked_indices: np.ndarray
    ranking_is_peak_preview: bool
    timing_seconds: dict[str, float]


def reduce_finite_maps(metric_maps: np.ndarray, reducer: str) -> np.ndarray:
    reduced = np.zeros(metric_maps.shape[0], dtype=np.float32)
    for unit_index, metric_map in enumerate(metric_maps):
        finite_values = metric_map[np.isfinite(metric_map)]
        if finite_values.size == 0:
            reduced[unit_index] = 0.0
        elif reducer == "max":
            reduced[unit_index] = float(np.max(finite_values))
        else:
            reduced[unit_index] = float(np.mean(finite_values))
    return reduced


def mean_finite_inside_mask(metric_map: np.ndarray, field_mask: np.ndarray) -> float:
    field_values = metric_map[np.isfinite(metric_map) & field_mask]
    if field_values.size == 0:
        return 0.0
    return float(np.mean(field_values))


def finite_fraction_inside_mask(metric_map: np.ndarray, field_mask: np.ndarray) -> float:
    total_field_bins = int(np.count_nonzero(field_mask))
    if total_field_bins == 0:
        return 0.0
    supported_bins = int(np.count_nonzero(np.isfinite(metric_map) & field_mask))
    return float(supported_bins / total_field_bins)


def nanmean_or_nan(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def nanmax_or_nan(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.max(finite_values))


def resolve_rate_map_metric_settings(config: dict) -> RateMapMetricSettings:
    """Read every knob the bundle needs out of the analysis config, once."""
    gate_thresholds = resolve_place_cell_gate_thresholds(config)
    return RateMapMetricSettings(
        num_bins_x=int(config.get("num_bins_x", 60)),
        num_bins_y=int(config.get("num_bins_y", 60)),
        smoothing_sigma=float(config.get("smoothing_sigma", 0.3)),
        min_occupancy=float(config.get("min_occupancy", 1e-6)),
        threshold_fraction=float(config.get("reliability_threshold_fraction", 0.2)),
        threshold_quantile=float(config.get("reliability_threshold_quantile", 0.95)),
        field_threshold_fraction=float(config.get("place_field_threshold_fraction", 0.2)),
        per_bin_cv_min_episodes=int(config.get("per_bin_cv_min_episodes", 3)),
        split_half_agreement_min_episodes_per_half=int(
            config.get("split_half_agreement_min_episodes_per_half", 2)
        ),
        use_absolute_activations=bool(
            config.get("reliability_use_absolute_activations", False)
        ),
        bin_consistency_active_bin_peak_fraction=float(
            config.get("bin_consistency_active_bin_peak_fraction", 0.05)
        ),
        bin_consistency_active_episode_threshold_fraction=float(
            config.get("bin_consistency_active_episode_threshold_fraction", 0.5)
        ),
        split_half_num_random_splits=int(config.get("split_half_num_random_splits", 20)),
        split_half_random_split_seed=int(config.get("split_half_random_split_seed", 0)),
        place_cell_gate_minimum_split_half=gate_thresholds.minimum_split_half,
        place_cell_gate_minimum_coherence=gate_thresholds.minimum_coherence,
        place_cell_gate_maximum_confound=gate_thresholds.maximum_confound,
        field_traversal_minimum_traversals=int(
            config.get("field_traversal_minimum_traversals", 5)
        ),
        field_traversal_heading_sectors=int(
            config.get("field_traversal_heading_sectors", 8)
        ),
        field_core_threshold_fraction=float(
            config.get("place_field_core_threshold_fraction", 0.5)
        ),
        negative_tolerance=float(config.get("place_metric_negative_tolerance", 1e-8)),
        max_negative_bin_fraction=float(
            config.get("place_metric_max_negative_bin_fraction", 0.01)
        ),
        max_negative_peak_fraction=float(
            config.get("place_metric_max_negative_peak_fraction", 0.05)
        ),
        null_num_shuffles=int(
            config.get(
                "spatial_information_null_shuffles",
                analysis_config_default("spatial_information_null_shuffles"),
            )
        ),
        null_seed=int(config.get("spatial_information_null_seed", 0)),
    )


@dataclass(frozen=True, slots=True)
class _ReliabilityPass:
    """The reliability maps before the lift maps, which are computed after the field masks."""

    thresholded_maps: np.ndarray
    thresholded_visit_counts: np.ndarray
    quantile_maps: np.ndarray
    quantile_visit_counts: np.ndarray
    bin_consistency_maps: np.ndarray
    bin_coefficient_of_variation_maps: np.ndarray
    bin_consistency_visit_counts: np.ndarray
    split_half_agreement_maps: np.ndarray
    split_half_agreement_support_counts: np.ndarray
    split_half_rate_map_correlation: np.ndarray
    episode_rate_map_correlation: np.ndarray


def _compute_reliability_pass(
    analysis_input: AnalysisInput,
    settings: RateMapMetricSettings,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    episode_statistics: EpisodeBinStatistics,
    timing_seconds: dict[str, float],
) -> _ReliabilityPass:
    """Both thresholded reliability maps plus the one-pass revisit metrics."""
    section_started_at = perf_counter()
    reliability_maps, visited_episode_counts = compute_reliability_maps(
        analysis_input.representation,
        analysis_input.position_xy,
        analysis_input.valid_mask,
        num_bins_x=settings.num_bins_x,
        num_bins_y=settings.num_bins_y,
        bounds=bounds,
        threshold_mode="peak_fraction",
        threshold_fraction=settings.threshold_fraction,
        threshold_quantile=settings.threshold_quantile,
        use_absolute_activations=settings.use_absolute_activations,
        episode_statistics=episode_statistics,
    )
    record_timing(timing_seconds, "reliability_peak_fraction", section_started_at)
    section_started_at = perf_counter()
    quantile_reliability_maps, quantile_visited_episode_counts = compute_reliability_maps(
        analysis_input.representation,
        analysis_input.position_xy,
        analysis_input.valid_mask,
        num_bins_x=settings.num_bins_x,
        num_bins_y=settings.num_bins_y,
        bounds=bounds,
        threshold_mode="quantile_per_unit",
        threshold_fraction=settings.threshold_fraction,
        threshold_quantile=settings.threshold_quantile,
        use_absolute_activations=False,
        episode_statistics=episode_statistics,
    )
    record_timing(timing_seconds, "reliability_quantile", section_started_at)

    section_started_at = perf_counter()
    revisit_metrics = compute_revisit_activity_metrics(
        analysis_input.representation,
        analysis_input.position_xy,
        analysis_input.valid_mask,
        num_bins_x=settings.num_bins_x,
        num_bins_y=settings.num_bins_y,
        smoothing_sigma=settings.smoothing_sigma,
        min_occupancy=settings.min_occupancy,
        bounds=bounds,
        minimum_visited_episodes=settings.per_bin_cv_min_episodes,
        active_bin_peak_fraction=settings.bin_consistency_active_bin_peak_fraction,
        active_episode_threshold_fraction_of_bin_mean=(
            settings.bin_consistency_active_episode_threshold_fraction
        ),
        minimum_episodes_per_half=settings.split_half_agreement_min_episodes_per_half,
        episode_statistics=episode_statistics,
        num_random_splits=settings.split_half_num_random_splits,
        rng_seed=settings.split_half_random_split_seed,
    )
    record_timing(timing_seconds, "revisit_activity_metrics", section_started_at)
    return _ReliabilityPass(
        thresholded_maps=reliability_maps,
        thresholded_visit_counts=visited_episode_counts,
        quantile_maps=quantile_reliability_maps,
        quantile_visit_counts=quantile_visited_episode_counts,
        bin_consistency_maps=revisit_metrics.bin_consistency_maps,
        bin_coefficient_of_variation_maps=revisit_metrics.bin_coefficient_of_variation_maps,
        bin_consistency_visit_counts=revisit_metrics.consistency_visit_counts,
        split_half_agreement_maps=revisit_metrics.split_half_agreement_maps,
        split_half_agreement_support_counts=(
            revisit_metrics.split_half_agreement_support_counts
        ),
        split_half_rate_map_correlation=revisit_metrics.split_half_rate_map_correlation,
        episode_rate_map_correlation=revisit_metrics.episode_rate_map_correlation,
    )


def _compute_place_metric_scores(
    analysis_input: AnalysisInput,
    settings: RateMapMetricSettings,
    *,
    rate_map_result: RateMapComputation,
    split_half_rate_map_correlation: np.ndarray,
    timing_seconds: dict[str, float],
) -> tuple[PlaceMetricScores, np.ndarray]:
    """Spatial information, coherence, confounds and the place-cell gates built on them."""
    section_started_at = perf_counter()
    prepared_place_metric_maps = prepare_place_metric_rate_maps(
        rate_map_result.rate_maps,
        negative_tolerance=settings.negative_tolerance,
        max_negative_bin_fraction=settings.max_negative_bin_fraction,
        max_negative_peak_fraction=settings.max_negative_peak_fraction,
    )
    record_timing(timing_seconds, "prepare_place_metric_maps", section_started_at)
    section_started_at = perf_counter()
    spatial_information_bits = np.asarray(
        skaggs_spatial_information(
            rate_map_result.rate_maps,
            rate_map_result.occupancy,
            negative_tolerance=settings.negative_tolerance,
            max_negative_bin_fraction=settings.max_negative_bin_fraction,
            max_negative_peak_fraction=settings.max_negative_peak_fraction,
        ),
        dtype=np.float32,
    )
    record_timing(timing_seconds, "spatial_information", section_started_at)
    section_started_at = perf_counter()
    spatial_coherence_scores = batched_spatial_coherence(rate_map_result.rate_maps)
    record_timing(timing_seconds, "spatial_coherence", section_started_at)
    section_started_at = perf_counter()
    confound_scores = get_or_compute_confound_scores(analysis_input)
    max_available_confound_score = confound_scores["max_available_confound_score"]
    record_timing(timing_seconds, "confound_scores", section_started_at)
    section_started_at = perf_counter()
    reliability_weighted_information_scores = reliability_weighted_information(
        spatial_information_bits,
        split_half_rate_map_correlation,
    )
    coding_purity_scores = coding_purity_score(
        spatial_information_bits,
        spatial_coherence_scores,
        split_half_rate_map_correlation,
        max_available_confound_score,
    )
    (
        unit_passes_place_cell_gates,
        unit_gates_assessable,
        unit_replicates_split_half,
        _,
        _,
    ) = place_cell_pass_mask(
        split_half_rate_map_correlation,
        spatial_coherence_scores,
        max_available_confound_score,
        supported_mask=prepared_place_metric_maps.supported_mask,
        minimum_split_half=settings.place_cell_gate_minimum_split_half,
        minimum_coherence=settings.place_cell_gate_minimum_coherence,
        maximum_confound=settings.place_cell_gate_maximum_confound,
    )
    place_cell_gate_summary = fraction_place_cells(
        split_half_rate_map_correlation,
        spatial_coherence_scores,
        max_available_confound_score,
        supported_mask=prepared_place_metric_maps.supported_mask,
        minimum_split_half=settings.place_cell_gate_minimum_split_half,
        minimum_coherence=settings.place_cell_gate_minimum_coherence,
        maximum_confound=settings.place_cell_gate_maximum_confound,
    )
    qualifying_field_coverage = field_coverage_fraction(
        rate_map_result.rate_maps,
        rate_map_result.raw_occupancy,
        unit_passes_place_cell_gates,
        threshold_fraction=settings.field_threshold_fraction,
    )
    record_timing(timing_seconds, "quality_scores", section_started_at)
    scores = PlaceMetricScores(
        prepared_maps=prepared_place_metric_maps,
        rate_maps=prepared_place_metric_maps.clipped_rate_maps,
        supported=prepared_place_metric_maps.supported_mask,
        spatial_information_bits=spatial_information_bits,
        spatial_coherence=spatial_coherence_scores,
        max_available_confound=max_available_confound_score,
        reliability_weighted_information=reliability_weighted_information_scores,
        reliability_weighted_information_excess=np.full_like(
            reliability_weighted_information_scores, np.nan
        ),
        coding_purity=coding_purity_scores,
        passes_gates=unit_passes_place_cell_gates,
        gates_assessable=unit_gates_assessable,
        replicates_split_half=unit_replicates_split_half,
        gate_summary=place_cell_gate_summary,
        field_coverage_fraction=qualifying_field_coverage,
    )
    return scores, spatial_information_bits


def _with_null_excess(
    scores: PlaceMetricScores,
    excess_scores: np.ndarray,
) -> PlaceMetricScores:
    """The same scores with the null-referenced excess variant filled in."""
    return PlaceMetricScores(
        prepared_maps=scores.prepared_maps,
        rate_maps=scores.rate_maps,
        supported=scores.supported,
        spatial_information_bits=scores.spatial_information_bits,
        spatial_coherence=scores.spatial_coherence,
        max_available_confound=scores.max_available_confound,
        reliability_weighted_information=scores.reliability_weighted_information,
        reliability_weighted_information_excess=excess_scores,
        coding_purity=scores.coding_purity,
        passes_gates=scores.passes_gates,
        gates_assessable=scores.gates_assessable,
        replicates_split_half=scores.replicates_split_half,
        gate_summary=scores.gate_summary,
        field_coverage_fraction=scores.field_coverage_fraction,
    )


def _compute_spatial_information_null(
    analysis_input: AnalysisInput,
    settings: RateMapMetricSettings,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    spatial_information_bits: np.ndarray,
    passes_gates: np.ndarray,
    gates_assessable: np.ndarray,
    timing_seconds: dict[str, float],
) -> tuple[SpatialInformationNull, np.ndarray]:
    """Calibrate raw Skaggs bits against the episode-preserving circular-shift null."""
    section_started_at = perf_counter()
    unit_count = spatial_information_bits.shape[0]
    spatial_information_null_p = np.full(unit_count, np.nan, dtype=np.float32)
    spatial_information_null_95 = np.full(unit_count, np.nan, dtype=np.float32)
    spatial_information_significant = np.zeros(unit_count, dtype=bool)
    observed_is_finite = np.isfinite(spatial_information_bits)
    if settings.null_num_shuffles > 0 and observed_is_finite.any():
        null_matrix = analysis_input.get_cached_metric(
            (
                "spatial_information_null",
                settings.num_bins_x,
                settings.num_bins_y,
                settings.smoothing_sigma,
                settings.min_occupancy,
                bounds,
                settings.null_num_shuffles,
                settings.null_seed,
                observed_is_finite.tobytes(),
            ),
            lambda: circular_shift_spatial_information_null(
                analysis_input.representation,
                analysis_input.position_xy,
                analysis_input.valid_mask,
                num_bins_x=settings.num_bins_x,
                num_bins_y=settings.num_bins_y,
                smoothing_sigma=settings.smoothing_sigma,
                min_occupancy=settings.min_occupancy,
                bounds=bounds,
                num_shuffles=settings.null_num_shuffles,
                rng_seed=settings.null_seed,
                unit_mask=observed_is_finite,
            ),
        )
        null_is_finite = np.isfinite(null_matrix)
        finite_draw_counts = null_is_finite.sum(axis=0)
        exceed_counts = np.sum(
            null_is_finite & (null_matrix >= spatial_information_bits[None, :]), axis=0
        )
        assessable_null = observed_is_finite & (finite_draw_counts > 0)
        spatial_information_null_p[assessable_null] = (
            (1.0 + exceed_counts[assessable_null]) / (1.0 + finite_draw_counts[assessable_null])
        ).astype(np.float32)
        for unit_index in np.flatnonzero(assessable_null):
            spatial_information_null_95[unit_index] = float(
                np.percentile(null_matrix[null_is_finite[:, unit_index], unit_index], 95.0)
            )
        spatial_information_significant = benjamini_hochberg(spatial_information_null_p)
    fraction_significant_spatial_information = (
        float(spatial_information_significant[observed_is_finite].mean())
        if settings.null_num_shuffles > 0 and observed_is_finite.any()
        else float("nan")
    )
    fraction_place_cells_strict = (
        float(
            (passes_gates & spatial_information_significant).sum() / gates_assessable.sum()
        )
        if settings.null_num_shuffles > 0 and gates_assessable.any()
        else float("nan")
    )
    record_timing(timing_seconds, "spatial_information_null", section_started_at)
    return (
        SpatialInformationNull(
            null_p=spatial_information_null_p,
            null_95=spatial_information_null_95,
            significant=spatial_information_significant,
            num_shuffles=settings.null_num_shuffles,
            fraction_significant=fraction_significant_spatial_information,
            fraction_place_cells_strict=fraction_place_cells_strict,
        ),
        spatial_information_null_95,
    )


def _compute_place_field_summaries(
    settings: RateMapMetricSettings,
    *,
    episode_statistics: EpisodeBinStatistics,
    place_metric_rate_maps: np.ndarray,
    supports_place_metrics: np.ndarray,
    reliability: _ReliabilityPass,
    timing_seconds: dict[str, float],
    flat_headings: np.ndarray | None = None,
) -> tuple[PlaceFieldSummaries, list[np.ndarray]]:
    """Place-field masks plus every metric mean taken inside them."""
    section_started_at = perf_counter()
    field_masks: list[np.ndarray] = []
    unit_count = place_metric_rate_maps.shape[0]
    field_counts = np.full(unit_count, np.nan, dtype=np.float32)
    field_areas = np.full(unit_count, np.nan, dtype=np.float32)
    field_reliability = np.full(unit_count, np.nan, dtype=np.float32)
    field_quantile_reliability = np.full(unit_count, np.nan, dtype=np.float32)
    field_bin_consistency = np.full(unit_count, np.nan, dtype=np.float32)
    field_split_half_agreement = np.full(unit_count, np.nan, dtype=np.float32)
    field_reliability_supported_fraction = np.full(unit_count, np.nan, dtype=np.float32)
    field_quantile_reliability_supported_fraction = np.full(unit_count, np.nan, dtype=np.float32)
    field_bin_consistency_supported_fraction = np.full(unit_count, np.nan, dtype=np.float32)
    field_split_half_agreement_supported_fraction = np.full(unit_count, np.nan, dtype=np.float32)
    core_masks: list[np.ndarray] = []
    for unit_index, rate_map in enumerate(place_metric_rate_maps):
        if not supports_place_metrics[unit_index]:
            field_masks.append(np.zeros_like(rate_map, dtype=bool))
            core_masks.append(np.zeros_like(rate_map, dtype=bool))
            continue
        field_mask, field_count, field_area = compute_place_field_mask(
            rate_map,
            settings.field_threshold_fraction,
        )
        field_masks.append(field_mask)
        core_mask, _, _ = compute_place_field_mask(
            rate_map,
            settings.field_core_threshold_fraction,
        )
        core_masks.append(core_mask)
        field_counts[unit_index] = float(field_count)
        field_areas[unit_index] = float(field_area)
        field_reliability[unit_index] = mean_finite_inside_mask(
            reliability.thresholded_maps[unit_index],
            field_mask,
        )
        field_reliability_supported_fraction[unit_index] = finite_fraction_inside_mask(
            reliability.thresholded_maps[unit_index],
            field_mask,
        )
        field_quantile_reliability[unit_index] = mean_finite_inside_mask(
            reliability.quantile_maps[unit_index],
            field_mask,
        )
        field_quantile_reliability_supported_fraction[unit_index] = finite_fraction_inside_mask(
            reliability.quantile_maps[unit_index],
            field_mask,
        )
        field_bin_consistency[unit_index] = mean_finite_inside_mask(
            reliability.bin_consistency_maps[unit_index],
            field_mask,
        )
        field_split_half_agreement[unit_index] = mean_finite_inside_mask(
            reliability.split_half_agreement_maps[unit_index],
            field_mask,
        )
        field_bin_consistency_supported_fraction[unit_index] = finite_fraction_inside_mask(
            reliability.bin_consistency_maps[unit_index],
            field_mask,
        )
        field_split_half_agreement_supported_fraction[unit_index] = finite_fraction_inside_mask(
            reliability.split_half_agreement_maps[unit_index],
            field_mask,
        )

    field_masks_array = np.stack(field_masks, axis=0).astype(bool, copy=False)
    record_timing(timing_seconds, "place_field_masks", section_started_at)
    section_started_at = perf_counter()
    (
        field_traversal_reliability,
        field_traversal_counts,
        field_traversal_reliability_directional,
        field_traversal_directional_counts,
    ) = compute_field_traversal_reliability(
        episode_statistics,
        field_masks_array,
        threshold_mode="peak_fraction",
        threshold_fraction=settings.threshold_fraction,
        threshold_quantile=settings.threshold_quantile,
        use_absolute_activations=settings.use_absolute_activations,
        minimum_traversals=settings.field_traversal_minimum_traversals,
        flat_headings=flat_headings,
        num_heading_sectors=settings.field_traversal_heading_sectors,
    )
    core_masks_array = np.stack(core_masks, axis=0).astype(bool, copy=False)
    (
        field_core_traversal_reliability,
        field_core_traversal_counts,
        _,
        _,
    ) = compute_field_traversal_reliability(
        episode_statistics,
        core_masks_array,
        threshold_mode="peak_fraction",
        threshold_fraction=settings.threshold_fraction,
        threshold_quantile=settings.threshold_quantile,
        use_absolute_activations=settings.use_absolute_activations,
        minimum_traversals=settings.field_traversal_minimum_traversals,
    )
    record_timing(timing_seconds, "field_traversal_reliability", section_started_at)
    summaries = PlaceFieldSummaries(
        masks=field_masks_array,
        counts=field_counts,
        areas=field_areas,
        reliability=field_reliability,
        reliability_lift=np.full(unit_count, np.nan, dtype=np.float32),
        traversal_reliability=field_traversal_reliability,
        traversal_counts=field_traversal_counts,
        traversal_reliability_directional=field_traversal_reliability_directional,
        traversal_directional_counts=field_traversal_directional_counts,
        core_traversal_reliability=field_core_traversal_reliability,
        core_traversal_counts=field_core_traversal_counts,
        quantile_reliability=field_quantile_reliability,
        bin_consistency=field_bin_consistency,
        split_half_agreement=field_split_half_agreement,
        reliability_supported_fraction=field_reliability_supported_fraction,
        quantile_reliability_supported_fraction=field_quantile_reliability_supported_fraction,
        bin_consistency_supported_fraction=field_bin_consistency_supported_fraction,
        split_half_agreement_supported_fraction=field_split_half_agreement_supported_fraction,
    )
    return summaries, field_masks


def _fill_field_reliability_lift(
    summaries: PlaceFieldSummaries,
    field_masks: list[np.ndarray],
    reliability_lift_maps: np.ndarray,
) -> PlaceFieldSummaries:
    """In-field lift, the discriminating reliability scalar for sparse codes."""
    field_reliability_lift = np.full(summaries.counts.shape[0], np.nan, dtype=np.float32)
    for unit_index, field_mask in enumerate(field_masks):
        if field_mask.any():
            field_reliability_lift[unit_index] = mean_finite_inside_mask(
                reliability_lift_maps[unit_index],
                field_mask,
            )
    return PlaceFieldSummaries(
        masks=summaries.masks,
        counts=summaries.counts,
        areas=summaries.areas,
        reliability=summaries.reliability,
        reliability_lift=field_reliability_lift,
        traversal_reliability=summaries.traversal_reliability,
        traversal_counts=summaries.traversal_counts,
        traversal_reliability_directional=summaries.traversal_reliability_directional,
        traversal_directional_counts=summaries.traversal_directional_counts,
        core_traversal_reliability=summaries.core_traversal_reliability,
        core_traversal_counts=summaries.core_traversal_counts,
        quantile_reliability=summaries.quantile_reliability,
        bin_consistency=summaries.bin_consistency,
        split_half_agreement=summaries.split_half_agreement,
        reliability_supported_fraction=summaries.reliability_supported_fraction,
        quantile_reliability_supported_fraction=(
            summaries.quantile_reliability_supported_fraction
        ),
        bin_consistency_supported_fraction=summaries.bin_consistency_supported_fraction,
        split_half_agreement_supported_fraction=(
            summaries.split_half_agreement_supported_fraction
        ),
    )


def _compute_unit_map_summaries(
    settings: RateMapMetricSettings,
    *,
    reliability: ReliabilityMaps,
    supports_place_metrics: np.ndarray,
) -> UnitMapSummaries:
    """Whole-map mean/max per unit, over bins with enough revisit support."""
    summary_bin_support = (
        reliability.thresholded_visit_counts >= float(settings.per_bin_cv_min_episodes)
    )
    supported_reliability_maps = np.where(
        summary_bin_support[None, :, :], reliability.thresholded_maps, np.nan
    )
    supported_quantile_maps = np.where(
        summary_bin_support[None, :, :], reliability.quantile_maps, np.nan
    )
    supported_lift_maps = np.where(
        summary_bin_support[None, :, :], reliability.thresholded_lift_maps, np.nan
    )
    supported_quantile_lift_maps = np.where(
        summary_bin_support[None, :, :], reliability.quantile_lift_maps, np.nan
    )
    mean_bin_coefficient_of_variation = reduce_finite_maps(
        reliability.bin_coefficient_of_variation_maps,
        reducer="mean",
    )
    mean_bin_coefficient_of_variation[~supports_place_metrics] = np.nan
    return UnitMapSummaries(
        mean_reliability=reduce_finite_maps(supported_reliability_maps, reducer="mean"),
        max_reliability=reduce_finite_maps(supported_reliability_maps, reducer="max"),
        mean_reliability_lift=reduce_finite_maps(supported_lift_maps, reducer="mean"),
        max_reliability_lift=reduce_finite_maps(supported_lift_maps, reducer="max"),
        mean_quantile_reliability=reduce_finite_maps(supported_quantile_maps, reducer="mean"),
        max_quantile_reliability=reduce_finite_maps(supported_quantile_maps, reducer="max"),
        mean_quantile_reliability_lift=reduce_finite_maps(
            supported_quantile_lift_maps, reducer="mean"
        ),
        max_quantile_reliability_lift=reduce_finite_maps(
            supported_quantile_lift_maps, reducer="max"
        ),
        mean_bin_consistency=reduce_finite_maps(
            reliability.bin_consistency_maps, reducer="mean"
        ),
        max_bin_consistency=reduce_finite_maps(reliability.bin_consistency_maps, reducer="max"),
        mean_split_half_agreement=reduce_finite_maps(
            reliability.split_half_agreement_maps, reducer="mean"
        ),
        max_split_half_agreement=reduce_finite_maps(
            reliability.split_half_agreement_maps, reducer="max"
        ),
        mean_bin_coefficient_of_variation=mean_bin_coefficient_of_variation,
    )


def _compute_signed_map_diagnostics(rate_maps: np.ndarray) -> SignedMapDiagnostics:
    """Negative-mass diagnostics on the signed (un-clipped) rate maps."""
    flattened_rate_maps = rate_maps.reshape(rate_maps.shape[0], -1)
    peak_magnitude = np.nanmax(np.abs(flattened_rate_maps), axis=1)
    clean_signed_maps = np.nan_to_num(flattened_rate_maps, nan=0.0, posinf=0.0, neginf=0.0)
    peak_abs_activation = np.abs(clean_signed_maps).max(axis=1)
    positive_mass = np.clip(clean_signed_maps, 0.0, None).sum(axis=1)
    negative_mass = np.clip(-clean_signed_maps, 0.0, None).sum(axis=1)
    negative_fraction = negative_mass / np.maximum(positive_mass + negative_mass, 1e-9)
    return SignedMapDiagnostics(
        peak_magnitude=peak_magnitude,
        peak_abs_activation=peak_abs_activation,
        negative_fraction=negative_fraction,
        median_negative_fraction=(
            float(np.median(negative_fraction)) if negative_fraction.size else 0.0
        ),
        fraction_units_majority_negative=(
            float(np.mean(negative_fraction > 0.5)) if negative_fraction.size else 0.0
        ),
    )


def _rank_units(
    *,
    place_metrics: PlaceMetricScores,
    significant: np.ndarray,
    peak_magnitude: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Lexicographic eligibility ranking, with a flagged peak-activation fallback."""
    ranking_is_peak_preview = not bool(place_metrics.gates_assessable.any())
    if ranking_is_peak_preview:
        return np.argsort(peak_magnitude, kind="stable")[::-1], True
    return (
        np.lexsort(
            (
                np.arange(peak_magnitude.shape[0]),
                -np.nan_to_num(place_metrics.spatial_coherence, nan=-np.inf),
                -np.nan_to_num(place_metrics.spatial_information_bits, nan=-np.inf),
                ~place_metrics.replicates_split_half,
                ~significant,
                ~place_metrics.gates_assessable,
            )
        ),
        False,
    )


def compute_rate_map_metric_bundle(
    analysis_input: AnalysisInput,
    config: dict,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
) -> RateMapMetricBundle:
    timing_seconds: dict[str, float] = {}
    settings = resolve_rate_map_metric_settings(config)

    section_started_at = perf_counter()
    episode_statistics = get_or_compute_episode_bin_statistics(
        analysis_input,
        num_bins_x=settings.num_bins_x,
        num_bins_y=settings.num_bins_y,
        bounds=bounds,
    )
    record_timing(timing_seconds, "prepare_episode_statistics", section_started_at)
    if episode_statistics is None:
        raise ValueError("Cannot compute rate maps without any valid timesteps.")

    section_started_at = perf_counter()
    rate_map_result = get_or_compute_rate_maps(
        analysis_input,
        num_bins_x=settings.num_bins_x,
        num_bins_y=settings.num_bins_y,
        smoothing_sigma=settings.smoothing_sigma,
        min_occupancy=settings.min_occupancy,
        bounds=bounds,
        episode_statistics=episode_statistics,
    )
    record_timing(timing_seconds, "rate_map_computation", section_started_at)

    reliability_pass = _compute_reliability_pass(
        analysis_input,
        settings,
        bounds=rate_map_result.bounds,
        episode_statistics=episode_statistics,
        timing_seconds=timing_seconds,
    )
    rate_map_result.reliability_maps = reliability_pass.thresholded_maps
    rate_map_result.visited_episode_counts = reliability_pass.thresholded_visit_counts

    place_metrics, spatial_information_bits = _compute_place_metric_scores(
        analysis_input,
        settings,
        rate_map_result=rate_map_result,
        split_half_rate_map_correlation=reliability_pass.split_half_rate_map_correlation,
        timing_seconds=timing_seconds,
    )
    null, spatial_information_null_95 = _compute_spatial_information_null(
        analysis_input,
        settings,
        bounds=rate_map_result.bounds,
        spatial_information_bits=spatial_information_bits,
        passes_gates=place_metrics.passes_gates,
        gates_assessable=place_metrics.gates_assessable,
        timing_seconds=timing_seconds,
    )
    place_metrics = _with_null_excess(
        place_metrics,
        reliability_weighted_information_excess(
            spatial_information_bits,
            spatial_information_null_95,
            reliability_pass.split_half_rate_map_correlation,
        ),
    )

    flat_headings = None
    if analysis_input.heading is not None:
        heading_flat = np.asarray(analysis_input.heading, dtype=np.float64).reshape(-1)
        valid_mask = analysis_input.valid_mask
        if valid_mask is not None and not bool(np.all(valid_mask)):
            heading_flat = heading_flat[valid_mask.reshape(-1).astype(bool, copy=False)]
        if heading_flat.shape[0] == episode_statistics.flat_values.shape[0]:
            flat_headings = heading_flat
    field_summaries, field_masks = _compute_place_field_summaries(
        settings,
        episode_statistics=episode_statistics,
        place_metric_rate_maps=place_metrics.rate_maps,
        supports_place_metrics=place_metrics.supported,
        reliability=reliability_pass,
        timing_seconds=timing_seconds,
        flat_headings=flat_headings,
    )

    section_started_at = perf_counter()
    reliability = ReliabilityMaps(
        thresholded_maps=reliability_pass.thresholded_maps,
        thresholded_lift_maps=compute_reliability_lift_maps(
            reliability_pass.thresholded_maps, reliability_pass.thresholded_visit_counts
        ),
        thresholded_visit_counts=reliability_pass.thresholded_visit_counts,
        quantile_maps=reliability_pass.quantile_maps,
        quantile_lift_maps=compute_reliability_lift_maps(
            reliability_pass.quantile_maps, reliability_pass.thresholded_visit_counts
        ),
        quantile_visit_counts=reliability_pass.quantile_visit_counts,
        bin_consistency_maps=reliability_pass.bin_consistency_maps,
        bin_coefficient_of_variation_maps=reliability_pass.bin_coefficient_of_variation_maps,
        bin_consistency_visit_counts=reliability_pass.bin_consistency_visit_counts,
        split_half_agreement_maps=reliability_pass.split_half_agreement_maps,
        split_half_agreement_support_counts=(
            reliability_pass.split_half_agreement_support_counts
        ),
        split_half_rate_map_correlation=reliability_pass.split_half_rate_map_correlation,
        episode_rate_map_correlation=reliability_pass.episode_rate_map_correlation,
    )
    unit_summaries = _compute_unit_map_summaries(
        settings,
        reliability=reliability,
        supports_place_metrics=place_metrics.supported,
    )
    field_summaries = _fill_field_reliability_lift(
        field_summaries, field_masks, reliability.thresholded_lift_maps
    )
    signed_diagnostics = _compute_signed_map_diagnostics(rate_map_result.rate_maps)
    ranked_indices, ranking_is_peak_preview = _rank_units(
        place_metrics=place_metrics,
        significant=null.significant,
        peak_magnitude=signed_diagnostics.peak_magnitude,
    )
    record_timing(timing_seconds, "field_summaries_and_ranking", section_started_at)

    return RateMapMetricBundle(
        settings=settings,
        rate_map_result=rate_map_result,
        reliability=reliability,
        place_metrics=place_metrics,
        spatial_information_null=null,
        fields=field_summaries,
        summaries=unit_summaries,
        signed_diagnostics=signed_diagnostics,
        ranked_indices=ranked_indices,
        ranking_is_peak_preview=ranking_is_peak_preview,
        timing_seconds=timing_seconds,
    )
