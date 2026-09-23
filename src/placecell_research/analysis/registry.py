"""Analysis registry and orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from .band_score import BandScoreModule
from .base import AnalysisInput, AnalysisResult
from .border_score import BorderScoreModule
from .code_timescale import CodeTimescaleModule
from .cognitive_map_geometry import CognitiveMapGeometryModule
from .combined_episode_dynamics import CombinedEpisodeDynamicsModule
from .conformal_isometry import ConformalIsometryModule
from .confounds import ConfoundsModule
from .dataset_coverage import DatasetCoverageModule
from .decode_convergence import DecodeConvergenceModule
from .decode_extrapolation import DecodeExtrapolationModule
from .decode_region import DecodeRegionModule
from .decode_structural_room import DecodeStructuralRoomModule
from .decode_xy import DecodeXYModule
from .directionality import DirectionalityModule
from .effective_dimensionality import EffectiveDimensionalityModule
from .eigenmode_morphology import EigenmodeMorphologyModule
from .episode_dynamics import EpisodeDynamicsModule
from .excess_stability import ExcessStabilityModule
from .field_stability import (
    FieldStabilityMetricsModule,
    FieldStabilitySummaryModule,
    FieldStabilityTrajectoriesModule,
)
from .fourier_ring import FourierRingModule
from .gridness import GridnessModule
from .head_direction_tuning import HeadDirectionTuningModule
from .heading_rate_map_overlay import HeadingRateMapOverlayModule
from .landmark_visibility_error import LandmarkVisibilityErrorModule
from .manifold_topology import ManifoldTopologyModule
from .neighborhood_preservation import NeighborhoodPreservationModule
from .per_episode_gridness import PerEpisodeGridnessModule
from .per_episode_rate_maps import PerEpisodeRateMapsModule
from .place_field_detection import PlaceFieldDetectionModule
from .place_field_overlay import PlaceFieldOverlayModule
from .population_coverage import PopulationCoverageModule
from .probing import ProbingModule
from .rate_maps import (
    RateMapBinConsistencyModule,
    RateMapCodingPurityModule,
    RateMapEpisodeCorrelationModule,
    RateMapExtraReliabilityPanelsModule,
    RateMapFieldsModule,
    RateMapGridModule,
    RateMapPanelModule,
    RateMapReliabilityModule,
    RateMapSplitHalfModule,
)
from .reanchoring_gridness import ReanchoringGridnessModule
from .redundancy_metrics import RedundancyMetricsModule
from .remapping import RemappingComparisonModule
from .representation_drift import RepresentationDriftModule
from .selectivity import SelectivityPartitionModule, SpatialCodeTypeModule
from .sparsity_metrics import SparsityModule
from .spatial_code_dynamics import SpatialCodeDynamicsModule
from .spatial_info import SpatialInfoModule
from .sr_oracle import SuccessorOracleComparisonModule
from .successor_return import SuccessorReturnComparisonModule
from .timing import log_timing, log_timing_line, with_deferred_module_timing_logs
from .topology_umap import (
    TopologyIsomapPooledModule,
    TopologyMDSPooledModule,
    TopologyUMAPPooledModule,
    TopologyUMAPStepsModule,
)
from .transition_geometry import (
    TransitionGeometryAlignmentModule,
    TransitionGeometryGraphModule,
    TransitionGeometryPanelModule,
)
from .uncertainty_signature import UncertaintySignatureModule
from .vector_cell_score import BoundaryVectorScoreModule, ObjectVectorScoreModule
from .within_heading_reliability import WithinHeadingReliabilityModule

ANALYSIS_MODULES = {
    "dataset_coverage": DatasetCoverageModule,
    "rate_map_fields": RateMapFieldsModule,
    "rate_map_reliability": RateMapReliabilityModule,
    "within_heading_reliability": WithinHeadingReliabilityModule,
    "rate_map_bin_consistency": RateMapBinConsistencyModule,
    "rate_map_split_half": RateMapSplitHalfModule,
    "rate_map_episode_correlation": RateMapEpisodeCorrelationModule,
    "rate_map_coding_purity": RateMapCodingPurityModule,
    "rate_map_panel": RateMapPanelModule,
    "rate_map_grid": RateMapGridModule,
    "rate_map_extra_reliability_panels": RateMapExtraReliabilityPanelsModule,
    "population_coverage": PopulationCoverageModule,
    "field_stability_metrics": FieldStabilityMetricsModule,
    "field_stability_summary": FieldStabilitySummaryModule,
    "field_stability_trajectories": FieldStabilityTrajectoriesModule,
    "redundancy_metrics": RedundancyMetricsModule,
    "spatial_info": SpatialInfoModule,
    "spatial_code_dynamics": SpatialCodeDynamicsModule,
    "directionality": DirectionalityModule,
    "selectivity_partition": SelectivityPartitionModule,
    "spatial_code_type": SpatialCodeTypeModule,
    "head_direction_tuning": HeadDirectionTuningModule,
    "heading_rate_map_overlay": HeadingRateMapOverlayModule,
    "gridness": GridnessModule,
    "per_episode_gridness": PerEpisodeGridnessModule,
    "per_episode_rate_maps": PerEpisodeRateMapsModule,
    "reanchoring_gridness": ReanchoringGridnessModule,
    "sparsity": SparsityModule,
    "code_timescale": CodeTimescaleModule,
    "excess_stability": ExcessStabilityModule,
    "decode_xy": DecodeXYModule,
    "decode_convergence": DecodeConvergenceModule,
    "uncertainty_signature": UncertaintySignatureModule,
    "landmark_visibility_error": LandmarkVisibilityErrorModule,
    "decode_region": DecodeRegionModule,
    "decode_structural_room": DecodeStructuralRoomModule,
    "probing": ProbingModule,
    "topology_umap_steps": TopologyUMAPStepsModule,
    "topology_umap_pooled": TopologyUMAPPooledModule,
    "topology_mds_pooled": TopologyMDSPooledModule,
    "topology_isomap_pooled": TopologyIsomapPooledModule,
    "neighborhood_preservation": NeighborhoodPreservationModule,
    "transition_geometry_graph": TransitionGeometryGraphModule,
    "transition_geometry_alignment": TransitionGeometryAlignmentModule,
    "transition_geometry_panel": TransitionGeometryPanelModule,
    "eigenmode_morphology": EigenmodeMorphologyModule,
    "cognitive_map_geometry": CognitiveMapGeometryModule,
    "confounds": ConfoundsModule,
    "place_field_detection": PlaceFieldDetectionModule,
    "episode_dynamics": EpisodeDynamicsModule,
    "place_field_overlay": PlaceFieldOverlayModule,
    "band_score": BandScoreModule,
    "border_score": BorderScoreModule,
    "boundary_vector_score": BoundaryVectorScoreModule,
    "object_vector_score": ObjectVectorScoreModule,
    "effective_dimensionality": EffectiveDimensionalityModule,
    "conformal_isometry": ConformalIsometryModule,
    "decode_extrapolation": DecodeExtrapolationModule,
    "manifold_topology": ManifoldTopologyModule,
    "fourier_ring": FourierRingModule,
}

COMPARATIVE_MODULES = {
    "remapping_comparison": RemappingComparisonModule,
    "representation_drift": RepresentationDriftModule,
    "combined_episode_dynamics": CombinedEpisodeDynamicsModule,
    "sr_oracle": SuccessorOracleComparisonModule,
    "successor_return": SuccessorReturnComparisonModule,
}

_COST_ORDER = {"light": 0, "standard": 1, "heavy": 2}
_MODULE_TIMING_METADATA_KEYS = (
    "rate_map_timing_seconds",
    "decode_timing_seconds",
    "field_stability_timing_seconds",
    "per_episode_gridness_timing_seconds",
    "redundancy_timing_seconds",
    "topology_umap_timing_seconds",
    "topology_manifold_timing_seconds",
    "neighborhood_preservation_timing_seconds",
    "transition_geometry_timing_seconds",
    "confounds_timing_seconds",
)


def _should_run_module(module_cost_tier: str, configured_max_cost_tier: str) -> bool:
    return _COST_ORDER[module_cost_tier] <= _COST_ORDER[configured_max_cost_tier]


def _strip_artifact_paths(result: AnalysisResult) -> AnalysisResult:
    """Return a notebook-friendly copy with file outputs removed."""
    return AnalysisResult(
        metrics=dict(result.metrics),
        per_unit_metrics=dict(result.per_unit_metrics),
        figures={},
        tables={},
        metadata=dict(result.metadata),
    )


def _store_analysis_module_timing(
    result: AnalysisResult,
    *,
    elapsed_seconds: float,
) -> None:
    result.metadata.setdefault("analysis_module_timing_seconds", {})["total"] = elapsed_seconds


def _emit_analysis_module_timing(
    *,
    source_name: str,
    split_name: str,
    module_name: str,
    elapsed_seconds: float,
) -> None:
    log_timing_line(
        "analysis_module",
        (source_name, split_name, module_name),
        {"total": elapsed_seconds},
        section_names=("total",),
    )


def _emit_deferred_module_timing_logs(
    result: AnalysisResult,
    *,
    source_name: str,
    split_name: str,
    module_name: str,
) -> None:
    for metadata_key in _MODULE_TIMING_METADATA_KEYS:
        timing_seconds = result.metadata.get(metadata_key)
        if not isinstance(timing_seconds, dict) or not timing_seconds:
            continue
        log_timing(module_name, source_name, split_name, timing_seconds)


def _store_comparative_module_timing(
    result: AnalysisResult,
    *,
    elapsed_seconds: float,
) -> None:
    result.metadata.setdefault("analysis_module_timing_seconds", {})["total"] = elapsed_seconds


def _emit_comparative_module_timing(
    *,
    analysis_name: str,
    module_name: str,
    elapsed_seconds: float,
) -> None:
    log_timing_line(
        "comparative_analysis_module",
        (analysis_name, module_name),
        {"total": elapsed_seconds},
        section_names=("total",),
    )


def run_analysis_modules(
    analysis_input: AnalysisInput,
    output_dir: Path | None,
    config: dict[str, Any],
    module_names: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, AnalysisResult]:
    """Run configured single-source modules."""
    results: dict[str, AnalysisResult] = {}
    max_cost_tier = str(config.get("max_cost_tier", "heavy"))
    if output_dir is None:
        with TemporaryDirectory(prefix="placecell_analysis_") as temporary_dir:
            scratch_dir = Path(temporary_dir)
            for module_name in module_names:
                module = ANALYSIS_MODULES[module_name]()
                if not _should_run_module(module.cost_tier, max_cost_tier):
                    continue
                section_started_at = perf_counter()
                module_result = module.run(
                    analysis_input,
                    scratch_dir,
                    with_deferred_module_timing_logs(config),
                )
                elapsed_seconds = perf_counter() - section_started_at
                _store_analysis_module_timing(module_result, elapsed_seconds=elapsed_seconds)
                results[module_name] = _strip_artifact_paths(module_result)
                if progress_callback is not None:
                    progress_callback(module_name)
                _emit_deferred_module_timing_logs(
                    results[module_name],
                    source_name=analysis_input.source_name,
                    split_name=analysis_input.split_name,
                    module_name=module_name,
                )
                _emit_analysis_module_timing(
                    source_name=analysis_input.source_name,
                    split_name=analysis_input.split_name,
                    module_name=module_name,
                    elapsed_seconds=elapsed_seconds,
                )
        return results

    for module_name in module_names:
        module = ANALYSIS_MODULES[module_name]()
        if not _should_run_module(module.cost_tier, max_cost_tier):
            continue
        section_started_at = perf_counter()
        module_result = module.run(
            analysis_input,
            output_dir,
            with_deferred_module_timing_logs(config),
        )
        elapsed_seconds = perf_counter() - section_started_at
        _store_analysis_module_timing(module_result, elapsed_seconds=elapsed_seconds)
        results[module_name] = module_result
        if progress_callback is not None:
            progress_callback(module_name)
        _emit_deferred_module_timing_logs(
            module_result,
            source_name=analysis_input.source_name,
            split_name=analysis_input.split_name,
            module_name=module_name,
        )
        _emit_analysis_module_timing(
            source_name=analysis_input.source_name,
            split_name=analysis_input.split_name,
            module_name=module_name,
            elapsed_seconds=elapsed_seconds,
        )
    return results


def run_comparative_modules(
    analysis_inputs: list[AnalysisInput],
    labels: list[str],
    output_dir: Path | None,
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, AnalysisResult]:
    """Run configured comparative modules."""
    results: dict[str, AnalysisResult] = {}
    comparative_config = config.get("comparative", {}) if isinstance(config, dict) else {}
    if output_dir is None:
        with TemporaryDirectory(prefix="placecell_comparative_analysis_") as temporary_dir:
            scratch_dir = Path(temporary_dir)
            for analysis_name, analysis_config in comparative_config.items():
                if not analysis_config.get("enabled", True):
                    continue
                module_name = str(analysis_config.get("module", ""))
                if module_name not in COMPARATIVE_MODULES:
                    raise KeyError(f"Unknown comparative analysis module: {module_name}")
                module = COMPARATIVE_MODULES[module_name]()
                max_cost_tier = str(
                    analysis_config.get("max_cost_tier", config.get("max_cost_tier", "heavy"))
                )
                if not _should_run_module(module.cost_tier, max_cost_tier):
                    continue
                section_started_at = perf_counter()
                module_result = module.run(analysis_inputs, labels, scratch_dir, analysis_config)
                elapsed_seconds = perf_counter() - section_started_at
                _store_comparative_module_timing(module_result, elapsed_seconds=elapsed_seconds)
                results[analysis_name] = _strip_artifact_paths(module_result)
                if progress_callback is not None:
                    progress_callback(analysis_name)
                _emit_comparative_module_timing(
                    analysis_name=analysis_name,
                    module_name=module_name,
                    elapsed_seconds=elapsed_seconds,
                )
        return results

    for analysis_name, analysis_config in comparative_config.items():
        if not analysis_config.get("enabled", True):
            continue
        module_name = str(analysis_config.get("module", ""))
        if module_name not in COMPARATIVE_MODULES:
            raise KeyError(f"Unknown comparative analysis module: {module_name}")
        module = COMPARATIVE_MODULES[module_name]()
        max_cost_tier = str(
            analysis_config.get("max_cost_tier", config.get("max_cost_tier", "heavy"))
        )
        if not _should_run_module(module.cost_tier, max_cost_tier):
            continue
        section_started_at = perf_counter()
        module_result = module.run(analysis_inputs, labels, output_dir, analysis_config)
        elapsed_seconds = perf_counter() - section_started_at
        _store_comparative_module_timing(module_result, elapsed_seconds=elapsed_seconds)
        results[analysis_name] = module_result
        if progress_callback is not None:
            progress_callback(analysis_name)
        _emit_comparative_module_timing(
            analysis_name=analysis_name,
            module_name=module_name,
            elapsed_seconds=elapsed_seconds,
        )
    return results
