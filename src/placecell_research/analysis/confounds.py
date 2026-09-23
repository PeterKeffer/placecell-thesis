"""Confound analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .base import AnalysisInput, AnalysisResult
from .place_cell_quality import get_or_compute_confound_scores
from .timing import log_timing


@dataclass(slots=True)
class ConfoundsModule:
    """Simple confound scores for step_displacement, heading, and time."""

    name: str = "confounds"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        del output_dir
        timing_seconds: dict[str, float] = {}
        section_started_at = perf_counter()
        confound_scores = get_or_compute_confound_scores(analysis_input)
        timing_seconds["compute_scores"] = perf_counter() - section_started_at

        section_started_at = perf_counter()
        step_displacement_scores_array = confound_scores["step_displacement_score"]
        heading_scores_array = confound_scores["heading_score"]
        time_scores_array = confound_scores["time_score"]
        max_available_confound_score = confound_scores["max_available_confound_score"]
        result = AnalysisResult(
            metrics={
                "mean_step_displacement_score": (
                    float(step_displacement_scores_array.mean())
                    if len(step_displacement_scores_array)
                    else 0.0
                ),
                "mean_heading_score": (
                    float(heading_scores_array.mean()) if len(heading_scores_array) else 0.0
                ),
                "mean_time_score": (
                    float(time_scores_array.mean()) if len(time_scores_array) else 0.0
                ),
                "mean_max_available_confound_score": float(max_available_confound_score.mean())
                if len(max_available_confound_score)
                else 0.0,
            },
            per_unit_metrics={
                "step_displacement_score": step_displacement_scores_array,
                "heading_score": heading_scores_array,
                "time_score": time_scores_array,
                "max_available_confound_score": max_available_confound_score,
            },
            figures={},
            tables={},
            metadata={"confounds_timing_seconds": timing_seconds},
        )
        timing_seconds["assemble_result"] = perf_counter() - section_started_at
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )
        return result
