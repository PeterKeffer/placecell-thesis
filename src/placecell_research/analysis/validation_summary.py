"""Aggregate analysis outputs into summary files."""

from __future__ import annotations

import json
from pathlib import Path

from .base import AnalysisResult
from .helpers import write_csv


def write_summary_json(path: Path, summary: dict) -> Path:
    """Write a summary JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def write_summary_csv(path: Path, summary: dict[str, float]) -> Path:
    """Write a flat summary CSV."""
    return write_csv(
        path, ["metric", "value"], [[key, value] for key, value in sorted(summary.items())]
    )


def flatten_analysis_results(
    single_results: dict[str, AnalysisResult],
    comparative_results: dict[str, AnalysisResult],
) -> dict[str, float]:
    """Flatten analysis results into one summary dict."""
    flattened: dict[str, float] = {}
    for target_name, result in single_results.items():
        flattened.update(
            {f"{target_name}.{metric_name}": value for metric_name, value in result.metrics.items()}
        )
    for analysis_name, result in comparative_results.items():
        flattened.update(
            {
                f"comparative.{analysis_name}.{metric_name}": value
                for metric_name, value in result.metrics.items()
            }
        )
    return flattened
