"""Comparative representation-drift analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import AnalysisInput, AnalysisResult
from .helpers import save_heatmap


def _flatten_valid_representation(analysis_input: AnalysisInput) -> np.ndarray:
    valid_representation = analysis_input.representation[analysis_input.valid_mask]
    if valid_representation.size == 0:
        return np.zeros((0, analysis_input.representation.shape[-1]), dtype=np.float32)
    return valid_representation.astype(np.float32, copy=False)


def _mean_feature_vector(analysis_input: AnalysisInput) -> np.ndarray:
    flattened = _flatten_valid_representation(analysis_input)
    if flattened.shape[0] == 0:
        return np.zeros((analysis_input.representation.shape[-1],), dtype=np.float32)
    return flattened.mean(axis=0).astype(np.float32, copy=False)


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1.0e-8 or second_norm < 1.0e-8:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


@dataclass(slots=True)
class RepresentationDriftModule:
    """Cross-input similarity in representation space."""

    name: str = "representation_drift"
    cost_tier: str = "heavy"

    def run(
        self,
        inputs: list[AnalysisInput],
        labels: list[str],
        output_dir: Path,
        config: dict,
    ) -> AnalysisResult:
        if len(inputs) < 2:
            raise ValueError("Representation drift analysis needs at least two inputs.")

        feature_vectors = [_mean_feature_vector(analysis_input) for analysis_input in inputs]
        similarity_matrix = np.eye(len(inputs), dtype=np.float32)
        for row_index, first_vector in enumerate(feature_vectors):
            for column_index, second_vector in enumerate(feature_vectors):
                if column_index <= row_index:
                    continue
                similarity = _cosine_similarity(first_vector, second_vector)
                similarity_matrix[row_index, column_index] = similarity
                similarity_matrix[column_index, row_index] = similarity

        module_dir = output_dir / self.name
        heatmap_path = save_heatmap(
            module_dir
            / f"representation_drift__{inputs[0].source_name}__{inputs[0].split_name}.png",
            similarity_matrix,
            "Representation drift similarity",
            labels,
            labels,
        )
        upper_triangle = similarity_matrix[np.triu_indices(len(inputs), k=1)]
        return AnalysisResult(
            metrics={
                "mean_pairwise_similarity": float(upper_triangle.mean())
                if len(upper_triangle)
                else 0.0,
                "min_pairwise_similarity": float(upper_triangle.min())
                if len(upper_triangle)
                else 0.0,
            },
            per_unit_metrics={},
            figures={"similarity_heatmap": heatmap_path},
            tables={},
        )
