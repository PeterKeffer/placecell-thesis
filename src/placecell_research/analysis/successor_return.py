"""Direct held-out successor-feature return validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import AnalysisInput, AnalysisResult


def _fit_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    centered_target = target - target.mean(axis=0, keepdims=True)
    target_standard_deviation = float(np.sqrt(np.mean(np.square(centered_target))))
    if target_standard_deviation > np.finfo(np.float64).eps:
        nrmse = rmse / target_standard_deviation
        r2 = 1.0 - float(np.sum(np.square(error)) / np.sum(np.square(centered_target)))
    else:
        nrmse = float("nan")
        r2 = float("nan")

    denominator = float(np.sum(np.square(prediction)))
    epsilon = float(np.finfo(np.float64).eps)
    if denominator > epsilon and target_standard_deviation > epsilon:
        scale = float(np.sum(prediction * target)) / denominator
        scaled_error = scale * prediction - target
        r2_scaled = 1.0 - float(
            np.sum(np.square(scaled_error)) / np.sum(np.square(centered_target))
        )
    else:
        scale = float("nan")
        r2_scaled = float("nan")

    prediction_norm = np.linalg.norm(prediction, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    nonzero = (prediction_norm > 0.0) & (target_norm > 0.0)
    cosine = float(
        np.mean(
            np.sum(prediction[nonzero] * target[nonzero], axis=-1)
            / (prediction_norm[nonzero] * target_norm[nonzero])
        )
    ) if np.any(nonzero) else float("nan")
    mean_target_norm = float(target_norm.mean())
    norm_ratio = (
        float(prediction_norm.mean()) / mean_target_norm
        if mean_target_norm > np.finfo(np.float64).eps
        else float("nan")
    )
    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
        "r2_scaled": r2_scaled,
        "fitted_scale": scale,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
    }


def discounted_feature_returns(
    features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    discount_gamma: float,
    normalized: bool,
) -> np.ndarray:
    """Compute episode-bounded Monte Carlo successor-feature targets."""
    feature_array = np.asarray(features, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if feature_array.ndim != 3 or valid.shape != feature_array.shape[:2]:
        raise ValueError(
            "features must be [episode, time, feature] and valid_mask must match its "
            f"first two axes; got {feature_array.shape} and {valid.shape}."
        )
    if not 0.0 <= discount_gamma < 1.0:
        raise ValueError(f"discount_gamma must be in [0, 1), got {discount_gamma}.")

    feature_weight = 1.0 - discount_gamma if normalized else 1.0
    returns = np.zeros_like(feature_array)
    running_return = np.zeros((feature_array.shape[0], feature_array.shape[2]), dtype=np.float64)
    for timestep in range(feature_array.shape[1] - 1, -1, -1):
        step_valid = valid[:, timestep]
        running_return[~step_valid] = 0.0
        running_return[step_valid] = (
            feature_weight * feature_array[step_valid, timestep]
            + discount_gamma * running_return[step_valid]
        )
        returns[step_valid, timestep] = running_return[step_valid]
    return returns


def successor_return_metrics(
    successor: np.ndarray,
    features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    discount_gamma: float,
    normalized: bool,
    shuffle_seed: int,
) -> dict[str, float]:
    """Score a successor output against held-out Monte Carlo feature returns."""
    successor_array = np.asarray(successor, dtype=np.float64)
    feature_array = np.asarray(features, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if successor_array.shape != feature_array.shape:
        raise ValueError(
            "successor and features must have the same [episode, time, feature] shape; "
            f"got {successor_array.shape} and {feature_array.shape}."
        )
    returns = discounted_feature_returns(
        feature_array,
        valid,
        discount_gamma=discount_gamma,
        normalized=normalized,
    )
    if not np.any(valid):
        raise ValueError("successor return validation requires at least one valid step.")

    flat_successor = successor_array[valid]
    flat_features = feature_array[valid]
    flat_returns = returns[valid]
    successor_fit = _fit_metrics(flat_successor, flat_returns)
    current_feature_fit = _fit_metrics(flat_features, flat_returns)
    feature_weight = 1.0 - discount_gamma if normalized else 1.0
    immediate_feature_fit = _fit_metrics(feature_weight * flat_features, flat_returns)
    shuffled_indices = np.random.default_rng(shuffle_seed).permutation(len(flat_returns))
    shuffled_returns = flat_returns[shuffled_indices]
    shuffled_fit = _fit_metrics(flat_successor, shuffled_returns)

    transition_valid = valid[:, :-1] & valid[:, 1:]
    if np.any(transition_valid):
        bellman_target = (
            feature_weight * feature_array[:, :-1]
            + discount_gamma * successor_array[:, 1:]
        )[transition_valid]
        bellman_fit = _fit_metrics(successor_array[:, :-1][transition_valid], bellman_target)
    else:
        bellman_fit = {key: float("nan") for key in _fit_metrics(flat_successor, flat_returns)}

    return {
        "mc_return_rmse": successor_fit["rmse"],
        "mc_return_nrmse": successor_fit["nrmse"],
        "mc_return_r2": successor_fit["r2"],
        "mc_return_r2_scaled": successor_fit["r2_scaled"],
        "mc_return_fitted_scale": successor_fit["fitted_scale"],
        "mc_return_cosine": successor_fit["cosine"],
        "mc_return_norm_ratio": successor_fit["norm_ratio"],
        "current_feature_rmse": current_feature_fit["rmse"],
        "current_feature_nrmse": current_feature_fit["nrmse"],
        "current_feature_r2": current_feature_fit["r2"],
        "immediate_feature_rmse": immediate_feature_fit["rmse"],
        "immediate_feature_nrmse": immediate_feature_fit["nrmse"],
        "immediate_feature_r2": immediate_feature_fit["r2"],
        "immediate_feature_r2_scaled": immediate_feature_fit["r2_scaled"],
        "immediate_feature_cosine": immediate_feature_fit["cosine"],
        "mc_nrmse_improvement_over_current": (
            current_feature_fit["nrmse"] - successor_fit["nrmse"]
        ),
        "mc_nrmse_improvement_over_immediate": (
            immediate_feature_fit["nrmse"] - successor_fit["nrmse"]
        ),
        "shuffled_return_nrmse": shuffled_fit["nrmse"],
        "shuffled_return_r2": shuffled_fit["r2"],
        "shuffled_return_r2_scaled": shuffled_fit["r2_scaled"],
        "bellman_residual_rmse": bellman_fit["rmse"],
        "bellman_residual_nrmse": bellman_fit["nrmse"],
    }


@dataclass(slots=True)
class SuccessorReturnComparisonModule:
    """Compare a learned psi directly with empirical discounted returns of its phi."""

    name: str = "successor_return"
    cost_tier: str = "light"

    def run(
        self,
        inputs: list[AnalysisInput],
        labels: list[str],
        output_dir: Path,
        config: dict[str, Any],
    ) -> AnalysisResult:
        del output_dir
        if len(inputs) != 2:
            raise ValueError(
                f"successor_return requires exactly two inputs, got {len(inputs)}."
            )
        successor_input, feature_input = inputs
        for metadata_key in ("dataset_artifact_id", "split_artifact_id"):
            successor_reference = successor_input.metadata.get(metadata_key)
            feature_reference = feature_input.metadata.get(metadata_key)
            if (
                successor_reference is not None
                and feature_reference is not None
                and successor_reference != feature_reference
            ):
                raise ValueError(
                    "successor_return inputs must be sample-aligned; "
                    f"{metadata_key} differs "
                    f"({successor_reference!r} vs {feature_reference!r})."
                )

        successor = np.asarray(successor_input.representation)
        features = np.asarray(feature_input.representation)
        if successor.shape != features.shape or successor.ndim != 3:
            raise ValueError(
                "successor_return requires psi and phi with the same "
                f"[episode, time, feature] shape, got {successor.shape} and {features.shape}."
            )
        if successor_input.position_xy.shape != feature_input.position_xy.shape or not np.allclose(
            successor_input.position_xy,
            feature_input.position_xy,
            atol=1e-5,
            equal_nan=True,
        ):
            raise ValueError("successor_return inputs must share aligned positions.")

        successor_valid = np.asarray(successor_input.valid_mask, dtype=bool)
        feature_valid = np.asarray(feature_input.valid_mask, dtype=bool)
        if (
            successor_valid.shape != successor.shape[:2]
            or feature_valid.shape != successor.shape[:2]
        ):
            raise ValueError(
                "successor_return valid masks must match the shared [episode, time] axes."
            )
        valid = successor_valid & feature_valid
        discount_gamma = float(config.get("successor_return_discount_gamma", 0.95))
        normalized = bool(config.get("successor_return_normalized", True))
        metrics = successor_return_metrics(
            successor,
            features,
            valid,
            discount_gamma=discount_gamma,
            normalized=normalized,
            shuffle_seed=int(config.get("successor_return_shuffle_seed", 0)),
        )
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={
                "successor_return_discount_gamma": discount_gamma,
                "successor_return_normalized": normalized,
                "successor_return_valid_steps": int(valid.sum()),
                "successor_return_valid_transitions": int(
                    np.sum(valid[:, :-1] & valid[:, 1:])
                ),
                "successor_label": labels[0] if labels else successor_input.label,
                "feature_label": labels[1] if len(labels) > 1 else feature_input.label,
            },
        )
