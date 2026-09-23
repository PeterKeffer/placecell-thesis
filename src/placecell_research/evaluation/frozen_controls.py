"""Matched code transformations and controlled recurrent inference interventions."""

from __future__ import annotations

import numpy as np
import torch

from placecell_research.evaluation.matched_decode import MatchedPositionDecoder
from placecell_research.numerics.error_metrics import rmse_from_coordinate_mse
from placecell_research.numerics.rate_map_kernels import (
    compute_rate_maps,
    skaggs_spatial_information,
)


def fixed_topk(values: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest signed activations per sample, without fitting or rescaling."""
    if not 1 <= k <= values.shape[-1]:
        raise ValueError("k must be between 1 and the representation width.")
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    indices = tensor.topk(k, dim=-1).indices
    return torch.zeros_like(tensor).scatter(-1, indices, tensor.gather(-1, indices)).numpy()


def code_organisation_metrics(
    values: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[dict, dict]:
    """Signed activity statistics and explicitly positive-part tuning and field coverage."""
    rows = values[valid.astype(bool)]
    if not len(rows):
        raise ValueError("Organisation metrics require valid samples.")
    maps = compute_rate_maps(
        np.maximum(values, 0), positions, valid, 60, 60, 0.3, 1e-6, bounds=bounds
    )
    visited = maps.raw_occupancy > 0
    rates = np.nan_to_num(maps.rate_maps, nan=0.0)
    peaks = rates[:, visited].max(axis=1)
    fields = (rates >= 0.2 * peaks[:, None, None]) & (peaks[:, None, None] > 0)
    fields &= visited[None]
    information = skaggs_spatial_information(maps.rate_maps, maps.occupancy)
    metrics = {
        "fraction_active_abs_gt_1e-4": float((np.abs(rows) > 1e-4).mean()),
        "fraction_nonzero": float((rows != 0).mean()),
        "recruited_fraction_abs_gt_1e-4": float((np.abs(rows) > 1e-4).any(0).mean()),
        "activation_rms": float(np.sqrt(np.square(rows, dtype=np.float64).mean())),
        "mean_unit_std": float(rows.std(0, dtype=np.float64).mean()),
        "positive_part_mean_spatial_information_bits": float(np.mean(information)),
        "positive_part_field_unit_fraction": float(fields.any(axis=(1, 2)).mean()),
        "positive_part_visited_bin_coverage": float(fields.any(0)[visited].mean()),
    }
    arrays = {
        "positive_part_rate_maps": maps.rate_maps,
        "raw_occupancy": maps.raw_occupancy,
        "positive_part_spatial_information_bits": information,
        "positive_part_field_area_fraction": fields[:, visited].mean(1),
        "unit_recruitment": (np.abs(rows) > 1e-4).mean(0),
    }
    return metrics, arrays


@torch.inference_mode()
def perturbed_codes(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    onset: int,
    duration: int,
    reset: bool = False,
    blackout: bool = False,
    source: str = "encoder.place_codes",
) -> torch.Tensor:
    """Replay identical chunk boundaries; reset all carried state and/or zero latent input."""
    steps = batch["valid_steps"].shape[1]
    if not 0 < onset < onset + duration < steps:
        raise ValueError("The intervention must have clean observations before and after it.")
    if "latent" not in batch:
        raise ValueError("This protocol defines blackout as zero latent observations.")
    training = model.training
    model.eval()
    state = None
    outputs = []
    try:
        for start, stop in zip(
            (0, onset, onset + duration), (onset, onset + duration, steps), strict=False
        ):
            chunk = {key: value[:, start:stop] for key, value in batch.items()}
            if start == onset:
                if reset:
                    state = None
                if blackout:
                    chunk["latent"] = torch.zeros_like(chunk["latent"])
            bundle, state = model.forward_chunk(chunk, state)
            outputs.append(bundle.get_representation(source))
        return torch.cat(outputs, dim=1)
    finally:
        model.train(training)


def localization_curves(
    decoder: MatchedPositionDecoder,
    codes: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Frozen clean-data decoder; retain paired per-episode errors for recovery analysis."""
    if decoder.selected is None:
        raise ValueError("Select the decoder on clean validation data before perturbing test data.")
    predictions = (codes - decoder.mean) @ decoder.weights[decoder.selected] + decoder.target_mean
    error = np.where(valid[..., None], (predictions - positions) ** 2, np.nan)
    per_axis_mse = np.nanmean(error, axis=0)
    center = np.nanmean(np.where(valid[..., None], positions, np.nan), axis=0)
    total = np.nansum(np.where(valid[..., None], (positions - center) ** 2, np.nan), axis=0)
    residual = np.nansum(error, axis=0)
    r2 = np.full_like(total, np.nan)
    np.divide(residual, total, out=r2, where=total > 0)
    return {
        "decode_rmse_by_step": rmse_from_coordinate_mse(per_axis_mse),
        "decode_r2_by_step": (1 - r2).mean(-1),
        "episode_squared_error_xy": error,
        "episode_localization_error": np.sqrt(error.sum(-1)),
    }
