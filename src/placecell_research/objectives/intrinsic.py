"""Model-global intrinsic losses that are not declared as objective entries."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from placecell_research.config.schema import SpatialModelConfig
from placecell_research.spatial_model.batch_access import resolve_kinematics
from placecell_research.spatial_model.types import RepresentationBundle

from .masking import flatten_valid_steps, transition_mask, valid_steps


def orthonormality_loss(codes: Tensor, variance_floor: float = 0.0) -> Tensor:
    """Single-view orthonormality regularizer with an optional per-dimension std floor."""

    sample_count, feature_dim = codes.shape
    if sample_count < 2 or feature_dim < 1:
        return torch.zeros((), device=codes.device, dtype=codes.dtype)
    normalized_codes = F.normalize(codes, dim=-1)
    feature_gram = normalized_codes.T @ normalized_codes
    row_norm_squared = normalized_codes.square().sum(dim=1)
    off_diagonal_sum = feature_gram.square().sum() - row_norm_squared.square().sum()
    diagonal_penalty = (row_norm_squared - 1.0).square().sum()
    decorrelation_loss = (off_diagonal_sum + diagonal_penalty) / (
        sample_count * max(sample_count - 1, 1)
    )
    norm_preservation_loss = (codes.norm(dim=-1) - 1.0).pow(2).mean()
    loss_value = decorrelation_loss + norm_preservation_loss
    if variance_floor > 0.0:
        per_dimension_std = codes.std(dim=0, unbiased=False)
        floor = torch.as_tensor(variance_floor, device=codes.device, dtype=codes.dtype)
        loss_value = loss_value + F.relu(floor - per_dimension_std).mean()
    return loss_value


def compute_intrinsic_losses(
    bundle: RepresentationBundle,
    batch: dict[str, Tensor],
    model_config: SpatialModelConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute always-available model-global losses controlled by model config."""

    device = bundle.infer_device()
    zero = torch.zeros((), device=device)
    extra_loss = zero
    metrics: dict[str, Tensor] = {}

    orthonormality = model_config.orthonormality
    if orthonormality.encoder_weight > 0.0:
        encoder_codes = bundle.get_representation("encoder.place_codes")
        encoder_mask = valid_steps(bundle, batch)
        valid_encoder_codes = flatten_valid_steps(encoder_codes, encoder_mask)
        if valid_encoder_codes.shape[0] > 1:
            encoder_loss = orthonormality.encoder_weight * orthonormality_loss(
                valid_encoder_codes,
                variance_floor=orthonormality.variance_floor,
            )
            extra_loss = extra_loss + encoder_loss
            metrics["loss/orthonormality_encoder"] = encoder_loss.detach()

    if orthonormality.predictor_weight > 0.0:
        predictor_codes = bundle.get_representation("predictor.place_codes")[:, 1:]
        predictor_mask = transition_mask(bundle, batch, horizon=1)
        valid_predictor_codes = flatten_valid_steps(predictor_codes, predictor_mask)
        if valid_predictor_codes.shape[0] > 1:
            predictor_loss = orthonormality.predictor_weight * orthonormality_loss(
                valid_predictor_codes,
                variance_floor=orthonormality.variance_floor,
            )
            extra_loss = extra_loss + predictor_loss
            metrics["loss/orthonormality_predictor"] = predictor_loss.detach()

    if model_config.inverse_dynamics.enabled:
        predictions = bundle.auxiliary_outputs.get("inverse_dynamics.prediction")
        kinematics = bundle.inputs.get("kinematics")
        if kinematics is not None and kinematics.numel() == 0:
            kinematics = None
        if kinematics is None:
            kinematics = resolve_kinematics(batch)
        if predictions is not None and kinematics is not None and predictions.shape[1] > 0:
            transition_valid_mask = transition_mask(bundle, batch, horizon=1)
            if "vision_mask" in bundle.masks:
                vision_mask = bundle.masks["vision_mask"].bool()
                transition_valid_mask = (
                    transition_valid_mask & vision_mask[:, :-1] & vision_mask[:, 1:]
                )
            if transition_valid_mask.any():
                targets = kinematics[:, 1:]
                per_transition_loss = (predictions - targets).pow(2).mean(dim=-1)
                weighted = (
                    model_config.inverse_dynamics.weight
                    * per_transition_loss[transition_valid_mask].mean()
                )
                extra_loss = extra_loss + weighted
                metrics["loss/inverse_dynamics"] = weighted.detach()

    return extra_loss, metrics
