"""Prediction-alignment objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from placecell_research.spatial_model.types import RepresentationBundle

from .base import ConfiguredObjective, ObjectiveResult
from .masking import flatten_valid_pairs, transition_mask
from .predictor_rollout import bootstrap_future_prediction, replay_predictor_states

if TYPE_CHECKING:
    from placecell_research.spatial_model.protocol import PlaceModel


def _balanced_smooth_l1(predictor: Tensor, target: Tensor) -> Tensor:
    elementwise_loss = F.smooth_l1_loss(predictor, target, reduction="none")
    support = target != 0
    active_count = support.sum(dim=-1).clamp_min(1)
    inactive_count = (~support).sum(dim=-1).clamp_min(1)
    active_loss = (elementwise_loss * support).sum(dim=-1) / active_count
    inactive_loss = (elementwise_loss * ~support).sum(dim=-1) / inactive_count
    return 0.5 * (active_loss + inactive_loss).mean()


def _magnitude_support_ranking(predictor: Tensor, target: Tensor) -> Tensor:
    support = target != 0
    active_count = support.sum(dim=-1)
    inactive_count = (~support).sum(dim=-1)
    if bool((active_count == 0).any()) or bool((inactive_count == 0).any()):
        raise ValueError(
            "support ranking requires targets with active and inactive dimensions."
        )

    support_scores = predictor.abs()
    negative_inactive = torch.where(
        ~support,
        support_scores,
        torch.full_like(support_scores, -torch.inf),
    )
    negative_active = torch.where(
        support,
        -support_scores,
        torch.full_like(support_scores, -torch.inf),
    )
    log_mean_pair_violation = (
        torch.logsumexp(negative_inactive, dim=-1)
        + torch.logsumexp(negative_active, dim=-1)
        - (active_count * inactive_count).log()
    )
    return F.softplus(log_mean_pair_violation).mean()


@dataclass
class PredictionAlignmentObjective(ConfiguredObjective):
    model: PlaceModel | None = None

    def _resolved_targets(self) -> tuple[str, str]:
        if len(self.config.targets) >= 2:
            return self.config.targets[0], self.config.targets[1]
        return "predictor.place_codes", "teacher.place_codes"

    def required_representations(self) -> set[str]:
        predictor_source, target_source = self._resolved_targets()
        return {predictor_source, target_source}

    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        predictor_source, target_source = self._resolved_targets()
        prediction_valid_steps = bundle.masks.get("prediction_valid_steps")
        bootstrap_config = bundle.metadata.get("prediction_bootstrap", {})
        if self.config.target_offset == 0 and (
            prediction_valid_steps is not None or bootstrap_config.get("active", False)
        ):
            raise ValueError("Same-step prediction requires full sequences without bootstrapping.")
        if prediction_valid_steps is None:
            predictor = bundle.get_representation(predictor_source)[:, 1:]
            target_sequence = bundle.get_representation(target_source)
            target = (
                target_sequence[:, 1:]
                if self.config.target_offset == 1
                else target_sequence[:, :-1]
            )
            mask = transition_mask(bundle, batch)
        else:
            predictor = bundle.get_representation(predictor_source)
            target = bundle.get_representation(target_source)
            mask = prediction_valid_steps.bool()
        if isinstance(bootstrap_config, dict) and bool(bootstrap_config.get("active", False)):
            bootstrap_loss_type = str(bootstrap_config.get("loss_type", "mse"))
            if bootstrap_loss_type not in {"mse", "l1"}:
                raise ValueError(
                    "prediction_bootstrap.loss_type must be 'mse' or 'l1', "
                    f"got {bootstrap_loss_type!r}."
                )
            if self.model is None:
                raise RuntimeError(
                    "prediction_bootstrap requires an objective bound to the active place model."
                )
            _, state_after = replay_predictor_states(self.model, bundle)
            bootstrap_target = target.detach().clone()
            for offset in range(target.shape[1] - 1):
                actual_step = offset + 1
                bootstrap_target[:, offset] = bootstrap_target[:, offset] + float(
                    bootstrap_config["gamma"]
                ) * bootstrap_future_prediction(
                    self.model,
                    bundle,
                    actual_step,
                    state_after[actual_step],
                )
            target = bootstrap_target
        predictor_flat, target_flat = flatten_valid_pairs(predictor, target, mask)
        if predictor_flat.numel() == 0:
            loss = predictor.sum() * 0.0
        elif isinstance(bootstrap_config, dict) and bool(bootstrap_config.get("active", False)):
            if bootstrap_loss_type == "l1":
                loss = (predictor_flat - target_flat).abs().mean()
            else:
                loss = F.mse_loss(predictor_flat, target_flat)
        elif self.config.loss_type == "cosine":
            cosine = F.cosine_similarity(predictor_flat, target_flat, dim=-1)
            loss = 1.0 - cosine.mean()
        elif self.config.loss_type == "mse":
            loss = F.mse_loss(predictor_flat, target_flat)
        elif self.config.loss_type == "balanced_smooth_l1":
            loss = _balanced_smooth_l1(predictor_flat, target_flat)
        elif self.config.loss_type == "support_rank_smooth_l1":
            loss = _balanced_smooth_l1(predictor_flat, target_flat) + float(
                self.config.support_rank_weight
            ) * _magnitude_support_ranking(predictor_flat, target_flat)
        else:
            raise ValueError(
                "prediction_alignment.loss_type must be 'cosine', 'mse', "
                "'balanced_smooth_l1', or 'support_rank_smooth_l1', "
                f"got {self.config.loss_type!r}."
            )
        return ObjectiveResult(loss=loss, metrics={"alignment": 1.0 - loss.detach()})
