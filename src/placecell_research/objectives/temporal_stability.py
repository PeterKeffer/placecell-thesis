"""Temporal stability objective (Wyss/Koenig/Verschure)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from placecell_research.spatial_model.types import RepresentationBundle

from .base import ConfiguredObjective, ObjectiveResult
from .masking import valid_steps

_VARIANCE_EPS = 1.0e-6
_DEAD_UNIT_VARIANCE_FLOOR = 1.0e-4


def full_window_pair_mask(step_mask: Tensor, lag: int) -> Tensor:
    """Valid lag pairs whose ENTIRE window [t, t + lag] is valid, shape (B, T - lag)."""
    counts = torch.nn.functional.pad(step_mask.to(torch.long).cumsum(dim=1), (1, 0))
    window_counts = counts[:, lag + 1 :] - counts[:, : -(lag + 1)]
    return window_counts == lag + 1


@dataclass
class TemporalStabilityObjective(ConfiguredObjective):
    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        lag = self.config.horizon
        step_mask = valid_steps(bundle, batch)
        pair_mask = full_window_pair_mask(step_mask, lag)
        losses: list[Tensor] = []
        stability_terms: list[Tensor] = []
        decorrelation_terms: list[Tensor] = []
        variance_means: list[Tensor] = []
        dead_fractions: list[Tensor] = []
        code_rms_terms: list[Tensor] = []
        for target_name in self.config.targets:
            target = bundle.get_representation(target_name)
            num_units = target.shape[-1]
            step_weights = step_mask.to(target.dtype).unsqueeze(-1)
            episode_counts = step_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            episode_means = (target * step_weights).sum(dim=1, keepdim=True) / episode_counts
            centered = (target - episode_means) * step_weights
            total_count = step_weights.sum().clamp_min(1.0)
            variance = centered.pow(2).sum(dim=(0, 1)) / total_count
            detached_variance = variance.detach()
            psi_variance = (
                detached_variance if self.config.detach_variance_denominator else variance
            )
            variance_means.append(detached_variance.mean())
            mean_squared_activity = (target.pow(2) * step_weights).sum() / (total_count * num_units)
            code_rms_terms.append(torch.sqrt(mean_squared_activity).detach())
            assessable = detached_variance > _DEAD_UNIT_VARIANCE_FLOOR * detached_variance.mean()
            dead_fractions.append(1.0 - assessable.to(target.dtype).mean())

            pair_weights = pair_mask.to(target.dtype).unsqueeze(-1)
            pair_count = pair_weights.sum()
            if pair_count == 0 or not bool(assessable.any()):
                zero = target.sum() * 0.0
                stability_terms.append(zero.detach())
                decorrelation_terms.append(zero.detach())
                losses.append(zero)
                continue
            squared_change = (target[:, lag:] - target[:, :-lag]).pow(2)
            mean_squared_change = (squared_change * pair_weights).sum(dim=(0, 1)) / pair_count
            psi = mean_squared_change / (psi_variance + _VARIANCE_EPS)
            stability = psi[assessable].mean()
            decorrelation = self._decorrelation(centered, variance, total_count, assessable)
            stability_terms.append(stability.detach())
            decorrelation_terms.append(decorrelation.detach())
            losses.append(stability + self.config.decorrelation_weight * decorrelation)
        loss = torch.stack(losses).mean()
        return ObjectiveResult(
            loss=loss,
            metrics={
                "temporal_stability": torch.stack(stability_terms).mean(),
                "decorrelation": torch.stack(decorrelation_terms).mean(),
                "mean_variance": torch.stack(variance_means).mean(),
                "dead_unit_fraction": torch.stack(dead_fractions).mean(),
                "mean_code_rms": torch.stack(code_rms_terms).mean(),
            },
        )

    def _decorrelation(
        self,
        centered: Tensor,
        variance: Tensor,
        total_count: Tensor,
        assessable: Tensor,
    ) -> Tensor:
        """Mean over off-diagonal LIVE unit pairs of the squared correlation (Wyss's beta term)."""
        if self.config.decorrelation_weight <= 0.0:
            return centered.sum() * 0.0
        live_centered = centered[..., assessable]
        num_live = live_centered.shape[-1]
        if num_live < 2:
            return centered.sum() * 0.0
        flat_centered = live_centered.reshape(-1, num_live)
        covariance = flat_centered.T @ flat_centered / total_count
        std = torch.sqrt(variance[assessable].clamp_min(0.0))
        correlation = covariance / (std[:, None] * std[None, :] + _VARIANCE_EPS)
        squared_correlation = correlation.pow(2)
        off_diagonal_sum = squared_correlation.sum() - squared_correlation.diagonal().sum()
        return off_diagonal_sum / (num_live * (num_live - 1))
