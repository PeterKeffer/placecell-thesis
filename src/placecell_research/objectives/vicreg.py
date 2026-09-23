"""VICReg objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from placecell_research.spatial_model.types import RepresentationBundle

from .base import ConfiguredObjective, ObjectiveResult
from .masking import flatten_valid_steps, sample_time_mask, valid_steps


@dataclass
class VICRegObjective(ConfiguredObjective):
    def required_representations(self) -> set[str]:
        representations = super().required_representations()
        if self.config.routing_mask_source:
            representations.add(self.config.routing_mask_source)
        return representations

    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        losses: list[Tensor] = []
        mean_std_values: list[Tensor] = []
        variance_loss_values: list[Tensor] = []
        covariance_loss_values: list[Tensor] = []
        dead_dim_fraction_values: list[Tensor] = []
        target_mask = sample_time_mask(
            valid_steps(bundle, batch),
            stride=self.config.anchor_stride,
            offset=self.config.anchor_offset,
        )
        if self.config.routing_mask_source:
            routing = bundle.get_representation(self.config.routing_mask_source)
            if routing.shape[:2] != target_mask.shape:
                raise ValueError(
                    "VICReg routing mask must share the target batch and time dimensions."
                )
            if self.config.routing_mask_index >= routing.shape[-1]:
                raise ValueError(
                    f"VICReg routing_mask_index={self.config.routing_mask_index} is outside "
                    f"the routing width {routing.shape[-1]}."
                )
            target_mask = target_mask & (
                routing.argmax(dim=-1) == self.config.routing_mask_index
            )
        for target_name in self.config.targets:
            target = bundle.get_representation(target_name)
            flat_target = flatten_valid_steps(target, target_mask)
            if flat_target.shape[0] < 2:
                losses.append(target.sum() * 0.0)
                continue
            centered = flat_target - flat_target.mean(dim=0, keepdim=True)
            std_per_dim = torch.sqrt(centered.var(dim=0) + 1e-4)
            variance_loss = torch.relu(self.config.minimum_std - std_per_dim).mean()
            covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
            num_features = centered.shape[1]
            off_diagonal = covariance - torch.diag(torch.diag(covariance))
            if self.config.covariance_normalization == "feature_count_squared":
                covariance_loss = off_diagonal.pow(2).mean()
            else:
                covariance_loss = off_diagonal.pow(2).sum() / num_features
            losses.append(
                self.config.variance_weight * variance_loss
                + self.config.covariance_weight * covariance_loss
            )
            mean_std_values.append(std_per_dim.mean())
            variance_loss_values.append(variance_loss)
            covariance_loss_values.append(covariance_loss)
            dead_dim_fraction_values.append((std_per_dim <= 1.05e-2).float().mean())
        total_loss = torch.stack(losses).mean()
        mean_std = (
            torch.stack(mean_std_values).mean()
            if mean_std_values
            else total_loss.detach() * 0.0
        )
        variance_loss = (
            torch.stack(variance_loss_values).mean()
            if variance_loss_values
            else total_loss.detach() * 0.0
        )
        covariance_loss = (
            torch.stack(covariance_loss_values).mean()
            if covariance_loss_values
            else total_loss.detach() * 0.0
        )
        dead_dim_fraction = (
            torch.stack(dead_dim_fraction_values).mean()
            if dead_dim_fraction_values
            else total_loss.detach() * 0.0
        )
        return ObjectiveResult(
            loss=total_loss,
            metrics={
                "mean_std": mean_std.detach(),
                "variance_loss": variance_loss.detach(),
                "covariance_loss": covariance_loss.detach(),
                "dead_dim_fraction": dead_dim_fraction.detach(),
            },
        )
