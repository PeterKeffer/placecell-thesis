"""Sparsity objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from placecell_research.spatial_model.types import RepresentationBundle

from .base import ConfiguredObjective, ObjectiveResult
from .masking import masked_mean, valid_steps


@dataclass
class L1SparsityObjective(ConfiguredObjective):
    """Mean |activity| over valid steps AND units."""

    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        mask = valid_steps(bundle, batch)
        penalties = [
            masked_mean(bundle.get_representation(target).abs().mean(dim=-1), mask)
            for target in self.config.targets
        ]
        loss = torch.stack(penalties).mean()
        return ObjectiveResult(loss=loss, metrics={"mean_abs": loss.detach()})


@dataclass
class L1CapacityObjective(L1SparsityObjective):
    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        invalid_targets = [
            target for target in self.config.targets if not target.endswith(".place_codes")
        ]
        if invalid_targets:
            raise ValueError(
                "l1_capacity may only target bounded place-code representations. "
                f"Invalid targets: {invalid_targets!r}."
            )
        result = super().compute(bundle, batch)
        return ObjectiveResult(loss=-result.loss, metrics={"mean_abs": result.metrics["mean_abs"]})


@dataclass
class NormalizedL1CapacityObjective(ConfiguredObjective):
    """Negative L1 on unit-L2, nonnegative activity, scaled by sqrt(unit count)."""

    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        mask = valid_steps(bundle, batch)
        scores = []
        rms, population_std, step_change = [], [], []
        for target in self.config.targets:
            activity = bundle.get_representation(target)
            positive = activity.clamp_min(0)
            unit = torch.nn.functional.normalize(positive, p=2, dim=-1, eps=1e-12)
            score = unit.sum(-1) / activity.shape[-1] ** 0.5
            scores.append(masked_mean(score, mask))
            with torch.no_grad():
                rms.append(masked_mean(activity.square().mean(-1), mask).sqrt())
                population_std.append(masked_mean(activity.std(-1, unbiased=False), mask))
                changes = (activity[:, 1:] - activity[:, :-1]).square().mean(-1)
                step_change.append(masked_mean(changes, mask[:, 1:] & mask[:, :-1]).sqrt())
        score = torch.stack(scores).mean()
        return ObjectiveResult(
            loss=-score,
            metrics={
                "normalized_l1": score.detach(),
                "activity_rms": torch.stack(rms).mean(),
                "activity_population_std": torch.stack(population_std).mean(),
                "activity_step_change_rms": torch.stack(step_change).mean(),
            },
        )
