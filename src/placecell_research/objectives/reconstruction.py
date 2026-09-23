"""Latent-reconstruction objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from placecell_research.spatial_model.types import RepresentationBundle

from .base import ConfiguredObjective, ObjectiveResult
from .masking import flatten_valid_pairs, transition_mask


@dataclass
class LatentReconstructionObjective(ConfiguredObjective):
    def required_representations(self) -> set[str]:
        return set()

    def required_auxiliary_outputs(self) -> set[str]:
        return {f"{self.name}.reconstruction"}

    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        target = batch.get("latent")
        if target is None:
            raise KeyError(
                "latent_reconstruction requires batch['latent']; it only applies to "
                "latent-input runs (spatial_model.inputs.observation_source == 'latent')."
            )
        prediction = bundle.get_auxiliary(f"{self.name}.reconstruction")[:, 1:]
        target = target[:, 1:]
        mask = transition_mask(bundle, batch)
        prediction_flat, target_flat = flatten_valid_pairs(prediction, target, mask)
        if prediction_flat.numel() == 0:
            loss = prediction.sum() * 0.0
        else:
            loss = F.mse_loss(prediction_flat, target_flat)
        return ObjectiveResult(loss=loss, metrics={"reconstruction_mse": loss.detach()})
