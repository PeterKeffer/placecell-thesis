"""Predict-yourself-at-timescale-tau: the single readout principle behind cell types."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from placecell_research.config.schema import SpatialModelConfig
from placecell_research.spatial_model.representation_contract import base_representation_shapes
from placecell_research.spatial_model.types import RepresentationBundle

from .base import ConfiguredObjective, ObjectiveResult
from .masking import valid_steps


def _configured_width(model_config: SpatialModelConfig, target: str) -> int | None:
    """Width of a target as the CONFIG declares it, or None when the config cannot say."""
    shape = base_representation_shapes(model_config).get(target)
    if isinstance(shape, list) and shape and isinstance(shape[-1], int):
        return int(shape[-1])
    return None


@dataclass
class TimescaleAlignmentObjective(ConfiguredObjective):
    """Cosine-align head codes with the temporally low-passed target representation."""

    def validate_configuration(
        self,
        available_representations: set[str],
        available_auxiliary_outputs: set[str],
    ) -> None:
        super().validate_configuration(available_representations, available_auxiliary_outputs)
        if len(self.config.targets) != 2:
            raise ValueError(
                f"{self.name} requires exactly two targets [head_code, target_code], "
                f"got {self.config.targets}."
            )

    def validate_against_model(self, model_config: SpatialModelConfig) -> None:
        """Cosine alignment is elementwise: mismatched widths crash mid-epoch, not at build."""
        if len(self.config.targets) != 2:
            return
        head_target, alignment_target = self.config.targets
        head_width = _configured_width(model_config, head_target)
        alignment_width = _configured_width(model_config, alignment_target)
        if head_width is None or alignment_width is None or head_width == alignment_width:
            return
        raise ValueError(
            f"objective {self.name!r}: timescale_alignment aligns the two targets elementwise, "
            f"so they must be equally wide, but {head_target!r} is {head_width}-dimensional and "
            f"{alignment_target!r} is {alignment_width}-dimensional. There is no projection "
            "between them: give the two heads the same code_dim, or align against a source of "
            "the head's own width."
        )

    def compute(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ObjectiveResult:
        head_codes = bundle.get_representation(self.config.targets[0])
        target_codes = bundle.get_representation(self.config.targets[1]).detach()
        mask = valid_steps(bundle, batch)
        timescale = int(self.config.timescale)
        if timescale > 0:
            decay = timescale / (timescale + 1.0)
            step_decay = None
            if self.config.event_gated:
                expectation = bundle.get_representation("predictor.pre_sparsifier")
                reference = bundle.get_representation("teacher.pre_sparsifier")
                surprise = 1.0 - F.cosine_similarity(
                    expectation.detach(), reference.detach(), dim=-1
                )
                surprise[:, 0] = 0.0
                mask_f = mask.to(surprise.dtype)
                count = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean = (surprise * mask_f).sum(dim=1, keepdim=True) / count
                var = ((surprise - mean).pow(2) * mask_f).sum(dim=1, keepdim=True) / count
                z_scores = (surprise - mean) / (var.sqrt() + 1e-6)
                step_decay = decay * torch.exp(-torch.relu(z_scores - 1.0)).unsqueeze(-1)
            step_mask = mask.to(target_codes.dtype).unsqueeze(-1)
            trace = target_codes[:, 0] * step_mask[:, 0]
            traces = [trace]
            for t in range(1, target_codes.shape[1]):
                decay_t = decay if step_decay is None else step_decay[:, t]
                update = decay_t * trace + (1.0 - decay_t) * target_codes[:, t]
                trace = torch.where(step_mask[:, t] > 0, update, trace)
                traces.append(trace)
            target_codes = torch.stack(traces, dim=1)
        flat_mask = mask.reshape(-1)
        flat_head = head_codes.reshape(-1, head_codes.shape[-1])[flat_mask]
        flat_target = target_codes.reshape(-1, target_codes.shape[-1])[flat_mask]
        if flat_head.numel() == 0:
            loss = head_codes.sum() * 0.0
            return ObjectiveResult(loss=loss, metrics={"alignment": loss.detach()})
        alignment = F.cosine_similarity(flat_head, flat_target, dim=-1).mean()
        return ObjectiveResult(loss=1.0 - alignment, metrics={"alignment": alignment.detach()})
