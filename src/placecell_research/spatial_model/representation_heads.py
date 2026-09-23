"""Cell-type readout heads trained beside the place model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from placecell_research.config.schema import SpatialModelConfig

from .components.sparsifiers import KWinnersSparsifier
from .types import ModuleOutputs, RepresentationBundle


def run_representation_heads(
    representation_heads: nn.ModuleDict,
    bundle: RepresentationBundle,
    batch: dict[str, Tensor],
) -> None:
    """Run each head and register its outputs under bundle.modules[head.namespace]."""
    for head in representation_heads.values():
        namespace = head.namespace
        if namespace in bundle.modules:
            raise ValueError(
                f"Representation head namespace '{namespace}' already exists in bundle."
            )
        bundle.modules[namespace] = head(bundle, batch)


class CellTypeReadoutHead(nn.Module):
    """Generic staged cell-type readout: dense trunk -> linear -> [relu] -> [k-winners]."""

    def __init__(
        self,
        *,
        namespace: str,
        source: str,
        input_dim: int,
        code_dim: int,
        sparsifier: nn.Module | None,
        nonnegative: bool,
        predictor_hidden_dim: int = 0,
        target_dim: int = 0,
        num_actions: int | None = None,
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.source = source
        self.projection = nn.Linear(input_dim, code_dim)
        self.nonnegative = bool(nonnegative)
        self.sparsifier = sparsifier
        self.num_actions = num_actions
        self.prediction = None
        if predictor_hidden_dim:
            if not num_actions or target_dim <= 0:
                raise ValueError("A predictive readout requires action and target dimensions.")
            self.prediction = nn.Sequential(
                nn.Linear(code_dim + num_actions, predictor_hidden_dim),
                nn.GELU(),
                nn.Linear(predictor_hidden_dim, target_dim),
            )

    def forward(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> ModuleOutputs:
        features = bundle.get_representation(self.source).detach()
        pre_sparsifier = self.projection(features)
        if self.nonnegative:
            pre_sparsifier = torch.relu(pre_sparsifier)
        codes = self.sparsifier(pre_sparsifier) if self.sparsifier is not None else pre_sparsifier
        predicted = None
        if self.prediction is not None:
            actions = torch.nn.functional.one_hot(
                bundle.inputs["actions"][:, :-1].long(), num_classes=self.num_actions + 1
            )[..., : self.num_actions].to(codes.dtype)
            next_code = self.prediction(torch.cat((codes[:, :-1], actions), dim=-1))
            predicted = torch.cat(
                (next_code.new_zeros(codes.shape[0], 1, next_code.shape[-1]), next_code), dim=1
            )
        auxiliary = (
            dict(getattr(self.sparsifier, "last_auxiliary_outputs", {}))
            if self.sparsifier is not None
            else {}
        )
        return ModuleOutputs(
            place_codes=codes,
            place_logits=predicted,
            pre_sparsifier=pre_sparsifier,
            auxiliary=auxiliary,
        )


def build_representation_heads(
    config: SpatialModelConfig,
    *,
    num_actions: int | None = None,
    optimizer_steps_per_epoch: int = 1,
) -> nn.ModuleDict:
    """Build configured post-base representation heads."""
    heads: dict[str, nn.Module] = {}
    source_dims = {
        "encoder.hidden_state": config.encoder.output_size,
        "predictor.hidden_state": config.predictor.output_size,
    }
    for head_config in config.cell_type_heads:
        if head_config.name in heads:
            raise ValueError(f"cell_type_heads name collides with a built head: {head_config.name}")
        if head_config.sparsifier is not None:
            from .builder import build_sparsifier

            sparsifier = build_sparsifier(
                head_config.sparsifier,
                head_config.code_dim,
                optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            )
        elif head_config.active_units > 0:
            sparsifier = KWinnersSparsifier(
                k_fraction=head_config.active_units / head_config.code_dim,
                num_units=head_config.code_dim,
            )
        else:
            sparsifier = None
        heads[head_config.name] = CellTypeReadoutHead(
            namespace=head_config.name,
            source=head_config.source,
            input_dim=source_dims[head_config.source],
            code_dim=head_config.code_dim,
            sparsifier=sparsifier,
            nonnegative=head_config.nonnegative,
            predictor_hidden_dim=head_config.predictor_hidden_dim,
            target_dim=config.training.code_dim,
            num_actions=num_actions,
        )
    return nn.ModuleDict(heads)
