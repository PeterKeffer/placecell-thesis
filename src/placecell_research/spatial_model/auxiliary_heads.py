"""Trainable auxiliary heads built beside objectives."""

from __future__ import annotations

from torch import Tensor, nn

from .types import RepresentationBundle


class ReconstructionHead(nn.Module):
    """Decoder from place codes back to the observation latent."""

    def __init__(
        self,
        name: str,
        source: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        target_module: str = "predictor",
    ) -> None:
        super().__init__()
        self._name = name
        self.source = source
        self._target_module = target_module
        self.output_dim = output_dim
        if hidden_dim is not None and hidden_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.projection = nn.Linear(input_dim, output_dim)

    @property
    def name(self) -> str:
        return self._name

    @property
    def target_module(self) -> str:
        return self._target_module

    def required_representations(self) -> set[str]:
        return {self.source}

    def output_names(self) -> set[str]:
        return {f"{self.name}.reconstruction"}

    def forward(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        del batch
        source_representation = bundle.get_representation(self.source)
        return {f"{self.name}.reconstruction": self.projection(source_representation)}


def run_auxiliary_heads(
    auxiliary_heads: dict[str, nn.Module],
    bundle: RepresentationBundle,
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    outputs: dict[str, Tensor] = {}
    for head in auxiliary_heads.values():
        head_outputs = head(bundle, batch)
        overlap = set(outputs) & set(head_outputs)
        if overlap:
            raise ValueError(f"Auxiliary head outputs collided: {sorted(overlap)}")
        target_module = getattr(head, "target_module", "predictor")
        module_outputs = bundle.modules.get(target_module)
        if module_outputs is None:
            raise ValueError(f"Auxiliary head targets missing module '{target_module}'.")
        module_outputs.auxiliary.update(head_outputs)
        outputs.update(head_outputs)
    bundle.auxiliary_outputs.update(outputs)
    return outputs
