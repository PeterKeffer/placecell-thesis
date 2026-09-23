"""Model-side protocols."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor
from torch.nn import Parameter

from .types import RepresentationBundle


class PlaceModel(Protocol):
    """Stable protocol for place models."""

    def forward_sequence(self, batch: dict[str, Tensor]) -> RepresentationBundle:
        ...

    def parameters_by_group(self) -> dict[str, list[Parameter]]:
        ...

    def update_teacher(self, current_step: int | None = None) -> None:
        ...

    def architecture_summary(self) -> str:
        ...

    def model_contract(self) -> dict:
        ...


class AuxiliaryHead(Protocol):
    """Trainable, stateless-between-batches auxiliary computation."""

    @property
    def name(self) -> str:
        ...

    def required_representations(self) -> set[str]:
        ...

    def output_names(self) -> set[str]:
        ...

    @property
    def target_module(self) -> str:
        ...

    def forward(self, bundle: RepresentationBundle, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        ...
