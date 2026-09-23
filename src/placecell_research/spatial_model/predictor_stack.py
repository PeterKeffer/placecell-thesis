"""The typed contract a predictor rollout runs against."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

from torch import nn

from placecell_research.config.schema import SpatialModelConfig


class PredictorStackComponents(Protocol):
    """The slice of PlaceModelComponents a rollout reads."""

    config: SpatialModelConfig


class PredictorStack(Protocol):
    """Everything a predictor rollout touches: the config that shapes it, the modules it runs."""

    components: PredictorStackComponents
    predictor_input_assembler: nn.Module
    predictor_temporal: nn.Module
    predictor_head: nn.Module
    predictor_sparsifier: nn.Module
    action_embedding: nn.Module | None


@dataclass(frozen=True, slots=True)
class PredictorModules:
    """The predictor stack's modules, under the attribute names the rollout reads them by."""

    predictor_input_assembler: nn.Module
    predictor_temporal: nn.Module
    predictor_head: nn.Module
    predictor_sparsifier: nn.Module
    action_embedding: nn.Module | None = None

    @staticmethod
    def module_names() -> tuple[str, ...]:
        return tuple(field.name for field in fields(PredictorModules))

    def present(self) -> dict[str, nn.Module]:
        """Name -> module for every module this stack actually has, in registration order."""
        return {
            name: module
            for name in self.module_names()
            if (module := getattr(self, name)) is not None
        }

    def bind(self, components: PredictorStackComponents) -> DetachedPredictorStack:
        return DetachedPredictorStack(
            components=components,
            predictor_input_assembler=self.predictor_input_assembler,
            predictor_temporal=self.predictor_temporal,
            predictor_head=self.predictor_head,
            predictor_sparsifier=self.predictor_sparsifier,
            action_embedding=self.action_embedding,
        )


@dataclass(frozen=True, slots=True)
class DetachedPredictorStack:
    """A PredictorStack whose modules are not the live model's."""

    components: PredictorStackComponents
    predictor_input_assembler: nn.Module
    predictor_temporal: nn.Module
    predictor_head: nn.Module
    predictor_sparsifier: nn.Module
    action_embedding: nn.Module | None
    training_regularizer: None = None
