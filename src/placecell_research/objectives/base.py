"""Objective contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from torch import Tensor

from placecell_research.config.schema import ObjectiveConfig, SpatialModelConfig
from placecell_research.spatial_model.types import RepresentationBundle


@dataclass
class ObjectiveResult:
    """Loss plus metrics from a stateless objective."""

    loss: Tensor
    metrics: dict[str, Tensor | float] = field(default_factory=dict)
    debug: dict[str, Tensor | float] = field(default_factory=dict)


@dataclass
class ConfiguredObjective:
    """Shared config-backed objective contract behavior."""

    name: str
    config: ObjectiveConfig

    def required_representations(self) -> set[str]:
        return set(self.config.targets)

    def required_auxiliary_outputs(self) -> set[str]:
        return set()

    def validate_configuration(
        self,
        available_representations: set[str],
        available_auxiliary_outputs: set[str],
    ) -> None:
        missing_representations = self.required_representations() - available_representations
        if missing_representations:
            raise ValueError(
                f"{self.name} requires missing representations: {sorted(missing_representations)}"
            )
        missing_auxiliary_outputs = self.required_auxiliary_outputs() - available_auxiliary_outputs
        if missing_auxiliary_outputs:
            raise ValueError(
                f"{self.name} requires missing auxiliary outputs: "
                f"{sorted(missing_auxiliary_outputs)}"
            )

    def validate_against_model(self, model_config: SpatialModelConfig) -> None:
        """Reject objective/model-config combinations at build time."""
        return None


class ObjectiveModule(Protocol):
    @property
    def name(self) -> str: ...

    def required_representations(self) -> set[str]: ...

    def required_auxiliary_outputs(self) -> set[str]: ...

    def validate_configuration(
        self,
        available_representations: set[str],
        available_auxiliary_outputs: set[str],
    ) -> None: ...

    def validate_against_model(self, model_config: SpatialModelConfig) -> None: ...

    def compute(
        self, bundle: RepresentationBundle, batch: dict[str, Tensor]
    ) -> ObjectiveResult: ...
