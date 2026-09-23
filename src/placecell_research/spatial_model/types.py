"""Stable spatial model output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .representation_contract import MODULE_OUTPUT_FIELDS


@dataclass
class ModuleOutputs:
    """Outputs from one module in the place model stack."""

    place_codes: Tensor | None = None
    place_logits: Tensor | None = None
    pre_sparsifier: Tensor | None = None
    hidden_state: Tensor | None = None
    backbone_output: Tensor | None = None
    state_readout: Tensor | None = None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class RepresentationBundle:
    """Contract exported by every place model."""

    modules: dict[str, ModuleOutputs] = field(default_factory=dict)
    masks: dict[str, Tensor] = field(default_factory=dict)
    inputs: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary_outputs: dict[str, Tensor] = field(default_factory=dict)
    views: dict[str, tuple[str, int, int]] = field(default_factory=dict)

    def get_representation(self, dotted_name: str) -> Tensor:
        view = self.views.get(dotted_name)
        if view is not None:
            source_name, start, end = view
            return self.get_representation(source_name)[..., start:end]
        module_name, field_name = dotted_name.split(".", 1)
        module = self.modules.get(module_name)
        if module is None:
            raise KeyError(
                f"Module '{module_name}' not available. Available: {sorted(self.modules)}"
            )
        value = getattr(module, field_name, None)
        if value is None:
            value = module.auxiliary.get(field_name)
        if value is None:
            raise KeyError(
                f"'{dotted_name}' not exported. Available: {self.available_representations()}"
            )
        return value

    def get_auxiliary(self, name: str) -> Tensor:
        for module in self.modules.values():
            if name in module.auxiliary:
                return module.auxiliary[name]
        if name not in self.auxiliary_outputs:
            raise KeyError(
                f"Auxiliary output '{name}' missing. Available: {sorted(self.auxiliary_outputs)}"
            )
        return self.auxiliary_outputs[name]

    def available_representations(self) -> list[str]:
        representations: list[str] = []
        for module_name, module in sorted(self.modules.items()):
            for field_name in MODULE_OUTPUT_FIELDS:
                if getattr(module, field_name) is not None:
                    representations.append(f"{module_name}.{field_name}")
            for auxiliary_name, value in module.auxiliary.items():
                if value is not None:
                    representations.append(f"{module_name}.{auxiliary_name}")
        representations.extend(self.views.keys())
        return representations

    def infer_device(self) -> torch.device:
        for module in self.modules.values():
            for field_name in MODULE_OUTPUT_FIELDS:
                tensor = getattr(module, field_name, None)
                if tensor is not None:
                    return tensor.device
            for tensor in module.auxiliary.values():
                if tensor is not None:
                    return tensor.device
        return torch.device("cpu")
