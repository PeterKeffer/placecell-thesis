"""Training-time tensor regularization."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from placecell_research.config.schema import RegularizationConfig

REGULARIZATION_TARGETS: tuple[str, ...] = (
    "vision.output",
    "encoder.hidden",
    "encoder.logits",
    "encoder.pre_sparsifier",
    "encoder.output",
    "predictor.input",
    "predictor.hidden",
    "predictor.logits",
    "predictor.pre_sparsifier",
    "predictor.output",
)

PREDICTOR_FUSION_COMPATIBLE_TARGETS: frozenset[str] = frozenset(
    {
        "predictor.input",
        "predictor.hidden",
        "predictor.logits",
        "predictor.pre_sparsifier",
        "predictor.output",
    }
)


class LockedDropout(nn.Module):
    """Dropout with optional time-locked masks for sequence tensors."""

    def __init__(self, dropout_probability: float = 0.0, *, mode: str = "independent") -> None:
        super().__init__()
        self.dropout_probability = float(dropout_probability)
        self.mode = str(mode).lower()
        self._cached_keep_mask: Tensor | None = None

    def reset(self) -> None:
        self._cached_keep_mask = None

    def _keep_mask_shape(self, tensor: Tensor) -> tuple[int, ...]:
        if tensor.dim() == 2:
            batch_size, features = tensor.shape
            return batch_size, features
        if tensor.dim() == 3:
            batch_size, time_steps, features = tensor.shape
            if self.mode == "variational":
                return batch_size, 1, features
            return batch_size, time_steps, features
        raise ValueError("LockedDropout expects tensors with 2 or 3 dimensions.")

    def _should_cache_for(self, tensor: Tensor) -> bool:
        return self.mode == "variational" and tensor.dim() == 2

    def _sample_keep_mask(self, tensor: Tensor) -> Tensor:
        if self.dropout_probability <= 0.0:
            return torch.ones_like(tensor)
        if self.dropout_probability >= 1.0:
            return torch.zeros(
                self._keep_mask_shape(tensor),
                device=tensor.device,
                dtype=tensor.dtype,
            )

        keep_probability = 1.0 - self.dropout_probability
        mask_shape = self._keep_mask_shape(tensor)
        if self._should_cache_for(tensor) and self._cached_keep_mask is not None:
            cached = self._cached_keep_mask
            if (
                cached.shape == mask_shape
                and cached.device == tensor.device
                and cached.dtype == tensor.dtype
            ):
                return cached

        keep_mask = torch.empty(
            mask_shape,
            device=tensor.device,
            dtype=tensor.dtype,
        ).bernoulli_(keep_probability)
        keep_mask = keep_mask / keep_probability
        if self._should_cache_for(tensor):
            self._cached_keep_mask = keep_mask
        return keep_mask

    def forward(self, tensor: Tensor) -> Tensor:
        if not self.training or self.dropout_probability <= 0.0:
            return tensor
        return tensor * self._sample_keep_mask(tensor)


class DropoutAndNoise(nn.Module):
    """Dropout followed by independent additive Gaussian noise."""

    def __init__(
        self,
        dropout_probability: float = 0.0,
        noise_scale: float = 0.0,
        *,
        mode: str = "independent",
    ) -> None:
        super().__init__()
        self.noise_scale = float(noise_scale)
        self.dropout = LockedDropout(dropout_probability, mode=mode)

    def reset(self) -> None:
        self.dropout.reset()

    def forward(self, tensor: Tensor) -> Tensor:
        if not self.training:
            return tensor

        if self.dropout.dropout_probability > 0.0:
            tensor = self.dropout(tensor)
        if self.noise_scale > 0.0:
            noise = self.noise_scale * torch.randn_like(tensor)
            tensor = tensor + noise
        return tensor


class SequenceRegularizer(nn.Module):
    """Named-site regularization for one sequence forward pass."""

    def __init__(self, config: RegularizationConfig) -> None:
        super().__init__()
        site_modules: dict[str, DropoutAndNoise] = {}
        target_to_key: dict[str, str] = {}
        for site in config.sites:
            if not site.target:
                continue
            module_key = site.target.replace(".", "__")
            site_modules[module_key] = DropoutAndNoise(
                dropout_probability=site.dropout,
                noise_scale=site.noise_scale,
                mode=config.dropout_mode,
            )
            target_to_key[site.target] = module_key
        self.site_modules = nn.ModuleDict(site_modules)
        self.target_to_key = target_to_key

    def reset_sequence(self) -> None:
        for module in self.site_modules.values():
            module.reset()

    def active_targets(self) -> list[str]:
        return sorted(self.target_to_key)

    def has_active_site(self, prefix: str) -> bool:
        for target, module_key in self.target_to_key.items():
            if not target.startswith(prefix):
                continue
            module = self.site_modules[module_key]
            if module.dropout.dropout_probability > 0.0 or module.noise_scale > 0.0:
                return True
        return False

    def predictor_requires_stepwise(self) -> bool:
        """Return whether active predictor regularization must run inside recurrence."""
        for target, module_key in self.target_to_key.items():
            if not target.startswith("predictor."):
                continue
            module = self.site_modules[module_key]
            is_active = (
                module.dropout.dropout_probability > 0.0 or module.noise_scale > 0.0
            )
            if is_active and target not in PREDICTOR_FUSION_COMPATIBLE_TARGETS:
                return True
        return False

    def apply(self, target: str, tensor: Tensor) -> Tensor:
        module_key = self.target_to_key.get(target)
        if module_key is None:
            return tensor
        return self.site_modules[module_key](tensor)
