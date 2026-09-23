"""Vision artifact config export helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from placecell_research.config.schema import VisionConfig

from .autoencoder import ConvAutoEncoder, ConvBetaVAE


def _encoder_channels_from_model(model: torch.nn.Module) -> list[int]:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, torch.nn.Sequential):
        raise ValueError(f"Model {type(model).__name__} does not expose a sequential encoder.")
    channels = [int(layer.out_channels) for layer in encoder if isinstance(layer, torch.nn.Conv2d)]
    if not channels:
        raise ValueError(
            f"Model {type(model).__name__} does not expose any encoder convolution layers."
        )
    return channels


def export_effective_vision_config_payload(
    requested_config: VisionConfig,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Serialize the effective train-time vision spec from the actual built model."""
    payload = asdict(requested_config)
    if isinstance(model, torch.nn.Identity):
        payload["type"] = "identity"
        payload["channels"] = []
        return payload
    if isinstance(model, ConvAutoEncoder):
        payload["type"] = "autoencoder"
        payload["latent_dim"] = int(model.fc_latent.out_features)
        payload["channels"] = _encoder_channels_from_model(model)
        return payload
    if isinstance(model, ConvBetaVAE):
        payload["type"] = "beta_vae"
        payload["latent_dim"] = int(model.fc_mu.out_features)
        payload["beta"] = float(model.beta)
        payload["channels"] = _encoder_channels_from_model(model)
        return payload
    raise TypeError(f"Unsupported vision model type for artifact export: {type(model).__name__}.")
