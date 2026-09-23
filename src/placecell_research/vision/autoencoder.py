"""Vision encoder models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _encoder_layers(in_channels: int, channels: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_channels
    for next_channels in channels:
        layers.append(nn.Conv2d(current, next_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        current = next_channels
    return nn.Sequential(*layers)


def _decoder_layers(channels: Sequence[int], out_channels: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    reversed_channels = list(channels)[::-1]
    for index, current in enumerate(reversed_channels):
        next_channels = (
            reversed_channels[index + 1] if index + 1 < len(reversed_channels) else out_channels
        )
        layers.append(
            nn.ConvTranspose2d(current, next_channels, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.ReLU(inplace=True) if index + 1 < len(reversed_channels) else nn.Sigmoid())
    return nn.Sequential(*layers)


def _gaussian_window(
    window_size: int, sigma: float, channels: int, reference: torch.Tensor
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=reference.dtype, device=reference.device)
    coords = coords - (window_size - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] * gauss[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 7,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Single-scale SSIM with a Gaussian window, returned as a mean scalar in roughly [0, 1]."""
    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction)
    mean_prediction = F.conv2d(prediction, window, groups=channels)
    mean_target = F.conv2d(target, window, groups=channels)
    mean_prediction_sq = mean_prediction * mean_prediction
    mean_target_sq = mean_target * mean_target
    mean_cross = mean_prediction * mean_target
    var_prediction = F.conv2d(prediction * prediction, window, groups=channels) - mean_prediction_sq
    var_target = F.conv2d(target * target, window, groups=channels) - mean_target_sq
    covariance = F.conv2d(prediction * target, window, groups=channels) - mean_cross
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mean_cross + c1) * (2 * covariance + c2)) / (
        (mean_prediction_sq + mean_target_sq + c1) * (var_prediction + var_target + c2)
    )
    return ssim_map.mean()


def _sobel_edges(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = image.shape[1]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    )
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.expand(channels, 1, 3, 3).contiguous()
    kernel_y = kernel_y.expand(channels, 1, 3, 3).contiguous()
    gradient_x = F.conv2d(image, kernel_x, groups=channels, padding=1)
    gradient_y = F.conv2d(image, kernel_y, groups=channels, padding=1)
    return gradient_x, gradient_y


@dataclass(slots=True)
class AutoEncoderOutput:
    reconstruction: torch.Tensor
    latents: torch.Tensor
    mean: torch.Tensor | None = None
    logvar: torch.Tensor | None = None


class ConvBetaVAE(nn.Module):
    """Convolutional beta-VAE with a spatially faithful latent projection."""

    def __init__(
        self,
        latent_dim: int = 64,
        beta: float = 4.0,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        input_shape: tuple[int, int, int] = (3, 64, 64),
    ) -> None:
        super().__init__()
        in_channels, height, width = input_shape
        self.beta = float(beta)
        self.last_loss_components: dict[str, torch.Tensor] = {}
        self.encoder = _encoder_layers(in_channels, channels)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            conv_shape = self.encoder(dummy).shape[2:]
        self.conv_shape = tuple(int(value) for value in conv_shape)
        encoded_channels = int(channels[-1])
        self.encoded_shape = (encoded_channels, *self.conv_shape)
        encoded_flat = encoded_channels * self.conv_shape[0] * self.conv_shape[1]
        self.fc_mu = nn.Linear(encoded_flat, latent_dim)
        self.fc_logvar = nn.Linear(encoded_flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, encoded_flat)
        self.decoder = _decoder_layers(channels, in_channels)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs).reshape(inputs.shape[0], -1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.fc_decode(latents).view(latents.shape[0], *self.encoded_shape)
        return self.decoder(hidden)

    def forward(self, inputs: torch.Tensor) -> AutoEncoderOutput:
        mean, logvar = self.encode(inputs)
        latents = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latents)
        if reconstruction.shape[2:] != inputs.shape[2:]:
            reconstruction = F.interpolate(
                reconstruction, size=inputs.shape[2:], mode="bilinear", align_corners=False
            )
        return AutoEncoderOutput(
            reconstruction=reconstruction,
            latents=latents,
            mean=mean,
            logvar=logvar,
        )

    def loss(self, inputs: torch.Tensor, output: AutoEncoderOutput) -> torch.Tensor:
        if output.mean is None or output.logvar is None:
            raise ValueError("ConvBetaVAE loss requires mean and logvar in AutoEncoderOutput.")
        reconstruction_loss = (
            F.mse_loss(
                output.reconstruction,
                inputs,
                reduction="none",
            )
            .sum(dim=(1, 2, 3))
            .mean()
        )
        kl = (
            -0.5
            * torch.sum(
                1 + output.logvar - output.mean.square() - output.logvar.exp(),
                dim=1,
            ).mean()
        )
        return reconstruction_loss + self.beta * kl


class ConvAutoEncoder(nn.Module):
    """Deterministic convolutional autoencoder with a spatially faithful bottleneck."""

    def __init__(
        self,
        latent_dim: int = 64,
        channels: Sequence[int] = (32, 64, 128, 256),
        input_shape: tuple[int, int, int] = (3, 64, 64),
        loss_type: str = "mse",
        loss_l1_weight: float = 1.0,
        loss_ssim_weight: float = 0.5,
        loss_edge_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.loss_type = str(loss_type)
        self.loss_l1_weight = float(loss_l1_weight)
        self.loss_ssim_weight = float(loss_ssim_weight)
        self.loss_edge_weight = float(loss_edge_weight)
        self.last_loss_components: dict[str, torch.Tensor] = {}
        in_channels, height, width = input_shape
        self.encoder = _encoder_layers(in_channels, channels)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            conv_shape = self.encoder(dummy).shape[2:]
        self.conv_shape = tuple(int(value) for value in conv_shape)
        encoded_channels = int(channels[-1])
        self.encoded_shape = (encoded_channels, *self.conv_shape)
        encoded_flat = encoded_channels * self.conv_shape[0] * self.conv_shape[1]
        self.fc_latent = nn.Linear(encoded_flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, encoded_flat)
        self.decoder = _decoder_layers(channels, in_channels)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(inputs).reshape(inputs.shape[0], -1)
        return self.fc_latent(hidden)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.fc_decode(latents).view(latents.shape[0], *self.encoded_shape)
        return self.decoder(hidden)

    def forward(self, inputs: torch.Tensor) -> AutoEncoderOutput:
        latents = self.encode(inputs)
        reconstruction = self.decode(latents)
        if reconstruction.shape[2:] != inputs.shape[2:]:
            reconstruction = F.interpolate(
                reconstruction, size=inputs.shape[2:], mode="bilinear", align_corners=False
            )
        return AutoEncoderOutput(reconstruction=reconstruction, latents=latents)

    def loss(self, inputs: torch.Tensor, output: AutoEncoderOutput) -> torch.Tensor:
        reconstruction = output.reconstruction
        if self.loss_type == "structure_l1":
            return self._structure_l1_loss(inputs, reconstruction)
        return F.mse_loss(reconstruction, inputs, reduction="none").sum(dim=(1, 2, 3)).mean()

    def _structure_l1_loss(
        self, inputs: torch.Tensor, reconstruction: torch.Tensor
    ) -> torch.Tensor:
        l1 = (reconstruction - inputs).abs().mean()
        structure = 1.0 - structural_similarity(reconstruction, inputs, data_range=1.0)
        gradient_x_recon, gradient_y_recon = _sobel_edges(reconstruction)
        gradient_x_target, gradient_y_target = _sobel_edges(inputs)
        edge = (
            (gradient_x_recon - gradient_x_target).abs()
            + (gradient_y_recon - gradient_y_target).abs()
        ).mean()
        self.last_loss_components = {
            "l1": l1.detach(),
            "ssim": structure.detach(),
            "edge": edge.detach(),
        }
        return (
            self.loss_l1_weight * l1
            + self.loss_ssim_weight * structure
            + self.loss_edge_weight * edge
        )
