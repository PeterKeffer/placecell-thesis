"""Observation backbones."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor, nn


class IdentityBackbone(nn.Module):
    """Pass latent observations through unchanged."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output_dim = input_dim

    def forward(self, observations: Tensor) -> Tensor:
        return observations


class MLPBackbone(nn.Module):
    """Per-timestep latent projection."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, observations: Tensor) -> Tensor:
        batch_size, time_steps, feature_dim = observations.shape
        del feature_dim
        flattened = rearrange(observations, "batch time features -> (batch time) features")
        projected = self.network(flattened)
        return rearrange(
            projected,
            "(batch time) features -> batch time features",
            batch=batch_size,
            time=time_steps,
        )


class SimpleCNNBackbone(nn.Module):
    """Compact CNN for RGB sequences."""

    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        output_dim: int,
        pool_size: int = 1,
        max_frames_per_chunk: int = 256,
    ) -> None:
        super().__init__()
        pool_size = max(1, int(pool_size))
        layers: list[nn.Module] = []
        current_channels = input_channels
        for channel in channels:
            layers.extend(
                [
                    nn.Conv2d(current_channels, channel, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                ]
            )
            current_channels = channel
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.projection = nn.Linear(current_channels * pool_size * pool_size, output_dim)
        self.output_dim = output_dim
        self.max_frames_per_chunk = max(1, int(max_frames_per_chunk))

    def forward(self, observations: Tensor) -> Tensor:
        batch_size, time_steps, _channels, _height, _width = observations.shape
        flattened = rearrange(
            observations,
            "batch time channels height width -> (batch time) channels height width",
        )
        encoded_chunks: list[Tensor] = []
        for start_index in range(0, flattened.shape[0], self.max_frames_per_chunk):
            chunk = flattened[start_index : start_index + self.max_frames_per_chunk]
            encoded_chunks.append(self.pool(self.conv(chunk)).flatten(start_dim=1))
        encoded = torch.cat(encoded_chunks, dim=0)
        projected = self.projection(encoded)
        return rearrange(
            projected,
            "(batch time) features -> batch time features",
            batch=batch_size,
            time=time_steps,
        )


class ViTBackbone(nn.Module):
    """Small ViT feature extractor for RGB sequences."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        patch_size: int = 8,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames_per_chunk: int = 256,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError(
                "ViTBackbone output_dim must be divisible by num_heads, "
                f"got {output_dim} and {num_heads}."
            )
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Conv2d(
            input_channels,
            output_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=max(1, int(output_dim * mlp_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim
        self.max_frames_per_chunk = max(1, int(max_frames_per_chunk))

    @staticmethod
    def _axis_position_embedding(values: Tensor, width: int) -> Tensor:
        if width <= 0:
            return values.new_zeros(values.shape[0], 0)
        frequency_count = (width + 1) // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(frequency_count, device=values.device, dtype=values.dtype)
            / max(1, frequency_count - 1)
        )
        angles = values[:, None] * frequencies[None, :] * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)[:, :width]

    def _position_embedding(
        self,
        patch_rows: int,
        patch_cols: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        row_values = torch.linspace(
            -1.0, 1.0, patch_rows, device=device, dtype=dtype
        ).repeat_interleave(patch_cols)
        col_values = torch.linspace(-1.0, 1.0, patch_cols, device=device, dtype=dtype).repeat(
            patch_rows
        )
        row_width = self.output_dim // 2
        col_width = self.output_dim - row_width
        patch_positions = torch.cat(
            [
                self._axis_position_embedding(row_values, row_width),
                self._axis_position_embedding(col_values, col_width),
            ],
            dim=1,
        )
        class_position = patch_positions.new_zeros(1, self.output_dim)
        return torch.cat([class_position, patch_positions], dim=0).unsqueeze(0)

    def _encode_frames(self, frames: Tensor) -> Tensor:
        if frames.shape[-2] < self.patch_size or frames.shape[-1] < self.patch_size:
            raise ValueError(
                f"ViTBackbone patch_size={self.patch_size} is larger than image shape "
                f"{tuple(frames.shape[-2:])}."
            )
        patches = self.patch_embed(frames)
        tokens = rearrange(
            patches,
            "frames features height width -> frames (height width) features",
        )
        class_tokens = self.class_token.expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype)
        tokens = torch.cat([class_tokens, tokens], dim=1)
        tokens = tokens + self._position_embedding(
            patches.shape[-2],
            patches.shape[-1],
            tokens.device,
            tokens.dtype,
        )
        return self.norm(self.encoder(tokens)[:, 0])

    def forward(self, observations: Tensor) -> Tensor:
        batch_size, time_steps, _channels, _height, _width = observations.shape
        flattened = rearrange(
            observations,
            "batch time channels height width -> (batch time) channels height width",
        )
        encoded_chunks: list[Tensor] = []
        for start_index in range(0, flattened.shape[0], self.max_frames_per_chunk):
            chunk = flattened[start_index : start_index + self.max_frames_per_chunk]
            encoded_chunks.append(self._encode_frames(chunk))
        encoded = torch.cat(encoded_chunks, dim=0)
        return rearrange(
            encoded,
            "(batch time) features -> batch time features",
            batch=batch_size,
            time=time_steps,
        )


class ResNetBackbone(nn.Module):
    """ResNet18 feature extractor for RGB sequences."""

    def __init__(
        self,
        output_dim: int,
        pretrained: bool,
        input_channels: int = 3,
        max_frames_per_chunk: int = 256,
    ) -> None:
        super().__init__()
        if int(input_channels) != 3:
            raise ValueError(f"ResNetBackbone requires 3 RGB channels, got {input_channels}.")
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torchvision is required for ResNetBackbone") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.projection = nn.Linear(backbone.fc.in_features, output_dim)
        self.output_dim = output_dim
        self.max_frames_per_chunk = max(1, int(max_frames_per_chunk))
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, observations: Tensor) -> Tensor:
        batch_size, time_steps, _channels, _height, _width = observations.shape
        flattened = rearrange(
            observations,
            "batch time channels height width -> (batch time) channels height width",
        )
        flattened = flattened.float()
        normalized = (flattened - self.image_mean) / self.image_std
        encoded_chunks: list[Tensor] = []
        for start_index in range(0, normalized.shape[0], self.max_frames_per_chunk):
            chunk = normalized[start_index : start_index + self.max_frames_per_chunk]
            encoded_chunks.append(self.stem(chunk).flatten(start_dim=1))
        encoded = torch.cat(encoded_chunks, dim=0)
        projected = self.projection(encoded)
        return rearrange(
            projected,
            "(batch time) features -> batch time features",
            batch=batch_size,
            time=time_steps,
        )
