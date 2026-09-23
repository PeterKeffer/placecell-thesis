"""Custom SB3 feature extractors for downstream RL."""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from placecell_research.config.downstream_schema import DownstreamObservationConfig

_POSITION_SOURCES = frozenset({"synthetic_grid_cells", "synthetic_place_cells"})
_GOAL_SOURCES = frozenset(
    {
        "goal_xy",
        "goal_xy_scaled",
        "goal_xy_map01",
        "goal_grid_code",
        "goal_place_code",
        "goal_rbf_code",
    }
)
_KNOWN_TWO_DIM_SOURCES = frozenset(
    {
        "goal_xy",
        "goal_xy_scaled",
        "goal_xy_map01",
        "goal_delta_xy",
        "heading_sin_cos",
        "current_position_xy",
        "current_position_xy_scaled",
        "current_position_xy_map01",
    }
)


def _resolve_feature_dims(
    observation: DownstreamObservationConfig,
    *,
    total_dim: int,
) -> list[tuple[str, int]]:
    """(source_name, dim) for each entry in observation.feature_sources, in concatenation order."""
    resolved_dims: list[tuple[str, int | None]] = []
    for name in observation.feature_sources:
        if name in {"synthetic_grid_cells", "goal_grid_code"}:
            resolved_dims.append((name, int(observation.synthetic_grid_cells.num_cells)))
        elif name == "synthetic_place_cells":
            resolved_dims.append((name, int(observation.synthetic_place_cells.num_cells)))
        elif name in _KNOWN_TWO_DIM_SOURCES:
            resolved_dims.append((name, 2))
        elif name in {"goal_place_code", "goal_rbf_code"}:
            resolved_dims.append((name, None))
        else:
            raise ValueError(
                f"split_position_goal feature extractor does not know the dimension of feature "
                f"source {name!r}. Add it to feature_extractors._resolve_feature_dims."
            )
    unresolved_indices = [
        index for index, (_, dimension) in enumerate(resolved_dims) if dimension is None
    ]
    if len(unresolved_indices) > 1:
        unresolved_sources = [resolved_dims[index][0] for index in unresolved_indices]
        raise ValueError(
            "split_position_goal cannot infer multiple runtime-defined feature dimensions: "
            f"{unresolved_sources}."
        )
    known_dim = sum(dimension or 0 for _, dimension in resolved_dims)
    if unresolved_indices:
        inferred_dim = int(total_dim) - known_dim
        if inferred_dim < 1:
            raise ValueError(
                "split_position_goal could not infer a positive runtime-defined feature "
                f"dimension from observation width {total_dim} and known width {known_dim}."
            )
        index = unresolved_indices[0]
        resolved_dims[index] = (resolved_dims[index][0], inferred_dim)
    elif known_dim != int(total_dim):
        raise ValueError(
            f"Configured feature dimensions sum to {known_dim}, but the observation space has "
            f"width {total_dim}."
        )
    return [(name, int(dimension)) for name, dimension in resolved_dims if dimension is not None]


def _gather_indices(block_dims: list[tuple[str, int]], names: frozenset[str]) -> list[int]:
    indices: list[int] = []
    offset = 0
    for name, dim in block_dims:
        if name in names:
            indices.extend(range(offset, offset + dim))
        offset += dim
    return indices


class SplitPositionGoalExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        observation_config: DownstreamObservationConfig,
        embed_dim: int = 64,
    ) -> None:
        block_dims = _resolve_feature_dims(
            observation_config,
            total_dim=int(observation_space.shape[0]),
        )
        position_indices = _gather_indices(block_dims, _POSITION_SOURCES)
        goal_indices = _gather_indices(block_dims, _GOAL_SOURCES)
        other_indices = _gather_indices(
            block_dims,
            frozenset(name for name, _ in block_dims) - _POSITION_SOURCES - _GOAL_SOURCES,
        )
        if not position_indices or not goal_indices:
            raise ValueError(
                "split_position_goal requires at least one position-like source "
                f"({sorted(_POSITION_SOURCES)}) and one goal-like source ({sorted(_GOAL_SOURCES)}) "
                f"in observation.feature_sources; got {[name for name, _ in block_dims]}."
            )
        features_dim = embed_dim * 4 + len(other_indices)
        super().__init__(observation_space, features_dim=features_dim)
        self.register_buffer(
            "_position_indices", torch.as_tensor(position_indices, dtype=torch.long)
        )
        self.register_buffer("_goal_indices", torch.as_tensor(goal_indices, dtype=torch.long))
        self.register_buffer("_other_indices", torch.as_tensor(other_indices, dtype=torch.long))
        self.position_encoder = nn.Sequential(
            nn.Linear(len(position_indices), embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(len(goal_indices), embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        position_embedding = self.position_encoder(
            observations.index_select(1, self._position_indices)
        )
        goal_embedding = self.goal_encoder(observations.index_select(1, self._goal_indices))
        parts = [
            position_embedding,
            goal_embedding,
            position_embedding - goal_embedding,
            position_embedding * goal_embedding,
        ]
        if self._other_indices.numel() > 0:
            parts.append(observations.index_select(1, self._other_indices))
        return torch.cat(parts, dim=1)


class PairedGridCodeExtractor(BaseFeaturesExtractor):
    """Expose aligned interactions between current and goal codes from one fixed grid bank."""

    def __init__(self, observation_space, observation_config: DownstreamObservationConfig) -> None:
        block_dims = _resolve_feature_dims(
            observation_config,
            total_dim=int(observation_space.shape[0]),
        )
        current_indices = _gather_indices(block_dims, frozenset({"synthetic_grid_cells"}))
        goal_indices = _gather_indices(block_dims, frozenset({"goal_grid_code"}))
        other_indices = _gather_indices(
            block_dims,
            frozenset(name for name, _ in block_dims)
            - {"synthetic_grid_cells", "goal_grid_code"},
        )
        if not current_indices or len(current_indices) != len(goal_indices):
            raise ValueError(
                "paired_grid_code requires one synthetic_grid_cells block and one equal-sized "
                "goal_grid_code block."
            )
        grid_dim = len(current_indices)
        super().__init__(observation_space, features_dim=4 * grid_dim + len(other_indices))
        self.register_buffer("_current_indices", torch.as_tensor(current_indices, dtype=torch.long))
        self.register_buffer("_goal_indices", torch.as_tensor(goal_indices, dtype=torch.long))
        self.register_buffer("_other_indices", torch.as_tensor(other_indices, dtype=torch.long))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        current_code = observations.index_select(1, self._current_indices)
        goal_code = observations.index_select(1, self._goal_indices)
        parts = [
            current_code,
            goal_code,
            current_code - goal_code,
            current_code * goal_code,
        ]
        if self._other_indices.numel() > 0:
            parts.append(observations.index_select(1, self._other_indices))
        return torch.cat(parts, dim=1)
