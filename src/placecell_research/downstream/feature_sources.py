"""Composable downstream feature sources for RL observations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import yaml

from placecell_research.config import materialize_dataclass
from placecell_research.config.downstream_feature_sources import (
    PLACE_CODE_SOURCES,
    PlaceCodeTransformMode,
)
from placecell_research.config.schema import VisionConfig
from placecell_research.downstream.frozen_extractor import FrozenRepresentationExtractor
from placecell_research.downstream.place_code_stats import PlaceCodeStats
from placecell_research.downstream.synthetic_grid_cells import synthetic_grid_cell_code
from placecell_research.downstream.synthetic_place_cells import synthetic_place_cell_code
from placecell_research.vision.builder import build_vision_model

GOAL_XY_SCALE = 36.0
_EPSILON = 1e-8


@dataclass(slots=True)
class StepContext:
    rgb: np.ndarray
    position_xy: np.ndarray
    heading: float
    kinematics: np.ndarray | None
    goal_position_xy: np.ndarray | None
    current_place_code: np.ndarray | None = None
    goal_place_code: np.ndarray | None = None


class VectorFeatureSource(Protocol):
    name: str
    feature_dim: int

    def reset(self) -> None: ...

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray: ...


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def transform_place_code_batch(
    codes: np.ndarray,
    *,
    mode: PlaceCodeTransformMode,
    stats: PlaceCodeStats | None = None,
) -> np.ndarray:
    matrix = np.asarray(codes, dtype=np.float32)
    if mode == "identity":
        return matrix.astype(np.float32, copy=False)
    if mode == "l2":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms <= _EPSILON, 1.0, norms)
        return (matrix / safe_norms).astype(np.float32, copy=False)
    if mode == "binary":
        return (np.abs(matrix) > _EPSILON).astype(np.float32)
    if stats is None:
        raise ValueError(f"place-code transform {mode!r} requires fixed stats.")
    if mode == "active_rms":
        return (matrix / stats.active_rms.reshape(1, -1)).astype(np.float32, copy=False)
    if mode == "active_rms_l2":
        scaled = matrix / stats.active_rms.reshape(1, -1)
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        return (scaled / np.where(norms <= _EPSILON, 1.0, norms)).astype(np.float32, copy=False)
    if mode == "zscore":
        return ((matrix - stats.mean.reshape(1, -1)) / stats.std.reshape(1, -1)).astype(
            np.float32,
            copy=False,
        )
    raise ValueError(f"Unsupported place-code transform mode: {mode!r}.")


def apply_place_code_source(
    source_name: str,
    codes: np.ndarray,
    *,
    head_row_norms: np.ndarray | None,
    stats: PlaceCodeStats | None,
) -> np.ndarray:
    spec = PLACE_CODE_SOURCES[source_name]
    matrix = np.asarray(codes, dtype=np.float32)
    if spec.pre_scale == "head_row_norm":
        if head_row_norms is None:
            raise ValueError(f"feature source '{source_name}' requires encoder head-row norms.")
        matrix = matrix / np.asarray(head_row_norms, dtype=np.float32).reshape(1, -1)
    return transform_place_code_batch(matrix, mode=spec.transform, stats=stats)


def scale_goal_xy(goal_xy: np.ndarray) -> np.ndarray:
    return (np.asarray(goal_xy, dtype=np.float32) / GOAL_XY_SCALE).astype(np.float32, copy=False)


def scale_xy_to_unit_box(xy: np.ndarray, bounds_xz: Sequence[float]) -> np.ndarray:
    min_x, max_x, min_z, max_z = (float(value) for value in bounds_xz)
    span = np.asarray([max_x - min_x, max_z - min_z], dtype=np.float32)
    if np.any(span <= 0.0):
        raise ValueError(f"Invalid map bounds for xy normalization: {tuple(bounds_xz)!r}.")
    origin = np.asarray([min_x, min_z], dtype=np.float32)
    return ((np.asarray(xy, dtype=np.float32) - origin) / span).astype(np.float32, copy=False)


def heading_sin_cos(heading: np.ndarray | float) -> np.ndarray:
    heading_array = np.asarray(heading, dtype=np.float32)
    return np.stack([np.sin(heading_array), np.cos(heading_array)], axis=-1).astype(
        np.float32, copy=False
    )


def _artifact_file(path_or_directory: Path, filename: str) -> Path:
    if path_or_directory.is_dir():
        return path_or_directory / filename
    return path_or_directory


class PlaceCodeFeatureRuntime:
    """Shared frozen place-model runtime for single-env transformed views."""

    def __init__(
        self,
        model_artifact_path: Path,
        vision_encoder_path: Path | None,
        representation_source: str,
        device: str,
        checkpoint_selection: str,
        place_code_stats: PlaceCodeStats | None = None,
    ) -> None:
        self.extractor = FrozenRepresentationExtractor(
            model_checkpoint=model_artifact_path,
            vision_encoder=vision_encoder_path,
            device=device,
            representation_source=representation_source,
            checkpoint_selection=checkpoint_selection,
        )
        self.feature_dim = int(self.extractor.feature_dim)
        self.place_code_stats = place_code_stats

    def reset(self) -> None:
        self.extractor.reset()

    def extract_raw(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        if context.current_place_code is not None:
            return np.asarray(context.current_place_code, dtype=np.float32).reshape(-1)
        context.current_place_code = self.extractor.extract(
            rgb=context.rgb,
            previous_action=previous_action,
            kinematics=context.kinematics,
        )
        return np.asarray(context.current_place_code, dtype=np.float32).reshape(-1)


class PlaceCodeFeatureSource:
    """Transformed view over a shared frozen place-model runtime."""

    def __init__(
        self,
        *,
        runtime: PlaceCodeFeatureRuntime,
        source_name: str,
    ) -> None:
        self.name = source_name
        self.runtime = runtime
        self.feature_dim = int(runtime.feature_dim)

    def reset(self) -> None:
        self.runtime.reset()

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        code = self.runtime.extract_raw(context, previous_action)
        spec = PLACE_CODE_SOURCES[self.name]
        return apply_place_code_source(
            self.name,
            code.reshape(1, -1),
            head_row_norms=(
                self.runtime.extractor.head_row_norms if spec.pre_scale == "head_row_norm" else None
            ),
            stats=self.runtime.place_code_stats,
        ).reshape(-1)


class AELatentFeatureSource:
    """Frozen vision encoder latent source."""

    def __init__(
        self, vision_encoder_path: Path, device: str, projection_path: str = ""
    ) -> None:
        self.name = "ae_latent"
        self.device = torch.device(device)
        self.vision_encoder_path = Path(vision_encoder_path)
        self._lazy_state_dict: dict[str, torch.Tensor] | None = None
        self._lazy_config: VisionConfig | None = None
        self.model = self._load_model(self.vision_encoder_path)
        self.feature_dim = self._infer_feature_dim(self.vision_encoder_path)
        self.projection = None
        if projection_path:
            projection = np.load(projection_path, allow_pickle=False)
            if (
                projection.ndim != 2
                or projection.shape[0] != self.feature_dim
                or not np.isfinite(projection).all()
            ):
                raise ValueError(
                    "Latent projection must be a finite [latent_dim, output_dim] matrix."
                )
            self.projection = np.asarray(projection, dtype=np.float32)
            self.feature_dim = int(projection.shape[1])

    def _load_model(self, artifact_path: Path) -> torch.nn.Module:
        weights_path = _artifact_file(artifact_path, "weights.pt")
        payload = torch.load(weights_path, map_location=self.device, weights_only=False)
        if hasattr(payload, "encode"):
            model = payload
        elif (
            isinstance(payload, dict) and "model" in payload and hasattr(payload["model"], "encode")
        ):
            model = payload["model"]
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            config_path = weights_path.parent / "training_config.yaml"
            if not config_path.exists():
                raise RuntimeError("Vision encoder artifact is missing training_config.yaml.")
            vision_config = materialize_dataclass(
                VisionConfig, yaml.safe_load(config_path.read_text()) or {}
            )
            self._lazy_state_dict = payload["model_state_dict"]
            self._lazy_config = vision_config
            return None  # type: ignore[return-value]
        else:
            raise RuntimeError("Vision encoder artifact does not expose an encode() method.")
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _ensure_model(self, rgb_tensor: torch.Tensor) -> torch.nn.Module:
        if self.model is not None:
            return self.model
        if self._lazy_state_dict is None or self._lazy_config is None:
            raise RuntimeError("Vision encoder could not be initialized.")
        input_shape = tuple(int(value) for value in rgb_tensor.shape[-3:])
        model = build_vision_model(self._lazy_config, input_shape)
        model.load_state_dict(self._lazy_state_dict)
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self._lazy_state_dict = None
        self._lazy_config = None
        return model

    def _infer_feature_dim(self, artifact_path: Path) -> int:
        contract_path = (
            Path(artifact_path) / "manifest.json"
            if Path(artifact_path).is_dir()
            else Path(artifact_path)
        )
        if contract_path.exists():
            payload = json.loads(contract_path.read_text())
            latent_dim = payload.get("summary", {}).get("latent_dim")
            if latent_dim is not None:
                return int(latent_dim)
        if self._lazy_config is not None:
            return int(self._lazy_config.latent_dim)
        if self.model is None:
            raise RuntimeError(
                "Vision encoder artifact is missing both manifest latent_dim and buildable model."
            )
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 64, 64), dtype=torch.float32, device=self.device)
            latent = self.model.encode(dummy)
        if isinstance(latent, tuple):
            latent = latent[0]
        return int(latent.reshape(1, -1).shape[-1])

    def reset(self) -> None:
        return None

    def _rgb_to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3:
            raise ValueError(f"Expected RGB frame with 3 dims, got {tuple(rgb.shape)}.")
        if rgb.shape[0] in {1, 3}:
            chw = rgb
        else:
            chw = np.transpose(rgb, (2, 0, 1))
        return torch.as_tensor(chw, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        rgb_tensor = self._rgb_to_tensor(context.rgb)
        model = self._ensure_model(rgb_tensor)
        with torch.no_grad():
            latent = model.encode(rgb_tensor)
        if isinstance(latent, tuple):
            latent = latent[0]
        features = latent.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
        return self.project(features)

    def project(self, features: np.ndarray) -> np.ndarray:
        return features if self.projection is None else features @ self.projection

    def reconstruct_rgb(self, rgb: np.ndarray) -> np.ndarray | None:
        rgb_tensor = self._rgb_to_tensor(rgb)
        model = self._ensure_model(rgb_tensor)
        if not hasattr(model, "decode"):
            return None
        with torch.no_grad():
            latent = model.encode(rgb_tensor)
            if isinstance(latent, tuple):
                latent = latent[0]
            reconstruction = model.decode(latent)
        if reconstruction.shape[-2:] != rgb_tensor.shape[-2:]:
            reconstruction = torch.nn.functional.interpolate(
                reconstruction,
                size=rgb_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        reconstruction_hwc = (
            reconstruction[0].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        )
        return np.rint(reconstruction_hwc * 255.0).astype(np.uint8)


class GoalXYFeatureSource:
    """Absolute or relative goal coordinates."""

    def __init__(self, *, delta: bool, scaled: bool = False) -> None:
        if delta:
            self.name = "goal_delta_xy"
        elif scaled:
            self.name = "goal_xy_scaled"
        else:
            self.name = "goal_xy"
        self._delta = bool(delta)
        self._scaled = bool(scaled)
        self.feature_dim = 2

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        if context.goal_position_xy is None:
            raise ValueError(
                f"{self.name} requires a goal position, but the environment did not provide one."
            )
        goal_xy = np.asarray(context.goal_position_xy, dtype=np.float32)
        if self._delta:
            goal_xy = goal_xy - np.asarray(context.position_xy, dtype=np.float32)
        if self._scaled:
            goal_xy = scale_goal_xy(goal_xy)
        return goal_xy.astype(np.float32, copy=False)


class GoalXYMap01FeatureSource:
    name = "goal_xy_map01"
    feature_dim = 2

    def __init__(self, *, bounds_xz: Sequence[float]) -> None:
        self.bounds_xz = tuple(float(value) for value in bounds_xz)

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        if context.goal_position_xy is None:
            raise ValueError(
                "goal_xy_map01 requires a goal position, but the environment did not provide one."
            )
        return scale_xy_to_unit_box(
            np.asarray(context.goal_position_xy, dtype=np.float32), self.bounds_xz
        )


class HeadingSinCosFeatureSource:
    name = "heading_sin_cos"
    feature_dim = 2

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        return heading_sin_cos(float(context.heading)).reshape(2)


class GoalRBFCodeFeatureSource:
    """Coordinate-derived goal code over configured candidate goal positions."""

    def __init__(self, candidate_positions_xy: list[list[float]], sigma: float) -> None:
        if not candidate_positions_xy:
            raise ValueError("goal_rbf_code requires at least one candidate goal position.")
        self.name = "goal_rbf_code"
        self.centers = np.asarray(candidate_positions_xy, dtype=np.float32)
        self.sigma = float(sigma)
        self.feature_dim = int(self.centers.shape[0])

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        if context.goal_position_xy is None:
            raise ValueError(
                "goal_rbf_code requires a goal position, but the environment did not provide one."
            )
        goal_xy = np.asarray(context.goal_position_xy, dtype=np.float32).reshape(1, 2)
        squared_distance = np.square(self.centers - goal_xy).sum(axis=1)
        if self.sigma <= 0.0:
            weights = np.zeros_like(squared_distance, dtype=np.float32)
            weights[int(np.argmin(squared_distance))] = 1.0
            return weights
        weights = np.exp(-squared_distance / (2.0 * self.sigma * self.sigma))
        return _normalize_vector(weights.astype(np.float32, copy=False))


class GoalPlaceCodeFeatureSource:
    """Precomputed place-code target for the active goal."""

    def __init__(self, feature_dim: int) -> None:
        self.name = "goal_place_code"
        self.feature_dim = int(feature_dim)

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        if context.goal_place_code is None:
            raise ValueError(
                "goal_place_code requires the runtime goal place code, but it was not available."
            )
        return np.asarray(context.goal_place_code, dtype=np.float32).reshape(-1)


class CurrentPositionFeatureSource:
    """Current agent position in world XY coordinates."""

    def __init__(self, *, scaled: bool = False) -> None:
        self._scaled = bool(scaled)
        self.name = "current_position_xy_scaled" if self._scaled else "current_position_xy"
        self.feature_dim = 2

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        position_xy = np.asarray(context.position_xy, dtype=np.float32)
        if self._scaled:
            position_xy = scale_goal_xy(position_xy)
        return position_xy.astype(np.float32, copy=False)


class CurrentPositionMap01FeatureSource:
    name = "current_position_xy_map01"
    feature_dim = 2

    def __init__(self, *, bounds_xz: Sequence[float]) -> None:
        self.bounds_xz = tuple(float(value) for value in bounds_xz)

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        return scale_xy_to_unit_box(
            np.asarray(context.position_xy, dtype=np.float32), self.bounds_xz
        )


class SyntheticPlaceCellsFeatureSource:
    """Sorscher-style oracle place-cell code over the agent's ground-truth (x, z) position."""

    name = "synthetic_place_cells"

    def __init__(
        self,
        *,
        centers: np.ndarray,
        sigma_center: float,
        sigma_surround: float,
        normalization: str = "none",
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float32)
        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError(
                f"synthetic place-cell centers must be (num_cells, 2), got {self.centers.shape}."
            )
        self.sigma_center = float(sigma_center)
        self.sigma_surround = float(sigma_surround)
        self.normalization = str(normalization)
        self.feature_dim = int(self.centers.shape[0])

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        position_xz = np.asarray(context.position_xy, dtype=np.float32).reshape(1, 2)
        code = synthetic_place_cell_code(
            position_xz, self.centers, self.sigma_center, self.sigma_surround, self.normalization
        )
        return code.reshape(-1)


class SyntheticGridCellsFeatureSource:
    """Solstad-style oracle grid-cell code over the agent's ground-truth (x, z) position."""

    name = "synthetic_grid_cells"

    def __init__(
        self,
        *,
        wave_vectors: np.ndarray,
        phases: np.ndarray,
        normalization: str = "none",
    ) -> None:
        self.wave_vectors = np.asarray(wave_vectors, dtype=np.float32)
        self.phases = np.asarray(phases, dtype=np.float32)
        if self.wave_vectors.ndim != 3 or self.wave_vectors.shape[1:] != (3, 2):
            raise ValueError(
                f"grid wave_vectors must be (num_cells, 3, 2), got {self.wave_vectors.shape}."
            )
        if self.phases.shape != (self.wave_vectors.shape[0], 2):
            raise ValueError(f"grid phases must be (num_cells, 2), got {self.phases.shape}.")
        self.normalization = str(normalization)
        self.feature_dim = int(self.wave_vectors.shape[0])

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        position_xz = np.asarray(context.position_xy, dtype=np.float32).reshape(1, 2)
        code = synthetic_grid_cell_code(
            position_xz, self.wave_vectors, self.phases, self.normalization
        )
        return code.reshape(-1)


class GoalGridCellsFeatureSource:
    """Grid code of the GOAL position, in the same bank as SyntheticGridCellsFeatureSource."""

    name = "goal_grid_code"

    def __init__(
        self,
        *,
        wave_vectors: np.ndarray,
        phases: np.ndarray,
        normalization: str = "none",
    ) -> None:
        self.wave_vectors = np.asarray(wave_vectors, dtype=np.float32)
        self.phases = np.asarray(phases, dtype=np.float32)
        if self.wave_vectors.ndim != 3 or self.wave_vectors.shape[1:] != (3, 2):
            raise ValueError(
                f"grid wave_vectors must be (num_cells, 3, 2), got {self.wave_vectors.shape}."
            )
        if self.phases.shape != (self.wave_vectors.shape[0], 2):
            raise ValueError(f"grid phases must be (num_cells, 2), got {self.phases.shape}.")
        self.normalization = str(normalization)
        self.feature_dim = int(self.wave_vectors.shape[0])

    def reset(self) -> None:
        return None

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del previous_action
        if context.goal_position_xy is None:
            raise ValueError(
                "goal_grid_code requires a goal position, but the environment did not provide one."
            )
        goal_xz = np.asarray(context.goal_position_xy, dtype=np.float32).reshape(1, 2)
        code = synthetic_grid_cell_code(goal_xz, self.wave_vectors, self.phases, self.normalization)
        return code.reshape(-1)


class FeatureConcatenator:
    """Concatenate multiple vector feature sources into one observation vector."""

    def __init__(self, sources: list[VectorFeatureSource], normalize: bool) -> None:
        if not sources:
            raise ValueError("FeatureConcatenator requires at least one source.")
        self.sources = list(sources)
        self.normalize = bool(normalize)
        self.feature_dim = int(sum(source.feature_dim for source in self.sources))

    def reset(self) -> None:
        for source in self.sources:
            source.reset()

    def extract(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        concatenated = np.concatenate(
            [source.extract(context, previous_action).reshape(-1) for source in self.sources],
            axis=0,
        ).astype(np.float32, copy=False)
        if not self.normalize:
            return concatenated
        return _normalize_vector(concatenated)
