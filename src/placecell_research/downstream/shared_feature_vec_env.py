"""Parent-side shared feature extraction for vectorized downstream RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.downstream_feature_sources import (
    PLACE_CODE_SOURCES,
    missing_place_code_stats_sources,
    resolved_place_code_stats_path,
)
from placecell_research.config.downstream_schema import (
    DownstreamModelConfig,
    DownstreamObservationConfig,
)
from placecell_research.downstream.feature_sources import (
    AELatentFeatureSource,
    apply_place_code_source,
    encoder_head_row_norms,
    heading_sin_cos,
    scale_goal_xy,
    scale_xy_to_unit_box,
)
from placecell_research.downstream.frozen_extractor import FrozenRepresentationExtractor
from placecell_research.downstream.place_code_stats import (
    PlaceCodeStats,
    load_place_code_stats,
)
from placecell_research.downstream.synthetic_grid_cells import (
    build_synthetic_grid_cell_bank,
    synthetic_grid_cell_code,
)
from placecell_research.downstream.synthetic_place_cells import (
    build_place_bank_from_config,
    resolve_environment_xz_bounds,
    synthetic_place_cell_code,
)

from .runtime import (
    place_model_requires_external_vision_encoder,
    resolve_downstream_model_artifacts,
    resolve_inference_device,
)

_FEATURE_MICROBATCH_SIZE = 16


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms <= 1e-8, 1.0, norms)
    return (matrix / safe_norms).astype(np.float32, copy=False)


def _rgb_batch_to_chw(rgb_batch: np.ndarray) -> np.ndarray:
    rgb_batch = np.asarray(rgb_batch)
    if rgb_batch.ndim != 4:
        raise ValueError(
            f"Expected batched RGB frames with 4 dims, got shape {tuple(rgb_batch.shape)}."
        )
    if rgb_batch.shape[1] in {1, 3}:
        return rgb_batch.astype(np.uint8, copy=False)
    return np.transpose(rgb_batch, (0, 3, 1, 2)).astype(np.uint8, copy=False)


def _rgb_frame_to_hwc(rgb_frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb_frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected RGB frame with 3 dims, got shape {tuple(frame.shape)}.")
    if frame.shape[0] in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame.astype(np.uint8, copy=False)


class _SharedAELatentRuntime:
    def __init__(
        self, vision_encoder_path: Path, device: str, projection_path: str = ""
    ) -> None:
        self.source = AELatentFeatureSource(
            vision_encoder_path=vision_encoder_path, device=device, projection_path=projection_path
        )
        self.feature_dim = int(self.source.feature_dim)

    def _encode_microbatch(self, rgb_batch: np.ndarray) -> np.ndarray:
        chw_batch = _rgb_batch_to_chw(rgb_batch)
        rgb_tensor = (
            torch.as_tensor(chw_batch, dtype=torch.float32, device=self.source.device) / 255.0
        )
        model = self.source._ensure_model(rgb_tensor)
        with torch.inference_mode():
            latent = model.encode(rgb_tensor)
        if isinstance(latent, tuple):
            latent = latent[0]
        features = (
            latent.reshape(latent.shape[0], -1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        return self.source.project(features)

    def encode(
        self,
        rgb_batch: np.ndarray,
        microbatch_size: int = _FEATURE_MICROBATCH_SIZE,
    ) -> np.ndarray:
        batch_size = int(rgb_batch.shape[0])
        chunks = []
        for start in range(0, batch_size, int(microbatch_size)):
            end = min(start + int(microbatch_size), batch_size)
            chunks.append(self._encode_microbatch(rgb_batch[start:end]))
        return np.concatenate(chunks, axis=0)


class _SharedPlaceRepresentationRuntime:
    def __init__(
        self,
        *,
        model_artifact_path: Path,
        vision_encoder_path: Path | None,
        representation_source: str,
        device: str,
        checkpoint_selection: str,
    ) -> None:
        self.extractor = FrozenRepresentationExtractor(
            model_checkpoint=model_artifact_path,
            vision_encoder=vision_encoder_path,
            device=device,
            representation_source=representation_source,
            checkpoint_selection=checkpoint_selection,
        )
        self.device = self.extractor.device
        self.feature_dim = int(self.extractor.feature_dim)
        self._head_row_norms: np.ndarray | None = None

    @property
    def head_row_norms(self) -> np.ndarray:
        if self._head_row_norms is None:
            self._head_row_norms = encoder_head_row_norms(self.extractor)
        return self._head_row_norms

    def _encode_microbatch(
        self,
        *,
        rgb_batch: np.ndarray,
        previous_actions: np.ndarray,
        kinematics_batch: np.ndarray | None,
        indices: np.ndarray | None,
    ) -> np.ndarray:
        return self.extractor.extract_batch(
            rgb_batch=rgb_batch,
            previous_actions=previous_actions,
            kinematics_batch=kinematics_batch,
            indices=indices,
        )

    def encode(
        self,
        *,
        rgb_batch: np.ndarray,
        previous_actions: np.ndarray,
        kinematics_batch: np.ndarray | None,
        indices: np.ndarray | None = None,
        microbatch_size: int = _FEATURE_MICROBATCH_SIZE,
    ) -> np.ndarray:
        if self.extractor.uses_stateful_encoder:
            return self._encode_microbatch(
                rgb_batch=rgb_batch,
                previous_actions=previous_actions,
                kinematics_batch=kinematics_batch,
                indices=indices,
            )
        batch_size = int(rgb_batch.shape[0])
        chunks = []
        for start in range(0, batch_size, int(microbatch_size)):
            end = min(start + int(microbatch_size), batch_size)
            chunks.append(
                self._encode_microbatch(
                    rgb_batch=rgb_batch[start:end],
                    previous_actions=previous_actions[start:end],
                    kinematics_batch=(
                        None if kinematics_batch is None else kinematics_batch[start:end]
                    ),
                    indices=None,
                )
            )
        return np.concatenate(chunks, axis=0)

    def reset(self, indices: np.ndarray | None = None) -> None:
        self.extractor.reset(indices)


class _BatchedKinematicsTracker:
    def __init__(self, num_envs: int) -> None:
        self._previous_position_xy = np.zeros((num_envs, 2), dtype=np.float32)
        self._previous_heading = np.zeros((num_envs,), dtype=np.float32)
        self._initialized = np.zeros((num_envs,), dtype=bool)

    def reset(self, indices: np.ndarray | None = None) -> None:
        if indices is None:
            self._initialized[:] = False
            return
        self._initialized[np.asarray(indices, dtype=np.int64)] = False

    def observe(
        self,
        positions_xy: np.ndarray,
        headings: np.ndarray,
        *,
        indices: np.ndarray,
        update: bool,
    ) -> np.ndarray:
        positions_xy = np.asarray(positions_xy, dtype=np.float32).reshape(len(indices), 2)
        headings = np.asarray(headings, dtype=np.float32).reshape(len(indices))
        tracker_indices = np.asarray(indices, dtype=np.int64)
        previous_position = self._previous_position_xy[tracker_indices]
        previous_heading = self._previous_heading[tracker_indices]
        initialized = self._initialized[tracker_indices]
        displacement = positions_xy - previous_position
        step_displacement = np.linalg.norm(displacement, axis=1)
        heading_delta = (headings - previous_heading + np.pi) % (2 * np.pi) - np.pi
        step_displacement = np.where(initialized, step_displacement, 0.0).astype(np.float32)
        heading_delta = np.where(initialized, heading_delta, 0.0).astype(np.float32)
        kinematics = np.stack(
            [step_displacement, heading_delta, np.sin(headings), np.cos(headings)],
            axis=1,
        ).astype(np.float32, copy=False)
        if update:
            self._previous_position_xy[tracker_indices] = positions_xy
            self._previous_heading[tracker_indices] = headings
            self._initialized[tracker_indices] = True
        return kinematics


@dataclass(slots=True)
class SharedFeaturePipeline:
    observation: DownstreamObservationConfig
    goal_candidate_positions_xy: np.ndarray
    goal_rbf_sigma: float
    place_runtime: _SharedPlaceRepresentationRuntime | None
    ae_runtime: _SharedAELatentRuntime | None
    place_code_stats: PlaceCodeStats | None = None
    synthetic_centers: np.ndarray | None = None
    synthetic_sigma_center: float = 0.0
    synthetic_sigma_surround: float = 0.0
    synthetic_normalization: str = "none"
    grid_wave_vectors: np.ndarray | None = None
    grid_phases: np.ndarray | None = None
    grid_normalization: str = "none"
    map_bounds_xz: tuple[float, float, float, float] | None = None

    @property
    def feature_dim(self) -> int:
        total = 0
        for source_name in self.observation.feature_sources:
            if source_name in PLACE_CODE_SOURCES:
                if self.place_runtime is None:
                    raise ValueError(f"{source_name} requested without a place-model runtime.")
                total += int(self.place_runtime.feature_dim)
            elif source_name == "ae_latent":
                if self.ae_runtime is None:
                    raise ValueError("ae_latent requested without a vision runtime.")
                total += int(self.ae_runtime.feature_dim)
            elif source_name in {
                "goal_xy",
                "goal_xy_scaled",
                "goal_xy_map01",
                "goal_delta_xy",
                "heading_sin_cos",
                "current_position_xy",
                "current_position_xy_scaled",
                "current_position_xy_map01",
            }:
                total += 2
            elif source_name == "goal_rbf_code":
                total += int(self.goal_candidate_positions_xy.shape[0])
            elif source_name == "goal_place_code":
                if self.place_runtime is None:
                    raise ValueError("goal_place_code requested without a place-model runtime.")
                total += int(self.place_runtime.feature_dim)
            elif source_name == "synthetic_place_cells":
                if self.synthetic_centers is None:
                    raise ValueError("synthetic_place_cells requested without resolved centers.")
                total += int(self.synthetic_centers.shape[0])
            elif source_name in {"synthetic_grid_cells", "goal_grid_code"}:
                if self.grid_wave_vectors is None:
                    raise ValueError(f"{source_name} requested without a resolved grid bank.")
                total += int(self.grid_wave_vectors.shape[0])
            else:
                raise ValueError(f"Unsupported downstream feature source: {source_name}")
        if self.observation.include_current_position_xy:
            total += 2
        return int(total)

    def reset(self, indices: np.ndarray | None = None) -> None:
        if self.place_runtime is not None:
            self.place_runtime.reset(indices)

    def encode(
        self,
        *,
        rgb_batch: np.ndarray,
        positions_xy: np.ndarray,
        headings: np.ndarray,
        kinematics_batch: np.ndarray,
        goal_positions_xy: np.ndarray | None,
        goal_place_codes: np.ndarray | None,
        previous_actions: np.ndarray,
        indices: np.ndarray | None,
    ) -> np.ndarray:
        features: list[np.ndarray] = []
        place_codes: np.ndarray | None = None
        for source_name in self.observation.feature_sources:
            if source_name in PLACE_CODE_SOURCES:
                if self.place_runtime is None:
                    raise ValueError(f"{source_name} requested without a place-model runtime.")
                if place_codes is None:
                    place_codes = self.place_runtime.encode(
                        rgb_batch=rgb_batch,
                        previous_actions=previous_actions,
                        kinematics_batch=kinematics_batch,
                        indices=indices,
                    )
                features.append(
                    apply_place_code_source(
                        source_name,
                        place_codes,
                        head_row_norms=(
                            self.place_runtime.head_row_norms
                            if PLACE_CODE_SOURCES[source_name].pre_scale == "head_row_norm"
                            else None
                        ),
                        stats=self.place_code_stats,
                    )
                )
            elif source_name == "ae_latent":
                if self.ae_runtime is None:
                    raise ValueError("ae_latent requested without a vision runtime.")
                features.append(self.ae_runtime.encode(rgb_batch))
            elif source_name == "goal_xy":
                if goal_positions_xy is None:
                    raise ValueError(
                        "goal_xy requested, but the env did not expose goal coordinates."
                    )
                features.append(goal_positions_xy.astype(np.float32, copy=False))
            elif source_name == "goal_xy_scaled":
                if goal_positions_xy is None:
                    raise ValueError(
                        "goal_xy_scaled requested, but the env did not expose goal coordinates."
                    )
                features.append(scale_goal_xy(goal_positions_xy))
            elif source_name == "goal_xy_map01":
                if goal_positions_xy is None:
                    raise ValueError(
                        "goal_xy_map01 requested, but the env did not expose goal coordinates."
                    )
                if self.map_bounds_xz is None:
                    raise ValueError("goal_xy_map01 requested without resolved map bounds.")
                features.append(scale_xy_to_unit_box(goal_positions_xy, self.map_bounds_xz))
            elif source_name == "goal_delta_xy":
                if goal_positions_xy is None:
                    raise ValueError(
                        "goal_delta_xy requested, but the env did not expose goal coordinates."
                    )
                features.append((goal_positions_xy - positions_xy).astype(np.float32, copy=False))
            elif source_name == "heading_sin_cos":
                features.append(heading_sin_cos(headings).reshape(len(headings), 2))
            elif source_name == "current_position_xy":
                features.append(positions_xy.astype(np.float32, copy=False))
            elif source_name == "current_position_xy_scaled":
                features.append(scale_goal_xy(positions_xy))
            elif source_name == "current_position_xy_map01":
                if self.map_bounds_xz is None:
                    raise ValueError(
                        "current_position_xy_map01 requested without resolved map bounds."
                    )
                features.append(scale_xy_to_unit_box(positions_xy, self.map_bounds_xz))
            elif source_name == "goal_rbf_code":
                if goal_positions_xy is None:
                    raise ValueError(
                        "goal_rbf_code requested, but the env did not expose goal coordinates."
                    )
                deltas = (
                    self.goal_candidate_positions_xy[None, :, :] - goal_positions_xy[:, None, :]
                )
                squared_distance = np.square(deltas).sum(axis=2)
                if self.goal_rbf_sigma <= 0.0:
                    weights = np.zeros_like(squared_distance, dtype=np.float32)
                    weights[np.arange(weights.shape[0]), np.argmin(squared_distance, axis=1)] = 1.0
                else:
                    weights = np.exp(
                        -squared_distance / (2.0 * self.goal_rbf_sigma * self.goal_rbf_sigma)
                    ).astype(np.float32, copy=False)
                features.append(_normalize_rows(weights))
            elif source_name == "synthetic_place_cells":
                if self.synthetic_centers is None:
                    raise ValueError("synthetic_place_cells requested without resolved centers.")
                features.append(
                    synthetic_place_cell_code(
                        positions_xy,
                        self.synthetic_centers,
                        self.synthetic_sigma_center,
                        self.synthetic_sigma_surround,
                        self.synthetic_normalization,
                    )
                )
            elif source_name == "synthetic_grid_cells":
                if self.grid_wave_vectors is None or self.grid_phases is None:
                    raise ValueError("synthetic_grid_cells requested without a resolved grid bank.")
                features.append(
                    synthetic_grid_cell_code(
                        positions_xy,
                        self.grid_wave_vectors,
                        self.grid_phases,
                        self.grid_normalization,
                    )
                )
            elif source_name == "goal_grid_code":
                if self.grid_wave_vectors is None or self.grid_phases is None:
                    raise ValueError("goal_grid_code requested without a resolved grid bank.")
                if goal_positions_xy is None:
                    raise ValueError(
                        "goal_grid_code requested, but the env did not expose goal coordinates."
                    )
                features.append(
                    synthetic_grid_cell_code(
                        goal_positions_xy,
                        self.grid_wave_vectors,
                        self.grid_phases,
                        self.grid_normalization,
                    )
                )
            elif source_name == "goal_place_code":
                if goal_place_codes is None:
                    raise ValueError(
                        "goal_place_code requested, but the env did not expose goal place codes."
                    )
                features.append(goal_place_codes.astype(np.float32, copy=False))
            else:
                raise ValueError(f"Unsupported downstream feature source: {source_name}")
        if self.observation.include_current_position_xy:
            features.append(positions_xy.astype(np.float32, copy=False))
        concatenated = np.concatenate(
            [feature.reshape(feature.shape[0], -1) for feature in features], axis=1
        )
        concatenated = concatenated.astype(np.float32, copy=False)
        if self.observation.normalize_concatenated_features:
            concatenated = _normalize_rows(concatenated)
        return concatenated


def build_shared_feature_pipeline(
    *,
    artifact_registry: ArtifactRegistry,
    models: DownstreamModelConfig,
    observation: DownstreamObservationConfig,
    env_id: str,
    goal_candidate_positions_xy: list[list[float]],
    goal_rbf_sigma: float,
    device: str,
) -> SharedFeaturePipeline:
    resolved_device = resolve_inference_device(device)
    resolved_artifacts = resolve_downstream_model_artifacts(
        artifact_registry=artifact_registry,
        models=models,
        observation=observation,
    )
    vision_encoder_path = (
        None
        if resolved_artifacts.vision_encoder is None
        else resolved_artifacts.vision_encoder.path
    )
    place_runtime = None
    ae_runtime = None
    place_code_stats = None
    place_code_stats_path = resolved_place_code_stats_path(models)
    if place_code_stats_path:
        place_code_stats = load_place_code_stats(place_code_stats_path)
    place_code_source_names = set(observation.feature_sources).intersection(PLACE_CODE_SOURCES)
    missing_stats_sources = missing_place_code_stats_sources(
        place_code_source_names,
        stats_available=place_code_stats is not None,
    )
    if missing_stats_sources:
        raise ValueError(
            f"feature source '{missing_stats_sources[0]}' requires models.place_code_stats_path."
        )
    if place_code_source_names or "goal_place_code" in observation.feature_sources:
        place_model_artifact = resolved_artifacts.place_model
        if place_model_artifact is None:
            raise ValueError(
                "feature source 'place_codes*' or 'goal_place_code' "
                "requires models.place_model_artifact_id."
            )
        if vision_encoder_path is None and place_model_requires_external_vision_encoder(
            place_model_artifact
        ):
            raise ValueError(
                f"Place model '{place_model_artifact.artifact_id}' expects latent inputs from a "
                f"vision encoder. "
                "Set models.vision_encoder_artifact_id to an exact artifact id, a tag reference, "
                "or 'auto'."
            )
        place_runtime = _SharedPlaceRepresentationRuntime(
            model_artifact_path=place_model_artifact.path,
            vision_encoder_path=vision_encoder_path,
            representation_source=models.place_representation_source,
            device=resolved_device,
            checkpoint_selection=models.place_model_checkpoint,
        )
    if "ae_latent" in observation.feature_sources:
        if vision_encoder_path is None:
            raise ValueError(
                "feature source 'ae_latent' requires models.vision_encoder_artifact_id."
            )
        ae_runtime = _SharedAELatentRuntime(
            vision_encoder_path=vision_encoder_path,
            device=resolved_device,
            projection_path=models.ae_latent_projection_path,
        )
    synthetic_centers = None
    synthetic_sigma_center = 0.0
    synthetic_sigma_surround = 0.0
    synthetic_normalization = "none"
    if "synthetic_place_cells" in observation.feature_sources:
        synthetic_config = observation.synthetic_place_cells
        (
            synthetic_centers,
            synthetic_sigma_center,
            synthetic_sigma_surround,
        ) = build_place_bank_from_config(synthetic_config, env_id)
        synthetic_normalization = synthetic_config.normalization
    grid_wave_vectors = None
    grid_phases = None
    grid_normalization = "none"
    if {"synthetic_grid_cells", "goal_grid_code"}.intersection(observation.feature_sources):
        grid_config = observation.synthetic_grid_cells
        bounds_xz = resolve_environment_xz_bounds(env_id, grid_config.bounds_xz)
        grid_wave_vectors, grid_phases = build_synthetic_grid_cell_bank(
            bounds_xz,
            grid_config.num_cells,
            grid_config.num_modules,
            grid_config.min_period,
            grid_config.period_ratio,
            grid_config.orientation_degrees,
            grid_config.orientation_jitter_degrees,
            grid_config.seed,
        )
        grid_normalization = grid_config.normalization
    map_bounds_xz = None
    if {"current_position_xy_map01", "goal_xy_map01"}.intersection(observation.feature_sources):
        map_bounds_xz = resolve_environment_xz_bounds(env_id)
    return SharedFeaturePipeline(
        observation=observation,
        goal_candidate_positions_xy=np.asarray(goal_candidate_positions_xy, dtype=np.float32),
        goal_rbf_sigma=float(goal_rbf_sigma),
        place_runtime=place_runtime,
        ae_runtime=ae_runtime,
        place_code_stats=place_code_stats,
        synthetic_centers=synthetic_centers,
        synthetic_sigma_center=synthetic_sigma_center,
        synthetic_sigma_surround=synthetic_sigma_surround,
        synthetic_normalization=synthetic_normalization,
        grid_wave_vectors=grid_wave_vectors,
        grid_phases=grid_phases,
        grid_normalization=grid_normalization,
        map_bounds_xz=map_bounds_xz,
    )


class SharedFeatureVecEnvWrapper(VecEnvWrapper):
    """Transform raw-pixel vector env outputs into shared parent-side feature vectors."""

    def __init__(self, venv: VecEnv, pipeline: SharedFeaturePipeline):
        self.pipeline = pipeline
        self._pending_actions = np.zeros((venv.num_envs,), dtype=np.int64)
        self._kinematics = _BatchedKinematicsTracker(venv.num_envs)
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(pipeline.feature_dim),),
            dtype=np.float32,
        )
        super().__init__(venv, observation_space=observation_space)

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        super().step_async(actions)

    def _extract_arrays_from_info_batch(
        self,
        infos: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        batch_size = len(infos)
        positions_xy = np.asarray(
            [info["position_xy"] for info in infos], dtype=np.float32
        ).reshape(batch_size, 2)
        headings = np.asarray([float(info.get("heading", 0.0)) for info in infos], dtype=np.float32)
        goal_positions: list[np.ndarray] = []
        goal_place_codes: list[np.ndarray] = []
        has_goal = False
        has_goal_place_code = False
        for info in infos:
            goal_position_xy = info.get("goal_position_xy")
            if goal_position_xy is None:
                goal_positions.append(np.zeros((2,), dtype=np.float32))
            else:
                has_goal = True
                goal_positions.append(np.asarray(goal_position_xy, dtype=np.float32).reshape(2))
            goal_place_code = info.get("goal_place_code")
            if goal_place_code is None:
                if self.pipeline.place_runtime is None:
                    goal_place_codes.append(np.zeros((0,), dtype=np.float32))
                else:
                    goal_place_codes.append(
                        np.zeros((self.pipeline.place_runtime.feature_dim,), dtype=np.float32)
                    )
            else:
                has_goal_place_code = True
                goal_place_codes.append(np.asarray(goal_place_code, dtype=np.float32).reshape(-1))
        goal_batch = np.stack(goal_positions, axis=0) if has_goal else None
        goal_place_code_batch = np.stack(goal_place_codes, axis=0) if has_goal_place_code else None
        return positions_xy, headings, goal_batch, goal_place_code_batch

    def _encode_from_infos(
        self,
        raw_obs: np.ndarray,
        infos: list[dict[str, Any]],
        *,
        previous_actions: np.ndarray,
        indices: np.ndarray,
        update_kinematics: bool,
    ) -> np.ndarray:
        positions_xy, headings, goal_positions, goal_place_codes = (
            self._extract_arrays_from_info_batch(infos)
        )
        kinematics_batch = self._kinematics.observe(
            positions_xy,
            headings,
            indices=indices,
            update=update_kinematics,
        )
        return self.pipeline.encode(
            rgb_batch=np.asarray(raw_obs),
            positions_xy=positions_xy,
            headings=headings,
            kinematics_batch=kinematics_batch,
            goal_positions_xy=goal_positions,
            goal_place_codes=goal_place_codes,
            previous_actions=previous_actions,
            indices=indices,
        )

    def reset(self) -> np.ndarray:
        raw_obs = np.asarray(self.venv.reset())
        infos = [dict(info) for info in list(self.venv.reset_infos)]
        for env_index, info in enumerate(infos):
            info["rgb_frame"] = _rgb_frame_to_hwc(raw_obs[env_index])
        self.reset_infos = infos
        self.pipeline.reset()
        self._kinematics.reset()
        self._pending_actions = np.zeros((self.num_envs,), dtype=np.int64)
        return self._encode_from_infos(
            raw_obs,
            infos,
            previous_actions=np.zeros((self.num_envs,), dtype=np.int64),
            indices=np.arange(self.num_envs, dtype=np.int64),
            update_kinematics=True,
        )

    def step_wait(self):
        raw_obs, rewards, dones, infos = self.venv.step_wait()
        raw_obs = np.asarray(raw_obs)
        done_flags = np.asarray(dones, dtype=bool)
        info_dicts = [dict(info) for info in infos]

        terminal_indices = [
            env_index
            for env_index, done in enumerate(done_flags.tolist())
            if done and "terminal_observation" in info_dicts[env_index]
        ]
        if terminal_indices:
            terminal_infos = [info_dicts[env_index] for env_index in terminal_indices]
            terminal_obs = np.stack(
                [
                    np.asarray(info_dicts[env_index]["terminal_observation"])
                    for env_index in terminal_indices
                ],
                axis=0,
            )
            terminal_index_array = np.asarray(terminal_indices, dtype=np.int64)
            terminal_vector = self._encode_from_infos(
                terminal_obs,
                terminal_infos,
                previous_actions=self._pending_actions[terminal_index_array],
                indices=terminal_index_array,
                update_kinematics=False,
            )
            for output_index, env_index in enumerate(terminal_indices):
                info_dicts[env_index]["terminal_observation"] = terminal_vector[output_index]
                info_dicts[env_index]["rgb_frame"] = _rgb_frame_to_hwc(terminal_obs[output_index])

        if np.any(done_flags):
            reset_indices = np.flatnonzero(done_flags).astype(np.int64, copy=False)
            self.pipeline.reset(reset_indices)
            self._kinematics.reset(reset_indices)
        live_infos = []
        previous_actions = np.zeros((self.num_envs,), dtype=np.int64)
        for env_index in range(self.num_envs):
            if done_flags[env_index]:
                reset_info = dict(self.venv.reset_infos[env_index])
                reset_info["rgb_frame"] = _rgb_frame_to_hwc(raw_obs[env_index])
                next_start_position_xy = reset_info.get("position_xy")
                if next_start_position_xy is not None:
                    info_dicts[env_index]["next_start_position_xy"] = [
                        float(value)
                        for value in np.asarray(next_start_position_xy, dtype=np.float32).reshape(2)
                    ]
                live_infos.append(reset_info)
            else:
                info_dicts[env_index]["rgb_frame"] = _rgb_frame_to_hwc(raw_obs[env_index])
                live_infos.append(info_dicts[env_index])
                previous_actions[env_index] = int(self._pending_actions[env_index])
        self.reset_infos = [dict(info) for info in live_infos]
        live_obs = self._encode_from_infos(
            raw_obs,
            live_infos,
            previous_actions=previous_actions,
            indices=np.arange(self.num_envs, dtype=np.int64),
            update_kinematics=True,
        )
        return live_obs, rewards, dones, info_dicts
