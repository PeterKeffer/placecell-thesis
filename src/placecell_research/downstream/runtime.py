"""Runtime helpers for downstream RL commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from placecell_research.artifacts.manifests import ArtifactManifest
from placecell_research.artifacts.registry import ArtifactRegistry, RegisteredArtifact
from placecell_research.config.downstream_feature_sources import (
    PLACE_CODE_SOURCES,
    missing_place_code_stats_sources,
    resolved_place_code_stats_path,
)
from placecell_research.config.downstream_schema import (
    DownstreamGoalCodeConfig,
    DownstreamModelConfig,
    DownstreamObservationConfig,
)
from placecell_research.utils.device import resolve_device

from .feature_sources import (
    AELatentFeatureSource,
    CurrentPositionFeatureSource,
    CurrentPositionMap01FeatureSource,
    FeatureConcatenator,
    GoalGridCellsFeatureSource,
    GoalPlaceCodeFeatureSource,
    GoalRBFCodeFeatureSource,
    GoalXYFeatureSource,
    GoalXYMap01FeatureSource,
    HeadingSinCosFeatureSource,
    PlaceCodeFeatureRuntime,
    PlaceCodeFeatureSource,
    SyntheticGridCellsFeatureSource,
    SyntheticPlaceCellsFeatureSource,
)
from .goal_codes import GoalPlaceCodeRuntime
from .place_code_stats import load_place_code_stats
from .synthetic_grid_cells import build_grid_bank_from_config
from .synthetic_place_cells import (
    build_place_bank_from_config,
    resolve_environment_xz_bounds,
)


@dataclass(slots=True)
class ResolvedModelArtifact:
    """Resolved model artifact, either registry-backed or direct-path-backed."""

    artifact_id: str
    artifact_type: str
    path: Path
    manifest: ArtifactManifest | None


@dataclass(slots=True)
class ResolvedDownstreamModelArtifacts:
    """Resolved upstream artifacts used by downstream runs."""

    place_model: ResolvedModelArtifact | None
    vision_encoder: ResolvedModelArtifact | None


def resolve_inference_device(device: str) -> str:
    return str(resolve_device(device))


def _absolute_path_reference(artifact_reference: str) -> Path | None:
    path = Path(artifact_reference).expanduser()
    if path.is_absolute():
        return path
    return None


def _read_direct_path_manifest(path: Path) -> ArtifactManifest | None:
    manifest_path = path / "manifest.json" if path.is_dir() else path.with_name("manifest.json")
    if manifest_path.exists():
        return ArtifactManifest.read(manifest_path)
    if path.is_dir():
        raise FileNotFoundError(f"Artifact directory is missing manifest.json: {path}")
    return None


def _resolve_direct_path_artifact(
    artifact_reference: str, *, expected_type: str
) -> ResolvedModelArtifact | None:
    path = _absolute_path_reference(artifact_reference)
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {path}")
    manifest = _read_direct_path_manifest(path)
    if manifest is not None and manifest.artifact_type != expected_type:
        raise ValueError(
            f"Artifact path '{path}' has type '{manifest.artifact_type}', expected "
            f"'{expected_type}'."
        )
    artifact_id = manifest.artifact_id if manifest is not None else path.stem
    artifact_type = manifest.artifact_type if manifest is not None else expected_type
    return ResolvedModelArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        path=path,
        manifest=manifest,
    )


def _from_registered_artifact(artifact: RegisteredArtifact | None) -> ResolvedModelArtifact | None:
    if artifact is None:
        return None
    return ResolvedModelArtifact(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        path=artifact.path,
        manifest=artifact.manifest,
    )


def _resolve_optional_artifact(
    registry: ArtifactRegistry,
    artifact_reference: str,
    *,
    expected_type: str,
) -> ResolvedModelArtifact | None:
    normalized_reference = str(artifact_reference or "").strip()
    if not normalized_reference:
        return None
    direct_path_artifact = _resolve_direct_path_artifact(
        normalized_reference, expected_type=expected_type
    )
    if direct_path_artifact is not None:
        return direct_path_artifact
    if ArtifactRegistry.is_tag_reference(normalized_reference):
        return _from_registered_artifact(
            registry.resolve_completed_reference(expected_type, normalized_reference)
        )
    artifact = registry.find_by_id(normalized_reference)
    if artifact is None:
        raise FileNotFoundError(f"Could not find artifact '{normalized_reference}'.")
    if artifact.artifact_type != expected_type:
        raise ValueError(
            f"Artifact '{normalized_reference}' has type '{artifact.artifact_type}', expected "
            f"'{expected_type}'."
        )
    registry.require_completed(artifact)
    return _from_registered_artifact(artifact)


def _infer_matching_vision_encoder_artifact(
    registry: ArtifactRegistry,
    place_model_artifact: ResolvedModelArtifact | None,
) -> ResolvedModelArtifact | None:
    if place_model_artifact is None or place_model_artifact.manifest is None:
        return None
    summary = dict(place_model_artifact.manifest.summary or {})
    explicit_artifact_id = str(summary.get("vision_encoder_artifact_id") or "").strip()
    if explicit_artifact_id:
        artifact = registry.find_by_id(explicit_artifact_id)
        if artifact is None:
            raise FileNotFoundError(
                "The selected place model references vision encoder "
                f"'{explicit_artifact_id}', but that artifact is missing."
            )
        if artifact.artifact_type != "vision_encoder":
            raise ValueError(
                f"Place model '{place_model_artifact.artifact_id}' references "
                f"'{explicit_artifact_id}' as its vision encoder, but that artifact has type "
                f"'{artifact.artifact_type}'."
            )
        return _from_registered_artifact(artifact)
    matching_inputs = [
        artifact
        for input_artifact_id in place_model_artifact.manifest.input_artifact_ids
        if (artifact := registry.find_by_id(str(input_artifact_id))) is not None
        and artifact.artifact_type == "vision_encoder"
    ]
    if not matching_inputs:
        return None
    if len(matching_inputs) > 1:
        raise ValueError(
            f"Place model '{place_model_artifact.artifact_id}' has multiple vision-encoder "
            f"ancestors. "
            "Set models.vision_encoder_artifact_id explicitly."
        )
    return _from_registered_artifact(matching_inputs[0])


def resolve_downstream_model_artifacts(
    *,
    artifact_registry: ArtifactRegistry,
    models: DownstreamModelConfig,
    observation: DownstreamObservationConfig | None = None,
    require_vision_encoder: bool = False,
) -> ResolvedDownstreamModelArtifacts:
    normalized_place_model_reference = str(models.place_model_artifact_id or "").strip()
    normalized_vision_reference = str(models.vision_encoder_artifact_id or "").strip()
    place_model_artifact = _resolve_optional_artifact(
        artifact_registry,
        normalized_place_model_reference,
        expected_type="place_model",
    )
    should_try_inference = require_vision_encoder or normalized_vision_reference.lower() == "auto"
    if normalized_vision_reference and normalized_vision_reference.lower() != "auto":
        vision_encoder_artifact = _resolve_optional_artifact(
            artifact_registry,
            normalized_vision_reference,
            expected_type="vision_encoder",
        )
    elif should_try_inference:
        vision_encoder_artifact = _infer_matching_vision_encoder_artifact(
            artifact_registry,
            place_model_artifact,
        )
    else:
        vision_encoder_artifact = None
    if require_vision_encoder and vision_encoder_artifact is None:
        if place_model_artifact is None:
            raise ValueError(
                "A vision encoder is required here, but downstream could not infer one because "
                "models.place_model_artifact_id is empty."
            )
        raise ValueError(
            f"Downstream could not infer a matching vision encoder for place model "
            f"'{place_model_artifact.artifact_id}'. Set models.vision_encoder_artifact_id "
            f"explicitly."
        )
    return ResolvedDownstreamModelArtifacts(
        place_model=place_model_artifact,
        vision_encoder=vision_encoder_artifact,
    )


def place_model_requires_external_vision_encoder(
    place_model_artifact: ResolvedModelArtifact | None,
) -> bool:
    """Return whether the selected place model expects latent inputs from a vision encoder."""
    if place_model_artifact is None or place_model_artifact.manifest is None:
        return False
    observation_source = (
        str(place_model_artifact.manifest.summary.get("observation_source") or "").strip().lower()
    )
    return observation_source == "latent"


def build_feature_extractor(
    *,
    artifact_registry: ArtifactRegistry,
    models: DownstreamModelConfig,
    observation: DownstreamObservationConfig,
    env_id: str,
    goal_candidate_positions_xy: list[list[float]],
    goal_rbf_sigma: float,
    device: str,
    goal_place_code_dim: int | None = None,
) -> FeatureConcatenator | None:
    if not observation.feature_sources and not observation.include_current_position_xy:
        return None

    sources = []
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
    place_code_runtime = None
    place_code_stats = None
    place_code_stats_path = resolved_place_code_stats_path(models)
    if place_code_stats_path:
        place_code_stats = load_place_code_stats(place_code_stats_path)
    missing_stats_sources = missing_place_code_stats_sources(
        observation.feature_sources,
        stats_available=place_code_stats is not None,
    )
    if missing_stats_sources:
        raise ValueError(
            f"feature source '{missing_stats_sources[0]}' requires models.place_code_stats_path."
        )

    for source_name in observation.feature_sources:
        if source_name in PLACE_CODE_SOURCES:
            place_model_artifact = resolved_artifacts.place_model
            if place_model_artifact is None:
                raise ValueError(
                    f"feature source '{source_name}' requires models.place_model_artifact_id."
                )
            if vision_encoder_path is None and place_model_requires_external_vision_encoder(
                place_model_artifact
            ):
                raise ValueError(
                    f"Place model '{place_model_artifact.artifact_id}' expects latent inputs from "
                    f"a vision encoder. "
                    "Set models.vision_encoder_artifact_id to an exact artifact id, a tag "
                    "reference, or 'auto' "
                    "to infer the matching encoder from the place-model lineage."
                )
            if place_code_runtime is None:
                place_code_runtime = PlaceCodeFeatureRuntime(
                    model_artifact_path=place_model_artifact.path,
                    vision_encoder_path=vision_encoder_path,
                    representation_source=models.place_representation_source,
                    device=resolved_device,
                    checkpoint_selection=models.place_model_checkpoint,
                    place_code_stats=place_code_stats,
                )
            sources.append(
                PlaceCodeFeatureSource(
                    runtime=place_code_runtime,
                    source_name=source_name,
                )
            )
        elif source_name == "ae_latent":
            if vision_encoder_path is None:
                raise ValueError(
                    "feature source 'ae_latent' requires models.vision_encoder_artifact_id."
                )
            sources.append(
                AELatentFeatureSource(
                    vision_encoder_path=vision_encoder_path,
                    device=resolved_device,
                    projection_path=models.ae_latent_projection_path,
                )
            )
        elif source_name == "goal_xy":
            sources.append(GoalXYFeatureSource(delta=False))
        elif source_name == "goal_xy_scaled":
            sources.append(GoalXYFeatureSource(delta=False, scaled=True))
        elif source_name == "goal_xy_map01":
            bounds_xz = resolve_environment_xz_bounds(env_id)
            sources.append(GoalXYMap01FeatureSource(bounds_xz=bounds_xz))
        elif source_name == "goal_delta_xy":
            sources.append(GoalXYFeatureSource(delta=True))
        elif source_name == "heading_sin_cos":
            sources.append(HeadingSinCosFeatureSource())
        elif source_name == "current_position_xy":
            sources.append(CurrentPositionFeatureSource(scaled=False))
        elif source_name == "current_position_xy_scaled":
            sources.append(CurrentPositionFeatureSource(scaled=True))
        elif source_name == "current_position_xy_map01":
            bounds_xz = resolve_environment_xz_bounds(env_id)
            sources.append(CurrentPositionMap01FeatureSource(bounds_xz=bounds_xz))
        elif source_name == "goal_rbf_code":
            sources.append(
                GoalRBFCodeFeatureSource(
                    candidate_positions_xy=goal_candidate_positions_xy,
                    sigma=goal_rbf_sigma,
                )
            )
        elif source_name == "goal_place_code":
            if goal_place_code_dim is None:
                raise ValueError(
                    "feature source 'goal_place_code' requires a resolved goal-place runtime "
                    "dimension."
                )
            sources.append(GoalPlaceCodeFeatureSource(feature_dim=goal_place_code_dim))
        elif source_name == "synthetic_place_cells":
            synthetic_config = observation.synthetic_place_cells
            centers, sigma_center, sigma_surround = build_place_bank_from_config(
                synthetic_config, env_id
            )
            sources.append(
                SyntheticPlaceCellsFeatureSource(
                    centers=centers,
                    sigma_center=sigma_center,
                    sigma_surround=sigma_surround,
                    normalization=synthetic_config.normalization,
                )
            )
        elif source_name == "synthetic_grid_cells":
            grid_config = observation.synthetic_grid_cells
            wave_vectors, phases = build_grid_bank_from_config(grid_config, env_id)
            sources.append(
                SyntheticGridCellsFeatureSource(
                    wave_vectors=wave_vectors,
                    phases=phases,
                    normalization=grid_config.normalization,
                )
            )
        elif source_name == "goal_grid_code":
            grid_config = observation.synthetic_grid_cells
            wave_vectors, phases = build_grid_bank_from_config(grid_config, env_id)
            sources.append(
                GoalGridCellsFeatureSource(
                    wave_vectors=wave_vectors,
                    phases=phases,
                    normalization=grid_config.normalization,
                )
            )
        else:
            raise ValueError(f"Unsupported downstream feature source: {source_name}")

    if observation.include_current_position_xy:
        sources.append(CurrentPositionFeatureSource())
    return FeatureConcatenator(sources, normalize=observation.normalize_concatenated_features)


def build_goal_place_code_runtime(
    *,
    artifact_registry: ArtifactRegistry,
    models: DownstreamModelConfig,
    goal_code: DownstreamGoalCodeConfig,
    device: str,
) -> GoalPlaceCodeRuntime:
    resolved_device = resolve_inference_device(device)
    resolved_artifacts = resolve_downstream_model_artifacts(
        artifact_registry=artifact_registry,
        models=models,
        observation=None,
    )
    place_model_artifact = resolved_artifacts.place_model
    if place_model_artifact is None:
        raise ValueError("Place-code goals require models.place_model_artifact_id.")
    vision_encoder_path = (
        None
        if resolved_artifacts.vision_encoder is None
        else resolved_artifacts.vision_encoder.path
    )
    if vision_encoder_path is None and place_model_requires_external_vision_encoder(
        place_model_artifact
    ):
        raise ValueError(
            f"Place model '{place_model_artifact.artifact_id}' expects latent inputs from a vision "
            f"encoder. "
            "Set models.vision_encoder_artifact_id to an exact artifact id, a tag reference, or "
            "'auto'."
        )
    return GoalPlaceCodeRuntime.build(
        model_artifact_path=place_model_artifact.path,
        vision_encoder_path=vision_encoder_path,
        representation_source=models.place_representation_source,
        device=resolved_device,
        checkpoint_selection=models.place_model_checkpoint,
        goal_code_config=goal_code,
    )
