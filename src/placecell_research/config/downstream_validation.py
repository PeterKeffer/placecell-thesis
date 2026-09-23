"""Validation of downstream navigation configs."""

from __future__ import annotations

from .downstream_feature_sources import (
    PLACE_CODE_FEATURE_SOURCES,
    PLACE_CODE_SOURCES,
    missing_place_code_stats_sources,
    resolved_place_code_stats_path,
)
from .downstream_schema import DownstreamRunConfig
from .loader import revalidate_config

GOAL_FEATURE_SOURCES = frozenset(
    {
        "goal_xy",
        "goal_xy_scaled",
        "goal_xy_map01",
        "goal_delta_xy",
        "goal_rbf_code",
        "goal_place_code",
        "goal_grid_code",
    }
)
RAW_PIXEL_AUXILIARY_SOURCES = frozenset(
    {"goal_xy", "goal_xy_scaled", "goal_xy_map01", "heading_sin_cos"}
)


def _validate_training_contract(config: DownstreamRunConfig) -> None:
    gradient_steps = config.training.gradient_steps
    if isinstance(gradient_steps, str):
        if gradient_steps.strip().lower() != "auto_default_utd":
            raise ValueError(
                "training.gradient_steps must be a positive integer or 'auto_default_utd'."
            )
    elif int(gradient_steps) < 1:
        raise ValueError(
            "training.gradient_steps must be a positive integer or 'auto_default_utd'."
        )
    if config.observation.mode == "raw_pixels":
        unsupported_raw_sources = sorted(
            set(config.observation.feature_sources) - RAW_PIXEL_AUXILIARY_SOURCES
        )
        if unsupported_raw_sources:
            raise ValueError(
                "raw_pixels mode may only define small auxiliary feature sources "
                f"{sorted(RAW_PIXEL_AUXILIARY_SOURCES)!r}; got {unsupported_raw_sources!r}."
            )
    if config.observation.mode == "feature_vector" and not config.observation.feature_sources:
        raise ValueError("feature_vector mode requires at least one feature source.")


def _validate_place_code_sources(config: DownstreamRunConfig) -> list[str]:
    warnings: list[str] = []
    feature_sources = config.observation.feature_sources
    missing_place_code_sources = sorted(
        PLACE_CODE_FEATURE_SOURCES.intersection(feature_sources)
        if not config.models.place_model_artifact_id
        else []
    )
    if missing_place_code_sources:
        raise ValueError(
            f"feature source {missing_place_code_sources[0]!r} "
            "requires models.place_model_artifact_id."
        )
    missing_stats_sources = missing_place_code_stats_sources(
        feature_sources,
        stats_available=resolved_place_code_stats_path(config.models) is not None,
    )
    if missing_stats_sources:
        raise ValueError(
            f"feature source {missing_stats_sources[0]!r} requires models.place_code_stats_path."
        )
    head_norm_sources = sorted(
        source
        for source in feature_sources
        if source in PLACE_CODE_SOURCES and PLACE_CODE_SOURCES[source].pre_scale == "head_row_norm"
    )
    if head_norm_sources and config.models.place_representation_source != "encoder.place_codes":
        raise ValueError(
            f"feature source {head_norm_sources[0]!r} uses head_row_norm pre-scaling, which reads "
            "the encoder code-head weights and requires "
            "models.place_representation_source='encoder.place_codes'; got "
            f"{config.models.place_representation_source!r}."
        )
    if "goal_place_code" in feature_sources and not config.models.place_model_artifact_id:
        raise ValueError(
            "feature source 'goal_place_code' requires models.place_model_artifact_id."
        )
    if "goal_place_code" in feature_sources and int(config.training.n_envs) > 1:
        raise ValueError("goal_place_code feature source currently requires training.n_envs=1.")
    if not any(
        source in PLACE_CODE_FEATURE_SOURCES | {"ae_latent", "goal_place_code"}
        for source in feature_sources
    ):
        return warnings
    vision_reference = config.models.vision_encoder_artifact_id or ""
    normalized_vision_reference = str(vision_reference).strip().lower()
    if not normalized_vision_reference and not config.models.place_model_artifact_id:
        warnings.append(
            "A learned visual feature source is configured without "
            "models.vision_encoder_artifact_id. This is valid only if the place model "
            "artifact already embeds raw-RGB support."
        )
    elif not normalized_vision_reference and config.models.place_model_artifact_id:
        warnings.append(
            "models.vision_encoder_artifact_id is empty. For latent-input place models, "
            "downstream will fail fast unless you set an exact vision-encoder artifact id, "
            "a tag reference, or 'auto' to infer the matching encoder from the place-model "
            "lineage."
        )
    elif normalized_vision_reference == "auto" and not config.models.place_model_artifact_id:
        warnings.append(
            "models.vision_encoder_artifact_id=auto requires "
            "models.place_model_artifact_id so downstream can infer the matching vision "
            "encoder from the place-model lineage."
        )
    return warnings


def _validate_goal_contract(config: DownstreamRunConfig) -> list[str]:
    warnings: list[str] = []
    feature_sources = config.observation.feature_sources
    schedule = str(config.goal_task.schedule).strip().lower()
    environment_kind = str(config.environment.kind).strip().lower()
    if schedule == "uniform_random" and environment_kind not in {"miniworld", "jaxenstein"}:
        raise ValueError(
            "goal_task.schedule='uniform_random' supports environment.kind 'miniworld' or "
            "'jaxenstein'."
        )
    if (
        GOAL_FEATURE_SOURCES.intersection(feature_sources)
        and not config.goal_task.candidate_positions_xy
        and schedule != "uniform_random"
    ):
        raise ValueError("Goal feature sources require goal_task.candidate_positions_xy.")
    if "goal_rbf_code" in feature_sources and not config.goal_task.candidate_positions_xy:
        raise ValueError(
            "observation.feature_sources=['goal_rbf_code'] requires "
            "goal_task.candidate_positions_xy."
        )
    if (
        config.preview.save_ae_reconstruction_gif
        and not str(config.models.vision_encoder_artifact_id or "").strip()
    ):
        warnings.append(
            "preview.save_ae_reconstruction_gif is enabled, but "
            "models.vision_encoder_artifact_id is empty. Downstream previews will include RGB "
            "and trajectory outputs only."
        )
    return warnings


def _validate_curriculum(config: DownstreamRunConfig) -> list[str]:
    if config.curriculum is None:
        return []
    if str(config.environment.kind).strip().lower() != "miniworld":
        raise ValueError(
            "downstream curriculum currently supports only environment.kind='miniworld'."
        )
    warnings: list[str] = []
    candidate_goal_count = len(config.goal_task.candidate_positions_xy)
    feature_sources = config.observation.feature_sources
    for phase in config.curriculum.spawn_schedule:
        if phase.spawn_region_xz is not None and len(phase.spawn_region_xz) != 4:
            raise ValueError(
                f"Curriculum phase '{phase.name}' spawn_region_xz must contain exactly 4 floats."
            )
        if phase.goal_index is not None and phase.goal_schedule == "uniform_random":
            raise ValueError(
                f"Curriculum phase '{phase.name}' may not combine goal_index with "
                "goal_schedule='uniform_random'."
            )
        if phase.goal_index is not None and candidate_goal_count == 0:
            raise ValueError(
                f"Curriculum phase '{phase.name}' sets goal_index, but "
                "goal_task.candidate_positions_xy is empty."
            )
        if phase.goal_index is not None and phase.goal_index >= candidate_goal_count:
            raise ValueError(
                f"Curriculum phase '{phase.name}' goal_index={phase.goal_index} exceeds the "
                f"configured candidate goal count {candidate_goal_count}."
            )
        if phase.goal_schedule in {"fixed", "cycle", "random"} and candidate_goal_count == 0:
            raise ValueError(
                f"Curriculum phase '{phase.name}' overrides goal scheduling, but "
                "goal_task.candidate_positions_xy is empty."
            )
        if phase.goal_schedule == "uniform_random" and "goal_rbf_code" in feature_sources:
            warnings.append(
                f"Curriculum phase '{phase.name}' uses goal_schedule='uniform_random' while "
                "observation.feature_sources includes 'goal_rbf_code'. The RBF code will still "
                "be computed, but it will encode distances from arbitrary runtime goals to the "
                "fixed candidate anchor set."
            )
        if phase.goal_schedule == "uniform_random" and "goal_place_code" in feature_sources:
            warnings.append(
                f"Curriculum phase '{phase.name}' uses goal_schedule='uniform_random' while "
                "observation.feature_sources includes 'goal_place_code'. Downstream will snapshot "
                "and encode those runtime goals on demand instead of using the cached candidate "
                "goal codes."
            )
    return warnings


def validate_downstream_run_config(config: DownstreamRunConfig) -> list[str]:
    """Raise on an invalid downstream config, return warnings otherwise."""
    config = revalidate_config(config)
    _validate_training_contract(config)
    return [
        *_validate_place_code_sources(config),
        *_validate_goal_contract(config),
        *_validate_curriculum(config),
    ]
