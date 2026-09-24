"""Typed configs for downstream RL evaluation."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from .schema import (
    PYDANTIC_CONFIG,
    EnvironmentConfig,
    LauncherConfig,
    TrackingConfig,
    dump_config,
)

VectorFeatureSource = Literal[
    "place_codes",
    "place_codes_l2",
    "place_codes_binary",
    "place_codes_headnorm_l2",
    "place_codes_active_rms",
    "place_codes_rms_l2",
    "place_codes_zscore",
    "ae_latent",
    "goal_xy",
    "goal_xy_scaled",
    "goal_xy_map01",
    "goal_delta_xy",
    "goal_rbf_code",
    "goal_place_code",
    "goal_grid_code",
    "heading_sin_cos",
    "current_position_xy",
    "current_position_xy_scaled",
    "current_position_xy_map01",
    "synthetic_place_cells",
    "synthetic_grid_cells",
]
DownstreamAlgorithm = Literal["ppo", "dqn"]
DownstreamTrainFreqUnit = Literal["step", "episode"]
DownstreamGradientSteps = int | Literal["auto_default_utd"]
DownstreamFeatureExtractor = Literal["concat_mlp", "split_position_goal", "paired_grid_code"]
DownstreamReplayBufferType = Literal["uniform", "nstep"]
DownstreamGoalSchedule = Literal["fixed", "cycle", "random", "uniform_random"]


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamModelConfig:
    place_model_artifact_id: str = ""
    place_model_checkpoint: Literal["best_primary", "last"] | None = None
    vision_encoder_artifact_id: str = ""
    place_representation_source: str = "encoder.place_codes"
    place_code_stats_path: str = ""
    ae_latent_projection_path: str = ""


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamGoalTaskConfig:
    candidate_positions_xy: list[list[float]] = field(default_factory=list)
    schedule: DownstreamGoalSchedule = "fixed"
    change_interval_episodes: int = Field(default=1, ge=1)
    initial_goal_index: int = Field(default=0, ge=0)
    rbf_sigma: float = Field(default=1.5, gt=0.0)


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamGoalCodeConfig:
    snapshot_heading_degrees: float = 0.0
    normalize_codes: bool = True


@dataclass(config=PYDANTIC_CONFIG)
class SyntheticPlaceCellsConfig:
    """Sorscher-style oracle place-cell baseline (difference-of-softmax over true position)."""

    num_cells: int = Field(default=512, ge=1)
    sigma_center: float = Field(default=0.0, ge=0.0)
    surround_scale: float = Field(default=2.0, gt=0.0)
    seed: int = 0
    bounds_xz: list[float] = field(default_factory=list)
    normalization: Literal["none", "l2"] = "none"


@dataclass(config=PYDANTIC_CONFIG)
class SyntheticGridCellsConfig:
    """Solstad-style oracle grid-cell baseline (three-plane-wave hexagonal fields over position)."""

    num_cells: int = Field(default=512, ge=1)
    num_modules: int = Field(default=4, ge=1)
    min_period: float = Field(default=0.0, ge=0.0)
    period_ratio: float = Field(default=1.42, gt=0.0)
    orientation_degrees: float = 0.0
    orientation_jitter_degrees: float = Field(default=0.0, ge=0.0)
    seed: int = 0
    bounds_xz: list[float] = field(default_factory=list)
    normalization: Literal["none", "l2"] = "none"


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamObservationConfig:
    mode: Literal["feature_vector", "raw_pixels"] = "feature_vector"
    feature_sources: list[VectorFeatureSource] = field(default_factory=lambda: ["place_codes"])
    normalize_concatenated_features: bool = False
    include_current_position_xy: bool = False
    synthetic_place_cells: SyntheticPlaceCellsConfig = field(
        default_factory=SyntheticPlaceCellsConfig
    )
    synthetic_grid_cells: SyntheticGridCellsConfig = field(
        default_factory=SyntheticGridCellsConfig
    )


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamTrainingConfig:
    algorithm: DownstreamAlgorithm = "ppo"
    total_timesteps: int = Field(default=100_000, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    n_steps: int = Field(default=1024, ge=1)
    batch_size: int = Field(default=64, ge=1)
    n_epochs: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    clip_range: float = Field(default=0.2, ge=0.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, ge=0.0)
    policy_hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    feature_extractor: DownstreamFeatureExtractor = "concat_mlp"
    feature_extractor_embed_dim: int = Field(default=64, ge=1)
    cnn_features_dim: int = Field(default=0, ge=0)
    buffer_size: int = Field(default=100_000, ge=1)
    learning_starts: int = Field(default=1_000, ge=0)
    tau: float = Field(default=1.0, gt=0.0, le=1.0)
    train_freq: int = Field(default=4, ge=1)
    train_freq_unit: DownstreamTrainFreqUnit = "step"
    gradient_steps: DownstreamGradientSteps = 1
    target_update_interval: int = Field(default=1_000, ge=1)
    replay_buffer_type: DownstreamReplayBufferType = "uniform"
    n_step_returns: int = Field(default=3, ge=1)
    exploration_fraction: float = Field(default=0.1, ge=0.0)
    exploration_initial_eps: float = Field(default=1.0, ge=0.0, le=1.0)
    exploration_final_eps: float = Field(default=0.05, ge=0.0, le=1.0)
    n_envs: int = Field(default=1, ge=1)
    device: str = "auto"
    log_interval: int = Field(default=20, ge=1)
    stats_window_size: int = Field(default=100, ge=1)
    progress_bar: bool = False
    checkpoint_freq: int = Field(default=20_000, ge=1)
    eval_freq: int = Field(default=10_000, ge=0)
    eval_episodes: int = Field(default=8, ge=0)
    eval_stochastic: bool = True
    final_eval_episodes: int = Field(default=8, ge=0)
    final_eval_stochastic: bool = True
    final_eval_diagnostics: bool = True
    memory_watchdog_limit_mb: int = Field(default=0, ge=0)


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamSpawnCurriculumPhaseConfig:
    name: str = "phase"
    start_episode: int = Field(default=0, ge=0)
    spawn_regions: list[str] = field(default_factory=list)
    spawn_region_xz: list[float] | None = None
    max_spawn_distance: float | None = Field(default=None, gt=0.0)
    goal_tolerance: float | None = Field(default=None, gt=0.0)
    goal_schedule: DownstreamGoalSchedule | None = None
    goal_change_interval_episodes: int | None = Field(default=None, ge=1)
    goal_index: int | None = Field(default=None, ge=0)


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamCurriculumConfig:
    spawn_schedule: list[DownstreamSpawnCurriculumPhaseConfig] = field(default_factory=list)


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamPreviewConfig:
    enabled: bool = True
    every_n_episodes: int = Field(default=10, ge=1)
    episodes_per_preview: int = Field(default=2, ge=1)
    max_steps_per_episode: int = Field(default=128, ge=1)
    max_animation_frames: int = Field(default=96, ge=2)
    deterministic_policy: bool = True
    gif_frame_duration: float = Field(default=0.12, gt=0.0)
    save_rgb_gif: bool = True
    save_ae_reconstruction_gif: bool = True
    save_trajectory_summary: bool = True


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamRolloutConfig:
    episodes: int = Field(default=3, ge=1)
    max_steps_per_episode: int = Field(default=256, ge=1)
    policy: Literal["random", "forward"] = "random"


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamRunConfig:
    name: str = "downstream_run"
    seed: int = 0
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    models: DownstreamModelConfig = field(default_factory=DownstreamModelConfig)
    goal_task: DownstreamGoalTaskConfig = field(default_factory=DownstreamGoalTaskConfig)
    eval_candidate_positions_xy: list[list[float]] = field(default_factory=list)
    goal_code: DownstreamGoalCodeConfig = field(default_factory=DownstreamGoalCodeConfig)
    observation: DownstreamObservationConfig = field(default_factory=DownstreamObservationConfig)
    training: DownstreamTrainingConfig = field(default_factory=DownstreamTrainingConfig)
    curriculum: DownstreamCurriculumConfig | None = None
    preview: DownstreamPreviewConfig = field(default_factory=DownstreamPreviewConfig)
    rollout: DownstreamRolloutConfig = field(default_factory=DownstreamRolloutConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)

    def to_dict(self) -> dict[str, Any]:
        return dump_config(self)
