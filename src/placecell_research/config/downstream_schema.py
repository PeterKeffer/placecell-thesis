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
    _dump_config,
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
DownstreamHerGoalSelectionStrategy = Literal["future", "final", "episode"]
DownstreamGoalCodeDistanceMetric = Literal["l2", "cosine"]
DownstreamHerGoalRepresentation = Literal["goal_xy", "goal_place_code", "goal_grid_code"]
DownstreamFeatureExtractor = Literal["concat_mlp", "split_position_goal", "paired_grid_code"]
DownstreamReplayBufferType = Literal["uniform", "nstep"]
DownstreamGoalSchedule = Literal["fixed", "cycle", "random", "uniform_random", "codebook"]


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamModelConfig:
    place_model_artifact_id: str = ""
    place_model_checkpoint: Literal["best_primary", "last"] | None = None
    vision_encoder_artifact_id: str = ""
    online_pcdt_checkpoint_path: str = ""
    place_representation_source: str = "encoder.place_codes"
    place_code_stats_path: str = ""
    ae_latent_projection_path: str = ""
    goal_codebook_path: str = ""


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
    distance_metric: DownstreamGoalCodeDistanceMetric = "l2"
    success_threshold: float = Field(default=0.35, gt=0.0)


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
class OnlinePCDTConfig:
    """Sequence, predictive, and IQL settings used only by online_pcdt."""

    context_length: int = Field(default=8, ge=1)
    prediction_horizon: int = Field(default=16, ge=1)
    action_repeat: int = Field(default=1, ge=1)
    mask_stalled_forward: bool = False
    exclude_stalled_forward_control_anchors: bool = False
    stalled_forward_min_displacement: float = Field(default=0.01, gt=0.0)
    stalled_forward_turn_steps: int = Field(default=1, ge=1)
    normalize_position_inputs: bool = False
    position_bounds_low: list[float] = field(default_factory=list)
    position_bounds_high: list[float] = field(default_factory=list)
    hidden_dim: int = Field(default=128, ge=8)
    encoder_layers: int = Field(default=2, ge=1)
    policy_layers: int = Field(default=2, ge=1)
    critic_layers: int = Field(default=2, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    mask_probability: float = Field(default=0.3, ge=0.0, lt=1.0)
    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    lean_masked_reconstruction: bool = False
    future_prediction_weight: float = Field(default=1.0, ge=0.0)
    goal_prediction_weight: float = Field(default=1.0, gt=0.0)
    predictive_learning_rate: float = Field(default=1e-4, gt=0.0)
    expectile: float = Field(default=0.7, gt=0.0, lt=1.0)
    advantage_inverse_temperature: float = Field(default=3.0, ge=0.0)
    max_advantage_weight: float = Field(default=100.0, ge=1.0)
    control_objective: Literal["iql", "gcsl", "lean_gcsl"] = "iql"
    direct_goal_conditioning: bool = False
    hindsight_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    hindsight_near_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    hindsight_near_horizon: int = Field(default=20, ge=1)
    encoder_hindsight_near_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    hindsight_min_goal_distance: float = Field(default=0.0, ge=0.0)
    hindsight_control_segment_length: int = Field(default=1, ge=1)
    segment_goal_contrast_weight: float = Field(default=0.0, ge=0.0)
    segment_goal_contrast_margin: float = Field(default=1.0, ge=0.0)
    hindsight_same_goal_segment: bool = False
    hindsight_near_same_goal_segment: bool = False
    stationary_hindsight_ablation: bool = False
    spatially_balanced_replay: bool = False
    spatial_replay_bin_size: float = Field(default=1.0, gt=0.0)
    route_stitching: bool = False
    route_stitching_min_transitions: int = Field(default=1024, ge=1)
    route_stitching_waypoint_distance: float = Field(default=2.0, gt=0.0)
    practice_goal_relabeling: bool = False
    practice_goal_strategy: Literal[
        "annulus", "progress", "route_progress", "route_frontier"
    ] = "annulus"
    practice_goal_min_transitions: int = Field(default=1024, ge=1)
    practice_goal_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    practice_goal_min_distance: float = Field(default=2.0, gt=0.0)
    practice_goal_max_distance: float = Field(default=8.0, gt=0.0)
    practice_goal_min_progress: float = Field(default=0.5, ge=0.0)
    plan_kl_weight: float = Field(default=1e-3, ge=0.0)
    plan_commit_steps: int = Field(default=10, ge=1)
    target_entropy: float = Field(default=0.3, ge=0.0)
    entropy_dual_learning_rate: float = Field(default=3e-3, gt=0.0)
    initial_entropy_coefficient: float = Field(default=0.05, gt=0.0)
    target_tau: float = Field(default=0.005, gt=0.0, le=1.0)
    encoder_target_tau: float | None = Field(default=None, gt=0.0, le=1.0)
    use_evaluation_actor: bool = False
    evaluation_actor_tau: float = Field(default=0.005, gt=0.0, le=1.0)
    goal_chain_on_success: bool = False
    reward_scale: float = Field(default=1.0, gt=0.0)
    checkpoint_include_replay: bool = True
    resume_checkpoint_path: str = ""


@dataclass(config=PYDANTIC_CONFIG)
class DownstreamTrainingConfig:
    algorithm: DownstreamAlgorithm = "ppo"
    her_goal_representation: DownstreamHerGoalRepresentation = "goal_xy"
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
    bootstrap_heads: int = Field(default=10, ge=2)
    bootstrap_prior_scale: float = Field(default=1.0, gt=0.0)
    bootstrap_mask_probability: float = Field(default=0.5, gt=0.0, le=1.0)
    success_replay_capacity_per_goal: int = Field(default=32, ge=1)
    success_replay_batch_size: int = Field(default=128, ge=1)
    success_replay_loss_coefficient: float = Field(default=1.0, gt=0.0)
    success_replay_greedy_epsilon: float = Field(default=0.02, ge=0.0, le=1.0)
    her_n_sampled_goal: int = Field(default=4, ge=1)
    her_goal_selection_strategy: DownstreamHerGoalSelectionStrategy = "future"
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
    online_pcdt: OnlinePCDTConfig | None = None


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
class DownstreamGoalTeacherConfig:
    """Adaptive frontier teacher for goal-conditioned downstream training."""

    kind: Literal["amigo_frontier"] = "amigo_frontier"
    max_spawn_distance_levels: list[float] = field(
        default_factory=lambda: [5.0, 8.0, 12.0, 18.0, 26.0, 36.0, 50.0]
    )
    initial_level: int = Field(default=0, ge=0)
    window_episodes: int = Field(default=20, ge=1)
    minimum_challenge_steps: int = Field(default=7, ge=1)
    maximum_challenge_steps: int = Field(default=256, ge=1)
    frontier_successes_to_raise_target: int = Field(default=10, ge=1)
    promote_too_easy_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    demote_too_hard_rate: float = Field(default=0.6, ge=0.0, le=1.0)


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
    goal_teacher: DownstreamGoalTeacherConfig | None = None
    preview: DownstreamPreviewConfig = field(default_factory=DownstreamPreviewConfig)
    rollout: DownstreamRolloutConfig = field(default_factory=DownstreamRolloutConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)

    def to_dict(self) -> dict[str, Any]:
        return _dump_config(self)
