"""Typed config schemas."""

from __future__ import annotations

import math
import warnings
from dataclasses import field, is_dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

from .context_channels import allowed_context_channel_hint, is_context_channel

PYDANTIC_CONFIG = ConfigDict(validate_assignment=True)
ContextChannel = str


def _dump_config(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError(f"Expected a dataclass config object, got {type(value)!r}.")
    return TypeAdapter(type(value)).dump_python(value, mode="json")


@dataclass(config=PYDANTIC_CONFIG)
class ThreadingSafetyConfig:
    omp_num_threads: int = Field(default=1, ge=1)
    mkl_num_threads: int = Field(default=1, ge=1)
    openblas_num_threads: int = Field(default=1, ge=1)
    torch_num_threads: int = Field(default=1, ge=1)


@dataclass(config=PYDANTIC_CONFIG)
class SeedBundleConfig:
    global_seed: int = 42
    collection_seed: int | None = None
    split_seed: int | None = None
    training_seed: int | None = None


@dataclass(config=PYDANTIC_CONFIG)
class PipelineConfig:
    stages: list[str] = field(
        default_factory=lambda: [
            "collect_dataset",
            "create_split",
            "train_vision_encoder",
            "encode_dataset",
            "train_place_model",
            "evaluate_model",
            "analyze_model",
        ]
    )
    stop_after_stage: str | None = None


@dataclass(config=PYDANTIC_CONFIG)
class LauncherConfig:
    type: Literal["local", "slurm"] = "local"
    partition: str = "gpu"
    exclude_nodes: list[str] = field(default_factory=list)
    gpus: int = Field(default=1, ge=0)
    gpu_type: str | None = None
    time_hours: int = Field(default=48, ge=1)
    memory_gb: int = Field(default=32, ge=1)
    cpus_per_task: int = Field(default=4, ge=1)
    env_setup: str = ""
    stream_logs: bool = True
    threading_safety: ThreadingSafetyConfig = field(default_factory=ThreadingSafetyConfig)
    analysis_workers: int = Field(default=0, ge=0)


@dataclass(config=PYDANTIC_CONFIG)
class TrackingOutputTagsConfig:
    vision_encoder: list[str] = field(default_factory=list)
    place_model: list[str] = field(default_factory=list)


@dataclass(config=PYDANTIC_CONFIG)
class TrackingConfig:
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_failure_mode: Literal["warn", "raise"] = "warn"
    log_interval_steps: int = Field(default=50, ge=1)
    run_root: Path = Path("runs")
    artifact_root: Path = Path("artifacts")
    study_name: str = "default_study"
    variant_name: str = "baseline"
    tags: list[str] = field(default_factory=list)
    campaign_name: str = ""
    campaign_group: str = ""
    experiment_id: str = ""
    experiment_arm: str = ""
    output_tags: TrackingOutputTagsConfig = field(default_factory=TrackingOutputTagsConfig)


@dataclass(config=PYDANTIC_CONFIG)
class PolicyConfig:
    artifact_reuse: Literal["error", "reuse_if_config_match", "force_recompute"] = "error"
    training_resume: Literal["fresh", "weights_only", "weights_and_optimizer"] = "fresh"
    auto_resume_interrupted: bool = True
    checkpoint_selection: Literal["auto", "best", "last"] = "last"


@dataclass(config=PYDANTIC_CONFIG)
class ReuseConfig:
    vision_encoder_artifact_id: str = ""
    place_model_artifact_id: str = ""
    representation_set_artifact_id: str = ""
    place_model_checkpoint_path: str = ""


@dataclass(config=PYDANTIC_CONFIG)
class EnvironmentConfig:
    kind: str = "miniworld"
    env_id: str = "MiniWorld-WallGapAsymLarge-v0"
    episode_length: int = Field(default=256, ge=1)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    randomize_agent_start: bool | None = None


@dataclass(config=PYDANTIC_CONFIG)
class CollectionSafetyConfig:
    multiprocessing_start_method: Literal["spawn"] = "spawn"
    num_workers: int = Field(default=1, ge=0)
    stagger_worker_start_seconds: float = Field(default=2.0, ge=0.0)
    watchdog_timeout_seconds: float = Field(default=300.0, gt=0.0)
    memory_watchdog_limit_mb: int = Field(default=0, ge=0)
    worker_max_episodes: int = Field(default=0, ge=0)
    isolate_cuda_miniworld: bool = True
    skip_env_close_on_slurm: bool = True
    scratch_root: Path | None = None
    render_timeout_seconds: float = Field(default=30.0, gt=0.0)
    cpu_encoder_during_collection: bool = False


@dataclass(config=PYDANTIC_CONFIG)
class ContinuousMotionConfig:
    """Rayleigh distances, Gaussian turns, and a turn bias after blocked movement."""

    forward_distance_scale: float = Field(default=0.26, gt=0.0, allow_inf_nan=False)
    turn_std_radians: float = Field(default=0.11519173063162574, gt=0.0, allow_inf_nan=False)
    wall_turn_radians: float = Field(
        default=1.0471975511965976, ge=0.0, le=3.141592653589793, allow_inf_nan=False
    )


@dataclass(config=PYDANTIC_CONFIG)
class CollectionConfig:
    episodes: int = Field(default=128, ge=1)
    episode_length: int = Field(default=256, ge=1)
    policy: Literal[
        "ou_smoothed_random",
        "independent_random",
        "uniform_random",
        "motion_bouts",
        "continuous_random",
    ] = "ou_smoothed_random"
    action_probabilities: dict[str, float] = field(default_factory=dict)
    bout_switch_probability: float = Field(default=0.1, gt=0.0, le=1.0)
    continuous_motion: ContinuousMotionConfig | None = None
    save_rgb: bool = True
    save_topdown: bool = True
    spawn_regions: list[str] = field(default_factory=list)
    safety: CollectionSafetyConfig = field(default_factory=CollectionSafetyConfig)
    vectorized: bool = False
    num_parallel_envs: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_motion_bouts(self) -> CollectionConfig:
        if self.policy == "motion_bouts":
            if self.vectorized or self.action_probabilities:
                raise ValueError(
                    "motion_bouts needs the stepwise sampler and no action probabilities."
                )
        elif self.bout_switch_probability != 0.1:
            raise ValueError("bout_switch_probability requires policy=motion_bouts.")
        return self

    @model_validator(mode="after")
    def validate_continuous_motion(self) -> CollectionConfig:
        if self.policy != "continuous_random":
            if self.continuous_motion is not None:
                raise ValueError("continuous_motion requires policy=continuous_random.")
            return self
        if self.vectorized or self.action_probabilities:
            raise ValueError(
                "continuous_random requires stepwise collection and no action probabilities."
            )
        if self.continuous_motion is None:
            self.continuous_motion = ContinuousMotionConfig()
        return self

    @field_validator("action_probabilities")
    @classmethod
    def validate_action_probabilities(cls, probabilities: dict[str, float]) -> dict[str, float]:
        total_probability = 0.0
        for action_name, probability_value in probabilities.items():
            if not action_name:
                raise ValueError(
                    "collection.action_probabilities keys must be non-empty action names."
                )
            if probability_value != probability_value or probability_value in {
                float("inf"),
                float("-inf"),
            }:
                raise ValueError(
                    f"collection.action_probabilities[{action_name!r}] must be finite, "
                    f"got {probability_value!r}."
                )
            if probability_value < 0.0:
                raise ValueError(
                    f"collection.action_probabilities[{action_name!r}] must be non-negative, "
                    f"got {probability_value!r}."
                )
            total_probability += float(probability_value)
        if probabilities and total_probability <= 0.0:
            raise ValueError("collection.action_probabilities must sum to a positive value.")
        return probabilities


@dataclass(config=PYDANTIC_CONFIG)
class MultiWorldDatasetRecord:
    dataset_id: str = ""
    split_id: str = ""

    @model_validator(mode="after")
    def validate_ids_present(self) -> MultiWorldDatasetRecord:
        missing = [
            name
            for name, value in (("dataset_id", self.dataset_id), ("split_id", self.split_id))
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(
                f"dataset.extra_worlds entries need non-empty {' and '.join(missing)}."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class DatasetReferenceConfig:
    artifact_id: str = ""
    artifact_type: Literal["raw_dataset", "encoded_dataset"] = "encoded_dataset"
    extra_worlds: list[MultiWorldDatasetRecord] = field(default_factory=list)
    canonicality_policy: Literal[
        "rgb_canonical",
        "latent_canonical",
        "action_canonical",
        "hybrid_debug",
    ] = "rgb_canonical"
    keep_rgb: bool = False
    keep_latent: bool = True


@dataclass(config=PYDANTIC_CONFIG)
class SplitPolicyConfig:
    artifact_id: str = ""
    strategy: Literal[
        "episode_random",
        "episode_random_stratified_by_environment",
        "chronological",
        "manual_ids",
    ] = "episode_random"
    seed: int = 42
    train_fraction: float = Field(default=0.6, ge=0.0, le=1.0)
    validation_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    test_fraction: float = Field(default=0.3, ge=0.0, le=1.0)
    train_episode_ids: list[int] = field(default_factory=list)
    validation_episode_ids: list[int] = field(default_factory=list)
    test_episode_ids: list[int] = field(default_factory=list)
    constraints: dict[str, Any] = field(
        default_factory=lambda: {
            "environment_balanced": False,
            "minimum_episodes_per_split": 32,
        }
    )


@dataclass(config=PYDANTIC_CONFIG)
class VisionDatasetRecipe:
    artifact_id: str = ""
    environment: str = ""
    split_id: str = ""

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> VisionDatasetRecipe:
        if bool(self.artifact_id) == bool(self.environment):
            raise ValueError(
                "vision.datasets entries need exactly one of artifact_id or environment."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class VisionConfig:
    type: Literal["autoencoder", "beta_vae", "identity"] = "autoencoder"
    artifact_id: str = ""
    latent_dim: int = Field(default=64, ge=1)
    beta: float = Field(default=4.0, ge=0.0)
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    learning_rate: float = Field(default=1e-3, gt=0.0)
    loss_type: Literal["mse", "structure_l1"] = "mse"
    loss_l1_weight: float = Field(default=1.0, ge=0.0)
    loss_ssim_weight: float = Field(default=0.5, ge=0.0)
    loss_edge_weight: float = Field(default=0.25, ge=0.0)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=1)
    max_episodes: int = Field(default=16, ge=0)
    max_frames_per_episode: int = Field(default=64, ge=0)
    frame_cache_mode: Literal["auto", "memory", "disk", "none"] = "auto"
    frame_cache_memory_fraction: float = Field(default=0.7, gt=0.0, le=1.0)
    data_loader_num_workers: int = Field(default=0, ge=0)
    data_loader_pin_memory: bool = True
    data_loader_persistent_workers: bool = True
    datasets: list[VisionDatasetRecipe] = field(default_factory=list)


@dataclass(config=PYDANTIC_CONFIG)
class VisualMaskingConfig:
    """Random and periodic visual masking for the non-corruption path."""

    blackout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    blackout_num_segments: int = Field(default=1, ge=1)
    blackout_min_length: int = Field(default=3, ge=1)
    blackout_max_length: int = Field(default=10, ge=1)
    stride: int = Field(default=1, ge=1)
    stride_random_offset: bool = True

    @model_validator(mode="after")
    def validate_blackout_length_range(self) -> VisualMaskingConfig:
        if self.blackout_min_length > self.blackout_max_length:
            raise ValueError(
                "visual_masking.blackout_min_length must not exceed blackout_max_length."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class InputCorruptionConfig:
    """Latent input corruption: noise injection plus blackout blocks."""

    enabled: bool = False
    noise_sigma_rel: float = Field(default=0.0, ge=0.0)
    noise_sigma_abs: float = Field(default=0.5, ge=0.0)
    noise_type: Literal["additive", "multiplicative"] = "additive"
    noise_num_blocks: int = Field(default=1, ge=0)
    noise_min_length: int = Field(default=1, ge=1)
    noise_max_length: int = Field(default=10, ge=1)
    blackout_num_blocks: int = Field(default=1, ge=0)
    blackout_min_length: int = Field(default=1, ge=1)
    blackout_max_length: int = Field(default=0, ge=0)
    blackout_schedule: Literal["random", "periodic"] = "random"
    blackout_period_grounded: int = Field(default=8, ge=1)
    blackout_period_masked: int = Field(default=8, ge=1)
    context_warmup: int = Field(default=2, ge=0)
    use_blackout_token: bool = False
    append_temporal_offset: bool = False

    @model_validator(mode="after")
    def validate_corruption_length_ranges(self) -> InputCorruptionConfig:
        if self.noise_sigma_abs > 0.0 and self.noise_sigma_rel > 0.0:
            raise ValueError(
                "input_corruption.noise_sigma_abs and noise_sigma_rel are mutually exclusive."
            )
        if self.noise_min_length > self.noise_max_length:
            raise ValueError("input_corruption.noise_min_length must not exceed noise_max_length.")
        if self.blackout_max_length > 0 and self.blackout_min_length > self.blackout_max_length:
            raise ValueError(
                "input_corruption.blackout_min_length must not exceed "
                "blackout_max_length when the maximum is bounded."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class RegularizationSiteConfig:
    """Training-time dropout and Gaussian noise applied at one named tensor site."""

    target: str = ""
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    noise_scale: float = Field(default=0.0, ge=0.0)


@dataclass(config=PYDANTIC_CONFIG)
class RegularizationConfig:
    """Training-time tensor perturbations grouped by tensor site."""

    dropout_mode: Literal["independent", "variational"] = "independent"
    sites: list[RegularizationSiteConfig] = field(default_factory=list)


@dataclass(config=PYDANTIC_CONFIG)
class PredictionBootstrapConfig:
    """Optional TD-style bootstrap added to one-step prediction targets."""

    enabled: bool = False
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    loss_type: Literal["mse", "l1"] = "mse"


@dataclass(config=PYDANTIC_CONFIG)
class InverseDynamicsConfig:
    """Auxiliary transition decoder from consecutive encoder codes to kinematics."""

    enabled: bool = False
    hidden_dim: int = Field(default=256, ge=1)
    weight: float = Field(default=1.0, ge=0.0)


@dataclass(config=PYDANTIC_CONFIG)
class OrthonormalityConfig:
    """Single-view code-structure regularizers for encoder and predictor codes."""

    encoder_weight: float = Field(default=0.0, ge=0.0)
    predictor_weight: float = Field(default=0.0, ge=0.0)
    variance_floor: float = Field(default=0.0, ge=0.0)


@dataclass(config=PYDANTIC_CONFIG)
class RolloutConfig:
    """Predictor rollout behavior for self-fed belief updates."""

    detach_intermediate_predictions: bool = False


@dataclass(config=PYDANTIC_CONFIG)
class SpatialModelInputsConfig:
    observation_source: Literal["latent", "rgb", "action"] = "latent"
    encoder_observation_delay_steps: int = Field(default=0, ge=0)
    encoder_context_channels: list[ContextChannel] = field(default_factory=list)
    predictor_context_channels: list[ContextChannel] = field(
        default_factory=lambda: ["action", "step_displacement", "angular_velocity"]
    )
    predictor_input_mode: Literal["encoder", "belief", "dual", "conditional", "gated"] = "encoder"
    kinematics_velocity_scale: float = 1.0
    visual_masking: VisualMaskingConfig = field(default_factory=VisualMaskingConfig)
    input_corruption: InputCorruptionConfig = field(default_factory=InputCorruptionConfig)

    @field_validator("encoder_context_channels", "predictor_context_channels")
    @classmethod
    def _validate_context_channel_names(cls, channels: list[str]) -> list[str]:
        invalid_channels = [channel for channel in channels if not is_context_channel(channel)]
        if invalid_channels:
            raise ValueError(
                f"unknown context channel(s) {invalid_channels!r}; {allowed_context_channel_hint()}"
            )
        return channels

    @model_validator(mode="after")
    def _validate_observation_delay(self) -> SpatialModelInputsConfig:
        if self.observation_source == "action" and self.encoder_observation_delay_steps > 0:
            raise ValueError(
                "encoder_observation_delay_steps is unsupported for observation_source='action': "
                "action observations are already shifted into history tokens."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class TemporalFamilyConfig:
    family: Literal[
        "mlp",
        "rnn",
        "gru",
        "gru_softplus",
        "gru_relu",
        "lstm",
        "lstm_softplus",
        "lstm_relu",
    ] = "gru"
    layer_sizes: list[int] = field(default_factory=lambda: [256])
    stability_timescale: float = Field(default=1.0, gt=0.0)
    stability_statistics_tau: float = Field(default=1000.0, gt=1.0)
    head_activation: Literal["none", "relu", "softplus"] = "none"
    head_weight_sparsity: float = Field(default=1.0, gt=0.0, le=1.0)
    normalize_codes: bool = True
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    num_heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=512, ge=1)
    context_length: int = Field(default=64, ge=1)
    position_encoding: Literal["sinusoidal", "learned", "alibi", "rope"] = "sinusoidal"
    alibi_slope: float = Field(default=0.125, gt=0.0)
    causal: bool = True
    transformer_memory_slots: int = Field(default=0, ge=0)
    backbone_type: Literal["identity", "mlp", "cnn", "vit"] = "identity"
    cnn_type: Literal["simple", "resnet18"] = "simple"
    cnn_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_pretrained: bool = False
    cnn_gradient_checkpointing: bool = False
    cnn_freeze: bool = False
    cnn_pool_size: int = Field(default=1, ge=1)
    vit_patch_size: int = Field(default=8, ge=1)
    vit_depth: int = Field(default=4, ge=1)
    vit_num_heads: int = Field(default=8, ge=1)
    vit_mlp_ratio: float = Field(default=4.0, gt=0.0)
    action_embedding_dim: int = Field(default=32, ge=1)
    slow_leak_alpha: float = Field(default=0.0, ge=0.0, lt=1.0)
    slow_leak_mode: Literal["ema", "union"] = "ema"
    forget_bias: float = Field(default=0.0)
    chrono_init: bool = False
    chrono_t_max: int = Field(default=2, ge=2)
    state_tau: float = Field(default=1.0, ge=1.0)
    adaptive_state_alpha: float | None = Field(default=None, gt=0.0, lt=1.0)
    adaptive_state_gate_initial_open_probability: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
    )
    clockwork_periods: list[int] = field(default_factory=list)
    mtrnn_time_constants: list[float] = field(default_factory=list)
    state_dim: int = Field(default=64, ge=1)
    dt_min: float = Field(default=0.001, gt=0.0)
    dt_max: float = Field(default=0.1, gt=0.0)
    ssm_feedforward: bool = Field(default=True)
    ema_ssm_min_timescale: float = Field(default=4.0, ge=1.0)
    ema_ssm_max_timescale: float = Field(default=512.0, ge=1.0)
    mamba_d_state: int = Field(default=16, ge=1)
    mamba_d_conv: int = Field(default=4, ge=1)
    mamba_expand: int = Field(default=2, ge=1)
    xlstm_num_heads: int = Field(default=8, ge=1)
    xlstm_variant: Literal["simple_mlstm", "simple_slstm", "block_stack", "large_block"] = (
        "simple_mlstm"
    )
    xlstm_block_ratio: tuple[int, int] = (7, 1)
    xlstm_slstm_at: list[int] | None = None
    xlstm_backend: Literal["reimpl", "nxai", "auto"] = "reimpl"
    fla_variant: Literal["gla", "gated_deltanet", "mamba2"] = "gla"
    fla_head_dim: int = Field(default=64, ge=1)
    fla_expand: int = Field(default=2, ge=1)
    fla_state_size: int = Field(default=128, ge=1)
    fla_conv_kernel: int = Field(default=4, ge=1)
    readout_cell_size: int = Field(default=0, ge=0)
    readout_cell_family: Literal["gru", "lstm"] = "gru"
    chart_memory_heads: int = Field(default=0, ge=0)
    chart_memory_head_dim: int = Field(default=64, ge=1)
    chart_memory_half_life_steps: float = Field(default=2048.0, gt=0.0)
    chart_readout: Literal["residual", "code"] = "residual"
    chart_code_value: Literal["mask", "activation"] = "mask"
    chart_code_checkpoint_chunk: int = Field(default=128, ge=0)
    chart_code_tied_query_init: bool = True
    code_head_frozen: bool = False
    chart_sensory_heads: int = Field(default=0, ge=0)
    chart_sensory_head_dim: int = Field(default=64, ge=1)
    chart_sensory_half_life_steps: float = Field(default=2048.0, gt=0.0)
    chart_sensory_checkpoint_chunk: int = Field(default=128, ge=0)
    chart_transition_heads: int = Field(default=0, ge=0)
    chart_transition_head_dim: int = Field(default=64, ge=1)
    chart_transition_half_life_steps: float = Field(default=2048.0, gt=0.0)
    chart_transition_checkpoint_chunk: int = Field(default=128, ge=0)

    @property
    def input_size(self) -> int:
        if not self.layer_sizes:
            raise ValueError("TemporalFamilyConfig.layer_sizes must contain at least one width.")
        return int(self.layer_sizes[0])

    @property
    def mixer_output_size(self) -> int:
        """Width of the temporal mixer's own last layer, before any readout cell."""
        if not self.layer_sizes:
            raise ValueError("TemporalFamilyConfig.layer_sizes must contain at least one width.")
        return int(self.layer_sizes[-1])

    @property
    def output_size(self) -> int:
        """Width presented to the code head: the readout cell when enabled, else the mixer."""
        if self.readout_cell_size > 0:
            return int(self.readout_cell_size)
        return self.mixer_output_size

    @model_validator(mode="after")
    def _validate_stability_family(self) -> TemporalFamilyConfig:
        if self.family not in {"wyss", "leaky_hierarchy"} and (
            self.stability_timescale != 1.0 or self.stability_statistics_tau != 1000.0
        ):
            raise ValueError("stability_* knobs require wyss or leaky_hierarchy.")
        if self.family in {"wyss", "leaky_hierarchy"} and (
            self.dropout != 0 or self.readout_cell_size > 0 or self.chart_memory_heads > 0
        ):
            raise ValueError("Local-memory hierarchies require zero dropout and no extra memory.")
        return self

    @model_validator(mode="after")
    def _validate_ssm_dt_range(self) -> TemporalFamilyConfig:
        if self.family == "ssm" and self.dt_min >= self.dt_max:
            raise ValueError("ssm dt_min must be less than dt_max.")
        return self

    @model_validator(mode="after")
    def _validate_ssm_knobs_family(self) -> TemporalFamilyConfig:
        if self.family != "ssm" and (
            self.state_dim != 64
            or self.dt_min != 0.001
            or self.dt_max != 0.1
            or self.ssm_feedforward is not True
        ):
            raise ValueError(
                "state_dim / dt_min / dt_max / ssm_feedforward are only supported when "
                f"family='ssm'; got family={self.family!r}. Leave them at their defaults "
                "(64 / 0.001 / 0.1 / true)."
            )
        return self

    @model_validator(mode="after")
    def _validate_ema_ssm_timescales(self) -> TemporalFamilyConfig:
        if self.ema_ssm_min_timescale > self.ema_ssm_max_timescale:
            raise ValueError("ema_ssm_min_timescale must not exceed ema_ssm_max_timescale.")
        if self.family != "ema_ssm" and (
            self.ema_ssm_min_timescale != 4.0 or self.ema_ssm_max_timescale != 512.0
        ):
            raise ValueError(
                "ema_ssm_min_timescale / ema_ssm_max_timescale are only supported when "
                f"family='ema_ssm'; got family={self.family!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_mamba_knobs_family(self) -> TemporalFamilyConfig:
        if self.family != "mamba" and (
            self.mamba_d_state != 16 or self.mamba_d_conv != 4 or self.mamba_expand != 2
        ):
            raise ValueError(
                "mamba_d_state / mamba_d_conv / mamba_expand are only supported when "
                f"family='mamba'; got family={self.family!r}. Leave them at their defaults "
                "(16 / 4 / 2)."
            )
        return self

    @model_validator(mode="after")
    def _validate_transformer_memory_family(self) -> TemporalFamilyConfig:
        if self.family != "transformer" and self.transformer_memory_slots != 0:
            raise ValueError(
                "transformer_memory_slots is only supported when family='transformer'; "
                f"got family={self.family!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_fla_knobs_family(self) -> TemporalFamilyConfig:
        if self.family != "fla" and (
            self.fla_variant != "gla"
            or self.fla_head_dim != 64
            or self.fla_expand != 2
            or self.fla_state_size != 128
            or self.fla_conv_kernel != 4
        ):
            raise ValueError(
                "fla_variant / fla_head_dim / fla_expand / fla_state_size / fla_conv_kernel are "
                f"only supported when family='fla'; got family={self.family!r}. Leave them at "
                "their defaults (gla / 64 / 2 / 128 / 4)."
            )
        return self

    @model_validator(mode="after")
    def _validate_fla_heads(self) -> TemporalFamilyConfig:
        if (
            self.family == "fla"
            and self.fla_variant == "mamba2"
            and (self.fla_expand * self.mixer_output_size) % self.fla_head_dim != 0
        ):
            raise ValueError(
                f"fla layer width ({self.mixer_output_size}) * fla_expand ({self.fla_expand}) "
                f"must be divisible by fla_head_dim ({self.fla_head_dim}) so num_heads is an "
                "integer."
            )
        if (
            self.family == "fla"
            and self.fla_variant == "gated_deltanet"
            and 4 * self.num_heads * self.fla_head_dim != 3 * self.mixer_output_size
        ):
            raise ValueError(
                "fla gated_deltanet requires num_heads * fla_head_dim = "
                f"0.75 * width; got {self.num_heads} * {self.fla_head_dim} for width "
                f"{self.mixer_output_size}."
            )
        return self

    @model_validator(mode="after")
    def _validate_xlstm_knobs_family(self) -> TemporalFamilyConfig:
        if self.family == "xlstm":
            return self
        xlstm_knobs = {
            "xlstm_num_heads": (self.xlstm_num_heads, 8),
            "xlstm_variant": (self.xlstm_variant, "simple_mlstm"),
            "xlstm_block_ratio": (self.xlstm_block_ratio, (7, 1)),
            "xlstm_slstm_at": (self.xlstm_slstm_at, None),
            "xlstm_backend": (self.xlstm_backend, "reimpl"),
        }
        set_knobs = sorted(
            name for name, (value, default) in xlstm_knobs.items() if value != default
        )
        if set_knobs:
            raise ValueError(
                f"xLSTM knobs {set_knobs} are only supported when family='xlstm'; "
                f"got family={self.family!r}. Leave them at their defaults."
            )
        return self

    @model_validator(mode="after")
    def _validate_xlstm_heads(self) -> TemporalFamilyConfig:
        if self.family == "xlstm" and self.mixer_output_size % self.xlstm_num_heads != 0:
            raise ValueError(
                f"xlstm layer width ({self.mixer_output_size}) must be divisible by "
                f"xlstm_num_heads ({self.xlstm_num_heads})."
            )
        if self.family == "xlstm" and self.xlstm_variant == "block_stack":
            num_blocks = len(self.layer_sizes)
            if self.xlstm_slstm_at is None:
                mlstm_per_period, slstm_per_period = self.xlstm_block_ratio
                period = mlstm_per_period + slstm_per_period
                has_mlstm_block = period > 0 and any(
                    (index % period) < mlstm_per_period for index in range(num_blocks)
                )
            else:
                has_mlstm_block = len(set(self.xlstm_slstm_at)) < num_blocks
            if has_mlstm_block and (2 * self.mixer_output_size) % 4 != 0:
                raise ValueError(
                    f"xlstm block_stack mLSTM inner width ({2 * self.mixer_output_size}) must be "
                    "divisible by qkv block size 4; use an even layer width."
                )
        return self

    @model_validator(mode="after")
    def _validate_xlstm_variant_backend_combo(self) -> TemporalFamilyConfig:
        if self.family != "xlstm":
            return self
        unbuilt = {
            ("simple_mlstm", "nxai"),
            ("simple_slstm", "nxai"),
            ("block_stack", "nxai"),
            ("large_block", "reimpl"),
        }
        if (self.xlstm_variant, self.xlstm_backend) in unbuilt:
            raise ValueError(
                f"xlstm (variant={self.xlstm_variant!r}, backend={self.xlstm_backend!r}) is not a "
                "built combination. Built: (simple_mlstm, reimpl), (simple_slstm, reimpl), "
                "(block_stack, reimpl) and (large_block, nxai) -- the reimpl variants are pure "
                "torch, large_block is nxai-only (official GPU/triton lib). Use "
                "xlstm_backend='auto' to resolve by hardware."
            )
        if self.xlstm_variant != "block_stack":
            if self.xlstm_block_ratio != (7, 1):
                raise ValueError(
                    "xlstm_block_ratio is only supported when "
                    "xlstm_variant='block_stack'; leave it at the default (7, 1) for "
                    f"xlstm_variant={self.xlstm_variant!r}. validate_assignment re-checks the "
                    "whole model on every set, so switch variant and ratio together with "
                    "dataclasses.replace rather than one field at a time."
                )
            if self.xlstm_slstm_at is not None:
                raise ValueError(
                    "xlstm_slstm_at is only supported when xlstm_variant='block_stack'."
                )
            return self
        if self.xlstm_slstm_at is not None:
            if len(set(self.xlstm_slstm_at)) != len(self.xlstm_slstm_at):
                raise ValueError("xlstm_slstm_at must contain unique block indices.")
            invalid_indices = [
                index
                for index in self.xlstm_slstm_at
                if index < 0 or index >= len(self.layer_sizes)
            ]
            if invalid_indices:
                raise ValueError(
                    "xlstm_slstm_at indices must be within the configured block stack; "
                    f"got {invalid_indices!r} for {len(self.layer_sizes)} block(s)."
                )
            return self
        mlstm_per_period, slstm_per_period = self.xlstm_block_ratio
        if mlstm_per_period + slstm_per_period == 0:
            raise ValueError("xlstm_block_ratio (0, 0) selects no blocks.")
        num_blocks = len(self.layer_sizes)
        period = mlstm_per_period + slstm_per_period
        realized_slstm = sum(
            1 for index in range(num_blocks) if (index % period) >= mlstm_per_period
        )
        if slstm_per_period > 0 and realized_slstm == 0:
            raise ValueError(
                f"xlstm_block_ratio {tuple(self.xlstm_block_ratio)} over {num_blocks} block(s) "
                f"yields no sLSTM blocks (period {period} > depth), so the stack would be "
                "pure-mLSTM -- i.e. no memory mixing -- while claiming this ratio. Use a ratio "
                f"whose period fits the depth (e.g. (1, 1)), add blocks (>= {period} needed), or "
                "set the ratio to (1, 0) if mLSTM-only is what you want."
            )
        return self

    @model_validator(mode="after")
    def _validate_forget_bias_family(self) -> TemporalFamilyConfig:
        recurrent_families = {
            "gru",
            "gru_softplus",
            "gru_relu",
            "lstm",
            "lstm_softplus",
            "lstm_relu",
        }
        if self.forget_bias != 0.0 and self.family not in recurrent_families:
            raise ValueError(
                "forget_bias is only supported for gated RNN families "
                f"{sorted(recurrent_families)} (LSTM forget gate / GRU update gate); "
                f"got family={self.family!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_chrono_init(self) -> TemporalFamilyConfig:
        lstm_families = {"lstm", "lstm_softplus", "lstm_relu"}
        if self.chrono_init and self.family not in lstm_families:
            raise ValueError(
                "chrono_init is only supported for LSTM families "
                f"{sorted(lstm_families)} (Tallec & Ollivier 2018); got family={self.family!r}."
            )
        if self.chrono_t_max != 2 and not self.chrono_init:
            raise ValueError(
                "chrono_t_max is only meaningful when chrono_init=True; leave it at the inert "
                f"sentinel 2, got chrono_t_max={self.chrono_t_max}."
            )
        if self.chrono_init and self.forget_bias != 0.0:
            raise ValueError(
                "chrono_init and forget_bias are mutually exclusive retention-gate inits "
                f"(both write the forget-gate bias slice); got forget_bias={self.forget_bias}."
            )
        return self

    @model_validator(mode="after")
    def _validate_state_tau_family(self) -> TemporalFamilyConfig:
        gru_families = {"gru", "gru_softplus", "gru_relu"}
        if self.state_tau != 1.0 and self.family not in gru_families:
            raise ValueError(
                "state_tau>1 (leaky-integrator recurrent state) is implemented for the GRU "
                f"families {sorted(gru_families)} only; got family={self.family!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_adaptive_state_gate(self) -> TemporalFamilyConfig:
        supported_families = {
            "gru",
            "gru_softplus",
            "gru_relu",
            "lstm",
            "lstm_softplus",
            "lstm_relu",
        }
        if self.adaptive_state_alpha is not None and self.family not in supported_families:
            raise ValueError(
                "adaptive_state_alpha is only supported for GRU/LSTM families "
                f"{sorted(supported_families)}; got family={self.family!r}."
            )
        if (
            self.adaptive_state_alpha is None
            and self.adaptive_state_gate_initial_open_probability != 0.01
        ):
            raise ValueError(
                "adaptive_state_gate_initial_open_probability is only meaningful when "
                "adaptive_state_alpha is set; leave it at 0.01 when the adaptive gate is off."
            )
        if self.adaptive_state_alpha is not None and self.state_tau != 1.0:
            raise ValueError(
                "adaptive_state_alpha and state_tau>1 are mutually exclusive recurrent-state "
                "update rules; leave state_tau=1 when the adaptive gate is enabled."
            )
        return self

    @model_validator(mode="after")
    def _validate_readout_cell_knobs(self) -> TemporalFamilyConfig:
        if self.readout_cell_size == 0 and self.readout_cell_family != "gru":
            raise ValueError(
                "readout_cell_family is only meaningful when readout_cell_size > 0; leave it at "
                f"'gru' while the readout cell is off, got {self.readout_cell_family!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_chart_memory_knobs(self) -> TemporalFamilyConfig:
        if self.chart_memory_heads == 0 and (
            self.chart_memory_head_dim != 64 or self.chart_memory_half_life_steps != 2048.0
        ):
            raise ValueError(
                "chart_memory_head_dim / chart_memory_half_life_steps are only used when "
                "chart_memory_heads > 0; remove them or enable the chart."
            )
        if self.chart_memory_heads > 0 and self.readout_cell_size > 0:
            raise ValueError(
                "chart_memory_heads and readout_cell_size are both mixer wrappers; pick one."
            )
        if self.chart_readout != "residual" and self.chart_memory_heads == 0:
            raise ValueError(
                f"chart_readout={self.chart_readout!r} needs chart_memory_heads > 0; there is no "
                "memory to read."
            )
        if self.code_head_frozen and self.chart_readout != "code":
            raise ValueError(
                "code_head_frozen freezes hidden -> code at its random init, so the map needs a "
                "code-memory binder to live in: set chart_readout='code' with "
                "chart_memory_heads > 0, or unfreeze the head. Frozen without a binder there is "
                "no trainable hidden-to-code mapping at all."
            )
        code_only_defaults = (
            self.chart_code_value != "mask"
            or self.chart_code_checkpoint_chunk != 128
            or not self.chart_code_tied_query_init
        )
        if code_only_defaults and self.chart_readout != "code":
            raise ValueError(
                "chart_code_value / chart_code_checkpoint_chunk / chart_code_tied_query_init are "
                "only used when chart_readout='code'; leave them at their defaults."
            )
        if self.chart_sensory_heads == 0 and (
            self.chart_sensory_head_dim != 64
            or self.chart_sensory_half_life_steps != 2048.0
            or self.chart_sensory_checkpoint_chunk != 128
        ):
            raise ValueError("chart_sensory_* knobs need chart_sensory_heads > 0.")
        if self.chart_transition_heads == 0 and (
            self.chart_transition_head_dim != 64
            or self.chart_transition_half_life_steps != 2048.0
            or self.chart_transition_checkpoint_chunk != 128
        ):
            raise ValueError("chart_transition_* knobs need chart_transition_heads > 0.")
        return self

    @model_validator(mode="after")
    def _validate_clockwork_periods(self) -> TemporalFamilyConfig:
        if self.family == "clockwork":
            if not self.clockwork_periods:
                raise ValueError("clockwork_periods must be non-empty when family='clockwork'.")
            if any(period < 1 for period in self.clockwork_periods):
                raise ValueError(
                    "every clockwork_periods entry must be a positive int (>= 1); "
                    f"got {self.clockwork_periods}."
                )
            if self.clockwork_periods != sorted(self.clockwork_periods):
                raise ValueError(
                    "clockwork_periods must be ordered fastest-to-slowest (ascending); "
                    f"got {self.clockwork_periods}."
                )
        elif self.clockwork_periods:
            raise ValueError(
                "clockwork_periods is only supported when family='clockwork'; "
                f"got family={self.family!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_mtrnn_time_constants(self) -> TemporalFamilyConfig:
        if self.family == "mtrnn":
            if not self.mtrnn_time_constants:
                raise ValueError("mtrnn_time_constants must be non-empty when family='mtrnn'.")
            if any(tau < 1.0 for tau in self.mtrnn_time_constants):
                raise ValueError(
                    "every mtrnn_time_constants entry must be >= 1.0 (tau<1 diverges); "
                    f"got {self.mtrnn_time_constants}."
                )
        elif self.mtrnn_time_constants:
            raise ValueError(
                "mtrnn_time_constants is only supported when family='mtrnn'; "
                f"got family={self.family!r}."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class SparsifierConfig:
    type: Literal[
        "none",
        "kwinners",
        "grouped_kwinners",
        "sparsemax",
        "soft_wta",
        "entmax",
        "lateral_inhibition",
    ] = "sparsemax"
    temperature: float = Field(default=1.0, gt=0.0)
    k_fraction: float = Field(default=0.06, gt=0.0, le=1.0)
    kwinners_boost_strength: float = Field(default=0.0, ge=0.0)
    kwinners_boost_update_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    kwinners_boost_anneal_steps: int = Field(default=0, ge=0)
    kwinners_balance_bias_rate: float = Field(default=0.0, ge=0.0)
    kwinners_balance_bias_strategy: Literal["age_liveness", "load_sign"] = "age_liveness"
    kwinners_balance_liveness_rate_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
    kwinners_balance_liveness_patience_epochs: float = Field(default=2.0, ge=1.0, le=3.0)
    kwinners_balance_bias_clamp: float = Field(default=0.0, ge=0.0)
    kwinners_balance_bias_leak: float = Field(default=0.0, ge=0.0, lt=1.0)
    kwinners_selection_noise_scale: float = Field(default=0.0, ge=0.0)
    kwinners_selection_noise_anneal_steps: int = Field(default=0, ge=0)
    kwinners_selection_noise_final_scale: float = Field(default=0.0, ge=0.0)
    kwinners_k_anneal_start: int = Field(default=0, ge=0)
    kwinners_k_anneal_steps: int = Field(default=0, ge=0)
    kwinners_binarize: bool = False
    grouped_num_groups: int = Field(default=1, ge=1)
    grouped_group_bias_rate: float = Field(default=0.0, ge=0.0)
    lateral_inhibition_strength: float = Field(default=0.5, ge=0.0)
    lateral_inhibition_rectify: bool = True
    entmax_alpha: float = Field(default=1.5, gt=1.0)
    soft_wta_beta: float = Field(default=1.0, gt=0.0)
    soft_wta_learnable_beta: bool = False
    soft_wta_use_homeostatic: bool = False
    soft_wta_target_sparsity: float = Field(default=0.1, ge=0.0, le=1.0)
    soft_wta_homeostatic_rate: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _validate_kwinners_knobs(self) -> SparsifierConfig:
        if (
            self.type == "kwinners"
            and self.kwinners_balance_bias_strategy != "age_liveness"
            and self.kwinners_balance_bias_rate == 0.0
        ):
            raise ValueError(
                "kwinners_balance_bias_strategy requires a positive kwinners_balance_bias_rate."
            )
        if (
            self.kwinners_selection_noise_anneal_steps > 0
            and self.kwinners_selection_noise_final_scale > self.kwinners_selection_noise_scale
        ):
            raise ValueError(
                "kwinners_selection_noise_final_scale must not exceed "
                "kwinners_selection_noise_scale when annealing is active."
            )
        if self.kwinners_boost_strength > 0.0 and self.kwinners_balance_bias_rate > 0.0:
            raise ValueError(
                "kwinners_boost_strength and kwinners_balance_bias_rate cannot both be enabled."
            )
        k_anneal_configured = self.kwinners_k_anneal_start > 0 or self.kwinners_k_anneal_steps > 0
        if (self.kwinners_k_anneal_start > 0) != (self.kwinners_k_anneal_steps > 0):
            raise ValueError(
                "kwinners_k_anneal_start and kwinners_k_anneal_steps must both be positive or "
                "both be zero."
            )
        noise_configured = (
            self.kwinners_selection_noise_scale > 0.0
            or self.kwinners_selection_noise_anneal_steps > 0
            or self.kwinners_selection_noise_final_scale > 0.0
        )
        if k_anneal_configured and (
            self.kwinners_boost_strength > 0.0
            or self.kwinners_balance_bias_rate > 0.0
            or noise_configured
        ):
            raise ValueError(
                "k-Winners k annealing cannot be combined with boost, balance bias, or "
                "selection noise."
            )
        if self.type == "grouped_kwinners" and self.grouped_num_groups < 2:
            raise ValueError(
                "grouped_kwinners needs grouped_num_groups >= 2; with one group it is plain "
                "top-1. Use type='kwinners' for ungrouped competition."
            )
        if self.type != "grouped_kwinners" and self.grouped_num_groups != 1:
            raise ValueError(
                "grouped_num_groups is only used by the 'grouped_kwinners' sparsifier; "
                f"got type={self.type!r}."
            )
        if self.type == "kwinners":
            return self
        kwinners_knobs = {
            "kwinners_boost_strength": (self.kwinners_boost_strength, 0.0),
            "kwinners_boost_update_rate": (self.kwinners_boost_update_rate, 0.01),
            "kwinners_boost_anneal_steps": (self.kwinners_boost_anneal_steps, 0),
            "kwinners_balance_bias_rate": (self.kwinners_balance_bias_rate, 0.0),
            "kwinners_balance_bias_strategy": (
                self.kwinners_balance_bias_strategy,
                "age_liveness",
            ),
            "kwinners_balance_liveness_rate_ratio": (
                self.kwinners_balance_liveness_rate_ratio,
                0.1,
            ),
            "kwinners_balance_liveness_patience_epochs": (
                self.kwinners_balance_liveness_patience_epochs,
                2.0,
            ),
            "kwinners_selection_noise_scale": (
                self.kwinners_selection_noise_scale,
                0.0,
            ),
            "kwinners_selection_noise_anneal_steps": (
                self.kwinners_selection_noise_anneal_steps,
                0,
            ),
            "kwinners_selection_noise_final_scale": (
                self.kwinners_selection_noise_final_scale,
                0.0,
            ),
            "kwinners_k_anneal_start": (self.kwinners_k_anneal_start, 0),
            "kwinners_k_anneal_steps": (self.kwinners_k_anneal_steps, 0),
            "kwinners_binarize": (self.kwinners_binarize, False),
        }
        set_knobs = sorted(
            name for name, (value, default) in kwinners_knobs.items() if value != default
        )
        if set_knobs:
            raise ValueError(
                f"kwinners_* knobs {set_knobs} are only used by the 'kwinners' sparsifier; "
                f"got type={self.type!r}. Remove them or set type='kwinners'."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class CodeBlockConfig:
    name: str = ""
    start: int = Field(default=0, ge=0)
    end: int = Field(default=0, ge=0)
    sparsifier: SparsifierConfig = field(default_factory=lambda: SparsifierConfig(type="none"))
    slow_leak_alpha: float | None = Field(default=None, ge=0.0, lt=1.0)


@dataclass(config=PYDANTIC_CONFIG)
class MaskedPredictorConfig:
    enabled: bool = False
    layer_sizes: list[int] = field(default_factory=lambda: [128, 128, 128, 128])
    head_activation: Literal["none", "relu", "softplus"] = "none"
    normalize_codes: bool = True
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    num_heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=256, ge=1)
    context_length: int = Field(default=256, ge=1)
    position_encoding: Literal["sinusoidal", "learned", "rope"] = "learned"
    sparsifier: SparsifierConfig = field(default_factory=lambda: SparsifierConfig(type="none"))

    @property
    def output_size(self) -> int:
        if not self.layer_sizes:
            raise ValueError("MaskedPredictorConfig.layer_sizes must contain at least one width.")
        return int(self.layer_sizes[-1])


@dataclass(config=PYDANTIC_CONFIG)
class TeacherStudentConfig:
    mode: Literal[
        "none",
        "ema_byol",
        "jepa_temporal",
        "jepa_noise_blackout",
        "jepa_masked",
    ] = "ema_byol"
    ema_decay: float = Field(default=0.995, ge=0.0, le=1.0)
    ema_schedule: Literal["constant", "cosine"] = "constant"
    ema_decay_end: float = Field(default=1.0, ge=0.0, le=1.0)
    teacher_sees_unmasked: bool = False
    predictor_ema: bool = False
    predictor_ema_decay: float | None = Field(default=None, ge=0.0, le=1.0)
    predictor_ema_pullback: float = Field(default=0.0, ge=0.0, le=1.0)
    slow_operator_side: Literal["target", "online"] = "target"

    @model_validator(mode="after")
    def _validate_predictor_ema(self) -> TeacherStudentConfig:
        if self.slow_operator_side == "online" and not self.predictor_ema:
            raise ValueError(
                "teacher_student.slow_operator_side='online' requires predictor_ema=true; without "
                "the EMA predictor there is no slow operator to put on the online side."
            )
        if self.predictor_ema and self.mode == "none":
            raise ValueError(
                "teacher_student.predictor_ema=true requires a teacher mode: the slow operator is "
                "applied to teacher.place_codes, which mode='none' does not produce."
            )
        if self.predictor_ema_pullback > 0.0 and not self.predictor_ema:
            raise ValueError(
                "teacher_student.predictor_ema_pullback requires predictor_ema=true; without the "
                "slow copy there is nothing to pull the online predictor toward."
            )
        if self.predictor_ema_decay is not None and not self.predictor_ema:
            raise ValueError(
                "teacher_student.predictor_ema_decay is set but predictor_ema is false, so the "
                "predictor teacher would never be built and the decay silently ignored."
            )
        return self

    @property
    def resolved_predictor_ema_decay(self) -> float:
        """Decay for the predictor teacher; inherits ema_decay when left unset."""
        return self.ema_decay if self.predictor_ema_decay is None else self.predictor_ema_decay


@dataclass(config=PYDANTIC_CONFIG)
class AuxiliaryHeadConfig:
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    head_hidden_dim: int = Field(default=256, ge=1)
    aggregation: str = "gru"
    target_module: str = "predictor"
    orthogonal_init: bool = False
    identity_projection: bool = False
    detach_source: bool = False


_OBJECTIVE_FIELD_RULES: dict[str, tuple[Any, frozenset[str]]] = {
    "loss_type": (
        "cosine",
        frozenset(
            {
                "blackout_alignment",
                "action_conditioned_encoder_alignment",
                "conformal_isometry",
                "grid_prediction",
                "masked_prediction_alignment",
                "multistep_rollout",
                "prediction_alignment",
                "successor_representation",
            }
        ),
    ),
    "support_rank_weight": (0.1, frozenset({"prediction_alignment"})),
    "target_offset": (1, frozenset({"prediction_alignment"})),
    "variance_weight": (0.5, frozenset({"vicreg"})),
    "covariance_weight": (0.5, frozenset({"vicreg"})),
    "minimum_std": (0.8, frozenset({"vicreg"})),
    "covariance_normalization": ("feature_count", frozenset({"vicreg"})),
    "routing_mask_source": ("", frozenset({"vicreg"})),
    "routing_mask_index": (0, frozenset({"vicreg"})),
    "step_displacement_scale": (3.0, frozenset({"slowness"})),
    "timescale": (0, frozenset({"timescale_alignment"})),
    "event_gated": (False, frozenset({"timescale_alignment"})),
    "equivariant_dim": (32, frozenset({"action_equivariance"})),
    "horizon": (8, frozenset({"multistep_rollout", "successor_features", "temporal_stability"})),
    "decorrelation_weight": (0.0, frozenset({"temporal_stability", "normalized_slowness"})),
    "slowness_lags": ([1], frozenset({"normalized_slowness"})),
    "whiten": (False, frozenset({"normalized_slowness"})),
    "detach_variance_denominator": (True, frozenset({"temporal_stability"})),
    "rollout_horizons": ([], frozenset({"multistep_rollout"})),
    "normalize_successor_scale": (
        None,
        frozenset({"successor_representation", "successor_features"}),
    ),
    "constrain_successor_head": (False, frozenset({"successor_representation"})),
    "max_rollout_start_steps": (32, frozenset({"multistep_rollout"})),
    "step_decay": (0.9, frozenset({"multistep_rollout"})),
    "horizon_discount_gamma": (None, frozenset({"multistep_rollout"})),
    "rollout_state_mode": ("replay", frozenset({"multistep_rollout"})),
    "temperature": (0.1, frozenset({"cpc_multi_horizon", "infonce"})),
    "horizons": ([1, 2, 4, 8], frozenset({"cpc_multi_horizon"})),
    "anchor_stride": (1, frozenset({"cpc_multi_horizon", "vicreg"})),
    "anchor_offset": (0, frozenset({"cpc_multi_horizon", "vicreg"})),
    "positive_window": (1, frozenset({"cpc_multi_horizon"})),
    "aggregation": ("mean", frozenset()),
    "head_dim": (
        128,
        frozenset({"action_conditioned_encoder_alignment", "cpc_multi_horizon", "infonce"}),
    ),
    "discount_gamma": (
        0.99,
        frozenset({"successor_alignment", "successor_features", "successor_representation"}),
    ),
    "target_rate": (0.125, frozenset({"boundary_rate"})),
    "binarity_weight": (0.01, frozenset({"boundary_rate"})),
    "bootstrap_source": (
        "",
        frozenset({"successor_features", "successor_representation"}),
    ),
    "prediction_decay_tau": (0.0, frozenset({"grid_prediction"})),
    "auxiliary_head": (
        None,
        frozenset(
            {
                "cpc_multi_horizon",
                "infonce",
                "latent_reconstruction",
                "successor_features",
                "successor_representation",
            }
        ),
    ),
}


L1_SEMANTICS_VERSION = 2
L1_OBJECTIVE_TYPES = ("l1_sparsity", "l1_capacity")


@dataclass(config=PYDANTIC_CONFIG)
class ObjectiveConfig:
    type: str = ""
    weight: float = Field(default=1.0, ge=0.0)
    targets: list[str] = field(default_factory=list)
    loss_type: str = "cosine"
    target_offset: Literal[0, 1] = 1
    timescale: int = Field(default=0, ge=0)
    event_gated: bool = False
    equivariant_dim: int = Field(default=32, ge=1)
    normalize_successor_scale: bool | None = None
    constrain_successor_head: bool = False
    support_rank_weight: float = Field(default=0.1, ge=0.0)
    variance_weight: float = Field(default=0.5, ge=0.0)
    covariance_weight: float = Field(default=0.5, ge=0.0)
    minimum_std: float = Field(default=0.8, ge=0.0)
    covariance_normalization: Literal["feature_count", "feature_count_squared"] = "feature_count"
    routing_mask_source: str = ""
    routing_mask_index: int = Field(default=0, ge=0)
    step_displacement_scale: float = Field(default=3.0, ge=0.0)
    decorrelation_weight: float = Field(default=0.0, ge=0.0)
    slowness_lags: list[int] = field(default_factory=lambda: [1])
    whiten: bool = False
    detach_variance_denominator: bool = True
    horizon: int = Field(default=8, ge=1)
    rollout_horizons: list[int] = field(default_factory=list)
    max_rollout_start_steps: int = Field(default=32, ge=1)
    step_decay: float = Field(default=0.9, ge=0.0)
    horizon_discount_gamma: float | None = Field(default=None, gt=0.0, le=1.0)
    rollout_state_mode: Literal["replay", "reset"] = "replay"
    temperature: float = Field(default=0.1, gt=0.0)
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    anchor_stride: int = Field(default=1, ge=1)
    anchor_offset: int = Field(default=0, ge=0)
    positive_window: int = Field(default=1, ge=1)
    aggregation: str = "mean"
    head_dim: int = Field(default=128, ge=1)
    discount_gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    target_rate: float = Field(default=0.125, gt=0.0, lt=1.0)
    binarity_weight: float = Field(default=0.01, ge=0.0)
    bootstrap_source: str = ""
    prediction_decay_tau: float = Field(default=0.0, ge=0.0)
    auxiliary_head: AuxiliaryHeadConfig | None = None
    semantics_version: int = Field(default=L1_SEMANTICS_VERSION, ge=1)

    @model_validator(mode="after")
    def validate_temporal_sampling(self) -> ObjectiveConfig:
        if self.type == "normalized_slowness":
            if not self.targets:
                raise ValueError("normalized_slowness requires explicit targets.")
            if (
                not self.slowness_lags
                or min(self.slowness_lags) < 1
                or len(set(self.slowness_lags)) != len(self.slowness_lags)
            ):
                raise ValueError("slowness_lags must contain distinct positive offsets.")
            if self.whiten and self.decorrelation_weight:
                raise ValueError(
                    "Whitening already constrains covariance; omit decorrelation_weight."
                )
        if self.type in L1_OBJECTIVE_TYPES and self.semantics_version != L1_SEMANTICS_VERSION:
            raise ValueError(
                f"objective type {self.type!r} declares semantics_version "
                f"{self.semantics_version}, but this code computes version {L1_SEMANTICS_VERSION} "
                "(|A| averaged over units as well as steps). A version-1 weight is code_dim times "
                "stronger than the same number means now, so multiply the weight by code_dim (or "
                "divide, going the other way) and set semantics_version explicitly. No silent "
                "rescale."
            )
        if self.anchor_offset >= self.anchor_stride:
            raise ValueError("anchor_offset must be below anchor_stride.")
        if any(horizon < 1 for horizon in self.rollout_horizons):
            raise ValueError("rollout_horizons must contain only positive integers.")
        if len(set(self.rollout_horizons)) != len(self.rollout_horizons):
            raise ValueError("rollout_horizons must not contain duplicates.")
        if (
            self.type == "cpc_multi_horizon"
            and self.horizons
            and self.positive_window > min(self.horizons)
        ):
            raise ValueError(
                "cpc_multi_horizon positive_window must not exceed its shortest horizon."
            )
        return self

    @model_validator(mode="after")
    def validate_objective_specific_fields(self) -> ObjectiveConfig:
        for field_name, (default_value, consumers) in _OBJECTIVE_FIELD_RULES.items():
            if getattr(self, field_name) == default_value or self.type in consumers:
                continue
            raise ValueError(
                f"{field_name} is not used by objective type {self.type!r}; it is only used by "
                f"objective types {sorted(consumers)}. Leave it at its default {default_value!r}."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class RepresentationViewConfig:
    """A named contiguous slice source[..., start:end] of a base representation."""

    name: str = ""
    source: str = ""
    start: int = Field(default=0, ge=0)
    end: int = Field(default=0, ge=0)


@dataclass(config=PYDANTIC_CONFIG)
class SelectionPolicyConfig:
    primary_metric: str = "validation.xy_decode_rmse"
    primary_mode: Literal["min", "max"] = "min"
    tie_break_metric: str = "validation.total_loss"
    tie_break_mode: Literal["min", "max"] = "min"
    save_best_primary: bool = True
    save_best_validation_loss: bool = True
    save_last: bool = True


@dataclass(config=PYDANTIC_CONFIG)
class OnlineTrainingConfig:
    """On-the-fly trajectory generation instead of an offline encoded dataset (JAXenstein only)."""

    enabled: bool = False
    steps_per_epoch: int = Field(default=256, ge=1)
    validation_episodes: int = Field(default=256, ge=0)
    seed: int = Field(default=0, ge=0)
    prefetch_batches: int = Field(default=2, ge=0)


@dataclass(config=PYDANTIC_CONFIG)
class PhaseConfig:
    name: str
    epochs: int = Field(ge=1)
    train: list[str] = field(default_factory=list)


@dataclass(config=PYDANTIC_CONFIG)
class ReplayConfig:
    enabled: bool = False
    source_encoded_dataset_artifact_id: str | None = None
    source_split_artifact_id: str | None = None
    clip_length: int = Field(default=64, ge=1)
    replay_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass(config=PYDANTIC_CONFIG)
class SpatialTrainingConfig:
    learning_rate: float = Field(default=3e-4, gt=0.0)
    group_learning_rates: dict[str, float] = field(default_factory=dict)
    checkpoint_every_n_epochs: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    exclude_bias_and_norm_from_weight_decay: bool = True
    gradient_clip_norm: float = Field(default=1.0, ge=0.0)
    gradient_clip_mode: Literal["global", "component"] = "global"
    predictor_update_interval: int = Field(default=1, ge=1)
    epochs: int = Field(default=32, ge=1)
    code_dim: int = Field(default=256, ge=1)
    batch_size: int = Field(default=32, ge=1)
    shuffle_episodes: bool = True
    bptt_window: int = Field(default=0, ge=0)
    validation_batch_size: int = Field(default=0, ge=0)
    num_workers: int = Field(default=0, ge=0)
    train_episode_ids: list[int] | None = None
    max_validation_episodes: int = Field(default=512, ge=0)
    optimizer: Literal["adamw", "adam", "rmsprop", "sgd"] = "adamw"
    rmsprop_alpha: float = Field(default=0.99, gt=0.0, lt=1.0)
    rmsprop_momentum: float = Field(default=0.0, ge=0.0, lt=1.0)
    rmsprop_eps: float = Field(default=1e-8, gt=0.0)
    lr_schedule: Literal["constant", "cosine"] = "constant"
    lr_warmup_epochs: int = Field(default=0, ge=0)
    allow_tf32: bool = True
    selection: SelectionPolicyConfig = field(default_factory=SelectionPolicyConfig)
    online: OnlineTrainingConfig = field(default_factory=OnlineTrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    phases: list[PhaseConfig] = field(default_factory=list)

    @model_validator(mode="after")
    def validate_phase_epochs_cover_total(self) -> SpatialTrainingConfig:
        if self.phases:
            scheduled_epochs = sum(phase.epochs for phase in self.phases)
            if scheduled_epochs != self.epochs:
                raise ValueError(
                    f"training.phases epochs must sum to training.epochs ({self.epochs}), "
                    f"got {scheduled_epochs} across {len(self.phases)} phase(s)."
                )
        if self.predictor_update_interval > 1:
            predictor_side_names = {"predictor", "sparsifier", "embeddings"}
            for phase in self.phases:
                phase_selectors = set(phase.train)
                required_encoders = {
                    f"{selector.rpartition(':')[0]}:encoder" if ":" in selector else "encoder"
                    for selector in phase_selectors
                    if selector.rpartition(":")[2] in predictor_side_names
                }
                missing_encoders = sorted(required_encoders - phase_selectors)
                if missing_encoders:
                    raise ValueError(
                        f"training phase {phase.name!r} uses predictor_update_interval="
                        f"{self.predictor_update_interval} but is missing paired encoder "
                        f"selector(s) {missing_encoders}. Intervening predictor-frozen steps "
                        "must keep the corresponding encoder trainable."
                    )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class DagNodeConfig:
    """One additional FULL, INDEPENDENT place model in the DAG, fed live by an earlier node."""

    name: str
    input_source: str = "vision_latent"
    detach_input: bool = True
    temporal_stride: int = Field(default=1, ge=1)
    chunk_input: Literal["last", "delta", "last_delta", "last_delta_mean"] = "last"
    chunk_matched_slots: bool = False
    boundary_gate: BoundaryGateConfig = field(default_factory=lambda: BoundaryGateConfig())
    model: SpatialModelConfig = field(default_factory=lambda: SpatialModelConfig())

    @model_validator(mode="after")
    def validate_chunk_input_requires_chunking(self) -> DagNodeConfig:
        if self.temporal_stride == 1 and self.chunk_input != "last":
            raise ValueError("non-last chunk_input requires temporal_stride greater than one.")
        return self


@dataclass(config=PYDANTIC_CONFIG)
class BoundaryGateConfig:
    """Learned causal hold gate on a DAG node's input sequence."""

    enabled: bool = False
    hidden_dim: int = Field(default=32, ge=1)
    initial_rate: float = Field(default=0.125, gt=0.0, lt=1.0)


@dataclass(config=PYDANTIC_CONFIG)
class CompetenceRouterConfig:
    """Task-free routing over DAG experts by self-supervised prediction error."""

    support_window: int = Field(default=16, ge=1)
    include_root: bool = True
    diversity_coefficient: float = Field(default=0.0, ge=0.0)
    router_mode: Literal[
        "competence", "competence_ema", "competence_relative", "competence_zscore", "random"
    ] = "competence"
    ema_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    baseline_decay: float = Field(default=0.99, gt=0.0, le=1.0)
    error_horizon: Literal["onestep", "successor"] = "onestep"
    successor_gamma: float = Field(default=0.95, ge=0.0, lt=1.0)
    soft_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    soft_floor_temperature: float = Field(default=1.0, gt=0.0)
    soft_floor_anneal_steps: int = Field(default=0, ge=0)
    soft_floor_min: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_soft_floor_range(self) -> CompetenceRouterConfig:
        if self.soft_floor_min > self.soft_floor:
            raise ValueError(
                "soft_floor_min cannot exceed soft_floor; the anneal must not increase routing "
                "softness."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class LateralConsensusConfig:
    enabled: bool = False
    alpha: float = Field(default=0.1, ge=0.0)
    threshold: float = Field(default=0.3, ge=-1.0, le=1.0)
    iters: int = Field(default=1, ge=1, le=5)


@dataclass(config=PYDANTIC_CONFIG)
class ThalamicRouterConfig:
    """Learned sparse recurrent gate over DAG experts."""

    hidden: int = Field(default=64, ge=1)
    top_k: int = Field(default=1, ge=1)
    support_window: int = Field(default=16, ge=1)
    include_root: bool = True
    gate_temperature: float = Field(default=1.0, gt=0.0)


@dataclass(config=PYDANTIC_CONFIG)
class EncoderContextAdaptersConfig:
    """Per-context low-rank adapters on the encoder binding layer (disabled by default)."""

    enabled: bool = False
    rank: int = Field(default=8, ge=1)
    num_contexts: int = Field(default=4, ge=1)
    active_context: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_active_context(self) -> EncoderContextAdaptersConfig:
        if self.active_context >= self.num_contexts:
            raise ValueError(
                f"encoder_context_adapters.active_context={self.active_context} is out of range "
                f"for num_contexts={self.num_contexts}."
            )
        return self


MISMATCH_GATE_DEFAULTS: dict[str, Any] = {
    "loss_metric_key": "loss/prediction_cosine",
    "dead_fraction_metric_key": "vicreg_encoder/dead_dim_fraction",
    "fast_ema_steps": 64,
    "slow_ema_steps": 2048,
    "ratio_on": 1.20,
    "ratio_off": 1.05,
    "min_steps": 200,
}


NON_MODEL_OWNED_SIGNAL_PREFIXES = ("env/", "dataset/")


def signal_is_model_owned(metric_key: str) -> bool:
    """True when the gate's watched signal is a feature the model itself produces."""
    return not metric_key.startswith(NON_MODEL_OWNED_SIGNAL_PREFIXES)


@dataclass(config=PYDANTIC_CONFIG)
class MismatchRegimeConfig:
    """The constant knob values one mismatch-gate regime holds."""

    balance_bias_rate: float | None = Field(default=None, ge=0.0)
    selection_noise_scale: float | None = Field(default=None, ge=0.0)
    group_lr_multipliers: dict[str, float] = field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_multipliers(self) -> MismatchRegimeConfig:
        non_positive = sorted(
            name for name, value in self.group_lr_multipliers.items() if value <= 0.0
        )
        if non_positive:
            raise ValueError(
                f"mismatch_gate group_lr_multipliers {non_positive} must be positive; a zero or "
                "negative multiplier is a freeze or a sign flip, not a regime."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class MismatchGateConfig:
    """Prediction error as a DISCRETE event that switches a recruitment/plasticity regime."""

    enabled: bool = False
    loss_metric_key: str = MISMATCH_GATE_DEFAULTS["loss_metric_key"]
    dead_fraction_metric_key: str = MISMATCH_GATE_DEFAULTS["dead_fraction_metric_key"]
    fast_ema_steps: int = Field(default=MISMATCH_GATE_DEFAULTS["fast_ema_steps"], ge=1)
    slow_ema_steps: int = Field(default=MISMATCH_GATE_DEFAULTS["slow_ema_steps"], ge=1)
    ratio_on: float = Field(default=MISMATCH_GATE_DEFAULTS["ratio_on"], gt=1.0)
    ratio_off: float = Field(default=MISMATCH_GATE_DEFAULTS["ratio_off"], gt=0.0)
    min_steps: int = Field(default=MISMATCH_GATE_DEFAULTS["min_steps"], ge=1)
    stable: MismatchRegimeConfig = field(default_factory=MismatchRegimeConfig)
    novelty: MismatchRegimeConfig = field(default_factory=MismatchRegimeConfig)

    @model_validator(mode="after")
    def _validate_gate(self) -> MismatchGateConfig:
        if self.ratio_off >= self.ratio_on:
            raise ValueError(
                f"mismatch_gate.ratio_off ({self.ratio_off}) must be below ratio_on "
                f"({self.ratio_on}); equal thresholds are a comparator, not hysteresis."
            )
        if self.slow_ema_steps <= self.fast_ema_steps:
            raise ValueError(
                f"mismatch_gate.slow_ema_steps ({self.slow_ema_steps}) must exceed fast_ema_steps "
                f"({self.fast_ema_steps}); otherwise the ratio compares an average with itself."
            )
        if self.enabled:
            noise_regimes = sorted(
                regime_name
                for regime_name in ("stable", "novelty")
                if getattr(self, regime_name).selection_noise_scale is not None
            )
            if noise_regimes and signal_is_model_owned(self.loss_metric_key):
                raise ValueError(
                    f"mismatch_gate regime(s) {noise_regimes} set selection_noise_scale while the "
                    f"gate watches {self.loss_metric_key!r}, a model-owned signal. That closes the "
                    "loop the masterplan forbids: 'Kein Novelty-gated Noise aus modelleigenen "
                    "Features (Closed-Loop-Verbot)'. Drop selection_noise_scale, or watch a "
                    f"signal whose key starts with one of {list(NON_MODEL_OWNED_SIGNAL_PREFIXES)}."
                )
            return self
        set_knobs = [
            name
            for name, default in MISMATCH_GATE_DEFAULTS.items()
            if getattr(self, name) != default
        ]
        set_knobs.extend(
            regime_name
            for regime_name in ("stable", "novelty")
            for regime in [getattr(self, regime_name)]
            if regime.balance_bias_rate is not None
            or regime.selection_noise_scale is not None
            or regime.group_lr_multipliers
        )
        set_knobs.sort()
        if set_knobs:
            raise ValueError(
                f"mismatch_gate knobs {set_knobs} are set while mismatch_gate.enabled=false, so "
                "they would tune nothing. Set enabled=true or remove them."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class CellTypeHeadConfig:
    """One staged cell-type readout over the frozen/slow dense trunk."""

    name: str
    source: str = "encoder.hidden_state"
    code_dim: int = Field(default=512, ge=1)
    active_units: int = Field(default=10, ge=0)
    nonnegative: bool = True
    sparsifier: SparsifierConfig | None = None
    predictor_hidden_dim: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_source(self) -> CellTypeHeadConfig:
        allowed = {"encoder.hidden_state", "predictor.hidden_state"}
        if self.source not in allowed:
            raise ValueError(f"cell_type_heads.source must be one of {sorted(allowed)}.")
        if self.sparsifier is not None and self.active_units != 0:
            raise ValueError(
                "cell_type_heads.active_units=0 is required when sparsifier is configured."
            )
        if self.active_units > self.code_dim:
            raise ValueError(
                f"cell_type_heads.active_units={self.active_units} exceeds "
                f"code_dim={self.code_dim}; a head cannot keep more winners than it has units."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class AttractorBindingConfig:
    """Self-organizing context as an attractor state (disabled by default)."""

    enabled: bool = False
    num_components: int = Field(default=8, ge=1)
    active_components: int = Field(default=1, ge=1)
    persistence: float = Field(default=0.8, ge=0.0, lt=1.0)
    drive_gain: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_active_components(self) -> AttractorBindingConfig:
        if self.active_components > self.num_components:
            raise ValueError(
                f"attractor_binding.active_components={self.active_components} exceeds "
                f"num_components={self.num_components}; the pool cannot elect more winners "
                "than it holds components."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class ComparatorConfig:
    """CA1-style gated comparator head (disabled by default)."""

    enabled: bool = False
    gate_init: float = 2.0
    closed_loop: bool = False
    state_dependent_gate: bool = False

    @model_validator(mode="after")
    def _validate_closed_loop(self) -> ComparatorConfig:
        if self.closed_loop and not self.enabled:
            raise ValueError("comparator.closed_loop requires comparator.enabled=true.")
        if self.state_dependent_gate and not self.enabled:
            raise ValueError("comparator.state_dependent_gate requires comparator.enabled=true.")
        return self


@dataclass(config=PYDANTIC_CONFIG)
class GridStreamConfig:
    """Coordinate-free grid-cell path-integration stream (disabled by default)."""

    enabled: bool = False
    functional: bool = False
    hidden_dim: int = Field(default=1024, ge=1)
    lateral_scale: float = Field(default=0.3, ge=0.0)
    alpha: float = Field(default=0.5, ge=0.0)
    ema_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    knn_k: int = Field(default=16, ge=0)
    velocity_scale: float = Field(default=1.0, gt=0.0)
    velocity_scales: list[float] = field(default_factory=list)
    recurrent_weight_decay: float = Field(default=0.0, ge=0.0)
    place_warmup_epochs: int = Field(default=0, ge=0)
    stabilize_feedback: bool = True
    head_direction_input: bool = False
    teacher_representation: Literal["place_codes", "pre_sparsifier"] = "place_codes"
    velocity_shuffle: bool = False
    correction_interval: int = Field(default=0, ge=0)
    correction_beta: float = Field(default=0.1, ge=0.0, le=1.0)
    feedback_enabled: bool = False
    feedback_gate_init: float = 0.0
    bptt_window: int = Field(default=0, ge=0)
    bptt_curriculum: list[int] = field(default_factory=list)
    anchor_on_clean_steps: bool = False


@dataclass(config=PYDANTIC_CONFIG)
class RecurrentFeedbackConfig:
    """Previous upper state conditions the lower input at every step."""

    enabled: bool = False
    bptt_steps: int = Field(default=64, ge=1)


@dataclass(config=PYDANTIC_CONFIG)
class PredictiveContextConfig:
    """Probabilistic context over a frozen dense encoder; time is measured in input steps."""

    enabled: bool = False
    source: Literal["encoder.hidden_state"] = "encoder.hidden_state"
    hidden_dim: int = Field(default=128, ge=1)
    bptt_steps: int = Field(default=64, ge=1)


@dataclass(config=PYDANTIC_CONFIG)
class DaleMotionConfig:
    """Constraints for the shared excitatory/inhibitory motion circuit."""

    input_driven_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    excitatory_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    leak: float = Field(default=0.5, gt=0.0, le=1.0)
    activity_noise: float = Field(default=0.1, ge=0.0)
    observation_mode: Literal["correction", "direct"] = "correction"
    readout_population: Literal["all", "input_driven"] = "all"


@dataclass(config=PYDANTIC_CONFIG)
class MotionFatigueConfig:
    """Per-episode activity adaptation; decay is measured in observation steps."""

    strength: float = Field(default=0.0, ge=0.0)
    decay: float = Field(default=0.95, ge=0.0, lt=1.0)


@dataclass(config=PYDANTIC_CONFIG)
class AnchoredMotionConfig:
    """Frozen-trunk anchoring and motion-only recurrence between observations."""

    enabled: bool = False
    anchor_source: Literal[
        "encoder.hidden_state",
        "encoder.place_codes",
        "observation.backbone_output",
    ] = "encoder.hidden_state"
    target_source: str = "encoder.place_codes"
    hidden_dim: int = Field(default=128, ge=1)
    input_mode: Literal[
        "actions",
        "relative_odometry",
        "odometry_only",
        "egocentric_odometry",
    ] = "actions"
    dynamics: Literal[
        "gated_sigmoid",
        "relu_rnn",
        "lstm",
        "persistent_lstm",
        "dale_rnn",
        "predict_correct_gru",
    ] = "gated_sigmoid"
    active_units: int = Field(default=0, ge=0)
    dale: DaleMotionConfig | None = None
    fatigue: MotionFatigueConfig | None = None
    rnn_activation: Literal["relu", "softplus"] = "relu"
    rnn_leak: float = Field(default=1.0, gt=0.0, le=1.0)
    normalize_activity: bool = False
    readout_dim: int = Field(default=512, ge=1)
    readout_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    regularized_weights: Literal["none", "recurrent", "decoder"] = "none"
    weight_decay: float = Field(default=0.0, ge=0.0)
    anchor_intervals: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    bptt_steps: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def validate_intervals(self) -> AnchoredMotionConfig:
        if self.active_units and self.dynamics != "predict_correct_gru":
            raise ValueError("motion.active_units applies only to predict_correct_gru.")
        if self.dynamics == "predict_correct_gru":
            if self.hidden_dim < 2:
                raise ValueError("predict_correct_gru requires hidden_dim >= 2 for normalization.")
            if not 1 <= self.active_units <= self.readout_dim:
                raise ValueError("predict_correct_gru requires 1 <= active_units <= readout_dim.")
            if self.input_mode not in {"actions", "relative_odometry", "odometry_only"}:
                raise ValueError("predict_correct_gru supports actions and relative odometry.")
            if self.fatigue is not None or self.regularized_weights != "none":
                raise ValueError("predict_correct_gru does not support fatigue or targeted decay.")
        if self.normalize_activity and self.dynamics not in {"relu_rnn", "dale_rnn"}:
            raise ValueError("Unit-L2 recurrent activity requires ReLU or Dale dynamics.")
        if self.dynamics != "relu_rnn" and (self.rnn_activation != "relu" or self.rnn_leak != 1.0):
            raise ValueError("rnn_activation and rnn_leak apply only to relu_rnn dynamics.")
        if self.dynamics == "gated_sigmoid" and self.input_mode not in {
            "actions",
            "relative_odometry",
        }:
            raise ValueError("gated_sigmoid supports actions or relative_odometry inputs.")
        if self.dynamics in {"relu_rnn", "lstm"} and len(self.anchor_intervals) != 1:
            raise ValueError("Population motion requires one fixed anchor interval.")
        if (self.dynamics == "dale_rnn") != (self.dale is not None):
            raise ValueError("Set motion.dale exactly when dynamics is dale_rnn.")
        if self.dale is not None:
            for fraction in (self.dale.input_driven_fraction, self.dale.excitatory_fraction):
                if not 1 <= int(self.hidden_dim * fraction) < self.hidden_dim:
                    raise ValueError("Dale populations must each contain at least one unit.")
            if self.readout_dropout:
                raise ValueError("Dale motion uses activity_noise, not readout_dropout.")
        if self.regularized_weights != "none" and self.dynamics == "gated_sigmoid":
            raise ValueError("Targeted motion decay requires population dynamics.")
        if (self.regularized_weights == "none") != (self.weight_decay == 0):
            raise ValueError("Set both motion.regularized_weights and positive weight_decay.")
        if not self.anchor_intervals or any(i < 1 for i in self.anchor_intervals):
            raise ValueError("motion.anchor_intervals must contain positive step counts.")
        if self.anchor_intervals != sorted(set(self.anchor_intervals)):
            raise ValueError("motion.anchor_intervals must be increasing and unique.")
        if self.dynamics not in {"persistent_lstm", "dale_rnn"} and (
            self.bptt_steps < max(self.anchor_intervals)
        ):
            raise ValueError("motion.bptt_steps must cover the longest training anchor interval.")
        return self


@dataclass(config=PYDANTIC_CONFIG)
class PathColoringArmConfig:
    """Sparse raw-action transport over a frozen visual place code."""

    enabled: bool = False
    hidden_dim: int = Field(default=512, ge=1)
    active_cells: int = Field(default=16, ge=1)
    action_context_dim: int = Field(default=32, ge=1)
    transport_rank: int = Field(default=16, ge=1)
    anchor_interval: int = Field(default=0, ge=0)
    internal_bptt_steps: int = Field(default=32, ge=0)
    topk_temperature: float = Field(default=0.5, gt=0.0)
    homeostasis_rate: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def validate_active_cells(self) -> PathColoringArmConfig:
        if self.active_cells > self.hidden_dim:
            raise ValueError(
                "path_coloring_arm.active_cells cannot exceed hidden_dim, "
                f"got {self.active_cells} > {self.hidden_dim}."
            )
        if self.transport_rank > self.hidden_dim:
            raise ValueError(
                "path_coloring_arm.transport_rank cannot exceed hidden_dim, "
                f"got {self.transport_rank} > {self.hidden_dim}."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class PathColoringFusionArmConfig:
    """Path coloring corrected by vision after a relative-odometry prediction."""

    enabled: bool = False
    hidden_dim: int = Field(default=512, ge=1)
    active_cells: int = Field(default=16, ge=1)
    action_context_dim: int = Field(default=32, ge=1)
    transport_rank: int = Field(default=16, ge=1)
    internal_bptt_steps: int = Field(default=32, ge=0)
    topk_temperature: float = Field(default=0.5, gt=0.0)
    homeostasis_rate: float = Field(default=0.01, ge=0.0)
    vision_correction_initial: float = Field(default=0.5, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_active_cells(self) -> PathColoringFusionArmConfig:
        if self.active_cells > self.hidden_dim:
            raise ValueError(
                "path_coloring_fusion_arm.active_cells cannot exceed hidden_dim, "
                f"got {self.active_cells} > {self.hidden_dim}."
            )
        if self.transport_rank > self.hidden_dim:
            raise ValueError(
                "path_coloring_fusion_arm.transport_rank cannot exceed hidden_dim, "
                f"got {self.transport_rank} > {self.hidden_dim}."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class WangGridArmConfig:
    """Action-only Wang-inspired shared circuit with recurrent-only candidate cells."""

    enabled: bool = False
    input_driven_dim: int = Field(default=1024, ge=1)
    recurrent_only_dim: int = Field(default=1024, ge=1)
    internal_bptt_steps: int = Field(default=10, ge=0)
    decay_rate: float = Field(default=0.5, gt=0.0, le=1.0)
    noise_level: float = Field(default=1.0, ge=0.0)
    homeostasis_rate: float = Field(default=0.001, ge=0.0, le=1.0)
    excitatory_fraction: float = Field(default=0.8, gt=0.0, le=1.0)


@dataclass(config=PYDANTIC_CONFIG)
class EncoderExpertsConfig:
    """Multiple encoder code heads combined into one code."""

    count: int = Field(default=1, ge=1)
    combiner: Literal["product", "mixture", "gated_product", "attention"] = "product"
    top_m: int | None = Field(default=None, ge=1)
    warmup_steps: int = Field(default=0, ge=0)
    precision_weighted: bool = False
    streams: list[str] = field(default_factory=list)

    @model_validator(mode="after")
    def _validate_precision_weighted(self) -> EncoderExpertsConfig:
        if self.precision_weighted and self.combiner != "product":
            raise ValueError(
                "encoder_experts.precision_weighted requires combiner='product' "
                f"(inverse-variance weighting is a product operation), got {self.combiner!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_streams(self) -> EncoderExpertsConfig:
        if not self.streams:
            return self
        if len(self.streams) != self.count:
            raise ValueError(
                f"encoder_experts.streams must have length count={self.count}, "
                f"got {len(self.streams)}."
            )
        allowed = {"vision", "kinematics"}
        invalid = [stream for stream in self.streams if stream not in allowed]
        if invalid:
            raise ValueError(
                f"encoder_experts.streams entries must be in {allowed}, got {invalid}."
            )
        if self.streams[0] != "vision":
            raise ValueError("encoder_experts.streams[0] must be 'vision' (the base encoder path).")
        return self


@dataclass(config=PYDANTIC_CONFIG)
class PredictorExpertsConfig:
    count: int = Field(default=1, ge=1)
    combiner: Literal["product", "mixture"] = "product"


@dataclass(config=PYDANTIC_CONFIG)
class EnvironmentCodeConfig:
    enabled: bool = False
    z_dim: int = Field(default=16, ge=1)
    inference: Literal["episode", "causal"] = "episode"


@dataclass(config=PYDANTIC_CONFIG)
class SpatialModelConfig:
    architecture: Literal["composite"] = "composite"
    inputs: SpatialModelInputsConfig = field(default_factory=SpatialModelInputsConfig)
    encoder: TemporalFamilyConfig = field(
        default_factory=lambda: TemporalFamilyConfig(
            family="lstm",
            head_activation="none",
        )
    )
    encoder_pre_head_norm: Literal["none", "layernorm", "rmsnorm_fixed"] = "none"
    encoder_post_head_norm: Literal["none", "rmsnorm_fixed", "rmsnorm"] = "none"
    encoder_gradient_scale: float = Field(default=1.0, gt=0.0)
    slow_operator_projection_hidden: int = Field(default=0, ge=0)
    encoder_readout: Literal["mixed", "state"] = "mixed"
    predictor: TemporalFamilyConfig = field(
        default_factory=lambda: TemporalFamilyConfig(
            family="gru",
            head_activation="none",
        )
    )
    predictor_free_partition_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    predictor_residual_dynamics: bool = False
    sparsifier: SparsifierConfig = field(default_factory=SparsifierConfig)
    encoder_experts: EncoderExpertsConfig = field(default_factory=EncoderExpertsConfig)
    predictor_experts: PredictorExpertsConfig = field(default_factory=PredictorExpertsConfig)
    environment_code: EnvironmentCodeConfig = field(default_factory=EnvironmentCodeConfig)
    predictor_sparsifier: SparsifierConfig = field(
        default_factory=lambda: SparsifierConfig(type="none")
    )
    masked_predictor: MaskedPredictorConfig = field(default_factory=MaskedPredictorConfig)
    grid_stream: GridStreamConfig = field(default_factory=GridStreamConfig)
    comparator: ComparatorConfig = field(default_factory=ComparatorConfig)
    encoder_context_adapters: EncoderContextAdaptersConfig = field(
        default_factory=EncoderContextAdaptersConfig
    )
    mismatch_gate: MismatchGateConfig = field(default_factory=MismatchGateConfig)
    attractor_binding: AttractorBindingConfig = field(default_factory=AttractorBindingConfig)
    cell_type_heads: list[CellTypeHeadConfig] = field(default_factory=list)
    motion: AnchoredMotionConfig = field(default_factory=AnchoredMotionConfig)
    predictive_context: PredictiveContextConfig = field(default_factory=PredictiveContextConfig)
    recurrent_feedback: RecurrentFeedbackConfig = field(default_factory=RecurrentFeedbackConfig)
    path_coloring_arm: PathColoringArmConfig = field(default_factory=PathColoringArmConfig)
    path_coloring_fusion_arm: PathColoringFusionArmConfig = field(
        default_factory=PathColoringFusionArmConfig
    )
    wang_grid_arm: WangGridArmConfig = field(default_factory=WangGridArmConfig)
    dag_nodes: list[DagNodeConfig] = field(default_factory=list)
    node_input_uses_teacher: dict[str, bool] = field(default_factory=dict)
    top_down_source: str = ""
    top_down_context_dim: int = 0
    top_down_context_limit: float = 0.0
    top_down_mode: Literal["aligned", "zero", "cross_twin"] = "aligned"
    dag_expert_combiner: Literal["none", "mixture", "attention", "competence", "thalamic"] = "none"
    competence_router: CompetenceRouterConfig = field(default_factory=CompetenceRouterConfig)
    lateral_consensus: LateralConsensusConfig = field(default_factory=LateralConsensusConfig)
    thalamic_router: ThalamicRouterConfig = field(default_factory=ThalamicRouterConfig)
    teacher_student: TeacherStudentConfig = field(default_factory=TeacherStudentConfig)
    prediction_bootstrap: PredictionBootstrapConfig = field(
        default_factory=PredictionBootstrapConfig
    )
    inverse_dynamics: InverseDynamicsConfig = field(default_factory=InverseDynamicsConfig)
    orthonormality: OrthonormalityConfig = field(default_factory=OrthonormalityConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    num_code_blocks: int = Field(default=0, ge=0)
    code_blocks: list[CodeBlockConfig] = field(default_factory=list)
    objectives: dict[str, ObjectiveConfig] = field(default_factory=dict)
    representation_views: list[RepresentationViewConfig] = field(default_factory=list)
    training: SpatialTrainingConfig = field(default_factory=SpatialTrainingConfig)

    def enabled_emergent_grid_arm(self) -> str | None:
        enabled = [
            name
            for name, arm_config in (
                ("path_coloring", self.path_coloring_arm),
                ("path_coloring_fusion", self.path_coloring_fusion_arm),
                ("wang", self.wang_grid_arm),
            )
            if arm_config.enabled
        ]
        return enabled[0] if len(enabled) == 1 else None

    def removed_feature_settings(self) -> list[str]:
        """Names of enabled settings whose implementation is not part of this repository."""
        settings = {
            "spatial_model.dag_nodes": bool(self.dag_nodes),
            "spatial_model.recurrent_feedback.enabled": self.recurrent_feedback.enabled,
            "spatial_model.encoder_experts.combiner": (
                self.encoder_experts.count > 1
                and self.encoder_experts.combiner in {"mixture", "attention", "gated_product"}
            ),
            "spatial_model.predictor_experts.count": self.predictor_experts.count > 1,
            "spatial_model.encoder_context_adapters.enabled": self.encoder_context_adapters.enabled,
            "spatial_model.environment_code.enabled": self.environment_code.enabled,
            "spatial_model.attractor_binding.enabled": self.attractor_binding.enabled,
            "spatial_model.masked_predictor.enabled": self.masked_predictor.enabled,
            "spatial_model.comparator.enabled": self.comparator.enabled,
            "spatial_model.grid_stream.enabled": self.grid_stream.enabled,
            "spatial_model.predictive_context.enabled": self.predictive_context.enabled,
            "spatial_model.motion.enabled": self.motion.enabled,
            "spatial_model.mismatch_gate.enabled": self.mismatch_gate.enabled,
            "spatial_model emergent grid arm": any(
                arm.enabled
                for arm in (
                    self.path_coloring_arm,
                    self.path_coloring_fusion_arm,
                    self.wang_grid_arm,
                )
            ),
            "spatial_model.encoder.readout_cell_size": self.encoder.readout_cell_size > 0,
            "spatial_model.encoder.chart_memory_heads": self.encoder.chart_memory_heads > 0,
            "spatial_model.encoder.chart_readout": self.encoder.chart_readout != "residual",
            "spatial_model.encoder.chart_sensory_heads": self.encoder.chart_sensory_heads > 0,
            "spatial_model.predictor.chart_transition_heads": (
                self.predictor.chart_transition_heads > 0
            ),
            "spatial_model.inputs.observation_source=action": (
                self.inputs.observation_source == "action"
            ),
            "spatial_model.training.online.enabled": self.training.online.enabled,
            "spatial_model.training.replay.enabled": self.training.replay.enabled,
        }
        return sorted(name for name, is_set in settings.items() if is_set)

    @model_validator(mode="after")
    def validate_kwinners_k_anneal_widths(self) -> SpatialModelConfig:
        code_dim = int(self.training.code_dim)
        if self.code_blocks:
            sparsifiers = [
                (block.sparsifier, int(block.end - block.start)) for block in self.code_blocks
            ]
        elif self.num_code_blocks > 1:
            widths = [
                ((index + 1) * code_dim) // self.num_code_blocks
                - (index * code_dim) // self.num_code_blocks
                for index in range(self.num_code_blocks)
            ]
            sparsifiers = [(self.sparsifier, min(widths))]
        else:
            sparsifiers = [(self.sparsifier, code_dim)]

        for sparsifier, num_units in sparsifiers:
            start_k = int(sparsifier.kwinners_k_anneal_start)
            if start_k <= 0:
                continue
            final_k = max(1, round(num_units * sparsifier.k_fraction))
            if start_k < final_k or start_k > num_units:
                raise ValueError(
                    "kwinners_k_anneal_start must be between "
                    f"final k={final_k} "
                    f"and num_units={num_units}; got {start_k}."
                )
        return self

    @model_validator(mode="after")
    def validate_emergent_grid_arm_selection(self) -> SpatialModelConfig:
        enabled_count = sum(
            int(arm_config.enabled)
            for arm_config in (
                self.path_coloring_arm,
                self.path_coloring_fusion_arm,
                self.wang_grid_arm,
            )
        )
        if enabled_count > 1:
            raise ValueError("Enable only one emergent grid arm per model.")
        if enabled_count == 1 and self.grid_stream.enabled:
            raise ValueError(
                "The legacy grid_stream and an emergent grid arm cannot be enabled together."
            )
        return self

    @model_validator(mode="after")
    def validate_comparator_closed_loop(self) -> SpatialModelConfig:
        if self.attractor_binding.enabled and not self.comparator.closed_loop:
            raise ValueError(
                "spatial_model.attractor_binding requires comparator.closed_loop=true: the pool "
                "scores candidate bindings against the per-step expectation, which only the "
                "closed loop produces."
            )
        if not self.comparator.closed_loop:
            return self
        if self.grid_stream.enabled and self.grid_stream.functional:
            raise ValueError(
                "comparator.closed_loop and grid_stream.functional both claim the predictor's "
                "belief channel; enable only one (the N-prior comparator that fuses both is v2)."
            )
        from placecell_research.spatial_model.components.temporal import (
            temporal_backend_capabilities,
        )

        capabilities = temporal_backend_capabilities(
            self.predictor.family, self.predictor.fla_variant
        )
        if not capabilities.supports_stepwise:
            raise ValueError(
                "comparator.closed_loop needs the stepwise predictor path. "
                + capabilities.stepwise_unsupported_message()
            )
        return self

    def exports_state_readout(self) -> bool:
        """Whether the model must export the encoder/teacher state readout."""
        return self.encoder_readout == "state" or any(
            (objective.type == "blackout_alignment" and not objective.targets)
            or "encoder.state_readout" in objective.targets
            or "teacher.state_readout" in objective.targets
            for objective in self.objectives.values()
        )

    @model_validator(mode="after")
    def validate_readout_cell_placement(self) -> SpatialModelConfig:
        """The readout cell is an encoder-side module, and it owns the head's input."""
        if self.predictor.readout_cell_size > 0:
            raise ValueError(
                "spatial_model.predictor.readout_cell_size is not implemented: the readout cell "
                "sits between the ENCODER mixer and the code head. Set "
                "spatial_model.encoder.readout_cell_size instead."
            )
        if self.predictor.chart_memory_heads > 0:
            raise ValueError(
                "spatial_model.predictor.chart_memory_heads is not implemented: the chart sits "
                "between the ENCODER mixer and the code head. Set "
                "spatial_model.encoder.chart_memory_heads instead."
            )
        if self.predictor.chart_sensory_heads > 0:
            raise ValueError(
                "spatial_model.predictor.chart_sensory_heads is not implemented: the sensory chart "
                "sits in front of the ENCODER trunk. Set spatial_model.encoder.chart_sensory_heads."
            )
        if self.encoder.chart_transition_heads > 0:
            raise ValueError(
                "spatial_model.encoder.chart_transition_heads is not implemented: the transition "
                "chart sits in the PREDICTOR. Set spatial_model.predictor.chart_transition_heads."
            )
        if self.encoder.readout_cell_size > 0 and self.exports_state_readout():
            raise ValueError(
                "spatial_model.encoder.readout_cell_size cannot be combined with a state readout "
                f"(encoder_readout={self.encoder_readout!r}, or an objective targeting "
                "encoder.state_readout / teacher.state_readout): the mixer state and the readout "
                "cell state are two different states, and which one the code head reads would be "
                "ambiguous. Use encoder_readout='mixed' with the readout cell."
            )
        return self

    @model_validator(mode="after")
    def validate_chart_code_readout_placement(self) -> SpatialModelConfig:
        """The chart-code binder writes into ONE k-winners competition on ONE head."""
        if self.predictor.chart_readout != "residual" or self.predictor.code_head_frozen:
            raise ValueError(
                "spatial_model.predictor.chart_readout / code_head_frozen are not implemented: "
                "the chart-code binder sits between the ENCODER head and its sparsifier. Set "
                "spatial_model.encoder.* instead."
            )
        if self.encoder.chart_readout != "code":
            return self
        if self.sparsifier.type != "kwinners" or self.num_code_blocks > 1 or self.code_blocks:
            raise ValueError(
                "spatial_model.encoder.chart_readout='code' needs one global k-winners "
                f"competition to store the winners of (sparsifier.type={self.sparsifier.type!r}, "
                f"num_code_blocks={self.num_code_blocks}, "
                f"{len(self.code_blocks)} explicit code_blocks)."
            )
        if self.encoder_experts.count > 1:
            raise ValueError(
                "spatial_model.encoder.chart_readout='code' cannot be combined with "
                f"encoder_experts.count={self.encoder_experts.count}: the stored map would be the "
                "base head's winners while the competition resolves the combined logits."
            )
        return self

    @model_validator(mode="after")
    def validate_encoder_readout_contract(self) -> SpatialModelConfig:
        if not self.exports_state_readout():
            return self
        unsupported_state_families = {"mlp", "fla"}
        if self.encoder.family in unsupported_state_families:
            raise ValueError(
                "state_readout requires an encoder with an explicit recurrent state; "
                f"family={self.encoder.family!r} does not export one."
            )
        if self.encoder.family == "transformer" and self.encoder.transformer_memory_slots == 0:
            raise ValueError(
                "state_readout with family='transformer' requires "
                "encoder.transformer_memory_slots > 0."
            )
        if self.encoder.family == "xlstm" and self.encoder.xlstm_variant == "large_block":
            raise ValueError(
                "state_readout is unavailable for xlstm_variant='large_block': the NX-AI wrapper "
                "does not expose a skip-free memory branch. Use xlstm_variant='block_stack' with "
                "xlstm_backend='reimpl'."
            )
        return self

    @model_validator(mode="after")
    def validate_stateful_bptt_contract(self) -> SpatialModelConfig:
        if self.training.bptt_window == 0:
            return self
        if self.inputs.encoder_observation_delay_steps > 0:
            raise ValueError(
                "training.bptt_window does not carry encoder observation history across chunks; "
                "use training.bptt_window=0 when encoder_observation_delay_steps > 0."
            )
        supported_recurrent_families = {"rnn", "gru", "lstm", "wyss", "leaky_hierarchy"}
        if self.encoder.family not in supported_recurrent_families:
            raise ValueError(
                "training.bptt_window currently requires encoder.family in "
                f"{sorted(supported_recurrent_families)}, got {self.encoder.family!r}."
            )
        if self.predictor.family not in {"rnn", "gru", "lstm"}:
            raise ValueError(
                "training.bptt_window currently requires predictor.family in "
                f"['gru', 'lstm', 'rnn'], got {self.predictor.family!r}."
            )
        if self.inputs.observation_source == "action":
            raise ValueError(
                "training.bptt_window does not yet carry action-history observation tokens."
            )
        if self.inputs.predictor_input_mode != "encoder":
            raise ValueError(
                "training.bptt_window currently requires inputs.predictor_input_mode='encoder'."
            )
        unsupported_features = []
        if self.training.online.enabled:
            unsupported_features.append("training.online")
        if self.training.replay.enabled:
            unsupported_features.append("training.replay")
        if self.dag_nodes:
            unsupported_features.append("dag_nodes")
        if self.grid_stream.enabled:
            unsupported_features.append("grid_stream")
        if self.motion.enabled:
            unsupported_features.append("motion")
        if self.predictive_context.enabled:
            unsupported_features.append("predictive_context")
        if self.enabled_emergent_grid_arm() is not None:
            unsupported_features.append("emergent_grid_arm")
        if self.masked_predictor.enabled:
            unsupported_features.append("masked_predictor")
        if self.prediction_bootstrap.enabled:
            unsupported_features.append("prediction_bootstrap")
        if self.inverse_dynamics.enabled:
            unsupported_features.append("inverse_dynamics")
        if self.environment_code.enabled:
            unsupported_features.append("environment_code")
        if self.encoder_experts.count != 1 or self.encoder_experts.streams:
            unsupported_features.append("encoder_experts")
        if self.predictor_experts.count != 1:
            unsupported_features.append("predictor_experts")
        if self.inputs.input_corruption.enabled:
            unsupported_features.append("inputs.input_corruption")
        visual_masking = self.inputs.visual_masking
        if visual_masking.blackout_probability > 0.0 or visual_masking.stride > 1:
            unsupported_features.append("inputs.visual_masking")
        if self.regularization.sites:
            unsupported_features.append("regularization")
        if self.encoder.slow_leak_alpha != 0.0:
            unsupported_features.append("encoder.slow_leak_alpha")
        if any(block.slow_leak_alpha is not None for block in self.code_blocks):
            unsupported_features.append("code_blocks.slow_leak_alpha")
        if unsupported_features:
            raise ValueError(
                "training.bptt_window does not yet support: "
                + ", ".join(unsupported_features)
                + "."
            )
        supported_objectives = {
            "prediction_alignment",
            "vicreg",
            "normalized_slowness",
            "l1_sparsity",
            "multistep_rollout",
        }
        unsupported_objectives = sorted(
            {
                objective.type
                for objective in self.objectives.values()
                if objective.type not in supported_objectives
            }
        )
        if unsupported_objectives:
            raise ValueError(
                f"training.bptt_window supports {sorted(supported_objectives)}; "
                f"got unsupported objective types {unsupported_objectives}."
            )
        return self


@dataclass(config=PYDANTIC_CONFIG)
class ExpertProbeConfig:
    """Bounded in-training diagnostics for competence-routed DAG experts."""

    enabled: bool = False
    routing_source: str = "experts.routing_onehot"
    routing_labels: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    max_episodes: int = Field(default=128, ge=2)
    steps_per_episode: int = Field(default=256, ge=1)
    spatial_bins: int = Field(default=12, ge=2)
    checkpoint_source: str = "experts.place_codes"

    @model_validator(mode="after")
    def validate_enabled_probe(self) -> ExpertProbeConfig:
        if not self.enabled:
            return self
        if not self.sources:
            raise ValueError("evaluation.expert_probe.sources must not be empty when enabled.")
        if self.checkpoint_source not in self.sources:
            raise ValueError(
                "evaluation.expert_probe.checkpoint_source must be listed in expert_probe.sources."
            )
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("evaluation.expert_probe.sources must not contain duplicates.")
        if len(self.routing_labels) != len(set(self.routing_labels)):
            raise ValueError("evaluation.expert_probe.routing_labels must not contain duplicates.")
        return self


@dataclass(config=PYDANTIC_CONFIG)
class HistoryAblationConfig:
    """Held-out blackout test for dependence on persistent encoder state."""

    enabled: bool = False
    blackout_start_step: int = Field(default=128, ge=1)
    blackout_length: int = Field(default=96, ge=1)
    max_episodes: int = Field(default=64, ge=2)


@dataclass(config=PYDANTIC_CONFIG)
class RepresentationCollectionConfig:
    """Inputs for the collect-representations stage."""

    model_artifact_id: str = ""
    dataset_artifact_id: str = ""
    dataset_artifact_type: str = ""
    split_artifact_id: str = ""
    split_names: list[str] = field(default_factory=lambda: ["validation", "test"])
    sources: list[str] = field(default_factory=list)
    batch_size: int = Field(default=8, ge=1)
    include_batch_keys: list[str] = field(default_factory=lambda: ["rgb", "latent"])
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    max_episodes: int = Field(default=0, ge=0)


@dataclass(config=PYDANTIC_CONFIG)
class EvaluationConfig:
    model_artifact_id: str = ""
    dataset_artifact_id: str = ""
    dataset_artifact_type: str = ""
    split_artifact_id: str = ""
    device: str = "auto"
    split_name: str = "validation"
    split_names: list[str] = field(default_factory=lambda: ["validation", "test"])
    online_decode_source: str = "encoder.place_codes"
    sources: list[str] = field(default_factory=lambda: ["encoder.place_codes"])
    action_future_horizons: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    action_ngram_orders: list[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    action_persistence_lags: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    eval_every_n_epochs: int = Field(default=1, ge=1)
    eval_schedule: Literal["uniform", "front_loaded"] = "uniform"
    max_eval_episodes: int = Field(default=0, ge=0)
    decode_train_fraction: float = Field(default=0.8, ge=0.0, le=1.0)
    decode_ridge_alpha: float = Field(default=1e-3, gt=0.0)
    matched_decode: bool = False
    decode_include_shuffle: bool = True
    nonlinear_decode_enabled: bool = True
    nonlinear_decode_hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    nonlinear_decode_max_epochs: int = Field(default=200, ge=1)
    nonlinear_decode_batch_size: int = Field(default=1024, ge=1)
    nonlinear_decode_max_train_samples: int = Field(default=65_536, ge=1)
    nonlinear_decode_max_validation_samples: int = Field(default=16_384, ge=1)
    nonlinear_decode_random_seed: int = 0
    compute_spatial_info: bool = True
    spatial_info_top_k: int = Field(default=16, ge=1)
    online_split_half_num_random_splits: int = Field(default=20, ge=0)
    compute_gridness: bool = False
    gridness_threshold: float = 0.37
    evaluate_training_split: bool = False
    eval_directionality: bool = True
    expert_probe: ExpertProbeConfig = field(default_factory=ExpertProbeConfig)
    history_ablation: HistoryAblationConfig = field(default_factory=HistoryAblationConfig)

    @model_validator(mode="after")
    def _validate_action_temporal_settings(self) -> EvaluationConfig:
        positive_fields = (
            ("action_future_horizons", self.action_future_horizons),
            ("action_persistence_lags", self.action_persistence_lags),
        )
        for field_name, values in positive_fields:
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{field_name} must contain positive integers.")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates.")
        if not self.action_ngram_orders or any(value < 0 for value in self.action_ngram_orders):
            raise ValueError("action_ngram_orders must contain non-negative integers.")
        if len(self.action_ngram_orders) != len(set(self.action_ngram_orders)):
            raise ValueError("action_ngram_orders must not contain duplicates.")
        return self


class AnalysisTargetConfig(BaseModel):
    """Typed target core with passthrough support for module-specific target payloads."""

    model_config = ConfigDict(extra="allow")

    source: str = "encoder.place_codes"
    enabled: bool = True
    modules: list[str] = Field(default_factory=list)


_FDR_PERMUTATION_COUNT_FIELDS: dict[str, float | str] = {
    "spatial_information_null_shuffles": 0.05,
    "directionality_null_shuffles": 0.05,
    "head_direction_null_shuffles": 0.05,
    "eigenmode_morphology_shuffle_count": "eigenmode_morphology_fdr_alpha",
    "reanchoring_shuffle_count": "reanchoring_fdr_alpha",
}
_MINIMUM_SUPPORTABLE_DISCOVERY_FRACTION = 0.05


@dataclass(config=ConfigDict(validate_assignment=True, extra="forbid"))
class AnalysisConfig:
    model_artifact_id: str = ""
    dataset_artifact_id: str = ""
    dataset_artifact_type: str = ""
    split_artifact_id: str = ""
    split_name: str = "test"
    batch_size: int = Field(default=8, ge=1)
    device: str = "auto"

    num_bins_x: int = Field(default=60, ge=1)
    num_bins_y: int = Field(default=60, ge=1)
    num_bins: int = Field(default=24, ge=1)
    render_dpi: int = Field(default=160, ge=1)
    max_eval_episodes: int = Field(default=0, ge=0)
    smoothing_sigma: float = Field(default=0.3, ge=0.0)
    code_timescale_lags: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    excess_stability_lag: int = Field(default=16, ge=1)
    min_occupancy: float = Field(default=1e-6, ge=0.0)
    per_episode_num_bins_x: int = Field(default=20, ge=1)
    per_episode_num_bins_y: int = Field(default=20, ge=1)
    per_episode_smoothing_sigma: float = Field(default=1.5, ge=0.0)
    per_episode_min_occupancy: float = Field(default=1e-6, ge=0.0)
    per_episode_minimum_visited_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    per_episode_minimum_valid_steps: int = Field(default=200, ge=1)
    field_stability_top_k: int = Field(default=6, ge=1)
    reliability_threshold_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    reliability_threshold_quantile: float = Field(default=0.95, ge=0.0, le=1.0)
    reliability_use_absolute_activations: bool = False
    field_traversal_minimum_traversals: int = Field(default=5, ge=1)
    field_traversal_heading_sectors: int = Field(default=8, ge=1)
    place_field_core_threshold_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    spatial_information_null_shuffles: int = Field(default=999, ge=0)
    spatial_information_null_seed: int = 0
    split_half_num_random_splits: int = Field(default=20, ge=0)
    split_half_random_split_seed: int = 0
    place_cell_gate_minimum_split_half: float = Field(default=0.8, ge=-1.0, le=1.0)
    place_cell_gate_minimum_coherence: float = Field(default=0.3, ge=-1.0, le=1.0)
    place_cell_gate_maximum_confound: float = Field(default=0.5, ge=0.0, le=1.0)
    directionality_null_shuffles: int = Field(default=999, ge=0)
    head_direction_null_shuffles: int = Field(default=999, ge=0)
    head_direction_min_heading_quadrants: int = Field(default=3, ge=1, le=4)
    place_field_threshold_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    place_field_core_threshold_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    omnidirectional_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    directional_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    directionality_num_bins: int = Field(default=24, ge=1)
    directionality_num_bins_x: int = Field(default=24, ge=1)
    directionality_num_bins_y: int = Field(default=24, ge=1)
    min_occupancy_per_quadrant: int = Field(default=5, ge=1)
    min_heading_quadrants: int = Field(default=3, ge=1, le=4)
    min_field_bins: int = Field(default=3, ge=1)
    min_fire_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    within_heading_num_bins: int = Field(default=4, ge=1)
    within_heading_min_steps_per_cell: int = Field(default=5, ge=1)
    within_heading_conjunctive_gain: float = Field(default=0.2, ge=0.0)

    head_direction_num_bins: int = Field(default=36, ge=1)
    head_direction_spatial_num_bins: int = Field(default=24, ge=1)
    head_direction_spatial_num_bins_x: int = Field(default=24, ge=1)
    head_direction_spatial_num_bins_y: int = Field(default=24, ge=1)
    head_direction_num_bins_x: int = Field(default=24, ge=1)
    head_direction_num_bins_y: int = Field(default=24, ge=1)
    head_direction_vector_length_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    head_direction_spatial_coverage_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    head_direction_position_invariance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    head_direction_hd_cell_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    head_direction_min_active_spatial_bins: int = Field(default=4, ge=1)
    head_direction_min_occupancy_per_bin: int = Field(default=5, ge=1)
    head_direction_min_heading_bins: int = Field(default=9, ge=1)
    head_direction_active_threshold_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    head_direction_min_fire_rate: float = Field(default=0.01, ge=0.0, le=1.0)

    place_field_overlay_active_threshold: float = 0.0
    place_field_overlay_blend_mode: Literal["max", "additive", "alpha"] = "max"
    place_field_overlay_max_cells: int | None = Field(default=None, ge=0)
    place_field_overlay_max_frames: int = Field(default=240, ge=1)
    place_field_overlay_kwinners_k_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    place_field_overlay_frame_duration: float = Field(default=0.5, gt=0.0)
    place_metric_negative_tolerance: float = Field(default=1e-8, ge=0.0)
    place_metric_max_negative_bin_fraction: float = Field(default=0.01, ge=0.0, le=1.0)
    place_metric_max_negative_peak_fraction: float = Field(default=0.05, ge=0.0)
    per_bin_cv_min_episodes: int = Field(default=3, ge=2)
    bin_consistency_active_bin_peak_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    bin_consistency_active_episode_threshold_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    split_half_agreement_min_episodes_per_half: int = Field(default=2, ge=1)
    rate_map_panel_reliability_metric: Literal[
        "thresholded_reliability",
        "quantile_thresholded_reliability",
        "bin_consistency",
        "split_half_agreement",
    ] = "quantile_thresholded_reliability"
    rate_map_panel_emit_thresholded_reliability_figure: bool = True
    rate_map_panel_emit_quantile_thresholded_reliability_figure: bool = True
    rate_map_panel_top_k: int = Field(default=128, ge=1)
    rate_map_panel_show_all_units: bool = False
    rate_map_export_per_unit: bool = False
    rate_map_grid_top_k: int = Field(default=128, ge=1)
    rate_map_grid_show_all_units: bool = False
    rate_map_spike_overlay_max_points: int = Field(default=180, ge=0)
    rate_map_unit_order: Literal["ranking_score", "field_position"] = "field_position"
    heading_rate_map_overlay_num_bins_x: int = Field(default=60, ge=1)
    heading_rate_map_overlay_num_bins_y: int = Field(default=60, ge=1)
    heading_rate_map_overlay_smoothing_sigma: float = Field(default=0.3, ge=0.0)
    heading_rate_map_overlay_min_occupancy: float = Field(default=1e-6, ge=0.0)
    heading_rate_map_overlay_top_k: int = Field(default=12, ge=1)
    heading_rate_map_overlay_page_size: int = Field(default=8, ge=1)
    heading_rate_map_overlay_render_dpi: int = Field(default=160, ge=1)
    heading_rate_map_overlay_local_num_bins_x: int = Field(default=20, ge=1)
    heading_rate_map_overlay_local_num_bins_y: int = Field(default=20, ge=1)
    heading_rate_map_overlay_num_heading_bins: int = Field(default=36, ge=1)
    heading_rate_map_overlay_min_occupancy_per_heading_bin: int = Field(default=5, ge=1)
    heading_rate_map_overlay_active_threshold: float = 0.0
    heading_rate_map_overlay_field_threshold_fraction: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
    )
    rate_map_shared_color_scale: bool = False
    rate_map_colormap_mode: Literal[
        "reds", "inferno", "turbo", "viridis", "magma", "cividis", "coolwarm", "plasma", "auto"
    ] = "reds"
    rate_map_panel_metric_fill_sigma_bins: float = Field(default=1.0, ge=0.0)
    decode_train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    decode_ridge_alpha: float = Field(default=1e-3, ge=0.0)
    decode_include_shuffle: bool = True
    decode_bias_min_samples_per_bin: int = Field(default=8, ge=1)
    decode_max_samples: int = Field(default=0, ge=0)
    decode_random_seed: int = 0
    decode_error_num_bins_x: int = Field(default=60, ge=1)
    decode_error_num_bins_y: int = Field(default=60, ge=1)
    decode_region_num_regions_x: int = Field(default=4, ge=1)
    decode_region_num_regions_y: int = Field(default=4, ge=1)
    decode_region_max_iter: int = Field(default=200, ge=1)
    structural_room_decode_max_samples: int = Field(default=50_000, ge=0)
    decode_nonlinear_enabled: bool = True
    decode_nonlinear_hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    decode_nonlinear_max_epochs: int = Field(default=200, ge=1)
    decode_nonlinear_batch_size: int = Field(default=1024, ge=1)
    decode_nonlinear_max_train_samples: int = Field(default=65_536, ge=1)
    decode_nonlinear_max_validation_samples: int = Field(default=16_384, ge=1)
    decode_nonlinear_random_seed: int = 0
    redundancy_max_samples: int = Field(default=4096, ge=64)
    redundancy_random_seed: int = 0
    example_episode_index: int = Field(default=-1, ge=-1)
    example_episode_random_seed: int | None = 0
    example_episode_top_k: int = Field(default=0, ge=0)
    example_episode_frame_duration: float = Field(default=0.14, gt=0.0)
    per_episode_rate_maps_top_k: int = Field(default=8, ge=0)
    per_episode_rate_maps_num_episodes: int = Field(default=6, ge=1)
    per_episode_rate_maps_render_dpi: int = Field(default=160, ge=1)
    per_episode_rate_maps_page_size: int = Field(default=6, ge=1)

    selectivity_position_centers_x: int = Field(default=8, ge=1)
    selectivity_position_centers_y: int = Field(default=8, ge=1)
    selectivity_position_rbf_width_scale: float = Field(default=1.25, gt=0.0)
    selectivity_heading_harmonics: int = Field(default=2, ge=1)
    selectivity_ridge_alpha: float = Field(default=1e-2, ge=0.0)
    selectivity_train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    selectivity_split_seed: int = 0
    selectivity_min_r2: float = 0.02
    selectivity_mixed_dominance: float = Field(default=0.6, ge=0.0, le=1.0)
    selectivity_max_samples: int = Field(default=150_000, ge=0)
    selectivity_visual_pca_components: int = Field(default=64, ge=1)
    spatial_code_min_r2: float = 0.02
    spatial_code_sufficiency_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    umap_max_points: int = Field(default=4096, ge=1)
    umap_n_neighbors: int = Field(default=25, ge=1)
    umap_min_dist: float = Field(default=0.1, ge=0.0)
    umap_metric: str = "euclidean"
    umap_random_seed: int = 42
    umap_pool_mode: Literal["none", "spatial_bins"] = "spatial_bins"
    umap_pool_num_bins_x: int = Field(default=20, ge=1)
    umap_pool_num_bins_y: int = Field(default=20, ge=1)
    umap_pool_min_count: int = Field(default=2, ge=1)
    umap_trustworthiness_neighbors: int = Field(default=15, ge=1)
    umap_transition_min_count: int = Field(default=2, ge=1)
    umap_transition_metric: Literal["correlation", "cosine", "euclidean"] = "correlation"
    umap_transition_physical_match_caliper: float = Field(default=0.25, ge=0.0)
    manifold_max_points: int = Field(default=4096, ge=4)
    manifold_pool_num_bins_x: int = Field(default=20, ge=1)
    manifold_pool_num_bins_y: int = Field(default=20, ge=1)
    manifold_pool_min_count: int = Field(default=2, ge=1)
    manifold_metric: Literal["euclidean", "cosine"] = "euclidean"
    manifold_trustworthiness_neighbors: int = Field(default=15, ge=1)
    manifold_random_seed: int = 42
    isomap_n_neighbors: int = Field(default=10, ge=2)
    neighborhood_preservation_neighbors: int = Field(default=15, ge=1)
    neighborhood_preservation_max_points: int = Field(default=4096, ge=4)
    neighborhood_preservation_pool_num_bins_x: int = Field(default=20, ge=1)
    neighborhood_preservation_pool_num_bins_y: int = Field(default=20, ge=1)
    neighborhood_preservation_pool_min_count: int = Field(default=2, ge=1)
    neighborhood_preservation_random_seed: int = 0
    transition_geometry_num_bins_x: int = Field(default=24, ge=2)
    transition_geometry_num_bins_y: int = Field(default=24, ge=2)
    transition_geometry_num_modes: int = Field(default=6, ge=1)
    transition_geometry_alignment_top_k: int = Field(default=16, ge=1)
    transition_geometry_include_self_transitions: bool = False
    eigenmode_morphology_max_frequency: int = Field(default=4, ge=1)
    eigenmode_morphology_shuffle_count: int = Field(default=999, ge=0)
    eigenmode_morphology_shuffle_top_k: int = Field(default=32, ge=1)
    eigenmode_morphology_fdr_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    eigenmode_morphology_degeneracy_tolerance: float = Field(default=0.05, ge=0.0)
    eigenmode_morphology_render_top_k: int = Field(default=6, ge=1)
    eigenmode_morphology_random_seed: int = 0
    cognitive_map_num_bins_x: int = Field(default=20, ge=1)
    cognitive_map_num_bins_y: int = Field(default=20, ge=1)
    cognitive_map_max_scatter_points: int = Field(default=4000, ge=1)
    cognitive_map_render_rdm: bool = True

    spatial_code_dynamics_lags: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    spatial_code_dynamics_support_epsilon: float = Field(default=1e-8, ge=0.0)
    spatial_code_dynamics_event_thresholds: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1]
    )
    dwell_shuffle_seed: int = 0

    probing_future_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    probing_past_horizons: list[int] = field(default_factory=lambda: [1, 5, 10])
    probing_include_heading: bool = True
    probing_include_time: bool = True
    probing_include_wall_distance: bool = True
    probing_include_landmark_distances: bool = True
    probing_include_novelty: bool = True
    probing_train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    probing_ridge_alpha: float = Field(default=1e-3, ge=0.0)
    probing_include_shuffle: bool = True
    probing_shuffle_seed: int = 0
    probing_novelty_num_bins_x: int = Field(default=60, ge=1)
    probing_novelty_num_bins_y: int = Field(default=60, ge=1)

    expert_map_num_bins: int = Field(default=12, ge=1)
    active_peak_rate_threshold: float = Field(default=1e-6, ge=0.0)
    remapping_shuffle_iterations: int = Field(default=100, ge=0)
    remapping_shuffle_seed: int = 0

    reanchoring_num_bins_x: int = Field(default=60, ge=1)
    reanchoring_num_bins_y: int = Field(default=60, ge=1)
    reanchoring_per_episode_num_bins_x: int = Field(default=20, ge=1)
    reanchoring_per_episode_num_bins_y: int = Field(default=20, ge=1)
    reanchoring_smoothing_sigma: float = Field(default=1.0, ge=0.0)
    reanchoring_min_occupancy: float = Field(default=1e-6, ge=0.0)
    reanchoring_minimum_valid_steps: int = Field(default=50, ge=1)
    reanchoring_minimum_visited_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    reanchoring_max_episodes: int = Field(default=256, ge=0)
    reanchoring_shuffle_count: int = Field(default=999, ge=0)
    reanchoring_shuffle_top_k: int = Field(default=32, ge=1)
    reanchoring_fdr_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    reanchoring_render_top_k: int = Field(default=8, ge=1)
    reanchoring_render_example_episodes: int = Field(default=4, ge=1)
    reanchoring_random_seed: int = 0

    sr_oracle_discount_gamma: float = Field(default=0.95, ge=0.0, lt=1.0)
    sr_oracle_num_bins_x: int = Field(default=20, ge=1)
    sr_oracle_num_bins_y: int = Field(default=20, ge=1)
    sr_oracle_num_eigen_modes: int = Field(default=8, ge=1)
    successor_return_discount_gamma: float = Field(default=0.95, ge=0.0, lt=1.0)
    successor_return_normalized: bool = True
    successor_return_shuffle_seed: int = 0

    vector_num_distance_bins: int = Field(default=12, ge=1)
    vector_num_angle_bins: int = Field(default=24, ge=1)
    vector_max_distance: float = Field(default=0.0, ge=0.0)
    vector_min_peak_distance_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    vector_reliability_threshold: float = Field(default=0.4, ge=-1.0, le=1.0)
    vector_shuffle_count: int = Field(default=0, ge=0)
    vector_shuffle_top_k: int = Field(default=32, ge=1)
    band_score_threshold: float = Field(default=0.35, ge=0.0)
    border_field_threshold_fraction: float = Field(default=0.3, ge=0.0, le=1.0)
    border_score_threshold: float = Field(default=0.5)
    geometry_max_samples: int = Field(default=5000, ge=2)
    conformal_max_points: int = Field(default=1500, ge=10)
    conformal_neighbor_k: int = Field(default=8, ge=2)
    decode_extrapolation_axis: Literal["x", "y"] = "x"
    topology_max_points: int = Field(default=700, ge=10)
    topology_lifetime_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    fourier_ring_band_pass_ratio: float = Field(default=1.2, ge=1.0)
    fourier_ring_num_bins: int = Field(default=48, ge=8)
    fourier_ring_smoothing_sigma: float = Field(default=0.4, ge=0.0)

    dataset_coverage_extra_splits: list[str] = field(default_factory=lambda: ["train", "test"])
    max_cost_tier: Literal["light", "standard", "heavy"] = "standard"
    targets: dict[str, AnalysisTargetConfig] = field(default_factory=dict)
    target_order: list[str] = field(default_factory=list)
    comparative: dict[str, dict[str, Any]] = field(default_factory=dict)

    @model_validator(mode="after")
    def _check_permutation_counts_support_fdr(self) -> AnalysisConfig:
        """Refuse permutation counts a BH-FDR step cannot turn into a discovery."""
        for count_field, alpha in _FDR_PERMUTATION_COUNT_FIELDS.items():
            num_permutations = int(getattr(self, count_field))
            if num_permutations <= 0:
                continue
            level = alpha if isinstance(alpha, float) else float(getattr(self, alpha))
            smallest_p = 1.0 / (num_permutations + 1)
            required_fraction = smallest_p / level
            detail = (
                f"{count_field}={num_permutations} gives a smallest reachable p of "
                f"1/{num_permutations + 1} = {smallest_p:.6g}, and BH-FDR at q={level:g} rejects "
                f"nothing until {required_fraction:.1%} of the tested units sit on that floor"
            )
            if required_fraction > 1.0:
                raise ValueError(
                    f"{detail} -- more units than are tested, so no unit can ever be called "
                    f"significant. Raise it to at least {math.ceil(1.0 / level) - 1} "
                    f"(preferably 999), or set it to 0 to disable the null."
                )
            if required_fraction > _MINIMUM_SUPPORTABLE_DISCOVERY_FRACTION:
                warnings.warn(
                    f"{detail}, so only dense effects are discoverable (of 1000 tested units, "
                    f"{math.ceil(required_fraction * 1000) - 1} on the floor give 0 "
                    f"discoveries). Raise it to 999 for a "
                    f"{1.0 / (1000 * level):.1%} floor.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


def analysis_config_default(field_name: str) -> Any:
    """The default AnalysisConfig declares for field_name."""
    return AnalysisConfig.__pydantic_fields__[field_name].default


@dataclass(config=PYDANTIC_CONFIG)
class ExperimentConfig:
    name: str = "placecell_experiment"
    seed: SeedBundleConfig = field(default_factory=SeedBundleConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    dataset: DatasetReferenceConfig = field(default_factory=DatasetReferenceConfig)
    splits: SplitPolicyConfig = field(default_factory=SplitPolicyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    spatial_model: SpatialModelConfig = field(default_factory=SpatialModelConfig)
    representation_collection: RepresentationCollectionConfig = field(
        default_factory=RepresentationCollectionConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reuse: ReuseConfig = field(default_factory=ReuseConfig)
    policies: PolicyConfig = field(default_factory=PolicyConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> dict[str, Any]:
        return _dump_config(self)
