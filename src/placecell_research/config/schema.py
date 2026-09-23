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

PYDANTIC_CONFIG = ConfigDict(validate_assignment=True, extra="forbid")
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


REPRODUCTION_JOB_KINDS = ("data", "train", "representations", "measures", "navigation", "summary")


@dataclass(config=PYDANTIC_CONFIG)
class JobResourcesConfig:
    """Launcher values that one kind of reproduction job overrides."""

    partition: str | None = None
    exclude_nodes: list[str] | None = None
    gpus: int | None = Field(default=None, ge=0)
    gpu_type: str | None = None
    cpus_per_task: int | None = Field(default=None, ge=1)
    memory_gb: int | None = Field(default=None, ge=1)
    time_hours: int | None = Field(default=None, ge=1)
    omp_num_threads: int | None = Field(default=None, ge=1)


@dataclass(config=PYDANTIC_CONFIG)
class LauncherConfig:
    type: Literal["local", "slurm"] = "local"
    partition: str = "gpu"
    account: str = ""
    qos: str = ""
    constraint: str = ""
    exclude_nodes: list[str] = field(default_factory=list)
    gpus: int = Field(default=1, ge=0)
    gpu_type: str | None = None
    mig_gpu_types: list[str] = field(default_factory=list)
    time_hours: int = Field(default=48, ge=1)
    memory_gb: int = Field(default=32, ge=1)
    cpus_per_task: int = Field(default=4, ge=1)
    env_setup: str = ""
    site_env_script: str = ""
    exports: dict[str, str] = field(default_factory=dict)
    threading_safety: ThreadingSafetyConfig = field(default_factory=ThreadingSafetyConfig)
    analysis_workers: int = Field(default=0, ge=0)
    max_concurrent_gpu_jobs: int = Field(default=2, ge=1)
    max_concurrent_cpu_jobs: int = Field(default=8, ge=1)
    job_resources: dict[str, JobResourcesConfig] = field(default_factory=dict)

    @field_validator("partition", "account", "qos", "constraint", "env_setup", mode="before")
    @classmethod
    def coerce_scheduler_text(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int | float) else value

    @field_validator("exports", mode="before")
    @classmethod
    def coerce_export_values(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        return value

    @field_validator("job_resources")
    @classmethod
    def validate_job_resource_kinds(
        cls, value: dict[str, JobResourcesConfig]
    ) -> dict[str, JobResourcesConfig]:
        unknown = sorted(set(value) - set(REPRODUCTION_JOB_KINDS))
        if unknown:
            raise ValueError(
                f"Unknown launcher.job_resources kinds {unknown}; "
                f"expected any of {list(REPRODUCTION_JOB_KINDS)}."
            )
        return value


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
    allow_domain_transfer: bool = False


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
    device: str | None = None
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
class EncodingConfig:
    """How encode-dataset runs the frozen vision encoder over a raw dataset."""

    device: str = "auto"
    read_workers: int | None = Field(default=None, ge=0)
    episode_batch: int = Field(default=1, ge=1)


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
    observation_source: Literal["latent", "rgb"] = "latent"
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
        "clockwork",
        "mtrnn",
    ] = "gru"
    layer_sizes: list[int] = field(default_factory=lambda: [256])
    head_activation: Literal["none", "relu", "softplus"] = "none"
    head_weight_sparsity: float = Field(default=1.0, gt=0.0, le=1.0)
    normalize_codes: bool = True
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
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

    @property
    def input_size(self) -> int:
        if not self.layer_sizes:
            raise ValueError("TemporalFamilyConfig.layer_sizes must contain at least one width.")
        return int(self.layer_sizes[0])

    @property
    def output_size(self) -> int:
        """Width of the last temporal layer, which the code head reads."""
        if not self.layer_sizes:
            raise ValueError("TemporalFamilyConfig.layer_sizes must contain at least one width.")
        return int(self.layer_sizes[-1])

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
    soft_wta_target_sparsity: float = Field(default=0.1, ge=0.0, le=1.0)

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
class TeacherStudentConfig:
    mode: Literal["none", "ema_byol"] = "ema_byol"
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
    head_hidden_dim: int = Field(default=256, ge=1)
    target_module: str = "predictor"


_OBJECTIVE_FIELD_RULES: dict[str, tuple[Any, frozenset[str]]] = {
    "loss_type": ("cosine", frozenset({"prediction_alignment"})),
    "support_rank_weight": (0.1, frozenset({"prediction_alignment"})),
    "target_offset": (1, frozenset({"prediction_alignment"})),
    "variance_weight": (0.5, frozenset({"vicreg"})),
    "covariance_weight": (0.5, frozenset({"vicreg"})),
    "minimum_std": (0.8, frozenset({"vicreg"})),
    "covariance_normalization": ("feature_count", frozenset({"vicreg"})),
    "timescale": (0, frozenset({"timescale_alignment"})),
    "event_gated": (False, frozenset({"timescale_alignment"})),
    "horizon": (8, frozenset({"temporal_stability"})),
    "decorrelation_weight": (0.0, frozenset({"temporal_stability"})),
    "detach_variance_denominator": (True, frozenset({"temporal_stability"})),
    "anchor_stride": (1, frozenset({"vicreg"})),
    "anchor_offset": (0, frozenset({"vicreg"})),
    "auxiliary_head": (None, frozenset({"latent_reconstruction"})),
}


@dataclass(config=PYDANTIC_CONFIG)
class ObjectiveConfig:
    type: str = ""
    weight: float = Field(default=1.0, ge=0.0)
    targets: list[str] = field(default_factory=list)
    loss_type: str = "cosine"
    target_offset: Literal[0, 1] = 1
    timescale: int = Field(default=0, ge=0)
    event_gated: bool = False
    support_rank_weight: float = Field(default=0.1, ge=0.0)
    variance_weight: float = Field(default=0.5, ge=0.0)
    covariance_weight: float = Field(default=0.5, ge=0.0)
    minimum_std: float = Field(default=0.8, ge=0.0)
    covariance_normalization: Literal["feature_count", "feature_count_squared"] = "feature_count"
    decorrelation_weight: float = Field(default=0.0, ge=0.0)
    detach_variance_denominator: bool = True
    horizon: int = Field(default=8, ge=1)
    anchor_stride: int = Field(default=1, ge=1)
    anchor_offset: int = Field(default=0, ge=0)
    auxiliary_head: AuxiliaryHeadConfig | None = None

    @model_validator(mode="after")
    def validate_anchor_offset(self) -> ObjectiveConfig:
        if self.anchor_offset >= self.anchor_stride:
            raise ValueError("anchor_offset must be below anchor_stride.")
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
class PhaseConfig:
    name: str
    epochs: int = Field(ge=1)
    train: list[str] = field(default_factory=list)


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
    device: str = "auto"
    selection: SelectionPolicyConfig = field(default_factory=SelectionPolicyConfig)
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
                required_encoders = (
                    {"encoder"} if phase_selectors & predictor_side_names else set()
                )
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
    predictor_sparsifier: SparsifierConfig = field(
        default_factory=lambda: SparsifierConfig(type="none")
    )
    cell_type_heads: list[CellTypeHeadConfig] = field(default_factory=list)
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

    def exports_state_readout(self) -> bool:
        """Whether the model must export the encoder/teacher state readout."""
        return self.encoder_readout == "state" or any(
            "encoder.state_readout" in objective.targets
            or "teacher.state_readout" in objective.targets
            for objective in self.objectives.values()
        )

    @model_validator(mode="after")
    def validate_encoder_readout_contract(self) -> SpatialModelConfig:
        if self.exports_state_readout() and self.encoder.family == "mlp":
            raise ValueError(
                "state_readout requires an encoder with an explicit recurrent state; "
                f"family={self.encoder.family!r} does not export one."
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
        supported_recurrent_families = {"rnn", "gru", "lstm"}
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
        if self.inputs.predictor_input_mode != "encoder":
            raise ValueError(
                "training.bptt_window currently requires inputs.predictor_input_mode='encoder'."
            )
        unsupported_features = []
        if self.prediction_bootstrap.enabled:
            unsupported_features.append("prediction_bootstrap")
        if self.inverse_dynamics.enabled:
            unsupported_features.append("inverse_dynamics")
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
        supported_objectives = {"prediction_alignment", "vicreg", "l1_sparsity"}
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
    batch_size: int = Field(default=16, ge=1)
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
    evaluate_training_split: bool = False
    eval_directionality: bool = True

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


@dataclass(config=PYDANTIC_CONFIG)
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
class MeasuresConfig:
    """Settings of pc measures."""

    null_shuffles: int = Field(default=999, ge=1)
    traversal_shifts: int = Field(default=999, ge=0)
    read_time_top_k: int = Field(default=0, ge=0)
    output_dir: str = "measures"


@dataclass(config=PYDANTIC_CONFIG)
class ExperimentConfig:
    name: str = "placecell_experiment"
    seed: SeedBundleConfig = field(default_factory=SeedBundleConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    dataset: DatasetReferenceConfig = field(default_factory=DatasetReferenceConfig)
    splits: SplitPolicyConfig = field(default_factory=SplitPolicyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
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
    measures: MeasuresConfig = field(default_factory=MeasuresConfig)

    def to_dict(self) -> dict[str, Any]:
        return _dump_config(self)


@dataclass(config=PYDANTIC_CONFIG)
class SweepConfig:
    method: Literal["grid", "paired"] = "grid"
    base_experiment: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [0])
    objective_metric: str = "validation.xy_decode_rmse"
    direction: Literal["minimize", "maximize"] = "minimize"


@dataclass(config=PYDANTIC_CONFIG)
class CurriculumAnalyzeAfter:
    datasets: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    comparative_modules: list[str] = field(default_factory=list)
    checkpoints: list[str] | str | None = None


@dataclass(config=PYDANTIC_CONFIG)
class CurriculumPhaseConfig:
    name: str = ""
    dataset: str = ""
    epochs: int = Field(default=32, ge=1)
    resume_from: str | None = None
    resume_policy: Literal["fresh", "weights_only", "weights_and_optimizer"] = "weights_only"
    analyze_after: CurriculumAnalyzeAfter = field(default_factory=CurriculumAnalyzeAfter)


@dataclass(config=PYDANTIC_CONFIG)
class CurriculumSourceConfig:
    raw_dataset: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    collection: dict[str, Any] = field(default_factory=dict)


@dataclass(config=PYDANTIC_CONFIG)
class CurriculumConfig:
    name: str = "curriculum"
    base_experiment: str = ""
    sources: dict[str, CurriculumSourceConfig] = field(default_factory=dict)
    vision_encoder: dict[str, Any] = field(default_factory=dict)
    encoding: dict[str, Any] = field(default_factory=dict)
    phases: list[CurriculumPhaseConfig] = field(default_factory=list)
    final_analysis: CurriculumAnalyzeAfter = field(default_factory=CurriculumAnalyzeAfter)


@dataclass(config=PYDANTIC_CONFIG)
class StudyConfig:
    name: str = "study"
    sweep: SweepConfig | None = None
    curriculum: CurriculumConfig | None = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)

    def to_dict(self) -> dict[str, Any]:
        return _dump_config(self)
