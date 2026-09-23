"""Run tracking helpers."""

from .console import (
    emit_metrics_block,
    emit_text_block,
    render_metrics_block,
    render_stage_result_summary,
    render_text_block,
)
from .naming import (
    RunIdentity,
    capture_git_state,
    generate_signature,
    generate_variant_slug,
    make_run_id,
)
from .progress import ConsoleProgressReporter, ProgressTracker, ProgressUpdate
from .run_directory import (
    RunDirectory,
    RunResultShortcutSummary,
    RunSlurmLogLinkSummary,
    repair_result_shortcut_links,
    repair_slurm_log_links,
)
from .run_index import (
    RunIndexSummary,
    rebuild_stage_run_index,
    rebuild_variant_run_index,
    write_stage_run_link,
    write_variant_run_link,
)
from .stage_run import ManagedStageRun, managed_stage_run
from .tags import (
    curriculum_tags,
    default_wandb_group,
    merge_tags,
    stage_tags,
    study_tags,
    sweep_tags,
)
from .wandb_logger import WandbLogger, build_wandb_run_config, flatten_wandb_config

__all__ = [
    "emit_metrics_block",
    "emit_text_block",
    "render_metrics_block",
    "render_stage_result_summary",
    "render_text_block",
    "RunDirectory",
    "RunResultShortcutSummary",
    "RunSlurmLogLinkSummary",
    "RunIndexSummary",
    "RunIdentity",
    "ManagedStageRun",
    "ConsoleProgressReporter",
    "ProgressTracker",
    "ProgressUpdate",
    "WandbLogger",
    "build_wandb_run_config",
    "curriculum_tags",
    "default_wandb_group",
    "flatten_wandb_config",
    "merge_tags",
    "managed_stage_run",
    "stage_tags",
    "study_tags",
    "sweep_tags",
    "capture_git_state",
    "generate_signature",
    "generate_variant_slug",
    "make_run_id",
    "rebuild_stage_run_index",
    "rebuild_variant_run_index",
    "repair_result_shortcut_links",
    "repair_slurm_log_links",
    "write_stage_run_link",
    "write_variant_run_link",
]
