"""Progress phases shared by downstream trainers and the isolated-run watchdog."""

from __future__ import annotations

from collections.abc import Callable

ProgressReporter = Callable[[int], None]

TRAINING_PHASE = "training"
EVALUATION_PHASE = "evaluation"
POST_TRAINING_PHASE = "post_training"


def report_progress_phase(
    reporter: ProgressReporter | None,
    phase: str,
    timestep: int,
) -> None:
    """Report a phase when the reporter supports the isolated-run phase protocol."""
    if reporter is None:
        return
    phase_reporter = getattr(reporter, "report_phase", None)
    if callable(phase_reporter):
        phase_reporter(str(phase), int(timestep))


def report_progress_heartbeat(
    reporter: ProgressReporter | None,
    phase: str,
    timestep: int,
) -> None:
    """Report liveness without implying that the training timestep advanced."""
    if reporter is None:
        return
    heartbeat_reporter = getattr(reporter, "report_heartbeat", None)
    if callable(heartbeat_reporter):
        heartbeat_reporter(str(phase), int(timestep))
