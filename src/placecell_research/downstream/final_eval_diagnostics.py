"""Final deterministic eval diagnostics lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import EvaluationObserver
from .evaluation_observers import (
    PolicyLayerRecorderObserver,
    RGBInputRecorderObserver,
    TrajectoryRecorderObserver,
)
from .policy_layer_maps import PolicyLayerActivationRecorder
from .trajectory_logging import RGBInputRecorder, TrainingTrajectoryRecorder


@dataclass(slots=True)
class FinalEvalDiagnostics:
    label: str
    trajectory_recorder: TrainingTrajectoryRecorder
    rgb_input_recorder: RGBInputRecorder
    policy_layer_recorder: PolicyLayerActivationRecorder | None
    output_dir: Path

    @classmethod
    def create(
        cls,
        *,
        model: Any,
        output_dir: Path,
        env_id: str,
        num_envs: int,
        max_episodes: int,
        label: str = "final",
    ) -> FinalEvalDiagnostics:
        max_eval_episodes = max(1, int(max_episodes))
        trajectory_recorder = TrainingTrajectoryRecorder(
            output_dir=output_dir / "eval_trajectory_diagnostics" / label,
            env_id=env_id,
            num_envs=max(1, int(num_envs)),
            max_rendered_trajectories=max_eval_episodes,
            max_recorded_episodes=max_eval_episodes,
            file_prefix="eval",
            write_gif=True,
            individual_trajectory_gif_count=max_eval_episodes,
        )
        rgb_input_recorder = RGBInputRecorder(
            output_dir=output_dir / "eval_rgb_input_diagnostics" / label,
            max_recorded_episodes=max_eval_episodes,
        )
        policy_layer_recorder = PolicyLayerActivationRecorder(
            model=model,
            output_dir=output_dir / "eval_policy_layer_diagnostics" / label,
            env_id=env_id,
            max_episodes=max_episodes,
        )
        return cls(
            label=label,
            trajectory_recorder=trajectory_recorder,
            rgb_input_recorder=rgb_input_recorder,
            policy_layer_recorder=policy_layer_recorder if policy_layer_recorder.enabled else None,
            output_dir=output_dir,
        )

    def observers(self) -> list[EvaluationObserver]:
        observers: list[EvaluationObserver] = [
            TrajectoryRecorderObserver(self.trajectory_recorder),
            RGBInputRecorderObserver(self.rgb_input_recorder),
        ]
        if self.policy_layer_recorder is not None:
            observers.append(PolicyLayerRecorderObserver(self.policy_layer_recorder))
        return observers

    def close(self) -> dict[str, Any]:
        self.trajectory_recorder.close()
        self.rgb_input_recorder.close()
        policy_layer_diagnostics = None
        if self.policy_layer_recorder is not None:
            policy_layer_diagnostics = self.policy_layer_recorder.close()
        return {
            "final_eval_trajectory_diagnostics": self.trajectory_payload(),
            "final_eval_rgb_input_diagnostics": self.rgb_input_payload(),
            "final_policy_layer_diagnostics": policy_layer_diagnostics,
        }

    def trajectory_payload(self) -> dict[str, str | None]:
        diagnostics_dir = self.output_dir / "eval_trajectory_diagnostics" / self.label
        jsonl_path = diagnostics_dir / "eval_trajectories.jsonl"
        start_heatmap_path = diagnostics_dir / "eval_start_position_heatmap.png"
        start_outcomes_path = diagnostics_dir / "eval_start_position_outcomes.png"
        trajectories_path = diagnostics_dir / "eval_trajectories.png"
        gallery_gif_path = diagnostics_dir / "eval_trajectory_gallery.gif"
        individual_gif_paths = _sorted_numbered_gifs(diagnostics_dir, "eval_trajectory_")
        return {
            "eval_trajectory_jsonl_path": str(jsonl_path) if jsonl_path.exists() else None,
            "eval_start_position_heatmap_path": str(start_heatmap_path)
            if start_heatmap_path.exists()
            else None,
            "eval_start_position_outcomes_path": str(start_outcomes_path)
            if start_outcomes_path.exists()
            else None,
            "eval_trajectories_path": str(trajectories_path)
            if trajectories_path.exists()
            else None,
            "eval_trajectory_gallery_gif_path": str(gallery_gif_path)
            if gallery_gif_path.exists()
            else None,
            "eval_trajectory_gif_paths": [str(path) for path in individual_gif_paths],
        }

    def rgb_input_payload(self) -> dict[str, list[str]]:
        diagnostics_dir = self.output_dir / "eval_rgb_input_diagnostics" / self.label
        gif_paths = _sorted_numbered_gifs(diagnostics_dir, "eval_rgb_input_")
        return {"eval_rgb_input_gif_paths": [str(path) for path in gif_paths]}


def _sorted_numbered_gifs(directory: Path, prefix: str) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.glob(f"{prefix}*.gif")
            if path.stem.removeprefix(prefix).isdigit()
        ),
        key=lambda path: int(path.stem.removeprefix(prefix)),
    )
