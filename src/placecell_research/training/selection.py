"""Checkpoint selection policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from placecell_research.config.schema import SelectionPolicyConfig


def _is_better(mode: str, new_value: float, old_value: float) -> bool:
    return new_value < old_value if mode == "min" else new_value > old_value


def _is_tied(new_value: float, old_value: float) -> bool:
    return math.isclose(new_value, old_value, rel_tol=1e-12, abs_tol=1e-12)


@dataclass
class CheckpointSelector:
    config: SelectionPolicyConfig
    best_primary: float | None = None
    best_primary_tie_break: float | None = None
    best_validation_loss: float | None = None

    def update(self, metrics: dict[str, float]) -> dict[str, bool]:
        save_primary = False
        save_validation_loss = False
        primary_value = metrics.get(
            self.config.primary_metric,
            metrics.get("validation.total_loss", metrics.get("loss/total")),
        )
        if primary_value is None:
            raise KeyError(
                f"Missing primary selection metric '{self.config.primary_metric}' and no fallback "
                "total loss."
            )
        validation_loss = metrics.get(
            "validation.total_loss", metrics.get("loss/total", primary_value)
        )
        tie_break_value = metrics.get(self.config.tie_break_metric, validation_loss)
        if self.best_primary is None:
            self.best_primary = primary_value
            self.best_primary_tie_break = tie_break_value
            save_primary = self.config.save_best_primary
        elif _is_better(self.config.primary_mode, primary_value, self.best_primary):
            self.best_primary = primary_value
            self.best_primary_tie_break = tie_break_value
            save_primary = self.config.save_best_primary
        elif _is_tied(primary_value, self.best_primary):
            reference_tie_break = self.best_primary_tie_break
            if reference_tie_break is None or _is_better(
                self.config.tie_break_mode, tie_break_value, reference_tie_break
            ):
                self.best_primary = primary_value
                self.best_primary_tie_break = tie_break_value
                save_primary = self.config.save_best_primary
        if self.best_validation_loss is None or _is_better(
            "min", validation_loss, self.best_validation_loss
        ):
            self.best_validation_loss = validation_loss
            save_validation_loss = self.config.save_best_validation_loss
        return {
            "save_best_primary": save_primary,
            "save_best_validation_loss": save_validation_loss,
            "save_last": self.config.save_last,
        }
