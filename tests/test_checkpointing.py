from __future__ import annotations

from pathlib import Path

import pytest
import torch

from placecell_research.config.schema import PolicyConfig
from placecell_research.spatial_model.loading import select_place_model_checkpoint
from placecell_research.training.checkpointing import load_checkpoint


def test_load_checkpoint_sets_weights_only_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 2)
    auxiliary_heads = torch.nn.ModuleDict()
    payload = {
        "epoch": 3,
        "step": 7,
        "metrics": {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "auxiliary_state_dict": auxiliary_heads.state_dict(),
        "scheduler_state_dict": None,
        "build_context": None,
    }
    captured_kwargs: dict[str, object] = {}

    def fake_load(path: Path, **kwargs):
        del path
        captured_kwargs.update(kwargs)
        return payload

    monkeypatch.setattr("placecell_research.training.checkpointing.torch.load", fake_load)

    checkpoint_state = load_checkpoint(tmp_path / "weights.pt", model, auxiliary_heads)

    assert checkpoint_state.step == 7
    assert captured_kwargs["weights_only"] is False


def _touch_checkpoints(directory: Path, names: list[str]) -> None:
    for name in names:
        (directory / name).write_text("x")


def test_checkpoint_policy_defaults_to_last() -> None:
    assert PolicyConfig().checkpoint_selection == "last"


def test_select_checkpoint_path_prefers_last_by_default(tmp_path: Path) -> None:
    _touch_checkpoints(
        tmp_path, ["weights_best_primary.pt", "weights_best_validation_loss.pt", "weights_last.pt"]
    )
    assert select_place_model_checkpoint(tmp_path).name == "weights_last.pt"


def test_select_checkpoint_path_last_skips_stale_best_primary(tmp_path: Path) -> None:
    _touch_checkpoints(
        tmp_path, ["weights_best_primary.pt", "weights_best_validation_loss.pt", "weights_last.pt"]
    )
    assert select_place_model_checkpoint(tmp_path, selection="last").name == "weights_last.pt"


def test_select_checkpoint_path_last_falls_back_when_last_absent(tmp_path: Path) -> None:
    _touch_checkpoints(tmp_path, ["weights_best_primary.pt"])
    assert (
        select_place_model_checkpoint(tmp_path, selection="last").name == "weights_best_primary.pt"
    )


def test_select_checkpoint_path_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        select_place_model_checkpoint(tmp_path)


def test_explicit_best_overrides_the_last_default(tmp_path: Path) -> None:
    _touch_checkpoints(tmp_path, ["weights_best_primary.pt", "weights_last.pt"])
    assert (
        select_place_model_checkpoint(tmp_path, selection="best").name == "weights_best_primary.pt"
    )


def test_checkpoint_selection_rejects_unknown_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint selection"):
        select_place_model_checkpoint(tmp_path, selection="newest")
