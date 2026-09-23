from __future__ import annotations

import fcntl
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from placecell_research.config import artifact_match_fingerprint
from placecell_research.config.schema import ExperimentConfig
from placecell_research.spatial_model import select_place_model_checkpoint
from placecell_research.stages.train_place_model import (
    _place_model_resume_fingerprint_payload,
)
from placecell_research.training.checkpointing import (
    capture_rng_state,
    claim_latest_compatible_recovery_checkpoint,
    find_latest_compatible_recovery_checkpoint,
    hold_recovery_checkpoint_lock,
    load_checkpoint,
    save_checkpoint,
)


def _checkpoint_components():
    model = nn.Linear(3, 2)
    auxiliary_heads = nn.ModuleDict({"projection": nn.Linear(2, 2)})
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *auxiliary_heads.parameters()],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return model, auxiliary_heads, optimizer, scheduler


def test_checkpoint_round_trip_restores_random_state(tmp_path: Path) -> None:
    model, auxiliary_heads, optimizer, scheduler = _checkpoint_components()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    checkpoint_path = tmp_path / "weights_last.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        auxiliary_heads,
        optimizer,
        scheduler,
        epoch=4,
        step=17,
        metrics={"loss": 0.5},
        loop_state={"best_primary": 0.5},
        data_loader_state={"generator_state": torch.Generator().manual_seed(14).get_state()},
    )
    expected_random_values = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    loaded = load_checkpoint(
        checkpoint_path,
        model,
        auxiliary_heads,
        optimizer,
        scheduler,
        restore_random_state=True,
    )

    assert loaded.epoch == 4
    assert loaded.step == 17
    assert loaded.loop_state == {"best_primary": 0.5}
    assert random.random() == expected_random_values[0]
    assert float(np.random.random()) == expected_random_values[1]
    torch.testing.assert_close(torch.rand(3), expected_random_values[2])
    assert not (tmp_path / ".weights_last.pt.tmp").exists()


def test_full_rng_snapshot_includes_mps_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.tensor([7], dtype=torch.uint8))

    state = capture_rng_state()

    torch.testing.assert_close(state["mps"], torch.tensor([7], dtype=torch.uint8))


def test_failed_checkpoint_write_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, auxiliary_heads, optimizer, scheduler = _checkpoint_components()
    checkpoint_path = tmp_path / "weights_last.pt"
    checkpoint_path.write_bytes(b"previous checkpoint")

    def fail_after_partial_write(_payload, path: Path) -> None:
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("simulated preemption")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        save_checkpoint(
            checkpoint_path,
            model,
            auxiliary_heads,
            optimizer,
            scheduler,
            epoch=0,
            step=1,
            metrics={},
        )

    assert checkpoint_path.read_bytes() == b"previous checkpoint"
    assert not (tmp_path / ".weights_last.pt.tmp").exists()


def test_full_artifact_resume_prefers_last_checkpoint(tmp_path: Path) -> None:
    best_path = tmp_path / "weights_best_primary.pt"
    last_path = tmp_path / "weights_last.pt"
    best_path.touch()
    last_path.touch()

    assert select_place_model_checkpoint(tmp_path, selection="best") == best_path
    assert select_place_model_checkpoint(tmp_path, selection="last") == last_path


def test_artifact_checkpoint_falls_back_to_last_when_best_is_disabled(tmp_path: Path) -> None:
    last_path = tmp_path / "weights_last.pt"
    last_path.touch()

    assert select_place_model_checkpoint(tmp_path, selection="best") == last_path


def test_last_checkpoint_fallback_matches_canonical_validation_preference(
    tmp_path: Path,
) -> None:
    validation_path = tmp_path / "weights_best_validation_loss.pt"
    validation_path.touch()
    (tmp_path / "weights_best_primary.pt").touch()

    assert select_place_model_checkpoint(tmp_path, selection="last") == validation_path


def test_recovery_fingerprint_includes_seed_contract() -> None:
    config = ExperimentConfig()
    raw_config = {
        "spatial_model": {"training": {"epochs": 2}},
        "seed": {"global_seed": 1, "training_seed": 2},
    }
    first = artifact_match_fingerprint(
        _place_model_resume_fingerprint_payload(config, raw_config)
    )
    raw_config["seed"] = {"global_seed": 3, "training_seed": 4}
    second = artifact_match_fingerprint(
        _place_model_resume_fingerprint_payload(config, raw_config)
    )

    assert first != second


def _write_recovery_candidate(
    run_root: Path,
    *,
    run_id: str,
    fingerprint: str,
    status: str = "initialized",
    modified_time_ns: int,
) -> Path:
    run_dir = run_root / "by_id" / run_id
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage_name": "train_place_model",
                "status": status,
                "summary": {"place_model_resume_fingerprint": fingerprint},
            }
        )
    )
    checkpoint_dir = run_dir / "results" / "place_model_training_checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / ".training.lock").touch()
    checkpoint_path = checkpoint_dir / "weights_last.pt"
    checkpoint_path.touch()
    os.utime(checkpoint_path, ns=(modified_time_ns, modified_time_ns))
    return checkpoint_path


def test_auto_resume_selects_latest_compatible_incomplete_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    older_path = _write_recovery_candidate(
        run_root,
        run_id="older",
        fingerprint="matching",
        modified_time_ns=100,
    )
    newest_path = _write_recovery_candidate(
        run_root,
        run_id="newest",
        fingerprint="matching",
        modified_time_ns=300,
    )
    _write_recovery_candidate(
        run_root,
        run_id="different_config",
        fingerprint="different",
        modified_time_ns=500,
    )
    _write_recovery_candidate(
        run_root,
        run_id="completed",
        fingerprint="matching",
        status="completed",
        modified_time_ns=600,
    )
    _write_recovery_candidate(
        run_root,
        run_id="current",
        fingerprint="matching",
        modified_time_ns=700,
    )

    candidate = find_latest_compatible_recovery_checkpoint(
        run_root,
        resume_fingerprint="matching",
        exclude_run_id="current",
    )

    assert candidate is not None
    assert candidate.path == newest_path
    assert candidate.path != older_path
    assert candidate.run_id == "newest"


@pytest.mark.parametrize("status", ["failed", "error"])
def test_auto_resume_does_not_recover_failed_runs(tmp_path: Path, status: str) -> None:
    run_root = tmp_path / "runs"
    _write_recovery_candidate(
        run_root,
        run_id=status,
        fingerprint="matching",
        status=status,
        modified_time_ns=300,
    )

    candidate = find_latest_compatible_recovery_checkpoint(
        run_root,
        resume_fingerprint="matching",
        exclude_run_id="current",
    )

    assert candidate is None


def test_auxiliary_head_mismatch_requires_explicit_weights_only_tolerance(
    tmp_path: Path,
) -> None:
    model, auxiliary_heads, optimizer, scheduler = _checkpoint_components()
    checkpoint_path = tmp_path / "weights_last.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        auxiliary_heads,
        optimizer,
        scheduler,
        epoch=0,
        step=1,
        metrics={},
    )
    changed_heads = nn.ModuleDict({"replacement": nn.Linear(2, 2)})

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        load_checkpoint(checkpoint_path, model, changed_heads)

    with pytest.warns(RuntimeWarning, match="fresh-initialized"):
        load_checkpoint(
            checkpoint_path,
            model,
            changed_heads,
            allow_auxiliary_mismatch=True,
        )


def test_weights_only_tolerance_skips_changed_auxiliary_head_shapes(
    tmp_path: Path,
) -> None:
    model, auxiliary_heads, optimizer, scheduler = _checkpoint_components()
    checkpoint_path = tmp_path / "weights_last.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        auxiliary_heads,
        optimizer,
        scheduler,
        epoch=0,
        step=1,
        metrics={},
    )
    changed_heads = nn.ModuleDict({"projection": nn.Linear(2, 3)})
    fresh_state = {
        key: value.detach().clone() for key, value in changed_heads.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(checkpoint_path, model, changed_heads)

    with pytest.warns(RuntimeWarning, match="incompatible shape"):
        load_checkpoint(
            checkpoint_path,
            model,
            changed_heads,
            allow_auxiliary_mismatch=True,
        )

    for key, value in changed_heads.state_dict().items():
        torch.testing.assert_close(value, fresh_state[key])


def test_auto_resume_skips_checkpoint_held_by_active_training_process(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    active_path = _write_recovery_candidate(
        run_root,
        run_id="active",
        fingerprint="matching",
        modified_time_ns=300,
    )
    inactive_path = _write_recovery_candidate(
        run_root,
        run_id="inactive",
        fingerprint="matching",
        modified_time_ns=200,
    )

    with hold_recovery_checkpoint_lock(active_path.parent):
        candidate = find_latest_compatible_recovery_checkpoint(
            run_root,
            resume_fingerprint="matching",
            exclude_run_id="current",
        )

    assert candidate is not None
    assert candidate.path == inactive_path


def test_auto_resume_claim_prevents_a_second_process_from_adopting_the_same_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    claimed_path = _write_recovery_candidate(
        run_root,
        run_id="interrupted",
        fingerprint="matching",
        modified_time_ns=300,
    )

    with claim_latest_compatible_recovery_checkpoint(
        run_root,
        resume_fingerprint="matching",
        exclude_run_id="first",
    ) as first_claim:
        assert first_claim is not None
        assert first_claim.path == claimed_path
        with claim_latest_compatible_recovery_checkpoint(
            run_root,
            resume_fingerprint="matching",
            exclude_run_id="second",
        ) as second_claim:
            assert second_claim is None


def test_auto_resume_claim_revalidates_manifest_after_acquiring_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "runs"
    checkpoint_path = _write_recovery_candidate(
        run_root,
        run_id="finished_during_discovery",
        fingerprint="matching",
        modified_time_ns=300,
    )
    manifest_path = (
        run_root
        / "by_id"
        / "finished_during_discovery"
        / "manifests"
        / "run_manifest.json"
    )
    real_flock = fcntl.flock
    manifest_completed = False

    def complete_before_lock(lock_handle, operation):
        nonlocal manifest_completed
        if operation & fcntl.LOCK_NB and not manifest_completed:
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "completed"
            manifest_path.write_text(json.dumps(manifest))
            manifest_completed = True
        return real_flock(lock_handle, operation)

    monkeypatch.setattr(fcntl, "flock", complete_before_lock)
    with claim_latest_compatible_recovery_checkpoint(
        run_root,
        resume_fingerprint="matching",
        exclude_run_id="current",
    ) as claim:
        assert claim is None
    assert checkpoint_path.is_file()
