"""Checkpoint I/O."""

from __future__ import annotations

import fcntl
import json
import os
import random
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class CheckpointState:
    epoch: int
    step: int
    metrics: dict[str, float]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    auxiliary_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None = None
    optimizer_parameter_identity: list[dict[str, Any]] | None = None
    build_context: dict[str, Any] | None = None
    rng_state: dict[str, Any] | None = None
    loop_state: dict[str, Any] | None = None
    data_loader_state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RecoveryCheckpoint:
    path: Path
    run_id: str
    checkpoint_modified_time_ns: int


_RECOVERY_LOCK_FILE_NAME = ".training.lock"
_INCOMPLETE_RUN_STATUSES = {
    "initialized",
    "running",
    "submitted",
    "queued",
    "interrupted",
    "cancelled",
    "canceled",
    "timeout",
}


@contextmanager
def hold_recovery_checkpoint_lock(checkpoint_dir: Path) -> Iterator[None]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (checkpoint_dir / _RECOVERY_LOCK_FILE_NAME).open("a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def _compatible_recovery_checkpoints(
    run_root: Path,
    *,
    resume_fingerprint: str,
    exclude_run_id: str,
) -> list[RecoveryCheckpoint]:
    """Return compatible candidates without making a non-atomic availability decision."""
    runs_by_id = run_root / "by_id"
    if not runs_by_id.is_dir():
        return []
    candidates: list[RecoveryCheckpoint] = []
    for run_dir in runs_by_id.iterdir():
        if not run_dir.is_dir() or run_dir.name == exclude_run_id:
            continue
        manifest_path = run_dir / "manifests" / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if manifest.get("stage_name") != "train_place_model":
            continue
        if str(manifest.get("status", "")).strip().lower() not in _INCOMPLETE_RUN_STATUSES:
            continue
        summary = manifest.get("summary")
        if not isinstance(summary, dict):
            continue
        if summary.get("place_model_resume_fingerprint") != resume_fingerprint:
            continue
        checkpoint_path = (
            run_dir / "results" / "place_model_training_checkpoints" / "weights_last.pt"
        )
        if not checkpoint_path.is_file():
            continue
        candidates.append(
            RecoveryCheckpoint(
                path=checkpoint_path,
                run_id=run_dir.name,
                checkpoint_modified_time_ns=checkpoint_path.stat().st_mtime_ns,
            )
        )
    return sorted(
        candidates,
        key=lambda candidate: (candidate.checkpoint_modified_time_ns, candidate.run_id),
        reverse=True,
    )


@contextmanager
def claim_latest_compatible_recovery_checkpoint(
    run_root: Path,
    *,
    resume_fingerprint: str,
    exclude_run_id: str,
) -> Iterator[RecoveryCheckpoint | None]:
    """Atomically claim one recovery source and hold its lock until the caller finishes."""
    for candidate in _compatible_recovery_checkpoints(
        run_root,
        resume_fingerprint=resume_fingerprint,
        exclude_run_id=exclude_run_id,
    ):
        lock_path = candidate.path.parent / _RECOVERY_LOCK_FILE_NAME
        try:
            lock_handle = lock_path.open("a+")
        except OSError:
            continue
        try:
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            refreshed_candidates = _compatible_recovery_checkpoints(
                run_root,
                resume_fingerprint=resume_fingerprint,
                exclude_run_id=exclude_run_id,
            )
            refreshed = next(
                (item for item in refreshed_candidates if item.run_id == candidate.run_id),
                None,
            )
            if refreshed is None:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                continue
            try:
                yield refreshed
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
            return
        finally:
            lock_handle.close()
    yield None


def capture_torch_random_state() -> dict[str, Any]:
    """Capture every PyTorch RNG stream available to this process."""
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_torch_random_state(state: dict[str, Any]) -> None:
    """Restore streams captured by capture_torch_random_state."""
    torch.random.set_rng_state(state["cpu"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([device_state.cpu() for device_state in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def capture_rng_state() -> dict[str, Any]:
    torch_state = capture_torch_random_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch_state.pop("cpu"),
        **torch_state,
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    restore_torch_random_state(
        {
            "cpu": state["torch"],
            **{key: state[key] for key in ("cuda", "mps") if key in state},
        }
    )


def _optimizer_group_layout(param_groups: list[dict[str, Any]]) -> list[tuple[Any, int]]:
    return [(group.get("name"), len(group["params"])) for group in param_groups]


def _optimizer_groups_by_name(
    param_groups: list[dict[str, Any]], *, origin: str
) -> dict[str, dict[str, Any]]:
    groups_by_name: dict[str, dict[str, Any]] = {}
    for group in param_groups:
        name = group.get("name")
        if not isinstance(name, str):
            raise ValueError(
                f"Cannot migrate optimizer state: a {origin} parameter group carries no name, "
                "so groups cannot be aligned across the change."
            )
        if name in groups_by_name:
            raise ValueError(
                f"Cannot migrate optimizer state: the {origin} has two parameter groups named "
                f"'{name}', so the alignment is ambiguous."
            )
        groups_by_name[name] = group
    return groups_by_name


def load_optimizer_state_dict(
    optimizer: Optimizer, checkpoint_optimizer_state: dict[str, Any]
) -> list[str]:
    """Load optimizer state, migrating across a changed set of parameter groups."""
    current_state = optimizer.state_dict()
    current_groups = current_state["param_groups"]
    checkpoint_groups = checkpoint_optimizer_state["param_groups"]
    if _optimizer_group_layout(current_groups) == _optimizer_group_layout(checkpoint_groups):
        optimizer.load_state_dict(checkpoint_optimizer_state)
        return []

    checkpoint_by_name = _optimizer_groups_by_name(checkpoint_groups, origin="checkpoint")
    current_by_name = _optimizer_groups_by_name(current_groups, origin="optimizer")
    checkpoint_state = checkpoint_optimizer_state["state"]
    migrated_state: dict[Any, Any] = {}
    migrated_groups: list[dict[str, Any]] = []
    fresh_group_names: list[str] = []
    for current_group in current_groups:
        name = current_group["name"]
        checkpoint_group = checkpoint_by_name.get(name)
        if checkpoint_group is None:
            fresh_group_names.append(name)
            migrated_groups.append(dict(current_group))
            continue
        if len(checkpoint_group["params"]) != len(current_group["params"]):
            raise ValueError(
                f"Cannot migrate optimizer state: parameter group '{name}' holds "
                f"{len(current_group['params'])} parameter(s) but the checkpoint recorded "
                f"{len(checkpoint_group['params'])}, so its per-parameter moment estimates cannot "
                "be matched. Resume this checkpoint with training_resume=weights_only to "
                "warm-start from the weights with a fresh optimizer."
            )
        for current_index, checkpoint_index in zip(
            current_group["params"], checkpoint_group["params"], strict=False
        ):
            if checkpoint_index in checkpoint_state:
                migrated_state[current_index] = checkpoint_state[checkpoint_index]
        migrated_group = dict(checkpoint_group)
        migrated_group["params"] = list(current_group["params"])
        migrated_groups.append(migrated_group)

    optimizer.load_state_dict({"state": migrated_state, "param_groups": migrated_groups})

    if fresh_group_names:
        warnings.warn(
            f"[checkpoint] optimizer parameter group(s) {fresh_group_names} are absent from the "
            "resumed checkpoint and start from a fresh optimizer state (no moment estimates, no "
            "step count).",
            RuntimeWarning,
            stacklevel=2,
        )
    dropped_group_names = sorted(set(checkpoint_by_name) - set(current_by_name))
    if dropped_group_names:
        warnings.warn(
            f"[checkpoint] optimizer state for parameter group(s) {dropped_group_names} is in the "
            "checkpoint but not in this model, and was discarded.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fresh_group_names


def optimizer_parameter_identity(
    optimizer: Optimizer,
    model: nn.Module,
    auxiliary_heads: nn.ModuleDict,
) -> list[dict[str, Any]]:
    """Identity of every optimizer parameter, in torch's flat state-key order."""
    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    names_by_identity.update(
        (id(parameter), f"auxiliary_heads.{name}")
        for name, parameter in auxiliary_heads.named_parameters()
    )
    identity: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names_by_identity.get(id(parameter))
            if name is None:
                raise ValueError(
                    "Cannot record optimizer parameter identity: the optimizer holds a parameter "
                    "that belongs to neither the model nor the auxiliary heads."
                )
            identity.append(
                {
                    "name": name,
                    "shape": tuple(int(size) for size in parameter.shape),
                    "dtype": str(parameter.dtype),
                    "group": group.get("name"),
                }
            )
    return identity


def _migrate_optimizer_state_by_name(
    optimizer: Optimizer,
    checkpoint_optimizer_state: dict[str, Any],
    checkpoint_identity: list[dict[str, Any]],
    current_identity: list[dict[str, Any]],
) -> list[str]:
    """Load optimizer state matched on parameter NAME."""
    checkpoint_entries = {
        entry["name"]: (index, entry) for index, entry in enumerate(checkpoint_identity)
    }
    if len(checkpoint_entries) != len(checkpoint_identity):
        raise ValueError(
            "Cannot migrate optimizer state: the checkpoint records a parameter name twice, so "
            "its moment estimates cannot be matched by name."
        )
    checkpoint_state = checkpoint_optimizer_state["state"]
    checkpoint_groups_by_name = {
        group.get("name"): group for group in checkpoint_optimizer_state["param_groups"]
    }
    migrated_state: dict[Any, Any] = {}
    fresh_names: list[str] = []
    mismatched: list[str] = []
    unmatched_names: list[str] = []
    for current_index, entry in enumerate(current_identity):
        matched = checkpoint_entries.get(entry["name"])
        if matched is None:
            if entry["group"] in checkpoint_groups_by_name:
                unmatched_names.append(entry["name"])
            else:
                fresh_names.append(entry["name"])
            continue
        checkpoint_index, checkpoint_entry = matched
        if (
            tuple(checkpoint_entry["shape"]) != tuple(entry["shape"])
            or checkpoint_entry["dtype"] != entry["dtype"]
        ):
            mismatched.append(
                f"{entry['name']}: checkpoint={tuple(checkpoint_entry['shape'])}"
                f"/{checkpoint_entry['dtype']}, current={tuple(entry['shape'])}/{entry['dtype']}"
            )
            continue
        if checkpoint_index in checkpoint_state:
            migrated_state[current_index] = checkpoint_state[checkpoint_index]
    dropped_names = sorted(
        name
        for name, (_index, checkpoint_entry) in checkpoint_entries.items()
        if name not in {current["name"] for current in current_identity}
        and checkpoint_entry["group"] in {current["group"] for current in current_identity}
    )
    if unmatched_names or dropped_names:
        raise ValueError(
            "Cannot migrate optimizer state: parameter names in a group both sides hold do not "
            f"agree. Absent from the checkpoint: {sorted(unmatched_names)}. Absent from this "
            f"model: {dropped_names}. Resume with training_resume=weights_only to warm-start "
            "from the weights with a fresh optimizer."
        )
    if mismatched:
        raise ValueError(
            "Cannot migrate optimizer state: parameter(s) "
            f"{sorted(mismatched)} changed shape or dtype, so their moment estimates do not "
            "describe the parameter of the same name any more. Resume with "
            "training_resume=weights_only to warm-start from the weights alone."
        )
    migrated_groups: list[dict[str, Any]] = []
    fresh_group_names: list[str] = []
    for group in optimizer.state_dict()["param_groups"]:
        checkpoint_group = checkpoint_groups_by_name.get(group.get("name"))
        if checkpoint_group is None:
            fresh_group_names.append(group.get("name"))
            migrated_groups.append(dict(group))
            continue
        migrated_group = dict(checkpoint_group)
        migrated_group["params"] = list(group["params"])
        migrated_groups.append(migrated_group)
    optimizer.load_state_dict({"state": migrated_state, "param_groups": migrated_groups})
    if fresh_names:
        warnings.warn(
            f"[checkpoint] {len(fresh_names)} optimizer parameter(s) are absent from the resumed "
            f"checkpoint and start without moment estimates: {sorted(fresh_names)} "
            f"(parameter group(s) {sorted(name for name in fresh_group_names if name)}).",
            RuntimeWarning,
            stacklevel=2,
        )
    discarded = sorted(set(checkpoint_entries) - {entry["name"] for entry in current_identity})
    if discarded:
        warnings.warn(
            f"[checkpoint] optimizer state for {len(discarded)} parameter(s) of removed group(s) "
            f"was discarded: {discarded}",
            RuntimeWarning,
            stacklevel=2,
        )
    return [name for name in fresh_group_names if isinstance(name, str)]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    auxiliary_heads: nn.ModuleDict,
    optimizer: Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    build_context: dict[str, Any] | None = None,
    loop_state: dict[str, Any] | None = None,
    data_loader_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = CheckpointState(
        epoch=epoch,
        step=step,
        metrics=metrics,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        optimizer_parameter_identity=optimizer_parameter_identity(
            optimizer, model, auxiliary_heads
        ),
        auxiliary_state_dict=auxiliary_heads.state_dict(),
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        build_context=build_context,
        rng_state=capture_rng_state(),
        loop_state=loop_state,
        data_loader_state=data_loader_state,
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload.__dict__, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    auxiliary_heads: nn.ModuleDict,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    restore_random_state: bool = False,
    allow_auxiliary_mismatch: bool = False,
) -> CheckpointState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if optimizer is not None and payload.get("optimizer_parameter_identity") is None:
        raise ValueError(
            "Full optimizer resume requires optimizer_parameter_identity in the checkpoint. "
            "Use policies.training_resume=weights_only to start a new optimizer explicitly."
        )
    model.load_state_dict(payload["model_state_dict"])
    checkpoint_auxiliary_state = payload["auxiliary_state_dict"]
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[str] = []
    if not allow_auxiliary_mismatch:
        auxiliary_heads.load_state_dict(checkpoint_auxiliary_state, strict=True)
    else:
        current_auxiliary_state = auxiliary_heads.state_dict()
        compatible_auxiliary_state = {
            key: checkpoint_value
            for key, checkpoint_value in checkpoint_auxiliary_state.items()
            if key in current_auxiliary_state
            and current_auxiliary_state[key].shape == checkpoint_value.shape
        }
        missing_keys = sorted(current_auxiliary_state.keys() - checkpoint_auxiliary_state.keys())
        unexpected_keys = sorted(checkpoint_auxiliary_state.keys() - current_auxiliary_state.keys())
        shape_mismatches = sorted(
            key
            for key in checkpoint_auxiliary_state.keys() & current_auxiliary_state.keys()
            if checkpoint_auxiliary_state[key].shape != current_auxiliary_state[key].shape
        )
        auxiliary_heads.load_state_dict(compatible_auxiliary_state, strict=False)
    if allow_auxiliary_mismatch and missing_keys:
        warnings.warn(
            f"[checkpoint] {len(missing_keys)} auxiliary-head param(s) absent from "
            f"the resumed checkpoint were left fresh-initialized (new objectives): "
            f"{missing_keys}",
            RuntimeWarning,
            stacklevel=2,
        )
    if allow_auxiliary_mismatch and unexpected_keys:
        warnings.warn(
            f"[checkpoint] {len(unexpected_keys)} auxiliary-head param(s) in the "
            f"checkpoint are absent from the current model and were ignored: "
            f"{unexpected_keys}",
            RuntimeWarning,
            stacklevel=2,
        )
    if allow_auxiliary_mismatch and shape_mismatches:
        mismatch_descriptions = [
            (
                f"{key}: checkpoint={tuple(checkpoint_auxiliary_state[key].shape)}, "
                f"current={tuple(current_auxiliary_state[key].shape)}"
            )
            for key in shape_mismatches
        ]
        warnings.warn(
            f"[checkpoint] {len(shape_mismatches)} auxiliary-head param(s) have an incompatible "
            "shape and were left fresh-initialized: "
            f"{mismatch_descriptions}",
            RuntimeWarning,
            stacklevel=2,
        )
    if optimizer is not None:
        _migrate_optimizer_state_by_name(
            optimizer,
            payload["optimizer_state_dict"],
            payload["optimizer_parameter_identity"],
            optimizer_parameter_identity(optimizer, model, auxiliary_heads),
        )
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_random_state and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return CheckpointState(**payload)
