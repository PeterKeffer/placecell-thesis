"""Structured W&B tag helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from hashlib import sha1

_MAX_TAG_LENGTH = 64


def _normalize_tag_value(value: object) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return normalized.strip("_") or "unknown"


def _structured_tag(label: str, value: object) -> str:
    prefix = f"{label}:"
    normalized_value = _normalize_tag_value(value)
    tag = f"{prefix}{normalized_value}"
    if len(tag) <= _MAX_TAG_LENGTH:
        return tag
    suffix = sha1(normalized_value.encode("utf-8")).hexdigest()[:8]
    max_value_length = max(1, _MAX_TAG_LENGTH - len(prefix) - len(suffix) - 1)
    shortened_value = normalized_value[:max_value_length].rstrip("._-") or "x"
    return f"{prefix}{shortened_value}_{suffix}"


def merge_tags(*tag_groups: Iterable[str]) -> list[str]:
    """Merge tag groups while preserving order and uniqueness."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in tag_groups:
        for tag in group:
            normalized = str(tag).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


def default_wandb_group(study_name: str | None, variant_slug: str) -> str:
    if study_name:
        return _structured_tag("study", study_name)
    return _structured_tag("variant", variant_slug)


def stage_tags(
    stage_name: str,
    env_id: str,
    *,
    base_tags: Iterable[str] = (),
    study_name: str | None = None,
    variant_name: str | None = None,
    variant_slug: str | None = None,
    seed: int | None = None,
) -> list[str]:
    structured_tags = [
        _structured_tag("stage", stage_name),
        _structured_tag("env", env_id),
    ]
    if study_name:
        structured_tags.append(_structured_tag("study", study_name))
    if variant_name:
        structured_tags.append(_structured_tag("variant", variant_name))
    if variant_slug:
        structured_tags.append(_structured_tag("variant_slug", variant_slug))
    if seed is not None:
        structured_tags.append(_structured_tag("seed", seed))
    return merge_tags(base_tags, structured_tags)


def study_tags(
    study_name: str,
    *,
    mode: str,
    base_tags: Iterable[str] = (),
) -> list[str]:
    return merge_tags(
        base_tags,
        [
            _structured_tag("study", study_name),
            _structured_tag("study_mode", mode),
        ],
    )


def sweep_tags(
    study_name: str,
    trial_index: int,
    seed: int | None,
    *,
    base_tags: Iterable[str] = (),
) -> list[str]:
    structured_tags = [
        _structured_tag("study", study_name),
        "study_mode:sweep",
        _structured_tag("trial", trial_index),
    ]
    if seed is not None:
        structured_tags.append(_structured_tag("seed", seed))
    return merge_tags(base_tags, structured_tags)


def curriculum_tags(
    study_name: str,
    phase_name: str,
    phase_index: int,
    dataset_alias: str,
    *,
    base_tags: Iterable[str] = (),
) -> list[str]:
    return merge_tags(
        base_tags,
        [
            _structured_tag("study", study_name),
            "study_mode:curriculum",
            _structured_tag("phase", f"{phase_index}_{_normalize_tag_value(phase_name)}"),
            _structured_tag("dataset", dataset_alias),
        ],
    )
