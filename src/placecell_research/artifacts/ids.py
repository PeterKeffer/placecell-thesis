"""Artifact and run id helpers."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from placecell_research.utils.timing import utc_now

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Convert free text into a stable lowercase slug."""
    slug = _NON_ALNUM.sub("_", value.lower()).strip("_")
    return slug or "artifact"


def short_fingerprint(*parts: str, length: int = 6) -> str:
    """Generate a short stable hash."""
    digest = sha256("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def config_fingerprint(config_text: str) -> str:
    """Return a sha256-prefixed config fingerprint."""
    return f"sha256:{sha256(config_text.encode('utf-8')).hexdigest()}"


def generate_artifact_id(prefix: str, summary_name: str, run_id: str) -> str:
    """Generate a timestamped artifact id."""
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    fingerprint = short_fingerprint(run_id, summary_name)
    return f"{slugify(prefix)}_{slugify(summary_name)}_{timestamp}_{fingerprint}"


def relative_to(path: Path, root: Path) -> str:
    """Format a stable relative path for manifests."""
    return str(path.resolve().relative_to(root.resolve()))
