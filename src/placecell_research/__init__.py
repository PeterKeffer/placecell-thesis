"""PlaceCell Research package."""

from __future__ import annotations

import os

_CODE_SNAPSHOT_ROOT = os.environ.get("PLACECELL_CODE_SNAPSHOT", "").strip()
if _CODE_SNAPSHOT_ROOT:
    _EXPECTED_PACKAGE_PREFIX = os.path.join(_CODE_SNAPSHOT_ROOT, "src", "")
    if not os.path.abspath(__file__).startswith(_EXPECTED_PACKAGE_PREFIX):
        raise ImportError(
            f"PLACECELL_CODE_SNAPSHOT={_CODE_SNAPSHOT_ROOT} but placecell_research was "
            f"imported from {os.path.abspath(__file__)}. The job would run code the snapshot "
            "does not pin. Check that PYTHONPATH still leads with the snapshot's src/."
        )
