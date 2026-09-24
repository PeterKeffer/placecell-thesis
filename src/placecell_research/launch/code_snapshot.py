"""Per-job code snapshots for the remote SLURM submit path."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

SNAPSHOT_DIR_NAME = ".code_snapshots"
SNAPSHOT_ENV_VAR = "PLACECELL_CODE_SNAPSHOT"

SNAPSHOT_SOURCE_DIRS = ("src", "configs", "scripts/slurm")
SNAPSHOT_EXCLUDED_DIRS = ("__pycache__",)
SNAPSHOT_EXCLUDED_FILE_GLOBS = ("*.pyc", "*.pyo", ".DS_Store")

MAX_SNAPSHOT_FILES = 20_000
MAX_SNAPSHOT_BYTES = 50 * 1024 * 1024

SNAPSHOT_RETENTION_DAYS = 14
SNAPSHOT_RETENTION_KEEP = 30

SNAPSHOT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
SNAPSHOT_DIR_PATTERN = r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$"

_OUTPUT_FIELD_PATTERN = re.compile(r"^\[code-snapshot\] ([a-z_]+)=(.*)$")


@dataclass(slots=True)
class CodeSnapshot:
    """One frozen copy of the importable code on the remote host."""

    path: str
    content_hash: str
    file_count: int
    total_bytes: int
    reused: bool

    @property
    def python_path(self) -> str:
        return f"{self.path}/src"

    def describe(self) -> str:
        verb = "reused" if self.reused else "created"
        megabytes = self.total_bytes / (1024 * 1024)
        return (
            f"{verb} {self.path} "
            f"(hash={self.content_hash}, files={self.file_count}, {megabytes:.1f} MB)"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "reused": self.reused,
        }


def snapshot_timestamp(moment: datetime | None = None) -> str:
    resolved = moment or datetime.now(UTC)
    return resolved.astimezone(UTC).strftime(SNAPSHOT_TIMESTAMP_FORMAT)


def retention_cutoff(moment: datetime | None = None) -> str:
    """Snapshot-name prefix below which a directory is prunable."""
    resolved = moment or datetime.now(UTC)
    return snapshot_timestamp(resolved - timedelta(days=SNAPSHOT_RETENTION_DAYS))


def _find_expression() -> str:
    parts = [*SNAPSHOT_SOURCE_DIRS, "-type", "f"]
    for excluded_dir in SNAPSHOT_EXCLUDED_DIRS:
        parts.extend(["!", "-path", shlex.quote(f"*/{excluded_dir}/*")])
    for glob in SNAPSHOT_EXCLUDED_FILE_GLOBS:
        parts.extend(["!", "-name", shlex.quote(glob)])
    return " ".join(parts)


def _rsync_exclude_flags() -> str:
    patterns = [f"{name}/" for name in SNAPSHOT_EXCLUDED_DIRS]
    patterns.extend(SNAPSHOT_EXCLUDED_FILE_GLOBS)
    return " ".join(f"--exclude {shlex.quote(pattern)}" for pattern in patterns)


def _rsync_source_args() -> str:
    return " ".join(f"./{directory}" for directory in SNAPSHOT_SOURCE_DIRS)


def build_code_snapshot_script(
    *,
    remote_repo_root: str | PurePosixPath,
    timestamp: str,
    cutoff: str,
) -> str:
    """Render the remote bash that measures, snapshots, verifies and prunes."""
    quoted_repo_root = shlex.quote(str(remote_repo_root))
    file_list = f"find {_find_expression()} -print0 | LC_ALL=C sort -z"
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {quoted_repo_root}",
            f"snapshot_root={quoted_repo_root}/{SNAPSHOT_DIR_NAME}",
            'mkdir -p "${snapshot_root}"',
            f"manifest=$({file_list} | xargs -0 -r sha256sum)",
            "file_count=$(printf '%s\\n' \"${manifest}\" | grep -c . || true)",
            f"total_bytes=$({file_list} | xargs -0 -r cat | wc -c | tr -d ' ')",
            f'if [[ "${{file_count}}" -gt {MAX_SNAPSHOT_FILES} ]]; then',
            (
                '  echo "[code-snapshot] refusing: ${file_count} files exceeds the '
                f'{MAX_SNAPSHOT_FILES} file budget" >&2'
            ),
            "  exit 1",
            "fi",
            f'if [[ "${{total_bytes}}" -gt {MAX_SNAPSHOT_BYTES} ]]; then',
            (
                '  echo "[code-snapshot] refusing: ${total_bytes} bytes exceeds the '
                f'{MAX_SNAPSHOT_BYTES} byte budget" >&2'
            ),
            "  exit 1",
            "fi",
            'if [[ "${file_count}" -eq 0 ]]; then',
            '  echo "[code-snapshot] refusing: no files selected" >&2',
            "  exit 1",
            "fi",
            ("content_hash=$(printf '%s\\n' \"${manifest}\" | sha256sum | cut -c1-8)"),
            'existing=$(ls -1d "${snapshot_root}"/*_"${content_hash}" 2>/dev/null '
            "| LC_ALL=C sort | tail -n 1 || true)",
            'if [[ -n "${existing}" && -f "${existing}/.snapshot_hash" ]]; then',
            '  snapshot_dir="${existing%/}"',
            "  reused=1",
            "else",
            f'  snapshot_dir="${{snapshot_root}}/{timestamp}_${{content_hash}}"',
            "  reused=0",
            '  link_dest=$(ls -1d "${snapshot_root}"/*/ 2>/dev/null '
            "| LC_ALL=C sort | tail -n 1 || true)",
            '  mkdir -p "${snapshot_dir}"',
            (
                f"  rsync -a --checksum -R {_rsync_exclude_flags()} "
                '${link_dest:+--link-dest="${link_dest%/}"} '
                f"{_rsync_source_args()} "
                '"${snapshot_dir}/"'
            ),
            '  if [[ -e "${snapshot_dir}/pyproject.toml" ]]; then',
            (
                '    echo "[code-snapshot] refusing: snapshot contains pyproject.toml, which '
                'would redirect the artifact root" >&2'
            ),
            "    exit 1",
            "  fi",
            f'  copied_manifest=$(cd "${{snapshot_dir}}" && {file_list} | xargs -0 -r sha256sum)',
            '  if [[ "${copied_manifest}" != "${manifest}" ]]; then',
            (
                '    echo "[code-snapshot] refusing: ${snapshot_dir} does not match the source '
                'it was copied from" >&2'
            ),
            "    exit 1",
            "  fi",
            '  printf \'%s\\n\' "${manifest}" > "${snapshot_dir}/.snapshot_manifest"',
            '  printf \'%s\\n\' "${content_hash}" > "${snapshot_dir}/.snapshot_hash"',
            "fi",
            'echo "[code-snapshot] path=${snapshot_dir}"',
            'echo "[code-snapshot] hash=${content_hash}"',
            'echo "[code-snapshot] files=${file_count}"',
            'echo "[code-snapshot] bytes=${total_bytes}"',
            'echo "[code-snapshot] reused=${reused}"',
            "kept=0",
            "while read -r candidate; do",
            '  [[ -n "${candidate}" ]] || continue',
            '  candidate="${candidate%/}"',
            '  name="${candidate##*/}"',
            f'  [[ "${{name}}" =~ {SNAPSHOT_DIR_PATTERN} ]] || continue',
            '  [[ "${snapshot_root}/${name}" != "${snapshot_dir}" ]] || continue',
            "  kept=$((kept + 1))",
            f'  [[ "${{kept}}" -gt {SNAPSHOT_RETENTION_KEEP} ]] || continue',
            f'  [[ "${{name%%_*}}" < "{cutoff}" ]] || continue',
            '  rm -rf -- "${snapshot_root}/${name}"',
            '  echo "[code-snapshot] pruned=${name}"',
            'done < <(ls -1d "${snapshot_root}"/*/ 2>/dev/null | LC_ALL=C sort -r || true)',
        ]
    )


def parse_code_snapshot_output(output: str) -> CodeSnapshot:
    """Read the [code-snapshot] key=value lines out of a noisy remote shell."""
    fields: dict[str, str] = {}
    for line in output.splitlines():
        match = _OUTPUT_FIELD_PATTERN.match(line.strip())
        if match is not None:
            fields[match.group(1)] = match.group(2).strip()
    missing = [name for name in ("path", "hash", "files", "bytes", "reused") if name not in fields]
    if missing:
        raise RuntimeError(
            "Remote code snapshot did not report "
            f"{', '.join(missing)}:\n{output.strip() or '<empty output>'}"
        )
    return CodeSnapshot(
        path=fields["path"],
        content_hash=fields["hash"],
        file_count=int(fields["files"]),
        total_bytes=int(fields["bytes"]),
        reused=fields["reused"] == "1",
    )


def rewrite_config_path_for_snapshot(
    *,
    remote_config_path: PurePosixPath,
    remote_repo_root: PurePosixPath,
    snapshot_path: str,
) -> PurePosixPath:
    """Point a mirrored config path at the snapshot copy of the same file."""
    relative_path = remote_config_path.relative_to(remote_repo_root)
    if relative_path.parts[0] != "configs":
        raise ValueError(
            "Code snapshots only cover configs/; cannot resolve a snapshot path for "
            f"{remote_config_path}."
        )
    return PurePosixPath(snapshot_path).joinpath(*relative_path.parts)
