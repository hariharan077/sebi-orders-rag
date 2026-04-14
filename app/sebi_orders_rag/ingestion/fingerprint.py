"""Local file fingerprint helpers for Phase 1."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..constants import HASH_CHUNK_SIZE_BYTES
from ..schemas import FileFingerprint, LocalFileSnapshot
from ..utils.files import iter_file_chunks


def inspect_local_file(path: Path) -> LocalFileSnapshot:
    """Check local file availability and compute fingerprint metadata when present."""

    resolved_path = path.resolve(strict=False)
    if not resolved_path.exists() or not resolved_path.is_file():
        return LocalFileSnapshot(path=resolved_path, exists=False, fingerprint=None)

    return LocalFileSnapshot(
        path=resolved_path,
        exists=True,
        fingerprint=compute_file_fingerprint(resolved_path),
    )


def compute_file_fingerprint(path: Path) -> FileFingerprint:
    """Compute the byte size and SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    file_size_bytes = 0
    for chunk in iter_file_chunks(path, HASH_CHUNK_SIZE_BYTES):
        digest.update(chunk)
        file_size_bytes += len(chunk)
    return FileFingerprint(
        file_size_bytes=file_size_bytes,
        file_sha256=digest.hexdigest(),
    )
