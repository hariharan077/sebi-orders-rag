"""File helpers for SEBI Orders RAG."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def discover_named_files(root: Path, file_name: str) -> list[Path]:
    """Return sorted absolute file paths beneath root for a single filename."""

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted(path.resolve(strict=False) for path in root.rglob(file_name) if path.is_file())


def iter_file_chunks(path: Path, chunk_size: int) -> Iterator[bytes]:
    """Yield file chunks without loading the full file into memory."""

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def read_text_file(path: Path) -> str:
    """Read UTF-8 text content from disk."""

    return path.read_text(encoding="utf-8")
