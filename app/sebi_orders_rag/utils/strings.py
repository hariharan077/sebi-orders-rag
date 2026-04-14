"""String helpers shared across SEBI Orders RAG modules."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")


def normalize_newlines(text: str) -> str:
    """Normalize mixed newline styles to LF."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_inline_whitespace(text: str) -> str:
    """Collapse repeated inline whitespace without touching newlines."""

    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


def sha256_hexdigest(text: str) -> str:
    """Return a stable SHA-256 hex digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def uppercase_ratio(text: str) -> float:
    """Return the ratio of uppercase alphabetic characters in the text."""

    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0

    uppercase_count = sum(1 for character in letters if character.isupper())
    return uppercase_count / len(letters)


def join_heading_path(parts: Iterable[str]) -> str | None:
    """Serialize a heading path tuple for persistence."""

    cleaned_parts = [part.strip() for part in parts if part and part.strip()]
    if not cleaned_parts:
        return None
    return " > ".join(cleaned_parts)


def split_heading_path(value: str | None) -> tuple[str, ...]:
    """Deserialize a persisted heading path string into its component parts."""

    if value is None:
        return ()
    cleaned_parts = [part.strip() for part in value.split(">") if part.strip()]
    return tuple(cleaned_parts)
