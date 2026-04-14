"""Token counting helpers shared by Phase 2 extraction and later phases."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..exceptions import MissingDependencyError


def _import_tiktoken() -> Any:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise MissingDependencyError(
            "tiktoken is required for Phase 2 token counting. "
            "Install the dependencies from requirements-sebi-orders-rag.txt."
        ) from exc
    return tiktoken


@lru_cache(maxsize=8)
def _get_encoding(model_name: str) -> Any:
    tiktoken = _import_tiktoken()
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def token_count(text: str, *, model_name: str) -> int:
    """Return the token count for text under the configured model encoding."""

    if not text:
        return 0
    encoding = _get_encoding(model_name)
    return len(encoding.encode(text, disallowed_special=()))


def take_last_tokens(text: str, *, model_name: str, max_tokens: int) -> str:
    """Return the last max_tokens from text, decoded back into a string."""

    if max_tokens <= 0 or not text:
        return ""

    encoding = _get_encoding(model_name)
    encoded = encoding.encode(text, disallowed_special=())
    if len(encoded) <= max_tokens:
        return text
    return encoding.decode(encoded[-max_tokens:])


def split_token_windows(
    text: str,
    *,
    model_name: str,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """Split text into token windows for oversized fallback cases."""

    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    encoding = _get_encoding(model_name)
    encoded = encoding.encode(text, disallowed_special=())
    if len(encoded) <= max_tokens:
        return [text]

    windows: list[str] = []
    step = max_tokens - overlap_tokens
    start = 0
    while start < len(encoded):
        stop = min(start + max_tokens, len(encoded))
        windows.append(encoding.decode(encoded[start:stop]).strip())
        if stop >= len(encoded):
            break
        start += step
    return [window for window in windows if window]
