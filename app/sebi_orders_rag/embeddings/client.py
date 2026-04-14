"""OpenAI embedding client wrapper for the SEBI Orders RAG module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import SebiOrdersRagSettings
from ..exceptions import ConfigurationError, MissingDependencyError


class OpenAIEmbeddingClient:
    """Provider-encapsulated embedding client with batched requests."""

    def __init__(self, settings: SebiOrdersRagSettings) -> None:
        api_key = (settings.openai_api_key or "").strip()
        if not api_key or api_key.upper() in {"YOUR_KEY", "YOUR_API_KEY"}:
            raise ConfigurationError(
                "SEBI_ORDERS_RAG_OPENAI_API_KEY must be set to a real API key "
                "for Phase 3 embedding and search."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local runtime
            raise MissingDependencyError(
                "openai is required for Phase 3 embeddings. "
                "Install the dependencies from requirements-sebi-orders-rag.txt."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 3,
            "timeout": 60.0,
        }
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**client_kwargs)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int,
        batch_size: int,
    ) -> list[tuple[float, ...]]:
        """Create embeddings for a sequence of non-empty texts."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        sanitized = [_sanitize_text(text) for text in texts]
        if not sanitized:
            return []

        results: list[tuple[float, ...] | None] = [None] * len(sanitized)
        for start in range(0, len(sanitized), batch_size):
            batch = sanitized[start : start + batch_size]
            response = self._client.embeddings.create(
                model=model,
                input=batch,
                dimensions=dimensions,
            )
            for item in response.data:
                results[start + item.index] = tuple(float(value) for value in item.embedding)

        missing_indexes = [index for index, value in enumerate(results) if value is None]
        if missing_indexes:
            raise RuntimeError(
                f"Embedding response was missing rows for batch indexes: {missing_indexes}"
            )

        return [value for value in results if value is not None]


def _sanitize_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("embedding text must not be empty")
    return cleaned
