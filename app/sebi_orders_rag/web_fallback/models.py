"""Typed models for controlled web-search fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WebSourceType = Literal["official_web", "general_web"]
WebSearchAnswerStatus = Literal["answered", "insufficient_context", "unavailable"]
SearchContextSize = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class WebSearchRequest:
    """One structured web-search request."""

    query: str
    instructions: str
    lookup_type: str
    source_type: WebSourceType
    allowed_domains: tuple[str, ...] = ()
    search_context_size: SearchContextSize = "medium"
    max_results: int | None = None


@dataclass(frozen=True)
class WebSearchSource:
    """One source returned by a web-search provider."""

    source_title: str
    source_url: str
    domain: str
    source_type: WebSourceType
    snippet: str | None = None
    record_key: str | None = None


@dataclass(frozen=True)
class WebSearchResult:
    """Normalized output from a web-search provider."""

    answer_status: WebSearchAnswerStatus
    answer_text: str
    sources: tuple[WebSearchSource, ...] = ()
    confidence: float = 0.0
    provider_name: str = "unavailable"
    lookup_type: str = "web_search"
    debug: dict[str, Any] = field(default_factory=dict)
