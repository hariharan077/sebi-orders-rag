"""Provider interfaces for current official information lookups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..schemas import ChatSessionStateRecord
from ..web_fallback.models import WebSearchSource
from ..web_fallback.ranking import extract_domain

CurrentInfoAnswerStatus = Literal["answered", "insufficient_context", "unavailable"]


@dataclass(frozen=True)
class CurrentInfoSource:
    """One official source page used for a current-information answer."""

    title: str
    url: str
    record_key: str
    domain: str | None = None
    source_type: Literal["official_web", "general_web", "structured", "corpus"] = "official_web"
    snippet: str | None = None


@dataclass(frozen=True)
class CurrentInfoResult:
    """Normalized result returned by a current-information provider."""

    answer_status: CurrentInfoAnswerStatus
    answer_text: str
    sources: tuple[CurrentInfoSource, ...] = ()
    confidence: float = 0.0
    provider_name: str = "unavailable"
    lookup_type: str = "unsupported"
    debug: dict[str, Any] = field(default_factory=dict)


class CurrentInfoProvider(ABC):
    """Abstract provider for live official current-fact lookups."""

    @abstractmethod
    def lookup(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult:
        """Return a current-fact answer for one user query."""

    def preview_internal_person_priority(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult | None:
        """Return a structured person-priority result when it should override broad routes."""

        del query, session_state
        return None


class UnavailableCurrentInfoProvider(CurrentInfoProvider):
    """Provider used when live current-fact lookup is disabled."""

    def __init__(self, *, reason: str) -> None:
        self._reason = reason

    def lookup(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult:
        return CurrentInfoResult(
            answer_status="unavailable",
            answer_text=self._reason,
            provider_name="unavailable",
            lookup_type="disabled",
            debug={"query": query, "provider_available": False},
        )


def normalize_current_info_sources(
    sources: tuple[CurrentInfoSource, ...],
    *,
    default_source_type: Literal["official_web", "general_web", "structured", "corpus"],
) -> tuple[CurrentInfoSource, ...]:
    """Fill stable source metadata required by the citation layer."""

    normalized: list[CurrentInfoSource] = []
    for source in sources:
        normalized.append(
            CurrentInfoSource(
                title=source.title,
                url=source.url,
                record_key=source.record_key,
                domain=source.domain or extract_domain(source.url),
                source_type=source.source_type or default_source_type,
                snippet=source.snippet,
            )
        )
    return tuple(normalized)


def web_sources_to_current_info_sources(
    sources: tuple[WebSearchSource, ...],
) -> tuple[CurrentInfoSource, ...]:
    """Convert web-search source objects into current-info source objects."""

    return tuple(
        CurrentInfoSource(
            title=source.source_title,
            url=source.source_url,
            record_key=source.record_key or f"{source.source_type}:{source.domain}",
            domain=source.domain,
            source_type=source.source_type,
            snippet=source.snippet,
        )
        for source in sources
    )
