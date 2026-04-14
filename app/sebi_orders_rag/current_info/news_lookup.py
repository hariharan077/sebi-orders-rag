"""Current-news lookup provider for explicit official-news queries."""

from __future__ import annotations

from typing import Any, Callable

from ..answering.confidence import assess_web_fallback_confidence
from ..config import SebiOrdersRagSettings
from ..schemas import ChatSessionStateRecord
from ..web_fallback import WebSearchProvider, WebSearchRequest
from ..web_fallback.provider import build_official_web_search_provider
from .provider import (
    CurrentInfoProvider,
    CurrentInfoResult,
    CurrentInfoSource,
    web_sources_to_current_info_sources,
)

NewsLookupCallable = Callable[[str], tuple[str, tuple[CurrentInfoSource, ...], float, dict[str, Any]]]


class CurrentNewsLookupProvider(CurrentInfoProvider):
    """Route explicit SEBI current-news questions away from the orders corpus."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings | None = None,
        lookup_callable: NewsLookupCallable | None = None,
        official_search_provider: WebSearchProvider | None = None,
    ) -> None:
        self._lookup_callable = lookup_callable
        self._settings = settings
        self._official_search = official_search_provider or (
            build_official_web_search_provider(settings) if settings is not None else None
        )

    def lookup(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult:
        if self._lookup_callable is not None:
            answer_text, sources, confidence, debug = self._lookup_callable(query)
            return CurrentInfoResult(
                answer_status="answered" if answer_text else "insufficient_context",
                answer_text=answer_text or "I could not find a reliable current-news answer for that query.",
                sources=sources,
                confidence=confidence if answer_text else 0.0,
                provider_name="current_news_provider",
                lookup_type="sebi_current_news",
                debug={
                    "query": query,
                    "provider_available": True,
                    "official_web_attempted": False,
                    "general_web_attempted": False,
                    **debug,
                },
            )

        if self._official_search is None:
            return CurrentInfoResult(
                answer_status="unavailable",
                answer_text="Current news lookup is not available in this environment right now.",
                provider_name="current_news_unavailable",
                lookup_type="sebi_current_news",
                debug={
                    "query": query,
                    "provider_available": False,
                    "official_web_attempted": False,
                    "general_web_attempted": False,
                },
            )

        web_result = self._official_search.search(
            request=WebSearchRequest(
                query=self._build_search_query(query),
                instructions=(
                    "Find the latest official SEBI public development that clearly matches the query. "
                    "Prefer sebi.gov.in press releases, circulars, official updates, and government pages. "
                    "Answer from the most recent clearly relevant official item. "
                    "Include the date when the source clearly states it."
                ),
                lookup_type="sebi_current_news",
                source_type="official_web",
                allowed_domains=(
                    self._settings.official_allowed_domains if self._settings is not None else ()
                ),
                search_context_size="medium",
                max_results=(
                    self._settings.web_search_max_results if self._settings is not None else None
                ),
            )
        )
        confidence = assess_web_fallback_confidence(
            answer_status=web_result.answer_status,
            sources=web_result.sources,
            preferred_source_type="official_web",
            preferred_domains=self._settings.official_allowed_domains if self._settings is not None else (),
        )
        if web_result.answer_status == "answered" and not confidence.should_abstain:
            answer_text = web_result.answer_text
            if confidence.should_hedge:
                answer_text = "Official search found limited support. " + answer_text
            return CurrentInfoResult(
                answer_status="answered",
                answer_text=answer_text,
                sources=web_sources_to_current_info_sources(web_result.sources),
                confidence=confidence.confidence,
                provider_name=web_result.provider_name,
                lookup_type=web_result.lookup_type,
                debug={
                    **dict(web_result.debug),
                    "official_web_attempted": True,
                    "general_web_attempted": False,
                },
            )

        return CurrentInfoResult(
            answer_status=web_result.answer_status,
            answer_text=web_result.answer_text or "I could not find a reliable current-news answer for that query.",
            sources=web_sources_to_current_info_sources(web_result.sources),
            confidence=0.0,
            provider_name=web_result.provider_name,
            lookup_type=web_result.lookup_type,
            debug={
                **dict(web_result.debug),
                "official_web_attempted": True,
                "general_web_attempted": False,
            },
        )

    @staticmethod
    def _build_search_query(query: str) -> str:
        normalized = " ".join(query.lower().split())
        if "circular" in normalized:
            return "latest SEBI circular official update"
        if "press release" in normalized:
            return "latest SEBI press release official"
        return "latest SEBI press release circular public notice official update"
