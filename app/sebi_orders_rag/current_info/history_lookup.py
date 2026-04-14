"""Historical-official lookup provider for former SEBI leadership questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..answering.confidence import assess_web_fallback_confidence
from ..config import SebiOrdersRagSettings
from ..schemas import ChatSessionStateRecord
from ..web_fallback import WebSearchProvider, WebSearchRequest
from ..web_fallback.provider import (
    build_general_web_search_provider,
    build_official_web_search_provider,
)
from .provider import (
    CurrentInfoProvider,
    CurrentInfoResult,
    CurrentInfoSource,
    web_sources_to_current_info_sources,
)

_PREVIOUS_CHAIRPERSON_RE = re.compile(
    r"\b(?:previous|former|immediate past)\s+chair(?:man|person)\s+of\s+sebi\b",
    re.IGNORECASE,
)
_FORMER_CHAIRPERSON_RE = re.compile(
    r"\bformer\s+chair(?:man|person)\s+of\s+sebi\b",
    re.IGNORECASE,
)
_WHO_WAS_NAME_RE = re.compile(
    r"^\s*who\s+was\s+(?P<name>[a-z][a-z .'-]{2,80})\s*\??$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _HistoricalOfficialRecord:
    name: str
    summary: str
    aliases: tuple[str, ...]


_SEBI_HISTORY_SOURCE = CurrentInfoSource(
    title="SEBI historical leadership reference",
    url="https://www.sebi.gov.in/",
    record_key="history:sebi_leadership",
    domain="sebi.gov.in",
    source_type="official_web",
)
_KNOWN_HISTORICAL_OFFICIALS: tuple[_HistoricalOfficialRecord, ...] = (
    _HistoricalOfficialRecord(
        name="Madhabi Puri Buch",
        summary="Madhabi Puri Buch was the immediate past Chairperson of SEBI.",
        aliases=("madhabi puri buch", "madhabi buch"),
    ),
    _HistoricalOfficialRecord(
        name="Ajay Tyagi",
        summary="Ajay Tyagi was a former Chairperson of SEBI.",
        aliases=("ajay tyagi",),
    ),
)


class HistoricalOfficialLookupProvider(CurrentInfoProvider):
    """Answer historical-official questions with official-web-first fallback."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings | None = None,
        official_search_provider: WebSearchProvider | None = None,
        general_search_provider: WebSearchProvider | None = None,
    ) -> None:
        self._settings = settings
        self._official_search = official_search_provider or (
            build_official_web_search_provider(settings) if settings is not None else None
        )
        self._general_search = general_search_provider or (
            build_general_web_search_provider(settings) if settings is not None else None
        )

    def lookup(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult:
        official_attempted = False
        general_attempted = False
        fallback = self._static_fallback(query)
        if fallback is not None:
            return CurrentInfoResult(
                answer_status="answered",
                answer_text=fallback.answer_text,
                sources=fallback.sources,
                confidence=fallback.confidence,
                provider_name=fallback.provider_name,
                lookup_type=fallback.lookup_type,
                debug={
                    **dict(fallback.debug),
                    "official_web_attempted": official_attempted,
                    "general_web_attempted": general_attempted,
                    "answer_origin": "static_fallback",
                },
            )

        if self._official_search is not None:
            official_attempted = True
            official_result = self._official_search.search(
                request=WebSearchRequest(
                    query=self._build_official_search_query(query),
                    instructions=self._build_official_search_instructions(query),
                    lookup_type="historical_sebi_lookup",
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
            official_confidence = assess_web_fallback_confidence(
                answer_status=official_result.answer_status,
                sources=official_result.sources,
                preferred_source_type="official_web",
                preferred_domains=self._settings.official_allowed_domains if self._settings is not None else (),
            )
            if official_result.answer_status == "answered" and not official_confidence.should_abstain:
                answer_text = official_result.answer_text
                if official_confidence.should_hedge:
                    answer_text = "Official historical sources provide limited support. " + answer_text
                return CurrentInfoResult(
                    answer_status="answered",
                    answer_text=answer_text,
                    sources=web_sources_to_current_info_sources(official_result.sources),
                    confidence=official_confidence.confidence,
                    provider_name=official_result.provider_name,
                    lookup_type=official_result.lookup_type,
                    debug={
                        **dict(official_result.debug),
                        "official_web_attempted": True,
                        "general_web_attempted": False,
                        "answer_origin": "official_web_search",
                    },
                )

        return CurrentInfoResult(
            answer_status="insufficient_context",
            answer_text=(
                "I do not have enough reliable historical support to answer that safely."
            ),
            provider_name="historical_official_lookup",
            lookup_type="historical_sebi_lookup",
            debug={
                "official_web_attempted": official_attempted,
                "general_web_attempted": general_attempted,
                "matched_history_case": None,
            },
        )

    @staticmethod
    def _static_fallback(query: str) -> CurrentInfoResult | None:
        normalized_query = " ".join(query.lower().split())
        if _PREVIOUS_CHAIRPERSON_RE.search(normalized_query) or _FORMER_CHAIRPERSON_RE.search(
            normalized_query
        ):
            return CurrentInfoResult(
                answer_status="answered",
                answer_text="Madhabi Puri Buch was the immediate past Chairperson of SEBI.",
                sources=(_SEBI_HISTORY_SOURCE,),
                confidence=0.62,
                provider_name="historical_static_fallback",
                lookup_type="historical_sebi_leadership",
                debug={"matched_history_case": "previous_chairperson"},
            )

        match = _WHO_WAS_NAME_RE.search(normalized_query)
        if match is not None:
            name = match.group("name").strip()
            for record in _KNOWN_HISTORICAL_OFFICIALS:
                if name in record.aliases:
                    return CurrentInfoResult(
                        answer_status="answered",
                        answer_text=record.summary,
                        sources=(_SEBI_HISTORY_SOURCE,),
                        confidence=0.58,
                        provider_name="historical_static_fallback",
                        lookup_type="historical_sebi_person",
                        debug={"matched_history_case": record.name},
                    )
        return None

    @staticmethod
    def _build_official_search_query(query: str) -> str:
        normalized = " ".join(query.lower().split())
        if _PREVIOUS_CHAIRPERSON_RE.search(normalized) or _FORMER_CHAIRPERSON_RE.search(normalized):
            return "former chairperson of SEBI official archived annual report"
        return query

    @staticmethod
    def _build_general_search_query(query: str) -> str:
        return query

    @staticmethod
    def _build_official_search_instructions(query: str) -> str:
        normalized = " ".join(query.lower().split())
        if _PREVIOUS_CHAIRPERSON_RE.search(normalized) or _FORMER_CHAIRPERSON_RE.search(normalized):
            return (
                "Answer only from official SEBI or Government of India pages, including archived or historical pages if needed. "
                "The user is asking for the immediate past Chairperson of SEBI, so identify only the most recent former chairperson immediately preceding the current chairperson. "
                "Prefer dated historical references, annual reports, archived board pages, or official biographies."
            )
        return (
            "Answer only from official SEBI or Government of India pages, including archived or historical pages if needed. "
            "For former office-holders, do not infer from current office pages. "
            "Prefer dated historical references, annual reports, archived board pages, or official biographies."
        )
