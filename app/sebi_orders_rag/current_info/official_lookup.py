"""Structured and official-web lookup provider for current SEBI public facts."""

from __future__ import annotations

import re
from typing import Any

from ..answering.confidence import assess_web_fallback_confidence
from ..config import SebiOrdersRagSettings
from ..directory_data.fetcher import OfficialDirectoryHtmlFetcher
from ..directory_data.models import DirectoryReferenceDataset
from ..directory_data.service import _parse_source
from ..directory_data.sources import configured_directory_sources
from ..repositories.directory import DirectoryRepository
from ..repositories.structured_info import StructuredInfoRepository, build_structured_info_snapshot
from ..schemas import ChatSessionStateRecord
from ..structured_info.query_service import StructuredInfoQueryService
from .query_normalization import normalize_current_info_query
from ..web_fallback import WebSearchProvider, WebSearchRequest
from ..web_fallback.provider import build_official_web_search_provider
from ..web_fallback.ranking import extract_domain
from .institutional_facts import classify_institutional_facts_query
from .provider import (
    CurrentInfoProvider,
    CurrentInfoResult,
    CurrentInfoSource,
    normalize_current_info_sources,
    web_sources_to_current_info_sources,
)

_DEA_ORGANISATIONS_URL = "https://dea.gov.in/index.php/our-organisations/department-economic-affairs"
_MINISTRY_QUERY_RE = re.compile(
    r"\b(?:does\s+sebi\s+come\s+under|which\s+ministry\s+does\s+sebi\s+come\s+under|under\s+which\s+ministry\s+does\s+sebi\s+come|which\s+department\s+does\s+sebi\s+come\s+under)\b",
    re.IGNORECASE,
)
_OFFICIAL_FACT_QUERY_RE = re.compile(
    r"\b(?:sebi|securities and exchange board of india|ministry of finance|department of economic affairs|government of india)\b",
    re.IGNORECASE,
)


class OfficialWebsiteCurrentInfoProvider(CurrentInfoProvider):
    """Prefer structured directory data, then controlled official-web fallback."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        connection: Any | None = None,
        fetcher: OfficialDirectoryHtmlFetcher | None = None,
        official_search_provider: WebSearchProvider | None = None,
    ) -> None:
        self._settings = settings
        self._fetcher = fetcher or OfficialDirectoryHtmlFetcher(
            timeout_seconds=settings.directory_timeout_seconds,
            user_agent=settings.directory_user_agent,
        )
        has_db_cursor = connection is not None and hasattr(connection, "cursor")
        self._repository = DirectoryRepository(connection) if has_db_cursor else None
        self._structured_repository = (
            StructuredInfoRepository(connection) if has_db_cursor else None
        )
        self._structured_lookup = (
            StructuredInfoQueryService(
                snapshot_loader=self._structured_repository.load_snapshot,
                provider_name="canonical_structured_info",
            )
            if self._structured_repository is not None and settings.directory_enabled
            else None
        )
        self._live_lookup = StructuredInfoQueryService(
            snapshot_loader=self._load_live_snapshot,
            provider_name="official_web_live",
        )
        self._official_search = official_search_provider or build_official_web_search_provider(
            settings
        )

    def preview_internal_person_priority(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult | None:
        """Probe the canonical structured layer for strong internal SEBI person candidates."""

        if self._structured_lookup is None:
            return None
        intent = normalize_current_info_query(query, session_state=session_state)
        if intent.lookup_type != "person_lookup" or not intent.person_name:
            return None
        structured_result = self._structured_lookup.lookup(query=query, session_state=session_state)
        if _should_prioritize_internal_person_result(structured_result):
            return self._decorate_result(
                structured_result,
                structured_attempted=True,
                live_structured_attempted=False,
                official_web_attempted=False,
                general_web_attempted=False,
                web_fallback_allowed=False,
                web_fallback_block_reason="internal_person_priority_override",
                answer_origin="structured_person_priority",
                default_source_type="structured",
            )

        live_result = self._live_lookup.lookup(query=query, session_state=session_state)
        if not _should_prioritize_internal_person_result(live_result):
            return None
        return self._decorate_result(
            live_result,
            structured_attempted=True,
            live_structured_attempted=True,
            official_web_attempted=False,
            general_web_attempted=False,
            web_fallback_allowed=False,
            web_fallback_block_reason="internal_person_priority_override",
            answer_origin="live_structured_person_priority",
            default_source_type="structured",
        )

    def lookup(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult:
        structured_attempted = False
        live_structured_attempted = False
        official_web_attempted = False
        web_fallback_allowed = self._supports_official_web_search(query)
        web_fallback_block_reason = None if web_fallback_allowed else "query_not_official_fact_lookup"
        structured_supported = self._live_lookup.supports_query(
            query=query,
            session_state=session_state,
        )

        structured_result: CurrentInfoResult | None = None
        if structured_supported:
            if self._structured_lookup is not None:
                structured_attempted = True
                structured_result = self._structured_lookup.lookup(
                    query=query,
                    session_state=session_state,
                )
                if (
                    structured_result.answer_status == "answered"
                    and not _should_refresh_live_structured_result(structured_result)
                ):
                    return self._decorate_result(
                        structured_result,
                        structured_attempted=structured_attempted,
                        live_structured_attempted=live_structured_attempted,
                        official_web_attempted=official_web_attempted,
                        general_web_attempted=False,
                        web_fallback_allowed=False,
                        web_fallback_block_reason="structured_current_info_only",
                        answer_origin="structured",
                        default_source_type="structured",
                    )
            live_structured_attempted = True
            live_result = self._live_lookup.lookup(query=query, session_state=session_state)
            if _should_use_live_structured_result(
                structured_result=structured_result,
                live_result=live_result,
            ):
                return self._decorate_result(
                    live_result,
                    structured_attempted=structured_attempted,
                    live_structured_attempted=live_structured_attempted,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    web_fallback_allowed=False,
                    web_fallback_block_reason="structured_current_info_only",
                    answer_origin="live_structured",
                    default_source_type="structured",
                )
            return self._decorate_result(
                structured_result or live_result,
                structured_attempted=structured_attempted,
                live_structured_attempted=live_structured_attempted,
                official_web_attempted=False,
                general_web_attempted=False,
                web_fallback_allowed=False,
                web_fallback_block_reason="structured_current_info_only",
                answer_origin="structured_miss",
                default_source_type="structured",
            )

        if _MINISTRY_QUERY_RE.search(query):
            ministry_result = self._lookup_ministry_relation()
            if ministry_result.answer_status == "answered":
                return self._decorate_result(
                    ministry_result,
                    structured_attempted=structured_attempted,
                    live_structured_attempted=live_structured_attempted,
                    official_web_attempted=True,
                    general_web_attempted=False,
                    web_fallback_allowed=True,
                    web_fallback_block_reason=None,
                    answer_origin="official_web_fetch",
                    default_source_type="official_web",
                )
            structured_result = structured_result or ministry_result

        institutional_plan = classify_institutional_facts_query(query)
        if web_fallback_allowed:
            official_web_attempted = True
            web_result = self._official_search.search(
                request=WebSearchRequest(
                    query=(
                        institutional_plan.search_query
                        if institutional_plan is not None
                        else self._build_official_search_query(query)
                    ),
                    instructions=(
                        institutional_plan.search_instructions
                        if institutional_plan is not None
                        else self._build_official_search_instructions(query)
                    ),
                    lookup_type=(
                        institutional_plan.lookup_type
                        if institutional_plan is not None
                        else self._lookup_type_for_query(query)
                    ),
                    source_type="official_web",
                    allowed_domains=self._settings.official_allowed_domains,
                    search_context_size="low",
                    max_results=self._settings.web_search_max_results,
                )
            )
            confidence = assess_web_fallback_confidence(
                answer_status=web_result.answer_status,
                sources=web_result.sources,
                preferred_source_type="official_web",
                preferred_domains=self._settings.official_allowed_domains,
            )
            if web_result.answer_status == "answered" and not confidence.should_abstain:
                answer_text = web_result.answer_text
                if confidence.should_hedge:
                    answer_text = (
                        "Official web sources provide only limited support. " + answer_text
                    )
                result = CurrentInfoResult(
                    answer_status="answered",
                    answer_text=answer_text,
                    sources=web_sources_to_current_info_sources(web_result.sources),
                    confidence=confidence.confidence,
                    provider_name=web_result.provider_name,
                    lookup_type=web_result.lookup_type,
                    debug=dict(web_result.debug),
                )
                return self._decorate_result(
                    result,
                    structured_attempted=structured_attempted,
                    live_structured_attempted=live_structured_attempted,
                    official_web_attempted=official_web_attempted,
                    general_web_attempted=False,
                    web_fallback_allowed=True,
                    web_fallback_block_reason=None,
                    answer_origin=(
                        "institutional_facts_official_web_search"
                        if institutional_plan is not None
                        else "official_web_search"
                    ),
                    default_source_type="official_web",
                )
            if structured_result is None:
                return self._decorate_result(
                    CurrentInfoResult(
                        answer_status=web_result.answer_status,
                        answer_text=web_result.answer_text,
                        sources=web_sources_to_current_info_sources(web_result.sources),
                        confidence=0.0,
                        provider_name=web_result.provider_name,
                        lookup_type=web_result.lookup_type,
                        debug=dict(web_result.debug),
                    ),
                    structured_attempted=structured_attempted,
                    live_structured_attempted=live_structured_attempted,
                    official_web_attempted=official_web_attempted,
                    general_web_attempted=False,
                    web_fallback_allowed=True,
                    web_fallback_block_reason=None,
                    answer_origin=(
                        "institutional_facts_official_web_search_failed"
                        if institutional_plan is not None
                        else "official_web_search_failed"
                    ),
                    default_source_type="official_web",
                )

        if structured_result is not None:
            return self._decorate_result(
                structured_result,
                structured_attempted=structured_attempted,
                live_structured_attempted=live_structured_attempted,
                official_web_attempted=official_web_attempted,
                general_web_attempted=False,
                web_fallback_allowed=web_fallback_allowed,
                web_fallback_block_reason=web_fallback_block_reason,
                answer_origin="structured_miss",
                default_source_type="structured",
            )

        return self._decorate_result(
            CurrentInfoResult(
                answer_status="insufficient_context",
                answer_text=(
                    "I could not confidently verify that from structured SEBI data or official web sources."
                ),
                provider_name="official_web",
                lookup_type=self._lookup_type_for_query(query),
                debug={"query": query},
            ),
            structured_attempted=structured_attempted,
            live_structured_attempted=live_structured_attempted,
            official_web_attempted=official_web_attempted,
            general_web_attempted=False,
            web_fallback_allowed=web_fallback_allowed,
            web_fallback_block_reason=web_fallback_block_reason,
            answer_origin="no_support",
            default_source_type="official_web",
        )

    def _load_live_snapshot(self):
        dataset = DirectoryReferenceDataset()
        for source in configured_directory_sources(self._settings):
            try:
                fetched = self._fetcher.fetch(source)
                parsed = _parse_source(source.source_type, fetched)
            except Exception:
                continue
            dataset = dataset.merged(parsed.as_dataset())
        return build_structured_info_snapshot(dataset)

    def _lookup_ministry_relation(self) -> CurrentInfoResult:
        source = CurrentInfoSource(
            title="Department of Economic Affairs",
            url=_DEA_ORGANISATIONS_URL,
            record_key="official:dea.gov.in",
            domain=extract_domain(_DEA_ORGANISATIONS_URL),
            source_type="official_web",
        )
        try:
            fetched = self._fetcher.fetch(
                type(
                    "Source",
                    (),
                    {
                        "source_type": "dea",
                        "title": source.title,
                        "url": source.url,
                    },
                )()
            )
        except Exception as exc:
            return CurrentInfoResult(
                answer_status="unavailable",
                answer_text=(
                    "Current official lookup is unavailable in this environment right now. "
                    "Please verify on the official Department of Economic Affairs website."
                ),
                provider_name="official_web",
                lookup_type="sebi_ministry_relation",
                debug={"error": f"{type(exc).__name__}: {exc}"},
            )

        page_text = fetched.raw_html.lower()
        has_sebi_listing = "securities and exchange board of india" in page_text
        has_dea_listing = "institutions under department of economic affairs" in page_text
        has_ministry_banner = (
            "ministry of finance" in page_text and "department of economic affairs" in page_text
        )
        if not (has_sebi_listing and has_dea_listing and has_ministry_banner):
            return CurrentInfoResult(
                answer_status="insufficient_context",
                answer_text=(
                    "I could not confidently confirm SEBI's current administrative placement "
                    "from the official Department of Economic Affairs page."
                ),
                sources=(source,),
                provider_name="official_web",
                lookup_type="sebi_ministry_relation",
                debug={
                    "has_sebi_listing": has_sebi_listing,
                    "has_dea_listing": has_dea_listing,
                    "has_ministry_banner": has_ministry_banner,
                },
            )

        return CurrentInfoResult(
            answer_status="answered",
            answer_text=(
                "The Department of Economic Affairs page of the Ministry of Finance lists "
                "SEBI as an institution under the Department of Economic Affairs."
            ),
            sources=(source,),
            confidence=0.89,
            provider_name="official_web",
            lookup_type="sebi_ministry_relation",
            debug={"parsed": True},
        )

    @staticmethod
    def _supports_official_web_search(query: str) -> bool:
        normalized = " ".join(query.lower().split())
        return bool(
            _OFFICIAL_FACT_QUERY_RE.search(normalized)
            or classify_institutional_facts_query(normalized) is not None
        )

    @staticmethod
    def _lookup_type_for_query(query: str) -> str:
        normalized = " ".join(query.lower().split())
        if _MINISTRY_QUERY_RE.search(normalized):
            return "sebi_ministry_relation"
        institutional_plan = classify_institutional_facts_query(normalized)
        if institutional_plan is not None:
            return institutional_plan.lookup_type
        return "current_official_lookup"

    @staticmethod
    def _build_official_search_instructions(query: str) -> str:
        institutional_plan = classify_institutional_facts_query(query)
        if institutional_plan is not None:
            return institutional_plan.search_instructions
        return (
            "Answer only from current official SEBI or Government of India webpages. "
            "Prefer direct institutional fact pages over summaries."
        )

    @staticmethod
    def _build_official_search_query(query: str) -> str:
        normalized = " ".join(query.lower().split())
        if _MINISTRY_QUERY_RE.search(normalized):
            return "SEBI ministry Department of Economic Affairs Ministry of Finance official"
        institutional_plan = classify_institutional_facts_query(normalized)
        if institutional_plan is not None:
            return institutional_plan.search_query
        return query

    @staticmethod
    def _decorate_result(
        result: CurrentInfoResult,
        *,
        structured_attempted: bool,
        live_structured_attempted: bool,
        official_web_attempted: bool,
        general_web_attempted: bool,
        web_fallback_allowed: bool,
        web_fallback_block_reason: str | None,
        answer_origin: str,
        default_source_type: str,
    ) -> CurrentInfoResult:
        sources = normalize_current_info_sources(
            result.sources,
            default_source_type=default_source_type,
        )
        source_domains = [source.domain or extract_domain(source.url) for source in sources]
        return CurrentInfoResult(
            answer_status=result.answer_status,
            answer_text=result.answer_text,
            sources=sources,
            confidence=result.confidence,
            provider_name=result.provider_name,
            lookup_type=result.lookup_type,
            debug={
                **dict(result.debug),
                "structured_attempted": structured_attempted,
                "live_structured_attempted": live_structured_attempted,
                "official_web_attempted": official_web_attempted,
                "general_web_attempted": general_web_attempted,
                "web_fallback_allowed": web_fallback_allowed,
                "web_fallback_not_allowed_reason": web_fallback_block_reason,
                "answer_origin": answer_origin,
                "institutional_facts_lookup": result.lookup_type in {
                    "sebi_income_sources",
                    "sebi_fee_or_charge_query",
                },
                "source_domains": source_domains,
            },
        )


def _should_prioritize_internal_person_result(result: CurrentInfoResult) -> bool:
    if result.lookup_type != "person_lookup":
        return False
    debug = dict(result.debug)
    if result.answer_status == "answered":
        return True
    if str(debug.get("fallback_reason") or "") == "person_match_clarify":
        return True
    if int(debug.get("matched_people_rows_count") or 0) > 0:
        return True
    return str(debug.get("fuzzy_band") or "") in {"high", "medium"}


def _should_refresh_live_structured_result(result: CurrentInfoResult) -> bool:
    if result.lookup_type != "designation_count":
        return False
    count_debug = dict(result.debug.get("count_debug") or {})
    return int(count_debug.get("count") or 0) == 0


def _should_use_live_structured_result(
    *,
    structured_result: CurrentInfoResult | None,
    live_result: CurrentInfoResult,
) -> bool:
    if structured_result is None:
        return True
    if structured_result.answer_status != "answered":
        return True
    if _should_prioritize_internal_person_result(live_result):
        return True
    if live_result.answer_status != "answered":
        return False
    if live_result.lookup_type != "designation_count":
        return structured_result is None or structured_result.answer_status != "answered"
    live_count = _designation_count_value(live_result)
    stored_count = _designation_count_value(structured_result)
    return live_count > stored_count


def _designation_count_value(result: CurrentInfoResult | None) -> int:
    if result is None or result.lookup_type != "designation_count":
        return 0
    count_debug = dict(result.debug.get("count_debug") or {})
    return int(count_debug.get("count") or 0)
