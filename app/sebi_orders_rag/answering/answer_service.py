"""Adaptive RAG orchestration for Phase 4 chat answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID, uuid4

from ..config import SebiOrdersRagSettings
from ..control import (
    MatterLockCandidate,
    StrictMatterLock,
    build_matter_clarification_candidates,
    build_person_clarification_candidates,
    dataclass_asdict,
    evaluate_mixed_record_guardrail,
    filter_items_to_locked_record_keys,
    looks_like_sat_court_query,
    load_control_pack,
    render_clarification_candidate_lines,
    select_exact_lookup_resolution,
)
from ..corpus_stats import CorpusStatsRepository, CorpusStatsService
from ..embeddings.client import OpenAIEmbeddingClient
from ..exceptions import ConfigurationError, MissingDependencyError
from ..current_info import (
    CurrentInfoProvider,
    CurrentNewsLookupProvider,
    HistoricalOfficialLookupProvider,
    build_current_info_provider,
    parse_company_role_query,
)
from ..memory.memory_service import GroundedMemoryService
from ..metadata import MetadataAnswer, OrderMetadataService
from ..repositories.answers import AnswerRepository
from ..repositories.metadata import OrderMetadataRepository
from ..repositories.retrieval import HierarchicalRetrievalRepository
from ..repositories.sessions import ChatSessionRepository
from ..retrieval.hierarchical_search import HierarchicalSearchService
from ..retrieval.query_intent import QueryIntent
from ..retrieval.scoring import ChunkSearchHit, HierarchicalSearchResult
from ..router.decision import AdaptiveQueryRouter
from ..router.planner import build_order_lookup_variants
from ..schemas import (
    Citation,
    ChatAnswerPayload,
    ClarificationCandidate,
    ClarificationContext,
    MetadataFilterInput,
    PromptContextChunk,
    QueryAnalysis,
    RouteDecision,
)
from ..web_fallback.models import WebSearchSource
from ..web_fallback import WebSearchRequest
from ..web_fallback.provider import build_general_web_search_provider
from ..web_fallback.ranking import extract_domain
from .citations import (
    build_external_citations,
    build_citations,
    extract_citation_numbers,
    filter_citations,
    resolve_cited_context_chunks,
    strip_inline_citation_markers,
)
from .confidence import (
    ConfidenceAssessment,
    assess_direct_llm_confidence,
    assess_retrieval_confidence,
    assess_web_fallback_confidence,
)
from .prompt_builder import (
    PromptPayload,
    build_general_knowledge_prompt,
    build_retrieval_answer_prompt,
)
from .style import apply_grounded_wording_caution

LOGGER = logging.getLogger(__name__)
_EMPTY_RETRIEVAL_ANSWER = (
    "I could not find enough reliable support in the retrieved SEBI order text to answer that."
)
_AMBIGUOUS_NAMED_MATTER_ANSWER = (
    "I found multiple similarly named SEBI matters and cannot safely answer from one record "
    "without risking cross-document contamination. Please specify the exact title, date, or record."
)
_STRICT_MATTER_WEAK_SUPPORT_ANSWER = (
    "I could not find enough reliable support inside the identified matter alone to answer that "
    "without mixing in another record."
)
_CANDIDATE_LIST_INTRO = (
    "I found multiple plausible internal SEBI matter matches. Please pick one of these exact records:"
)
_AMBIGUOUS_ENTITY_SUMMARY_PREFIXES: tuple[str, ...] = (
    "tell me more about",
    "tell me about",
    "summary of",
)
_IPO_PROCEEDS_CONTEXT_RE = re.compile(
    r"\bpreferential allotment\b|\bloan conversion\b|\bconversion of the outstanding unsecured loans into equity shares\b",
    re.IGNORECASE,
)


def _select_missing_ipo_proceeds_chunk(chunks: tuple[Any, ...]) -> Any | None:
    best_chunk = None
    best_score = -1
    for chunk in chunks:
        text = str(getattr(chunk, "text", "") or "")
        if not _IPO_PROCEEDS_CONTEXT_RE.search(text):
            continue
        section_type = str(getattr(chunk, "section_type", "") or "").strip().lower()
        score = 2 if section_type in {"facts", "background", "findings"} else 1
        if score > best_score:
            best_chunk = chunk
            best_score = score
    return best_chunk


def _should_abstain_ambiguous_entity_summary(*, query: str, analysis: QueryAnalysis) -> bool:
    normalized_query = " ".join(query.lower().split())
    if not any(normalized_query.startswith(prefix) for prefix in _AMBIGUOUS_ENTITY_SUMMARY_PREFIXES):
        return False
    if (
        analysis.appears_matter_specific
        or analysis.appears_general_explanatory
        or analysis.appears_structured_current_info
        or analysis.appears_current_official_lookup
        or analysis.appears_current_news_lookup
        or analysis.appears_company_role_current_fact
        or analysis.appears_non_sebi_person_query
    ):
        return False
    return True


@dataclass(frozen=True)
class _RetrievalExecution:
    route_mode: str
    search_result: HierarchicalSearchResult
    extracted_filters: dict[str, Any]


class OpenAIChatClient:
    """Minimal JSON-oriented OpenAI chat client for Phase 4 answers."""

    def __init__(self, settings: SebiOrdersRagSettings) -> None:
        api_key = (settings.openai_api_key or "").strip()
        if not api_key or api_key.upper() in {"YOUR_KEY", "YOUR_API_KEY"}:
            raise ConfigurationError(
                "SEBI_ORDERS_RAG_OPENAI_API_KEY must be set to a real API key for Phase 4 chat."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise MissingDependencyError(
                "openai is required for Phase 4 answering. "
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
        self._settings = settings

    def complete_json(self, prompt: PromptPayload) -> dict[str, Any]:
        """Execute one model call and parse the JSON response body."""

        response_format = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(
                model=self._settings.chat_model,
                temperature=self._settings.chat_temperature,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt},
                ],
            )
        except Exception:
            response = self._client.chat.completions.create(
                model=self._settings.chat_model,
                temperature=self._settings.chat_temperature,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt},
                ],
            )

        content = response.choices[0].message.content or "{}"
        return _parse_json_object(content)


class AdaptiveRagAnswerService:
    """Route, retrieve, ground, cite, log, and update session memory."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        connection: Any,
        search_service: HierarchicalSearchService | None = None,
        retrieval_repository: HierarchicalRetrievalRepository | None = None,
        session_repository: ChatSessionRepository | None = None,
        answer_repository: AnswerRepository | None = None,
        memory_service: GroundedMemoryService | None = None,
        router: AdaptiveQueryRouter | None = None,
        current_info_provider: CurrentInfoProvider | None = None,
        current_news_provider: CurrentInfoProvider | None = None,
        historical_info_provider: CurrentInfoProvider | None = None,
        general_web_provider: Any | None = None,
        metadata_service: OrderMetadataService | None = None,
        corpus_stats_service: CorpusStatsService | None = None,
        llm_client: OpenAIChatClient | Any | None = None,
    ) -> None:
        self._settings = settings
        self._connection = connection
        self._control_pack = load_control_pack(settings.control_pack_root)
        self._retrieval = retrieval_repository or HierarchicalRetrievalRepository(connection)
        self._sessions = session_repository or ChatSessionRepository(connection)
        self._answers = answer_repository or AnswerRepository(connection)
        self._search = search_service or HierarchicalSearchService(
            settings=settings,
            connection=connection,
            control_pack=self._control_pack,
        )
        self._memory = memory_service or GroundedMemoryService(
            session_repository=self._sessions,
            retrieval_repository=self._retrieval,
        )
        self._router = router or AdaptiveQueryRouter(control_pack=self._control_pack)
        self._current_info = current_info_provider or build_current_info_provider(
            settings,
            connection=connection,
        )
        self._current_news = current_news_provider or CurrentNewsLookupProvider(settings=settings)
        self._historical_info = historical_info_provider or HistoricalOfficialLookupProvider(
            settings=settings
        )
        self._general_web = general_web_provider or build_general_web_search_provider(settings)
        self._metadata_repository = OrderMetadataRepository(connection)
        self._metadata = metadata_service or OrderMetadataService(
            repository=self._metadata_repository
        )
        self._corpus_stats = corpus_stats_service or CorpusStatsService(
            repository=CorpusStatsRepository(data_root=settings.data_root)
        )
        self._llm = llm_client or OpenAIChatClient(settings)

    def answer_query(
        self,
        *,
        query: str,
        session_id: UUID | None = None,
    ) -> ChatAnswerPayload:
        """Answer one user query with deterministic routing and conservative grounding."""

        session = None
        decision = None
        try:
            session = self._memory.get_or_create_session(session_id=session_id)
            clarification_payload = self._maybe_answer_clarification_selection(
                query=query,
                session_id=session.session_id,
                session_state=session.state,
            )
            if clarification_payload is not None:
                payload = clarification_payload
            else:
                decision = self._router.decide(query=query, session_state=session.state)
                if decision.route_mode == "smalltalk":
                    payload = self._answer_smalltalk(query=query, session_id=session.session_id, decision=decision)
                elif decision.route_mode == "clarify":
                    payload = self._answer_direct_planner_clarify(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                elif decision.route_mode == "abstain":
                    payload = self._answer_direct_abstain(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                elif decision.route_mode == "structured_current_info":
                    payload = self._answer_structured_current_info(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                elif decision.route_mode == "corpus_metadata":
                    payload = self._answer_corpus_metadata(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                elif decision.route_mode in {"general_knowledge", "direct_llm"}:
                    payload = self._answer_general_knowledge(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                elif decision.route_mode == "current_news_lookup":
                    payload = self._answer_current_news_lookup(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                elif decision.route_mode == "historical_official_lookup":
                    payload = self._answer_historical_official_lookup(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                elif decision.route_mode == "current_official_lookup":
                    payload = self._answer_current_official_lookup(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                    )
                else:
                    payload = self._answer_with_retrieval(
                        query=query,
                        session_id=session.session_id,
                        decision=decision,
                        grounded_summary=session.state.grounded_summary if session.state else None,
                    )
            if self._settings.enable_memory and payload.answer_status != "clarify":
                self._memory.clear_clarification_context(session_id=session.session_id)
            if hasattr(self._connection, "commit"):
                self._connection.commit()
            return payload
        except Exception as exc:
            if hasattr(self._connection, "rollback"):
                self._connection.rollback()
            LOGGER.exception(
                "adaptive_rag_answer_failed",
                extra={
                    "query": query,
                    "session_id": str(session.session_id if session is not None else session_id),
                    "route_mode": decision.route_mode if decision is not None else None,
                },
            )
            return self._build_failure_payload(
                query=query,
                session_id=(session.session_id if session is not None else (session_id or uuid4())),
                decision=decision,
                exc=exc,
            )

    def _answer_smalltalk(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        normalized_query = " ".join(query.lower().split())
        if any(token in normalized_query for token in ("thank you", "thanks", "thx")):
            answer_text = "You're welcome."
        elif "what can you do" in normalized_query or normalized_query == "help":
            answer_text = (
                "I can answer grounded questions about SEBI enforcement orders, "
                "look up a specific matter, handle follow-ups within the active matter, "
                "and separate corpus answers from general or current official questions."
            )
        elif any(token in normalized_query for token in ("bye", "goodbye")):
            answer_text = "I can help whenever you're ready."
        else:
            answer_text = "I can help with SEBI orders and related follow-up questions."
        return self._build_non_corpus_payload(
            session_id=session_id,
            query=query,
            decision=decision,
            route_mode="smalltalk",
            answer_text=answer_text,
            confidence=0.94,
        )

    def _answer_general_knowledge(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        internal_person_priority_payload = self._maybe_answer_internal_person_priority(
            query=query,
            session_id=session_id,
            decision=decision,
        )
        if internal_person_priority_payload is not None:
            return internal_person_priority_payload

        static_answer = _static_general_knowledge_answer(query=query, analysis=decision.analysis)
        llm_answer_used = False
        if static_answer is not None:
            answer_text = static_answer
            answer_status = "answered"
        else:
            prompt = build_general_knowledge_prompt(query=query)
            response = self._llm.complete_json(prompt)
            answer_text = str(response.get("answer_text", "")).strip()
            answer_status = _normalize_answer_status(response.get("answer_status"))
            llm_answer_used = True
        if not answer_text:
            answer_text = "I do not have enough context to answer that clearly."
            answer_status = "insufficient_context"

        citations = ()
        confidence = assess_direct_llm_confidence(answer_text) if answer_status == "answered" else 0.0
        general_web_allowed, web_block_reason = _general_web_fallback_allowed(
            analysis=decision.analysis,
            query=query,
        )
        requires_live_web_check = _general_query_requires_live_web_check(
            analysis=decision.analysis,
            query=query,
        )
        general_web_attempted = False
        if (
            general_web_allowed
            and (
                requires_live_web_check
                or
                answer_status != "answered"
                or (llm_answer_used and confidence < 0.55)
            )
        ):
            general_web_attempted = True
            web_result = self._general_web.search(
                request=WebSearchRequest(
                    query=query,
                    instructions=(
                        "Answer only if broader web sources clearly support the requested general fact. "
                        "Prefer authoritative reference pages for people or organisations. "
                        "Abstain if the search results are weak or mixed."
                    ),
                    lookup_type="general_knowledge",
                    source_type="general_web",
                    search_context_size="medium",
                    max_results=self._settings.web_search_max_results,
                )
            )
            web_sources = _filter_general_web_sources(
                query=query,
                answer_text=web_result.answer_text,
                analysis=decision.analysis,
                sources=web_result.sources,
            )
            web_confidence = assess_web_fallback_confidence(
                answer_status=web_result.answer_status,
                sources=web_sources,
                preferred_source_type="general_web",
            )
            if web_result.answer_status == "answered" and not web_confidence.should_abstain:
                answer_text = web_result.answer_text
                if web_confidence.should_hedge:
                    answer_text = "Broader web sources provide limited support. " + answer_text
                answer_status = "answered"
                confidence = web_confidence.confidence
                citations = build_external_citations(web_sources)
            elif requires_live_web_check:
                answer_text = (
                    web_result.answer_text
                    or "I could not verify the current public information from reliable web sources."
                )
                answer_status = "insufficient_context"
                confidence = 0.0
                citations = ()
            elif answer_status != "answered":
                answer_text = web_result.answer_text or "I do not have enough context to answer that clearly."
                confidence = 0.0

        route_mode = "general_knowledge" if answer_status == "answered" else "abstain"
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode=route_mode,
            query_intent=decision.query_intent,
            answer_text=answer_text,
            confidence=confidence,
            citations=citations,
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="answered" if route_mode == "general_knowledge" else "abstained",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "current_lookup_debug": {
                    "used": False,
                },
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "metadata_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=general_web_attempted,
                    final_route_mode=route_mode,
                    citations=citations,
                    web_fallback_allowed=general_web_allowed,
                    web_fallback_not_allowed_reason=(
                        None if general_web_attempted else web_block_reason
                    ),
                ),
                "citation_debug": _build_citation_debug_payload(citations),
            },
        )
        if self._settings.enable_memory:
            self._memory.clear_current_lookup_context(session_id=session_id)
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _answer_direct_planner_clarify(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        reason = decision.plan.reason if decision.plan is not None else "clarify"
        if reason != "missing_order_scope_for_metadata" and _should_abstain_ambiguous_entity_summary(
            query=query,
            analysis=decision.analysis,
        ):
            return self._answer_direct_abstain(
                query=query,
                session_id=session_id,
                decision=decision,
            )
        if reason == "missing_order_scope_for_metadata":
            answer_text = (
                "Please specify the exact SEBI order or matter before I answer that exact-fact question."
            )
        else:
            answer_text = (
                "Please clarify whether you want a structured current fact, a specific SEBI matter, or a general explanation."
            )
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="clarify",
            query_intent=decision.query_intent,
            answer_text=answer_text,
            confidence=0.0,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="clarify",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "corpus_metadata_debug": {"used": False},
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode="clarify",
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason=reason,
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _answer_direct_abstain(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        reason = decision.plan.reason if decision.plan is not None else "abstain"
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="abstain",
            query_intent=decision.query_intent,
            answer_text="I do not have enough reliable context to answer that safely.",
            confidence=0.0,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="abstained",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "corpus_metadata_debug": {"used": False},
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode="abstain",
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason=reason,
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _answer_structured_current_info(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        return self._answer_current_lookup(
            query=query,
            session_id=session_id,
            decision=decision,
            requested_route_mode="structured_current_info",
        )

    def _answer_current_news_lookup(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        result = self._current_news.lookup(query=query, session_state=self._sessions.get_session_state(session_id=session_id))
        citations = build_external_citations(result.sources)
        answered = result.answer_status == "answered"
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="current_news_lookup",
            query_intent=decision.query_intent,
            answer_text=result.answer_text,
            confidence=result.confidence if answered else 0.0,
            citations=citations,
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="answered" if answered else "abstained",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {
                    "used": True,
                    "provider_name": result.provider_name,
                    "lookup_type": result.lookup_type,
                    "answer_status": result.answer_status,
                    **dict(result.debug),
                },
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=bool(result.debug.get("official_web_attempted", True)),
                    general_web_attempted=bool(result.debug.get("general_web_attempted", False)),
                    final_route_mode="current_news_lookup" if answered else "abstain",
                    citations=citations,
                    web_fallback_allowed=True,
                    web_fallback_not_allowed_reason=None,
                ),
                "citation_debug": _build_citation_debug_payload(citations),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _answer_historical_official_lookup(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        result = self._historical_info.lookup(query=query, session_state=self._sessions.get_session_state(session_id=session_id))
        citations = build_external_citations(result.sources)
        answered = result.answer_status == "answered"
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="historical_official_lookup",
            query_intent=decision.query_intent,
            answer_text=result.answer_text,
            confidence=result.confidence if answered else 0.0,
            citations=citations,
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="answered" if answered else "abstained",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {
                    "used": True,
                    "provider_name": result.provider_name,
                    "lookup_type": result.lookup_type,
                    "answer_status": result.answer_status,
                    **dict(result.debug),
                },
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=bool(result.debug.get("official_web_attempted", False)),
                    general_web_attempted=bool(result.debug.get("general_web_attempted", False)),
                    final_route_mode="historical_official_lookup" if answered else "abstain",
                    citations=citations,
                    web_fallback_allowed=True,
                    web_fallback_not_allowed_reason=None,
                ),
                "citation_debug": _build_citation_debug_payload(citations),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _answer_current_official_lookup(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        return self._answer_current_lookup(
            query=query,
            session_id=session_id,
            decision=decision,
            requested_route_mode="current_official_lookup",
        )

    def _answer_current_lookup(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
        requested_route_mode: str,
    ) -> ChatAnswerPayload:
        result = self._current_info.lookup(query=query, session_state=self._sessions.get_session_state(session_id=session_id))
        return self._build_current_lookup_payload(
            query=query,
            session_id=session_id,
            decision=decision,
            requested_route_mode=requested_route_mode,
            result=result,
        )

    def _answer_corpus_metadata(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload:
        answer = self._corpus_stats.answer_query(query)
        if answer is None:
            payload = ChatAnswerPayload(
                session_id=session_id,
                route_mode="abstain",
                query_intent=decision.query_intent,
                answer_text="I could not determine that corpus-metadata answer from the stored manifest stats.",
                confidence=0.0,
                citations=(),
                retrieved_chunk_ids=(),
                active_record_keys=self._existing_active_record_keys(session_id),
                answer_status="abstained",
                debug={
                    **_build_route_debug_payload(decision.analysis),
                    "planner_debug": _build_planner_debug_payload(decision),
                    "corpus_metadata_debug": {"used": True, "answered": False},
                    "metadata_debug": {"used": False},
                    "current_lookup_debug": {"used": False},
                    "news_lookup_debug": {"used": False},
                    "historical_lookup_debug": {"used": False},
                    "web_fallback_debug": _build_web_fallback_debug(
                        structured_attempted=False,
                        corpus_attempted=False,
                        official_web_attempted=False,
                        general_web_attempted=False,
                        final_route_mode="abstain",
                        citations=(),
                        web_fallback_allowed=False,
                        web_fallback_not_allowed_reason="corpus_metadata_missing",
                    ),
                    "citation_debug": _build_citation_debug_payload(()),
                },
            )
            self._log_answer(
                session_id=session_id,
                query=query,
                payload=payload,
                extracted_filters=payload.debug,
                reranked_chunk_ids=(),
            )
            return payload

        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="corpus_metadata",
            query_intent=decision.query_intent,
            answer_text=answer.answer_text,
            confidence=0.95,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="answered",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "corpus_metadata_debug": {
                    "used": True,
                    **dict(answer.debug),
                },
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode="corpus_metadata",
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="corpus_metadata_answer",
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _maybe_answer_clarification_selection(
        self,
        *,
        query: str,
        session_id: UUID,
        session_state,
    ) -> ChatAnswerPayload | None:
        if not self._settings.enable_memory:
            return None
        selection = self._memory.resolve_clarification_selection(
            query=query,
            state=session_state,
        )
        if not selection.active_context:
            return None
        context = session_state.clarification_context if session_state is not None else None
        if context is None:
            return None
        if len(selection.selected_candidates) == 1:
            candidate = selection.selected_candidates[0]
            if candidate.candidate_type == "person":
                payload = self._answer_selected_person_candidate(
                    query=query,
                    session_id=session_id,
                    candidate=candidate,
                    context=context,
                )
            else:
                payload = self._answer_selected_matter_candidate(
                    query=query,
                    session_id=session_id,
                    candidate=candidate,
                    context=context,
                )
            return _with_clarification_debug(
                payload,
                active_context=True,
                selection_query=query,
                matched_candidates=(candidate,),
                resolved_candidate=candidate,
                match_reason=selection.match_reason,
            )
        clarify_candidates = selection.selected_candidates or context.candidates
        prompt_text = (
            "I still found more than one candidate matching that selection. Please choose one of these exact records:"
            if selection.selected_candidates
            else "I could not match that selection confidently. Please choose one of these exact records:"
        )
        synthetic_analysis = (
            self._router.decide(
                query=context.source_query,
                session_state=session_state,
            ).analysis
        )
        synthetic_decision = RouteDecision(
            route_mode="clarify",
            query_intent=context.source_query_intent,
            analysis=synthetic_analysis,
            reason_codes=("clarification_selection",),
        )
        return self._build_clarify_payload(
            query=context.source_query,
            session_id=session_id,
            decision=synthetic_decision,
            clarification_candidates=clarify_candidates,
            candidate_source="clarification_selection",
            answer_intro=prompt_text,
            source_query=context.source_query,
            source_route_mode=context.source_route_mode,
            source_query_intent=context.source_query_intent,
            clarification_debug={
                "active_context": True,
                "selection_query": query,
                "match_reason": selection.match_reason,
            },
        )

    def _answer_selected_person_candidate(
        self,
        *,
        query: str,
        session_id: UUID,
        candidate: ClarificationCandidate,
        context: ClarificationContext,
    ) -> ChatAnswerPayload:
        resolved_query = candidate.resolution_query or candidate.title
        decision = RouteDecision(
            route_mode="structured_current_info",
            query_intent=context.source_query_intent,
            analysis=self._router.decide(
                query=resolved_query,
                session_state=self._sessions.get_session_state(session_id=session_id),
            ).analysis,
            reason_codes=("clarification_selection",),
        )
        result = self._current_info.lookup(
            query=resolved_query,
            session_state=self._sessions.get_session_state(session_id=session_id),
        )
        payload = self._build_current_lookup_payload(
            query=query,
            session_id=session_id,
            decision=decision,
            requested_route_mode="structured_current_info",
            result=result,
        )
        return payload

    def _answer_selected_matter_candidate(
        self,
        *,
        query: str,
        session_id: UUID,
        candidate: ClarificationCandidate,
        context: ClarificationContext,
    ) -> ChatAnswerPayload:
        base_decision = self._router.decide(
            query=context.source_query,
            session_state=self._sessions.get_session_state(session_id=session_id),
        )
        forced_lock = StrictMatterLock(
            named_matter_query=True,
            strict_scope_required=True,
            strict_single_matter=True,
            ambiguous=False,
            locked_record_keys=((candidate.record_key,) if candidate.record_key else ()),
            candidates=(
                MatterLockCandidate(
                    record_key=candidate.record_key or candidate.candidate_id,
                    title=candidate.title,
                    bucket_name=candidate.bucket_name or "clarify",
                    document_version_id=candidate.document_version_id,
                    canonical_entities=(),
                    score=1.0,
                    exact_title_match=True,
                ),
            ),
            reason_codes=("clarification_selection",),
        )
        forced_analysis = replace(
            base_decision.analysis,
            strict_scope_required=True,
            strict_single_matter=True,
            strict_lock_record_keys=((candidate.record_key,) if candidate.record_key else ()),
            strict_lock_titles=(candidate.title,),
            strict_lock_reason_codes=("clarification_selection",),
            strict_lock_ambiguous=False,
            strict_matter_lock=forced_lock,
        )
        forced_decision = RouteDecision(
            route_mode="exact_lookup",
            query_intent=context.source_query_intent,
            analysis=forced_analysis,
            reason_codes=("clarification_selection",),
        )
        state = self._sessions.get_session_state(session_id=session_id)
        return self._answer_with_retrieval(
            query=context.source_query,
            session_id=session_id,
            decision=forced_decision,
            grounded_summary=state.grounded_summary if state is not None else None,
        )

    def _maybe_answer_internal_person_priority(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload | None:
        preview = getattr(self._current_info, "preview_internal_person_priority", None)
        if not callable(preview):
            return None
        result = preview(
            query=query,
            session_state=self._sessions.get_session_state(session_id=session_id),
        )
        if result is None:
            return None
        override_decision = RouteDecision(
            route_mode="structured_current_info",
            query_intent="structured_current_info",
            analysis=decision.analysis,
            reason_codes=tuple(dict.fromkeys((*decision.reason_codes, "internal_person_priority"))),
        )
        return self._build_current_lookup_payload(
            query=query,
            session_id=session_id,
            decision=override_decision,
            requested_route_mode="structured_current_info",
            result=result,
            additional_debug={
                "internal_person_priority_debug": {
                    "used": True,
                    "blocked_general_or_web_route": True,
                    "blocked_reason": "strong_internal_structured_person_candidate",
                    "answer_status": result.answer_status,
                    "lookup_type": result.lookup_type,
                    "matched_people_rows_count": int(result.debug.get("matched_people_rows_count") or 0),
                    "person_match_status": result.debug.get("person_match_status"),
                    "fuzzy_band": result.debug.get("fuzzy_band"),
                },
            },
            web_fallback_not_allowed_reason_override="internal_person_priority_override",
        )

    def _build_current_lookup_payload(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
        requested_route_mode: str,
        result,
        additional_debug: dict[str, Any] | None = None,
        web_fallback_not_allowed_reason_override: str | None = None,
    ) -> ChatAnswerPayload:
        clarification_candidates = _extract_current_lookup_clarification_candidates(
            query=query,
            result=result,
        )
        if clarification_candidates:
            return self._build_clarify_payload(
                query=query,
                session_id=session_id,
                decision=decision,
                clarification_candidates=clarification_candidates,
                candidate_source="structured_current_info",
                answer_intro=result.answer_text,
                source_query=query,
                source_route_mode=requested_route_mode,
                source_query_intent=decision.query_intent,
            )
        citations = build_external_citations(result.sources)
        route_mode = _resolve_current_lookup_route_mode(
            requested_route_mode=requested_route_mode,
            result=result,
            citations=citations,
        )
        answer_status = "answered" if route_mode == "current_official_lookup" else "abstained"
        if route_mode == "structured_current_info":
            answer_status = "answered"
        payload_debug = {
            **_build_route_debug_payload(decision.analysis),
            "planner_debug": _build_planner_debug_payload(decision),
            "metadata_debug": {"used": False},
            "current_lookup_debug": {
                "used": True,
                "provider_name": result.provider_name,
                "lookup_type": result.lookup_type,
                "answer_status": result.answer_status,
                "source_urls": [source.url for source in result.sources],
                "source_types": list(
                    _ordered_unique(source.source_type for source in result.sources)
                ),
                "source_domains": list(
                    _ordered_unique(
                        source.domain or extract_domain(source.url) for source in result.sources
                    )
                ),
                **dict(result.debug),
            },
            "news_lookup_debug": {"used": False},
            "historical_lookup_debug": {"used": False},
            "web_fallback_debug": _build_web_fallback_debug(
                structured_attempted=bool(result.debug.get("structured_attempted", False)),
                corpus_attempted=False,
                official_web_attempted=bool(result.debug.get("official_web_attempted", False)),
                general_web_attempted=bool(result.debug.get("general_web_attempted", False)),
                final_route_mode=route_mode,
                citations=citations,
                web_fallback_allowed=bool(result.debug.get("web_fallback_allowed", True)),
                web_fallback_not_allowed_reason=(
                    web_fallback_not_allowed_reason_override
                    or result.debug.get("web_fallback_not_allowed_reason")
                ),
            ),
            "citation_debug": _build_citation_debug_payload(citations),
        }
        if additional_debug:
            payload_debug.update(additional_debug)
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode=route_mode,
            query_intent=decision.query_intent,
            answer_text=result.answer_text,
            confidence=result.confidence if route_mode != "abstain" else 0.0,
            citations=citations,
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status=answer_status,
            debug=payload_debug,
        )
        if self._settings.enable_memory:
            query_family = str(result.debug.get("detected_query_family") or result.lookup_type or "")
            if query_family == "office_contact":
                self._memory.update_current_lookup_context(
                    session_id=session_id,
                    family="office_contact",
                    focus=str(result.debug.get("extracted_city") or result.debug.get("matched_offices") or ""),
                    query=query,
                )
            else:
                self._memory.clear_current_lookup_context(session_id=session_id)
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _build_non_corpus_payload(
        self,
        *,
        session_id: UUID,
        query: str,
        decision: RouteDecision,
        route_mode: str,
        answer_text: str,
        confidence: float,
    ) -> ChatAnswerPayload:
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode=route_mode,
            query_intent=decision.query_intent,
            answer_text=answer_text,
            confidence=confidence,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="answered",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "metadata_debug": {"used": False},
                "current_lookup_debug": {
                    "used": False,
                },
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "metadata_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode=route_mode,
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="non_corpus_static_route",
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _build_clarify_payload(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
        clarification_candidates: tuple[ClarificationCandidate, ...],
        candidate_source: str,
        answer_intro: str,
        source_query: str,
        source_route_mode: str,
        source_query_intent: str,
        clarification_debug: dict[str, Any] | None = None,
    ) -> ChatAnswerPayload:
        rendered_candidates = render_clarification_candidate_lines(clarification_candidates)
        answer_lines = [answer_intro.strip()]
        if rendered_candidates:
            answer_lines.extend(rendered_candidates)
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="clarify",
            query_intent=decision.query_intent,
            answer_text="\n".join(line for line in answer_lines if line),
            confidence=0.0,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=self._existing_active_record_keys(session_id),
            answer_status="clarify",
            clarification_candidates=clarification_candidates,
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "candidate_list_debug": {
                    "used": True,
                    "candidate_source": candidate_source,
                    "candidate_count": len(clarification_candidates),
                    "record_keys": [
                        candidate.record_key
                        for candidate in clarification_candidates
                        if candidate.record_key
                    ],
                    "bucket_names": [
                        candidate.bucket_name
                        for candidate in clarification_candidates
                        if candidate.bucket_name
                    ],
                },
                "clarification_debug": {
                    "used": True,
                    "active_context": True,
                    "source_query": source_query,
                    "source_route_mode": source_route_mode,
                    "source_query_intent": source_query_intent,
                    "candidate_type": (
                        clarification_candidates[0].candidate_type
                        if clarification_candidates
                        else None
                    ),
                    **(clarification_debug or {}),
                },
                "corpus_metadata_debug": {"used": False},
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode="clarify",
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="clarification_required",
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )
        if self._settings.enable_memory and clarification_candidates:
            self._memory.update_clarification_context(
                session_id=session_id,
                context=ClarificationContext(
                    source_query=source_query,
                    source_route_mode=source_route_mode,
                    source_query_intent=source_query_intent,
                    candidate_type=clarification_candidates[0].candidate_type,
                    candidates=clarification_candidates,
                ),
            )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _build_metadata_payload(
        self,
        *,
        session_id: UUID,
        query: str,
        decision: RouteDecision,
        route_mode: str,
        metadata_answer,
    ) -> ChatAnswerPayload:
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode=route_mode,
            query_intent=decision.query_intent,
            answer_text=metadata_answer.answer_text,
            confidence=0.91,
            citations=metadata_answer.citations,
            retrieved_chunk_ids=(),
            active_record_keys=tuple(_ordered_unique(citation.record_key for citation in metadata_answer.citations)),
            answer_status="answered",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "planner_debug": _build_planner_debug_payload(decision),
                "metadata_debug": {
                    "used": True,
                    **dict(metadata_answer.debug),
                },
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode=route_mode,
                    citations=metadata_answer.citations,
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="metadata_answer",
                ),
                "citation_debug": _build_citation_debug_payload(metadata_answer.citations),
            },
        )
        if self._settings.enable_memory and metadata_answer.citations:
            self._memory.update_active_scope(
                session_id=session_id,
                document_ids=(),
                document_version_ids=_ordered_unique_values(
                    citation.document_version_id
                    for citation in metadata_answer.citations
                    if citation.document_version_id is not None
                ),
                record_keys=payload.active_record_keys,
            )
            payload = self._with_hydrated_active_matter(
                payload=payload,
                session_id=session_id,
                decision=decision,
                citations=metadata_answer.citations,
            )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _existing_active_record_keys(self, session_id: UUID) -> tuple[str, ...]:
        existing_state = self._sessions.get_session_state(session_id=session_id)
        return existing_state.active_record_keys if existing_state else ()

    def _with_hydrated_active_matter(
        self,
        *,
        payload: ChatAnswerPayload,
        session_id: UUID,
        decision: RouteDecision,
        citations: tuple[Any, ...],
    ) -> ChatAnswerPayload:
        document_version_ids = _ordered_unique_values(
            citation.document_version_id
            for citation in citations
            if citation.document_version_id is not None
        )
        if not document_version_ids:
            return payload

        try:
            metadata_rows = self._metadata_repository.fetch_order_metadata(
                document_version_ids=document_version_ids
            )
            legal_rows = self._metadata_repository.fetch_legal_provisions(
                document_version_ids=document_version_ids
            )
        except Exception:
            return payload
        primary_row = metadata_rows[0] if metadata_rows else None
        primary_title = primary_row.title if primary_row is not None else (citations[0].title if citations else None)
        primary_entity = (
            decision.analysis.strict_lock_matched_entities[0]
            if decision.analysis.strict_lock_matched_entities
            else _infer_main_entity(primary_title)
        )
        active_record_keys = (
            _ordered_unique_values(row.record_key for row in metadata_rows)
            or _ordered_unique_values(citation.record_key for citation in citations)
        )
        active_document_ids = _ordered_unique_values(row.document_id for row in metadata_rows)
        active_entities = _ordered_unique_values(
            value
            for value in (
                primary_entity,
                *(decision.analysis.strict_lock_matched_entities or ()),
            )
            if value
        )
        legal_provisions = _ordered_unique_values(
            f"{row.section_or_regulation} of {row.statute_name}"
            for row in legal_rows
        )
        updated_state = self._memory.update_active_matter_context(
            session_id=session_id,
            document_ids=active_document_ids,
            document_version_ids=document_version_ids,
            record_keys=active_record_keys,
            entities=active_entities,
            bucket_names=(),
            primary_title=primary_title,
            primary_entity=primary_entity,
            signatory_name=primary_row.signatory_name if primary_row is not None else None,
            signatory_designation=primary_row.signatory_designation if primary_row is not None else None,
            order_date=primary_row.order_date if primary_row is not None else None,
            order_place=primary_row.place if primary_row is not None else None,
            legal_provisions=legal_provisions,
        )
        return ChatAnswerPayload(
            session_id=payload.session_id,
            route_mode=payload.route_mode,
            query_intent=payload.query_intent,
            answer_text=payload.answer_text,
            confidence=payload.confidence,
            citations=payload.citations,
            retrieved_chunk_ids=payload.retrieved_chunk_ids,
            active_record_keys=updated_state.active_record_keys,
            answer_status=payload.answer_status,
            debug=payload.debug,
        )

    def _maybe_build_exact_lookup_candidate_payload(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload | None:
        candidates = tuple(
            self._retrieval.find_exact_lookup_candidates(
                query=query,
                limit=max(self._settings.retrieval_top_k_docs, 5),
                query_variants=decision.analysis.normalized_expansions or None,
            )
        )
        if not candidates:
            return None
        sat_court_query = looks_like_sat_court_query(
            query,
            sat_court_signals=decision.analysis.sat_court_signals,
        )
        resolution = select_exact_lookup_resolution(
            candidates,
            sat_court_query=sat_court_query,
            source_query=query,
        )
        if not resolution.should_clarify:
            return None
        return self._build_candidate_list_payload(
            query=query,
            session_id=session_id,
            decision=decision,
            candidates=resolution.ordered_candidates,
            candidate_source="exact_lookup",
        )

    def _build_candidate_list_payload(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
        candidates: tuple[Any, ...],
        candidate_source: str,
    ) -> ChatAnswerPayload | None:
        clarification_candidates = build_matter_clarification_candidates(
            candidates=candidates,
            source_query=query,
            control_pack=self._control_pack,
        )
        if not clarification_candidates:
            return None
        return self._build_clarify_payload(
            query=query,
            session_id=session_id,
            decision=decision,
            clarification_candidates=clarification_candidates,
            candidate_source=candidate_source,
            answer_intro=_CANDIDATE_LIST_INTRO,
            source_query=query,
            source_route_mode=decision.route_mode,
            source_query_intent=decision.query_intent,
        )

    def _answer_with_retrieval(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
        grounded_summary: str | None,
    ) -> ChatAnswerPayload:
        forced_single_candidate_decision = self._maybe_force_single_candidate_lock(
            query=query,
            decision=decision,
        )
        if forced_single_candidate_decision is not None:
            decision = forced_single_candidate_decision

        allow_session_scoped_override = bool(
            decision.route_mode == "memory_scoped_rag"
            and decision.analysis.has_session_scope
            and decision.analysis.active_order_override
        )
        if (
            decision.analysis.strict_scope_required
            and not decision.analysis.strict_single_matter
            and not allow_session_scoped_override
        ):
            candidate_payload = self._build_candidate_list_payload(
                query=query,
                session_id=session_id,
                decision=decision,
                candidates=tuple(decision.analysis.strict_matter_lock.candidates),
                candidate_source="strict_matter_lock",
            )
            if candidate_payload is not None:
                return candidate_payload
            payload = ChatAnswerPayload(
                session_id=session_id,
                route_mode="abstain",
                query_intent=decision.query_intent,
                answer_text=_AMBIGUOUS_NAMED_MATTER_ANSWER,
                confidence=0.0,
                citations=(),
                retrieved_chunk_ids=(),
                active_record_keys=(),
                answer_status="abstained",
                debug={
                    **_build_route_debug_payload(decision.analysis),
                    "mixed_record_guardrail": {
                        "guardrail_fired": True,
                        "reason_codes": ["strict_scope_unresolved"],
                    },
                    "metadata_debug": {"used": False},
                    "current_lookup_debug": {"used": False},
                    "news_lookup_debug": {"used": False},
                    "historical_lookup_debug": {"used": False},
                    "web_fallback_debug": _build_web_fallback_debug(
                        structured_attempted=False,
                        corpus_attempted=False,
                        official_web_attempted=False,
                        general_web_attempted=False,
                        final_route_mode="abstain",
                        citations=(),
                        web_fallback_allowed=False,
                        web_fallback_not_allowed_reason="ambiguous_named_matter_no_web_override",
                    ),
                    "citation_debug": _build_citation_debug_payload(()),
                },
            )
            self._log_answer(
                session_id=session_id,
                query=query,
                payload=payload,
                extracted_filters=payload.debug,
                reranked_chunk_ids=(),
            )
            return payload

        metadata_payload = self._answer_from_metadata_if_possible(
            query=query,
            session_id=session_id,
            decision=decision,
        )
        if metadata_payload is not None:
            return metadata_payload

        if decision.route_mode == "exact_lookup":
            if not (
                decision.analysis.strict_single_matter
                and decision.analysis.strict_lock_record_keys
            ):
                candidate_payload = self._maybe_build_exact_lookup_candidate_payload(
                    query=query,
                    session_id=session_id,
                    decision=decision,
                )
                if candidate_payload is not None:
                    return candidate_payload

        execution = self._execute_retrieval(query=query, decision=decision, session_id=session_id)
        retrieved_chunk_ids = tuple(chunk.chunk_id for chunk in execution.search_result.chunks)
        prompt_chunk_hits = execution.search_result.chunks
        guardrail_debug: dict[str, Any] = {
            "mixed_record_guardrail_fired": False,
            "retrieved_record_keys_before_filter": list(
                _ordered_unique(chunk.record_key for chunk in execution.search_result.chunks)
            ),
        }
        if decision.analysis.strict_single_matter and decision.analysis.strict_lock_record_keys:
            filtered_hits = filter_items_to_locked_record_keys(
                execution.search_result.chunks,
                locked_record_keys=decision.analysis.strict_lock_record_keys,
            )
            if filtered_hits and len(filtered_hits) != len(execution.search_result.chunks):
                prompt_chunk_hits = filtered_hits
                guardrail_debug["mixed_record_guardrail_fired"] = True
                guardrail_debug["guardrail_action"] = "locked_matter_only_context"
        context_chunks = self._select_context_chunks(
            execution.search_result,
            chunk_hits=prompt_chunk_hits,
            analysis=decision.analysis,
        )
        reranked_chunk_ids = tuple(chunk.chunk_id for chunk in context_chunks)

        if not context_chunks:
            payload = ChatAnswerPayload(
                session_id=session_id,
                route_mode="abstain",
                query_intent=decision.query_intent,
                answer_text=(
                    _STRICT_MATTER_WEAK_SUPPORT_ANSWER
                    if decision.analysis.strict_single_matter
                    else _EMPTY_RETRIEVAL_ANSWER
                ),
                confidence=0.0,
                citations=(),
                retrieved_chunk_ids=retrieved_chunk_ids,
                active_record_keys=(),
                answer_status="abstained",
                debug={
                    **execution.extracted_filters,
                    **_build_route_debug_payload(decision.analysis),
                    "mixed_record_guardrail": guardrail_debug,
                    "metadata_debug": {"used": False},
                    "current_lookup_debug": {"used": False},
                    "news_lookup_debug": {"used": False},
                    "historical_lookup_debug": {"used": False},
                    "web_fallback_debug": _build_web_fallback_debug(
                        structured_attempted=False,
                        corpus_attempted=True,
                        official_web_attempted=False,
                        general_web_attempted=False,
                        final_route_mode="abstain",
                        citations=(),
                        web_fallback_allowed=False,
                        web_fallback_not_allowed_reason=(
                            "named_matter_no_web_override"
                            if decision.analysis.strict_single_matter
                            else "corpus_route_no_web_fallback"
                        ),
                    ),
                    "citation_debug": _build_citation_debug_payload(()),
                },
            )
            self._log_answer(
                session_id=session_id,
                query=query,
                payload=payload,
                extracted_filters=payload.debug,
                reranked_chunk_ids=reranked_chunk_ids,
            )
            return payload

        prompt = build_retrieval_answer_prompt(
            query=query,
            context_chunks=context_chunks,
            analysis=decision.analysis,
            grounded_summary=grounded_summary,
            strict_rule_text=(
                self._control_pack.strict_answer_rule.text if self._control_pack else None
            ),
            locked_record_keys=decision.analysis.strict_lock_record_keys,
            comparison_allowed=decision.analysis.comparison_intent,
        )
        response = self._llm.complete_json(prompt)
        answer_text = str(response.get("answer_text", "")).strip()
        answer_status = _normalize_answer_status(response.get("answer_status"))
        declared_cited_numbers = _normalize_cited_numbers(response.get("cited_numbers"))
        payload = self._build_grounded_payload(
            session_id=session_id,
            query_intent=decision.query_intent,
            initial_route_mode=execution.route_mode,
            answer_text=answer_text,
            answer_status=answer_status,
            declared_cited_numbers=declared_cited_numbers,
            context_chunks=context_chunks,
            retrieved_chunks=execution.search_result.chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
            analysis=decision.analysis,
            debug_payload={
                **execution.extracted_filters,
                **_build_route_debug_payload(decision.analysis),
                "mixed_record_guardrail": guardrail_debug,
            },
        )
        if payload.route_mode == "abstain":
            fallback_payload = self._maybe_build_document_lookup_fallback(
                session_id=session_id,
                decision=decision,
                context_chunks=context_chunks,
                retrieved_chunk_ids=retrieved_chunk_ids,
                debug_payload={
                    **execution.extracted_filters,
                    **_build_route_debug_payload(decision.analysis),
                    "mixed_record_guardrail": guardrail_debug,
                },
            )
            if fallback_payload is not None:
                payload = fallback_payload
        if payload.route_mode != "abstain" and payload.citations and self._settings.enable_memory:
            updated_state = self._memory.update_from_grounded_answer(
                session_id=session_id,
                context_chunks=context_chunks,
                citations=payload.citations,
            )
            payload = self._with_hydrated_active_matter(
                payload=payload,
                session_id=session_id,
                decision=decision,
                citations=payload.citations,
            )
            hydrated_state = self._sessions.get_session_state(session_id=session_id) or updated_state
            payload = ChatAnswerPayload(
                session_id=payload.session_id,
                route_mode=payload.route_mode,
                query_intent=payload.query_intent,
                answer_text=payload.answer_text,
                confidence=payload.confidence,
                citations=payload.citations,
                retrieved_chunk_ids=payload.retrieved_chunk_ids,
                active_record_keys=hydrated_state.active_record_keys,
                answer_status=payload.answer_status,
                debug=payload.debug,
            )
        if self._settings.enable_memory:
            self._memory.clear_current_lookup_context(session_id=session_id)
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=reranked_chunk_ids,
        )
        return payload

    def _maybe_force_single_candidate_lock(
        self,
        *,
        query: str,
        decision: RouteDecision,
    ) -> RouteDecision | None:
        strict_lock = decision.analysis.strict_matter_lock
        if decision.analysis.strict_single_matter or not strict_lock.strict_scope_required:
            return None
        if len(strict_lock.candidates) != 1:
            return None
        candidate = strict_lock.candidates[0]
        if not _single_candidate_auto_resolve_allowed(candidate):
            return None
        forced_lock = StrictMatterLock(
            named_matter_query=True,
            strict_scope_required=True,
            strict_single_matter=True,
            ambiguous=False,
            comparison_intent=strict_lock.comparison_intent,
            comparison_terms=strict_lock.comparison_terms,
            matched_aliases=strict_lock.matched_aliases,
            matched_entities=strict_lock.matched_entities,
            matched_titles=(candidate.title,),
            locked_record_keys=(candidate.record_key,),
            candidates=(candidate,),
            reason_codes=tuple(dict.fromkeys((*strict_lock.reason_codes, "single_candidate_auto_resolve"))),
        )
        forced_analysis = replace(
            decision.analysis,
            query_family="named_order_query",
            strict_single_matter=True,
            strict_lock_record_keys=(candidate.record_key,),
            strict_lock_titles=(candidate.title,),
            strict_lock_reason_codes=forced_lock.reason_codes,
            strict_lock_ambiguous=False,
            strict_matter_lock=forced_lock,
        )
        return RouteDecision(
            route_mode="exact_lookup",
            query_intent=decision.query_intent,
            analysis=forced_analysis,
            reason_codes=tuple(dict.fromkeys((*decision.reason_codes, "single_candidate_auto_resolve"))),
        )

    def _answer_from_metadata_if_possible(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> ChatAnswerPayload | None:
        analysis = decision.analysis
        if not (
            analysis.asks_order_signatory
            or analysis.asks_order_date
            or analysis.asks_legal_provisions
            or analysis.asks_provision_explanation
            or analysis.asks_order_pan
            or analysis.asks_order_amount
            or analysis.asks_order_holding
            or analysis.asks_order_parties
            or analysis.asks_order_observations
            or analysis.asks_order_numeric_fact
            or analysis.active_matter_follow_up_intent is not None
        ):
            return None
        document_version_ids, candidate_payload = self._resolve_metadata_document_version_ids(
            query=query,
            session_id=session_id,
            decision=decision,
        )
        if candidate_payload is not None:
            return candidate_payload
        if not document_version_ids:
            return None
        if analysis.asks_order_signatory:
            answer = self._metadata.answer_signatory_question(
                document_version_ids=document_version_ids,
            )
            if answer is not None:
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=answer,
                )
        if analysis.asks_order_date:
            answer = self._metadata.answer_order_date_question(
                document_version_ids=document_version_ids,
            )
            if answer is not None:
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=answer,
                )
        if analysis.asks_order_numeric_fact:
            answer = self._metadata.answer_numeric_fact_question(
                query=query,
                document_version_ids=document_version_ids,
            )
            if answer is not None:
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=answer,
                )
            missing_numeric_payload = self._maybe_answer_missing_numeric_fact(
                query=query,
                session_id=session_id,
                decision=decision,
                document_version_ids=document_version_ids,
            )
            if missing_numeric_payload is not None:
                return missing_numeric_payload
        if analysis.asks_legal_provisions or analysis.asks_provision_explanation:
            answer = self._metadata.answer_legal_provisions_question(
                document_version_ids=document_version_ids,
                explain=analysis.asks_provision_explanation,
            )
            if answer is not None:
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=answer,
                )
        if (
            analysis.active_matter_follow_up_intent
            and analysis.active_matter_follow_up_intent != "da_observed"
        ):
            answer = self._metadata.answer_active_matter_follow_up(
                query=query,
                document_version_ids=document_version_ids,
                follow_up_intent=analysis.active_matter_follow_up_intent,
            )
            if answer is not None:
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=answer,
                )
            missing_scope_payload = self._maybe_answer_missing_active_matter_concept(
                query=query,
                session_id=session_id,
                decision=decision,
                follow_up_intent=analysis.active_matter_follow_up_intent,
            )
            if missing_scope_payload is not None:
                return missing_scope_payload
        if (
            analysis.asks_order_pan
            or analysis.asks_order_amount
            or analysis.asks_order_holding
            or analysis.asks_order_parties
        ):
            answer = self._metadata.answer_exact_fact_question(
                query=query,
                document_version_ids=document_version_ids,
            )
            if answer is not None:
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=answer,
                )
        if analysis.asks_order_observations and (
            analysis.active_order_override or analysis.strict_single_matter
        ):
            answer = self._metadata.answer_observation_question(
                query=query,
                document_version_ids=document_version_ids,
            )
            if answer is not None:
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=answer,
                )
        return None

    def _maybe_answer_missing_active_matter_concept(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
        follow_up_intent: str,
    ) -> ChatAnswerPayload | None:
        if decision.route_mode != "memory_scoped_rag":
            return None
        state = self._sessions.get_session_state(session_id=session_id)
        if state is None or not state.active_record_keys:
            return None
        if _active_scope_supports_follow_up_intent(state=state, follow_up_intent=follow_up_intent):
            return None
        answer_text = _render_missing_active_matter_concept_answer(
            state=state,
            follow_up_intent=follow_up_intent,
        )
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="memory_scoped_rag",
            query_intent=decision.query_intent,
            answer_text=answer_text,
            confidence=0.88,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=state.active_record_keys,
            answer_status="answered",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "metadata_debug": {
                    "used": True,
                    "metadata_type": f"active_matter_{follow_up_intent}_missing_concept",
                    "follow_up_intent": follow_up_intent,
                    "active_scope_negative_answer": True,
                },
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode="memory_scoped_rag",
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="active_matter_missing_concept",
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _maybe_answer_missing_numeric_fact(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
        document_version_ids: tuple[int, ...],
    ) -> ChatAnswerPayload | None:
        normalized_query = " ".join(query.lower().split())
        if "ipo proceeds" not in normalized_query:
            return None
        if not document_version_ids or not decision.analysis.strict_single_matter:
            return None
        metadata_rows = self._metadata_repository.fetch_order_metadata(
            document_version_ids=document_version_ids
        )
        load_chunks = getattr(self._metadata_repository, "load_chunks", None)
        if metadata_rows and callable(load_chunks):
            for row in metadata_rows:
                chunks = tuple(load_chunks(document_version_id=row.document_version_id))
                supporting_chunk = _select_missing_ipo_proceeds_chunk(chunks)
                if supporting_chunk is None:
                    continue
                citation = Citation(
                    citation_number=1,
                    record_key=row.record_key,
                    title=row.title,
                    page_start=supporting_chunk.page_start,
                    page_end=supporting_chunk.page_end,
                    section_type="metadata_exact_fact",
                    document_version_id=row.document_version_id,
                    chunk_id=supporting_chunk.chunk_id,
                    detail_url=row.detail_url,
                    pdf_url=row.pdf_url,
                )
                prompt_chunk = PromptContextChunk(
                    citation_number=1,
                    chunk_id=supporting_chunk.chunk_id,
                    document_version_id=row.document_version_id,
                    document_id=row.document_id,
                    record_key=row.record_key,
                    bucket_name="metadata",
                    title=row.title,
                    page_start=supporting_chunk.page_start,
                    page_end=supporting_chunk.page_end,
                    section_type=supporting_chunk.section_type or "facts",
                    section_title=supporting_chunk.section_title,
                    detail_url=row.detail_url,
                    pdf_url=row.pdf_url,
                    chunk_text=supporting_chunk.text,
                    token_count=max(len(supporting_chunk.text.split()), 1),
                    score=1.0,
                )
                answer_text, style_debug = apply_grounded_wording_caution(
                    answer_text="The cited order describes the transaction addressed in this matter.",
                    context_chunks=(prompt_chunk,),
                    analysis=decision.analysis,
                )
                return self._build_metadata_payload(
                    session_id=session_id,
                    query=query,
                    decision=decision,
                    route_mode=decision.route_mode,
                    metadata_answer=MetadataAnswer(
                        answer_text=answer_text,
                        citations=(citation,),
                        metadata_type="missing_numeric_fact",
                        debug={
                            "document_version_id": row.document_version_id,
                            "record_key": row.record_key,
                            "metadata_type": "missing_numeric_fact",
                            "missing_fact_query": "ipo_proceeds",
                            "style_debug": style_debug,
                        },
                    ),
                )
        payload = ChatAnswerPayload(
            session_id=session_id,
            route_mode="abstain",
            query_intent=decision.query_intent,
            answer_text=(
                "The cited order does not state IPO proceeds for this matter, so I cannot answer that from this record."
            ),
            confidence=0.0,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=decision.analysis.strict_lock_record_keys,
            answer_status="abstained",
            debug={
                **_build_route_debug_payload(decision.analysis),
                "metadata_debug": {
                    "used": True,
                    "metadata_type": "missing_numeric_fact",
                    "missing_fact_query": "ipo_proceeds",
                },
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=False,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode="abstain",
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="missing_numeric_fact_in_locked_matter",
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )
        self._log_answer(
            session_id=session_id,
            query=query,
            payload=payload,
            extracted_filters=payload.debug,
            reranked_chunk_ids=(),
        )
        return payload

    def _maybe_build_document_lookup_fallback(
        self,
        *,
        session_id: UUID,
        decision: RouteDecision,
        context_chunks: tuple[PromptContextChunk, ...],
        retrieved_chunk_ids: tuple[int, ...],
        debug_payload: dict[str, Any],
    ) -> ChatAnswerPayload | None:
        exact_title_locked = "exact_title_or_contained_match" in tuple(
            decision.analysis.strict_lock_reason_codes
        )
        if decision.query_intent != "document_lookup" and not exact_title_locked:
            return None
        if not decision.analysis.strict_single_matter or not context_chunks:
            return None
        title = context_chunks[0].title or (
            decision.analysis.strict_lock_titles[0]
            if decision.analysis.strict_lock_titles
            else None
        )
        if not title:
            return None
        citations = build_citations(context_chunks[:1])
        return ChatAnswerPayload(
            session_id=session_id,
            route_mode="exact_lookup",
            query_intent=decision.query_intent,
            answer_text=f'This refers to the SEBI matter titled "{title}".',
            confidence=0.62,
            citations=citations,
            retrieved_chunk_ids=retrieved_chunk_ids,
            active_record_keys=decision.analysis.strict_lock_record_keys,
            answer_status="answered",
            debug={
                **debug_payload,
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=False,
                    corpus_attempted=True,
                    official_web_attempted=False,
                    general_web_attempted=False,
                    final_route_mode="exact_lookup",
                    citations=citations,
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="strict_document_lookup_fallback",
                ),
                "citation_debug": _build_citation_debug_payload(citations),
                "document_lookup_fallback": {"used": True},
            },
        )

    def _metadata_document_version_ids(
        self,
        *,
        session_id: UUID,
        decision: RouteDecision,
    ) -> tuple[int, ...]:
        state = self._sessions.get_session_state(session_id=session_id)
        if decision.route_mode == "memory_scoped_rag" and state is not None:
            filters = self._memory.build_memory_filters(state=state)
            if filters.document_version_ids:
                return tuple(filters.document_version_ids)
        if decision.analysis.strict_lock_record_keys:
            return self._retrieval.resolve_current_document_version_ids(
                record_keys=decision.analysis.strict_lock_record_keys,
            )
        if decision.analysis.strict_matter_lock.candidates:
            return tuple(
                dict.fromkeys(
                    candidate.document_version_id
                    for candidate in decision.analysis.strict_matter_lock.candidates
                    if candidate.document_version_id is not None
                )
            )
        return ()

    def _resolve_metadata_document_version_ids(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision,
    ) -> tuple[tuple[int, ...], ChatAnswerPayload | None]:
        document_version_ids = self._metadata_document_version_ids(
            session_id=session_id,
            decision=decision,
        )
        if document_version_ids:
            return document_version_ids, None

        lookup_variants = build_order_lookup_variants(query=query, analysis=decision.analysis)
        if not lookup_variants:
            return (), None

        candidates = tuple(
            self._retrieval.find_exact_lookup_candidates(
                query=lookup_variants[0],
                limit=max(self._settings.retrieval_top_k_docs, 5),
                query_variants=lookup_variants,
            )
        )
        if not candidates:
            return (), None

        resolution = select_exact_lookup_resolution(
            candidates,
            sat_court_query=looks_like_sat_court_query(
                query,
                sat_court_signals=decision.analysis.sat_court_signals,
            ),
            source_query=lookup_variants[0],
        )
        if resolution.selected_document_id is not None:
            return (resolution.selected_document_id,), None
        if resolution.should_clarify:
            return (), self._build_candidate_list_payload(
                query=query,
                session_id=session_id,
                decision=decision,
                candidates=resolution.ordered_candidates,
                candidate_source="order_metadata",
            )
        return (), None

    def _execute_retrieval(
        self,
        *,
        query: str,
        decision: RouteDecision,
        session_id: UUID,
    ) -> _RetrievalExecution:
        if decision.route_mode == "exact_lookup":
            return self._run_exact_lookup(query=query, analysis=decision.analysis)
        if decision.route_mode == "memory_scoped_rag":
            state = self._sessions.get_session_state(session_id=session_id)
            return self._run_memory_scoped(
                query=query,
                session_state=state,
                analysis=decision.analysis,
            )
        search_result = self._search_with_optional_lock(
            query=query,
            strict_matter_lock=decision.analysis.strict_matter_lock,
        )
        return _RetrievalExecution(
            route_mode="hierarchical_rag",
            search_result=search_result,
            extracted_filters={
                **_build_search_debug_payload(search_result),
                **_build_route_debug_payload(decision.analysis),
            },
        )

    def _run_exact_lookup(self, *, query: str, analysis: QueryAnalysis) -> _RetrievalExecution:
        if analysis.strict_single_matter and analysis.strict_lock_record_keys:
            filters = MetadataFilterInput(record_key=analysis.strict_lock_record_keys[0])
            search_result = self._search_with_optional_lock(
                query=query,
                filters=filters,
                top_k_docs=1,
                strict_matter_lock=analysis.strict_matter_lock,
            )
            return _RetrievalExecution(
                route_mode="exact_lookup",
                search_result=search_result,
                extracted_filters={
                    "document_record_key": analysis.strict_lock_record_keys[0],
                    **_build_search_debug_payload(search_result),
                    **_build_route_debug_payload(analysis),
                },
            )

        candidates = self._retrieval.find_exact_lookup_candidates(
            query=query,
            limit=max(self._settings.retrieval_top_k_docs, 5),
            query_variants=analysis.normalized_expansions or None,
        )
        resolution = select_exact_lookup_resolution(
            candidates,
            sat_court_query=looks_like_sat_court_query(
                query,
                sat_court_signals=analysis.sat_court_signals,
            ),
            source_query=query,
        )
        if resolution.selected_document_id is None:
            fallback_result = self._search_with_optional_lock(
                query=query,
                strict_matter_lock=analysis.strict_matter_lock,
            )
            return _RetrievalExecution(
                route_mode="hierarchical_rag",
                search_result=fallback_result,
                extracted_filters={
                    **_build_search_debug_payload(fallback_result),
                    **_build_route_debug_payload(analysis),
                },
            )

        filters = MetadataFilterInput(document_version_ids=(resolution.selected_document_id,))
        search_result = self._search_with_optional_lock(
            query=query,
            filters=filters,
            top_k_docs=1,
            strict_matter_lock=analysis.strict_matter_lock,
        )
        return _RetrievalExecution(
            route_mode="exact_lookup",
            search_result=search_result,
            extracted_filters={
                "document_version_ids": [resolution.selected_document_id],
                "exact_lookup_debug": {
                    "sat_court_query": looks_like_sat_court_query(
                        query,
                        sat_court_signals=analysis.sat_court_signals,
                    ),
                    "candidate_bucket_priors_applied": bool(analysis.sat_court_signals),
                },
                "exact_lookup_match_scores": [
                    {
                        "document_version_id": candidate.document_version_id,
                        "record_key": getattr(candidate, "record_key", None),
                        "bucket_name": getattr(candidate, "bucket_name", None),
                        "match_score": candidate.match_score,
                    }
                    for candidate in resolution.ordered_candidates[:3]
                ],
                **_build_search_debug_payload(search_result),
                **_build_route_debug_payload(analysis),
            },
        )

    def _run_memory_scoped(
        self,
        *,
        query: str,
        session_state: Any,
        analysis: QueryAnalysis,
    ) -> _RetrievalExecution:
        if session_state is not None:
            filters = self._memory.build_memory_filters(state=session_state)
            scoped_result = self._search_with_optional_lock(
                query=query,
                filters=filters,
                strict_matter_lock=analysis.strict_matter_lock,
            )
            if scoped_result.chunks or analysis.active_order_override:
                return _RetrievalExecution(
                    route_mode="memory_scoped_rag",
                    search_result=scoped_result,
                    extracted_filters={
                        "document_version_ids": list(filters.document_version_ids),
                        "memory_scoped_debug": {
                            "scoped_first": True,
                            "global_fallback_attempted": False,
                            "active_order_override": analysis.active_order_override,
                        },
                        **_build_search_debug_payload(scoped_result),
                        **_build_route_debug_payload(analysis),
                    },
                )
        search_result = self._search_with_optional_lock(
            query=query,
            strict_matter_lock=analysis.strict_matter_lock,
        )
        return _RetrievalExecution(
            route_mode="memory_scoped_rag",
            search_result=search_result,
            extracted_filters={
                "memory_scoped_debug": {
                    "scoped_first": bool(session_state is not None),
                    "global_fallback_attempted": bool(session_state is not None),
                    "active_order_override": analysis.active_order_override,
                },
                **_build_search_debug_payload(search_result),
                **_build_route_debug_payload(analysis),
            },
        )

    def _select_context_chunks(
        self,
        search_result: HierarchicalSearchResult,
        *,
        chunk_hits: tuple[ChunkSearchHit, ...] | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> tuple[PromptContextChunk, ...]:
        ranked_hits = list(chunk_hits or search_result.chunks)
        if search_result.query_intent.intent == QueryIntent.SUBSTANTIVE_OUTCOME_QUERY:
            ranked_hits.sort(
                key=lambda hit: (
                    0 if hit.section_type in {"operative_order", "directions", "findings"} else 1,
                    -hit.score.final_score,
                    hit.chunk_id,
                )
            )
        elif analysis is not None and analysis.asks_brief_summary:
            summary_section_priority = {
                "findings": 4,
                "facts": 3,
                "background": 2,
                "operative_order": 1,
                "directions": 1,
            }
            ranked_hits.sort(
                key=lambda hit: (
                    -summary_section_priority.get(hit.section_type, 0),
                    -hit.score.final_score,
                    hit.chunk_id,
                )
            )

        context_chunks: list[PromptContextChunk] = []
        total_tokens = 0
        seen_chunk_ids: set[int] = set()
        seen_section_types: set[str] = set()
        prefer_summary_diversity = bool(analysis is not None and analysis.asks_brief_summary)
        for hit in ranked_hits:
            if hit.chunk_id in seen_chunk_ids:
                continue
            if (
                prefer_summary_diversity
                and context_chunks
                and hit.section_type in seen_section_types
                and any(
                    candidate.chunk_id not in seen_chunk_ids
                    and candidate.section_type not in seen_section_types
                    and candidate.section_type in {"facts", "background", "findings", "operative_order", "directions"}
                    for candidate in ranked_hits
                )
            ):
                continue
            token_count = hit.token_count or _approx_token_count(hit.chunk_text)
            if context_chunks and total_tokens + token_count > self._settings.max_context_tokens:
                continue
            context_chunks.append(
                PromptContextChunk(
                    citation_number=len(context_chunks) + 1,
                    chunk_id=hit.chunk_id,
                    document_version_id=hit.document_version_id,
                    document_id=hit.document_id,
                    record_key=hit.record_key,
                    bucket_name=hit.bucket_name,
                    title=hit.title,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    section_type=hit.section_type,
                    section_title=hit.section_title,
                    detail_url=hit.detail_url,
                    pdf_url=hit.pdf_url,
                    chunk_text=hit.chunk_text,
                    token_count=token_count,
                    score=hit.score.final_score,
                )
            )
            seen_chunk_ids.add(hit.chunk_id)
            seen_section_types.add(hit.section_type)
            total_tokens += token_count
            if len(context_chunks) >= self._settings.max_context_chunks:
                break
        return tuple(context_chunks)

    def _search_with_optional_lock(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None = None,
        top_k_docs: int | None = None,
        top_k_sections: int | None = None,
        top_k_chunks: int | None = None,
        strict_matter_lock=None,
    ) -> HierarchicalSearchResult:
        try:
            return self._search.search(
                query=query,
                filters=filters,
                top_k_docs=top_k_docs,
                top_k_sections=top_k_sections,
                top_k_chunks=top_k_chunks,
                strict_matter_lock=strict_matter_lock,
            )
        except TypeError as exc:
            if "strict_matter_lock" not in str(exc):
                raise
            return self._search.search(
                query=query,
                filters=filters,
                top_k_docs=top_k_docs,
                top_k_sections=top_k_sections,
                top_k_chunks=top_k_chunks,
            )

    def _build_grounded_payload(
        self,
        *,
        session_id: UUID,
        query_intent: str,
        initial_route_mode: str,
        answer_text: str,
        answer_status: str,
        declared_cited_numbers: tuple[int, ...],
        context_chunks: tuple[PromptContextChunk, ...],
        retrieved_chunks: tuple[ChunkSearchHit, ...],
        retrieved_chunk_ids: tuple[int, ...],
        analysis: QueryAnalysis,
        debug_payload: dict[str, Any],
    ) -> ChatAnswerPayload:
        all_citations = build_citations(context_chunks)
        marker_numbers = extract_citation_numbers(answer_text)
        cited_numbers = marker_numbers or declared_cited_numbers
        cleaned_answer_text = strip_inline_citation_markers(answer_text)
        valid_numbers = tuple(
            number for number in cited_numbers if 1 <= number <= len(context_chunks)
        )

        if (
            answer_status != "answered"
            or not cleaned_answer_text
            or not valid_numbers
            or len(valid_numbers) != len(cited_numbers)
        ):
            invalid_debug = dict(debug_payload)
            invalid_debug["web_fallback_debug"] = _build_web_fallback_debug(
                structured_attempted=False,
                corpus_attempted=True,
                official_web_attempted=False,
                general_web_attempted=False,
                final_route_mode="abstain",
                citations=(),
                web_fallback_allowed=False,
                web_fallback_not_allowed_reason=(
                    "named_matter_no_web_override"
                    if analysis.strict_single_matter
                    else "corpus_route_no_web_fallback"
                ),
            )
            invalid_debug["citation_debug"] = _build_citation_debug_payload(())
            return ChatAnswerPayload(
                session_id=session_id,
                route_mode="abstain",
                query_intent=query_intent,
                answer_text=_EMPTY_RETRIEVAL_ANSWER,
                confidence=0.0,
                citations=(),
                retrieved_chunk_ids=retrieved_chunk_ids,
                active_record_keys=(),
                answer_status="abstained",
                debug=invalid_debug,
            )

        citations = filter_citations(all_citations, cited_numbers=valid_numbers)
        cited_context_chunks = resolve_cited_context_chunks(
            context_chunks,
            cited_numbers=valid_numbers,
        )
        confidence = assess_retrieval_confidence(
            context_chunks=context_chunks,
            cited_context_chunks=cited_context_chunks,
            answer_status=answer_status,
            threshold=self._settings.low_confidence_threshold,
            strict_scope_required=analysis.strict_scope_required,
            strict_single_matter=analysis.strict_single_matter,
            locked_record_keys=analysis.strict_lock_record_keys,
        )
        guardrail_result = evaluate_mixed_record_guardrail(
            strict_lock=analysis.strict_matter_lock,
            retrieved_items=context_chunks,
            cited_items=cited_context_chunks,
        )
        final_debug = dict(debug_payload)
        final_debug["mixed_record_guardrail"] = {
            **dict(debug_payload.get("mixed_record_guardrail", {})),
            **dataclass_asdict(guardrail_result),
        }
        final_debug["current_lookup_debug"] = {
            "used": False,
        }
        final_debug["news_lookup_debug"] = {"used": False}
        final_debug["historical_lookup_debug"] = {"used": False}
        final_debug["metadata_debug"] = {"used": False}
        final_debug["web_fallback_debug"] = _build_web_fallback_debug(
            structured_attempted=False,
            corpus_attempted=True,
            official_web_attempted=False,
            general_web_attempted=False,
            final_route_mode=initial_route_mode,
            citations=citations,
            web_fallback_allowed=False,
            web_fallback_not_allowed_reason=(
                "named_matter_no_web_override"
                if analysis.strict_single_matter
                else "corpus_route_no_web_fallback"
            ),
        )
        final_debug["citation_debug"] = _build_citation_debug_payload(citations)
        if guardrail_result.should_abstain or not guardrail_result.single_matter_rule_respected:
            final_debug["web_fallback_debug"] = _build_web_fallback_debug(
                structured_attempted=False,
                corpus_attempted=True,
                official_web_attempted=False,
                general_web_attempted=False,
                final_route_mode="abstain",
                citations=(),
                web_fallback_allowed=False,
                web_fallback_not_allowed_reason="named_matter_no_web_override",
            )
            return ChatAnswerPayload(
                session_id=session_id,
                route_mode="abstain",
                query_intent=query_intent,
                answer_text=_STRICT_MATTER_WEAK_SUPPORT_ANSWER,
                confidence=min(confidence.confidence, 0.15),
                citations=(),
                retrieved_chunk_ids=retrieved_chunk_ids,
                active_record_keys=(),
                answer_status="abstained",
                debug=final_debug,
            )
        if confidence.should_abstain:
            final_debug["web_fallback_debug"] = _build_web_fallback_debug(
                structured_attempted=False,
                corpus_attempted=True,
                official_web_attempted=False,
                general_web_attempted=False,
                final_route_mode="abstain",
                citations=(),
                web_fallback_allowed=False,
                web_fallback_not_allowed_reason=(
                    "named_matter_no_web_override"
                    if analysis.strict_single_matter
                    else "corpus_route_no_web_fallback"
                ),
            )
            return ChatAnswerPayload(
                session_id=session_id,
                route_mode="abstain",
                query_intent=query_intent,
                answer_text=_EMPTY_RETRIEVAL_ANSWER,
                confidence=confidence.confidence,
                citations=(),
                retrieved_chunk_ids=retrieved_chunk_ids,
                active_record_keys=(),
                answer_status="abstained",
                debug=final_debug,
            )

        style_context_chunks = self._build_style_context_chunks(
            context_chunks=context_chunks,
            cited_context_chunks=cited_context_chunks,
            retrieved_chunks=retrieved_chunks,
            analysis=analysis,
        )
        styled_answer_text, style_debug = apply_grounded_wording_caution(
            answer_text=cleaned_answer_text,
            context_chunks=style_context_chunks,
            analysis=analysis,
        )
        final_debug["style_debug"] = style_debug

        final_answer_text = styled_answer_text
        final_status = "answered"
        if confidence.should_hedge:
            final_answer_text = (
                "The retrieved material provides limited support. "
                + styled_answer_text
            )
            final_status = "cautious"

        return ChatAnswerPayload(
            session_id=session_id,
            route_mode=initial_route_mode,
            query_intent=query_intent,
            answer_text=final_answer_text,
            confidence=confidence.confidence,
            citations=citations,
            retrieved_chunk_ids=retrieved_chunk_ids,
            active_record_keys=tuple(_ordered_unique(citation.record_key for citation in citations)),
            answer_status=final_status,
            debug=final_debug,
        )

    def _build_style_context_chunks(
        self,
        *,
        context_chunks: tuple[PromptContextChunk, ...],
        cited_context_chunks: tuple[PromptContextChunk, ...],
        retrieved_chunks: tuple[ChunkSearchHit, ...],
        analysis: QueryAnalysis,
    ) -> tuple[PromptContextChunk, ...]:
        style_chunks_by_id: dict[int, PromptContextChunk] = {
            chunk.chunk_id: chunk
            for chunk in (*context_chunks, *cited_context_chunks)
        }
        if analysis.asks_brief_summary:
            return tuple(style_chunks_by_id.values())
        if len(style_chunks_by_id) == len(retrieved_chunks):
            return tuple(style_chunks_by_id.values())

        for chunk in retrieved_chunks:
            style_chunks_by_id.setdefault(
                chunk.chunk_id,
                PromptContextChunk(
                    citation_number=0,
                    chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id,
                    document_id=chunk.document_id,
                    record_key=chunk.record_key,
                    bucket_name=chunk.bucket_name,
                    title=chunk.title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_type=chunk.section_type,
                    section_title=chunk.section_title,
                    detail_url=chunk.detail_url,
                    pdf_url=chunk.pdf_url,
                    chunk_text=chunk.chunk_text,
                    token_count=chunk.token_count,
                    score=chunk.score.final_score,
                ),
            )
        return tuple(style_chunks_by_id.values())

    def _build_failure_payload(
        self,
        *,
        query: str,
        session_id: UUID,
        decision: RouteDecision | None,
        exc: Exception,
    ) -> ChatAnswerPayload:
        analysis = (
            decision.analysis
            if decision is not None
            else QueryAnalysis(
                raw_query=query,
                normalized_query=" ".join(query.lower().split()),
                query_family="failure_safe_abstain",
            )
        )
        active_record_keys: tuple[str, ...] = ()
        try:
            active_record_keys = self._existing_active_record_keys(session_id)
        except Exception:
            active_record_keys = ()
        return ChatAnswerPayload(
            session_id=session_id,
            route_mode="abstain",
            query_intent=decision.query_intent if decision is not None else "ambiguous",
            answer_text="I’m temporarily unable to answer that safely right now.",
            confidence=0.0,
            citations=(),
            retrieved_chunk_ids=(),
            active_record_keys=active_record_keys,
            answer_status="abstained",
            debug={
                **_build_route_debug_payload(analysis),
                "metadata_debug": {"used": False},
                "current_lookup_debug": {"used": False},
                "news_lookup_debug": {"used": False},
                "historical_lookup_debug": {"used": False},
                "style_debug": {"used": False},
                "failure_safe": {
                    "used": True,
                    "exception_type": type(exc).__name__,
                },
                "web_fallback_debug": _build_web_fallback_debug(
                    structured_attempted=bool(
                        decision is not None and decision.route_mode == "structured_current_info"
                    ),
                    corpus_attempted=bool(
                        decision is not None
                        and decision.route_mode in {"exact_lookup", "hierarchical_rag", "memory_scoped_rag"}
                    ),
                    official_web_attempted=bool(
                        decision is not None
                        and decision.route_mode in {
                            "current_official_lookup",
                            "current_news_lookup",
                            "historical_official_lookup",
                        }
                    ),
                    general_web_attempted=False,
                    final_route_mode="abstain",
                    citations=(),
                    web_fallback_allowed=False,
                    web_fallback_not_allowed_reason="temporary_failure",
                ),
                "citation_debug": _build_citation_debug_payload(()),
            },
        )

    def _log_answer(
        self,
        *,
        session_id: UUID,
        query: str,
        payload: ChatAnswerPayload,
        extracted_filters: dict[str, Any],
        reranked_chunk_ids: tuple[int, ...],
    ) -> None:
        cited_chunk_ids = tuple(
            citation.chunk_id
            for citation in payload.citations
            if citation.chunk_id is not None
        )
        cited_record_keys = tuple(citation.record_key for citation in payload.citations)
        self._answers.insert_retrieval_log(
            session_id=session_id,
            user_query=query,
            route_mode=payload.route_mode,
            query_intent=payload.query_intent,
            extracted_filters=extracted_filters,
            retrieved_chunk_ids=payload.retrieved_chunk_ids,
            reranked_chunk_ids=reranked_chunk_ids,
            final_citation_chunk_ids=cited_chunk_ids,
            confidence=payload.confidence,
            answer_status=payload.answer_status,
            cited_record_keys=cited_record_keys,
        )
        self._answers.insert_answer_log(
            session_id=session_id,
            user_query=query,
            route_mode=payload.route_mode,
            query_intent=payload.query_intent,
            answer_text=payload.answer_text,
            answer_confidence=payload.confidence,
            cited_chunk_ids=cited_chunk_ids,
            cited_record_keys=cited_record_keys,
            citations=payload.citations,
        )
        LOGGER.info(
            "Phase 4 answer completed route_mode=%s query_intent=%s citations=%s confidence=%.4f",
            payload.route_mode,
            payload.query_intent,
            len(payload.citations),
            payload.confidence,
        )


def _build_search_debug_payload(search_result: HierarchicalSearchResult) -> dict[str, Any]:
    return {
        "search_debug": {
            **dict(search_result.debug),
            "top_documents": [_document_debug_row(hit) for hit in search_result.documents[:3]],
            "top_sections": [_section_debug_row(hit) for hit in search_result.sections[:3]],
            "top_chunks": [_chunk_debug_row(hit) for hit in search_result.chunks[:5]],
        }
    }


def _document_debug_row(hit: Any) -> dict[str, Any]:
    return {
        "document_version_id": hit.document_version_id,
        "record_key": hit.record_key,
        "bucket_name": hit.bucket_name,
        "title": hit.title,
        "base_score": round(hit.score.base_score, 6),
        "bucket_adjustment": round(hit.score.bucket_adjustment, 4),
        "query_alignment_adjustment": round(hit.score.query_alignment_adjustment, 4),
        "strict_lock_adjustment": round(hit.score.strict_lock_adjustment, 4),
        "confusion_penalty": round(hit.score.confusion_penalty, 4),
        "final_score": round(hit.score.final_score, 6),
    }


def _section_debug_row(hit: Any) -> dict[str, Any]:
    return {
        "section_node_id": hit.section_node_id,
        "document_version_id": hit.document_version_id,
        "record_key": hit.record_key,
        "bucket_name": hit.bucket_name,
        "section_type": hit.section_type,
        "section_key": hit.section_key,
        "base_score": round(hit.score.base_score, 6),
        "bucket_adjustment": round(hit.score.bucket_adjustment, 4),
        "section_prior": round(hit.score.section_prior, 4),
        "query_alignment_adjustment": round(hit.score.query_alignment_adjustment, 4),
        "content_adjustment": round(hit.score.content_adjustment, 4),
        "strict_lock_adjustment": round(hit.score.strict_lock_adjustment, 4),
        "confusion_penalty": round(hit.score.confusion_penalty, 4),
        "final_score": round(hit.score.final_score, 6),
    }


def _chunk_debug_row(hit: Any) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "document_version_id": hit.document_version_id,
        "record_key": hit.record_key,
        "bucket_name": hit.bucket_name,
        "section_type": hit.section_type,
        "section_key": hit.section_key,
        "base_score": round(hit.score.base_score, 6),
        "bucket_adjustment": round(hit.score.bucket_adjustment, 4),
        "section_prior": round(hit.score.section_prior, 4),
        "query_alignment_adjustment": round(hit.score.query_alignment_adjustment, 4),
        "content_adjustment": round(hit.score.content_adjustment, 4),
        "strict_lock_adjustment": round(hit.score.strict_lock_adjustment, 4),
        "confusion_penalty": round(hit.score.confusion_penalty, 4),
        "diversity_adjustment": round(hit.score.diversity_adjustment, 6),
        "final_score": round(hit.score.final_score, 6),
        "detail_url": hit.detail_url,
        "pdf_url": hit.pdf_url,
    }


def _build_route_debug_payload(analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        "route_debug": {
            "query_family": analysis.query_family,
            "normalized_expansions": list(analysis.normalized_expansions),
            "matched_abbreviations": list(analysis.matched_abbreviations),
            "smalltalk_signals": list(analysis.smalltalk_signals),
            "mentions_sebi": analysis.mentions_sebi,
            "structured_current_info_signals": list(analysis.structured_current_info_signals),
            "current_official_lookup_signals": list(analysis.current_official_lookup_signals),
            "current_news_signals": list(analysis.current_news_signals),
            "historical_official_signals": list(analysis.historical_official_signals),
            "current_public_fact_signals": list(analysis.current_public_fact_signals),
            "company_role_signals": list(analysis.company_role_signals),
            "order_context_override_signals": list(analysis.order_context_override_signals),
            "brief_summary_signals": list(analysis.brief_summary_signals),
            "current_info_query_family": analysis.current_info_query_family,
            "current_info_follow_up": analysis.current_info_follow_up,
            "general_explanatory_signals": list(analysis.general_explanatory_signals),
            "matter_reference_signals": list(analysis.matter_reference_signals),
            "sat_court_signals": list(analysis.sat_court_signals),
            "corpus_metadata_signals": list(analysis.corpus_metadata_signals),
            "asks_order_signatory": analysis.asks_order_signatory,
            "asks_order_date": analysis.asks_order_date,
            "asks_legal_provisions": analysis.asks_legal_provisions,
            "asks_provision_explanation": analysis.asks_provision_explanation,
            "asks_order_pan": analysis.asks_order_pan,
            "asks_order_amount": analysis.asks_order_amount,
            "asks_order_holding": analysis.asks_order_holding,
            "asks_order_parties": analysis.asks_order_parties,
            "asks_order_observations": analysis.asks_order_observations,
            "asks_order_numeric_fact": analysis.asks_order_numeric_fact,
            "active_matter_follow_up_intent": analysis.active_matter_follow_up_intent,
            "active_order_override": analysis.active_order_override,
            "fresh_query_override": analysis.fresh_query_override,
            "has_active_clarification": analysis.has_active_clarification,
            "appears_smalltalk": analysis.appears_smalltalk,
            "appears_structured_current_info": analysis.appears_structured_current_info,
            "appears_current_official_lookup": analysis.appears_current_official_lookup,
            "appears_current_news_lookup": analysis.appears_current_news_lookup,
            "appears_historical_official_lookup": analysis.appears_historical_official_lookup,
            "appears_corpus_metadata_query": analysis.appears_corpus_metadata_query,
            "appears_sat_court_style": analysis.appears_sat_court_style,
            "appears_non_sebi_person_query": analysis.appears_non_sebi_person_query,
            "appears_company_role_current_fact": analysis.appears_company_role_current_fact,
            "appears_general_explanatory": analysis.appears_general_explanatory,
            "appears_matter_specific": analysis.appears_matter_specific,
            "asks_brief_summary": analysis.asks_brief_summary,
            "requires_live_information": analysis.requires_live_information,
            "comparison_intent": analysis.comparison_intent,
            "comparison_terms": list(analysis.comparison_terms),
            "strict_scope_required": analysis.strict_scope_required,
            "strict_single_matter": analysis.strict_single_matter,
            "strict_lock_record_keys": list(analysis.strict_lock_record_keys),
            "strict_lock_titles": list(analysis.strict_lock_titles),
            "strict_lock_matched_aliases": list(analysis.strict_lock_matched_aliases),
            "strict_lock_matched_entities": list(analysis.strict_lock_matched_entities),
            "strict_lock_reason_codes": list(analysis.strict_lock_reason_codes),
            "strict_lock_ambiguous": analysis.strict_lock_ambiguous,
            "strict_matter_lock": dataclass_asdict(analysis.strict_matter_lock),
        }
    }


def _build_planner_debug_payload(decision: RouteDecision) -> dict[str, Any]:
    plan = decision.plan
    if plan is None:
        return {"used": False}
    return {
        "used": True,
        "route": plan.route,
        "reason": plan.reason,
        "confidence": plan.confidence,
        "use_structured_db": plan.use_structured_db,
        "use_order_metadata": plan.use_order_metadata,
        "use_order_rag": plan.use_order_rag,
        "use_official_web": plan.use_official_web,
        "use_general_web": plan.use_general_web,
        "force_fresh_named_matter_override": plan.force_fresh_named_matter_override,
        "execution_route_mode": decision.route_mode,
    }


def _extract_current_lookup_clarification_candidates(
    *,
    query: str,
    result: Any,
) -> tuple[ClarificationCandidate, ...]:
    if str(getattr(result, "lookup_type", "") or "") != "person_lookup":
        return ()
    debug = dict(getattr(result, "debug", {}) or {})
    if str(debug.get("fallback_reason") or "") != "person_match_clarify":
        return ()
    matched_people = debug.get("matched_people")
    if not isinstance(matched_people, list):
        return ()
    return build_person_clarification_candidates(
        matched_people,
        source_query=query,
        extracted_person_name=str(debug.get("extracted_person_name") or "") or None,
    )


def _active_scope_supports_follow_up_intent(*, state: Any, follow_up_intent: str) -> bool:
    title = " ".join(str(getattr(state, "active_primary_title", "") or "").lower().split())
    bucket_names = " ".join(
        " ".join(str(value or "").lower().split())
        for value in getattr(state, "active_bucket_names", ()) or ()
    )
    haystack = " ".join(part for part in (title, bucket_names) if part)
    if not haystack:
        return True
    if follow_up_intent == "settlement_amount":
        return "settlement" in haystack
    if follow_up_intent == "exemption_granted":
        return "exemption" in haystack
    if follow_up_intent in {"appellate_authority_decision", "sat_hold"}:
        return any(
            token in haystack
            for token in ("appeal", "sat", "tribunal", "court", "judgment", "writ petition", "vs sebi")
        )
    return True


def _render_missing_active_matter_concept_answer(*, state: Any, follow_up_intent: str) -> str:
    title = str(getattr(state, "active_primary_title", "") or "the active matter").strip()
    if follow_up_intent == "settlement_amount":
        return f"{title} does not appear to be a settlement matter, so there is no settlement amount exposed for the active matter."
    if follow_up_intent == "exemption_granted":
        return f"{title} does not appear to be an exemption matter, so the active matter does not expose an exemption decision."
    if follow_up_intent in {"appellate_authority_decision", "sat_hold"}:
        return f"{title} does not appear to be an appellate, SAT, or court matter, so the active matter does not expose an appellate decision."
    return f"{title} does not expose that concept in the active matter scope."


def _with_clarification_debug(
    payload: ChatAnswerPayload,
    *,
    active_context: bool,
    selection_query: str,
    matched_candidates: tuple[ClarificationCandidate, ...],
    resolved_candidate: ClarificationCandidate | None,
    match_reason: str | None,
) -> ChatAnswerPayload:
    debug = dict(payload.debug)
    debug["clarification_debug"] = {
        **dict(debug.get("clarification_debug", {})),
        "used": True,
        "active_context": active_context,
        "selection_query": selection_query,
        "match_reason": match_reason,
        "matched_candidate_ids": [candidate.candidate_id for candidate in matched_candidates],
        "resolved_candidate_id": (
            resolved_candidate.candidate_id if resolved_candidate is not None else None
        ),
    }
    return ChatAnswerPayload(
        session_id=payload.session_id,
        route_mode=payload.route_mode,
        query_intent=payload.query_intent,
        answer_text=payload.answer_text,
        confidence=payload.confidence,
        citations=payload.citations,
        retrieved_chunk_ids=payload.retrieved_chunk_ids,
        active_record_keys=payload.active_record_keys,
        answer_status=payload.answer_status,
        clarification_candidates=payload.clarification_candidates,
        debug=debug,
    )


def _single_candidate_auto_resolve_allowed(candidate: Any) -> bool:
    score = float(getattr(candidate, "score", 0.0) or 0.0)
    exact_title_match = bool(getattr(candidate, "exact_title_match", False))
    matched_aliases = tuple(getattr(candidate, "matched_aliases", ()) or ())
    matched_entities = tuple(getattr(candidate, "matched_entity_terms", ()) or ())
    return score >= 0.46 and (
        exact_title_match
        or bool(matched_aliases)
        or bool(matched_entities)
        or score >= 0.58
    )


def _static_general_knowledge_answer(*, query: str, analysis: QueryAnalysis) -> str | None:
    normalized_query = " ".join(query.lower().split())
    if "sebi_definition" not in analysis.general_explanatory_signals:
        if "exemption order under the takeover regulations" in normalized_query:
            return (
                "An exemption order under the takeover regulations is a SEBI order that relieves a proposed "
                "acquirer from specified open-offer or takeover-compliance obligations in a particular transaction, "
                "usually subject to stated conditions."
            )
        if "regulation 30a" in normalized_query:
            return (
                "Orders under Regulation 30A are SEBI orders dealing with suspension, cancellation, or surrender "
                "of an intermediary's certificate of registration under the SEBI (Intermediaries) Regulations, 2008, "
                "and they usually set out the consequences and continuing obligations that follow."
            )
        return None
    return (
        "SEBI is the Securities and Exchange Board of India, the statutory regulator for the "
        "Indian securities market."
    )


def _parse_json_object(raw_content: str) -> dict[str, Any]:
    stripped = raw_content.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}


def _normalize_answer_status(raw_status: Any) -> str:
    normalized = str(raw_status or "").strip().lower()
    if normalized == "answered":
        return "answered"
    if normalized == "clarify":
        return "clarify"
    return "insufficient_context"


def _approx_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def _normalize_cited_numbers(raw_value: Any) -> tuple[int, ...]:
    if not isinstance(raw_value, list):
        return ()
    numbers: list[int] = []
    for value in raw_value:
        try:
            numbers.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(numbers)


def _ordered_unique(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _ordered_unique_values(values: Any) -> tuple[Any, ...]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _infer_main_entity(title: str | None) -> str | None:
    if not title:
        return None
    lowered = title.lower()
    for marker in ("in the matter of ", "in respect of "):
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            excerpt = title[start:]
            excerpt = re.split(r"\bin the matter of\b|\bvs\.?\b|\bversus\b", excerpt, maxsplit=1, flags=re.IGNORECASE)[0]
            cleaned = " ".join(excerpt.split()).strip(" ,.-")
            return cleaned or None
    return None

def _build_citation_debug_payload(citations: tuple[Any, ...]) -> dict[str, Any]:
    cited_record_keys = list(_ordered_unique(citation.record_key for citation in citations))
    source_urls = [
        citation.source_url or citation.detail_url or citation.pdf_url
        for citation in citations
        if citation.source_url or citation.detail_url or citation.pdf_url
    ]
    return {
        "cited_record_keys": cited_record_keys,
        "citation_scope": (
            "none"
            if not citations
            else "single_matter"
            if len(set(cited_record_keys)) <= 1
            else "mixed"
        ),
        "source_urls": source_urls,
        "citations": [
            {
                "citation_number": citation.citation_number,
                "record_key": citation.record_key,
                "detail_url": citation.detail_url,
                "pdf_url": citation.pdf_url,
                "source_url": citation.source_url,
                "domain": citation.domain,
                "source_type": citation.source_type,
            }
            for citation in citations
        ],
    }


def _resolve_current_lookup_route_mode(
    *,
    requested_route_mode: str,
    result: Any,
    citations: tuple[Any, ...],
) -> str:
    if result.answer_status != "answered":
        return "abstain"
    source_types = {
        str(citation.source_type or "").strip()
        for citation in citations
        if str(citation.source_type or "").strip()
    }
    if not source_types:
        return requested_route_mode
    if source_types.issubset({"structured"}):
        return "structured_current_info"
    return "current_official_lookup"


def _build_web_fallback_debug(
    *,
    structured_attempted: bool,
    corpus_attempted: bool,
    official_web_attempted: bool,
    general_web_attempted: bool,
    final_route_mode: str,
    citations: tuple[Any, ...],
    web_fallback_allowed: bool,
    web_fallback_not_allowed_reason: str | None,
) -> dict[str, Any]:
    source_domains = list(
        _ordered_unique(
            citation.domain
            for citation in citations
            if getattr(citation, "domain", None)
        )
    )
    source_types = list(
        _ordered_unique(
            citation.source_type
            for citation in citations
            if getattr(citation, "source_type", None)
        )
    )
    attempt_order: list[str] = []
    if structured_attempted:
        attempt_order.append("structured_current_info")
    if corpus_attempted:
        attempt_order.append("corpus_rag")
    if official_web_attempted:
        attempt_order.append("official_web_fallback")
    if general_web_attempted:
        attempt_order.append("general_web_fallback")
    if final_route_mode == "abstain":
        attempt_order.append("abstain")
    return {
        "structured_attempted": structured_attempted,
        "corpus_attempted": corpus_attempted,
        "official_web_attempted": official_web_attempted,
        "general_web_attempted": general_web_attempted,
        "attempt_order": attempt_order,
        "final_route_mode": final_route_mode,
        "source_domains": source_domains,
        "source_types": source_types,
        "web_fallback_allowed": web_fallback_allowed,
        "web_fallback_not_allowed_reason": web_fallback_not_allowed_reason,
        "selected_source_tier": _resolve_selected_source_tier(
            final_route_mode=final_route_mode,
            source_types=tuple(source_types),
        ),
    }


def _general_web_fallback_allowed(
    *,
    analysis: QueryAnalysis,
    query: str,
) -> tuple[bool, str | None]:
    normalized_query = " ".join(query.lower().split())
    if analysis.mentions_sebi:
        return False, "sebi_official_material_prefers_official_sources"
    if analysis.appears_matter_specific or analysis.strict_scope_required:
        return False, "named_matter_no_web_override"
    if analysis.appears_company_role_current_fact:
        return True, None
    if analysis.appears_non_sebi_person_query:
        return True, None
    if normalized_query.startswith(("who is", "who was", "tell me about", "what is")):
        return True, None
    if analysis.requires_live_information or any(
        token in normalized_query for token in ("latest", "recent", "current", "today", "now")
    ):
        return True, None
    return False, "direct_llm_first_for_timeless_general_knowledge"


def _general_query_requires_live_web_check(
    *,
    analysis: QueryAnalysis,
    query: str,
) -> bool:
    normalized_query = " ".join(query.lower().split())
    if analysis.mentions_sebi or analysis.appears_matter_specific:
        return False
    if analysis.appears_company_role_current_fact:
        return True
    if analysis.requires_live_information:
        return True
    return any(
        token in normalized_query
        for token in ("latest", "recent", "current", "today", "now", "as of")
    )


_COMPANY_ROLE_SOURCE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "co",
        "company",
        "corp",
        "corporation",
        "inc",
        "limited",
        "ltd",
        "of",
        "pvt",
        "private",
        "the",
    }
)
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")


def _filter_general_web_sources(
    *,
    query: str,
    answer_text: str,
    analysis: QueryAnalysis,
    sources: tuple[WebSearchSource, ...],
) -> tuple[WebSearchSource, ...]:
    if not analysis.appears_company_role_current_fact or not sources:
        return sources
    company_role_query = parse_company_role_query(query)
    if company_role_query is None:
        return sources
    company_tokens = _source_relevance_tokens(
        company_role_query.company_name,
        drop_corporate_suffixes=True,
    )
    person_tokens = _company_role_answer_person_tokens(answer_text)
    filtered = tuple(
        source
        for source in sources
        if _company_role_source_matches(
            source=source,
            company_tokens=company_tokens,
            person_tokens=person_tokens,
        )
    )
    return filtered or sources


def _company_role_answer_person_tokens(answer_text: str) -> tuple[str, ...]:
    answer = (answer_text or "").strip()
    if not answer:
        return ()
    candidates = _PROPER_NAME_RE.findall(answer)
    if not candidates:
        return ()
    return _source_relevance_tokens(candidates[0], drop_corporate_suffixes=False)


def _company_role_source_matches(
    *,
    source: WebSearchSource,
    company_tokens: tuple[str, ...],
    person_tokens: tuple[str, ...],
) -> bool:
    haystack = " ".join(
        part
        for part in (
            source.source_title,
            source.source_url,
            source.snippet or "",
            source.domain,
        )
        if part
    ).lower()
    company_hits = sum(1 for token in company_tokens if token in haystack)
    person_hits = sum(1 for token in person_tokens if token in haystack)
    min_company_hits = 2 if len(company_tokens) >= 2 else 1 if company_tokens else 0
    min_person_hits = 2 if len(person_tokens) >= 2 else 1 if person_tokens else 0
    return bool(
        (min_company_hits and company_hits >= min_company_hits)
        or (min_person_hits and person_hits >= min_person_hits)
    )


def _source_relevance_tokens(value: str, *, drop_corporate_suffixes: bool) -> tuple[str, ...]:
    stopwords = _COMPANY_ROLE_SOURCE_STOPWORDS if drop_corporate_suffixes else frozenset({"a", "an", "and", "of", "the"})
    return tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if len(token) >= 3 and token not in stopwords
        )
    )


def _resolve_selected_source_tier(
    *,
    final_route_mode: str,
    source_types: tuple[str, ...],
) -> str:
    if not source_types:
        return {
            "structured_current_info": "structured_current_info",
            "current_official_lookup": "official_web_fallback",
            "current_news_lookup": "official_web_fallback",
            "historical_official_lookup": "official_web_fallback",
            "general_knowledge": "general_knowledge",
            "exact_lookup": "corpus_rag",
            "hierarchical_rag": "corpus_rag",
            "memory_scoped_rag": "corpus_rag",
            "abstain": "abstain",
        }.get(final_route_mode, final_route_mode)
    if set(source_types).issubset({"structured"}):
        return "structured_current_info"
    if "official_web" in source_types:
        return "official_web_fallback"
    if "general_web" in source_types:
        return "general_web_fallback"
    return "corpus_rag"
