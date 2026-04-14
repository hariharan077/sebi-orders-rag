"""Deterministic routing decisions for the adaptive RAG layer."""

from __future__ import annotations

from ..control import ControlPack
from ..schemas import ChatSessionStateRecord, QueryAnalysis, RouteDecision
from .intent_classifier import classify_query_intent
from .planner import QueryPlanner, build_order_lookup_variants
from .query_analyzer import analyze_query


class AdaptiveQueryRouter:
    """Conservative deterministic router for Phase 4 chat queries."""

    def __init__(self, *, control_pack: ControlPack | None = None) -> None:
        self._control_pack = control_pack
        self._planner = QueryPlanner()

    def decide(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> RouteDecision:
        """Choose an execution route from a canonical planner decision."""

        analysis = analyze_query(
            query,
            session_state=session_state,
            control_pack=self._control_pack,
        )
        query_intent = classify_query_intent(analysis)
        plan = self._planner.plan(query=query, analysis=analysis)
        route_mode, reason_codes = _resolve_execution_route_mode(
            query=query,
            analysis=analysis,
            planner_route=plan.route,
            planner_reason=plan.reason,
        )
        return self._decision(
            analysis,
            query_intent=query_intent,
            route_mode=route_mode,
            reason_codes=reason_codes,
            plan=plan,
        )

    @staticmethod
    def _decision(
        analysis: QueryAnalysis,
        *,
        query_intent: str,
        route_mode: str,
        reason_codes: tuple[str, ...],
        plan=None,
    ) -> RouteDecision:
        return RouteDecision(
            route_mode=route_mode,
            query_intent=query_intent,
            analysis=analysis,
            reason_codes=reason_codes,
            plan=plan,
        )


def _resolve_execution_route_mode(
    *,
    query: str,
    analysis: QueryAnalysis,
    planner_route: str,
    planner_reason: str,
) -> tuple[str, tuple[str, ...]]:
    if planner_route == "abstain":
        return "abstain", (planner_reason,)
    if planner_route == "clarify":
        return "clarify", (planner_reason,)
    if planner_route == "general_knowledge":
        return ("smalltalk", (planner_reason,)) if analysis.appears_smalltalk else (
            "general_knowledge",
            (planner_reason,),
        )
    if planner_route == "current_news":
        return "current_news_lookup", (planner_reason, "non_corpus_route")
    if planner_route == "structured_current_info":
        return "structured_current_info", (planner_reason, "structured_first")
    if planner_route == "official_web":
        if analysis.appears_company_role_current_fact:
            return "general_knowledge", (planner_reason, "non_corpus_route")
        if analysis.appears_historical_official_lookup:
            return "historical_official_lookup", (planner_reason, "non_corpus_route")
        return "current_official_lookup", (planner_reason, "official_sources_only")
    if planner_route == "order_metadata":
        if analysis.appears_corpus_metadata_query:
            return "corpus_metadata", (planner_reason, "metadata_first")
        if analysis.active_order_override and analysis.has_session_scope:
            return "memory_scoped_rag", (planner_reason, "session_scope", "metadata_first")
        return (
            "hierarchical_rag",
            (
                planner_reason,
                "metadata_first",
                *(() if not analysis.appears_settlement_specific else ("settlement_specific",)),
            ),
        )
    if planner_route == "order_corpus_rag":
        if analysis.active_order_override and analysis.has_session_scope:
            return "memory_scoped_rag", (planner_reason, "session_scope")
        if _should_prefer_exact_lookup(query=query, analysis=analysis):
            return "exact_lookup", (planner_reason, "document_identity_signal")
        return "hierarchical_rag", (planner_reason, "prefer_retrieval")
    return "general_knowledge", (planner_reason,)


def _should_prefer_exact_lookup(*, query: str, analysis: QueryAnalysis) -> bool:
    if analysis.strict_scope_required or analysis.title_or_party_lookup_signals:
        return True
    if analysis.appears_sat_court_style:
        return True
    return bool(build_order_lookup_variants(query=query, analysis=analysis))
