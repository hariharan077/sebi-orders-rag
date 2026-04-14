"""Deterministic query intent labels for the Phase 4 router."""

from __future__ import annotations

from ..schemas import QueryAnalysis


def classify_query_intent(analysis: QueryAnalysis) -> str:
    """Return a stable high-level intent label for routing and audit logs."""

    if analysis.appears_smalltalk:
        return "smalltalk"
    if analysis.appears_current_news_lookup:
        return "current_news_lookup"
    if analysis.appears_historical_official_lookup:
        return "historical_official_lookup"
    if analysis.appears_corpus_metadata_query:
        return "corpus_metadata"
    if analysis.appears_structured_current_info:
        return "structured_current_info"
    if analysis.active_order_override and analysis.has_session_scope:
        return "follow_up"
    if analysis.appears_company_role_current_fact:
        return "general_knowledge"
    if analysis.appears_current_official_lookup:
        return "current_official_lookup"
    if analysis.likely_follow_up and analysis.has_session_scope:
        return "follow_up"
    if analysis.title_or_party_lookup_signals:
        return "document_lookup"
    if analysis.asks_provision_explanation:
        return "legal_explanation"
    if analysis.appears_general_explanatory:
        return "general_knowledge"
    if analysis.procedural_or_outcome_signals or analysis.appears_settlement_specific:
        return "substantive_outcome"
    if analysis.appears_matter_specific:
        return "matter_specific"
    return "ambiguous"
