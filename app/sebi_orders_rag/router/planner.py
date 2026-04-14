"""Canonical query planner for SEBI Orders routing."""

from __future__ import annotations

import re

from ..schemas import QueryAnalysis, QueryPlan

_ORDER_CASE_RE = re.compile(
    r"\b(?:order|matter|case|appeal|sat|judgment|tribunal|writ|petition|vs\.?|versus|v\.)\b",
    re.IGNORECASE,
)
_CORPORATE_TOKEN_RE = re.compile(
    r"\b(?:limited|ltd|bank|digital|industries|securities|finance|research|capital|energy)\b",
    re.IGNORECASE,
)
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_STOPWORD_RE = re.compile(
    r"\b(?:who|what|when|was|is|the|this|that|for|in|of|did|give|show|tell|me|each|period|patch|wise|price|movement|signed|signatory|passed|issued|order|case|matter)\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_VARIANT_STOPWORDS = {
    "who",
    "which",
    "what",
    "when",
    "was",
    "is",
    "a",
    "an",
    "the",
    "this",
    "that",
    "for",
    "in",
    "of",
    "did",
    "give",
    "show",
    "tell",
    "me",
    "each",
    "period",
    "patch",
    "wise",
    "price",
    "movement",
    "signed",
    "signatory",
    "passed",
    "issued",
    "settlement",
    "penalty",
    "imposed",
    "amount",
    "finally",
    "direct",
    "directed",
    "proceedings",
    "granted",
    "observed",
    "observation",
    "observations",
    "order",
    "case",
    "matter",
}
_OF_SUBJECT_RE = re.compile(
    r"\bof\s+(?P<subject>[a-z0-9][a-z0-9 .&'/-]{2,160}?)(?:\s+(?:for each period|each period|each patch|period-wise|patch-wise))?\s*$",
    re.IGNORECASE,
)
_SIGNED_SUBJECT_RE = re.compile(
    r"\b(?:who|which)\b.*?\b(?:signed|signatory)\b(?:\s+the)?\s+(?P<subject>.+)$",
    re.IGNORECASE,
)
_DATED_SUBJECT_RE = re.compile(
    r"\bwhen\s+was\s+(?:the\s+)?(?P<subject>.+?)\s+(?:passed|issued)\b",
    re.IGNORECASE,
)
_MATTER_OF_RE = re.compile(r"\bmatter of\s+(?P<subject>.+)$", re.IGNORECASE)
_PRICE_INCREASE_SUBJECT_RE = re.compile(
    r"\bhow much did\s+(?P<subject>[a-z0-9][a-z0-9 .&'/-]{2,160}?)\s+share price increase\b",
    re.IGNORECASE,
)
_BEFORE_AFTER_SUBJECT_RE = re.compile(
    r"\bwhat was the price before and after(?: the increase)? in\s+(?P<subject>[a-z0-9][a-z0-9 .&'/-]{2,160})\b",
    re.IGNORECASE,
)
_HIGHEST_PRICE_SUBJECT_RE = re.compile(
    r"\bwhat was the (?:highest|peak|lowest|listing) price of\s+(?P<subject>[a-z0-9][a-z0-9 .&'/-]{2,160})\b",
    re.IGNORECASE,
)


class QueryPlanner:
    """Choose one canonical route before execution-specific routing."""

    def plan(
        self,
        *,
        query: str,
        analysis: QueryAnalysis,
    ) -> QueryPlan:
        if not analysis.normalized_query:
            return QueryPlan(route="abstain", reason="empty_query", confidence=1.0)

        if analysis.appears_smalltalk:
            return QueryPlan(
                route="general_knowledge",
                reason="smalltalk",
                confidence=0.99,
            )

        if analysis.appears_current_news_lookup:
            return QueryPlan(
                route="current_news",
                reason="current_news_lookup",
                confidence=0.99,
                use_official_web=True,
            )

        metadata_requested = _metadata_requested(analysis)
        named_order_reference = looks_like_named_order_reference(query=query, analysis=analysis)
        lookup_variants = build_order_lookup_variants(query=query, analysis=analysis)

        if analysis.appears_corpus_metadata_query:
            return QueryPlan(
                route="order_metadata",
                reason="corpus_metadata_first",
                confidence=0.98,
                use_order_metadata=True,
            )

        if analysis.active_order_override and metadata_requested:
            return QueryPlan(
                route="order_metadata",
                reason="active_matter_metadata_follow_up",
                confidence=0.99,
                use_order_metadata=True,
            )

        if analysis.appears_structured_current_info:
            return QueryPlan(
                route="structured_current_info",
                reason="structured_current_info",
                confidence=0.99,
                use_structured_db=True,
            )

        if analysis.appears_historical_official_lookup or analysis.appears_current_official_lookup:
            return QueryPlan(
                route="official_web",
                reason=(
                    "historical_official_lookup"
                    if analysis.appears_historical_official_lookup
                    else "official_public_fact"
                ),
                confidence=0.98,
                use_official_web=True,
            )

        if analysis.appears_company_role_current_fact:
            return QueryPlan(
                route="official_web",
                reason="company_role_current_fact",
                confidence=0.98,
                use_official_web=True,
                use_general_web=True,
            )

        if metadata_requested:
            if analysis.appears_general_explanatory and not analysis.appears_matter_specific:
                return QueryPlan(
                    route="general_knowledge",
                    reason="general_knowledge",
                    confidence=0.94,
                )
            if named_order_reference or lookup_variants or analysis.has_session_scope:
                return QueryPlan(
                    route="order_metadata",
                    reason="exact_order_fact_metadata_first",
                    confidence=0.97,
                    use_order_metadata=True,
                    force_fresh_named_matter_override=analysis.fresh_query_override,
                )
            return QueryPlan(
                route="clarify",
                reason="missing_order_scope_for_metadata",
                confidence=0.86,
            )

        if analysis.active_order_override:
            return QueryPlan(
                route="order_corpus_rag",
                reason="active_matter_corpus_follow_up",
                confidence=0.98,
                use_order_rag=True,
            )

        if analysis.appears_general_explanatory and not analysis.appears_matter_specific:
            return QueryPlan(
                route="general_knowledge",
                reason="general_knowledge",
                confidence=0.94,
            )

        if analysis.appears_matter_specific or named_order_reference:
            return QueryPlan(
                route="order_corpus_rag",
                reason="named_matter_query",
                confidence=0.96,
                use_order_rag=True,
                force_fresh_named_matter_override=analysis.fresh_query_override,
            )

        if analysis.appears_general_explanatory or analysis.appears_non_sebi_person_query:
            return QueryPlan(
                route="general_knowledge",
                reason="general_knowledge",
                confidence=0.94,
            )

        if analysis.requires_live_information:
            return QueryPlan(
                route="official_web",
                reason="live_public_fact",
                confidence=0.9,
                use_official_web=True,
                use_general_web=not analysis.mentions_sebi,
            )

        if analysis.query_family == "ambiguous":
            return QueryPlan(
                route="clarify",
                reason="ambiguous_query",
                confidence=0.72,
            )

        return QueryPlan(
            route="general_knowledge",
            reason="general_knowledge_default",
            confidence=0.7,
        )


def looks_like_named_order_reference(*, query: str, analysis: QueryAnalysis) -> bool:
    """Detect matter references that should stay on the internal order path."""

    if (
        analysis.appears_matter_specific
        or analysis.strict_scope_required
        or analysis.title_or_party_lookup_signals
        or analysis.matter_reference_signals
    ):
        return True
    normalized = _SPACE_RE.sub(" ", query.strip())
    if not normalized:
        return False
    if not _ORDER_CASE_RE.search(normalized):
        return False
    if _PROPER_NAME_RE.search(query):
        return True
    if _CORPORATE_TOKEN_RE.search(normalized):
        return True
    return bool(build_order_lookup_variants(query=query, analysis=analysis))


def build_order_lookup_variants(*, query: str, analysis: QueryAnalysis) -> tuple[str, ...]:
    """Extract matter-focused lookup variants from exact-fact and order queries."""

    raw = _SPACE_RE.sub(" ", query.strip())
    if not raw:
        return ()

    variants: list[str] = []
    for pattern in (
        _OF_SUBJECT_RE,
        _SIGNED_SUBJECT_RE,
        _DATED_SUBJECT_RE,
        _MATTER_OF_RE,
        _PRICE_INCREASE_SUBJECT_RE,
        _BEFORE_AFTER_SUBJECT_RE,
        _HIGHEST_PRICE_SUBJECT_RE,
    ):
        match = pattern.search(raw)
        if match:
            subject = _clean_variant(match.group("subject"))
            if subject:
                variants.append(subject)

    if analysis.asks_order_signatory or analysis.asks_order_date or analysis.asks_order_numeric_fact:
        cleaned = _clean_variant(_STOPWORD_RE.sub(" ", raw))
        if cleaned:
            variants.append(cleaned)

    order_trimmed = _clean_variant(re.sub(r"\b(?:order|case|matter)\b\s*$", "", raw, flags=re.IGNORECASE))
    if order_trimmed and order_trimmed.lower() != raw.lower():
        variants.append(order_trimmed)

    deduped: list[str] = []
    for variant in variants:
        if not _is_meaningful_variant(variant):
            continue
        lowered = variant.lower()
        if lowered not in {item.lower() for item in deduped}:
            deduped.append(variant)
    return tuple(deduped)


def _clean_variant(value: str) -> str:
    cleaned = _SPACE_RE.sub(" ", value.strip(" .?,-"))
    cleaned = re.sub(r"^(?:the|this)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+\b(?:for each period|each period|each patch|period-wise|patch-wise)\b$", "", cleaned, flags=re.IGNORECASE)
    return _SPACE_RE.sub(" ", cleaned).strip(" .?,-")


def _is_meaningful_variant(value: str) -> bool:
    tokens = [token.lower() for token in _TOKEN_RE.findall(value)]
    if len(tokens) < 2:
        return False
    non_stop_tokens = [token for token in tokens if token not in _VARIANT_STOPWORDS]
    return len(non_stop_tokens) >= 2


def _metadata_requested(analysis: QueryAnalysis) -> bool:
    return bool(
        analysis.asks_order_signatory
        or analysis.asks_order_date
        or analysis.asks_legal_provisions
        or analysis.asks_provision_explanation
        or analysis.asks_order_pan
        or analysis.asks_order_amount
        or analysis.asks_order_holding
        or analysis.asks_order_parties
        or (analysis.asks_order_observations and analysis.active_order_override)
        or analysis.asks_order_numeric_fact
        or analysis.active_matter_follow_up_intent is not None
    )
