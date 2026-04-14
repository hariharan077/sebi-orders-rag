"""Deterministic query-intent heuristics for Phase 3.1 retrieval tuning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_WHITESPACE_RE = re.compile(r"\s+")
_PARTY_OR_TITLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "appeal_no",
        re.compile(
            r"\b(?:appeal|petition|writ|case|suit|application|complaint|order)\s+no\.?\b",
            re.IGNORECASE,
        ),
    ),
    ("filed_by", re.compile(r"\bfiled by\b", re.IGNORECASE)),
    ("versus", re.compile(r"\b(?:vs\.?|versus|v\.)\b", re.IGNORECASE)),
    ("matter_style", re.compile(r"\bin the matter of\b", re.IGNORECASE)),
    ("petitioner", re.compile(r"\bpetitioner\b", re.IGNORECASE)),
    ("respondent", re.compile(r"\brespondent\b", re.IGNORECASE)),
)
_GENERIC_EXPLANATORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("what_is", re.compile(r"^\s*what is\b", re.IGNORECASE)),
    ("explain", re.compile(r"^\s*explain\b", re.IGNORECASE)),
    ("meaning_of", re.compile(r"\bmeaning of\b", re.IGNORECASE)),
)
_SUBSTANTIVE_OUTCOME_TERMS: tuple[str, ...] = (
    "dismissed",
    "allowed",
    "upheld",
    "quashed",
    "penalty",
    "penalties",
    "imposed",
    "restrained",
    "debarred",
    "directed",
    "direct",
    "refund",
    "settle",
    "settled",
    "settlement",
    "violated",
    "findings",
    "held",
    "order passed",
    "final order",
    "interim order",
)
_SETTLEMENT_TERMS: tuple[str, ...] = (
    "settle",
    "settlement",
    "settled",
    "settlement order",
    "terms of settlement",
    "settlement application",
    "settlement applications",
    "matter settled",
    "settlement proceedings",
    "settlement amount",
)
_SUBSTANTIVE_CONTEXT_TERMS: tuple[str, ...] = ("appeal",)
_REGULATION_OR_TOPIC_TERMS: tuple[str, ...] = (
    "act",
    "regulation",
    "regulations",
    "rule",
    "rules",
    "section",
    "rti act",
    "sast",
    "pfutp",
    "icdr",
    "lodr",
    "insider trading",
    "takeover",
    "adjudication",
)


class QueryIntent(str, Enum):
    """Supported deterministic retrieval intents."""

    SUBSTANTIVE_OUTCOME_QUERY = "substantive_outcome_query"
    PARTY_OR_TITLE_LOOKUP = "party_or_title_lookup"
    REGULATION_OR_TOPIC_LOOKUP = "regulation_or_topic_lookup"
    GENERIC_LOOKUP = "generic_lookup"


@dataclass(frozen=True)
class QueryIntentResult:
    """Intent classification and the matched heuristic markers."""

    intent: QueryIntent
    matched_terms: tuple[str, ...] = ()
    settlement_terms: tuple[str, ...] = ()
    generic_explanatory_terms: tuple[str, ...] = ()
    entity_terms: tuple[str, ...] = ()
    settlement_focused: bool = False


def detect_query_intent(query: str) -> QueryIntentResult:
    """Return a deterministic query intent classification for retrieval ranking."""

    normalized = _normalize_query(query)
    if not normalized:
        return QueryIntentResult(intent=QueryIntent.GENERIC_LOOKUP)

    settlement_matches = _matched_terms(normalized, _SETTLEMENT_TERMS)
    generic_explanatory_matches = _matched_pattern_labels(
        query,
        _GENERIC_EXPLANATORY_PATTERNS,
    )
    entity_terms = _extract_entity_terms(normalized)
    if generic_explanatory_matches and settlement_matches and not entity_terms:
        return QueryIntentResult(
            intent=QueryIntent.GENERIC_LOOKUP,
            matched_terms=_merge_terms(generic_explanatory_matches, settlement_matches),
            settlement_terms=settlement_matches,
            generic_explanatory_terms=generic_explanatory_matches,
        )

    substantive_matches = _matched_terms(normalized, _SUBSTANTIVE_OUTCOME_TERMS)
    if substantive_matches:
        return QueryIntentResult(
            intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
            matched_terms=_merge_terms(substantive_matches, settlement_matches),
            settlement_terms=settlement_matches,
            generic_explanatory_terms=generic_explanatory_matches,
            entity_terms=entity_terms,
            settlement_focused=bool(settlement_matches),
        )

    if settlement_matches:
        return QueryIntentResult(
            intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
            matched_terms=settlement_matches,
            settlement_terms=settlement_matches,
            generic_explanatory_terms=generic_explanatory_matches,
            entity_terms=entity_terms,
            settlement_focused=True,
        )

    party_or_title_matches = _matched_pattern_labels(query, _PARTY_OR_TITLE_PATTERNS)
    if party_or_title_matches:
        return QueryIntentResult(
            intent=QueryIntent.PARTY_OR_TITLE_LOOKUP,
            matched_terms=party_or_title_matches,
            generic_explanatory_terms=generic_explanatory_matches,
            entity_terms=entity_terms,
        )

    substantive_context_matches = _matched_terms(normalized, _SUBSTANTIVE_CONTEXT_TERMS)
    if substantive_context_matches:
        return QueryIntentResult(
            intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
            matched_terms=substantive_context_matches,
            generic_explanatory_terms=generic_explanatory_matches,
            entity_terms=entity_terms,
        )

    regulation_matches = _matched_terms(normalized, _REGULATION_OR_TOPIC_TERMS)
    if regulation_matches:
        return QueryIntentResult(
            intent=QueryIntent.REGULATION_OR_TOPIC_LOOKUP,
            matched_terms=regulation_matches,
            generic_explanatory_terms=generic_explanatory_matches,
        )

    return QueryIntentResult(
        intent=QueryIntent.GENERIC_LOOKUP,
        matched_terms=generic_explanatory_matches,
        generic_explanatory_terms=generic_explanatory_matches,
    )


def _normalize_query(query: str) -> str:
    return _WHITESPACE_RE.sub(" ", query.lower()).strip()


def _matched_terms(normalized_query: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    matches = [term for term in terms if _contains_term(normalized_query, term)]
    return tuple(matches)


def _matched_pattern_labels(
    query: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[str, ...]:
    matches = [label for label, pattern in patterns if pattern.search(query)]
    return tuple(matches)


def _contains_term(normalized_query: str, term: str) -> bool:
    pattern = rf"\b{re.escape(term)}\b"
    return re.search(pattern, normalized_query) is not None


def _extract_entity_terms(normalized_query: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", normalized_query)
    filtered = [
        token
        for token in tokens
        if len(token) >= 2 and token not in _ENTITY_STOPWORDS and not token.isdigit()
    ]
    return tuple(dict.fromkeys(filtered))


def _merge_terms(*groups: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


_ENTITY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "amount",
        "application",
        "applications",
        "did",
        "direct",
        "directed",
        "finally",
        "for",
        "in",
        "is",
        "matter",
        "of",
        "order",
        "ordered",
        "proceedings",
        "question",
        "sebi",
        "settled",
        "settlement",
        "terms",
        "the",
        "under",
        "was",
        "what",
        "explain",
        "meaning",
    }
)
