"""Helpers for current company leadership and role-style public-fact queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ROLE_RE = re.compile(
    r"\b(?P<role>ceo|cfo|coo|cto|md|managing director|chairman|chairperson|promoter|owner)\b",
    re.IGNORECASE,
)
_ROLE_QUERY_RE = re.compile(
    r"\b(?:who\s+is\s+the\s+)?(?:current\s+)?"
    r"(?P<role>ceo|cfo|coo|cto|md|managing director|chairman|chairperson|promoter|owner)"
    r"\s+of\s+(?P<company>[a-z0-9][a-z0-9 .&'/-]{2,160})\b",
    re.IGNORECASE,
)
_ORDER_CONTEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "according_to_order",
        re.compile(r"\b(?:according to|as per)\s+(?:the\s+)?(?:sebi\s+)?order\b", re.IGNORECASE),
    ),
    ("in_the_order_of", re.compile(r"\bin the order of\b", re.IGNORECASE)),
    ("in_the_matter_of", re.compile(r"\bin the matter of\b", re.IGNORECASE)),
    ("in_this_case", re.compile(r"\bin this case\b", re.IGNORECASE)),
    ("in_this_order", re.compile(r"\bin this order\b", re.IGNORECASE)),
    (
        "order_specific_reference",
        re.compile(r"\b(?:in|from)\s+(?:the\s+)?(?:sebi\s+)?order\b", re.IGNORECASE),
    ),
)
_SPACE_RE = re.compile(r"\s+")

_ROLE_NORMALIZATION = {
    "ceo": "ceo",
    "cfo": "cfo",
    "coo": "coo",
    "cto": "cto",
    "md": "md",
    "managing director": "managing_director",
    "chairman": "chairperson",
    "chairperson": "chairperson",
    "promoter": "promoter",
    "owner": "owner",
}


@dataclass(frozen=True)
class CompanyRoleQuery:
    """Parsed company-role current-fact query."""

    role: str
    company_name: str
    explicit_order_context: bool
    matched_signals: tuple[str, ...]


def parse_company_role_query(query: str) -> CompanyRoleQuery | None:
    """Return a normalized company-role query when the user asks for current leadership."""

    normalized_query = _SPACE_RE.sub(" ", query.strip())
    if not normalized_query:
        return None
    match = _ROLE_QUERY_RE.search(normalized_query)
    if match is None:
        return None
    role = _normalize_role(match.group("role"))
    company_name = _clean_company_name(match.group("company"))
    if not role or not company_name:
        return None
    explicit_order_context = False
    matched_signals: list[str] = [f"company_role:{role}"]
    for label, pattern in _ORDER_CONTEXT_PATTERNS:
        if pattern.search(normalized_query):
            explicit_order_context = True
            matched_signals.append(label)
    if not explicit_order_context:
        matched_signals.append("company_role_current_fact")
    return CompanyRoleQuery(
        role=role,
        company_name=company_name,
        explicit_order_context=explicit_order_context,
        matched_signals=tuple(dict.fromkeys(matched_signals)),
    )


def detect_company_role_order_context(query: str) -> tuple[str, ...]:
    """Return explicit order-context signals for company-role queries."""

    normalized_query = _SPACE_RE.sub(" ", query.strip())
    if not normalized_query or _ROLE_RE.search(normalized_query) is None:
        return ()
    return tuple(
        label
        for label, pattern in _ORDER_CONTEXT_PATTERNS
        if pattern.search(normalized_query)
    )


def _normalize_role(value: str) -> str:
    return _ROLE_NORMALIZATION.get(value.strip().lower(), "")


def _clean_company_name(value: str) -> str:
    cleaned = _SPACE_RE.sub(" ", value.strip(" .?,-"))
    cleaned = re.sub(
        r"\s+\b(?:according to|as per|in this case|in this order)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" .?,-")
    return cleaned
