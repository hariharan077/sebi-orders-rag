"""Institutional-facts routing helpers for current SEBI official lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass

_INCOME_QUERY_RE = re.compile(
    r"\b(?:sources?\s+of\s+income|income\s+sources?|revenue\s+sources?|how\s+does\s+sebi\s+earn|income\s+of\s+sebi|revenues?\s+of\s+sebi)\b",
    re.IGNORECASE,
)
_FEE_QUERY_RE = re.compile(
    r"\b(?:commission|charge(?:s)?|fee|fees|per trade|transaction charge(?:s)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InstitutionalFactsPlan:
    """Official-search plan for one institutional-facts query."""

    lookup_type: str
    search_query: str
    search_instructions: str


def classify_institutional_facts_query(query: str) -> InstitutionalFactsPlan | None:
    """Return an official-search plan for income/fee institutional facts."""

    normalized_query = " ".join(query.lower().split())
    if _INCOME_QUERY_RE.search(normalized_query):
        return InstitutionalFactsPlan(
            lookup_type="sebi_income_sources",
            search_query="SEBI Act section 14 general fund sources of income fees charges grants investments",
            search_instructions=(
                "Answer only from current official SEBI or Government of India webpages. "
                "Prefer pages that directly describe SEBI's sources of funds, revenue, or income. "
                "If the official sources do not directly answer the question, say insufficient."
            ),
        )
    if _FEE_QUERY_RE.search(normalized_query):
        return InstitutionalFactsPlan(
            lookup_type="sebi_fee_or_charge_query",
            search_query="Does SEBI charge commission per trade or only regulatory fees turnover based official",
            search_instructions=(
                "Answer only from current official SEBI or Government of India webpages. "
                "Do not guess about brokerage, exchange, or third-party charges. "
                "If official sources show that SEBI does not charge a flat investor-facing commission per trade, answer that explicitly."
            ),
        )
    return None


def is_institutional_facts_query(query: str) -> bool:
    """Return whether the query is an institutional-facts question."""

    return classify_institutional_facts_query(query) is not None
