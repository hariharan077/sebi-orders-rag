"""Reusable query expansion shared across routing and lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .abbreviations import AbbreviationExpansion, expand_abbreviations
from .aliases import generate_order_alias_variants, normalize_alias_text


@dataclass(frozen=True)
class ExpandedQuery:
    """Normalized query plus conservative expansion variants."""

    raw_query: str
    normalized_query: str
    expansions: tuple[str, ...]
    abbreviation_expansion: AbbreviationExpansion

    @property
    def matched_abbreviations(self) -> tuple[str, ...]:
        return self.abbreviation_expansion.matched_abbreviations


def expand_query(
    query: str,
    *,
    contexts: Iterable[str] = (),
) -> ExpandedQuery:
    """Expand abbreviations and entity aliases without changing the core architecture."""

    abbreviation_expansion = expand_abbreviations(query, contexts=contexts)
    ordered: list[str] = []
    for value in (normalize_alias_text(query), *abbreviation_expansion.expansions):
        normalized = normalize_alias_text(value)
        if normalized and normalized not in ordered:
            ordered.append(normalized)

    requested = {value for value in contexts if value}
    if requested.intersection({"order_lookup", "order_legal"}):
        for value in tuple(ordered):
            for variant in generate_order_alias_variants(value):
                if variant not in ordered:
                    ordered.append(variant)

    normalized_query = normalize_alias_text(
        abbreviation_expansion.normalized_query or (ordered[0] if ordered else "")
    )
    return ExpandedQuery(
        raw_query=query,
        normalized_query=normalized_query,
        expansions=tuple(ordered),
        abbreviation_expansion=abbreviation_expansion,
    )
