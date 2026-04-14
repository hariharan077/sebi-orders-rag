"""Conservative ambiguity handling for canonical SEBI person lookups."""

from __future__ import annotations

from dataclasses import dataclass

from ..normalization import FuzzyCandidate, rank_fuzzy_candidates
from .canonical_models import CanonicalPersonRecord
from .canonicalize_people import normalized_name_key


@dataclass(frozen=True)
class PersonMatchResult:
    """Outcome of one canonical person lookup attempt."""

    status: str
    match_stage: str | None = None
    matches: tuple[CanonicalPersonRecord, ...] = ()
    clarification_candidates: tuple[CanonicalPersonRecord, ...] = ()
    fuzzy_candidates: tuple[FuzzyCandidate[CanonicalPersonRecord], ...] = ()
    fuzzy_band: str | None = None


def match_people(
    people: tuple[CanonicalPersonRecord, ...],
    query_name: str,
) -> PersonMatchResult:
    """Return one deterministic high/clarify/abstain result for a person query."""

    normalized_query = normalized_name_key(query_name)
    if not normalized_query:
        return PersonMatchResult(status="no_match")

    exact_name_matches = tuple(
        person
        for person in people
        if person.normalized_name_key == normalized_query
    )
    if len(exact_name_matches) == 1:
        return PersonMatchResult(
            status="exact",
            match_stage="exact_name",
            matches=exact_name_matches,
        )
    if len(exact_name_matches) > 1:
        return PersonMatchResult(
            status="clarify",
            match_stage="exact_name",
            clarification_candidates=exact_name_matches[:5],
        )

    exact_alias_matches = tuple(
        person
        for person in people
        if any(
            normalized_name_key(alias) == normalized_query
            and normalized_name_key(alias) != person.normalized_name_key
            for alias in person.aliases
        )
    )
    if len(exact_alias_matches) == 1:
        return PersonMatchResult(
            status="high",
            match_stage="exact_alias",
            matches=exact_alias_matches,
        )
    if len(exact_alias_matches) > 1:
        return PersonMatchResult(
            status="clarify",
            match_stage="exact_alias",
            clarification_candidates=exact_alias_matches[:5],
        )

    token_matches = tuple(
        person
        for person in people
        if _token_exact_match(person, normalized_query)
    )
    if len(token_matches) == 1 and _allow_single_token_answer(normalized_query):
        return PersonMatchResult(
            status="high",
            match_stage="single_token_exact",
            matches=token_matches,
        )
    if len(token_matches) > 1:
        return PersonMatchResult(
            status="clarify",
            match_stage="single_token_exact",
            clarification_candidates=token_matches[:5],
        )

    fuzzy = rank_fuzzy_candidates(
        query_name,
        people,
        key=lambda person: person.canonical_name,
        min_score=0.72,
        medium_score=0.82,
        confident_score=0.86,
        ambiguity_gap=0.05,
    )
    if fuzzy.band == "high" and fuzzy.confident_match is not None:
        return PersonMatchResult(
            status="high",
            match_stage="fuzzy_high",
            matches=(fuzzy.confident_match,),
            fuzzy_candidates=fuzzy.candidates,
            fuzzy_band=fuzzy.band,
        )
    if fuzzy.band == "medium" and fuzzy.candidates:
        candidates = tuple(candidate.value for candidate in fuzzy.candidates[:5])
        return PersonMatchResult(
            status="clarify",
            match_stage="fuzzy_medium",
            clarification_candidates=candidates,
            fuzzy_candidates=fuzzy.candidates,
            fuzzy_band=fuzzy.band,
        )
    return PersonMatchResult(
        status="no_match",
        match_stage="fuzzy_low" if fuzzy.candidates else "no_match",
        fuzzy_candidates=fuzzy.candidates,
        fuzzy_band=fuzzy.band,
    )


def _token_exact_match(person: CanonicalPersonRecord, normalized_query: str) -> bool:
    if len(normalized_query.split()) != 1:
        return False
    person_tokens = tuple(token for token in person.normalized_name_key.split() if token)
    if not person_tokens:
        return False
    if normalized_query in person_tokens:
        return True
    return any(
        normalized_query in tuple(token for token in normalized_name_key(alias).split() if token)
        for alias in person.aliases
    )


def _allow_single_token_answer(normalized_query: str) -> bool:
    tokens = normalized_query.split()
    return len(tokens) == 1 and len(tokens[0]) >= 4
