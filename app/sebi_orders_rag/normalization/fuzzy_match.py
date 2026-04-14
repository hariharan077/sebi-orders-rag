"""Conservative fuzzy matching for people and entity names."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Generic, Iterable, Literal, TypeVar

_T = TypeVar("_T")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_HONORIFIC_RE = re.compile(r"\b(?:mr|mrs|ms|m/s|dr|shri|smt)\b", re.IGNORECASE)
FuzzyBand = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class FuzzyCandidate(Generic[_T]):
    """One ranked fuzzy match candidate."""

    value: _T
    display_name: str
    normalized_name: str
    score: float
    match_type: str
    token_overlap: float = 0.0
    first_token_similarity: float = 0.0
    last_token_similarity: float = 0.0


@dataclass(frozen=True)
class FuzzyMatchResult(Generic[_T]):
    """Ranked candidates plus conservative confidence flags."""

    query: str
    normalized_query: str
    candidates: tuple[FuzzyCandidate[_T], ...]
    confident_match: _T | None = None
    ambiguous: bool = False
    band: FuzzyBand = "low"
    clarification_candidate: FuzzyCandidate[_T] | None = None

    @property
    def top_candidate(self) -> FuzzyCandidate[_T] | None:
        return self.candidates[0] if self.candidates else None


def normalize_fuzzy_text(value: str) -> str:
    """Normalize free text for conservative typo-tolerant comparisons."""

    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = ascii_only.lower()
    cleaned = _HONORIFIC_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("&", " and ")
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def rank_fuzzy_candidates(
    query: str,
    values: Iterable[_T],
    *,
    key: Callable[[_T], str],
    min_score: float = 0.72,
    medium_score: float = 0.82,
    confident_score: float = 0.84,
    ambiguity_gap: float = 0.05,
) -> FuzzyMatchResult[_T]:
    """Rank candidates and return one confident match only when the gap is clear."""

    normalized_query = normalize_fuzzy_text(query)
    if not normalized_query:
        return FuzzyMatchResult(query=query, normalized_query="", candidates=())

    ranked: list[FuzzyCandidate[_T]] = []
    for value in values:
        display_name = key(value)
        normalized_name = normalize_fuzzy_text(display_name)
        if not normalized_name:
            continue
        score, match_type, details = _score_match(normalized_query, normalized_name)
        if score < min_score:
            continue
        ranked.append(
            FuzzyCandidate(
                value=value,
                display_name=display_name,
                normalized_name=normalized_name,
                score=round(score, 4),
                match_type=match_type,
                token_overlap=round(details["token_overlap"], 4),
                first_token_similarity=round(details["first_token_similarity"], 4),
                last_token_similarity=round(details["last_token_similarity"], 4),
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.display_name))
    candidates = tuple(ranked)
    if not candidates:
        return FuzzyMatchResult(query=query, normalized_query=normalized_query, candidates=())

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    band = _classify_match_band(
        normalized_query=normalized_query,
        top=top,
        second=second,
        medium_score=medium_score,
        confident_score=confident_score,
        ambiguity_gap=ambiguity_gap,
    )
    ambiguous = band == "medium" or bool(
        second and top.score < confident_score and second.score >= top.score - ambiguity_gap
    )
    confident_match = top.value if band == "high" else None
    return FuzzyMatchResult(
        query=query,
        normalized_query=normalized_query,
        candidates=candidates,
        confident_match=confident_match,
        ambiguous=ambiguous,
        band=band,
        clarification_candidate=top if band == "medium" else None,
    )


def _score_match(query: str, candidate: str) -> tuple[float, str, dict[str, float]]:
    if query == candidate:
        return 1.0, "exact", {
            "token_overlap": 1.0,
            "first_token_similarity": 1.0,
            "last_token_similarity": 1.0,
        }

    query_tokens = tuple(token for token in query.split() if token)
    candidate_tokens = tuple(token for token in candidate.split() if token)
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    overlap = query_set.intersection(candidate_set)
    token_overlap = len(overlap) / max(len(query_set), len(candidate_set))
    sequence = SequenceMatcher(a=query, b=candidate).ratio()
    compact_sequence = SequenceMatcher(a=query.replace(" ", ""), b=candidate.replace(" ", "")).ratio()
    contains = query in candidate or candidate in query
    suffix_match = bool(query_tokens and candidate_tokens and query_tokens[-1] == candidate_tokens[-1])
    prefix_similarity = 0.0
    suffix_similarity = 0.0
    if query_tokens and candidate_tokens:
        prefix_similarity = SequenceMatcher(a=query_tokens[0], b=candidate_tokens[0]).ratio()
        suffix_similarity = SequenceMatcher(a=query_tokens[-1], b=candidate_tokens[-1]).ratio()

    details = {
        "token_overlap": token_overlap,
        "first_token_similarity": prefix_similarity,
        "last_token_similarity": suffix_similarity,
    }

    if contains and (len(query) >= 5 or len(query_tokens) >= 2):
        score = 0.79 + (0.12 * compact_sequence)
        if suffix_match:
            score += 0.05
        return min(score, 0.97), "partial", details

    if len(query_tokens) >= 2 and suffix_match and prefix_similarity >= 0.72:
        score = 0.58 + (0.22 * prefix_similarity) + (0.16 * token_overlap)
        return min(score + (0.08 * compact_sequence), 0.96), "name_tokens", details

    score = (sequence * 0.42) + (compact_sequence * 0.33) + (token_overlap * 0.25)
    if suffix_match:
        score += 0.05
    if len(query_tokens) == 1 and len(query_tokens[0]) < 4:
        score -= 0.08
    return min(score, 0.95), "fuzzy", details


def _classify_match_band(
    *,
    normalized_query: str,
    top: FuzzyCandidate[object],
    second: FuzzyCandidate[object] | None,
    medium_score: float,
    confident_score: float,
    ambiguity_gap: float,
) -> FuzzyBand:
    query_tokens = tuple(token for token in normalized_query.split() if token)
    clear_gap = second is None or second.score <= top.score - ambiguity_gap
    if not query_tokens:
        return "low"
    if top.match_type == "exact":
        return "high"
    if len(query_tokens) == 1:
        if top.score >= max(confident_score + 0.08, 0.92) and clear_gap:
            return "high"
        if top.score >= medium_score and clear_gap:
            return "medium"
        return "low"

    strong_first_token = top.first_token_similarity >= 0.86
    exact_last_token = top.last_token_similarity >= 0.995
    good_token_overlap = top.token_overlap >= 0.5
    if top.score >= confident_score and strong_first_token and good_token_overlap and clear_gap:
        return "high"
    if top.score >= medium_score and exact_last_token and top.first_token_similarity >= 0.45 and clear_gap:
        return "medium"
    if top.score >= confident_score + 0.04 and good_token_overlap and clear_gap:
        return "medium"
    return "low"
