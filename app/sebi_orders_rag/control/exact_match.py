"""Deterministic matter-lock matching using the generated control pack."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from .candidate_selection import SAT_COURT_BUCKETS, looks_like_sat_court_query
from ..normalization import (
    expand_query,
    generate_order_alias_variants,
    normalize_alias_text,
    rank_fuzzy_candidates,
)
from .models import ControlPack, MatterLockCandidate, StrictMatterLock

_COMPARISON_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("compare", re.compile(r"\bcompare\b", re.IGNORECASE)),
    ("comparison", re.compile(r"\bcomparison\b", re.IGNORECASE)),
    ("contrast", re.compile(r"\bcontrast\b", re.IGNORECASE)),
    ("difference", re.compile(r"\bdifference(?:s)?\b", re.IGNORECASE)),
    ("distinguish", re.compile(r"\bdistinguish\b", re.IGNORECASE)),
)
_TITLE_PREFIXES: tuple[str, ...] = (
    "tell me more about",
    "what happened in",
    "what was decided in",
    "what did sebi direct in",
    "what did sebi direct for",
    "what was the ipo of",
    "what was the",
    "what was",
    "tell me about",
    "summary of",
)
_QUERY_FOCUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^how much did (?P<focus>.+?) share price increase$"),
    re.compile(r"^what was the price before and after the increase in (?P<focus>.+)$"),
    re.compile(r"^what was the price before and after (?P<focus>.+)$"),
    re.compile(r"^give the price movement of (?P<focus>.+?) for each period$"),
    re.compile(r"^give the price movement of (?P<focus>.+)$"),
    re.compile(r"^what was the price movement of (?P<focus>.+)$"),
    re.compile(r"^what was the listing price of (?P<focus>.+)$"),
    re.compile(r"^what was the highest price of (?P<focus>.+)$"),
    re.compile(r"^what was the lowest price of (?P<focus>.+)$"),
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MATTER_PREFIX_TERMS = (
    "appeal no",
    "petition no",
    "case no",
    "order in the matter of",
    "in the matter of",
    "filed by",
    "vs",
    "versus",
)
_GENERIC_NAMED_QUERY_TERMS = (
    "ipo",
    "matter",
    "order",
    "case",
    "appeal",
    "revocation",
    "settlement",
    "corrigendum",
    "judgment",
    "sentencing",
)
_QUERY_SUFFIX_TERMS = ("case", "matter", "order")
_CANDIDATE_LIMIT = 5
_GENERIC_MATCH_TERMS = frozenset({"sebi"})
_QUERY_TOKEN_STOPWORDS = frozenset(
    {
        "about",
        "against",
        "and",
        "company",
        "concerning",
        "did",
        "for",
        "in",
        "india",
        "limited",
        "ltd",
        "matter",
        "more",
        "of",
        "order",
        "tell",
        "the",
        "what",
    }
)
_MATERIAL_QUERY_TOKENS = frozenset(
    {
        "appeal",
        "confirmatory",
        "corrigendum",
        "exemption",
        "judgment",
        "penalty",
        "sentence",
        "sentencing",
        "settlement",
    }
)
_PROCEDURAL_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rti_appeal", ("rti", "appeal")),
    ("settlement", ("settlement",)),
    ("exemption", ("exemption",)),
    ("revocation", ("revocation",)),
    ("confirmatory", ("confirmatory",)),
    ("corrigendum", ("corrigendum",)),
    ("judgment", ("judgment",)),
    ("sentencing", ("sentencing",)),
    ("adjudication", ("adjudication",)),
)
_MATTER_STYLE_TITLE_PREFIXES: tuple[str, ...] = (
    "in the matter of ",
    "order in the matter of ",
    "settlement order in the matter of ",
    "exemption order in the matter of ",
    "confirmatory order in the matter of ",
)


def resolve_strict_matter_lock(
    *,
    query: str,
    control_pack: ControlPack | None,
    title_lookup_signals: tuple[str, ...] = (),
    matter_reference_signals: tuple[str, ...] = (),
) -> StrictMatterLock:
    """Resolve whether a query should be locked to one named matter."""

    comparison_terms = detect_comparison_terms(query)
    if control_pack is None:
        return StrictMatterLock(
            comparison_intent=bool(comparison_terms),
            comparison_terms=comparison_terms,
            reason_codes=("control_pack_unavailable",),
        )

    expanded_query = expand_query(query, contexts=("order_lookup", "order_legal"))
    query_variants = tuple(
        dict.fromkeys(
            normalize_match_text(value)
            for value in expanded_query.expansions
            if normalize_match_text(value)
        )
    ) or (normalize_match_text(query),)
    normalized_query = query_variants[0]
    query_focuses = tuple(_extract_query_focus(value) for value in query_variants)
    query_tokens = {
        token
        for variant in query_variants
        for token in _TOKEN_RE.findall(variant)
    }
    distinctive_query_tokens = _distinctive_query_tokens(query_tokens)
    sat_court_query = looks_like_sat_court_query(query)
    procedural_hints = _extract_procedural_hints(query_tokens)
    alias_matches = _match_alias_variants(
        query_variants,
        control_pack,
        focus_queries=query_focuses,
    )
    title_matches: list[str] = []
    candidate_rows: dict[str, MatterLockCandidate] = {}

    for document in control_pack.document_index:
        normalized_title = normalize_match_text(document.exact_title)
        exact_title_match = any(variant == normalized_title for variant in query_variants)
        title_contained = (
            any(normalized_title in variant for variant in query_variants)
            or any(normalized_title == focus for focus in query_focuses)
        )
        record_key_match = any(document.record_key.lower() in variant for variant in query_variants)
        matched_aliases = tuple(
            sorted(
                {
                    alias
                    for alias, record_keys in alias_matches.items()
                    if document.record_key in record_keys
                }
            )
        )
        matched_entity_terms = tuple(
            sorted(
                {
                    entity.lower()
                    for entity in document.main_entities
                    if entity and _entity_matches_query(
                        entity=entity,
                        query_variants=query_variants,
                        focus_queries=query_focuses,
                    )
                }
            )
        )
        distinctive_aliases = tuple(
            alias for alias in matched_aliases if alias not in _GENERIC_MATCH_TERMS
        )
        distinctive_entity_terms = tuple(
            entity for entity in matched_entity_terms if entity not in _GENERIC_MATCH_TERMS
        )
        title_similarity = max(
            max(_sequence_similarity(focus, normalized_title) for focus in query_focuses),
            max(_sequence_similarity(variant, normalized_title) for variant in query_variants),
        )
        title_overlap_ratio = _token_overlap_ratio(
            query_tokens,
            set(_TOKEN_RE.findall(normalize_match_text(document.exact_title))),
        )
        distinctive_overlap_ratio = _token_overlap_ratio(
            distinctive_query_tokens,
            set(_TOKEN_RE.findall(normalize_match_text(document.exact_title))),
        )
        distinctive_query_coverage = _query_token_coverage(
            distinctive_query_tokens,
            set(_TOKEN_RE.findall(normalize_match_text(document.exact_title))),
        )
        procedural_alignment = _resolve_procedural_alignment(
            procedural_hints=procedural_hints,
            title=document.exact_title,
            procedural_type=document.procedural_type,
            bucket_name=document.bucket_category,
        )
        procedural_preference = _resolve_default_procedural_preference(
            query_tokens=query_tokens,
            title=document.exact_title,
        )

        score = 0.0
        if record_key_match:
            score += 1.25
        if exact_title_match:
            score += 1.10
        elif title_contained:
            score += 0.85
        if distinctive_aliases:
            score += min(
                0.72,
                0.16 + (0.16 * len(distinctive_aliases)) + (0.04 * len(distinctive_aliases)),
            )
        if distinctive_entity_terms:
            score += min(
                0.45,
                (0.18 * len(distinctive_entity_terms)) + (0.03 * len(distinctive_entity_terms)),
            )
        score += title_similarity * 0.40
        score += title_overlap_ratio * 0.35
        score += distinctive_overlap_ratio * 0.30
        score += distinctive_query_coverage * 0.42
        if any(_matter_prefix_present(variant) for variant in query_variants):
            score += 0.08
        if query_tokens & set(_TOKEN_RE.findall(document.bucket_category.lower())):
            score += 0.03
        score *= procedural_alignment
        score *= procedural_preference
        score *= _resolve_bucket_alignment(
            sat_court_query=sat_court_query,
            bucket_name=document.bucket_category,
        )
        if (
            sat_court_query
            and (matched_aliases or matched_entity_terms)
            and not distinctive_aliases
            and not distinctive_entity_terms
            and title_overlap_ratio < 0.45
        ):
            score *= 0.62

        if score < 0.42:
            continue

        if exact_title_match or title_contained:
            title_matches.append(document.exact_title)
        candidate_rows[document.record_key] = MatterLockCandidate(
            record_key=document.record_key,
            title=document.exact_title,
            bucket_name=document.bucket_category,
            document_version_id=document.document_version_id,
            canonical_entities=document.main_entities,
            score=round(score, 4),
            exact_title_match=exact_title_match or title_contained,
            record_key_match=record_key_match,
            matched_aliases=matched_aliases,
            matched_entity_terms=matched_entity_terms,
            title_similarity=round(title_similarity, 4),
            title_overlap_ratio=round(title_overlap_ratio, 4),
        )

    ordered_candidates = tuple(
        sorted(
            candidate_rows.values(),
            key=lambda item: (-item.score, -int(item.exact_title_match), item.record_key),
        )[:_CANDIDATE_LIMIT]
    )
    promoted_primary_candidate = _promote_primary_matter_candidate(
        query_tokens=query_tokens,
        ordered_candidates=ordered_candidates,
        procedural_hints=procedural_hints,
    )
    suppressed_duplicate_title: str | None = None
    if promoted_primary_candidate is not None:
        suppressed_duplicate_title = normalize_match_text(ordered_candidates[0].title)
        ordered_candidates = (
            promoted_primary_candidate,
            *tuple(
                candidate
                for candidate in ordered_candidates
                if candidate.record_key != promoted_primary_candidate.record_key
            ),
        )
    matched_alias_values = tuple(
        sorted(alias for alias in alias_matches if alias not in _GENERIC_MATCH_TERMS)
    )
    named_matter_query = _is_named_matter_query(
        normalized_query=normalized_query,
        title_lookup_signals=title_lookup_signals,
        matter_reference_signals=matter_reference_signals,
        matched_aliases=matched_alias_values,
        candidates=ordered_candidates,
    )
    strict_scope_required = named_matter_query and not comparison_terms
    if not strict_scope_required:
        return StrictMatterLock(
            named_matter_query=named_matter_query,
            strict_scope_required=False,
            strict_single_matter=False,
            ambiguous=False,
            comparison_intent=bool(comparison_terms),
            comparison_terms=comparison_terms,
            matched_aliases=matched_alias_values,
            matched_entities=_collect_matched_entities(ordered_candidates),
            matched_titles=tuple(dict.fromkeys(title_matches)),
            locked_record_keys=(),
            candidates=ordered_candidates,
            reason_codes=("comparison_intent" if comparison_terms else "no_named_matter_lock",),
        )

    if not ordered_candidates:
        return StrictMatterLock(
            named_matter_query=named_matter_query,
            strict_scope_required=True,
            strict_single_matter=False,
            ambiguous=True,
            comparison_intent=False,
            matched_aliases=matched_alias_values,
            matched_entities=(),
            matched_titles=(),
            locked_record_keys=(),
            candidates=(),
            reason_codes=("named_matter_without_candidate",),
        )

    top_candidate = ordered_candidates[0]
    comparison_candidates = _comparison_candidates(
        ordered_candidates=ordered_candidates,
        suppressed_duplicate_title=suppressed_duplicate_title,
    )
    second_candidate = comparison_candidates[0] if comparison_candidates else None
    top_gap = top_candidate.score - (second_candidate.score if second_candidate else 0.0)
    repeated_title_ambiguity = bool(
        second_candidate
        and normalize_match_text(top_candidate.title) == normalize_match_text(second_candidate.title)
        and top_gap < 0.35
    )
    confusable_competitor = _has_confusable_competitor(
        top_candidate=top_candidate,
        ordered_candidates=(top_candidate, *comparison_candidates),
        control_pack=control_pack,
        procedural_hints=procedural_hints,
    )
    clearly_dominant = (
        not repeated_title_ambiguity
        and not confusable_competitor
        and (
            top_candidate.score >= 0.90
        or (
            second_candidate is None
            and top_candidate.score >= 0.52
            and (top_candidate.matched_entity_terms or top_candidate.matched_aliases)
        )
        or (
            second_candidate is None
            and top_candidate.score >= 0.44
            and top_candidate.title_overlap_ratio >= 0.14
        )
        or (
            top_candidate.score >= 0.72
            and (
                second_candidate is None
                or top_gap >= 0.18
                or (
                    top_candidate.exact_title_match
                    and top_gap >= 0.10
                    and second_candidate.record_key != top_candidate.record_key
                )
            )
        )
        or (
            second_candidate is not None
            and top_candidate.score >= 0.46
            and top_gap >= 0.02
            and (top_candidate.matched_entity_terms or top_candidate.matched_aliases)
            and not (second_candidate.matched_entity_terms or second_candidate.matched_aliases)
        )
        )
    )
    ambiguous = not clearly_dominant
    reason_codes = ["named_matter_query"]
    if sat_court_query:
        reason_codes.append("sat_court_priority")
    if top_candidate.record_key_match:
        reason_codes.append("record_key_match")
    if top_candidate.exact_title_match:
        reason_codes.append("exact_title_or_contained_match")
    if top_candidate.matched_aliases:
        reason_codes.append("alias_match")
    if ambiguous:
        reason_codes.append("ambiguous_named_matter")
        if repeated_title_ambiguity:
            reason_codes.append("repeated_title_ambiguity")
        if confusable_competitor:
            reason_codes.append("confusable_competitor")
    else:
        reason_codes.append("single_matter_lock")
        if promoted_primary_candidate is not None:
            reason_codes.append("primary_matter_preference")

    return StrictMatterLock(
        named_matter_query=named_matter_query,
        strict_scope_required=True,
        strict_single_matter=not ambiguous,
        ambiguous=ambiguous,
        comparison_intent=False,
        matched_aliases=matched_alias_values,
        matched_entities=_collect_matched_entities(ordered_candidates),
        matched_titles=tuple(dict.fromkeys(title_matches)),
        locked_record_keys=(top_candidate.record_key,) if not ambiguous else (),
        candidates=ordered_candidates,
        reason_codes=tuple(reason_codes),
    )


def detect_comparison_terms(query: str) -> tuple[str, ...]:
    """Return comparison terms that disable the single-matter lock."""

    return tuple(label for label, pattern in _COMPARISON_PATTERNS if pattern.search(query))


def normalize_match_text(text: str) -> str:
    """Normalize titles, aliases, and queries into comparable form."""

    return " ".join(_TOKEN_RE.findall(normalize_alias_text(text)))


def confusion_penalty_map(
    *,
    control_pack: ControlPack | None,
    strict_lock: StrictMatterLock | None,
) -> dict[str, float]:
    """Return per-record penalty multipliers for known confusing alternate matters."""

    if (
        control_pack is None
        or strict_lock is None
        or not strict_lock.strict_single_matter
        or not strict_lock.locked_record_keys
    ):
        return {}

    locked_record_key = strict_lock.locked_record_keys[0]
    penalties: dict[str, float] = {}
    for pair in control_pack.confusion_map.get(locked_record_key, ()):
        if pair.record_key_a == locked_record_key:
            penalties[pair.record_key_b] = 0.08
        elif pair.record_key_b == locked_record_key:
            penalties[pair.record_key_a] = 0.08
    return penalties


def _extract_query_focus(query: str) -> str:
    normalized = query if " " in query and query == query.lower() else normalize_match_text(query)
    for pattern in _QUERY_FOCUS_PATTERNS:
        match = pattern.match(normalized)
        if match:
            return _strip_query_suffix_terms(match.group("focus").strip())
    for prefix in _TITLE_PREFIXES:
        normalized_prefix = normalize_match_text(prefix)
        if normalized.startswith(normalized_prefix + " "):
            normalized = normalized[len(normalized_prefix) + 1 :].strip()
            break
    return _strip_query_suffix_terms(normalized)


def _strip_query_suffix_terms(value: str) -> str:
    tokens = value.split()
    while len(tokens) > 1 and tokens[-1] in _QUERY_SUFFIX_TERMS:
        tokens.pop()
    return " ".join(tokens)


def _match_alias_variants(
    normalized_queries: tuple[str, ...],
    control_pack: ControlPack,
    *,
    focus_queries: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    matched: dict[str, tuple[str, ...]] = {}
    alias_variants = tuple(control_pack.alias_variants)
    for variant, alias_rows in control_pack.alias_variants.items():
        if len(variant) < 3:
            continue
        variant_tokens = set(variant.split())
        partial_alias_match = bool(
            any(
                len(focus.split()) >= 2
                and set(focus.split()).issubset(variant_tokens)
                for focus in focus_queries
                if focus
            )
        )
        if not any(_contains_phrase(query, variant) for query in normalized_queries) and not partial_alias_match:
            continue
        record_keys: list[str] = []
        for alias_row in alias_rows:
            record_keys.extend(alias_row.related_record_keys)
        matched[variant] = tuple(dict.fromkeys(record_keys))
    for focus in focus_queries:
        if len(focus.split()) < 2:
            continue
        fuzzy_result = rank_fuzzy_candidates(
            focus,
            alias_variants,
            key=lambda value: value,
            min_score=0.82,
            medium_score=0.9,
            confident_score=0.93,
            ambiguity_gap=0.04,
        )
        if fuzzy_result.top_candidate is None:
            continue
        top_candidate = fuzzy_result.top_candidate
        second_candidate = fuzzy_result.candidates[1] if len(fuzzy_result.candidates) > 1 else None
        safe_alias_fuzzy = fuzzy_result.band == "high" or (
            top_candidate.score >= 0.84
            and top_candidate.first_token_similarity >= 0.9
            and top_candidate.last_token_similarity >= 0.9
            and top_candidate.token_overlap >= 0.5
            and (second_candidate is None or second_candidate.score <= top_candidate.score - 0.04)
        )
        if not safe_alias_fuzzy:
            continue
        variant = top_candidate.value
        if variant in matched:
            continue
        record_keys: list[str] = []
        for alias_row in control_pack.alias_variants.get(variant, ()):
            record_keys.extend(alias_row.related_record_keys)
        if record_keys:
            matched[variant] = tuple(dict.fromkeys(record_keys))
    return matched


def _contains_phrase(normalized_query: str, normalized_phrase: str) -> bool:
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_query} "


def _entity_matches_query(
    *,
    entity: str,
    query_variants: tuple[str, ...],
    focus_queries: tuple[str, ...],
) -> bool:
    aliases = tuple(
        dict.fromkeys(
            normalize_match_text(value)
            for value in (entity, *generate_order_alias_variants(entity))
            if normalize_match_text(value)
        )
    )
    for alias in aliases:
        alias_tokens = set(alias.split())
        if any(_contains_phrase(query, alias) for query in query_variants):
            return True
        if any(
            len(focus.split()) >= 2 and set(focus.split()).issubset(alias_tokens)
            for focus in focus_queries
            if focus
        ):
            return True
    return False


def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    return len(overlap) / max(len(left), len(right))


def _query_token_coverage(query_tokens: set[str], title_tokens: set[str]) -> float:
    if not query_tokens or not title_tokens:
        return 0.0
    overlap = query_tokens & title_tokens
    return len(overlap) / len(query_tokens)


def _distinctive_query_tokens(query_tokens: set[str]) -> set[str]:
    return {
        token
        for token in query_tokens
        if len(token) >= 3 and token not in _QUERY_TOKEN_STOPWORDS
    }


def _sequence_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def _extract_procedural_hints(query_tokens: set[str]) -> tuple[str, ...]:
    hints: list[str] = []
    for label, tokens in _PROCEDURAL_HINTS:
        if all(token in query_tokens for token in tokens):
            hints.append(label)
    return tuple(hints)


def _resolve_procedural_alignment(
    *,
    procedural_hints: tuple[str, ...],
    title: str,
    procedural_type: str | None,
    bucket_name: str,
) -> float:
    if not procedural_hints:
        return 1.0
    haystack = " ".join(
        value
        for value in (
            normalize_match_text(title),
            normalize_match_text(procedural_type or ""),
            normalize_match_text(bucket_name),
        )
        if value
    )
    matched = 0
    for hint in procedural_hints:
        hint_tokens = dict(_PROCEDURAL_HINTS).get(hint, ())
        if all(token in haystack for token in hint_tokens):
            matched += 1
    if matched == len(procedural_hints):
        return 1.0
    if matched > 0:
        return 0.72
    return 0.28


def _resolve_default_procedural_preference(
    *,
    query_tokens: set[str],
    title: str,
) -> float:
    normalized_title = normalize_match_text(title)
    if not normalized_title:
        return 1.0
    if "corrigendum" in normalized_title and "corrigendum" not in query_tokens:
        return 0.66
    if "sentencing" in normalized_title and not (query_tokens & {"sentencing", "sentence", "penalty"}):
        return 0.68
    if "judgment" in normalized_title and not (query_tokens & {"sentencing", "sentence", "penalty"}):
        return 1.08
    if (
        normalized_title.startswith("in the matter of ")
        or normalized_title.startswith("order in the matter of ")
        or normalized_title.startswith("settlement order in the matter of ")
        or normalized_title.startswith("exemption order in the matter of ")
        or normalized_title.startswith("confirmatory order in the matter of ")
    ) and not (query_tokens & _MATERIAL_QUERY_TOKENS):
        return 1.10
    return 1.0


def _promote_primary_matter_candidate(
    *,
    query_tokens: set[str],
    ordered_candidates: tuple[MatterLockCandidate, ...],
    procedural_hints: tuple[str, ...],
) -> MatterLockCandidate | None:
    if procedural_hints or query_tokens & _MATERIAL_QUERY_TOKENS or len(ordered_candidates) < 3:
        return None
    duplicate_title = normalize_match_text(ordered_candidates[0].title)
    if _looks_like_matter_style_title(duplicate_title):
        return None
    duplicate_cluster = tuple(
        candidate
        for candidate in ordered_candidates
        if normalize_match_text(candidate.title) == duplicate_title
    )
    if len(duplicate_cluster) < 2:
        return None
    alternatives = [
        candidate
        for candidate in ordered_candidates
        if normalize_match_text(candidate.title) != duplicate_title
        and _looks_like_matter_style_title(normalize_match_text(candidate.title))
        and (
            set(candidate.matched_aliases) & set(duplicate_cluster[0].matched_aliases)
            or set(candidate.matched_entity_terms) & set(duplicate_cluster[0].matched_entity_terms)
            or candidate.title_overlap_ratio >= duplicate_cluster[0].title_overlap_ratio - 0.02
        )
        and candidate.score >= duplicate_cluster[0].score * 0.62
    ]
    if not alternatives:
        return None
    return sorted(
        alternatives,
        key=lambda item: (-item.score, -item.title_overlap_ratio, item.record_key),
    )[0]


def _comparison_candidates(
    *,
    ordered_candidates: tuple[MatterLockCandidate, ...],
    suppressed_duplicate_title: str | None,
) -> tuple[MatterLockCandidate, ...]:
    if not suppressed_duplicate_title:
        return ordered_candidates[1:]
    return tuple(
        candidate
        for candidate in ordered_candidates[1:]
        if normalize_match_text(candidate.title) != suppressed_duplicate_title
    )


def _looks_like_matter_style_title(normalized_title: str) -> bool:
    return any(
        normalized_title.startswith(prefix)
        for prefix in _MATTER_STYLE_TITLE_PREFIXES
    )


def _has_confusable_competitor(
    *,
    top_candidate: MatterLockCandidate,
    ordered_candidates: tuple[MatterLockCandidate, ...],
    control_pack: ControlPack,
    procedural_hints: tuple[str, ...],
) -> bool:
    if procedural_hints:
        return False
    confusion_pairs = control_pack.confusion_map.get(top_candidate.record_key, ())
    if not confusion_pairs:
        return False
    confusing_keys = {
        pair.record_key_b if pair.record_key_a == top_candidate.record_key else pair.record_key_a
        for pair in confusion_pairs
    }
    for candidate in ordered_candidates[1:]:
        if candidate.record_key in confusing_keys and candidate.score >= top_candidate.score - 0.20:
            if top_candidate.title_overlap_ratio >= candidate.title_overlap_ratio + 0.12:
                continue
            if top_candidate.exact_title_match and not candidate.exact_title_match and candidate.score <= top_candidate.score - 0.08:
                continue
            return True
    return False


def _matter_prefix_present(normalized_query: str) -> bool:
    return any(term in normalized_query for term in _MATTER_PREFIX_TERMS)


def _is_named_matter_query(
    *,
    normalized_query: str,
    title_lookup_signals: tuple[str, ...],
    matter_reference_signals: tuple[str, ...],
    matched_aliases: tuple[str, ...],
    candidates: tuple[MatterLockCandidate, ...],
) -> bool:
    if title_lookup_signals or matter_reference_signals or matched_aliases:
        return True
    if candidates and (
        candidates[0].score >= 0.58
        or (
            candidates[0].score >= 0.52
            and (candidates[0].matched_entity_terms or candidates[0].matched_aliases)
        )
    ):
        return True
    query_tokens = set(_TOKEN_RE.findall(normalized_query))
    if len(query_tokens) >= 2 and query_tokens & set(_GENERIC_NAMED_QUERY_TERMS):
        return True
    return False


def _collect_matched_entities(candidates: tuple[MatterLockCandidate, ...]) -> tuple[str, ...]:
    values = []
    for candidate in candidates:
        values.extend(
            value
            for value in candidate.matched_entity_terms
            if value not in _GENERIC_MATCH_TERMS
        )
        values.extend(
            value
            for value in candidate.matched_aliases
            if value not in _GENERIC_MATCH_TERMS
        )
    return tuple(dict.fromkeys(values))


def _resolve_bucket_alignment(*, sat_court_query: bool, bucket_name: str) -> float:
    normalized_bucket = normalize_match_text(bucket_name)
    if not sat_court_query:
        return 1.0
    if normalized_bucket in {normalize_match_text(value) for value in SAT_COURT_BUCKETS}:
        return 1.22
    return 0.84
