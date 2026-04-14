"""Deterministic candidate ranking, clarify payloads, and selection aliases."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ..normalization import generate_order_alias_variants, normalize_alias_text

if TYPE_CHECKING:
    from ..schemas import ClarificationCandidate

SAT_COURT_BUCKETS: tuple[str, ...] = (
    "orders-of-sat",
    "orders-of-courts",
    "orders-of-special-courts",
)
_SAT_COURT_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:vs\.?\s+sebi|sebi\s+vs\.?|versus\s+sebi)\b", re.IGNORECASE),
    re.compile(r"\b(?:vs\.?|versus|v\.)\b", re.IGNORECASE),
    re.compile(r"\b(?:sat|appeal|w\.?\s*p\.?|writ petition|court|tribunal|judgment)\b", re.IGNORECASE),
)
_BUCKET_SELECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "orders-of-sat": ("sat", "sat case", "sat order", "tribunal"),
    "orders-of-courts": ("court", "court case", "writ petition", "wp", "w.p."),
    "orders-of-special-courts": ("special court", "special courts", "sebi special court"),
}
_MONTH_ALIAS_MAP: dict[int, tuple[str, ...]] = {
    index: (
        calendar.month_name[index].lower(),
        calendar.month_abbr[index].lower(),
    )
    for index in range(1, 13)
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HARD_QUERY_BUCKET_PREFERENCES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\brti\b|\bappellate authority\b", re.IGNORECASE),
        ("orders-of-aa-under-rti-act",),
    ),
    (
        re.compile(r"\bsettlement\b", re.IGNORECASE),
        ("settlement-orders",),
    ),
)
_SOFT_QUERY_BUCKET_PREFERENCES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\b(?:special court|judgment|sentencing)\b", re.IGNORECASE),
        ("orders-of-special-courts",),
    ),
    (
        re.compile(r"\b(?:sat|tribunal|appeal|court|writ petition|w\.?\s*p\.?)\b", re.IGNORECASE),
        SAT_COURT_BUCKETS,
    ),
)


@dataclass(frozen=True)
class ExactLookupResolution:
    """Direct-resolution outcome for exact-lookup candidates."""

    ordered_candidates: tuple[Any, ...]
    selected_document_id: int | None = None
    should_clarify: bool = False


def looks_like_sat_court_query(
    query: str,
    *,
    sat_court_signals: Sequence[str] = (),
) -> bool:
    """Return whether a query should strongly prefer SAT/court title matching."""

    if sat_court_signals:
        return True
    normalized = " ".join(query.lower().split())
    return any(pattern.search(normalized) for pattern in _SAT_COURT_QUERY_PATTERNS)


def sort_exact_lookup_candidates(
    candidates: Sequence[Any],
    *,
    sat_court_query: bool,
    source_query: str | None = None,
) -> tuple[Any, ...]:
    """Return candidates ordered with SAT/court bucket priors when requested."""

    normalized_query = normalize_alias_text(source_query or "")
    query_tokens = tuple(_TOKEN_RE.findall(normalized_query))
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                _bucket_priority_rank(getattr(item, "bucket_name", None), sat_court_query=sat_court_query),
                -int(_exact_title_lock(source_query=source_query, title=getattr(item, "title", None))),
                -_title_token_overlap_ratio(query_tokens=query_tokens, title=getattr(item, "title", None)),
                -(float(getattr(item, "match_score", 0.0) or 0.0)),
                str(getattr(item, "record_key", "") or ""),
            ),
        )
    )


def select_exact_lookup_resolution(
    candidates: Sequence[Any],
    *,
    sat_court_query: bool,
    source_query: str | None = None,
) -> ExactLookupResolution:
    """Choose either one dominant exact-match candidate or a clarify outcome."""

    ordered = sort_exact_lookup_candidates(
        candidates,
        sat_court_query=sat_court_query,
        source_query=source_query,
    )
    if not ordered:
        return ExactLookupResolution(ordered_candidates=())

    top_score = float(getattr(ordered[0], "match_score", 0.0) or 0.0)
    if top_score < 0.45:
        return ExactLookupResolution(ordered_candidates=ordered)

    second_candidate = ordered[1] if len(ordered) > 1 else None
    second_score = float(getattr(second_candidate, "match_score", 0.0) or 0.0)
    top_bucket_rank = _bucket_priority_rank(
        getattr(ordered[0], "bucket_name", None),
        sat_court_query=sat_court_query,
    )
    second_bucket_rank = _bucket_priority_rank(
        getattr(second_candidate, "bucket_name", None),
        sat_court_query=sat_court_query,
    )
    top_exact_title = _exact_title_lock(source_query=source_query, title=getattr(ordered[0], "title", None))
    second_exact_title = _exact_title_lock(
        source_query=source_query,
        title=getattr(second_candidate, "title", None) if second_candidate is not None else None,
    )

    clearly_dominant = (
        second_candidate is None
        or (
            top_exact_title
            and not second_exact_title
            and top_score >= 0.72
            and (second_candidate is None or top_score - second_score >= 0.05)
        )
        or top_score >= 0.92
        or (top_score >= 0.76 and top_score - second_score >= 0.12)
        or (
            sat_court_query
            and top_bucket_rank < second_bucket_rank
            and top_score >= 0.68
            and top_score - second_score >= 0.02
        )
        or (
            sat_court_query
            and top_bucket_rank == second_bucket_rank
            and top_score >= 0.72
            and top_score - second_score >= 0.08
        )
    )
    if clearly_dominant:
        return ExactLookupResolution(
            ordered_candidates=ordered,
            selected_document_id=int(getattr(ordered[0], "document_version_id")),
        )
    return ExactLookupResolution(
        ordered_candidates=ordered[:5],
        should_clarify=True,
    )


def _exact_title_lock(*, source_query: str | None, title: str | None) -> bool:
    if not source_query or not title:
        return False
    return normalize_alias_text(source_query) == normalize_alias_text(title)


def _title_token_overlap_ratio(*, query_tokens: Sequence[str], title: str | None) -> float:
    if not query_tokens or not title:
        return 0.0
    title_tokens = set(_TOKEN_RE.findall(normalize_alias_text(title)))
    if not title_tokens:
        return 0.0
    overlap = title_tokens & set(query_tokens)
    return len(overlap) / max(len(set(query_tokens)), 1)


def build_matter_clarification_candidates(
    candidates: Sequence[Any],
    *,
    source_query: str,
    control_pack: Any | None = None,
) -> tuple[ClarificationCandidate, ...]:
    """Build structured clarify candidates for ambiguous matter/title matches."""

    from ..schemas import ClarificationCandidate

    documents_by_record_key = getattr(control_pack, "documents_by_record_key", {}) or {}
    relevant_candidates = _filter_candidates_for_query_family(
        candidates,
        source_query=source_query,
    )
    structured: list[ClarificationCandidate] = []
    for index, candidate in enumerate(relevant_candidates[:5], start=1):
        record_key = str(getattr(candidate, "record_key", "") or "").strip()
        if not record_key:
            continue
        title = str(getattr(candidate, "title", "") or "").strip()
        bucket_name = str(getattr(candidate, "bucket_name", "") or "").strip() or None
        order_date = getattr(candidate, "order_date", None)
        document_version_id = getattr(candidate, "document_version_id", None)
        descriptor = None
        document_row = documents_by_record_key.get(record_key)
        if document_row is not None:
            title = title or str(getattr(document_row, "exact_title", "") or "").strip()
            bucket_name = bucket_name or str(getattr(document_row, "bucket_category", "") or "").strip() or None
            order_date = order_date or getattr(document_row, "order_date", None)
            document_version_id = document_version_id or getattr(document_row, "document_version_id", None)
            descriptor = _compact_descriptor(getattr(document_row, "short_summary", None))
        if not title:
            continue
        structured.append(
            ClarificationCandidate(
                candidate_id=record_key,
                candidate_index=index,
                candidate_type="matter",
                title=title,
                record_key=record_key,
                bucket_name=bucket_name,
                order_date=order_date,
                document_version_id=document_version_id,
                descriptor=descriptor,
                selection_aliases=_matter_selection_aliases(
                    title=title,
                    record_key=record_key,
                    bucket_name=bucket_name,
                    order_date=order_date,
                    source_query=source_query,
                ),
            )
        )
    return tuple(structured)


def _filter_candidates_for_query_family(
    candidates: Sequence[Any],
    *,
    source_query: str,
) -> tuple[Any, ...]:
    preferred_buckets, hard_filter = _preferred_buckets_for_query(source_query)
    if not preferred_buckets:
        return tuple(candidates)
    filtered = tuple(
        candidate
        for candidate in candidates
        if str(getattr(candidate, "bucket_name", "") or "").strip() in preferred_buckets
    )
    if filtered:
        return filtered
    return () if hard_filter else tuple(candidates)


def _preferred_buckets_for_query(source_query: str) -> tuple[tuple[str, ...], bool]:
    normalized_query = " ".join(source_query.lower().split())
    for pattern, buckets in _HARD_QUERY_BUCKET_PREFERENCES:
        if pattern.search(normalized_query):
            return buckets, True
    for pattern, buckets in _SOFT_QUERY_BUCKET_PREFERENCES:
        if pattern.search(normalized_query):
            return buckets, False
    return (), False


def build_person_clarification_candidates(
    people: Iterable[dict[str, Any]],
    *,
    source_query: str,
    extracted_person_name: str | None = None,
) -> tuple[ClarificationCandidate, ...]:
    """Build structured clarify candidates for ambiguous people queries."""

    from ..schemas import ClarificationCandidate

    ambiguous_name = extracted_person_name or _extract_single_name(source_query)
    structured: list[ClarificationCandidate] = []
    for index, person in enumerate(people, start=1):
        title = str(person.get("name") or "").strip()
        if not title:
            continue
        designation = str(person.get("designation") or "").strip()
        department_name = str(person.get("department_name") or "").strip()
        office_name = str(person.get("office_name") or "").strip()
        descriptor_parts = [value for value in (designation, department_name, office_name) if value]
        descriptor = ", ".join(descriptor_parts) if descriptor_parts else None
        structured.append(
            ClarificationCandidate(
                candidate_id=str(person.get("canonical_person_id") or f"person:{index}"),
                candidate_index=index,
                candidate_type="person",
                title=title,
                descriptor=descriptor,
                canonical_person_id=(
                    str(person["canonical_person_id"])
                    if person.get("canonical_person_id")
                    else None
                ),
                resolution_query=_rewrite_person_query(
                    source_query=source_query,
                    ambiguous_name=ambiguous_name,
                    selected_name=title,
                ),
                selection_aliases=_person_selection_aliases(
                    title=title,
                    designation=designation,
                    department_name=department_name,
                    office_name=office_name,
                ),
            )
        )
    return tuple(structured)


def render_clarification_candidate_lines(
    candidates: Sequence[ClarificationCandidate],
) -> tuple[str, ...]:
    """Render concise candidate lines for answer text."""

    rendered: list[str] = []
    for candidate in candidates:
        parts = [f"{candidate.candidate_index}. {candidate.title}"]
        if candidate.order_date is not None:
            parts.append(candidate.order_date.isoformat())
        if candidate.bucket_name:
            parts.append(candidate.bucket_name)
        if candidate.record_key:
            parts.append(candidate.record_key)
        if candidate.descriptor:
            parts.append(candidate.descriptor)
        rendered.append(" | ".join(parts))
    return tuple(rendered)


def _bucket_priority_rank(bucket_name: str | None, *, sat_court_query: bool) -> int:
    normalized = str(bucket_name or "").strip().lower()
    if sat_court_query and normalized in SAT_COURT_BUCKETS:
        return SAT_COURT_BUCKETS.index(normalized)
    if sat_court_query:
        return len(SAT_COURT_BUCKETS) + 1
    return 0


def _matter_selection_aliases(
    *,
    title: str,
    record_key: str,
    bucket_name: str | None,
    order_date: date | None,
    source_query: str,
) -> tuple[str, ...]:
    values = {
        _normalize_text(record_key),
        _normalize_text(title),
        *(_normalize_text(value) for value in generate_order_alias_variants(title)),
    }
    if bucket_name:
        values.add(_normalize_text(bucket_name))
        values.update(_normalize_text(alias) for alias in _BUCKET_SELECTION_ALIASES.get(bucket_name, ()))
    if order_date is not None:
        values.add(order_date.isoformat())
        values.add(str(order_date.year))
        for month_alias in _MONTH_ALIAS_MAP.get(order_date.month, ()):
            values.add(f"{month_alias} {order_date.year}")
            values.add(month_alias)
    return tuple(value for value in sorted(values) if value)


def _person_selection_aliases(
    *,
    title: str,
    designation: str,
    department_name: str,
    office_name: str,
) -> tuple[str, ...]:
    values = {
        _normalize_text(title),
        _normalize_text(designation),
        _normalize_text(department_name),
        _normalize_text(office_name),
    }
    if department_name:
        values.update(_department_aliases(department_name))
    return tuple(value for value in sorted(values) if value)


def _department_aliases(department_name: str) -> set[str]:
    tokens = [token.upper() for token in re.findall(r"\b[A-Za-z]{2,}\b", department_name)]
    aliases = {
        _normalize_text(department_name),
    }
    acronym = "".join(token[0] for token in tokens if token and token.lower() not in {"of", "and"})
    if len(acronym) >= 2:
        aliases.add(acronym.lower())
    return aliases


def _rewrite_person_query(
    *,
    source_query: str,
    ambiguous_name: str | None,
    selected_name: str,
) -> str:
    normalized_source = " ".join(source_query.split())
    if ambiguous_name:
        pattern = re.compile(re.escape(ambiguous_name), re.IGNORECASE)
        rewritten = pattern.sub(selected_name, normalized_source)
        if rewritten != normalized_source:
            return rewritten
    lowered = normalized_source.lower()
    if lowered.startswith("who is "):
        return f"who is {selected_name}"
    return selected_name


def _extract_single_name(source_query: str) -> str | None:
    normalized = _normalize_text(source_query)
    if normalized.startswith("who is "):
        tail = normalized[len("who is ") :].strip()
        if tail:
            return tail
    tokens = _TOKEN_RE.findall(normalized)
    if len(tokens) == 1:
        return tokens[0]
    return None


def _compact_descriptor(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) <= 96:
        return text
    return text[:93].rstrip() + "..."


def _normalize_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(normalize_alias_text(value or "")))
