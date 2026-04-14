"""Deterministic score combination for hybrid hierarchical retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Mapping

from ..control import StrictMatterLock
from .query_intent import QueryIntent, QueryIntentResult

SECTION_PRIOR_WEIGHTS: Mapping[str, float] = {
    "operative_order": 1.35,
    "directions": 1.30,
    "findings": 1.25,
    "issues": 1.15,
    "reply_or_submissions": 1.05,
    "background": 1.00,
    "facts": 1.00,
    "show_cause_notice": 0.98,
    "other": 0.92,
    "table_block": 0.90,
    "annexure": 0.88,
    "header": 0.70,
}
QUERY_INTENT_SECTION_MULTIPLIERS: Mapping[QueryIntent, Mapping[str, float]] = {
    QueryIntent.SUBSTANTIVE_OUTCOME_QUERY: {
        "operative_order": 1.20,
        "directions": 1.18,
        "findings": 1.15,
        "issues": 1.08,
        "reply_or_submissions": 1.03,
        "background": 0.98,
        "facts": 0.98,
        "show_cause_notice": 0.96,
        "other": 0.92,
        "table_block": 0.92,
        "annexure": 0.90,
        "header": 0.75,
    },
    QueryIntent.PARTY_OR_TITLE_LOOKUP: {
        "operative_order": 0.94,
        "directions": 0.94,
        "findings": 0.95,
        "issues": 0.96,
        "reply_or_submissions": 0.98,
        "background": 1.00,
        "facts": 1.00,
        "show_cause_notice": 0.98,
        "other": 1.05,
        "table_block": 0.95,
        "annexure": 0.94,
        "header": 2.00,
    },
    QueryIntent.REGULATION_OR_TOPIC_LOOKUP: {
        "operative_order": 1.00,
        "directions": 0.98,
        "findings": 1.06,
        "issues": 1.10,
        "reply_or_submissions": 1.02,
        "background": 1.04,
        "facts": 1.04,
        "show_cause_notice": 1.01,
        "other": 1.00,
        "table_block": 0.92,
        "annexure": 0.95,
        "header": 0.94,
    },
    QueryIntent.GENERIC_LOOKUP: {},
}
_NON_SUBSTANTIVE_SECTION_TYPES = frozenset({"header", "other", "table_block", "annexure"})
_STRONG_SUBSTANTIVE_SECTION_TYPES = frozenset({"operative_order", "findings", "directions"})
_HEADER_SUPPRESSION_MAX_SCORE_RATIO = 1.25
_HEADER_SUPPRESSION_TARGET_RATIO = 0.995
_SUBSTANTIVE_COMPETITIVE_RATIO = 0.93
_REPEATED_HEADER_CHUNK_PENALTY = 0.12
_MAX_REPEATED_HEADER_CHUNK_PENALTY = 0.30
_SETTLEMENT_BUCKET_NAME = "settlement-orders"
_SETTLEMENT_BUCKET_PRIOR = 1.12
_TITLE_MATCH_HIGH = 1.18
_TITLE_MATCH_MEDIUM = 1.12
_TITLE_MATCH_LOW = 1.06
_TITLE_MATCH_MISS = 0.96
_SETTLEMENT_BODY_BOOST = 1.10
_SETTLEMENT_HEADER_ONLY_PENALTY = 0.72
_SHORT_SETTLEMENT_CHUNK_WORD_LIMIT = 28
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SETTLEMENT_BODY_TERMS = (
    "settlement amount",
    "settled on the following terms",
    "terms proposed by the applicant",
    "notice of demand",
    "remittance of the aforesaid amount",
    "remitted the said settlement amount",
    "credit of the same",
    "credit of said amount",
    "it is hereby ordered",
    "no enforcement action",
    "proceedings that may be initiated",
    "settlement proceedings",
)
_SETTLEMENT_HEADING_TERMS = (
    "settlement order",
    "terms of settlement",
    "settlement proceedings",
    "settlement application",
)
_HEADER_ONLY_MARKERS = (
    "securities and exchange board of india",
    "settlement order",
    "in respect of",
    "in the matter of",
)
_STRICT_LOCK_MATCH_BOOST = 1.45
_STRICT_LOCK_OTHER_PENALTY = 0.62


@dataclass(frozen=True)
class ScoreBreakdown:
    """Intermediate lexical/vector scores exposed for debugging."""

    combined_score: float = 0.0
    base_score: float = 0.0
    lexical_rank: int | None = None
    lexical_score: float = 0.0
    fts_score: float = 0.0
    trigram_score: float = 0.0
    vector_rank: int | None = None
    vector_score: float = 0.0
    vector_distance: float | None = None
    parent_score: float = 0.0
    bucket_adjustment: float = 1.0
    query_alignment_adjustment: float = 1.0
    section_prior: float = 1.0
    query_intent_adjustment: float = 1.0
    content_adjustment: float = 1.0
    strict_lock_adjustment: float = 1.0
    confusion_penalty: float = 1.0
    diversity_adjustment: float = 0.0
    final_score: float = 0.0


@dataclass(frozen=True)
class DocumentSearchHit:
    """Document-level retrieval hit with explicit score details."""

    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    order_date: date | None
    title: str
    document_node_text: str | None
    score: ScoreBreakdown


@dataclass(frozen=True)
class SectionSearchHit:
    """Section-level retrieval hit with explicit score details."""

    section_node_id: int
    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    order_date: date | None
    title: str
    section_key: str
    section_type: str
    section_title: str | None
    heading_path: str | None
    page_start: int
    page_end: int
    section_node_text: str | None
    score: ScoreBreakdown


@dataclass(frozen=True)
class ChunkSearchHit:
    """Chunk-level retrieval hit with explicit score details."""

    chunk_id: int
    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    order_date: date | None
    title: str
    chunk_index: int
    page_start: int
    page_end: int
    section_key: str | None
    section_type: str
    section_title: str | None
    heading_path: str | None
    detail_url: str | None
    pdf_url: str | None
    chunk_text: str
    token_count: int
    score: ScoreBreakdown


@dataclass(frozen=True)
class HierarchicalSearchResult:
    """Hierarchical search output returned by the Phase 3 CLI and services."""

    query: str
    documents: tuple[DocumentSearchHit, ...]
    sections: tuple[SectionSearchHit, ...]
    chunks: tuple[ChunkSearchHit, ...]
    query_intent: QueryIntentResult = field(
        default_factory=lambda: QueryIntentResult(intent=QueryIntent.GENERIC_LOOKUP)
    )
    debug: Mapping[str, object] = field(default_factory=dict)


def combine_modal_scores(
    *,
    lexical_rank: int | None,
    vector_rank: int | None,
    parent_score: float = 0.0,
    rrf_k: float = 60.0,
) -> float:
    """Combine lexical and vector ranks with a small parent-score carry."""

    lexical_component = reciprocal_rank_score(lexical_rank, weight=1.0, k=rrf_k)
    vector_component = reciprocal_rank_score(vector_rank, weight=1.15, k=rrf_k)
    parent_component = max(parent_score, 0.0) * 0.15
    return lexical_component + vector_component + parent_component


def reciprocal_rank_score(rank: int | None, *, weight: float, k: float) -> float:
    """Return reciprocal-rank-fusion contribution for one modality."""

    if rank is None:
        return 0.0
    return weight / (k + rank)


def resolve_section_prior(section_type: str | None) -> float:
    """Return the deterministic section-type prior weight."""

    normalized_type = (section_type or "other").strip() or "other"
    return SECTION_PRIOR_WEIGHTS.get(normalized_type, SECTION_PRIOR_WEIGHTS["other"])


def resolve_query_intent_adjustment(
    *,
    query_intent: QueryIntent,
    section_type: str | None,
) -> float:
    """Return the intent-aware weight for a section or chunk type."""

    normalized_type = (section_type or "other").strip() or "other"
    adjustments = QUERY_INTENT_SECTION_MULTIPLIERS.get(query_intent, {})
    return adjustments.get(normalized_type, 1.0)


def resolve_bucket_adjustment(
    *,
    bucket_name: str,
    query_intent: QueryIntentResult | QueryIntent,
) -> float:
    resolved_query_intent = _coerce_query_intent_result(query_intent)
    if resolved_query_intent.settlement_focused and bucket_name == _SETTLEMENT_BUCKET_NAME:
        return _SETTLEMENT_BUCKET_PRIOR
    return 1.0


def resolve_query_alignment_adjustment(
    *,
    title: str,
    query_intent: QueryIntentResult | QueryIntent,
) -> float:
    resolved_query_intent = _coerce_query_intent_result(query_intent)
    if not resolved_query_intent.settlement_focused or len(resolved_query_intent.entity_terms) < 2:
        return 1.0

    query_terms = set(resolved_query_intent.entity_terms)
    title_terms = set(_normalized_terms(title))
    if not title_terms:
        return 1.0

    overlap = len(query_terms & title_terms)
    if overlap >= max(2, len(query_terms)):
        return _TITLE_MATCH_HIGH
    if overlap >= 2:
        return _TITLE_MATCH_MEDIUM
    if overlap == 1:
        return _TITLE_MATCH_LOW
    return _TITLE_MATCH_MISS


def resolve_content_adjustment(
    *,
    hit: SectionSearchHit | ChunkSearchHit,
    query_intent: QueryIntentResult | QueryIntent,
) -> float:
    resolved_query_intent = _coerce_query_intent_result(query_intent)
    if not resolved_query_intent.settlement_focused:
        return 1.0
    if isinstance(hit, ChunkSearchHit):
        if _looks_like_header_only_settlement_chunk(hit):
            return _SETTLEMENT_HEADER_ONLY_PENALTY
        if _looks_like_settlement_body_text(hit):
            return _SETTLEMENT_BODY_BOOST
        return 1.0
    if _looks_like_settlement_body_text(hit):
        return 1.04
    return 1.0


def merge_document_hits(
    lexical_hits: list[DocumentSearchHit],
    vector_hits: list[DocumentSearchHit],
    *,
    query_intent: QueryIntentResult = QueryIntentResult(intent=QueryIntent.GENERIC_LOOKUP),
    strict_matter_lock: StrictMatterLock | None = None,
    confusion_penalties: Mapping[str, float] | None = None,
) -> tuple[DocumentSearchHit, ...]:
    """Merge document hits from lexical and vector retrieval."""

    merged: dict[int, DocumentSearchHit] = {}
    for rank, hit in enumerate(lexical_hits, start=1):
        merged[hit.document_version_id] = replace(
            hit,
            score=_refresh_base_score(hit.score, lexical_rank=rank),
        )

    for rank, hit in enumerate(vector_hits, start=1):
        existing = merged.get(hit.document_version_id)
        if existing is None:
            merged[hit.document_version_id] = replace(
                hit,
                score=_refresh_base_score(hit.score, vector_rank=rank),
            )
            continue

        merged[hit.document_version_id] = replace(
            existing,
            score=_refresh_base_score(
                replace(
                    existing.score,
                    vector_score=hit.score.vector_score,
                    vector_distance=hit.score.vector_distance,
                ),
                vector_rank=rank,
            ),
            document_node_text=existing.document_node_text or hit.document_node_text,
        )

    return rerank_document_hits(
        merged.values(),
        query_intent=query_intent,
        strict_matter_lock=strict_matter_lock,
        confusion_penalties=confusion_penalties,
    )


def merge_section_hits(
    lexical_hits: list[SectionSearchHit],
    vector_hits: list[SectionSearchHit],
    *,
    parent_document_scores: Mapping[int, float] | None = None,
    query_intent: QueryIntentResult = QueryIntentResult(intent=QueryIntent.GENERIC_LOOKUP),
    strict_matter_lock: StrictMatterLock | None = None,
    confusion_penalties: Mapping[str, float] | None = None,
) -> tuple[SectionSearchHit, ...]:
    """Merge section hits from lexical and vector retrieval."""

    merged: dict[int, SectionSearchHit] = {}
    parent_document_scores = parent_document_scores or {}

    for rank, hit in enumerate(lexical_hits, start=1):
        parent_score = parent_document_scores.get(hit.document_version_id, 0.0)
        merged[hit.section_node_id] = replace(
            hit,
            score=_refresh_base_score(
                hit.score,
                lexical_rank=rank,
                parent_score=parent_score,
            ),
        )

    for rank, hit in enumerate(vector_hits, start=1):
        parent_score = parent_document_scores.get(hit.document_version_id, 0.0)
        existing = merged.get(hit.section_node_id)
        if existing is None:
            merged[hit.section_node_id] = replace(
                hit,
                score=_refresh_base_score(
                    hit.score,
                    vector_rank=rank,
                    parent_score=parent_score,
                ),
            )
            continue

        merged[hit.section_node_id] = replace(
            existing,
            score=_refresh_base_score(
                replace(
                    existing.score,
                    vector_score=hit.score.vector_score,
                    vector_distance=hit.score.vector_distance,
                ),
                vector_rank=rank,
                parent_score=parent_score,
            ),
            section_node_text=existing.section_node_text or hit.section_node_text,
        )

    return rerank_section_hits(
        merged.values(),
        query_intent=query_intent,
        strict_matter_lock=strict_matter_lock,
        confusion_penalties=confusion_penalties,
    )


def merge_chunk_hits(
    lexical_hits: list[ChunkSearchHit],
    vector_hits: list[ChunkSearchHit],
    *,
    parent_document_scores: Mapping[int, float] | None = None,
    parent_section_scores: Mapping[tuple[int, str], float] | None = None,
    query_intent: QueryIntentResult = QueryIntentResult(intent=QueryIntent.GENERIC_LOOKUP),
    strict_matter_lock: StrictMatterLock | None = None,
    confusion_penalties: Mapping[str, float] | None = None,
) -> tuple[ChunkSearchHit, ...]:
    """Merge chunk hits from lexical and vector retrieval."""

    merged: dict[int, ChunkSearchHit] = {}
    parent_document_scores = parent_document_scores or {}
    parent_section_scores = parent_section_scores or {}

    for rank, hit in enumerate(lexical_hits, start=1):
        parent_score = _resolve_chunk_parent_score(
            hit,
            parent_document_scores=parent_document_scores,
            parent_section_scores=parent_section_scores,
        )
        merged[hit.chunk_id] = replace(
            hit,
            score=_refresh_base_score(
                hit.score,
                lexical_rank=rank,
                parent_score=parent_score,
            ),
        )

    for rank, hit in enumerate(vector_hits, start=1):
        parent_score = _resolve_chunk_parent_score(
            hit,
            parent_document_scores=parent_document_scores,
            parent_section_scores=parent_section_scores,
        )
        existing = merged.get(hit.chunk_id)
        if existing is None:
            merged[hit.chunk_id] = replace(
                hit,
                score=_refresh_base_score(
                    hit.score,
                    vector_rank=rank,
                    parent_score=parent_score,
                ),
            )
            continue

        merged[hit.chunk_id] = replace(
            existing,
            score=_refresh_base_score(
                replace(
                    existing.score,
                    vector_score=hit.score.vector_score,
                    vector_distance=hit.score.vector_distance,
                ),
                vector_rank=rank,
                parent_score=parent_score,
            ),
        )

    return rerank_chunk_hits(
        merged.values(),
        query_intent=query_intent,
        strict_matter_lock=strict_matter_lock,
        confusion_penalties=confusion_penalties,
    )


def rerank_document_hits(
    hits: Iterable[DocumentSearchHit],
    *,
    query_intent: QueryIntentResult,
    strict_matter_lock: StrictMatterLock | None = None,
    confusion_penalties: Mapping[str, float] | None = None,
) -> tuple[DocumentSearchHit, ...]:
    """Apply deterministic query-aware document priors."""

    resolved_confusion_penalties = confusion_penalties or {}
    adjusted_hits = [
        _apply_document_adjustments(
            hit,
            query_intent=query_intent,
            strict_matter_lock=strict_matter_lock,
            confusion_penalties=resolved_confusion_penalties,
        )
        for hit in hits
    ]
    return tuple(sorted(adjusted_hits, key=_document_sort_key))


def rerank_section_hits(
    hits: Iterable[SectionSearchHit],
    *,
    query_intent: QueryIntentResult,
    strict_matter_lock: StrictMatterLock | None = None,
    confusion_penalties: Mapping[str, float] | None = None,
) -> tuple[SectionSearchHit, ...]:
    """Apply Phase 3.1 section priors and intent-aware reranking."""

    resolved_confusion_penalties = confusion_penalties or {}
    adjusted_hits = [
        _apply_structural_adjustments(
            hit,
            query_intent=query_intent,
            strict_matter_lock=strict_matter_lock,
            confusion_penalties=resolved_confusion_penalties,
        )
        for hit in hits
    ]
    adjusted_hits = _apply_header_suppression(
        adjusted_hits,
        query_intent=query_intent,
        key_resolver=lambda item: item.section_node_id,
    )
    return tuple(sorted(adjusted_hits, key=_section_sort_key))


def rerank_chunk_hits(
    hits: Iterable[ChunkSearchHit],
    *,
    query_intent: QueryIntentResult,
    strict_matter_lock: StrictMatterLock | None = None,
    confusion_penalties: Mapping[str, float] | None = None,
) -> tuple[ChunkSearchHit, ...]:
    """Apply Phase 3.1 chunk priors, suppression, and light diversity."""

    resolved_confusion_penalties = confusion_penalties or {}
    adjusted_hits = [
        _apply_structural_adjustments(
            hit,
            query_intent=query_intent,
            strict_matter_lock=strict_matter_lock,
            confusion_penalties=resolved_confusion_penalties,
        )
        for hit in hits
    ]
    adjusted_hits = _apply_header_suppression(
        adjusted_hits,
        query_intent=query_intent,
        key_resolver=lambda item: item.chunk_id,
    )
    adjusted_hits = _apply_chunk_diversity(adjusted_hits, query_intent=query_intent)
    return tuple(sorted(adjusted_hits, key=_chunk_sort_key))


def _refresh_base_score(
    score: ScoreBreakdown,
    *,
    lexical_rank: int | None | object = ...,
    vector_rank: int | None | object = ...,
    parent_score: float | object = ...,
) -> ScoreBreakdown:
    resolved_lexical_rank = score.lexical_rank if lexical_rank is ... else lexical_rank
    resolved_vector_rank = score.vector_rank if vector_rank is ... else vector_rank
    resolved_parent_score = score.parent_score if parent_score is ... else float(parent_score)
    base_score = combine_modal_scores(
        lexical_rank=resolved_lexical_rank,
        vector_rank=resolved_vector_rank,
        parent_score=resolved_parent_score,
    )
    return replace(
        score,
        lexical_rank=resolved_lexical_rank,
        vector_rank=resolved_vector_rank,
        parent_score=resolved_parent_score,
        base_score=base_score,
        bucket_adjustment=1.0,
        query_alignment_adjustment=1.0,
        section_prior=1.0,
        query_intent_adjustment=1.0,
        content_adjustment=1.0,
        strict_lock_adjustment=1.0,
        confusion_penalty=1.0,
        diversity_adjustment=0.0,
        final_score=base_score,
        combined_score=base_score,
    )


def _apply_document_adjustments(
    hit: DocumentSearchHit,
    *,
    query_intent: QueryIntentResult,
    strict_matter_lock: StrictMatterLock | None,
    confusion_penalties: Mapping[str, float],
) -> DocumentSearchHit:
    base_score = _resolve_base_score(hit.score)
    bucket_adjustment = resolve_bucket_adjustment(
        bucket_name=hit.bucket_name,
        query_intent=query_intent,
    )
    query_alignment_adjustment = resolve_query_alignment_adjustment(
        title=hit.title,
        query_intent=query_intent,
    )
    strict_lock_adjustment, confusion_penalty = _resolve_lock_adjustments(
        record_key=hit.record_key,
        strict_matter_lock=strict_matter_lock,
        confusion_penalties=confusion_penalties,
    )
    final_score = max(
        base_score
        * bucket_adjustment
        * query_alignment_adjustment
        * strict_lock_adjustment
        * confusion_penalty,
        0.0,
    )
    return replace(
        hit,
        score=replace(
            hit.score,
            base_score=base_score,
            bucket_adjustment=bucket_adjustment,
            query_alignment_adjustment=query_alignment_adjustment,
            section_prior=1.0,
            query_intent_adjustment=1.0,
            content_adjustment=1.0,
            strict_lock_adjustment=strict_lock_adjustment,
            confusion_penalty=confusion_penalty,
            diversity_adjustment=0.0,
            final_score=final_score,
            combined_score=final_score,
        ),
    )


def _apply_structural_adjustments(
    hit: SectionSearchHit | ChunkSearchHit,
    *,
    query_intent: QueryIntentResult,
    strict_matter_lock: StrictMatterLock | None,
    confusion_penalties: Mapping[str, float],
) -> SectionSearchHit | ChunkSearchHit:
    base_score = _resolve_base_score(hit.score)
    resolved_intent = _resolve_query_intent_enum(query_intent)
    bucket_adjustment = resolve_bucket_adjustment(
        bucket_name=hit.bucket_name,
        query_intent=query_intent,
    )
    query_alignment_adjustment = resolve_query_alignment_adjustment(
        title=hit.title,
        query_intent=query_intent,
    )
    section_prior = resolve_section_prior(hit.section_type)
    query_adjustment = resolve_query_intent_adjustment(
        query_intent=resolved_intent,
        section_type=hit.section_type,
    )
    content_adjustment = resolve_content_adjustment(
        hit=hit,
        query_intent=query_intent,
    )
    strict_lock_adjustment, confusion_penalty = _resolve_lock_adjustments(
        record_key=hit.record_key,
        strict_matter_lock=strict_matter_lock,
        confusion_penalties=confusion_penalties,
    )
    final_score = max(
        base_score
        * bucket_adjustment
        * query_alignment_adjustment
        * section_prior
        * query_adjustment
        * content_adjustment,
        0.0,
    )
    final_score = max(
        final_score
        * strict_lock_adjustment
        * confusion_penalty,
        0.0,
    )
    return replace(
        hit,
        score=replace(
            hit.score,
            base_score=base_score,
            bucket_adjustment=bucket_adjustment,
            query_alignment_adjustment=query_alignment_adjustment,
            section_prior=section_prior,
            query_intent_adjustment=query_adjustment,
            content_adjustment=content_adjustment,
            strict_lock_adjustment=strict_lock_adjustment,
            confusion_penalty=confusion_penalty,
            diversity_adjustment=0.0,
            final_score=final_score,
            combined_score=final_score,
        ),
    )


def _apply_header_suppression(
    hits: list[SectionSearchHit] | list[ChunkSearchHit],
    *,
    query_intent: QueryIntentResult,
    key_resolver,
) -> list[SectionSearchHit] | list[ChunkSearchHit]:
    if _resolve_query_intent_enum(query_intent) is not QueryIntent.SUBSTANTIVE_OUTCOME_QUERY:
        return hits

    grouped_hits: dict[int, list[SectionSearchHit | ChunkSearchHit]] = defaultdict(list)
    for hit in hits:
        grouped_hits[hit.document_version_id].append(hit)

    updated_hits = {key_resolver(hit): hit for hit in hits}
    for document_hits in grouped_hits.values():
        header_hits = [hit for hit in document_hits if hit.section_type == "header"]
        substantive_hits = [
            hit for hit in document_hits if hit.section_type in _STRONG_SUBSTANTIVE_SECTION_TYPES
        ]
        if not header_hits or not substantive_hits:
            continue

        top_substantive = max(substantive_hits, key=lambda item: _score_value(item.score))
        top_substantive_score = _score_value(top_substantive.score)
        if top_substantive_score <= 0.0:
            continue

        for header_hit in header_hits:
            header_score = _score_value(updated_hits[key_resolver(header_hit)].score)
            if header_score <= top_substantive_score:
                continue
            if header_score / top_substantive_score > _HEADER_SUPPRESSION_MAX_SCORE_RATIO:
                continue
            suppression_scale = (
                top_substantive_score * _HEADER_SUPPRESSION_TARGET_RATIO / header_score
            )
            updated_hits[key_resolver(header_hit)] = _scale_query_intent_adjustment(
                updated_hits[key_resolver(header_hit)],
                scale=suppression_scale,
            )

    return list(updated_hits.values())


def _apply_chunk_diversity(
    hits: list[ChunkSearchHit],
    *,
    query_intent: QueryIntentResult,
) -> list[ChunkSearchHit]:
    grouped_hits: dict[int, list[ChunkSearchHit]] = defaultdict(list)
    for hit in hits:
        grouped_hits[hit.document_version_id].append(hit)

    updated_hits = {hit.chunk_id: hit for hit in hits}
    for document_hits in grouped_hits.values():
        ordered_hits = sorted(document_hits, key=_chunk_sort_key)
        header_hits = [updated_hits[hit.chunk_id] for hit in ordered_hits if hit.section_type == "header"]
        substantive_hits = [
            updated_hits[hit.chunk_id] for hit in ordered_hits if _is_substantive_section(hit.section_type)
        ]

        if (
            _resolve_query_intent_enum(query_intent) is QueryIntent.SUBSTANTIVE_OUTCOME_QUERY
            and header_hits
            and substantive_hits
        ):
            top_header = header_hits[0]
            top_substantive = substantive_hits[0]
            header_score = _score_value(top_header.score)
            substantive_score = _score_value(top_substantive.score)
            if (
                substantive_score > 0.0
                and substantive_score < header_score
                and substantive_score >= header_score * _SUBSTANTIVE_COMPETITIVE_RATIO
            ):
                bonus = (header_score - substantive_score) + max(substantive_score * 0.01, 1e-6)
                updated_hits[top_substantive.chunk_id] = _apply_diversity_adjustment(
                    updated_hits[top_substantive.chunk_id],
                    delta=min(bonus, header_score * 0.08),
                )

        for header_index, header_hit in enumerate(header_hits[1:], start=1):
            penalty_ratio = min(
                _REPEATED_HEADER_CHUNK_PENALTY * header_index,
                _MAX_REPEATED_HEADER_CHUNK_PENALTY,
            )
            current_hit = updated_hits[header_hit.chunk_id]
            penalty = _score_value(current_hit.score) * penalty_ratio
            updated_hits[header_hit.chunk_id] = _apply_diversity_adjustment(
                current_hit,
                delta=-penalty,
            )

    return list(updated_hits.values())


def _scale_query_intent_adjustment(
    hit: SectionSearchHit | ChunkSearchHit,
    *,
    scale: float,
) -> SectionSearchHit | ChunkSearchHit:
    updated_adjustment = hit.score.query_intent_adjustment * max(scale, 0.0)
    return _replace_final_score(
        hit,
        query_intent_adjustment=updated_adjustment,
    )


def _apply_diversity_adjustment(
    hit: ChunkSearchHit,
    *,
    delta: float,
) -> ChunkSearchHit:
    updated_diversity_adjustment = hit.score.diversity_adjustment + delta
    return _replace_final_score(
        hit,
        diversity_adjustment=updated_diversity_adjustment,
    )


def _replace_final_score(
    hit: SectionSearchHit | ChunkSearchHit,
    *,
    query_intent_adjustment: float | object = ...,
    diversity_adjustment: float | object = ...,
) -> SectionSearchHit | ChunkSearchHit:
    resolved_query_adjustment = (
        hit.score.query_intent_adjustment
        if query_intent_adjustment is ...
        else float(query_intent_adjustment)
    )
    resolved_diversity_adjustment = (
        hit.score.diversity_adjustment
        if diversity_adjustment is ...
        else float(diversity_adjustment)
    )
    base_score = _resolve_base_score(hit.score)
    final_score = max(
        (
            base_score
            * hit.score.bucket_adjustment
            * hit.score.query_alignment_adjustment
            * hit.score.section_prior
            * resolved_query_adjustment
            * hit.score.content_adjustment
            * hit.score.strict_lock_adjustment
            * hit.score.confusion_penalty
        )
        + resolved_diversity_adjustment,
        0.0,
    )
    return replace(
        hit,
        score=replace(
            hit.score,
            base_score=base_score,
            query_intent_adjustment=resolved_query_adjustment,
            diversity_adjustment=resolved_diversity_adjustment,
            final_score=final_score,
            combined_score=final_score,
        ),
    )


def _resolve_query_intent_enum(query_intent: QueryIntentResult | QueryIntent) -> QueryIntent:
    if isinstance(query_intent, QueryIntentResult):
        return query_intent.intent
    return query_intent


def _coerce_query_intent_result(
    query_intent: QueryIntentResult | QueryIntent,
) -> QueryIntentResult:
    if isinstance(query_intent, QueryIntentResult):
        return query_intent
    return QueryIntentResult(intent=query_intent)


def _resolve_lock_adjustments(
    *,
    record_key: str,
    strict_matter_lock: StrictMatterLock | None,
    confusion_penalties: Mapping[str, float],
) -> tuple[float, float]:
    if (
        strict_matter_lock is None
        or not strict_matter_lock.strict_single_matter
        or not strict_matter_lock.locked_record_keys
    ):
        return 1.0, 1.0
    locked_set = set(strict_matter_lock.locked_record_keys)
    strict_lock_adjustment = (
        _STRICT_LOCK_MATCH_BOOST if record_key in locked_set else _STRICT_LOCK_OTHER_PENALTY
    )
    confusion_penalty = float(confusion_penalties.get(record_key, 1.0))
    return strict_lock_adjustment, confusion_penalty


def _normalized_terms(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(_TOKEN_RE.findall(text.lower()))


def _looks_like_header_only_settlement_chunk(hit: ChunkSearchHit) -> bool:
    normalized_text = _normalized_text(hit.chunk_text)
    if not normalized_text:
        return False
    word_count = len(normalized_text.split())
    marker_hits = sum(1 for marker in _HEADER_ONLY_MARKERS if marker in normalized_text)
    has_sentence_punctuation = ". " in normalized_text or ";" in normalized_text or ": " in normalized_text
    return (
        hit.section_type in {"header", "operative_order"}
        and word_count <= _SHORT_SETTLEMENT_CHUNK_WORD_LIMIT
        and marker_hits >= 2
        and not has_sentence_punctuation
    )


def _looks_like_settlement_body_text(hit: SectionSearchHit | ChunkSearchHit) -> bool:
    text = _normalized_text(
        hit.chunk_text if isinstance(hit, ChunkSearchHit) else hit.section_node_text
    )
    heading_text = _normalized_text(
        " ".join(part for part in (hit.section_title, hit.heading_path) if part)
    )
    if not text:
        return False
    word_count = len(text.split())
    if word_count < 30:
        return False
    if any(term in text for term in _SETTLEMENT_BODY_TERMS):
        return True
    if any(term in heading_text for term in _SETTLEMENT_HEADING_TERMS) and (
        "it is hereby ordered" in text
        or "settlement amount" in text
        or "notice of demand" in text
        or "remitted" in text
    ):
        return True
    return False


def _normalized_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def _resolve_chunk_parent_score(
    hit: ChunkSearchHit,
    *,
    parent_document_scores: Mapping[int, float],
    parent_section_scores: Mapping[tuple[int, str], float],
) -> float:
    section_score = 0.0
    if hit.section_key:
        section_score = parent_section_scores.get(
            (hit.document_version_id, hit.section_key),
            0.0,
        )
    document_score = parent_document_scores.get(hit.document_version_id, 0.0)
    return max(section_score, document_score)


def _resolve_base_score(score: ScoreBreakdown) -> float:
    if score.base_score > 0.0:
        return score.base_score
    if score.final_score > 0.0:
        return score.final_score
    return score.combined_score


def _score_value(score: ScoreBreakdown) -> float:
    if score.final_score > 0.0:
        return score.final_score
    return score.combined_score


def _is_substantive_section(section_type: str | None) -> bool:
    normalized_type = (section_type or "other").strip() or "other"
    return normalized_type not in _NON_SUBSTANTIVE_SECTION_TYPES


def _document_sort_key(hit: DocumentSearchHit) -> tuple[float, float, float, int]:
    return (
        -_score_value(hit.score),
        -hit.score.lexical_score,
        -hit.score.vector_score,
        hit.document_version_id,
    )


def _section_sort_key(hit: SectionSearchHit) -> tuple[float, float, float, int, int]:
    return (
        -_score_value(hit.score),
        -hit.score.lexical_score,
        -hit.score.vector_score,
        hit.document_version_id,
        hit.section_node_id,
    )


def _chunk_sort_key(hit: ChunkSearchHit) -> tuple[float, float, float, int, int]:
    return (
        -_score_value(hit.score),
        -hit.score.lexical_score,
        -hit.score.vector_score,
        hit.document_version_id,
        hit.chunk_id,
    )
