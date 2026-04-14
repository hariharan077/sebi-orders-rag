"""Hierarchical retrieval flow for Phase 3."""

from __future__ import annotations

from typing import Any

from ..config import SebiOrdersRagSettings
from ..control import ControlPack, StrictMatterLock, confusion_penalty_map, dataclass_asdict
from ..embeddings.client import OpenAIEmbeddingClient
from ..repositories.retrieval import HierarchicalRetrievalRepository, merge_filter_scope
from ..schemas import MetadataFilterInput
from .filters import normalize_metadata_filters
from .lexical_search import LexicalSearchService
from .query_intent import detect_query_intent
from .scoring import (
    HierarchicalSearchResult,
    merge_chunk_hits,
    merge_document_hits,
    merge_section_hits,
)
from .vector_search import VectorSearchService


class HierarchicalSearchService:
    """Execute metadata-aware hierarchical lexical + vector retrieval."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        connection: Any,
        embedding_client: OpenAIEmbeddingClient | None = None,
        control_pack: ControlPack | None = None,
    ) -> None:
        repository = HierarchicalRetrievalRepository(connection)
        self._settings = settings
        self._lexical = LexicalSearchService(repository)
        self._vector = VectorSearchService(
            settings=settings,
            repository=repository,
            embedding_client=embedding_client,
        )
        self._control_pack = control_pack

    def search(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None = None,
        top_k_docs: int | None = None,
        top_k_sections: int | None = None,
        top_k_chunks: int | None = None,
        strict_matter_lock: StrictMatterLock | None = None,
    ) -> HierarchicalSearchResult:
        """Run the Phase 3 hierarchical retrieval pipeline."""

        normalized_filters = normalize_metadata_filters(filters)
        normalized_filters = _merge_strict_lock_filter(
            normalized_filters,
            strict_matter_lock=strict_matter_lock,
        )
        document_limit = top_k_docs or self._settings.retrieval_top_k_docs
        section_limit = top_k_sections or self._settings.retrieval_top_k_sections
        chunk_limit = top_k_chunks or self._settings.retrieval_top_k_chunks
        expanded_document_limit = _expanded_limit(document_limit)
        expanded_section_limit = _expanded_limit(section_limit)
        expanded_chunk_limit = _expanded_limit(chunk_limit)
        query_intent = detect_query_intent(query)
        confusion_penalties = confusion_penalty_map(
            control_pack=self._control_pack,
            strict_lock=strict_matter_lock,
        )

        query_embedding = self._vector.embed_query_text(query)
        document_hits = merge_document_hits(
            self._lexical.search_documents(
                query=query,
                filters=normalized_filters,
                limit=expanded_document_limit,
            ),
            self._vector.search_documents(
                query_embedding=query_embedding,
                filters=normalized_filters,
                limit=expanded_document_limit,
            ),
            query_intent=query_intent,
            strict_matter_lock=strict_matter_lock,
            confusion_penalties=confusion_penalties,
        )[:document_limit]

        if not document_hits:
            return HierarchicalSearchResult(
                query=query,
                documents=(),
                sections=(),
                chunks=(),
                query_intent=query_intent,
                debug=_build_search_debug(
                    query_intent,
                    strict_matter_lock=strict_matter_lock,
                    confusion_penalties=confusion_penalties,
                    filters=normalized_filters,
                ),
            )

        document_scores = {
            hit.document_version_id: hit.score.combined_score for hit in document_hits
        }
        section_filters, chunk_filters = derive_hierarchical_filters(
            normalized_filters,
            document_hits=document_hits,
            section_hits=(),
        )
        section_hits = merge_section_hits(
            self._lexical.search_sections(
                query=query,
                filters=section_filters,
                limit=expanded_section_limit,
            ),
            self._vector.search_sections(
                query_embedding=query_embedding,
                filters=section_filters,
                limit=expanded_section_limit,
            ),
            parent_document_scores=document_scores,
            query_intent=query_intent,
            strict_matter_lock=strict_matter_lock,
            confusion_penalties=confusion_penalties,
        )[:section_limit]

        section_scores = {
            (hit.document_version_id, hit.section_key): hit.score.combined_score
            for hit in section_hits
        }
        _, chunk_filters = derive_hierarchical_filters(
            normalized_filters,
            document_hits=document_hits,
            section_hits=section_hits,
        )
        chunk_hits = merge_chunk_hits(
            self._lexical.search_chunks(
                query=query,
                filters=chunk_filters,
                limit=expanded_chunk_limit,
            ),
            self._vector.search_chunks(
                query_embedding=query_embedding,
                filters=chunk_filters,
                limit=expanded_chunk_limit,
            ),
            parent_document_scores=document_scores,
            parent_section_scores=section_scores,
            query_intent=query_intent,
            strict_matter_lock=strict_matter_lock,
            confusion_penalties=confusion_penalties,
        )[:chunk_limit]

        return HierarchicalSearchResult(
            query=query,
            documents=document_hits,
            sections=section_hits,
            chunks=chunk_hits,
            query_intent=query_intent,
            debug=_build_search_debug(
                query_intent,
                strict_matter_lock=strict_matter_lock,
                confusion_penalties=confusion_penalties,
                filters=normalized_filters,
            ),
        )


def _expanded_limit(limit: int) -> int:
    return max(limit, limit * 3)


def derive_hierarchical_filters(
    base_filters: MetadataFilterInput | None,
    *,
    document_hits: tuple[object, ...],
    section_hits: tuple[object, ...],
) -> tuple[MetadataFilterInput, MetadataFilterInput]:
    """Return the section and chunk filter scopes implied by current parent hits."""

    document_ids = [hit.document_version_id for hit in document_hits]
    section_filters = merge_filter_scope(
        base_filters,
        document_version_ids=document_ids,
    )
    chunk_filters = merge_filter_scope(
        base_filters,
        document_version_ids=document_ids,
        section_keys=[hit.section_key for hit in section_hits],
    )
    return section_filters, chunk_filters


def _merge_strict_lock_filter(
    filters: MetadataFilterInput,
    *,
    strict_matter_lock: StrictMatterLock | None,
) -> MetadataFilterInput:
    if (
        strict_matter_lock is None
        or not strict_matter_lock.strict_single_matter
        or not strict_matter_lock.locked_record_keys
    ):
        return filters
    return MetadataFilterInput(
        record_key=strict_matter_lock.locked_record_keys[0],
        bucket_name=filters.bucket_name,
        document_version_ids=(),
        section_keys=(),
        section_types=filters.section_types,
    )


def _build_search_debug(
    query_intent,
    *,
    strict_matter_lock: StrictMatterLock | None,
    confusion_penalties: dict[str, float],
    filters: MetadataFilterInput,
) -> dict[str, object]:
    return {
        "intent": query_intent.intent.value,
        "matched_terms": list(query_intent.matched_terms),
        "settlement_focused": query_intent.settlement_focused,
        "settlement_terms": list(query_intent.settlement_terms),
        "entity_terms": list(query_intent.entity_terms),
        "generic_explanatory_terms": list(query_intent.generic_explanatory_terms),
        "strict_single_matter": bool(strict_matter_lock and strict_matter_lock.strict_single_matter),
        "strict_scope_required": bool(strict_matter_lock and strict_matter_lock.strict_scope_required),
        "comparison_intent_disabled_lock": bool(
            strict_matter_lock
            and strict_matter_lock.comparison_intent
            and not strict_matter_lock.strict_single_matter
        ),
        "locked_candidate_record_keys": list(
            strict_matter_lock.locked_record_keys if strict_matter_lock else ()
        ),
        "matched_aliases": list(strict_matter_lock.matched_aliases if strict_matter_lock else ()),
        "matched_entities": list(strict_matter_lock.matched_entities if strict_matter_lock else ()),
        "strict_lock_reason_codes": list(
            strict_matter_lock.reason_codes if strict_matter_lock else ()
        ),
        "strict_lock_candidates": (
            dataclass_asdict(strict_matter_lock.candidates) if strict_matter_lock else []
        ),
        "confusion_penalties_applied": {
            key: round(value, 4) for key, value in confusion_penalties.items()
        },
        "normalized_filters": dataclass_asdict(filters),
    }
