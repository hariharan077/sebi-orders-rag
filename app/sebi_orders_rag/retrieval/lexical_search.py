"""Lexical search primitives for Phase 3 hierarchical retrieval."""

from __future__ import annotations

from ..repositories.retrieval import HierarchicalRetrievalRepository
from ..schemas import MetadataFilterInput
from .scoring import ChunkSearchHit, DocumentSearchHit, SectionSearchHit


class LexicalSearchService:
    """Thin wrapper around repository-backed lexical retrieval primitives."""

    def __init__(self, repository: HierarchicalRetrievalRepository) -> None:
        self._repository = repository

    def search_documents(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[DocumentSearchHit]:
        return self._repository.search_documents_lexical(
            query=query,
            filters=filters,
            limit=limit,
        )

    def search_sections(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[SectionSearchHit]:
        return self._repository.search_sections_lexical(
            query=query,
            filters=filters,
            limit=limit,
        )

    def search_chunks(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[ChunkSearchHit]:
        return self._repository.search_chunks_lexical(
            query=query,
            filters=filters,
            limit=limit,
        )
