"""Vector search primitives for Phase 3 hierarchical retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import SebiOrdersRagSettings
from ..embeddings.client import OpenAIEmbeddingClient
from ..repositories.retrieval import HierarchicalRetrievalRepository
from ..schemas import MetadataFilterInput
from .scoring import ChunkSearchHit, DocumentSearchHit, SectionSearchHit


class VectorSearchService:
    """Thin wrapper around query embedding creation and pgvector retrieval."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        repository: HierarchicalRetrievalRepository,
        embedding_client: OpenAIEmbeddingClient | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._embedding_client = embedding_client

    def embed_query_text(self, query: str) -> tuple[float, ...]:
        client = self._get_embedding_client()
        return client.embed_texts(
            [query],
            model=self._settings.embedding_model,
            dimensions=self._settings.embedding_dim,
            batch_size=1,
        )[0]

    def search_documents(
        self,
        *,
        query_embedding: Sequence[float],
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[DocumentSearchHit]:
        return self._repository.search_documents_vector(
            query_embedding=query_embedding,
            filters=filters,
            limit=limit,
        )

    def search_sections(
        self,
        *,
        query_embedding: Sequence[float],
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[SectionSearchHit]:
        return self._repository.search_sections_vector(
            query_embedding=query_embedding,
            filters=filters,
            limit=limit,
        )

    def search_chunks(
        self,
        *,
        query_embedding: Sequence[float],
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[ChunkSearchHit]:
        return self._repository.search_chunks_vector(
            query_embedding=query_embedding,
            filters=filters,
            limit=limit,
        )

    def _get_embedding_client(self) -> OpenAIEmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = OpenAIEmbeddingClient(self._settings)
        return self._embedding_client
