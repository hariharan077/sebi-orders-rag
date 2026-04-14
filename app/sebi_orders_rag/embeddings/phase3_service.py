"""Application service for Phase 3 hierarchical embedding generation."""

from __future__ import annotations

import logging
from typing import Any

from ..config import SebiOrdersRagSettings
from ..repositories.chunks import DocumentChunkRepository
from ..repositories.documents import DocumentRepository
from ..repositories.retrieval import HierarchicalRetrievalRepository
from ..schemas import ChunkEmbeddingUpdate, DocumentNodeUpsert, Phase3EmbeddingSummary, SectionNodeUpsert
from .client import OpenAIEmbeddingClient
from .payload_builder import (
    build_chunk_embedding_text,
    build_document_node_payload,
    build_section_node_payload,
)

LOGGER = logging.getLogger(__name__)


class Phase3EmbeddingService:
    """Coordinate deterministic hierarchical node generation and embeddings."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        connection: Any,
        embedding_client: OpenAIEmbeddingClient | None = None,
    ) -> None:
        self._settings = settings
        self._connection = connection
        self._documents = DocumentRepository(connection)
        self._chunks = DocumentChunkRepository(connection)
        self._retrieval = HierarchicalRetrievalRepository(connection)
        self._embedding_client = embedding_client

    def run(
        self,
        *,
        apply: bool,
        record_key: str | None = None,
        bucket_name: str | None = None,
        document_version_id: int | None = None,
        limit: int | None = None,
    ) -> Phase3EmbeddingSummary:
        """Run Phase 3 hierarchical embeddings for selected document versions."""

        summary = Phase3EmbeddingSummary()
        candidates = self._documents.list_embedding_candidates(
            record_key=record_key,
            bucket_name=bucket_name,
            document_version_id=document_version_id,
            limit=None,
        )

        for candidate in candidates:
            if limit is not None and summary.documents_selected >= limit:
                break

            summary.documents_selected += 1
            if not apply:
                LOGGER.info(
                    "[dry-run] Would embed document_version_id=%s record_key=%s title=%s",
                    candidate.document_version_id,
                    candidate.record_key,
                    candidate.title,
                )
                continue

            LOGGER.info(
                "Embedding document_version_id=%s record_key=%s",
                candidate.document_version_id,
                candidate.record_key,
            )
            try:
                self._documents.mark_embedding_processing(
                    document_version_id=candidate.document_version_id
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

            try:
                embedded_counts = self._embed_document(candidate)
                self._documents.mark_embedding_done(
                    document_version_id=candidate.document_version_id,
                    embedding_model=self._settings.embedding_model,
                    embedding_dim=self._settings.embedding_dim,
                )
                self._connection.commit()
            except Exception as exc:
                self._connection.rollback()
                LOGGER.exception(
                    "Phase 3 embedding failed for document_version_id=%s",
                    candidate.document_version_id,
                )
                self._documents.mark_embedding_failed(
                    document_version_id=candidate.document_version_id,
                    embedding_error=f"{type(exc).__name__}: {exc}",
                )
                self._connection.commit()
                summary.documents_failed += 1
                continue

            summary.documents_embedded += 1
            summary.document_nodes_written += embedded_counts[0]
            summary.section_nodes_written += embedded_counts[1]
            summary.chunk_embeddings_updated += embedded_counts[2]

        return summary

    def _embed_document(self, candidate: Any) -> tuple[int, int, int]:
        chunks = self._chunks.list_chunks_for_document_version(candidate.document_version_id)
        if not chunks:
            raise ValueError(
                "Cannot build hierarchical embeddings because no chunks were found for "
                f"document_version_id={candidate.document_version_id}"
            )
        sections = self._chunks.list_section_group_inputs(candidate.document_version_id)

        document_payload = build_document_node_payload(
            candidate,
            sections=sections,
            chunks=chunks,
            model_name=self._settings.embedding_model,
        )
        section_payloads = [
            build_section_node_payload(
                candidate,
                section,
                model_name=self._settings.embedding_model,
            )
            for section in sections
        ]
        chunk_texts = [build_chunk_embedding_text(chunk) for chunk in chunks]

        client = self._get_embedding_client()
        document_embedding = client.embed_texts(
            [document_payload.node_text],
            model=self._settings.embedding_model,
            dimensions=self._settings.embedding_dim,
            batch_size=self._settings.embed_batch_size,
        )[0]
        section_embeddings = client.embed_texts(
            [payload.node_text for payload in section_payloads],
            model=self._settings.embedding_model,
            dimensions=self._settings.embedding_dim,
            batch_size=self._settings.embed_batch_size,
        )
        chunk_embeddings = client.embed_texts(
            chunk_texts,
            model=self._settings.embedding_model,
            dimensions=self._settings.embedding_dim,
            batch_size=self._settings.embed_batch_size,
        )

        self._retrieval.replace_document_node(
            DocumentNodeUpsert(
                document_version_id=candidate.document_version_id,
                node_text=document_payload.node_text,
                token_count=document_payload.token_count,
                embedding=document_embedding,
                embedding_model=self._settings.embedding_model,
                metadata=document_payload.metadata,
            )
        )
        section_nodes_written = self._retrieval.replace_section_nodes(
            document_version_id=candidate.document_version_id,
            nodes=[
                SectionNodeUpsert(
                    document_version_id=section.document_version_id,
                    section_key=section.section_key,
                    section_type=section.section_type,
                    section_title=section.section_title,
                    heading_path=section.heading_path,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    node_text=payload.node_text,
                    token_count=payload.token_count,
                    embedding=embedding,
                    embedding_model=self._settings.embedding_model,
                    metadata=payload.metadata,
                )
                for section, payload, embedding in zip(
                    sections,
                    section_payloads,
                    section_embeddings,
                )
            ],
        )
        chunk_embeddings_updated = self._chunks.update_chunk_embeddings(
            document_version_id=candidate.document_version_id,
            updates=[
                ChunkEmbeddingUpdate(
                    chunk_id=chunk.chunk_id,
                    section_key=chunk.section_key or "",
                    chunk_metadata=chunk.chunk_metadata,
                    embedding=embedding,
                    embedding_model=self._settings.embedding_model,
                )
                for chunk, embedding in zip(chunks, chunk_embeddings)
            ],
        )
        return 1, section_nodes_written, chunk_embeddings_updated

    def _get_embedding_client(self) -> OpenAIEmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = OpenAIEmbeddingClient(self._settings)
        return self._embedding_client
