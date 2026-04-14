"""Chunk persistence helpers for extracted SEBI order text."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..schemas import ChunkEmbeddingUpdate, ChunkRecord, SectionGroupInput, StoredChunk
from ..utils.strings import join_heading_path, split_heading_path

_SECTION_KEY_SLUG_RE = re.compile(r"[^a-z0-9]+")


class DocumentChunkRepository:
    """Replace, read, and enrich chunk rows for a document version."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._supports_phase3_columns_cache: bool | None = None

    def replace_chunks(
        self,
        *,
        document_version_id: int,
        chunks: Sequence[ChunkRecord],
    ) -> int:
        """Delete and reinsert all chunks for a document version."""

        prepared_chunks = _prepare_chunk_records_for_persistence(chunks)
        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_version_id = %s",
                (document_version_id,),
            )
            if not prepared_chunks:
                return 0

            if self._supports_phase3_columns():
                cursor.executemany(
                    """
                    INSERT INTO document_chunks (
                        document_version_id,
                        chunk_index,
                        page_start,
                        page_end,
                        section_type,
                        section_title,
                        heading_path,
                        section_key,
                        chunk_text,
                        chunk_sha256,
                        token_count,
                        chunk_metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    [
                        (
                            document_version_id,
                            chunk.chunk_index,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.section_type,
                            chunk.section_title,
                            join_heading_path(chunk.heading_path),
                            chunk.section_key,
                            chunk.chunk_text,
                            chunk.chunk_sha256,
                            chunk.token_count,
                            json.dumps(chunk.chunk_metadata or {}, sort_keys=True),
                        )
                        for chunk in prepared_chunks
                    ],
                )
            else:
                cursor.executemany(
                    """
                    INSERT INTO document_chunks (
                        document_version_id,
                        chunk_index,
                        page_start,
                        page_end,
                        section_type,
                        section_title,
                        heading_path,
                        chunk_text,
                        chunk_sha256,
                        token_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            document_version_id,
                            chunk.chunk_index,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.section_type,
                            chunk.section_title,
                            join_heading_path(chunk.heading_path),
                            chunk.chunk_text,
                            chunk.chunk_sha256,
                            chunk.token_count,
                        )
                        for chunk in prepared_chunks
                    ],
                )
        return len(prepared_chunks)

    def list_chunks_for_document_version(self, document_version_id: int) -> list[StoredChunk]:
        """Return ordered chunks for one document version with document context."""

        grouped = self.list_chunks_grouped_by_document_version([document_version_id])
        return list(grouped.get(document_version_id, ()))

    def list_chunks_grouped_by_document_version(
        self,
        document_version_ids: Sequence[int],
    ) -> dict[int, tuple[StoredChunk, ...]]:
        """Return ordered chunks grouped by document version id."""

        ids = [int(value) for value in document_version_ids]
        if not ids:
            return {}

        sql = """
            SELECT
                dc.chunk_id,
                dc.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                dc.chunk_index,
                dc.page_start,
                dc.page_end,
                dc.section_type,
                dc.section_title,
                dc.heading_path,
                dc.section_key,
                dc.chunk_text,
                dc.chunk_sha256,
                dc.token_count,
                COALESCE(dc.chunk_metadata, '{}'::jsonb),
                dc.embedding_model,
                dc.embedding_created_at
            FROM document_chunks dc
            INNER JOIN document_versions dv
                ON dv.document_version_id = dc.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE dc.document_version_id = ANY(%s)
            ORDER BY dc.document_version_id ASC, dc.chunk_index ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (ids,))
            rows = cursor.fetchall()

        grouped_rows: dict[int, list[StoredChunk]] = defaultdict(list)
        for row in rows:
            chunk = _stored_chunk_from_row(row)
            grouped_rows[chunk.document_version_id].append(chunk)
        return {
            document_version_id: tuple(_normalize_stored_chunks(chunks))
            for document_version_id, chunks in grouped_rows.items()
        }

    def list_section_group_inputs(self, document_version_id: int) -> tuple[SectionGroupInput, ...]:
        """Build logical section inputs from persisted chunks for one document."""

        chunks = self.list_chunks_for_document_version(document_version_id)
        return build_section_group_inputs(chunks)

    def update_chunk_embeddings(
        self,
        *,
        document_version_id: int,
        updates: Sequence[ChunkEmbeddingUpdate],
    ) -> int:
        """Write chunk embeddings and normalized section metadata in place."""

        if not updates:
            return 0

        with self._connection.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE document_chunks
                SET
                    section_key = %s,
                    chunk_metadata = %s::jsonb,
                    embedding = %s::vector,
                    embedding_model = %s,
                    embedding_created_at = NOW()
                WHERE chunk_id = %s
                  AND document_version_id = %s
                """,
                [
                    (
                        update.section_key,
                        json.dumps(update.chunk_metadata, sort_keys=True),
                        _vector_literal(update.embedding),
                        update.embedding_model,
                        update.chunk_id,
                        document_version_id,
                    )
                    for update in updates
                ],
            )
        return len(updates)

    def _supports_phase3_columns(self) -> bool:
        if self._supports_phase3_columns_cache is not None:
            return self._supports_phase3_columns_cache

        sql = """
            SELECT COUNT(*)::INT
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'document_chunks'
              AND column_name IN ('section_key', 'chunk_metadata')
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
        self._supports_phase3_columns_cache = int(row[0] or 0) == 2
        return self._supports_phase3_columns_cache


def build_section_group_inputs(chunks: Sequence[StoredChunk]) -> tuple[SectionGroupInput, ...]:
    """Group normalized chunks into logical section inputs."""

    normalized_chunks = _normalize_stored_chunks(chunks)
    if not normalized_chunks:
        return ()

    sections: list[SectionGroupInput] = []
    current_chunks: list[StoredChunk] = []
    current_section_key: str | None = None

    for chunk in normalized_chunks:
        if current_chunks and chunk.section_key != current_section_key:
            sections.append(_section_group_from_chunks(current_chunks))
            current_chunks = []
        current_chunks.append(chunk)
        current_section_key = chunk.section_key

    if current_chunks:
        sections.append(_section_group_from_chunks(current_chunks))

    return tuple(sections)


def _prepare_chunk_records_for_persistence(chunks: Sequence[ChunkRecord]) -> list[ChunkRecord]:
    prepared: list[ChunkRecord] = []
    current_signature: tuple[str, str | None, str | None] | None = None
    current_section_key: str | None = None
    group_index = 0

    for chunk in chunks:
        heading_path_text = join_heading_path(chunk.heading_path)
        signature = (chunk.section_type, chunk.section_title, heading_path_text)
        if signature != current_signature:
            group_index += 1
            current_signature = signature
            current_section_key = chunk.section_key or _build_section_key(
                group_index=group_index,
                section_type=chunk.section_type,
                section_title=chunk.section_title,
                heading_path=chunk.heading_path,
            )

        section_key = chunk.section_key or current_section_key
        metadata = _build_chunk_metadata(
            existing=chunk.chunk_metadata,
            section_key=section_key,
            section_type=chunk.section_type,
            section_title=chunk.section_title,
            heading_path=chunk.heading_path,
        )
        prepared.append(
            replace(
                chunk,
                section_key=section_key,
                chunk_metadata=metadata,
            )
        )

    return prepared


def _normalize_stored_chunks(chunks: Sequence[StoredChunk]) -> list[StoredChunk]:
    normalized: list[StoredChunk] = []
    current_signature: tuple[str, str | None, str | None] | None = None
    current_section_key: str | None = None
    group_index = 0

    for chunk in sorted(chunks, key=lambda value: value.chunk_index):
        heading_path_parts = _chunk_heading_path_parts(chunk)
        heading_path_text = join_heading_path(heading_path_parts)
        signature = (chunk.section_type, chunk.section_title, heading_path_text)
        if chunk.section_key:
            section_key = chunk.section_key
            if section_key != current_section_key:
                group_index += 1
                current_signature = signature
                current_section_key = section_key
        else:
            if signature != current_signature:
                group_index += 1
                current_signature = signature
                current_section_key = _build_section_key(
                    group_index=group_index,
                    section_type=chunk.section_type,
                    section_title=chunk.section_title,
                    heading_path=heading_path_parts,
                )
            section_key = current_section_key

        metadata = _build_chunk_metadata(
            existing=chunk.chunk_metadata,
            section_key=section_key,
            section_type=chunk.section_type,
            section_title=chunk.section_title,
            heading_path=heading_path_parts,
        )
        normalized.append(
            replace(
                chunk,
                heading_path=heading_path_text,
                section_key=section_key,
                chunk_metadata=metadata,
            )
        )
    return normalized


def _section_group_from_chunks(chunks: Sequence[StoredChunk]) -> SectionGroupInput:
    first = chunks[0]
    return SectionGroupInput(
        document_version_id=first.document_version_id,
        document_id=first.document_id,
        record_key=first.record_key,
        bucket_name=first.bucket_name,
        external_record_id=first.external_record_id,
        order_date=first.order_date,
        title=first.title,
        section_key=first.section_key or "",
        section_type=first.section_type,
        section_title=first.section_title,
        heading_path=first.heading_path,
        page_start=min(chunk.page_start for chunk in chunks),
        page_end=max(chunk.page_end for chunk in chunks),
        chunks=tuple(chunks),
    )


def _chunk_heading_path_parts(chunk: StoredChunk) -> tuple[str, ...]:
    stored_parts = chunk.chunk_metadata.get("heading_path")
    if isinstance(stored_parts, list):
        return tuple(str(part).strip() for part in stored_parts if str(part).strip())
    return split_heading_path(chunk.heading_path)


def _build_chunk_metadata(
    *,
    existing: dict[str, Any] | None,
    section_key: str | None,
    section_type: str,
    section_title: str | None,
    heading_path: Sequence[str],
) -> dict[str, Any]:
    metadata = dict(existing or {})
    metadata["section_key"] = section_key
    metadata["section_type"] = section_type
    metadata["section_title"] = section_title
    metadata["heading_path"] = [part for part in heading_path if part]
    return metadata


def _build_section_key(
    *,
    group_index: int,
    section_type: str,
    section_title: str | None,
    heading_path: Sequence[str],
) -> str:
    label_source = section_title or (heading_path[-1] if heading_path else section_type)
    slug = _slugify(label_source)
    return f"section-{group_index:04d}-{section_type}-{slug}"


def _slugify(value: str) -> str:
    cleaned = _SECTION_KEY_SLUG_RE.sub("-", value.lower()).strip("-")
    return cleaned[:48] or "section"


def _stored_chunk_from_row(row: tuple[Any, ...]) -> StoredChunk:
    chunk_metadata = row[18] if isinstance(row[18], dict) else {}
    return StoredChunk(
        chunk_id=row[0],
        document_version_id=row[1],
        document_id=row[2],
        record_key=row[3],
        bucket_name=row[4],
        external_record_id=row[5],
        order_date=row[6],
        title=row[7],
        chunk_index=row[8],
        page_start=row[9],
        page_end=row[10],
        section_type=row[11],
        section_title=row[12],
        heading_path=row[13],
        section_key=row[14],
        chunk_text=row[15],
        chunk_sha256=row[16],
        token_count=row[17],
        chunk_metadata=chunk_metadata,
        embedding_model=row[19],
        embedding_created_at=row[20],
    )


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"
