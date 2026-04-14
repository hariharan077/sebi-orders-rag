"""Read-only repository helpers for SEBI Orders chunk QA and inspection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ProcessedDocumentVersionRow:
    """Processed document metadata used by the chunk QA workflow."""

    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    order_date: date | None
    title: str
    page_count: int
    chunk_count: int
    ingested_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class ChunkRow:
    """Materialized chunk row used for read-only QA inspection."""

    chunk_index: int
    page_start: int
    page_end: int
    section_type: str
    section_title: str | None
    heading_path: tuple[str, ...]
    chunk_text: str
    chunk_sha256: str
    token_count: int


@dataclass(frozen=True)
class CorpusAggregateRow:
    """High-level aggregate statistics for a processed document scope."""

    processed_document_versions: int
    total_chunks: int
    avg_chunks_per_document: float | None
    median_chunks_per_document: float | None
    avg_tokens_per_chunk: float | None
    median_tokens_per_chunk: float | None
    min_tokens_per_chunk: int | None
    max_tokens_per_chunk: int | None


@dataclass(frozen=True)
class BucketAggregateRow:
    """Per-bucket aggregate statistics for a processed document scope."""

    bucket_name: str
    processed_documents: int
    total_chunks: int
    avg_chunks_per_document: float | None
    avg_tokens_per_chunk: float | None


class ChunkQaRepository:
    """Explicit read-only queries for chunk QA reporting."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def list_processed_document_versions(
        self,
        *,
        document_version_ids: Sequence[int] | None = None,
        record_keys: Sequence[str] | None = None,
        bucket_name: str | None = None,
        limit: int | None = None,
    ) -> list[ProcessedDocumentVersionRow]:
        """Fetch processed document versions with stable metadata."""

        if document_version_ids is not None and not document_version_ids:
            return []
        if record_keys is not None and not record_keys:
            return []

        where_sql, params = self._build_processed_scope_filters(
            document_version_ids=document_version_ids,
            record_keys=record_keys,
            bucket_name=bucket_name,
        )
        sql = f"""
            WITH chunk_counts AS (
                SELECT
                    document_version_id,
                    COUNT(*)::INT AS chunk_count
                FROM document_chunks
                GROUP BY document_version_id
            )
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                dv.order_date,
                dv.title,
                COALESCE(dv.page_count, 0) AS page_count,
                COALESCE(dv.chunk_count, chunk_counts.chunk_count, 0) AS chunk_count,
                dv.ingested_at,
                dv.created_at
            FROM document_versions dv
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            LEFT JOIN chunk_counts
                ON chunk_counts.document_version_id = dv.document_version_id
            WHERE {where_sql}
            ORDER BY dv.document_version_id ASC
        """
        if limit is not None:
            sql += "\n LIMIT %s"
            params.append(limit)

        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [_processed_document_from_row(row) for row in rows]

    def list_chunks_for_document_version(self, *, document_version_id: int) -> list[ChunkRow]:
        """Fetch ordered chunks for one processed document version."""

        sql = """
            SELECT
                chunk_index,
                page_start,
                page_end,
                section_type,
                section_title,
                heading_path,
                chunk_text,
                chunk_sha256,
                token_count
            FROM document_chunks
            WHERE document_version_id = %s
            ORDER BY chunk_index ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (document_version_id,))
            rows = cursor.fetchall()
        return [_chunk_from_row(row) for row in rows]

    def get_corpus_aggregates(
        self,
        *,
        document_version_ids: Sequence[int] | None = None,
        record_keys: Sequence[str] | None = None,
        bucket_name: str | None = None,
    ) -> CorpusAggregateRow:
        """Fetch overall chunk QA aggregates for a processed document scope."""

        if document_version_ids is not None and not document_version_ids:
            return _empty_corpus_aggregate()
        if record_keys is not None and not record_keys:
            return _empty_corpus_aggregate()

        where_sql, params = self._build_processed_scope_filters(
            document_version_ids=document_version_ids,
            record_keys=record_keys,
            bucket_name=bucket_name,
        )
        sql = f"""
            WITH chunk_counts AS (
                SELECT
                    document_version_id,
                    COUNT(*)::INT AS chunk_count
                FROM document_chunks
                GROUP BY document_version_id
            ),
            selected_docs AS (
                SELECT
                    dv.document_version_id,
                    COALESCE(dv.chunk_count, chunk_counts.chunk_count, 0) AS chunk_count
                FROM document_versions dv
                INNER JOIN source_documents sd
                    ON sd.document_id = dv.document_id
                LEFT JOIN chunk_counts
                    ON chunk_counts.document_version_id = dv.document_version_id
                WHERE {where_sql}
            ),
            selected_chunks AS (
                SELECT dc.token_count
                FROM document_chunks dc
                INNER JOIN selected_docs
                    ON selected_docs.document_version_id = dc.document_version_id
            )
            SELECT
                (SELECT COUNT(*)::INT FROM selected_docs) AS processed_document_versions,
                COALESCE((SELECT SUM(chunk_count)::INT FROM selected_docs), 0) AS total_chunks,
                (SELECT AVG(chunk_count::DOUBLE PRECISION) FROM selected_docs) AS avg_chunks_per_document,
                (
                    SELECT PERCENTILE_CONT(0.5)
                    WITHIN GROUP (ORDER BY chunk_count)
                    FROM selected_docs
                ) AS median_chunks_per_document,
                (SELECT AVG(token_count::DOUBLE PRECISION) FROM selected_chunks) AS avg_tokens_per_chunk,
                (
                    SELECT PERCENTILE_CONT(0.5)
                    WITHIN GROUP (ORDER BY token_count)
                    FROM selected_chunks
                ) AS median_tokens_per_chunk,
                (SELECT MIN(token_count) FROM selected_chunks) AS min_tokens_per_chunk,
                (SELECT MAX(token_count) FROM selected_chunks) AS max_tokens_per_chunk
        """

        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
        return _corpus_aggregate_from_row(row)

    def get_per_bucket_aggregates(
        self,
        *,
        document_version_ids: Sequence[int] | None = None,
        record_keys: Sequence[str] | None = None,
        bucket_name: str | None = None,
    ) -> list[BucketAggregateRow]:
        """Fetch per-bucket aggregate statistics for a processed document scope."""

        if document_version_ids is not None and not document_version_ids:
            return []
        if record_keys is not None and not record_keys:
            return []

        where_sql, params = self._build_processed_scope_filters(
            document_version_ids=document_version_ids,
            record_keys=record_keys,
            bucket_name=bucket_name,
        )
        sql = f"""
            WITH chunk_counts AS (
                SELECT
                    document_version_id,
                    COUNT(*)::INT AS chunk_count
                FROM document_chunks
                GROUP BY document_version_id
            ),
            selected_docs AS (
                SELECT
                    dv.document_version_id,
                    sd.bucket_name,
                    COALESCE(dv.chunk_count, chunk_counts.chunk_count, 0) AS chunk_count
                FROM document_versions dv
                INNER JOIN source_documents sd
                    ON sd.document_id = dv.document_id
                LEFT JOIN chunk_counts
                    ON chunk_counts.document_version_id = dv.document_version_id
                WHERE {where_sql}
            ),
            bucket_doc_stats AS (
                SELECT
                    bucket_name,
                    COUNT(*)::INT AS processed_documents,
                    COALESCE(SUM(chunk_count)::INT, 0) AS total_chunks,
                    AVG(chunk_count::DOUBLE PRECISION) AS avg_chunks_per_document
                FROM selected_docs
                GROUP BY bucket_name
            ),
            bucket_chunk_stats AS (
                SELECT
                    selected_docs.bucket_name,
                    AVG(dc.token_count::DOUBLE PRECISION) AS avg_tokens_per_chunk
                FROM selected_docs
                INNER JOIN document_chunks dc
                    ON dc.document_version_id = selected_docs.document_version_id
                GROUP BY selected_docs.bucket_name
            )
            SELECT
                bucket_doc_stats.bucket_name,
                bucket_doc_stats.processed_documents,
                bucket_doc_stats.total_chunks,
                bucket_doc_stats.avg_chunks_per_document,
                bucket_chunk_stats.avg_tokens_per_chunk
            FROM bucket_doc_stats
            LEFT JOIN bucket_chunk_stats
                ON bucket_chunk_stats.bucket_name = bucket_doc_stats.bucket_name
            ORDER BY bucket_doc_stats.bucket_name ASC
        """

        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [_bucket_aggregate_from_row(row) for row in rows]

    def sample_document_version_ids(
        self,
        *,
        document_version_ids: Sequence[int] | None = None,
        record_keys: Sequence[str] | None = None,
        bucket_name: str | None = None,
        limit: int | None = None,
        sample_per_bucket: int | None = None,
    ) -> list[int]:
        """Fetch a stable ordered sample of processed document_version identifiers."""

        if document_version_ids is not None and not document_version_ids:
            return []
        if record_keys is not None and not record_keys:
            return []

        where_sql, params = self._build_processed_scope_filters(
            document_version_ids=document_version_ids,
            record_keys=record_keys,
            bucket_name=bucket_name,
        )
        if sample_per_bucket is not None:
            sql = f"""
                WITH ranked_docs AS (
                    SELECT
                        dv.document_version_id,
                        sd.bucket_name,
                        COALESCE(dv.ingested_at, dv.created_at) AS sort_ts,
                        ROW_NUMBER() OVER (
                            PARTITION BY sd.bucket_name
                            ORDER BY COALESCE(dv.ingested_at, dv.created_at) ASC, dv.document_version_id ASC
                        ) AS bucket_rank
                    FROM document_versions dv
                    INNER JOIN source_documents sd
                        ON sd.document_id = dv.document_id
                    WHERE {where_sql}
                )
                SELECT document_version_id
                FROM ranked_docs
                WHERE bucket_rank <= %s
                ORDER BY bucket_name ASC, sort_ts ASC, document_version_id ASC
            """
            params.append(sample_per_bucket)
        else:
            sql = f"""
                SELECT dv.document_version_id
                FROM document_versions dv
                INNER JOIN source_documents sd
                    ON sd.document_id = dv.document_id
                WHERE {where_sql}
                ORDER BY COALESCE(dv.ingested_at, dv.created_at) ASC, dv.document_version_id ASC
            """
            if limit is not None:
                sql += "\n LIMIT %s"
                params.append(limit)

        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [int(row[0]) for row in rows]

    @staticmethod
    def _build_processed_scope_filters(
        *,
        document_version_ids: Sequence[int] | None,
        record_keys: Sequence[str] | None,
        bucket_name: str | None,
    ) -> tuple[str, list[Any]]:
        conditions = ["dv.ingest_status = 'done'"]
        params: list[Any] = []

        if document_version_ids is not None:
            conditions.append("dv.document_version_id = ANY(%s)")
            params.append(list(document_version_ids))
        if record_keys is not None:
            conditions.append("sd.record_key = ANY(%s)")
            params.append(list(record_keys))
        if bucket_name is not None:
            conditions.append("sd.bucket_name = %s")
            params.append(bucket_name)

        return " AND ".join(conditions), params


def _processed_document_from_row(row: tuple[Any, ...]) -> ProcessedDocumentVersionRow:
    return ProcessedDocumentVersionRow(
        document_version_id=row[0],
        document_id=row[1],
        record_key=row[2],
        bucket_name=row[3],
        order_date=row[4],
        title=row[5],
        page_count=row[6],
        chunk_count=row[7],
        ingested_at=row[8],
        created_at=row[9],
    )


def _chunk_from_row(row: tuple[Any, ...]) -> ChunkRow:
    return ChunkRow(
        chunk_index=row[0],
        page_start=row[1],
        page_end=row[2],
        section_type=row[3],
        section_title=row[4],
        heading_path=_split_heading_path(row[5]),
        chunk_text=row[6],
        chunk_sha256=row[7],
        token_count=row[8],
    )


def _corpus_aggregate_from_row(row: tuple[Any, ...]) -> CorpusAggregateRow:
    return CorpusAggregateRow(
        processed_document_versions=int(row[0] or 0),
        total_chunks=int(row[1] or 0),
        avg_chunks_per_document=_coerce_float(row[2]),
        median_chunks_per_document=_coerce_float(row[3]),
        avg_tokens_per_chunk=_coerce_float(row[4]),
        median_tokens_per_chunk=_coerce_float(row[5]),
        min_tokens_per_chunk=row[6],
        max_tokens_per_chunk=row[7],
    )


def _bucket_aggregate_from_row(row: tuple[Any, ...]) -> BucketAggregateRow:
    return BucketAggregateRow(
        bucket_name=row[0],
        processed_documents=int(row[1]),
        total_chunks=int(row[2]),
        avg_chunks_per_document=_coerce_float(row[3]),
        avg_tokens_per_chunk=_coerce_float(row[4]),
    )


def _empty_corpus_aggregate() -> CorpusAggregateRow:
    return CorpusAggregateRow(
        processed_document_versions=0,
        total_chunks=0,
        avg_chunks_per_document=None,
        median_chunks_per_document=None,
        avg_tokens_per_chunk=None,
        median_tokens_per_chunk=None,
        min_tokens_per_chunk=None,
        max_tokens_per_chunk=None,
    )


def _coerce_float(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _split_heading_path(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(" > ") if part.strip())
