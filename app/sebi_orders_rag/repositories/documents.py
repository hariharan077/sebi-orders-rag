"""Document persistence helpers for the retrieval store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..constants import DONE_STATUS, FAILED_STATUS, PENDING_STATUS, PROCESSING_STATUS
from ..schemas import (
    DocumentVersionRecord,
    EmbeddingCandidate,
    PendingDocumentVersion,
    PlannedDocumentVersionCreate,
    SourceDocumentRecord,
)


class DocumentRepository:
    """Explicit CRUD operations used by Phase 1 ingestion."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def get_document_by_record_key(self, record_key: str) -> SourceDocumentRecord | None:
        sql = """
            SELECT
                document_id,
                record_key,
                bucket_name,
                external_record_id,
                first_seen_at,
                last_seen_at,
                current_version_id,
                is_active
            FROM source_documents
            WHERE record_key = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (record_key,))
            row = cursor.fetchone()
        if row is None:
            return None
        return _source_document_from_row(row)

    def create_document(
        self,
        *,
        record_key: str,
        bucket_name: str,
        external_record_id: str | None,
        first_seen_at: datetime,
        last_seen_at: datetime,
    ) -> SourceDocumentRecord:
        sql = """
            INSERT INTO source_documents (
                record_key,
                bucket_name,
                external_record_id,
                first_seen_at,
                last_seen_at
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING
                document_id,
                record_key,
                bucket_name,
                external_record_id,
                first_seen_at,
                last_seen_at,
                current_version_id,
                is_active
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    record_key,
                    bucket_name,
                    external_record_id,
                    first_seen_at,
                    last_seen_at,
                ),
            )
            row = cursor.fetchone()
        return _source_document_from_row(row)

    def update_document_seen_timestamps_and_current_version(
        self,
        *,
        document_id: int,
        bucket_name: str,
        external_record_id: str | None,
        first_seen_at: datetime,
        last_seen_at: datetime,
        current_version_id: int | None,
    ) -> SourceDocumentRecord:
        sql = """
            UPDATE source_documents
            SET
                bucket_name = %s,
                external_record_id = COALESCE(%s, external_record_id),
                first_seen_at = LEAST(first_seen_at, %s),
                last_seen_at = GREATEST(last_seen_at, %s),
                current_version_id = %s,
                is_active = TRUE
            WHERE document_id = %s
            RETURNING
                document_id,
                record_key,
                bucket_name,
                external_record_id,
                first_seen_at,
                last_seen_at,
                current_version_id,
                is_active
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    bucket_name,
                    external_record_id,
                    first_seen_at,
                    last_seen_at,
                    current_version_id,
                    document_id,
                ),
            )
            row = cursor.fetchone()
        return _source_document_from_row(row)

    def get_version_by_document_id_and_file_sha256(
        self,
        *,
        document_id: int,
        file_sha256: str,
    ) -> DocumentVersionRecord | None:
        sql = """
            SELECT
                document_version_id,
                document_id,
                order_date,
                title,
                detail_url,
                pdf_url,
                local_filename,
                local_path,
                file_size_bytes,
                file_sha256,
                manifest_status,
                parser_name,
                parser_version,
                extraction_status,
                ocr_used,
                page_count,
                extracted_char_count,
                ingest_status,
                ingest_error,
                ingested_at,
                created_at
            FROM document_versions
            WHERE document_id = %s
              AND file_sha256 = %s
            ORDER BY document_version_id DESC
            LIMIT 1
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (document_id, file_sha256))
            row = cursor.fetchone()
        if row is None:
            return None
        return _document_version_from_row(row)

    def create_document_version(
        self,
        *,
        document_id: int,
        version: PlannedDocumentVersionCreate,
    ) -> DocumentVersionRecord:
        sql = """
            INSERT INTO document_versions (
                document_id,
                order_date,
                title,
                detail_url,
                pdf_url,
                local_filename,
                local_path,
                file_size_bytes,
                file_sha256,
                manifest_status,
                parser_name,
                parser_version,
                extraction_status,
                ingest_status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING
                document_version_id,
                document_id,
                order_date,
                title,
                detail_url,
                pdf_url,
                local_filename,
                local_path,
                file_size_bytes,
                file_sha256,
                manifest_status,
                parser_name,
                parser_version,
                extraction_status,
                ocr_used,
                page_count,
                extracted_char_count,
                ingest_status,
                ingest_error,
                ingested_at,
                created_at
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    document_id,
                    version.order_date,
                    version.title,
                    version.detail_url,
                    version.pdf_url,
                    version.local_filename,
                    version.local_path,
                    version.file_size_bytes,
                    version.file_sha256,
                    version.manifest_status,
                    version.parser_name,
                    version.parser_version,
                    version.extraction_status,
                    version.ingest_status,
                ),
            )
            row = cursor.fetchone()
        return _document_version_from_row(row)

    def list_pending_versions(
        self,
        *,
        record_key: str | None = None,
        document_version_id: int | None = None,
        limit: int | None = None,
    ) -> list[PendingDocumentVersion]:
        """List Phase 2 candidates ordered oldest pending first."""

        conditions = [
            "((dv.extraction_status IN (%s, %s)) OR (dv.ingest_status IN (%s, %s)))"
        ]
        params: list[Any] = [PENDING_STATUS, FAILED_STATUS, PENDING_STATUS, FAILED_STATUS]

        if record_key is not None:
            conditions.append("sd.record_key = %s")
            params.append(record_key)
        if document_version_id is not None:
            conditions.append("dv.document_version_id = %s")
            params.append(document_version_id)

        sql = f"""
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                dv.order_date,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                dv.local_filename,
                dv.local_path,
                dv.file_size_bytes,
                dv.file_sha256,
                dv.manifest_status,
                dv.parser_name,
                dv.parser_version,
                dv.extraction_status,
                dv.ocr_used,
                dv.page_count,
                dv.extracted_char_count,
                dv.ingest_status,
                dv.ingest_error,
                dv.ingested_at,
                dv.created_at,
                dv.chunking_version,
                dv.chunk_count
            FROM document_versions dv
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE {' AND '.join(conditions)}
            ORDER BY dv.document_version_id ASC
        """
        if limit is not None:
            sql += "\n LIMIT %s"
            params.append(limit)

        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [_pending_document_version_from_row(row) for row in rows]

    def list_embedding_candidates(
        self,
        *,
        record_key: str | None = None,
        bucket_name: str | None = None,
        document_version_id: int | None = None,
        limit: int | None = None,
    ) -> list[EmbeddingCandidate]:
        """List Phase 3 candidates ordered oldest not-yet-embedded first."""

        conditions = [
            "dv.ingest_status = %s",
            "(dv.embedding_status IS NULL OR dv.embedding_status IN (%s, %s))",
        ]
        params: list[Any] = [DONE_STATUS, PENDING_STATUS, FAILED_STATUS]

        if record_key is not None:
            conditions.append("sd.record_key = %s")
            params.append(record_key)
        if bucket_name is not None:
            conditions.append("sd.bucket_name = %s")
            params.append(bucket_name)
        if document_version_id is not None:
            conditions.append("dv.document_version_id = %s")
            params.append(document_version_id)

        sql = f"""
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                dv.local_filename,
                dv.local_path,
                dv.ingest_status,
                dv.chunking_version,
                dv.chunk_count,
                dv.embedding_status,
                dv.embedding_error,
                dv.embedding_model,
                dv.embedding_dim,
                dv.embedded_at,
                dv.created_at
            FROM document_versions dv
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE {' AND '.join(conditions)}
            ORDER BY dv.document_version_id ASC
        """
        if limit is not None:
            sql += "\n LIMIT %s"
            params.append(limit)

        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [_embedding_candidate_from_row(row) for row in rows]

    def mark_version_done(
        self,
        *,
        document_version_id: int,
        parser_name: str,
        parser_version: str,
        chunking_version: str,
        page_count: int,
        extracted_char_count: int,
        ocr_used: bool,
        chunk_count: int,
    ) -> None:
        """Mark a document version as successfully extracted and chunked."""

        sql = """
            UPDATE document_versions
            SET
                parser_name = %s,
                parser_version = %s,
                extraction_status = %s,
                ocr_used = %s,
                page_count = %s,
                extracted_char_count = %s,
                ingest_status = %s,
                ingest_error = NULL,
                chunking_version = %s,
                chunk_count = %s,
                ingested_at = NOW()
            WHERE document_version_id = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    parser_name,
                    parser_version,
                    DONE_STATUS,
                    ocr_used,
                    page_count,
                    extracted_char_count,
                    DONE_STATUS,
                    chunking_version,
                    chunk_count,
                    document_version_id,
                ),
            )

    def mark_version_failed(
        self,
        *,
        document_version_id: int,
        parser_name: str,
        parser_version: str,
        chunking_version: str,
        ingest_error: str,
    ) -> None:
        """Mark a document version as failed without raising away the run."""

        sql = """
            UPDATE document_versions
            SET
                parser_name = %s,
                parser_version = %s,
                extraction_status = %s,
                ingest_status = %s,
                ingest_error = %s,
                chunking_version = %s
            WHERE document_version_id = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    parser_name,
                    parser_version,
                    FAILED_STATUS,
                    FAILED_STATUS,
                    ingest_error,
                    chunking_version,
                    document_version_id,
                ),
            )

    def mark_embedding_processing(self, *, document_version_id: int) -> None:
        """Mark a document version as actively being embedded."""

        sql = """
            UPDATE document_versions
            SET
                embedding_status = %s,
                embedding_error = NULL
            WHERE document_version_id = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (PROCESSING_STATUS, document_version_id))

    def mark_embedding_done(
        self,
        *,
        document_version_id: int,
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        """Mark a document version as successfully embedded."""

        sql = """
            UPDATE document_versions
            SET
                embedding_status = %s,
                embedding_error = NULL,
                embedding_model = %s,
                embedding_dim = %s,
                embedded_at = NOW()
            WHERE document_version_id = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (DONE_STATUS, embedding_model, embedding_dim, document_version_id),
            )

    def mark_embedding_failed(
        self,
        *,
        document_version_id: int,
        embedding_error: str,
    ) -> None:
        """Mark a document version as failed during Phase 3 embedding."""

        sql = """
            UPDATE document_versions
            SET
                embedding_status = %s,
                embedding_error = %s
            WHERE document_version_id = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (FAILED_STATUS, embedding_error, document_version_id),
            )


def _source_document_from_row(row: tuple[Any, ...]) -> SourceDocumentRecord:
    return SourceDocumentRecord(
        document_id=row[0],
        record_key=row[1],
        bucket_name=row[2],
        external_record_id=row[3],
        first_seen_at=row[4],
        last_seen_at=row[5],
        current_version_id=row[6],
        is_active=row[7],
    )


def _document_version_from_row(row: tuple[Any, ...]) -> DocumentVersionRecord:
    chunking_version = row[21] if len(row) > 21 else None
    chunk_count = row[22] if len(row) > 22 else None
    embedding_status = row[23] if len(row) > 23 else None
    embedding_error = row[24] if len(row) > 24 else None
    embedding_model = row[25] if len(row) > 25 else None
    embedding_dim = row[26] if len(row) > 26 else None
    embedded_at = row[27] if len(row) > 27 else None
    return DocumentVersionRecord(
        document_version_id=row[0],
        document_id=row[1],
        order_date=row[2],
        title=row[3],
        detail_url=row[4],
        pdf_url=row[5],
        local_filename=row[6],
        local_path=row[7],
        file_size_bytes=row[8],
        file_sha256=row[9],
        manifest_status=row[10],
        parser_name=row[11],
        parser_version=row[12],
        extraction_status=row[13],
        ocr_used=row[14],
        page_count=row[15],
        extracted_char_count=row[16],
        ingest_status=row[17],
        ingest_error=row[18],
        ingested_at=row[19],
        created_at=row[20],
        chunking_version=chunking_version,
        chunk_count=chunk_count,
        embedding_status=embedding_status,
        embedding_error=embedding_error,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedded_at=embedded_at,
    )


def _pending_document_version_from_row(row: tuple[Any, ...]) -> PendingDocumentVersion:
    embedding_status = row[25] if len(row) > 25 else None
    embedding_error = row[26] if len(row) > 26 else None
    embedding_model = row[27] if len(row) > 27 else None
    embedding_dim = row[28] if len(row) > 28 else None
    embedded_at = row[29] if len(row) > 29 else None
    return PendingDocumentVersion(
        document_version_id=row[0],
        document_id=row[1],
        record_key=row[2],
        bucket_name=row[3],
        order_date=row[4],
        title=row[5],
        detail_url=row[6],
        pdf_url=row[7],
        local_filename=row[8],
        local_path=row[9],
        file_size_bytes=row[10],
        file_sha256=row[11],
        manifest_status=row[12],
        parser_name=row[13],
        parser_version=row[14],
        extraction_status=row[15],
        ocr_used=row[16],
        page_count=row[17],
        extracted_char_count=row[18],
        ingest_status=row[19],
        ingest_error=row[20],
        ingested_at=row[21],
        created_at=row[22],
        chunking_version=row[23],
        chunk_count=row[24],
        embedding_status=embedding_status,
        embedding_error=embedding_error,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedded_at=embedded_at,
    )


def _embedding_candidate_from_row(row: tuple[Any, ...]) -> EmbeddingCandidate:
    return EmbeddingCandidate(
        document_version_id=row[0],
        document_id=row[1],
        record_key=row[2],
        bucket_name=row[3],
        external_record_id=row[4],
        order_date=row[5],
        title=row[6],
        detail_url=row[7],
        pdf_url=row[8],
        local_filename=row[9],
        local_path=row[10],
        ingest_status=row[11],
        chunking_version=row[12],
        chunk_count=row[13],
        embedding_status=row[14],
        embedding_error=row[15],
        embedding_model=row[16],
        embedding_dim=row[17],
        embedded_at=row[18],
        created_at=row[19],
    )
