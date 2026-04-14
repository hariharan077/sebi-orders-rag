"""Persistence helpers for extracted order metadata, legal provisions, and numeric facts."""

from __future__ import annotations

from typing import Any, Sequence

from ..metadata.models import (
    ExtractedLegalProvision,
    ExtractedNumericFact,
    ExtractedOrderMetadata,
    ExtractedPriceMovement,
    MetadataChunkText,
    MetadataExtractionTarget,
    MetadataPageText,
    StoredLegalProvision,
    StoredNumericFact,
    StoredOrderMetadata,
    StoredPriceMovement,
)


class OrderMetadataRepository:
    """Read and write extracted metadata without changing the RAG architecture."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def list_extraction_targets(
        self,
        *,
        record_key: str | None = None,
        document_version_id: int | None = None,
        limit: int | None = None,
        include_existing: bool = True,
        processed_only: bool = True,
    ) -> tuple[MetadataExtractionTarget, ...]:
        conditions = ["sd.current_version_id = dv.document_version_id"]
        params: list[Any] = []
        if processed_only:
            conditions.append("dv.ingest_status = 'done'")
        if record_key is not None:
            conditions.append("sd.record_key = %s")
            params.append(record_key)
        if document_version_id is not None:
            conditions.append("dv.document_version_id = %s")
            params.append(document_version_id)
        if not include_existing:
            conditions.append(
                "("
                "om.document_version_id IS NULL "
                "OR onf.document_version_id IS NULL "
                "OR opm.document_version_id IS NULL"
                ")"
            )
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT %s"
            params.append(limit)

        sql = f"""
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                dv.title,
                dv.order_date,
                dv.detail_url,
                dv.pdf_url
            FROM document_versions dv
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            LEFT JOIN order_metadata om
                ON om.document_version_id = dv.document_version_id
            LEFT JOIN (
                SELECT DISTINCT document_version_id
                FROM order_numeric_facts
            ) onf
                ON onf.document_version_id = dv.document_version_id
            LEFT JOIN (
                SELECT DISTINCT document_version_id
                FROM order_price_movements
            ) opm
                ON opm.document_version_id = dv.document_version_id
            WHERE {' AND '.join(conditions)}
            ORDER BY dv.document_version_id ASC
            {limit_clause}
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return tuple(
            MetadataExtractionTarget(
                document_version_id=int(row[0]),
                document_id=int(row[1]),
                record_key=row[2],
                title=row[3],
                order_date=row[4],
                detail_url=row[5],
                pdf_url=row[6],
            )
            for row in rows
        )

    def fetch_backfill_coverage(self) -> dict[str, int]:
        """Return aggregate metadata-backfill coverage across processed current versions."""

        sql = """
            WITH processed_docs AS (
                SELECT dv.document_version_id
                FROM document_versions dv
                INNER JOIN source_documents sd
                    ON sd.document_id = dv.document_id
                WHERE sd.current_version_id = dv.document_version_id
                  AND dv.ingest_status = 'done'
            ),
            metadata_counts AS (
                SELECT
                    COUNT(*)::INT AS metadata_docs,
                    COUNT(*) FILTER (WHERE COALESCE(NULLIF(TRIM(signatory_name), ''), NULL) IS NOT NULL)::INT AS signatory_name_docs,
                    COUNT(*) FILTER (WHERE COALESCE(NULLIF(TRIM(signatory_designation), ''), NULL) IS NOT NULL)::INT AS signatory_designation_docs,
                    COUNT(*) FILTER (WHERE order_date IS NOT NULL)::INT AS order_date_docs,
                    COUNT(*) FILTER (WHERE COALESCE(NULLIF(TRIM(place), ''), NULL) IS NOT NULL)::INT AS place_docs
                FROM order_metadata
                WHERE document_version_id IN (SELECT document_version_id FROM processed_docs)
            ),
            legal_counts AS (
                SELECT
                    COUNT(DISTINCT document_version_id)::INT AS legal_provision_docs,
                    COUNT(*)::INT AS legal_provision_rows
                FROM order_legal_provisions
                WHERE document_version_id IN (SELECT document_version_id FROM processed_docs)
            ),
            numeric_counts AS (
                SELECT
                    COUNT(DISTINCT document_version_id)::INT AS numeric_fact_docs,
                    COUNT(*)::INT AS numeric_fact_rows,
                    COUNT(DISTINCT document_version_id) FILTER (WHERE fact_type = 'listing_price')::INT AS listing_price_docs,
                    COUNT(DISTINCT document_version_id) FILTER (WHERE fact_type = 'highest_price')::INT AS highest_price_docs,
                    COUNT(DISTINCT document_version_id) FILTER (WHERE fact_type = 'settlement_amount')::INT AS settlement_amount_docs
                FROM order_numeric_facts
                WHERE document_version_id IN (SELECT document_version_id FROM processed_docs)
            ),
            price_counts AS (
                SELECT
                    COUNT(DISTINCT document_version_id)::INT AS price_movement_docs,
                    COUNT(*)::INT AS price_movement_rows
                FROM order_price_movements
                WHERE document_version_id IN (SELECT document_version_id FROM processed_docs)
            )
            SELECT
                (SELECT COUNT(*)::INT FROM processed_docs) AS processed_docs,
                metadata_counts.metadata_docs,
                metadata_counts.signatory_name_docs,
                metadata_counts.signatory_designation_docs,
                metadata_counts.order_date_docs,
                metadata_counts.place_docs,
                legal_counts.legal_provision_docs,
                legal_counts.legal_provision_rows,
                numeric_counts.numeric_fact_docs,
                numeric_counts.numeric_fact_rows,
                numeric_counts.listing_price_docs,
                numeric_counts.highest_price_docs,
                numeric_counts.settlement_amount_docs,
                price_counts.price_movement_docs,
                price_counts.price_movement_rows
            FROM metadata_counts, legal_counts, numeric_counts, price_counts
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
        if row is None:
            return {
                "processed_docs": 0,
                "metadata_docs": 0,
                "signatory_name_docs": 0,
                "signatory_designation_docs": 0,
                "order_date_docs": 0,
                "place_docs": 0,
                "legal_provision_docs": 0,
                "legal_provision_rows": 0,
                "numeric_fact_docs": 0,
                "numeric_fact_rows": 0,
                "listing_price_docs": 0,
                "highest_price_docs": 0,
                "settlement_amount_docs": 0,
                "price_movement_docs": 0,
                "price_movement_rows": 0,
            }
        return {
            "processed_docs": int(row[0] or 0),
            "metadata_docs": int(row[1] or 0),
            "signatory_name_docs": int(row[2] or 0),
            "signatory_designation_docs": int(row[3] or 0),
            "order_date_docs": int(row[4] or 0),
            "place_docs": int(row[5] or 0),
            "legal_provision_docs": int(row[6] or 0),
            "legal_provision_rows": int(row[7] or 0),
            "numeric_fact_docs": int(row[8] or 0),
            "numeric_fact_rows": int(row[9] or 0),
            "listing_price_docs": int(row[10] or 0),
            "highest_price_docs": int(row[11] or 0),
            "settlement_amount_docs": int(row[12] or 0),
            "price_movement_docs": int(row[13] or 0),
            "price_movement_rows": int(row[14] or 0),
        }

    def load_pages(self, *, document_version_id: int) -> tuple[MetadataPageText, ...]:
        sql = """
            SELECT page_no, final_text
            FROM document_pages
            WHERE document_version_id = %s
            ORDER BY page_no ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (document_version_id,))
            rows = cursor.fetchall()
        return tuple(MetadataPageText(page_no=int(row[0]), text=str(row[1] or "")) for row in rows)

    def load_chunks(self, *, document_version_id: int) -> tuple[MetadataChunkText, ...]:
        sql = """
            SELECT chunk_id, page_start, page_end, chunk_text, section_type, section_title
            FROM document_chunks
            WHERE document_version_id = %s
              AND is_active = TRUE
            ORDER BY chunk_index ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (document_version_id,))
            rows = cursor.fetchall()
        return tuple(
            MetadataChunkText(
                chunk_id=int(row[0]),
                page_start=int(row[1]),
                page_end=int(row[2]),
                text=str(row[3] or ""),
                section_type=str(row[4] or "").strip() or None,
                section_title=str(row[5] or "").strip() or None,
            )
            for row in rows
        )

    def upsert_order_metadata(self, metadata: ExtractedOrderMetadata) -> None:
        sql = """
            INSERT INTO order_metadata (
                document_version_id,
                signatory_name,
                signatory_designation,
                signatory_page_start,
                signatory_page_end,
                order_date,
                place,
                issuing_authority_type,
                authority_panel,
                metadata_confidence,
                extraction_version
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_version_id) DO UPDATE
            SET
                signatory_name = EXCLUDED.signatory_name,
                signatory_designation = EXCLUDED.signatory_designation,
                signatory_page_start = EXCLUDED.signatory_page_start,
                signatory_page_end = EXCLUDED.signatory_page_end,
                order_date = EXCLUDED.order_date,
                place = EXCLUDED.place,
                issuing_authority_type = EXCLUDED.issuing_authority_type,
                authority_panel = EXCLUDED.authority_panel,
                metadata_confidence = EXCLUDED.metadata_confidence,
                extraction_version = EXCLUDED.extraction_version,
                updated_at = NOW()
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    metadata.document_version_id,
                    metadata.signatory_name,
                    metadata.signatory_designation,
                    metadata.signatory_page_start,
                    metadata.signatory_page_end,
                    metadata.order_date,
                    metadata.place,
                    metadata.issuing_authority_type,
                    list(metadata.authority_panel),
                    metadata.metadata_confidence,
                    metadata.extraction_version,
                ),
            )

    def replace_legal_provisions(
        self,
        *,
        document_version_id: int,
        provisions: Sequence[ExtractedLegalProvision],
    ) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM order_legal_provisions WHERE document_version_id = %s",
                (document_version_id,),
            )
            if not provisions:
                return
            cursor.executemany(
                """
                INSERT INTO order_legal_provisions (
                    document_version_id,
                    statute_name,
                    section_or_regulation,
                    provision_type,
                    text_snippet,
                    page_start,
                    page_end,
                    row_sha256
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        row.document_version_id,
                        row.statute_name,
                        row.section_or_regulation,
                        row.provision_type,
                        row.text_snippet,
                        row.page_start,
                        row.page_end,
                        row.row_sha256,
                    )
                    for row in provisions
                ],
            )

    def replace_numeric_facts(
        self,
        *,
        document_version_id: int,
        facts: Sequence[ExtractedNumericFact],
    ) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM order_numeric_facts WHERE document_version_id = %s",
                (document_version_id,),
            )
            if not facts:
                return
            cursor.executemany(
                """
                INSERT INTO order_numeric_facts (
                    document_version_id,
                    fact_type,
                    subject,
                    value_text,
                    value_numeric,
                    unit,
                    context_label,
                    page_start,
                    page_end,
                    row_sha256
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        row.document_version_id,
                        row.fact_type,
                        row.subject,
                        row.value_text,
                        row.value_numeric,
                        row.unit,
                        row.context_label,
                        row.page_start,
                        row.page_end,
                        row.row_sha256,
                    )
                    for row in facts
                ],
            )

    def replace_price_movements(
        self,
        *,
        document_version_id: int,
        price_movements: Sequence[ExtractedPriceMovement],
    ) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM order_price_movements WHERE document_version_id = %s",
                (document_version_id,),
            )
            if not price_movements:
                return
            cursor.executemany(
                """
                INSERT INTO order_price_movements (
                    document_version_id,
                    period_label,
                    period_start_text,
                    period_end_text,
                    start_price,
                    high_price,
                    low_price,
                    end_price,
                    pct_change,
                    rationale,
                    page_start,
                    page_end,
                    row_sha256
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        row.document_version_id,
                        row.period_label,
                        row.period_start_text,
                        row.period_end_text,
                        row.start_price,
                        row.high_price,
                        row.low_price,
                        row.end_price,
                        row.pct_change,
                        row.rationale,
                        row.page_start,
                        row.page_end,
                        row.row_sha256,
                    )
                    for row in price_movements
                ],
            )

    def fetch_order_metadata(
        self,
        *,
        document_version_ids: Sequence[int],
    ) -> tuple[StoredOrderMetadata, ...]:
        if not document_version_ids:
            return ()
        sql = """
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                om.signatory_name,
                om.signatory_designation,
                om.signatory_page_start,
                om.signatory_page_end,
                COALESCE(om.order_date, dv.order_date) AS order_date,
                om.place,
                om.issuing_authority_type,
                om.authority_panel,
                om.metadata_confidence,
                om.extraction_version,
                om.updated_at
            FROM document_versions dv
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            LEFT JOIN order_metadata om
                ON om.document_version_id = dv.document_version_id
            WHERE dv.document_version_id = ANY(%s)
            ORDER BY dv.document_version_id ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (list(document_version_ids),))
            rows = cursor.fetchall()
        return tuple(
            StoredOrderMetadata(
                document_version_id=int(row[0]),
                document_id=int(row[1]),
                record_key=row[2],
                title=row[3],
                detail_url=row[4],
                pdf_url=row[5],
                signatory_name=row[6],
                signatory_designation=row[7],
                signatory_page_start=row[8],
                signatory_page_end=row[9],
                order_date=row[10],
                place=row[11],
                issuing_authority_type=row[12],
                authority_panel=tuple(row[13] or ()),
                metadata_confidence=float(row[14] or 0.0),
                extraction_version=row[15],
                updated_at=row[16],
            )
            for row in rows
        )

    def fetch_legal_provisions(
        self,
        *,
        document_version_ids: Sequence[int],
    ) -> tuple[StoredLegalProvision, ...]:
        if not document_version_ids:
            return ()
        sql = """
            SELECT
                olp.provision_id,
                olp.document_version_id,
                dv.document_id,
                sd.record_key,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                olp.statute_name,
                olp.section_or_regulation,
                olp.provision_type,
                olp.text_snippet,
                olp.page_start,
                olp.page_end,
                olp.row_sha256,
                olp.updated_at
            FROM order_legal_provisions olp
            INNER JOIN document_versions dv
                ON dv.document_version_id = olp.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE olp.document_version_id = ANY(%s)
            ORDER BY olp.document_version_id ASC, olp.page_start ASC NULLS LAST, olp.provision_id ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (list(document_version_ids),))
            rows = cursor.fetchall()
        return tuple(
            StoredLegalProvision(
                provision_id=int(row[0]),
                document_version_id=int(row[1]),
                document_id=int(row[2]),
                record_key=row[3],
                title=row[4],
                detail_url=row[5],
                pdf_url=row[6],
                statute_name=row[7],
                section_or_regulation=row[8],
                provision_type=row[9],
                text_snippet=row[10],
                page_start=row[11],
                page_end=row[12],
                row_sha256=row[13],
                updated_at=row[14],
            )
            for row in rows
        )

    def fetch_numeric_facts(
        self,
        *,
        document_version_ids: Sequence[int],
        fact_types: Sequence[str] | None = None,
    ) -> tuple[StoredNumericFact, ...]:
        if not document_version_ids:
            return ()
        params: list[Any] = [list(document_version_ids)]
        fact_type_clause = ""
        if fact_types:
            fact_type_clause = "AND onf.fact_type = ANY(%s)"
            params.append(list(fact_types))
        sql = f"""
            SELECT
                onf.numeric_fact_id,
                onf.document_version_id,
                dv.document_id,
                sd.record_key,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                onf.fact_type,
                onf.subject,
                onf.value_text,
                onf.value_numeric,
                onf.unit,
                onf.context_label,
                onf.page_start,
                onf.page_end,
                onf.row_sha256,
                onf.updated_at
            FROM order_numeric_facts onf
            INNER JOIN document_versions dv
                ON dv.document_version_id = onf.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE onf.document_version_id = ANY(%s)
              {fact_type_clause}
            ORDER BY
                onf.document_version_id ASC,
                onf.page_start ASC NULLS LAST,
                onf.numeric_fact_id ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return tuple(
            StoredNumericFact(
                numeric_fact_id=int(row[0]),
                document_version_id=int(row[1]),
                document_id=int(row[2]),
                record_key=row[3],
                title=row[4],
                detail_url=row[5],
                pdf_url=row[6],
                fact_type=row[7],
                subject=row[8],
                value_text=row[9],
                value_numeric=float(row[10]) if row[10] is not None else None,
                unit=row[11],
                context_label=row[12],
                page_start=row[13],
                page_end=row[14],
                row_sha256=row[15],
                updated_at=row[16],
            )
            for row in rows
        )

    def fetch_price_movements(
        self,
        *,
        document_version_ids: Sequence[int],
    ) -> tuple[StoredPriceMovement, ...]:
        if not document_version_ids:
            return ()
        sql = """
            SELECT
                opm.price_movement_id,
                opm.document_version_id,
                dv.document_id,
                sd.record_key,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                opm.period_label,
                opm.period_start_text,
                opm.period_end_text,
                opm.start_price,
                opm.high_price,
                opm.low_price,
                opm.end_price,
                opm.pct_change,
                opm.rationale,
                opm.page_start,
                opm.page_end,
                opm.row_sha256,
                opm.updated_at
            FROM order_price_movements opm
            INNER JOIN document_versions dv
                ON dv.document_version_id = opm.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE opm.document_version_id = ANY(%s)
            ORDER BY
                opm.document_version_id ASC,
                opm.page_start ASC NULLS LAST,
                opm.price_movement_id ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (list(document_version_ids),))
            rows = cursor.fetchall()
        return tuple(
            StoredPriceMovement(
                price_movement_id=int(row[0]),
                document_version_id=int(row[1]),
                document_id=int(row[2]),
                record_key=row[3],
                title=row[4],
                detail_url=row[5],
                pdf_url=row[6],
                period_label=row[7],
                period_start_text=row[8],
                period_end_text=row[9],
                start_price=float(row[10]) if row[10] is not None else None,
                high_price=float(row[11]) if row[11] is not None else None,
                low_price=float(row[12]) if row[12] is not None else None,
                end_price=float(row[13]) if row[13] is not None else None,
                pct_change=float(row[14]) if row[14] is not None else None,
                rationale=row[15],
                page_start=row[16],
                page_end=row[17],
                row_sha256=row[18],
                updated_at=row[19],
            )
            for row in rows
        )
