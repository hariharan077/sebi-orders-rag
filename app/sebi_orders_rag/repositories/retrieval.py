"""Hierarchical retrieval persistence and search queries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..schemas import (
    DocumentNodeUpsert,
    ExactLookupCandidate,
    MetadataFilterInput,
    SectionNodeUpsert,
)
from ..retrieval.filters import build_shared_filter_clauses, normalize_metadata_filters
from ..retrieval.scoring import (
    ChunkSearchHit,
    DocumentSearchHit,
    ScoreBreakdown,
    SectionSearchHit,
)


class HierarchicalRetrievalRepository:
    """Explicit SQL helpers for Phase 3 node storage and hybrid retrieval."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def replace_document_node(self, node: DocumentNodeUpsert) -> None:
        """Delete and replace the document node for one document version."""

        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM document_nodes WHERE document_version_id = %s",
                (node.document_version_id,),
            )
            cursor.execute(
                """
                INSERT INTO document_nodes (
                    document_version_id,
                    node_text,
                    token_count,
                    embedding,
                    embedding_model,
                    embedding_created_at,
                    metadata
                )
                VALUES (%s, %s, %s, %s::vector, %s, NOW(), %s::jsonb)
                """,
                (
                    node.document_version_id,
                    node.node_text,
                    node.token_count,
                    _vector_literal(node.embedding),
                    node.embedding_model,
                    json.dumps(node.metadata, sort_keys=True),
                ),
            )

    def replace_section_nodes(
        self,
        *,
        document_version_id: int,
        nodes: Sequence[SectionNodeUpsert],
    ) -> int:
        """Delete and replace all section nodes for one document version."""

        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM section_nodes WHERE document_version_id = %s",
                (document_version_id,),
            )
            if not nodes:
                return 0
            cursor.executemany(
                """
                INSERT INTO section_nodes (
                    document_version_id,
                    section_key,
                    section_type,
                    section_title,
                    heading_path,
                    page_start,
                    page_end,
                    node_text,
                    token_count,
                    embedding,
                    embedding_model,
                    embedding_created_at,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, NOW(), %s::jsonb)
                """,
                [
                    (
                        node.document_version_id,
                        node.section_key,
                        node.section_type,
                        node.section_title,
                        node.heading_path,
                        node.page_start,
                        node.page_end,
                        node.node_text,
                        node.token_count,
                        _vector_literal(node.embedding),
                        node.embedding_model,
                        json.dumps(node.metadata, sort_keys=True),
                    )
                    for node in nodes
                ],
            )
        return len(nodes)

    def find_exact_lookup_candidates(
        self,
        *,
        query: str,
        limit: int,
        query_variants: Sequence[str] | None = None,
    ) -> list[ExactLookupCandidate]:
        """Return strong document-identity candidates for exact lookup queries."""

        variants = tuple(dict.fromkeys(item.strip() for item in (query_variants or (query,)) if item and item.strip()))
        merged: dict[int, ExactLookupCandidate] = {}
        for variant in variants:
            for candidate in self._find_exact_lookup_candidates_single_query(query=variant, limit=limit):
                existing = merged.get(candidate.document_version_id)
                if existing is None or candidate.match_score > existing.match_score:
                    merged[candidate.document_version_id] = candidate
        return sorted(
            merged.values(),
            key=lambda item: (-item.match_score, item.document_version_id),
        )[:limit]

    def _find_exact_lookup_candidates_single_query(
        self,
        *,
        query: str,
        limit: int,
    ) -> list[ExactLookupCandidate]:
        """Return exact-lookup candidates for one normalized query variant."""

        normalized_query = " ".join(query.lower().split())
        sql = """
            WITH search_input AS (
                SELECT
                    trim(regexp_replace(%s::text, '[^a-z0-9]+', ' ', 'g')) AS normalized_query,
                    websearch_to_tsquery('english', %s) AS ts_query
            )
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                GREATEST(
                    CASE
                        WHEN lower(sd.record_key) = search_input.normalized_query THEN 1.0
                        ELSE 0.0
                    END,
                    CASE
                        WHEN trim(regexp_replace(lower(dv.title), '[^a-z0-9]+', ' ', 'g')) = search_input.normalized_query THEN 0.99
                        ELSE 0.0
                    END,
                    CASE
                        WHEN trim(regexp_replace(lower(dv.title), '[^a-z0-9]+', ' ', 'g')) LIKE '%%' || search_input.normalized_query || '%%' THEN 0.95
                        ELSE 0.0
                    END,
                    ts_rank_cd(
                        to_tsvector('english', COALESCE(dv.title, '')),
                        search_input.ts_query
                    ),
                    similarity(
                        trim(regexp_replace(lower(COALESCE(dv.title, '')), '[^a-z0-9]+', ' ', 'g')),
                        search_input.normalized_query
                    ),
                    similarity(lower(COALESCE(sd.record_key, '')), search_input.normalized_query)
                ) AS match_score
            FROM document_versions dv
            CROSS JOIN search_input
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE (
                sd.current_version_id = dv.document_version_id
                AND (
                lower(sd.record_key) = search_input.normalized_query
                OR trim(regexp_replace(lower(dv.title), '[^a-z0-9]+', ' ', 'g')) = search_input.normalized_query
                OR trim(regexp_replace(lower(dv.title), '[^a-z0-9]+', ' ', 'g')) LIKE '%%' || search_input.normalized_query || '%%'
                OR similarity(
                    trim(regexp_replace(lower(COALESCE(dv.title, '')), '[^a-z0-9]+', ' ', 'g')),
                    search_input.normalized_query
                ) >= 0.35
                OR similarity(lower(COALESCE(sd.record_key, '')), search_input.normalized_query) >= 0.75
                )
            )
            ORDER BY match_score DESC, dv.document_version_id ASC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (normalized_query, normalized_query, limit))
            rows = cursor.fetchall()
        return [_exact_lookup_candidate_from_row(row) for row in rows]

    def resolve_current_document_version_ids(
        self,
        *,
        record_keys: Sequence[str] = (),
        document_ids: Sequence[int] = (),
    ) -> tuple[int, ...]:
        """Resolve current document version ids for the active session scope."""

        cleaned_record_keys = [value.strip() for value in record_keys if value and value.strip()]
        cleaned_document_ids = [int(value) for value in document_ids if int(value) > 0]
        if not cleaned_record_keys and not cleaned_document_ids:
            return ()

        scope_conditions: list[str] = []
        params: list[Any] = []
        if cleaned_record_keys:
            scope_conditions.append("record_key = ANY(%s)")
            params.append(cleaned_record_keys)
        if cleaned_document_ids:
            scope_conditions.append("document_id = ANY(%s)")
            params.append(cleaned_document_ids)

        sql = f"""
            SELECT DISTINCT current_version_id
            FROM source_documents
            WHERE current_version_id IS NOT NULL
              AND ({' OR '.join(scope_conditions)})
            ORDER BY current_version_id ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return tuple(int(row[0]) for row in rows if row and row[0] is not None)

    def search_documents_lexical(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[DocumentSearchHit]:
        """Run lexical document search across title and document node text."""

        clauses, params = build_shared_filter_clauses(filters)
        fts_vector = (
            "setweight(to_tsvector('english', COALESCE(dv.title, '')), 'A') "
            "|| setweight(to_tsvector('english', COALESCE(dn.node_text, '')), 'B')"
        )
        search_condition = (
            "("
            f"({fts_vector}) @@ search_input.ts_query "
            "OR similarity(COALESCE(dv.title, ''), search_input.raw_query) >= 0.05 "
            "OR similarity(COALESCE(dn.node_text, ''), search_input.raw_query) >= 0.05"
            ")"
        )
        clauses.append(search_condition)
        query_params = [query, query]
        params.append(limit)

        sql = f"""
            WITH search_input AS (
                SELECT
                    websearch_to_tsquery('english', %s) AS ts_query,
                    %s::text AS raw_query
            )
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                dn.node_text,
                ts_rank_cd(
                    {fts_vector},
                    search_input.ts_query
                ) AS fts_score,
                GREATEST(
                    similarity(COALESCE(dv.title, ''), search_input.raw_query),
                    similarity(COALESCE(dn.node_text, ''), search_input.raw_query)
                ) AS trigram_score
            FROM document_versions dv
            CROSS JOIN search_input
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            LEFT JOIN document_nodes dn
                ON dn.document_version_id = dv.document_version_id
            WHERE {' AND '.join(clauses)}
            ORDER BY ((ts_rank_cd(
                        {fts_vector},
                        search_input.ts_query
                    ) * 0.8) +
                    (GREATEST(
                        similarity(COALESCE(dv.title, ''), search_input.raw_query),
                        similarity(COALESCE(dn.node_text, ''), search_input.raw_query)
                    ) * 0.2)) DESC,
                    dv.document_version_id ASC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(query_params + params))
            rows = cursor.fetchall()
        return [_document_lexical_hit_from_row(row) for row in rows]

    def search_documents_vector(
        self,
        *,
        query_embedding: Sequence[float],
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[DocumentSearchHit]:
        """Run vector document search against document nodes."""

        clauses, params = build_shared_filter_clauses(filters)
        clauses.append("dn.embedding IS NOT NULL")
        vector = _vector_literal(query_embedding)
        query_params = [vector]
        params.append(limit)

        sql = f"""
            WITH query_embedding AS (
                SELECT %s::vector AS embedding
            )
            SELECT
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                dn.node_text,
                (dn.embedding <=> query_embedding.embedding) AS vector_distance,
                (1 - (dn.embedding <=> query_embedding.embedding)) AS vector_score
            FROM document_versions dv
            CROSS JOIN query_embedding
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            INNER JOIN document_nodes dn
                ON dn.document_version_id = dv.document_version_id
            WHERE {' AND '.join(clauses)}
            ORDER BY dn.embedding <=> query_embedding.embedding ASC, dv.document_version_id ASC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(query_params + params))
            rows = cursor.fetchall()
        return [_document_vector_hit_from_row(row) for row in rows]

    def search_sections_lexical(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[SectionSearchHit]:
        """Run lexical section search within the narrowed document scope."""

        clauses, params = build_shared_filter_clauses(filters, section_alias="sn")
        fts_vector = (
            "setweight(to_tsvector('english', COALESCE(sn.section_title, '')), 'A') "
            "|| setweight(to_tsvector('english', COALESCE(sn.heading_path, '')), 'B') "
            "|| setweight(to_tsvector('english', COALESCE(sn.node_text, '')), 'C')"
        )
        search_condition = (
            "("
            f"({fts_vector}) @@ search_input.ts_query "
            "OR similarity(COALESCE(sn.section_title, ''), search_input.raw_query) >= 0.05 "
            "OR similarity(COALESCE(sn.node_text, ''), search_input.raw_query) >= 0.05"
            ")"
        )
        clauses.append(search_condition)
        query_params = [query, query]
        params.append(limit)

        sql = f"""
            WITH search_input AS (
                SELECT
                    websearch_to_tsquery('english', %s) AS ts_query,
                    %s::text AS raw_query
            )
            SELECT
                sn.section_node_id,
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                sn.section_key,
                sn.section_type,
                sn.section_title,
                sn.heading_path,
                sn.page_start,
                sn.page_end,
                sn.node_text,
                ts_rank_cd(
                    {fts_vector},
                    search_input.ts_query
                ) AS fts_score,
                GREATEST(
                    similarity(COALESCE(sn.section_title, ''), search_input.raw_query),
                    similarity(COALESCE(sn.node_text, ''), search_input.raw_query)
                ) AS trigram_score
            FROM section_nodes sn
            CROSS JOIN search_input
            INNER JOIN document_versions dv
                ON dv.document_version_id = sn.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE {' AND '.join(clauses)}
            ORDER BY ((ts_rank_cd(
                        {fts_vector},
                        search_input.ts_query
                    ) * 0.8) +
                    (GREATEST(
                        similarity(COALESCE(sn.section_title, ''), search_input.raw_query),
                        similarity(COALESCE(sn.node_text, ''), search_input.raw_query)
                    ) * 0.2)) DESC,
                    sn.section_node_id ASC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(query_params + params))
            rows = cursor.fetchall()
        return [_section_lexical_hit_from_row(row) for row in rows]

    def search_sections_vector(
        self,
        *,
        query_embedding: Sequence[float],
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[SectionSearchHit]:
        """Run vector section search within the narrowed document scope."""

        clauses, params = build_shared_filter_clauses(filters, section_alias="sn")
        clauses.append("sn.embedding IS NOT NULL")
        vector = _vector_literal(query_embedding)
        query_params = [vector]
        params.append(limit)

        sql = f"""
            WITH query_embedding AS (
                SELECT %s::vector AS embedding
            )
            SELECT
                sn.section_node_id,
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                sn.section_key,
                sn.section_type,
                sn.section_title,
                sn.heading_path,
                sn.page_start,
                sn.page_end,
                sn.node_text,
                (sn.embedding <=> query_embedding.embedding) AS vector_distance,
                (1 - (sn.embedding <=> query_embedding.embedding)) AS vector_score
            FROM section_nodes sn
            CROSS JOIN query_embedding
            INNER JOIN document_versions dv
                ON dv.document_version_id = sn.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE {' AND '.join(clauses)}
            ORDER BY sn.embedding <=> query_embedding.embedding ASC, sn.section_node_id ASC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(query_params + params))
            rows = cursor.fetchall()
        return [_section_vector_hit_from_row(row) for row in rows]

    def search_chunks_lexical(
        self,
        *,
        query: str,
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[ChunkSearchHit]:
        """Run lexical chunk search within the narrowed scope."""

        clauses, params = build_shared_filter_clauses(filters, chunk_alias="dc")
        fts_vector = (
            "setweight(to_tsvector('english', COALESCE(dc.section_title, '')), 'A') "
            "|| setweight(to_tsvector('english', COALESCE(dc.chunk_text, '')), 'B')"
        )
        search_condition = (
            "("
            f"({fts_vector}) @@ search_input.ts_query "
            "OR similarity(COALESCE(dc.chunk_text, ''), search_input.raw_query) >= 0.05"
            ")"
        )
        clauses.append(search_condition)
        query_params = [query, query]
        params.append(limit)

        sql = f"""
            WITH search_input AS (
                SELECT
                    websearch_to_tsquery('english', %s) AS ts_query,
                    %s::text AS raw_query
            )
            SELECT
                dc.chunk_id,
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                dc.chunk_index,
                dc.page_start,
                dc.page_end,
                dc.section_key,
                dc.section_type,
                dc.section_title,
                dc.heading_path,
                dc.chunk_text,
                dc.token_count,
                ts_rank_cd(
                    {fts_vector},
                    search_input.ts_query
                ) AS fts_score,
                similarity(COALESCE(dc.chunk_text, ''), search_input.raw_query) AS trigram_score
            FROM document_chunks dc
            CROSS JOIN search_input
            INNER JOIN document_versions dv
                ON dv.document_version_id = dc.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE {' AND '.join(clauses)}
            ORDER BY ((ts_rank_cd(
                        {fts_vector},
                        search_input.ts_query
                    ) * 0.85) +
                    (similarity(COALESCE(dc.chunk_text, ''), search_input.raw_query) * 0.15)) DESC,
                    dc.chunk_id ASC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(query_params + params))
            rows = cursor.fetchall()
        return [_chunk_lexical_hit_from_row(row) for row in rows]

    def search_chunks_vector(
        self,
        *,
        query_embedding: Sequence[float],
        filters: MetadataFilterInput | None,
        limit: int,
    ) -> list[ChunkSearchHit]:
        """Run vector chunk search within the narrowed scope."""

        clauses, params = build_shared_filter_clauses(filters, chunk_alias="dc")
        clauses.append("dc.embedding IS NOT NULL")
        vector = _vector_literal(query_embedding)
        query_params = [vector]
        params.append(limit)

        sql = f"""
            WITH query_embedding AS (
                SELECT %s::vector AS embedding
            )
            SELECT
                dc.chunk_id,
                dv.document_version_id,
                dv.document_id,
                sd.record_key,
                sd.bucket_name,
                sd.external_record_id,
                dv.order_date,
                dv.title,
                dv.detail_url,
                dv.pdf_url,
                dc.chunk_index,
                dc.page_start,
                dc.page_end,
                dc.section_key,
                dc.section_type,
                dc.section_title,
                dc.heading_path,
                dc.chunk_text,
                dc.token_count,
                (dc.embedding <=> query_embedding.embedding) AS vector_distance,
                (1 - (dc.embedding <=> query_embedding.embedding)) AS vector_score
            FROM document_chunks dc
            CROSS JOIN query_embedding
            INNER JOIN document_versions dv
                ON dv.document_version_id = dc.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE {' AND '.join(clauses)}
            ORDER BY dc.embedding <=> query_embedding.embedding ASC, dc.chunk_id ASC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, tuple(query_params + params))
            rows = cursor.fetchall()
        return [_chunk_vector_hit_from_row(row) for row in rows]


def merge_filter_scope(
    base_filters: MetadataFilterInput | None,
    *,
    document_version_ids: Sequence[int] | None = None,
    section_keys: Sequence[str] | None = None,
) -> MetadataFilterInput:
    """Return normalized filters narrowed to the provided ids."""

    normalized = normalize_metadata_filters(base_filters)
    merged_document_ids = set(normalized.document_version_ids)
    merged_section_keys = set(normalized.section_keys)

    for value in document_version_ids or ():
        if int(value) > 0:
            merged_document_ids.add(int(value))
    for value in section_keys or ():
        cleaned = str(value).strip()
        if cleaned:
            merged_section_keys.add(cleaned)

    return replace(
        normalized,
        document_version_ids=tuple(sorted(merged_document_ids)),
        section_keys=tuple(sorted(merged_section_keys)),
    )


def _document_lexical_hit_from_row(row: tuple[Any, ...]) -> DocumentSearchHit:
    fts_score = float(row[8] or 0.0)
    trigram_score = float(row[9] or 0.0)
    return DocumentSearchHit(
        document_version_id=row[0],
        document_id=row[1],
        record_key=row[2],
        bucket_name=row[3],
        external_record_id=row[4],
        order_date=row[5],
        title=row[6],
        document_node_text=row[7],
        score=ScoreBreakdown(
            lexical_score=(fts_score * 0.8) + (trigram_score * 0.2),
            fts_score=fts_score,
            trigram_score=trigram_score,
        ),
    )


def _exact_lookup_candidate_from_row(row: tuple[Any, ...]) -> ExactLookupCandidate:
    return ExactLookupCandidate(
        document_version_id=row[0],
        document_id=row[1],
        record_key=row[2],
        bucket_name=row[3],
        external_record_id=row[4],
        order_date=row[5],
        title=row[6],
        match_score=float(row[7] or 0.0),
    )


def _document_vector_hit_from_row(row: tuple[Any, ...]) -> DocumentSearchHit:
    vector_distance = float(row[8] or 0.0)
    vector_score = float(row[9] or 0.0)
    return DocumentSearchHit(
        document_version_id=row[0],
        document_id=row[1],
        record_key=row[2],
        bucket_name=row[3],
        external_record_id=row[4],
        order_date=row[5],
        title=row[6],
        document_node_text=row[7],
        score=ScoreBreakdown(
            vector_score=vector_score,
            vector_distance=vector_distance,
        ),
    )


def _section_lexical_hit_from_row(row: tuple[Any, ...]) -> SectionSearchHit:
    fts_score = float(row[15] or 0.0)
    trigram_score = float(row[16] or 0.0)
    return SectionSearchHit(
        section_node_id=row[0],
        document_version_id=row[1],
        document_id=row[2],
        record_key=row[3],
        bucket_name=row[4],
        external_record_id=row[5],
        order_date=row[6],
        title=row[7],
        section_key=row[8],
        section_type=row[9],
        section_title=row[10],
        heading_path=row[11],
        page_start=row[12],
        page_end=row[13],
        section_node_text=row[14],
        score=ScoreBreakdown(
            lexical_score=(fts_score * 0.8) + (trigram_score * 0.2),
            fts_score=fts_score,
            trigram_score=trigram_score,
        ),
    )


def _section_vector_hit_from_row(row: tuple[Any, ...]) -> SectionSearchHit:
    vector_distance = float(row[15] or 0.0)
    vector_score = float(row[16] or 0.0)
    return SectionSearchHit(
        section_node_id=row[0],
        document_version_id=row[1],
        document_id=row[2],
        record_key=row[3],
        bucket_name=row[4],
        external_record_id=row[5],
        order_date=row[6],
        title=row[7],
        section_key=row[8],
        section_type=row[9],
        section_title=row[10],
        heading_path=row[11],
        page_start=row[12],
        page_end=row[13],
        section_node_text=row[14],
        score=ScoreBreakdown(
            vector_score=vector_score,
            vector_distance=vector_distance,
        ),
    )


def _chunk_lexical_hit_from_row(row: tuple[Any, ...]) -> ChunkSearchHit:
    fts_score = float(row[19] or 0.0)
    trigram_score = float(row[20] or 0.0)
    return ChunkSearchHit(
        chunk_id=row[0],
        document_version_id=row[1],
        document_id=row[2],
        record_key=row[3],
        bucket_name=row[4],
        external_record_id=row[5],
        order_date=row[6],
        title=row[7],
        detail_url=row[8],
        pdf_url=row[9],
        chunk_index=row[10],
        page_start=row[11],
        page_end=row[12],
        section_key=row[13],
        section_type=row[14],
        section_title=row[15],
        heading_path=row[16],
        chunk_text=row[17],
        token_count=int(row[18] or 0),
        score=ScoreBreakdown(
            lexical_score=(fts_score * 0.85) + (trigram_score * 0.15),
            fts_score=fts_score,
            trigram_score=trigram_score,
        ),
    )


def _chunk_vector_hit_from_row(row: tuple[Any, ...]) -> ChunkSearchHit:
    vector_distance = float(row[19] or 0.0)
    vector_score = float(row[20] or 0.0)
    return ChunkSearchHit(
        chunk_id=row[0],
        document_version_id=row[1],
        document_id=row[2],
        record_key=row[3],
        bucket_name=row[4],
        external_record_id=row[5],
        order_date=row[6],
        title=row[7],
        detail_url=row[8],
        pdf_url=row[9],
        chunk_index=row[10],
        page_start=row[11],
        page_end=row[12],
        section_key=row[13],
        section_type=row[14],
        section_title=row[15],
        heading_path=row[16],
        chunk_text=row[17],
        token_count=int(row[18] or 0),
        score=ScoreBreakdown(
            vector_score=vector_score,
            vector_distance=vector_distance,
        ),
    )


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"
