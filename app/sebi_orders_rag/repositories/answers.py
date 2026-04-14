"""Answer and audit logging helpers for Phase 4 chat interactions."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from ..constants import LEGACY_RETRIEVAL_LOG_MODE_BY_ROUTE
from ..schemas import Citation


class AnswerRepository:
    """Write Phase 4 retrieval and answer audit rows."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def insert_retrieval_log(
        self,
        *,
        session_id: UUID,
        user_query: str,
        route_mode: str,
        query_intent: str,
        extracted_filters: dict[str, Any] | None,
        retrieved_chunk_ids: Sequence[int] = (),
        reranked_chunk_ids: Sequence[int] = (),
        final_citation_chunk_ids: Sequence[int] = (),
        confidence: float | None = None,
        answer_status: str | None = None,
        cited_record_keys: Sequence[str] = (),
    ) -> int:
        """Insert a retrieval log row and return its id."""

        sql = """
            INSERT INTO retrieval_logs (
                session_id,
                user_query,
                router_mode,
                extracted_filters,
                retrieved_chunk_ids,
                reranked_chunk_ids,
                final_citation_chunk_ids,
                confidence_score,
                query_intent,
                route_mode,
                answer_confidence,
                answer_status,
                cited_record_keys
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING retrieval_id
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    session_id,
                    user_query,
                    LEGACY_RETRIEVAL_LOG_MODE_BY_ROUTE.get(route_mode, "abstain"),
                    _json_dumps(extracted_filters or {}),
                    list(retrieved_chunk_ids),
                    list(reranked_chunk_ids),
                    list(final_citation_chunk_ids),
                    _decimal_or_none(confidence),
                    query_intent,
                    route_mode,
                    _decimal_or_none(confidence),
                    answer_status,
                    list(cited_record_keys),
                ),
            )
            row = cursor.fetchone()
        return int(row[0])

    def insert_answer_log(
        self,
        *,
        session_id: UUID,
        user_query: str,
        route_mode: str,
        query_intent: str,
        answer_text: str,
        answer_confidence: float | None,
        cited_chunk_ids: Sequence[int] = (),
        cited_record_keys: Sequence[str] = (),
        citations: Sequence[Citation] = (),
    ) -> int:
        """Insert an answer log row and return its id."""

        sql = """
            INSERT INTO answer_logs (
                session_id,
                user_query,
                route_mode,
                query_intent,
                answer_text,
                answer_confidence,
                cited_chunk_ids,
                cited_record_keys,
                citation_payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING answer_id
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    session_id,
                    user_query,
                    route_mode,
                    query_intent,
                    answer_text,
                    _decimal_or_none(answer_confidence),
                    list(cited_chunk_ids),
                    list(cited_record_keys),
                    _json_dumps(
                        [
                            {
                                "citation_number": citation.citation_number,
                                "record_key": citation.record_key,
                                "title": citation.title,
                                "page_start": citation.page_start,
                                "page_end": citation.page_end,
                                "section_type": citation.section_type,
                                "document_version_id": citation.document_version_id,
                                "chunk_id": citation.chunk_id,
                                "detail_url": citation.detail_url,
                                "pdf_url": citation.pdf_url,
                                "source_url": citation.source_url,
                                "source_title": citation.source_title,
                                "domain": citation.domain,
                                "source_type": citation.source_type,
                                "snippet": citation.snippet,
                            }
                            for citation in citations
                        ]
                    ),
                ),
            )
            row = cursor.fetchone()
        return int(row[0])


def _decimal_or_none(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(f"{value:.4f}")


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)
