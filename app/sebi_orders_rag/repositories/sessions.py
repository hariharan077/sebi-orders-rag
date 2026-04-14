"""Chat session and persisted history helpers for the portal chat surface."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

from ..schemas import (
    Citation,
    ChatSessionListEntry,
    ChatSessionSnapshot,
    ChatSessionStateRecord,
    ChatTurnRecord,
    ClarificationCandidate,
    ClarificationContext,
)


class ChatSessionRepository:
    """Read and write chat sessions plus persisted answer-log history."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def create_session_if_missing(self, *, session_id: UUID, user_name: str | None) -> None:
        sql = """
            INSERT INTO chat_sessions (session_id, user_name)
            VALUES (%s, %s)
            ON CONFLICT (session_id) DO UPDATE
            SET
                user_name = COALESCE(EXCLUDED.user_name, chat_sessions.user_name),
                updated_at = NOW()
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (session_id, user_name))

    def upsert_session_state(
        self,
        *,
        session_id: UUID,
        active_document_ids: Sequence[int] = (),
        active_document_version_ids: Sequence[int] = (),
        active_record_keys: Sequence[str] = (),
        active_entities: Sequence[str] = (),
        active_bucket_names: Sequence[str] = (),
        active_primary_title: str | None = None,
        active_primary_entity: str | None = None,
        active_signatory_name: str | None = None,
        active_signatory_designation: str | None = None,
        active_order_date=None,
        active_order_place: str | None = None,
        active_legal_provisions: Sequence[str] = (),
        last_chunk_ids: Sequence[int] = (),
        last_citation_chunk_ids: Sequence[int] = (),
        grounded_summary: str | None = None,
        current_lookup_family: str | None = None,
        current_lookup_focus: str | None = None,
        current_lookup_query: str | None = None,
        clarification_context: ClarificationContext | None = None,
    ) -> None:
        sql = """
            INSERT INTO chat_session_state (
                session_id,
                active_document_ids,
                active_document_version_ids,
                active_record_keys,
                active_entities,
                active_bucket_names,
                active_primary_title,
                active_primary_entity,
                active_signatory_name,
                active_signatory_designation,
                active_order_date,
                active_order_place,
                active_legal_provisions,
                last_chunk_ids,
                last_citation_chunk_ids,
                grounded_summary,
                current_lookup_family,
                current_lookup_focus,
                current_lookup_query,
                clarification_context
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (session_id) DO UPDATE
            SET
                active_document_ids = EXCLUDED.active_document_ids,
                active_document_version_ids = EXCLUDED.active_document_version_ids,
                active_record_keys = EXCLUDED.active_record_keys,
                active_entities = EXCLUDED.active_entities,
                active_bucket_names = EXCLUDED.active_bucket_names,
                active_primary_title = EXCLUDED.active_primary_title,
                active_primary_entity = EXCLUDED.active_primary_entity,
                active_signatory_name = EXCLUDED.active_signatory_name,
                active_signatory_designation = EXCLUDED.active_signatory_designation,
                active_order_date = EXCLUDED.active_order_date,
                active_order_place = EXCLUDED.active_order_place,
                active_legal_provisions = EXCLUDED.active_legal_provisions,
                last_chunk_ids = EXCLUDED.last_chunk_ids,
                last_citation_chunk_ids = EXCLUDED.last_citation_chunk_ids,
                grounded_summary = EXCLUDED.grounded_summary,
                current_lookup_family = EXCLUDED.current_lookup_family,
                current_lookup_focus = EXCLUDED.current_lookup_focus,
                current_lookup_query = EXCLUDED.current_lookup_query,
                clarification_context = EXCLUDED.clarification_context,
                updated_at = NOW()
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    session_id,
                    list(active_document_ids),
                    list(active_document_version_ids),
                    list(active_record_keys),
                    list(active_entities),
                    list(active_bucket_names),
                    active_primary_title,
                    active_primary_entity,
                    active_signatory_name,
                    active_signatory_designation,
                    active_order_date,
                    active_order_place,
                    list(active_legal_provisions),
                    list(last_chunk_ids),
                    list(last_citation_chunk_ids),
                    grounded_summary,
                    current_lookup_family,
                    current_lookup_focus,
                    current_lookup_query,
                    _json_dumps(_clarification_context_to_payload(clarification_context)),
                ),
            )

    def get_session_state(self, *, session_id: UUID) -> ChatSessionStateRecord | None:
        """Return the current grounded session state if it exists."""

        sql = """
            SELECT
                session_id,
                active_document_ids,
                active_document_version_ids,
                active_record_keys,
                active_entities,
                active_bucket_names,
                active_primary_title,
                active_primary_entity,
                active_signatory_name,
                active_signatory_designation,
                active_order_date,
                active_order_place,
                active_legal_provisions,
                last_chunk_ids,
                last_citation_chunk_ids,
                grounded_summary,
                current_lookup_family,
                current_lookup_focus,
                current_lookup_query,
                clarification_context,
                updated_at
            FROM chat_session_state
            WHERE session_id = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (session_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return _session_state_from_row(row)

    def get_session_snapshot(self, *, session_id: UUID) -> ChatSessionSnapshot | None:
        """Return the chat session row and any grounded state."""

        sql = """
            SELECT
                cs.session_id,
                cs.user_name,
                cs.created_at,
                cs.updated_at,
                css.session_id,
                css.active_document_ids,
                css.active_document_version_ids,
                css.active_record_keys,
                css.active_entities,
                css.active_bucket_names,
                css.active_primary_title,
                css.active_primary_entity,
                css.active_signatory_name,
                css.active_signatory_designation,
                css.active_order_date,
                css.active_order_place,
                css.active_legal_provisions,
                css.last_chunk_ids,
                css.last_citation_chunk_ids,
                css.grounded_summary,
                css.current_lookup_family,
                css.current_lookup_focus,
                css.current_lookup_query,
                css.clarification_context,
                css.updated_at
            FROM chat_sessions cs
            LEFT JOIN chat_session_state css
                ON css.session_id = cs.session_id
            WHERE cs.session_id = %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (session_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        state = None
        if row[4] is not None:
            state = _session_state_from_row(row[4:])
        return ChatSessionSnapshot(
            session_id=row[0],
            user_name=row[1],
            created_at=row[2],
            updated_at=row[3],
            state=state,
        )

    def list_recent_sessions(self, *, limit: int = 50) -> tuple[ChatSessionListEntry, ...]:
        """Return recent sessions ordered by most recently active first."""

        sql = """
            SELECT
                cs.session_id,
                cs.created_at,
                cs.updated_at,
                first_turn.user_query,
                latest_turn.user_query,
                latest_turn.created_at,
                COALESCE(turn_counts.turn_count, 0)
            FROM chat_sessions cs
            LEFT JOIN LATERAL (
                SELECT al.user_query
                FROM answer_logs al
                WHERE al.session_id = cs.session_id
                ORDER BY al.created_at ASC, al.answer_id ASC
                LIMIT 1
            ) AS first_turn ON TRUE
            LEFT JOIN LATERAL (
                SELECT al.user_query, al.created_at
                FROM answer_logs al
                WHERE al.session_id = cs.session_id
                ORDER BY al.created_at DESC, al.answer_id DESC
                LIMIT 1
            ) AS latest_turn ON TRUE
            LEFT JOIN LATERAL (
                SELECT COUNT(*) AS turn_count
                FROM answer_logs al
                WHERE al.session_id = cs.session_id
            ) AS turn_counts ON TRUE
            ORDER BY COALESCE(latest_turn.created_at, cs.updated_at) DESC, cs.created_at DESC
            LIMIT %s
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
        return tuple(
            ChatSessionListEntry(
                session_id=row[0],
                created_at=row[1],
                updated_at=row[2],
                first_user_query=row[3],
                latest_user_query=row[4],
                last_message_at=row[5],
                turn_count=int(row[6] or 0),
            )
            for row in rows
        )

    def get_session_turns(self, *, session_id: UUID) -> tuple[ChatTurnRecord, ...]:
        """Return persisted turns for one session in chronological order."""

        sql = """
            SELECT
                answer_id,
                session_id,
                user_query,
                route_mode,
                query_intent,
                answer_text,
                answer_confidence,
                cited_chunk_ids,
                citation_payload,
                created_at
            FROM answer_logs
            WHERE session_id = %s
            ORDER BY created_at ASC, answer_id ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (session_id,))
            rows = cursor.fetchall()
        if not rows:
            return ()

        citation_lookup = self._get_citation_lookup(
            chunk_ids=[
                chunk_id
                for row in rows
                for chunk_id in _int_tuple(row[7])
            ]
        )

        turns: list[ChatTurnRecord] = []
        for row in rows:
            cited_chunk_ids = _int_tuple(row[7])
            citations = _decode_citations(row[8])
            if not citations:
                citations = tuple(
                    Citation(
                        citation_number=index,
                        record_key=citation["record_key"],
                        title=citation["title"],
                        page_start=citation["page_start"],
                        page_end=citation["page_end"],
                        section_type=citation["section_type"],
                        document_version_id=citation["document_version_id"],
                        chunk_id=chunk_id,
                        detail_url=citation["detail_url"],
                        pdf_url=citation["pdf_url"],
                        source_title=citation["title"],
                    )
                    for index, chunk_id in enumerate(cited_chunk_ids, start=1)
                    if (citation := citation_lookup.get(chunk_id)) is not None
                )
            turns.append(
                ChatTurnRecord(
                    answer_id=int(row[0]),
                    session_id=row[1],
                    user_query=str(row[2] or ""),
                    route_mode=str(row[3] or "abstain"),
                    query_intent=row[4],
                    answer_text=str(row[5] or ""),
                    answer_confidence=_float_or_zero(row[6]),
                    citations=citations,
                    created_at=_optional_datetime(row[9]),
                )
            )
        return tuple(turns)

    def _get_citation_lookup(self, *, chunk_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        if not chunk_ids:
            return {}

        sql = """
            SELECT
                dc.chunk_id,
                sd.record_key,
                dv.title,
                dc.page_start,
                dc.page_end,
                dc.section_type,
                dc.document_version_id,
                dv.detail_url,
                dv.pdf_url
            FROM document_chunks dc
            INNER JOIN document_versions dv
                ON dv.document_version_id = dc.document_version_id
            INNER JOIN source_documents sd
                ON sd.document_id = dv.document_id
            WHERE dc.chunk_id = ANY(%s)
        """
        unique_chunk_ids = list(dict.fromkeys(int(chunk_id) for chunk_id in chunk_ids))
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (unique_chunk_ids,))
            rows = cursor.fetchall()
        return {
            int(row[0]): {
                "record_key": str(row[1]),
                "title": str(row[2]),
                "page_start": int(row[3]),
                "page_end": int(row[4]),
                "section_type": str(row[5]),
                "document_version_id": int(row[6]),
                "detail_url": row[7],
                "pdf_url": row[8],
            }
            for row in rows
        }


def _session_state_from_row(row: tuple[Any, ...]) -> ChatSessionStateRecord:
    return ChatSessionStateRecord(
        session_id=row[0],
        active_document_ids=_int_tuple(row[1]),
        active_document_version_ids=_int_tuple(row[2]),
        active_record_keys=_str_tuple(row[3]),
        active_entities=_str_tuple(row[4]),
        active_bucket_names=_str_tuple(row[5]),
        active_primary_title=row[6],
        active_primary_entity=row[7],
        active_signatory_name=row[8],
        active_signatory_designation=row[9],
        active_order_date=row[10],
        active_order_place=row[11],
        active_legal_provisions=_str_tuple(row[12]),
        last_chunk_ids=_int_tuple(row[13]),
        last_citation_chunk_ids=_int_tuple(row[14]),
        grounded_summary=row[15],
        current_lookup_family=row[16],
        current_lookup_focus=row[17],
        current_lookup_query=row[18],
        clarification_context=_clarification_context_from_payload(row[19]),
        updated_at=_optional_datetime(row[20]),
    )


def _clarification_context_to_payload(
    context: ClarificationContext | None,
) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "source_query": context.source_query,
        "source_route_mode": context.source_route_mode,
        "source_query_intent": context.source_query_intent,
        "candidate_type": context.candidate_type,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.candidate_index,
                "candidate_type": candidate.candidate_type,
                "title": candidate.title,
                "record_key": candidate.record_key,
                "bucket_name": candidate.bucket_name,
                "order_date": (
                    candidate.order_date.isoformat()
                    if candidate.order_date is not None
                    else None
                ),
                "document_version_id": candidate.document_version_id,
                "descriptor": candidate.descriptor,
                "resolution_query": candidate.resolution_query,
                "canonical_person_id": candidate.canonical_person_id,
                "selection_aliases": list(candidate.selection_aliases),
            }
            for candidate in context.candidates
        ],
    }


def _clarification_context_from_payload(raw_value: Any) -> ClarificationContext | None:
    if raw_value in (None, "", {}):
        return None
    payload = raw_value
    if isinstance(raw_value, str):
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None

    candidate_rows = payload.get("candidates")
    if not isinstance(candidate_rows, list):
        return None

    candidates: list[ClarificationCandidate] = []
    for item in candidate_rows:
        if not isinstance(item, dict):
            continue
        candidates.append(
            ClarificationCandidate(
                candidate_id=str(item.get("candidate_id") or ""),
                candidate_index=int(item.get("candidate_index") or len(candidates) + 1),
                candidate_type=str(item.get("candidate_type") or "matter"),
                title=str(item.get("title") or ""),
                record_key=(str(item["record_key"]) if item.get("record_key") else None),
                bucket_name=(str(item["bucket_name"]) if item.get("bucket_name") else None),
                order_date=_optional_date(item.get("order_date")),
                document_version_id=_optional_int(item.get("document_version_id")),
                descriptor=(str(item["descriptor"]) if item.get("descriptor") else None),
                resolution_query=(
                    str(item["resolution_query"]) if item.get("resolution_query") else None
                ),
                canonical_person_id=(
                    str(item["canonical_person_id"])
                    if item.get("canonical_person_id")
                    else None
                ),
                selection_aliases=_str_tuple(item.get("selection_aliases")),
            )
        )
    if not candidates:
        return None
    return ClarificationContext(
        source_query=str(payload.get("source_query") or ""),
        source_route_mode=str(payload.get("source_route_mode") or "clarify"),
        source_query_intent=str(payload.get("source_query_intent") or "ambiguous"),
        candidate_type=str(payload.get("candidate_type") or candidates[0].candidate_type),
        candidates=tuple(candidates),
    )


def _int_tuple(values: Sequence[Any] | None) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(int(value) for value in values)


def _str_tuple(values: Sequence[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


def _optional_date(value: Any):
    if value in (None, ""):
        return None
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value
    try:
        from datetime import date

        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _decode_citations(raw_value: Any) -> tuple[Citation, ...]:
    if raw_value in (None, "", []):
        return ()
    payload = raw_value
    if isinstance(raw_value, str):
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return ()
    if not isinstance(payload, list):
        return ()
    citations: list[Citation] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        citations.append(
            Citation(
                citation_number=int(item.get("citation_number", len(citations) + 1)),
                record_key=str(item.get("record_key") or ""),
                title=str(item.get("title") or ""),
                page_start=_optional_int(item.get("page_start")),
                page_end=_optional_int(item.get("page_end")),
                section_type=(str(item["section_type"]) if item.get("section_type") is not None else None),
                document_version_id=_optional_int(item.get("document_version_id")),
                chunk_id=_optional_int(item.get("chunk_id")),
                detail_url=(str(item["detail_url"]) if item.get("detail_url") else None),
                pdf_url=(str(item["pdf_url"]) if item.get("pdf_url") else None),
                source_url=(str(item["source_url"]) if item.get("source_url") else None),
                source_title=(str(item["source_title"]) if item.get("source_title") else None),
                domain=(str(item["domain"]) if item.get("domain") else None),
                source_type=(str(item["source_type"]) if item.get("source_type") else None),
                snippet=(str(item["snippet"]) if item.get("snippet") else None),
            )
        )
    return tuple(citations)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return value


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)
