from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.schemas import (
    ChatAnswerPayload,
    ChatSessionListEntry,
    ChatSessionSnapshot,
    ChatSessionStateRecord,
    ChatTurnRecord,
    Citation,
)
from app.services.sebi_orders_rag_service import SebiOrdersRagService


class PortalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SebiOrdersRagSettings(
            db_dsn="postgresql://unused",
            data_root=Path(".").resolve(),
        )
        self.connection = _FakeConnection()
        self.schema_calls = 0

    def test_query_maps_phase4_payload_to_portal_result(self) -> None:
        session_id = uuid4()
        citation = Citation(
            citation_number=1,
            record_key="external:100725",
            title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
            page_start=1,
            page_end=2,
            section_type="operative_order",
            document_version_id=101,
            chunk_id=11,
            detail_url="https://example.com/detail/100725",
            pdf_url="https://example.com/pdf/100725.pdf",
        )
        service = SebiOrdersRagService(
            settings_loader=lambda: self.settings,
            connection_factory=lambda settings: _ConnectionManager(self.connection),
            schema_initializer=self._schema_initializer,
            answer_service_factory=lambda settings, connection: _FakeAnswerService(
                ChatAnswerPayload(
                    session_id=session_id,
                    answer_text="The appeal was dismissed.",
                    route_mode="exact_lookup",
                    query_intent="document_lookup",
                    confidence=0.88,
                    citations=(citation,),
                    active_record_keys=("external:100725",),
                )
            ),
        )

        result = service.query(
            message="Appeal No. 6798 of 2026 filed by Hariom Yadav",
            session_id=session_id,
        )

        self.assertEqual(self.schema_calls, 1)
        self.assertEqual(result.session_id, session_id)
        self.assertEqual(result.route_mode, "exact_lookup")
        self.assertEqual(result.query_intent, "document_lookup")
        self.assertEqual(result.citations[0].record_key, "external:100725")
        self.assertEqual(result.citations[0].title_url, "https://example.com/detail/100725")
        self.assertEqual(result.citations[0].page_url, "https://example.com/pdf/100725.pdf#page=1")
        self.assertEqual(result.active_record_keys, ("external:100725",))

    def test_create_session_returns_portal_safe_snapshot(self) -> None:
        session_id = uuid4()
        repository = _FakeSessionRepository(session_id=session_id)
        service = SebiOrdersRagService(
            settings_loader=lambda: self.settings,
            connection_factory=lambda settings: _ConnectionManager(self.connection),
            schema_initializer=self._schema_initializer,
            session_repository_factory=lambda connection: repository,
        )

        session = service.create_session(session_id=session_id, user_name="portal-user")

        self.assertEqual(self.schema_calls, 1)
        self.assertEqual(repository.created_user_name, "portal-user")
        self.assertEqual(session.session_id, session_id)
        self.assertEqual(session.state.active_record_keys, ("external:100725",))
        self.assertEqual(session.state.active_document_ids, (201,))

    def test_list_sessions_returns_sidebar_safe_summaries(self) -> None:
        session_id = uuid4()
        repository = _FakeSessionRepository(session_id=session_id)
        service = SebiOrdersRagService(
            settings_loader=lambda: self.settings,
            connection_factory=lambda settings: _ConnectionManager(self.connection),
            schema_initializer=self._schema_initializer,
            session_repository_factory=lambda connection: repository,
        )

        sessions = service.list_sessions()

        self.assertEqual(self.schema_calls, 1)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].session_id, session_id)
        self.assertEqual(sessions[0].title, "Appeal No. 6798 of 2026 filed by Hariom Yadav")
        self.assertEqual(sessions[0].turn_count, 1)

    def test_get_session_history_returns_reconstructed_turns(self) -> None:
        session_id = uuid4()
        repository = _FakeSessionRepository(session_id=session_id)
        service = SebiOrdersRagService(
            settings_loader=lambda: self.settings,
            connection_factory=lambda settings: _ConnectionManager(self.connection),
            schema_initializer=self._schema_initializer,
            session_repository_factory=lambda connection: repository,
        )

        history = service.get_session_history(session_id=session_id)

        self.assertEqual(self.schema_calls, 1)
        self.assertIsNotNone(history)
        assert history is not None
        self.assertEqual(history.session.session_id, session_id)
        self.assertEqual(history.turns[0].user_message, "Appeal No. 6798 of 2026 filed by Hariom Yadav")
        self.assertEqual(history.turns[0].assistant_message, "The appeal was dismissed.")
        self.assertEqual(history.turns[0].citations[0].record_key, "external:100725")

    def _schema_initializer(self, connection, settings) -> None:
        self.schema_calls += 1


class _FakeConnection:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class _ConnectionManager:
    def __init__(self, connection) -> None:
        self._connection = connection

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeAnswerService:
    def __init__(self, payload: ChatAnswerPayload) -> None:
        self._payload = payload

    def answer_query(self, *, query: str, session_id):
        return self._payload


class _FakeSessionRepository:
    def __init__(self, *, session_id) -> None:
        now = datetime(2026, 4, 10, 10, 30, tzinfo=timezone.utc)
        self.created_user_name = None
        self.snapshot = ChatSessionSnapshot(
            session_id=session_id,
            user_name="portal-user",
            created_at=now,
            updated_at=now,
            state=ChatSessionStateRecord(
                session_id=session_id,
                active_document_ids=(201,),
                active_record_keys=("external:100725",),
            ),
        )
        citation = Citation(
            citation_number=1,
            record_key="external:100725",
            title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
            page_start=1,
            page_end=2,
            section_type="operative_order",
            document_version_id=101,
            chunk_id=11,
            detail_url="https://example.com/detail/100725",
            pdf_url="https://example.com/pdf/100725.pdf",
        )
        self.entries = (
            ChatSessionListEntry(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                first_user_query="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                latest_user_query="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                last_message_at=now,
                turn_count=1,
            ),
        )
        self.turns = (
            ChatTurnRecord(
                answer_id=1,
                session_id=session_id,
                user_query="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                route_mode="exact_lookup",
                query_intent="document_lookup",
                answer_text="The appeal was dismissed.",
                answer_confidence=0.88,
                citations=(citation,),
                created_at=now,
            ),
        )

    def create_session_if_missing(self, *, session_id, user_name):
        self.created_user_name = user_name

    def get_session_snapshot(self, *, session_id):
        return self.snapshot

    def list_recent_sessions(self, *, limit=50):
        return self.entries[:limit]

    def get_session_turns(self, *, session_id):
        return self.turns


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
