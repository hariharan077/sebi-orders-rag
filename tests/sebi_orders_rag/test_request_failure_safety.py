from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.schemas import ChatSessionSnapshot


class RequestFailureSafetyTests(unittest.TestCase):
    def test_top_level_failures_return_safe_abstain_payload(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            llm_client=_ExplodingLlmClient(),
        )

        payload = service.answer_query(query="Explain securities regulation", session_id=uuid4())

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertEqual(payload.answer_text, "I’m temporarily unable to answer that safely right now.")
        self.assertTrue(payload.debug["failure_safe"]["used"])
        self.assertEqual(payload.debug["failure_safe"]["exception_type"], "RuntimeError")


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("retrieval should not be used for this failure-safety test")


class _FakeSessionRepository:
    def create_session_if_missing(self, *, session_id, user_name):
        now = datetime.now(timezone.utc)
        self._snapshot = ChatSessionSnapshot(
            session_id=session_id,
            user_name=user_name,
            created_at=now,
            updated_at=now,
            state=None,
        )

    def get_session_snapshot(self, *, session_id):
        return getattr(self, "_snapshot", None)

    def get_session_state(self, *, session_id):
        return None

    def upsert_session_state(self, **kwargs):
        return None


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise RuntimeError("simulated llm failure")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
