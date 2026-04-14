from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.history_lookup import HistoricalOfficialLookupProvider
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter


class HistoricalPersonRoutingTests(unittest.TestCase):
    def test_router_uses_historical_official_route(self) -> None:
        decision = AdaptiveQueryRouter().decide(query="Who was the previous chairman of SEBI")

        self.assertEqual(decision.route_mode, "historical_official_lookup")
        self.assertEqual(decision.query_intent, "historical_official_lookup")
        self.assertTrue(decision.analysis.appears_historical_official_lookup)

    def test_historical_lookup_does_not_return_current_chairperson(self) -> None:
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
            historical_info_provider=HistoricalOfficialLookupProvider(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="Who was the previous chairman of SEBI",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "historical_official_lookup")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("Madhabi Puri Buch", payload.answer_text)
        self.assertNotIn("Tuhin Kanta Pandey", payload.answer_text)
        self.assertTrue(payload.debug["historical_lookup_debug"]["used"])


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("historical official lookup must not use corpus retrieval")


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()

    def fetch_order_metadata(self, *, document_version_ids):
        return ()

    def fetch_legal_provisions(self, *, document_version_ids):
        return ()


class _FakeSessionRepository:
    def create_session_if_missing(self, *, session_id, user_name):
        from datetime import datetime, timezone
        from app.sebi_orders_rag.schemas import ChatSessionSnapshot

        self._snapshot = ChatSessionSnapshot(
            session_id=session_id,
            user_name=user_name,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
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


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
