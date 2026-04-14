from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.provider import CurrentInfoResult
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter


class InstitutionalFactsRoutingTests(unittest.TestCase):
    def test_router_uses_current_official_lookup_for_institutional_facts(self) -> None:
        router = AdaptiveQueryRouter()

        for query in (
            "What are the sources of income for SEBI",
            "What is the current commission that SEBI charges per trade",
        ):
            with self.subTest(query=query):
                decision = router.decide(query=query)
                self.assertEqual(decision.route_mode, "current_official_lookup")
                self.assertEqual(decision.query_intent, "current_official_lookup")

    def test_answer_service_keeps_institutional_facts_off_the_orders_corpus(self) -> None:
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
            current_info_provider=_InstitutionalFactsProvider(),
            llm_client=_FakeLlmClient(),
        )

        payload = service.answer_query(
            query="What are the sources of income for SEBI",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "current_official_lookup")
        self.assertEqual(payload.answer_status, "answered")
        self.assertTrue(payload.debug["current_lookup_debug"]["used"])
        self.assertEqual(payload.debug["current_lookup_debug"]["lookup_type"], "sebi_income_sources")


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
        raise AssertionError("institutional facts must not use corpus retrieval")


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
    def complete_json(self, prompt):
        return {}


class _InstitutionalFactsProvider:
    def lookup(self, *, query: str, session_state=None) -> CurrentInfoResult:
        return CurrentInfoResult(
            answer_status="answered",
            answer_text="Official sources indicate that SEBI's income comes from fees, charges, and investments of its fund.",
            confidence=0.81,
            provider_name="official_web",
            lookup_type="sebi_income_sources",
            debug={"answer_origin": "institutional_facts_official_web_search"},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
