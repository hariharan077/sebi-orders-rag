from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot
from app.sebi_orders_rag.web_fallback.models import WebSearchResult, WebSearchSource


class WebFallbackRoutingTests(unittest.TestCase):
    def test_router_sends_news_and_history_queries_to_web_routes(self) -> None:
        router = AdaptiveQueryRouter()

        self.assertEqual(
            router.decide(query="What is the latest news about SEBI").route_mode,
            "current_news_lookup",
        )
        self.assertEqual(
            router.decide(query="Who was the previous chairman of SEBI").route_mode,
            "historical_official_lookup",
        )

    def test_router_sends_income_and_fee_queries_to_current_official_lookup(self) -> None:
        router = AdaptiveQueryRouter()

        for query in (
            "What are the sources of income for SEBI",
            "What is the current commission that SEBI charges per trade",
        ):
            with self.subTest(query=query):
                decision = router.decide(query=query)
                self.assertEqual(decision.route_mode, "current_official_lookup")
                self.assertEqual(decision.query_intent, "current_official_lookup")

    def test_router_keeps_non_sebi_people_on_general_knowledge(self) -> None:
        router = AdaptiveQueryRouter()

        for query in ("who is demis hassabis", "who is larry page"):
            with self.subTest(query=query):
                decision = router.decide(query=query)
                self.assertEqual(decision.route_mode, "general_knowledge")
                self.assertEqual(decision.query_intent, "general_knowledge")

    def test_general_person_query_can_use_general_web_fallback(self) -> None:
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
            general_web_provider=_FakeGeneralWebProvider(),
            llm_client=_FakeLlmClient(
                {
                    "answer_status": "insufficient_context",
                    "answer_text": "I do not know.",
                }
            ),
        )

        payload = service.answer_query(
            query="who is demis hassabis",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "general_knowledge")
        self.assertIn("Demis Hassabis", payload.answer_text)
        self.assertEqual(payload.citations[0].source_type, "general_web")
        self.assertTrue(payload.debug["web_fallback_debug"]["general_web_attempted"])

    def test_current_public_fact_for_general_query_forces_web_verification(self) -> None:
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
            general_web_provider=_FakeCurrentFactWebProvider(),
            llm_client=_FakeLlmClient(
                {
                    "answer_status": "answered",
                    "answer_text": "Larry Page is a technology entrepreneur.",
                }
            ),
        )

        payload = service.answer_query(
            query="who is the current CEO of Alphabet",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "general_knowledge")
        self.assertIn("Sundar Pichai", payload.answer_text)
        self.assertTrue(payload.debug["web_fallback_debug"]["general_web_attempted"])
        self.assertEqual(payload.citations[0].source_type, "general_web")


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("general web fallback should not use corpus retrieval")


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _FakeSessionRepository:
    def __init__(self) -> None:
        self._snapshots = {}

    def create_session_if_missing(self, *, session_id, user_name):
        now = datetime.now(timezone.utc)
        self._snapshots.setdefault(
            session_id,
            ChatSessionSnapshot(
                session_id=session_id,
                user_name=user_name,
                created_at=now,
                updated_at=now,
                state=None,
            ),
        )

    def get_session_snapshot(self, *, session_id):
        return self._snapshots.get(session_id)

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


class _FakeGeneralWebProvider:
    def search(self, *, request):
        return WebSearchResult(
            answer_status="answered",
            answer_text="Demis Hassabis is the co-founder and CEO of Google DeepMind.",
            sources=(
                WebSearchSource(
                    source_title="Demis Hassabis",
                    source_url="https://en.wikipedia.org/wiki/Demis_Hassabis",
                    domain="wikipedia.org",
                    source_type="general_web",
                    record_key="general_web:wikipedia.org",
                ),
            ),
            provider_name="general_web_search",
            lookup_type=request.lookup_type,
        )


class _FakeCurrentFactWebProvider:
    def search(self, *, request):
        return WebSearchResult(
            answer_status="answered",
            answer_text="Sundar Pichai is the CEO of Alphabet.",
            sources=(
                WebSearchSource(
                    source_title="Sundar Pichai",
                    source_url="https://en.wikipedia.org/wiki/Sundar_Pichai",
                    domain="wikipedia.org",
                    source_type="general_web",
                    record_key="general_web:wikipedia.org",
                ),
            ),
            provider_name="general_web_search",
            lookup_type=request.lookup_type,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
