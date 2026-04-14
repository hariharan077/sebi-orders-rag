from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.news_lookup import CurrentNewsLookupProvider
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.web_fallback.models import WebSearchResult


class CurrentNewsRoutingTests(unittest.TestCase):
    def test_router_uses_current_news_route(self) -> None:
        decision = AdaptiveQueryRouter().decide(query="What is the latest news about SEBI")

        self.assertEqual(decision.route_mode, "current_news_lookup")
        self.assertEqual(decision.query_intent, "current_news_lookup")
        self.assertTrue(decision.analysis.appears_current_news_lookup)

    def test_answer_service_does_not_use_corpus_for_current_news_query(self) -> None:
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
            current_news_provider=CurrentNewsLookupProvider(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="What is the latest news about SEBI",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "current_news_lookup")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertEqual(payload.query_intent, "current_news_lookup")
        self.assertIn("Current news lookup is not available", payload.answer_text)
        self.assertTrue(payload.debug["news_lookup_debug"]["used"])

    def test_current_news_provider_returns_insufficient_result_when_official_search_is_weak(self) -> None:
        provider = CurrentNewsLookupProvider(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
            ),
            official_search_provider=_WeakOfficialNewsProvider(),
        )

        result = provider.lookup(query="What is the latest news about SEBI")

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertIn("clearly relevant recent official item", result.answer_text)


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("current news must not use corpus retrieval")


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


class _WeakOfficialNewsProvider:
    def search(self, *, request):
        return WebSearchResult(
            answer_status="insufficient_context",
            answer_text="I could not find a clearly relevant recent official item.",
            sources=(),
            provider_name="official_web_search",
            lookup_type=request.lookup_type,
            debug={"official_web_attempted": True},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
