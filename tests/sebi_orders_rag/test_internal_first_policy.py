from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.provider import CurrentInfoResult, CurrentInfoSource
from app.sebi_orders_rag.retrieval.scoring import HierarchicalSearchResult
from app.sebi_orders_rag.schemas import ChatSessionSnapshot


class InternalFirstPolicyTests(unittest.TestCase):
    def test_structured_answer_does_not_fall_through_to_corpus_or_web(self) -> None:
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
            current_info_provider=_AnsweredCurrentInfoProvider(),
            current_news_provider=_ExplodingCurrentInfoProvider(),
            historical_info_provider=_ExplodingCurrentInfoProvider(),
            general_web_provider=_ExplodingWebProvider(),
            llm_client=_ExplodingLlmClient(),
        )

        payload = service.answer_query(
            query="Who is the chairperson of SEBI?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "structured_current_info")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("Tuhin Kanta Pandey", payload.answer_text)
        self.assertTrue(payload.debug["web_fallback_debug"]["structured_attempted"])
        self.assertFalse(payload.debug["web_fallback_debug"]["corpus_attempted"])
        self.assertFalse(payload.debug["web_fallback_debug"]["official_web_attempted"])
        self.assertFalse(payload.debug["web_fallback_debug"]["general_web_attempted"])

    def test_corpus_route_does_not_consult_current_or_general_web_layers(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
            ),
            connection=_FakeConnection(),
            search_service=_EmptySearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            current_info_provider=_ExplodingCurrentInfoProvider(),
            current_news_provider=_ExplodingCurrentInfoProvider(),
            historical_info_provider=_ExplodingCurrentInfoProvider(),
            general_web_provider=_ExplodingWebProvider(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="What was the settlement amount in the JP Morgan settlement order?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertTrue(payload.debug["web_fallback_debug"]["corpus_attempted"])
        self.assertFalse(payload.debug["web_fallback_debug"]["official_web_attempted"])
        self.assertFalse(payload.debug["web_fallback_debug"]["general_web_attempted"])


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("structured answers must not use corpus retrieval")


class _EmptySearchService:
    def search(self, **kwargs):
        return HierarchicalSearchResult(
            query=str(kwargs.get("query") or ""),
            documents=(),
            sections=(),
            chunks=(),
            debug={},
        )


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


class _AnsweredCurrentInfoProvider:
    def lookup(self, *, query: str, session_state=None):
        return CurrentInfoResult(
            answer_status="answered",
            answer_text="SEBI's Chairperson is Tuhin Kanta Pandey.",
            sources=(
                CurrentInfoSource(
                    title="SEBI Directory",
                    url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    record_key="official:directory",
                    domain="sebi.gov.in",
                    source_type="structured",
                ),
            ),
            confidence=0.97,
            provider_name="structured_directory",
            lookup_type="chairperson",
            debug={"structured_attempted": True},
        )


class _ExplodingCurrentInfoProvider:
    def lookup(self, *, query: str, session_state=None):
        raise AssertionError("this route must not consult this provider")


class _ExplodingWebProvider:
    def search(self, *, request):
        raise AssertionError("internal-first routes must not use general web fallback")


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("structured current-info answers should not hit the LLM")


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
