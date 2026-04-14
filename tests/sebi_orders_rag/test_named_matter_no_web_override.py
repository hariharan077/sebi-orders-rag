from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.retrieval.scoring import HierarchicalSearchResult
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class NamedMatterNoWebOverrideTests(unittest.TestCase):
    def test_named_order_query_stays_corpus_first(self) -> None:
        decision = AdaptiveQueryRouter(control_pack=load_control_pack(REAL_CONTROL_PACK)).decide(
            query="tell me more about the IPO of Vishvaraj Environment Limited"
        )

        self.assertEqual(decision.route_mode, "exact_lookup")
        self.assertTrue(decision.analysis.strict_single_matter)

    def test_named_order_query_does_not_call_web_when_internal_support_is_weak(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=REAL_CONTROL_PACK,
            ),
            connection=_FakeConnection(),
            search_service=_EmptySearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            general_web_provider=_ExplodingWebProvider(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="tell me more about the IPO of Vishvaraj Environment Limited",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertFalse(payload.debug["web_fallback_debug"]["general_web_attempted"])
        self.assertEqual(
            payload.debug["web_fallback_debug"]["web_fallback_not_allowed_reason"],
            "named_matter_no_web_override",
        )


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


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


class _ExplodingWebProvider:
    def search(self, *, request):
        raise AssertionError("named matter corpus routes must not call web fallback")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
