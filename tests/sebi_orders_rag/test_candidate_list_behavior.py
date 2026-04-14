from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.schemas import ChatSessionSnapshot
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class CandidateListBehaviorTests(unittest.TestCase):
    def test_ambiguous_named_matter_returns_candidate_list_instead_of_dead_abstain(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=REAL_CONTROL_PACK,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            llm_client=_ExplodingLlmClient(),
        )

        payload = service.answer_query(
            query="prime broking company",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "clarify")
        self.assertEqual(payload.answer_status, "clarify")
        self.assertIn("multiple plausible internal SEBI matter matches", payload.answer_text)
        self.assertIn("external:30222", payload.answer_text)
        self.assertIn("external:30223", payload.answer_text)
        self.assertGreaterEqual(len(payload.clarification_candidates), 2)
        self.assertTrue(payload.debug["candidate_list_debug"]["used"])
        self.assertEqual(
            payload.debug["web_fallback_debug"]["web_fallback_not_allowed_reason"],
            "clarification_required",
        )


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("candidate-list clarifications must short-circuit before retrieval")


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


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("candidate-list clarifications should not call the LLM")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
