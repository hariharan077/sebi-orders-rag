from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.provider import CurrentInfoResult, CurrentInfoSource
from app.sebi_orders_rag.schemas import ChatSessionSnapshot


class InternalPersonPriorityTests(unittest.TestCase):
    def test_general_route_is_overridden_by_internal_person_priority(self) -> None:
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
            current_info_provider=_PreviewCurrentInfoProvider(),
            current_news_provider=_ExplodingCurrentInfoProvider(),
            historical_info_provider=_ExplodingCurrentInfoProvider(),
            general_web_provider=_ExplodingWebProvider(),
            llm_client=_ExplodingLlmClient(),
        )

        payload = service.answer_query(
            query="who is dron amrit",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "structured_current_info")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("Dron Kumar Amrit", payload.answer_text)
        self.assertTrue(payload.debug["internal_person_priority_debug"]["used"])
        self.assertFalse(payload.debug["web_fallback_debug"]["general_web_attempted"])
        self.assertEqual(
            payload.debug["web_fallback_debug"]["web_fallback_not_allowed_reason"],
            "internal_person_priority_override",
        )


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("internal person priority should bypass corpus retrieval")


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


class _PreviewCurrentInfoProvider:
    def preview_internal_person_priority(self, *, query: str, session_state=None):
        if query.lower().strip() != "who is dron amrit":
            return None
        return CurrentInfoResult(
            answer_status="answered",
            answer_text="Dron Kumar Amrit is listed as Deputy General Manager in Investigation Department (IVD).",
            sources=(
                CurrentInfoSource(
                    title="SEBI Directory",
                    url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    record_key="official:directory",
                    domain="sebi.gov.in",
                    source_type="structured",
                ),
            ),
            confidence=0.93,
            provider_name="canonical_structured_info",
            lookup_type="person_lookup",
            debug={
                "structured_attempted": True,
                "matched_people_rows_count": 1,
                "person_match_status": "exact_alias",
                "fuzzy_band": None,
            },
        )

    def lookup(self, *, query: str, session_state=None):
        raise AssertionError("preview should have short-circuited the general route")


class _ExplodingCurrentInfoProvider:
    def lookup(self, *, query: str, session_state=None):
        raise AssertionError("this provider should not be used")


class _ExplodingWebProvider:
    def search(self, *, request):
        raise AssertionError("general web fallback must be blocked by internal person priority")


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("LLM should not run when internal person priority resolves the query")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
