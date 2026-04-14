from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.provider import CurrentInfoResult
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord


class StructuredInfoRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = AdaptiveQueryRouter()

    def test_validation_queries_route_to_structured_current_info(self) -> None:
        queries = (
            "who is chitra",
            "chitra bhandari sebi",
            "rajudeen",
            "what is the designation of lenin",
            "who is tuhin",
            "Whose staff id is 1668",
            "how many assistant managers are there in sebi",
            "how many assistant managers are currently serving in sebi",
            "who are the ed",
            "how many wtm are there",
            "who are the board members of sebi",
        )

        for query in queries:
            with self.subTest(query=query):
                decision = self.router.decide(query=query)
                self.assertEqual(decision.route_mode, "structured_current_info")
                self.assertEqual(decision.query_intent, "structured_current_info")

    def test_structured_count_queries_do_not_fall_through_to_orders_corpus(self) -> None:
        service = _build_answer_service(current_info_provider=_NoMatchStructuredProvider())

        payload = service.answer_query(
            query="how many assistant managers are there in sebi",
            session_id=uuid4(),
        )

        self.assertEqual(payload.query_intent, "structured_current_info")
        self.assertEqual(payload.route_mode, "abstain")
        self.assertIn("No matching current directory entry", payload.answer_text)

    def test_judgment_titles_with_city_names_do_not_route_to_structured_current_info(self) -> None:
        decision = self.router.decide(
            query=(
                "Judgment dated 08.10.2024 passed by the Hon'ble SEBI Special Court, Delhi "
                "in SEBI vs Kisley Plantation Limited & Ors."
            )
        )

        self.assertNotEqual(decision.route_mode, "structured_current_info")


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeSessionRepository:
    def __init__(self) -> None:
        self.snapshots = {}
        self.states = {}

    def create_session_if_missing(self, *, session_id, user_name):
        now = datetime.now(timezone.utc)
        self.snapshots.setdefault(
            session_id,
            ChatSessionSnapshot(
                session_id=session_id,
                user_name=user_name,
                created_at=now,
                updated_at=now,
                state=self.states.get(session_id),
            ),
        )

    def get_session_snapshot(self, *, session_id):
        return self.snapshots.get(session_id)

    def get_session_state(self, *, session_id):
        return self.states.get(session_id)

    def upsert_session_state(self, **kwargs) -> None:
        self.states[kwargs["session_id"]] = ChatSessionStateRecord(
            session_id=kwargs["session_id"],
            active_document_ids=tuple(kwargs["active_document_ids"]),
            active_record_keys=tuple(kwargs["active_record_keys"]),
            active_entities=tuple(kwargs["active_entities"]),
            active_bucket_names=tuple(kwargs["active_bucket_names"]),
            last_chunk_ids=tuple(kwargs["last_chunk_ids"]),
            last_citation_chunk_ids=tuple(kwargs["last_citation_chunk_ids"]),
            grounded_summary=kwargs["grounded_summary"],
            current_lookup_family=kwargs.get("current_lookup_family"),
            current_lookup_focus=kwargs.get("current_lookup_focus"),
            current_lookup_query=kwargs.get("current_lookup_query"),
        )


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("corpus retrieval should not be used for structured current-info routing")


class _FakeLlmClient:
    def complete_json(self, prompt):
        return {}


class _NoMatchStructuredProvider:
    def lookup(self, *, query: str, session_state=None) -> CurrentInfoResult:
        lookup_type = "designation_count" if "assistant manager" in query.lower() else "person_lookup"
        return CurrentInfoResult(
            answer_status="insufficient_context",
            answer_text="No matching current directory entry was found in the ingested official SEBI data.",
            provider_name="canonical_structured_info",
            lookup_type=lookup_type,
            debug={"detected_query_family": lookup_type},
        )


def _build_answer_service(*, current_info_provider) -> AdaptiveRagAnswerService:
    settings = SebiOrdersRagSettings(
        db_dsn="postgresql://unused",
        data_root=Path(".").resolve(),
        low_confidence_threshold=0.35,
        enable_memory=True,
    )
    return AdaptiveRagAnswerService(
        settings=settings,
        connection=_FakeConnection(),
        search_service=_ExplodingSearchService(),
        retrieval_repository=_FakeRetrievalRepository(),
        session_repository=_FakeSessionRepository(),
        answer_repository=_FakeAnswerRepository(),
        current_info_provider=current_info_provider,
        llm_client=_FakeLlmClient(),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
