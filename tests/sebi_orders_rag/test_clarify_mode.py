from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.provider import CurrentInfoResult
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord


class ClarifyModeTests(unittest.TestCase):
    def test_ambiguous_people_query_returns_clarify_not_abstain(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            current_info_provider=_AmbiguousPeopleProvider(),
            llm_client=_ExplodingLlmClient(),
        )

        payload = service.answer_query(query="who is chitra", session_id=uuid4())

        self.assertEqual(payload.route_mode, "clarify")
        self.assertEqual(payload.answer_status, "clarify")
        self.assertEqual(payload.query_intent, "structured_current_info")
        self.assertEqual(len(payload.clarification_candidates), 2)
        self.assertEqual(payload.clarification_candidates[0].candidate_type, "person")
        self.assertIn("Please choose", payload.answer_text)
        self.assertNotEqual(payload.route_mode, "abstain")
        self.assertTrue(payload.debug["clarification_debug"]["used"])


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("clarify should short-circuit before retrieval")


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _FakeSessionRepository:
    def __init__(self) -> None:
        self._states: dict[object, ChatSessionStateRecord] = {}
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
                state=self._states.get(session_id),
            ),
        )

    def get_session_snapshot(self, *, session_id):
        snapshot = self._snapshots.get(session_id)
        if snapshot is None:
            return None
        return ChatSessionSnapshot(
            session_id=snapshot.session_id,
            user_name=snapshot.user_name,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            state=self._states.get(session_id),
        )

    def get_session_state(self, *, session_id):
        return self._states.get(session_id)

    def upsert_session_state(self, **kwargs):
        self._states[kwargs["session_id"]] = ChatSessionStateRecord(
            session_id=kwargs["session_id"],
            active_document_ids=tuple(kwargs.get("active_document_ids", ())),
            active_document_version_ids=tuple(kwargs.get("active_document_version_ids", ())),
            active_record_keys=tuple(kwargs.get("active_record_keys", ())),
            active_entities=tuple(kwargs.get("active_entities", ())),
            active_bucket_names=tuple(kwargs.get("active_bucket_names", ())),
            active_primary_title=kwargs.get("active_primary_title"),
            active_primary_entity=kwargs.get("active_primary_entity"),
            active_signatory_name=kwargs.get("active_signatory_name"),
            active_signatory_designation=kwargs.get("active_signatory_designation"),
            active_order_date=kwargs.get("active_order_date"),
            active_order_place=kwargs.get("active_order_place"),
            active_legal_provisions=tuple(kwargs.get("active_legal_provisions", ())),
            last_chunk_ids=tuple(kwargs.get("last_chunk_ids", ())),
            last_citation_chunk_ids=tuple(kwargs.get("last_citation_chunk_ids", ())),
            grounded_summary=kwargs.get("grounded_summary"),
            current_lookup_family=kwargs.get("current_lookup_family"),
            current_lookup_focus=kwargs.get("current_lookup_focus"),
            current_lookup_query=kwargs.get("current_lookup_query"),
            clarification_context=kwargs.get("clarification_context"),
        )


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("clarify should not call the LLM")


class _AmbiguousPeopleProvider:
    def lookup(self, *, query: str, session_state=None) -> CurrentInfoResult:
        return CurrentInfoResult(
            answer_status="insufficient_context",
            answer_text="I found multiple current SEBI people named Chitra. Please choose one.",
            provider_name="structured_directory",
            lookup_type="person_lookup",
            debug={
                "detected_query_family": "person_lookup",
                "fallback_reason": "person_match_clarify",
                "extracted_person_name": "chitra",
                "matched_people": [
                    {
                        "name": "Chitra Ramkrishna",
                        "canonical_person_id": "person:chitra-1",
                        "designation": "Whole Time Member",
                        "department_name": "Integrated Surveillance Department",
                    },
                    {
                        "name": "Chitra Sharma",
                        "canonical_person_id": "person:chitra-2",
                        "designation": "Assistant Manager",
                        "department_name": "Investment Management Department",
                    },
                ],
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
