from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.metadata.service import MetadataAnswer
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord, Citation


class ActiveMatterFollowUpsTests(unittest.TestCase):
    def test_router_keeps_quasi_judicial_authority_inside_active_matter(self) -> None:
        router = AdaptiveQueryRouter()
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_record_keys=("external:kkp-marketing",),
        )

        decision = router.decide(
            query="who was the qasi judicial authority for this case?",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "memory_scoped_rag")
        self.assertTrue(decision.analysis.active_order_override)
        self.assertTrue(decision.analysis.asks_order_signatory)

    def test_service_answers_quasi_judicial_authority_from_active_matter_metadata(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:kkp-marketing",),
        )
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(state=state),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_FakeMetadataService(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="who was the qasi judicial authority for this case?",
            session_id=session_id,
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertIn("G. Mahalingam", payload.answer_text)
        self.assertTrue(payload.debug["metadata_debug"]["used"])


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return (777,)


class _FakeSessionRepository:
    def __init__(self, *, state: ChatSessionStateRecord) -> None:
        self._state = state
        now = datetime.now(timezone.utc)
        self._snapshot = ChatSessionSnapshot(
            session_id=state.session_id,
            user_name=None,
            created_at=now,
            updated_at=now,
            state=state,
        )

    def create_session_if_missing(self, *, session_id, user_name):
        return None

    def get_session_snapshot(self, *, session_id):
        return self._snapshot

    def get_session_state(self, *, session_id):
        return self._state

    def upsert_session_state(self, **kwargs):
        self._state = ChatSessionStateRecord(
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


class _FakeMetadataService:
    def answer_signatory_question(self, *, document_version_ids):
        return MetadataAnswer(
            answer_text="The order was signed by G. Mahalingam, Whole Time Member and quasi-judicial authority.",
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:kkp-marketing",
                    title="Order in the matter of KKP Marketing",
                    page_start=9,
                    page_end=9,
                    section_type="metadata_signatory",
                    document_version_id=777,
                    chunk_id=None,
                    detail_url="https://example.com/kkp",
                    pdf_url="https://example.com/kkp.pdf",
                ),
            ),
            metadata_type="signatory",
            debug={"metadata_type": "signatory"},
        )

    def answer_order_date_question(self, *, document_version_ids):
        return None

    def answer_numeric_fact_question(self, *, query: str, document_version_ids):
        return None

    def answer_legal_provisions_question(self, *, document_version_ids, explain: bool):
        return None

    def answer_active_matter_follow_up(self, *, query: str, document_version_ids, follow_up_intent: str):
        return None

    def answer_exact_fact_question(self, *, query: str, document_version_ids):
        return None

    def answer_observation_question(self, *, query: str, document_version_ids):
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("metadata follow-up should not use retrieval")


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
