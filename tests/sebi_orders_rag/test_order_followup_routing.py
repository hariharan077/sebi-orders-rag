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


class OrderFollowUpRoutingTests(unittest.TestCase):
    def test_router_prefers_active_order_override_for_signatory_follow_up(self) -> None:
        router = AdaptiveQueryRouter()
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_record_keys=("external:99753",),
        )

        decision = router.decide(
            query="which wtm signed the order",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "memory_scoped_rag")
        self.assertTrue(decision.analysis.active_order_override)
        self.assertTrue(decision.analysis.asks_order_signatory)

    def test_router_prefers_active_order_override_for_provision_explanation_follow_up(self) -> None:
        router = AdaptiveQueryRouter()
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_record_keys=("external:99753",),
        )

        decision = router.decide(
            query="what are these violation sections explain them",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "memory_scoped_rag")
        self.assertTrue(decision.analysis.active_order_override)
        self.assertTrue(decision.analysis.asks_provision_explanation)

    def test_router_keeps_da_observation_follow_up_inside_active_order(self) -> None:
        router = AdaptiveQueryRouter()
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_record_keys=("external:99753",),
        )

        decision = router.decide(
            query="what did the DA observe in this case",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "memory_scoped_rag")
        self.assertTrue(decision.analysis.active_order_override)

    def test_answer_service_uses_metadata_for_active_order_follow_ups(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:99753",),
        )
        metadata_service = _FakeMetadataService()
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
            metadata_service=metadata_service,
            llm_client=_FakeLlmClient({}),
        )

        signatory_payload = service.answer_query(
            query="which wtm signed the order",
            session_id=session_id,
        )
        provisions_payload = service.answer_query(
            query="what are these violation sections explain them",
            session_id=session_id,
        )

        self.assertEqual(signatory_payload.route_mode, "memory_scoped_rag")
        self.assertIn("Ananth Narayan G.", signatory_payload.answer_text)
        self.assertEqual(provisions_payload.route_mode, "memory_scoped_rag")
        self.assertIn("Section 12A(a)", provisions_payload.answer_text)
        self.assertTrue(metadata_service.signatory_called)
        self.assertTrue(metadata_service.provisions_called)

    def test_answer_service_uses_metadata_for_da_observation_follow_up(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:99753",),
        )
        metadata_service = _FakeMetadataService()
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
            metadata_service=metadata_service,
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="what did the DA observe in this case",
            session_id=session_id,
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("Designated Authority observed", payload.answer_text)
        self.assertTrue(payload.debug["metadata_debug"]["used"])
        self.assertTrue(metadata_service.observations_called)

    def test_answer_service_returns_graceful_no_da_message_in_active_matter(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:99753",),
        )
        metadata_service = _FakeMetadataService(no_da_observation=True)
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
            metadata_service=metadata_service,
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="what were the observations",
            session_id=session_id,
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("could not identify a Designated Authority observation", payload.answer_text)
        self.assertNotEqual(payload.route_mode, "abstain")


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return (99753,)


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


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("retrieval should not be used when metadata answers the follow-up")


class _FakeMetadataService:
    def __init__(self, no_da_observation: bool = False) -> None:
        self.signatory_called = False
        self.provisions_called = False
        self.observations_called = False
        self.no_da_observation = no_da_observation

    def answer_signatory_question(self, *, document_version_ids):
        self.signatory_called = True
        return MetadataAnswer(
            answer_text="The order was signed by Ananth Narayan G., Whole Time Member.",
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:99753",
                    title="Settlement Order in respect of SK Finance Limited",
                    page_start=8,
                    page_end=8,
                    section_type="metadata_signatory",
                    document_version_id=99753,
                    chunk_id=None,
                    detail_url="https://example.com/detail",
                    pdf_url="https://example.com/pdf",
                ),
            ),
            metadata_type="signatory",
            debug={"metadata_type": "signatory"},
        )

    def answer_legal_provisions_question(self, *, document_version_ids, explain: bool):
        self.provisions_called = True
        return MetadataAnswer(
            answer_text=(
                "The active order cites Section 12A(a) of SEBI Act, 1992 and Regulation 3(1) "
                "of PFUTP Regulations. In plain terms: Section 12A(a) prohibits fraudulent or "
                "manipulative conduct; Regulation 3(1) broadly prohibits unfair trade practices."
            ),
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:99753",
                    title="Settlement Order in respect of SK Finance Limited",
                    page_start=3,
                    page_end=4,
                    section_type="metadata_legal_provision",
                    document_version_id=99753,
                    chunk_id=None,
                    detail_url="https://example.com/detail",
                    pdf_url="https://example.com/pdf",
                ),
            ),
            metadata_type="legal_provisions",
            debug={"metadata_type": "legal_provisions"},
        )

    def answer_numeric_fact_question(self, *, query: str, document_version_ids):
        return None

    def answer_observation_question(self, *, query: str, document_version_ids):
        self.observations_called = True
        if self.no_da_observation:
            return MetadataAnswer(
                answer_text=(
                    "I could not identify a Designated Authority observation in this matter "
                    "from the indexed text."
                ),
                citations=(
                    Citation(
                        citation_number=1,
                        record_key="external:99753",
                        title="Settlement Order in respect of SK Finance Limited",
                        page_start=1,
                        page_end=1,
                        section_type="metadata_observation",
                        document_version_id=99753,
                        chunk_id=None,
                        detail_url="https://example.com/detail",
                        pdf_url="https://example.com/pdf",
                    ),
                ),
                metadata_type="observations",
                debug={"metadata_type": "observations", "observation_found": False},
            )
        return MetadataAnswer(
            answer_text=(
                "The Designated Authority observed that the noticee's explanation was not "
                "supported by the trading pattern and surrounding circumstances."
            ),
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:99753",
                    title="Settlement Order in respect of SK Finance Limited",
                    page_start=5,
                    page_end=6,
                    section_type="metadata_observation",
                    document_version_id=99753,
                    chunk_id=None,
                    detail_url="https://example.com/detail",
                    pdf_url="https://example.com/pdf",
                ),
            ),
            metadata_type="observations",
            debug={"metadata_type": "observations", "observation_found": True},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
