from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.metadata.models import MetadataChunkText, StoredOrderMetadata
from app.sebi_orders_rag.metadata.service import OrderMetadataService
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord


class ActiveMatterSemanticFollowUpTests(unittest.TestCase):
    def test_exemption_follow_up_uses_section_aware_metadata_answer(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e",),
            active_document_version_ids=(101,),
            active_primary_title="Exemption order in the matter of Hardcastle and Waud Manufacturing Ltd.",
        )
        metadata_repository = _SemanticMetadataRepository()
        service = _build_service(
            state=state,
            metadata_service=OrderMetadataService(repository=metadata_repository),
        )

        payload = service.answer_query(
            query="What exemption was granted?",
            session_id=session_id,
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertTrue(payload.debug["metadata_debug"]["used"])
        self.assertEqual(
            payload.debug["metadata_debug"]["follow_up_intent"],
            "exemption_granted",
        )
        self.assertIn("Regulation 11", payload.answer_text)
        self.assertEqual(
            payload.active_record_keys,
            ("derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e",),
        )

    def test_appellate_authority_follow_up_uses_result_section_before_generic_rag(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:100722",),
            active_document_version_ids=(202,),
            active_primary_title="Order of the Appellate Authority under the RTI Act in the matter of Rajat Kumar",
        )
        metadata_repository = _SemanticMetadataRepository()
        service = _build_service(
            state=state,
            metadata_service=OrderMetadataService(repository=metadata_repository),
        )

        payload = service.answer_query(
            query="What did the appellate authority decide?",
            session_id=session_id,
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertTrue(payload.debug["metadata_debug"]["used"])
        self.assertEqual(
            payload.debug["metadata_debug"]["follow_up_intent"],
            "appellate_authority_decision",
        )
        self.assertIn("appeal is dismissed", payload.answer_text.lower())
        self.assertEqual(payload.active_record_keys, ("external:100722",))


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


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


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("semantic active-matter follow-up should be answered from metadata first")


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("semantic active-matter follow-up should not call the LLM")


class _SemanticMetadataRepository:
    def __init__(self) -> None:
        self._rows = {
            101: StoredOrderMetadata(
                document_version_id=101,
                document_id=501,
                record_key="derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e",
                title="Exemption order in the matter of Hardcastle and Waud Manufacturing Ltd.",
                detail_url="https://example.com/detail/hardcastle",
                pdf_url="https://example.com/pdf/hardcastle.pdf",
                order_date=date(2026, 1, 7),
            ),
            202: StoredOrderMetadata(
                document_version_id=202,
                document_id=602,
                record_key="external:100722",
                title="Order of the Appellate Authority under the RTI Act in the matter of Rajat Kumar",
                detail_url="https://example.com/detail/rajat-kumar",
                pdf_url="https://example.com/pdf/rajat-kumar.pdf",
                order_date=date(2025, 7, 15),
            ),
        }
        self._chunks = {
            101: (
                MetadataChunkText(
                    chunk_id=7001,
                    page_start=7,
                    page_end=7,
                    section_type="operative_order",
                    section_title="OPERATIVE ORDER",
                    text=(
                        "It is hereby ordered that the proposed acquisition is exempt from the "
                        "applicability of Regulation 11 of the SEBI (Substantial Acquisition of "
                        "Shares and Takeovers) Regulations, 1997."
                    ),
                ),
            ),
            202: (
                MetadataChunkText(
                    chunk_id=8001,
                    page_start=3,
                    page_end=3,
                    section_type="findings",
                    section_title="ORDER",
                    text=(
                        "The appellate authority holds that the appeal is dismissed and the reply "
                        "of the CPIO is upheld."
                    ),
                ),
            ),
        }

    def fetch_order_metadata(self, *, document_version_ids):
        return tuple(
            self._rows[document_version_id]
            for document_version_id in document_version_ids
            if document_version_id in self._rows
        )

    def load_chunks(self, *, document_version_id: int):
        return self._chunks.get(document_version_id, ())

    def fetch_legal_provisions(self, *, document_version_ids):
        return ()


def _build_service(
    *,
    state: ChatSessionStateRecord,
    metadata_service: OrderMetadataService,
) -> AdaptiveRagAnswerService:
    return AdaptiveRagAnswerService(
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
        llm_client=_ExplodingLlmClient(),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
