from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.metadata.models import MetadataChunkText, StoredOrderMetadata
from app.sebi_orders_rag.metadata.service import MetadataAnswer, OrderMetadataService
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord, Citation


class ExactLookupFollowUpTests(unittest.TestCase):
    def test_signatory_and_order_date_follow_ups_use_active_matter_metadata_first(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:100663",),
            active_document_version_ids=(104,),
            active_primary_title="Order in the matter of Mr Yash Garg Proprietor of Yash Trading Academy",
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

        signatory_payload = service.answer_query(query="Who signed this case?", session_id=session_id)
        order_date_payload = service.answer_query(query="When was this order passed?", session_id=session_id)

        self.assertEqual(signatory_payload.route_mode, "memory_scoped_rag")
        self.assertTrue(signatory_payload.debug["metadata_debug"]["used"])
        self.assertIn("Ananth Narayan G.", signatory_payload.answer_text)

        self.assertEqual(order_date_payload.route_mode, "memory_scoped_rag")
        self.assertTrue(order_date_payload.debug["metadata_debug"]["used"])
        self.assertIn("2026-03-27", order_date_payload.answer_text)
        self.assertTrue(metadata_service.signatory_called)
        self.assertTrue(metadata_service.order_date_called)

    def test_signatory_footer_fallback_extracts_name_from_active_order_chunks(self) -> None:
        service = OrderMetadataService(repository=_FooterFallbackMetadataRepository())

        answer = service.answer_signatory_question(document_version_ids=(104,))

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("N. Murugan", answer.answer_text)
        self.assertIn("Quasi Judicial Authority", answer.answer_text)
        self.assertEqual(answer.citations[0].page_start, 32)

    def test_holding_fact_question_uses_metadata_fact_scan_before_retrieval(self) -> None:
        service = OrderMetadataService(repository=_HoldingFactMetadataRepository())

        answer = service.answer_exact_fact_question(
            query="how much shares does aruna dhanuka family trust hold in mint investment limited",
            document_version_ids=(501,),
        )

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("Aruna Dhanuka Family Trust was the proposed acquirer proposed to hold", answer.answer_text)
        self.assertIn("23.93%", answer.answer_text)
        self.assertEqual(answer.metadata_type, "holding_fact")


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return (104,)

    def fetch_order_metadata(self, *, document_version_ids):
        return ()

    def fetch_legal_provisions(self, *, document_version_ids):
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
            active_document_ids=tuple(kwargs["active_document_ids"]),
            active_document_version_ids=tuple(kwargs.get("active_document_version_ids", ())),
            active_record_keys=tuple(kwargs["active_record_keys"]),
            active_entities=tuple(kwargs["active_entities"]),
            active_bucket_names=tuple(kwargs["active_bucket_names"]),
            active_primary_title=kwargs.get("active_primary_title"),
            active_primary_entity=kwargs.get("active_primary_entity"),
            active_signatory_name=kwargs.get("active_signatory_name"),
            active_signatory_designation=kwargs.get("active_signatory_designation"),
            active_order_date=kwargs.get("active_order_date"),
            active_legal_provisions=tuple(kwargs.get("active_legal_provisions", ())),
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
    def __init__(self) -> None:
        self.signatory_called = False
        self.order_date_called = False

    def answer_signatory_question(self, *, document_version_ids):
        self.signatory_called = True
        return MetadataAnswer(
            answer_text="The order was signed by Ananth Narayan G., Whole Time Member.",
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:100663",
                    title="Order in the matter of Mr Yash Garg Proprietor of Yash Trading Academy",
                    page_start=8,
                    page_end=8,
                    section_type="metadata_signatory",
                    document_version_id=104,
                    chunk_id=None,
                    detail_url="https://example.com/detail",
                    pdf_url="https://example.com/pdf",
                ),
            ),
            metadata_type="signatory",
            debug={"metadata_type": "signatory"},
        )

    def answer_order_date_question(self, *, document_version_ids):
        self.order_date_called = True
        return MetadataAnswer(
            answer_text="The order date recorded for this matter is 2026-03-27.",
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:100663",
                    title="Order in the matter of Mr Yash Garg Proprietor of Yash Trading Academy",
                    page_start=1,
                    page_end=1,
                    section_type="metadata_signatory",
                    document_version_id=104,
                    chunk_id=None,
                    detail_url="https://example.com/detail",
                    pdf_url="https://example.com/pdf",
                ),
            ),
            metadata_type="order_date",
            debug={"metadata_type": "order_date"},
        )

    def answer_legal_provisions_question(self, *, document_version_ids, explain: bool):
        return None

    def answer_numeric_fact_question(self, *, query: str, document_version_ids):
        return None


class _FooterFallbackMetadataRepository:
    def fetch_order_metadata(self, *, document_version_ids):
        return (
            StoredOrderMetadata(
                document_version_id=104,
                document_id=123,
                record_key="external:100663",
                title="Order in the matter of Mr Yash Garg Proprietor of Yash Trading Academy",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                order_date=date(2026, 3, 27),
            ),
        )

    def load_chunks(self, *, document_version_id: int):
        self.last_document_version_id = document_version_id
        return (
            MetadataChunkText(
                chunk_id=2862,
                page_start=32,
                page_end=32,
                text=(
                    "BSE Administration and Supervision Ltd., to ensure that the directions "
                    "given above are strictly complied with.\n\n"
                    "Date: March 27, 2026 Place: Mumbai\n\n"
                    "N. MURUGAN\n\n"
                    "QUASI JUDICIAL AUTHORITY\n"
                ),
            ),
        )


class _HoldingFactMetadataRepository:
    def fetch_order_metadata(self, *, document_version_ids):
        return (
            StoredOrderMetadata(
                document_version_id=501,
                document_id=501,
                record_key="external:mint",
                title="Exemption Order in the matter of Mint Investment Limited",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                order_date=date(2026, 2, 11),
            ),
        )

    def load_chunks(self, *, document_version_id: int):
        return (
            MetadataChunkText(
                chunk_id=9901,
                page_start=3,
                page_end=3,
                text=(
                    "Mrs. Aruna Dhanuka individually held 5,65,818 shares = 10.21%. "
                    "Aruna Dhanuka Family Trust was the proposed acquirer proposed to hold "
                    "13,25,880 shares = 23.93% under an exemption from open offer."
                ),
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
