from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.provider import CurrentInfoResult, CurrentInfoSource
from app.sebi_orders_rag.retrieval.scoring import ChunkSearchHit, HierarchicalSearchResult, ScoreBreakdown
from app.sebi_orders_rag.schemas import ChatSessionSnapshot


class AnswerServiceTests(unittest.TestCase):
    def test_retrieval_mode_refuses_unsupported_claims_without_citations(self) -> None:
        service = _build_service(
            llm_payload={"answer_status": "answered", "answer_text": "SEBI directed compliance."},
            chunk_score=0.09,
        )

        payload = service.answer_query(
            query="What penalty was imposed in the 2026 order?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertEqual(payload.citations, ())

    def test_abstains_when_retrieval_support_is_too_weak(self) -> None:
        service = _build_service(
            llm_payload={"answer_status": "answered", "answer_text": "The appeal was dismissed [1]."},
            chunk_score=0.001,
        )

        payload = service.answer_query(
            query="What penalty was imposed in the 2026 order?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertLess(payload.confidence, 0.35)

    def test_retrieval_mode_strips_inline_citation_markers_from_answer_text(self) -> None:
        service = _build_service(
            llm_payload={
                "answer_status": "answered",
                "answer_text": "The appeal was dismissed [1].",
                "cited_numbers": [1],
            },
            chunk_score=0.09,
        )

        payload = service.answer_query(
            query="What happened in the 2026 order?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "hierarchical_rag")
        self.assertEqual(payload.answer_text, "The appeal was dismissed.")
        self.assertEqual(len(payload.citations), 1)
        self.assertEqual(payload.citations[0].detail_url, "https://example.com/detail/100725")
        self.assertEqual(payload.citations[0].pdf_url, "https://example.com/pdf/100725.pdf")

    def test_current_official_lookup_uses_provider_not_corpus(self) -> None:
        settings = SebiOrdersRagSettings(
            db_dsn="postgresql://unused",
            data_root=Path(".").resolve(),
            low_confidence_threshold=0.35,
            enable_memory=True,
        )
        service = AdaptiveRagAnswerService(
            settings=settings,
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            current_info_provider=_FakeCurrentInfoProvider(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="Who is the chairperson of SEBI?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "current_official_lookup")
        self.assertEqual(payload.query_intent, "structured_current_info")
        self.assertEqual(len(payload.citations), 1)
        self.assertEqual(payload.citations[0].source_url, "https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp")

    def test_current_person_lookup_no_match_abstains_without_using_corpus(self) -> None:
        settings = SebiOrdersRagSettings(
            db_dsn="postgresql://unused",
            data_root=Path(".").resolve(),
            low_confidence_threshold=0.35,
            enable_memory=True,
        )
        service = AdaptiveRagAnswerService(
            settings=settings,
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            current_info_provider=_FakeNoMatchCurrentInfoProvider(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="is there an assistant manager called Bhuvanesh? when did he join whats his number?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.query_intent, "structured_current_info")
        self.assertIn("No matching current directory entry was found", payload.answer_text)

    def test_current_office_lookup_does_not_use_corpus(self) -> None:
        settings = SebiOrdersRagSettings(
            db_dsn="postgresql://unused",
            data_root=Path(".").resolve(),
            low_confidence_threshold=0.35,
            enable_memory=True,
        )
        service = AdaptiveRagAnswerService(
            settings=settings,
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            current_info_provider=_FakeOfficeCurrentInfoProvider(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="SEBI office address Mumbai",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "current_official_lookup")
        self.assertIn("SEBI Bhavan BKC", payload.answer_text)

    def test_static_sebi_definition_does_not_abstain(self) -> None:
        service = _build_service(
            llm_payload={},
            chunk_score=0.09,
        )

        payload = service.answer_query(
            query="Who are SEBI",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "general_knowledge")
        self.assertIn("Securities and Exchange Board of India", payload.answer_text)


class _FakeConnection:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakeSearchService:
    def __init__(self, *, chunk_score: float) -> None:
        self._chunk_score = chunk_score

    def search(self, *, query: str, filters=None, top_k_docs=None, top_k_sections=None, top_k_chunks=None):
        return HierarchicalSearchResult(
            query=query,
            documents=(),
            sections=(),
            chunks=(
                ChunkSearchHit(
                    chunk_id=11,
                    document_version_id=101,
                    document_id=201,
                    record_key="external:100725",
                    bucket_name="orders-of-aa-under-rti-act",
                    external_record_id="100725",
                    order_date=None,
                    title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                    chunk_index=0,
                    page_start=1,
                    page_end=2,
                    section_key="section-1",
                    section_type="operative_order",
                    section_title="ORDER",
                    heading_path="ORDER",
                    detail_url="https://example.com/detail/100725",
                    pdf_url="https://example.com/pdf/100725.pdf",
                    chunk_text="The appeal is dismissed and the information is to be provided.",
                    token_count=48,
                    score=ScoreBreakdown(
                        combined_score=self._chunk_score,
                        final_score=self._chunk_score,
                    ),
                ),
            ),
        )


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


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
        from app.sebi_orders_rag.schemas import ChatSessionStateRecord

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


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt) -> dict[str, object]:
        return dict(self._payload)


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("corpus retrieval should not be used for current official lookups")


class _FakeCurrentInfoProvider:
    def lookup(self, *, query: str, session_state=None) -> CurrentInfoResult:
        return CurrentInfoResult(
            answer_status="answered",
            answer_text="SEBI's directory lists Tuhin Kanta Pandey as Chairman of SEBI.",
            sources=(
                CurrentInfoSource(
                    title="SEBI Directory",
                    url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    record_key="official:sebi.gov.in",
                ),
            ),
            confidence=0.91,
            provider_name="official_web",
            lookup_type="sebi_chairperson",
            debug={"detected_query_family": "chairperson"},
        )


class _FakeNoMatchCurrentInfoProvider:
    def lookup(self, *, query: str, session_state=None) -> CurrentInfoResult:
        return CurrentInfoResult(
            answer_status="insufficient_context",
            answer_text="No matching current directory entry was found in the ingested official SEBI data.",
            provider_name="structured_directory",
            lookup_type="person_lookup",
            debug={"detected_query_family": "person_lookup"},
        )


class _FakeOfficeCurrentInfoProvider:
    def lookup(self, *, query: str, session_state=None) -> CurrentInfoResult:
        return CurrentInfoResult(
            answer_status="answered",
            answer_text="SEBI Bhavan BKC: address: Plot No.C4-A, G Block, Bandra-Kurla Complex, Bandra (East), Mumbai - 400051, Maharashtra; phone: +91-22-26449000 / 40459000; email: sebi@sebi.gov.in.",
            sources=(
                CurrentInfoSource(
                    title="SEBI Contact Us",
                    url="https://www.sebi.gov.in/contact-us.html",
                    record_key="official:contact_us",
                ),
            ),
            confidence=0.94,
            provider_name="structured_directory",
            lookup_type="office_contact",
            debug={"detected_query_family": "office_contact", "extracted_city": "Mumbai"},
        )


def _build_service(*, llm_payload, chunk_score: float) -> AdaptiveRagAnswerService:
    settings = SebiOrdersRagSettings(
        db_dsn="postgresql://unused",
        data_root=Path(".").resolve(),
        low_confidence_threshold=0.35,
        enable_memory=True,
    )
    return AdaptiveRagAnswerService(
        settings=settings,
        connection=_FakeConnection(),
        search_service=_FakeSearchService(chunk_score=chunk_score),
        retrieval_repository=_FakeRetrievalRepository(),
        session_repository=_FakeSessionRepository(),
        answer_repository=_FakeAnswerRepository(),
        llm_client=_FakeLlmClient(llm_payload),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
