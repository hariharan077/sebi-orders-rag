from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.metadata.service import MetadataAnswer
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, Citation, ExactLookupCandidate


class MetadataFirstPlannerTests(unittest.TestCase):
    def test_router_sends_named_order_signatory_query_to_order_metadata(self) -> None:
        decision = AdaptiveQueryRouter().decide(query="who signed the Hemant Ghai order")

        self.assertEqual(decision.route_mode, "hierarchical_rag")
        self.assertIsNotNone(decision.plan)
        assert decision.plan is not None
        self.assertEqual(decision.plan.route, "order_metadata")
        self.assertEqual(decision.plan.reason, "exact_order_fact_metadata_first")

    def test_router_sends_freeform_numeric_entity_query_to_order_metadata(self) -> None:
        decision = AdaptiveQueryRouter().decide(
            query="give the price movement of DU Digital for each period"
        )

        self.assertEqual(decision.route_mode, "hierarchical_rag")
        self.assertTrue(decision.analysis.asks_order_numeric_fact)
        self.assertIsNotNone(decision.plan)
        assert decision.plan is not None
        self.assertEqual(decision.plan.route, "order_metadata")

    def test_router_keeps_plain_named_order_query_on_internal_corpus_path(self) -> None:
        decision = AdaptiveQueryRouter().decide(query="Hemant Ghai order")

        self.assertEqual(decision.route_mode, "exact_lookup")
        self.assertIsNotNone(decision.plan)
        assert decision.plan is not None
        self.assertEqual(decision.plan.route, "order_corpus_rag")

    def test_service_resolves_named_order_metadata_without_retrieval(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_MetadataCandidateRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_FakeMetadataService(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="who signed the Hemant Ghai order",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "hierarchical_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("Biju S.", payload.answer_text)
        self.assertTrue(payload.debug["metadata_debug"]["used"])

    def test_service_clarifies_missing_order_scope_for_exact_fact_question(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_MetadataCandidateRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_FakeMetadataService(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="who signed the order",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "clarify")
        self.assertEqual(payload.answer_status, "clarify")
        self.assertIn("Please specify the exact SEBI order", payload.answer_text)


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


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


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("named order metadata should resolve before retrieval")


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


class _MetadataCandidateRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        all_queries = [query, *(query_variants or ())]
        lowered = " ".join(item.lower() for item in all_queries)
        if "hemant ghai" in lowered:
            return [
                ExactLookupCandidate(
                    document_version_id=99001,
                    document_id=88001,
                    record_key="external:hemant-ghai",
                    bucket_name="settlement-orders",
                    external_record_id="99001",
                    order_date=date(2025, 12, 9),
                    title="Settlement Order in respect of Hemant Ghai",
                    match_score=0.94,
                )
            ]
        if "du digital" in lowered:
            return [
                ExactLookupCandidate(
                    document_version_id=99002,
                    document_id=88002,
                    record_key="external:du-digital",
                    bucket_name="orders-of-whole-time-member",
                    external_record_id="99002",
                    order_date=date(2025, 9, 1),
                    title="Order in the matter of DU Digital Technologies Private Limited",
                    match_score=0.91,
                )
            ]
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _FakeMetadataService:
    def answer_signatory_question(self, *, document_version_ids):
        if 99001 not in set(document_version_ids):
            return None
        return MetadataAnswer(
            answer_text="The order was signed by Biju S., Quasi Judicial Authority; date: 2025-12-09.",
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:hemant-ghai",
                    title="Settlement Order in respect of Hemant Ghai",
                    page_start=3,
                    page_end=3,
                    section_type="metadata_signatory",
                    document_version_id=99001,
                    chunk_id=None,
                    detail_url="https://example.com/hemant-ghai",
                    pdf_url="https://example.com/hemant-ghai.pdf",
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
