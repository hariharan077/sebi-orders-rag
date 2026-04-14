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


class NumericFactRoutingTests(unittest.TestCase):
    def test_router_prefers_order_metadata_for_du_digital_numeric_fact_queries(self) -> None:
        router = AdaptiveQueryRouter()

        for query in (
            "how much did DU Digital share price increase",
            "what was the price before and after the increase in DU Digital",
            "give the price movement of DU Digital for each period",
        ):
            with self.subTest(query=query):
                decision = router.decide(query=query)
                self.assertEqual(decision.route_mode, "hierarchical_rag")
                self.assertTrue(decision.analysis.asks_order_numeric_fact)
                self.assertIsNotNone(decision.plan)
                assert decision.plan is not None
                self.assertEqual(decision.plan.route, "order_metadata")

    def test_service_answers_du_digital_numeric_fact_from_metadata_before_retrieval(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_NumericRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_FakeMetadataService(),
            llm_client=_FakeLlmClient({}),
        )

        payload = service.answer_query(
            query="how much did DU Digital share price increase",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "hierarchical_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("1392.5%", payload.answer_text)
        self.assertTrue(payload.debug["metadata_debug"]["used"])


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("numeric fact queries should be answered from metadata first")


class _NumericRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        all_queries = [query, *(query_variants or ())]
        lowered = " ".join(item.lower() for item in all_queries)
        if "du digital" not in lowered:
            return []
        return [
            ExactLookupCandidate(
                document_version_id=118,
                document_id=137,
                record_key="external:du-digital",
                bucket_name="orders-of-whole-time-member",
                external_record_id="118",
                order_date=date(2025, 9, 1),
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited",
                match_score=0.91,
            )
        ]

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


class _FakeMetadataService:
    def answer_signatory_question(self, *, document_version_ids):
        return None

    def answer_order_date_question(self, *, document_version_ids):
        return None

    def answer_numeric_fact_question(self, *, query: str, document_version_ids):
        return MetadataAnswer(
            answer_text=(
                "The order metadata records that DU Digital moved from Rs.12/share at listing "
                "to Rs.179.10, a rise of 1392.5%."
            ),
            citations=(
                Citation(
                    citation_number=1,
                    record_key="external:du-digital",
                    title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited",
                    page_start=2,
                    page_end=2,
                    section_type="metadata_numeric_fact",
                    document_version_id=118,
                    chunk_id=None,
                    detail_url="https://example.com/du-digital",
                    pdf_url="https://example.com/du-digital.pdf",
                ),
            ),
            metadata_type="price_increase",
            debug={"metadata_type": "price_increase"},
        )

    def answer_legal_provisions_question(self, *, document_version_ids, explain: bool):
        return None

    def answer_active_matter_follow_up(self, *, query: str, document_version_ids, follow_up_intent: str):
        return None

    def answer_exact_fact_question(self, *, query: str, document_version_ids):
        return None

    def answer_observation_question(self, *, query: str, document_version_ids):
        return None


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
