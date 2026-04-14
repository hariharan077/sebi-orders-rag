from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.answering.style import apply_grounded_wording_caution
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.metadata.models import MetadataChunkText, StoredOrderMetadata
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.router.query_analyzer import analyze_query
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ExactLookupCandidate, PromptContextChunk
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT


class TrueBugRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(CONTROL_PACK_FIXTURE_ROOT)
        assert cls.pack is not None

    def test_jp_morgan_settlement_amount_query_locks_to_one_matter(self) -> None:
        analysis = analyze_query(
            "What was the settlement amount in the JP Morgan settlement order?",
            control_pack=self.pack,
        )

        self.assertTrue(analysis.strict_scope_required)
        self.assertTrue(analysis.strict_single_matter)
        self.assertEqual(analysis.strict_matter_lock.locked_record_keys, ("external:100486",))
        self.assertIn("alias_match", analysis.strict_matter_lock.reason_codes)

    def test_imaginary_capital_settlement_amount_abstains_without_candidate_list(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_NoopRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoopMetadataService(),
            llm_client=_FakeLlmClient({}),
            router=AdaptiveQueryRouter(control_pack=self.pack),
        )

        payload = service.answer_query(
            query="What was the settlement amount in the Imaginary Capital Limited settlement order?",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertNotIn("candidate_list_debug", payload.debug)

    def test_metadata_document_ids_prefer_locked_record_key(self) -> None:
        session_id = uuid4()
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_MetadataResolutionRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoopMetadataService(),
            llm_client=_FakeLlmClient({}),
            router=AdaptiveQueryRouter(control_pack=self.pack),
        )
        decision = service._router.decide(
            query="What was the settlement amount in the Mangalam Global Enterprise Limited settlement order?"
        )

        document_version_ids = service._metadata_document_version_ids(
            session_id=session_id,
            decision=decision,
        )

        self.assertEqual(document_version_ids, (191,))

    def test_comparison_queries_stay_on_retrieval_route(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.pack)

        decision = router.decide(
            query="Compare the JP Morgan Chase Bank N.A. and DDP- Standard Chartered Bank settlement orders."
        )

        self.assertEqual(decision.route_mode, "hierarchical_rag")
        self.assertTrue(decision.analysis.comparison_intent)

    def test_prime_broking_exact_title_query_does_not_lock_to_in_the_matter_variant(self) -> None:
        analysis = analyze_query(
            "Prime Broking Company (India) Limited",
            control_pack=self.pack,
        )

        self.assertFalse(analysis.strict_single_matter)
        self.assertEqual(
            tuple(candidate.record_key for candidate in analysis.strict_matter_lock.candidates[:2]),
            ("external:30222", "external:30223"),
        )
        self.assertNotEqual(
            analysis.strict_matter_lock.candidates[0].record_key,
            "external:30189",
        )

    def test_rti_prime_broking_query_abstains_without_unrelated_candidate_list(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_NoopRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoopMetadataService(),
            llm_client=_FakeLlmClient({}),
            router=AdaptiveQueryRouter(control_pack=self.pack),
        )

        payload = service.answer_query(
            query="Tell me more about the RTI appeal filed by Prime Broking Company India Limited.",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertNotIn("RTI Baba", payload.answer_text)
        self.assertFalse(payload.debug.get("candidate_list_debug", {}).get("used", False))

    def test_generic_cochin_entity_summary_abstains_instead_of_domain_clarify(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_NoopRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoopMetadataService(),
            llm_client=_FakeLlmClient({}),
            router=AdaptiveQueryRouter(control_pack=self.pack),
        )

        payload = service.answer_query(
            query="Tell me more about Cochin Stock Exchange Limited.",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertNotIn("structured current fact", payload.answer_text)

    def test_exact_title_lock_uses_document_lookup_fallback(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_NoopRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoopMetadataService(),
            llm_client=_FakeLlmClient({}),
            router=AdaptiveQueryRouter(control_pack=self.pack),
        )
        decision = service._router.decide(
            query="Final Order in respect of Basan Financial Services Limited"
        )

        payload = service._maybe_build_document_lookup_fallback(
            session_id=uuid4(),
            decision=decision,
            context_chunks=(
                _prompt_chunk(
                    title="Final Order in respect of Basan Financial Services Limited",
                    bucket_name="orders-of-ed-cgm",
                    chunk_text="Final Order in respect of Basan Financial Services Limited.",
                ),
            ),
            retrieved_chunk_ids=(1,),
            debug_payload={},
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.route_mode, "exact_lookup")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("Final Order in respect of Basan Financial Services Limited", payload.answer_text)

    def test_structured_people_department_filter_routes_to_structured_current_info(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.pack)

        decision = router.decide(query="Is there an abhishek in ITD")

        self.assertEqual(decision.route_mode, "structured_current_info")
        self.assertEqual(decision.query_intent, "structured_current_info")

    def test_settlement_order_does_not_answer_exemption_query_as_exemption(self) -> None:
        analysis = analyze_query(
            "What exemption did SEBI grant in the JP Morgan Chase Bank N.A. matter?",
            control_pack=self.pack,
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="SEBI granted an exemption to JP Morgan Chase Bank N.A.",
            context_chunks=(
                _prompt_chunk(
                    title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                    bucket_name="settlement-orders",
                    chunk_text=(
                        "It was alleged that the applicant violated the SEBI (Foreign Portfolio Investors) "
                        "Regulations. The specified proceedings were disposed of through settlement."
                    ),
                ),
            ),
            analysis=analysis,
        )

        self.assertIn("not an exemption order", answer_text)
        self.assertTrue(debug["used"])

    def test_preferential_allotment_context_does_not_become_ipo_proceeds(self) -> None:
        analysis = analyze_query(
            "What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
            control_pack=self.pack,
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="Pacheli raised IPO proceeds of approximately INR 849.99 crore.",
            context_chunks=(
                _prompt_chunk(
                    title="Confirmatory Order in the matter of Pacheli Industrial Finance Limited",
                    bucket_name="orders-of-ed-cgm",
                    chunk_text=(
                        "The company made a preferential allotment of equity shares for conversion of the "
                        "outstanding unsecured loans into equity shares."
                    ),
                    section_type="facts",
                ),
            ),
            analysis=analysis,
        )

        self.assertIn("does not describe IPO proceeds", answer_text)
        self.assertIn("preferential allotment", answer_text.lower())
        self.assertTrue(debug["used"])

    def test_missing_ipo_proceeds_returns_grounded_negative_answer(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_NoopRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoopMetadataService(),
            llm_client=_FakeLlmClient({}),
            router=AdaptiveQueryRouter(control_pack=self.pack),
        )
        service._metadata_repository = _MissingNumericFactMetadataRepository()
        decision = service._router.decide(
            query="What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?"
        )

        payload = service._maybe_answer_missing_numeric_fact(
            query="What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
            session_id=uuid4(),
            decision=decision,
            document_version_ids=(74,),
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIn(payload.route_mode, {"exact_lookup", "hierarchical_rag"})
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("does not describe IPO proceeds", payload.answer_text)
        self.assertIn("preferential allotment", payload.answer_text.lower())
        self.assertEqual(
            payload.active_record_keys,
            ("derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46",),
        )
        self.assertTrue(payload.citations)


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
        raise AssertionError("this regression should resolve before retrieval")


class _NoopRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _MetadataResolutionRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return [
            ExactLookupCandidate(
                document_version_id=191,
                document_id=191,
                record_key="external:100669",
                bucket_name="settlement-orders",
                external_record_id="100669",
                order_date=None,
                title="Settlement Order in the matter of Mangalam Global Enterprise Limited",
                match_score=0.98,
            ),
            ExactLookupCandidate(
                document_version_id=192,
                document_id=192,
                record_key="external:99922",
                bucket_name="settlement-orders",
                external_record_id="99922",
                order_date=None,
                title="Settlement Order in the matter of Kalyani Steels Limited",
                match_score=0.71,
            ),
        ]

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        if tuple(record_keys) == ("external:100669",):
            return (191,)
        return tuple(document_ids)


class _MissingNumericFactMetadataRepository:
    def fetch_order_metadata(self, *, document_version_ids):
        if tuple(document_version_ids) != (74,):
            return ()
        return (
            StoredOrderMetadata(
                document_version_id=74,
                document_id=74,
                record_key="derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46",
                title="Confirmatory Order in the matter of Pacheli Industrial Finance Limited",
                detail_url=None,
                pdf_url=None,
            ),
        )

    def load_chunks(self, *, document_version_id: int):
        if document_version_id != 74:
            return ()
        return (
            MetadataChunkText(
                chunk_id=1,
                page_start=1,
                page_end=1,
                text=(
                    "The company made a preferential allotment of equity shares for conversion "
                    "of the outstanding unsecured loans into equity shares."
                ),
                section_type="facts",
                section_title="Facts",
            ),
        )


class _NoopMetadataService:
    def answer_signatory_question(self, *, document_version_ids):
        return None

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


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


def _prompt_chunk(
    *,
    title: str,
    bucket_name: str,
    chunk_text: str,
    section_type: str = "operative_order",
) -> PromptContextChunk:
    return PromptContextChunk(
        citation_number=1,
        chunk_id=1,
        document_version_id=1,
        document_id=1,
        record_key="external:test",
        bucket_name=bucket_name,
        title=title,
        page_start=1,
        page_end=1,
        section_type=section_type,
        section_title="ORDER",
        detail_url=None,
        pdf_url=None,
        chunk_text=chunk_text,
        token_count=64,
        score=0.95,
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
