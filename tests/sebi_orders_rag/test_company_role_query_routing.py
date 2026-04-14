from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService, _filter_general_web_sources
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.control.models import ControlPack, DocumentIndexRow, EntityAliasRow, StrictAnswerRule
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot
from app.sebi_orders_rag.web_fallback.models import WebSearchResult, WebSearchSource


class CompanyRoleQueryRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control_pack = _build_control_pack()

    def test_router_keeps_company_role_queries_off_corpus_routes(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.control_pack)

        for query in (
            "who is the ceo of adani green?",
            "who is the ceo of reliance industries limited",
            "who is the ceo of adani port",
        ):
            with self.subTest(query=query):
                decision = router.decide(query=query)
                self.assertEqual(decision.route_mode, "general_knowledge")
                self.assertTrue(decision.analysis.appears_company_role_current_fact)
                self.assertFalse(decision.analysis.strict_scope_required)

    def test_explicit_order_context_can_still_use_corpus_matching(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.control_pack)

        decision = router.decide(query="who is the ceo of adani green according to sebi order")

        self.assertNotEqual(decision.route_mode, "general_knowledge")
        self.assertFalse(decision.analysis.appears_company_role_current_fact)

    def test_service_uses_general_web_fallback_for_company_role_query(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            general_web_provider=_FakeGeneralWebProvider(),
            llm_client=_FakeLlmClient(
                {
                    "answer_status": "insufficient_context",
                    "answer_text": "I do not know.",
                }
            ),
            router=AdaptiveQueryRouter(control_pack=self.control_pack),
        )

        payload = service.answer_query(
            query="who is the ceo of reliance industries limited",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "general_knowledge")
        self.assertIn("Mukesh Ambani", payload.answer_text)
        self.assertEqual(payload.citations[0].source_type, "general_web")
        self.assertTrue(payload.debug["web_fallback_debug"]["general_web_attempted"])

    def test_company_role_source_filter_keeps_only_relevant_domains(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.control_pack)
        analysis = router.decide(
            query="Who is the CEO of Adani Green Energy Limited",
        ).analysis

        filtered = _filter_general_web_sources(
            query="Who is the CEO of Adani Green Energy Limited",
            answer_text="Ashish Khanna is the CEO of Adani Green Energy Limited.",
            analysis=analysis,
            sources=_noisy_company_role_sources(),
        )

        domains = {source.domain for source in filtered}
        self.assertEqual(domains, {"economictimes.com", "adani.com", "zaubacorp.com"})
        self.assertNotIn("example.com", domains)


def _build_control_pack() -> ControlPack:
    adani_doc = DocumentIndexRow(
        record_key="external:adani-green-order",
        exact_title="Order in the matter of Adani Green Energy Limited",
        bucket_category="orders-of-whole-time-member",
        order_date=date(2024, 5, 1),
        main_entities=("Adani Green Energy Limited",),
        short_summary="SEBI order mentioning Adani Green Energy Limited.",
        summary_source="fixture",
        procedural_type="final_order",
        manifest_status="ingested",
        manifest_error=None,
        ingested=True,
        document_version_id=101,
        detail_url="https://example.com/adani-green",
        pdf_url="https://example.com/adani-green.pdf",
        local_filename="adani_green.pdf",
    )
    reliance_doc = DocumentIndexRow(
        record_key="external:reliance-order",
        exact_title="Order in the matter of Reliance Industries Limited",
        bucket_category="orders-of-whole-time-member",
        order_date=date(2024, 6, 1),
        main_entities=("Reliance Industries Limited",),
        short_summary="SEBI order mentioning Reliance Industries Limited.",
        summary_source="fixture",
        procedural_type="final_order",
        manifest_status="ingested",
        manifest_error=None,
        ingested=True,
        document_version_id=102,
        detail_url="https://example.com/reliance",
        pdf_url="https://example.com/reliance.pdf",
        local_filename="reliance.pdf",
    )
    alias_rows = (
        EntityAliasRow(
            canonical_name="Adani Green Energy Limited",
            short_name="Adani Green",
            abbreviations=(),
            old_name=None,
            new_name=None,
            related_record_keys=(adani_doc.record_key,),
            related_titles=(adani_doc.exact_title,),
        ),
        EntityAliasRow(
            canonical_name="Reliance Industries Limited",
            short_name="Reliance Industries",
            abbreviations=("RIL",),
            old_name=None,
            new_name=None,
            related_record_keys=(reliance_doc.record_key,),
            related_titles=(reliance_doc.exact_title,),
        ),
    )
    return ControlPack(
        root=Path(".").resolve(),
        document_index=(adani_doc, reliance_doc),
        confusion_pairs=(),
        eval_queries=(),
        wrong_answer_examples=(),
        entity_aliases=alias_rows,
        strict_answer_rule=StrictAnswerRule(
            text="Stay inside one matter.",
            strict_single_matter_required=True,
        ),
        documents_by_record_key={
            adani_doc.record_key: adani_doc,
            reliance_doc.record_key: reliance_doc,
        },
        aliases_by_record_key={
            adani_doc.record_key: (alias_rows[0],),
            reliance_doc.record_key: (alias_rows[1],),
        },
        alias_variants={
            "adani green": (alias_rows[0],),
            "adani green energy limited": (alias_rows[0],),
            "reliance industries": (alias_rows[1],),
            "reliance industries limited": (alias_rows[1],),
            "ril": (alias_rows[1],),
        },
        confusion_map={},
    )


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("company-role queries should not use corpus retrieval")


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

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


class _FakeLlmClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


class _FakeGeneralWebProvider:
    def search(self, *, request):
        return WebSearchResult(
            answer_status="answered",
            answer_text="Mukesh Ambani is the Chairman and Managing Director of Reliance Industries Limited.",
            sources=(
                WebSearchSource(
                    source_title="Mukesh Ambani",
                    source_url="https://en.wikipedia.org/wiki/Mukesh_Ambani",
                    domain="wikipedia.org",
                    source_type="general_web",
                    record_key="general_web:wikipedia.org",
                ),
            ),
            provider_name="general_web_search",
            lookup_type=request.lookup_type,
        )


def _noisy_company_role_sources() -> tuple[WebSearchSource, ...]:
    return (
        WebSearchSource(
            source_title="Amit Singh to step down as CEO of Adani Green Energy; Ashish Khanna to take over from April 2025",
            source_url="https://m.economictimes.com/industry/renewables/amit-singh-to-step-down-as-ceo-of-adani-green-energy-ashish-khanna-to-take-over-from-april-2025/articleshow/118760637.cms?utm_source=test",
            domain="economictimes.com",
            source_type="general_web",
            record_key="general_web:economictimes.com",
        ),
        WebSearchSource(
            source_title="adani.com",
            source_url="https://www.adani.com/en/about-us/leadership/ashish-khanna",
            domain="adani.com",
            source_type="general_web",
            record_key="general_web:adani.com",
        ),
        WebSearchSource(
            source_title="adani.com",
            source_url="https://www.adani.com/en/our-businesses/renewable-energy",
            domain="adani.com",
            source_type="general_web",
            record_key="general_web:adani.com",
        ),
        WebSearchSource(
            source_title="zaubacorp.com",
            source_url="https://www.zaubacorp.com/company/ADANI-GREEN-ENERGY-LIMITED/U40106GJ2015PLC082007",
            domain="zaubacorp.com",
            source_type="general_web",
            record_key="general_web:zaubacorp.com",
        ),
        WebSearchSource(
            source_title="zaubacorp.com",
            source_url="https://www.zaubacorp.com/company/IRRELEVANT-PRIVATE-LIMITED/U00000GJ2015PTC000000",
            domain="zaubacorp.com",
            source_type="general_web",
            record_key="general_web:zaubacorp.com",
        ),
        WebSearchSource(
            source_title="Example leadership page",
            source_url="https://example.com/leadership/jane-doe",
            domain="example.com",
            source_type="general_web",
            record_key="general_web:example.com",
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
