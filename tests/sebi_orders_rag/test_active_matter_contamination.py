from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.retrieval.scoring import HierarchicalSearchResult
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class ActiveMatterContaminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(REAL_CONTROL_PACK)
        assert cls.pack is not None

    def test_settlement_amount_follow_up_returns_scoped_negative_answer(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:100486",),
            active_document_version_ids=(100486,),
            active_primary_title="Exemption order in the matter of Hardcastle and Waud Manufacturing Ltd.",
        )
        search_service = _ScopedOnlySearchService()
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=REAL_CONTROL_PACK,
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=search_service,
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(state=state),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoOpMetadataService(),
            llm_client=_ExplodingLlmClient(),
        )

        payload = service.answer_query(
            query="What was the settlement amount?",
            session_id=session_id,
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("does not appear to be a settlement matter", payload.answer_text)
        self.assertFalse(search_service.scoped_calls)
        self.assertFalse(search_service.unscoped_called)
        self.assertEqual(payload.retrieved_chunk_ids, ())
        self.assertTrue(payload.debug["metadata_debug"]["active_scope_negative_answer"])

    def test_exemption_follow_up_returns_scoped_negative_answer_outside_exemption_matter(self) -> None:
        payload = self._answer_with_active_title(
            active_title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
            query="What exemption was granted?",
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("does not appear to be an exemption matter", payload.answer_text)
        self.assertTrue(payload.debug["metadata_debug"]["active_scope_negative_answer"])

    def test_appellate_follow_up_returns_scoped_negative_answer_outside_appellate_matter(self) -> None:
        payload = self._answer_with_active_title(
            active_title="Order in the matter of DU Digital Technologies Limited",
            query="What did the appellate authority decide?",
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("does not appear to be an appellate, SAT, or court matter", payload.answer_text)
        self.assertTrue(payload.debug["metadata_debug"]["active_scope_negative_answer"])

    def test_strong_new_named_query_overrides_stale_session_memory(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.pack)
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_record_keys=("external:100486",),
            active_document_version_ids=(100486,),
            active_primary_title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
        )

        decision = router.decide(
            query="alchemist infra vs sebi",
            session_state=state,
        )

        self.assertTrue(decision.analysis.fresh_query_override)
        self.assertFalse(decision.analysis.active_order_override)
        self.assertNotEqual(decision.route_mode, "memory_scoped_rag")
        self.assertEqual(
            decision.analysis.strict_lock_record_keys,
            ("external:29990",),
        )

    def _answer_with_active_title(self, *, active_title: str, query: str):
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:scoped",),
            active_document_version_ids=(999,),
            active_primary_title=active_title,
        )
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                control_pack_root=REAL_CONTROL_PACK,
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ScopedOnlySearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(state=state),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=_NoOpMetadataService(),
            llm_client=_ExplodingLlmClient(),
        )
        return service.answer_query(query=query, session_id=session_id)


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


class _ScopedOnlySearchService:
    def __init__(self) -> None:
        self.scoped_calls: list[tuple[int, ...]] = []
        self.unscoped_called = False

    def search(self, *, query: str, filters=None, top_k_docs=None, top_k_sections=None, top_k_chunks=None, strict_matter_lock=None):
        if filters is None or not getattr(filters, "document_version_ids", ()):
            self.unscoped_called = True
            raise AssertionError("memory-scoped follow-up should not fall back to global retrieval")
        self.scoped_calls.append(tuple(filters.document_version_ids))
        return HierarchicalSearchResult(
            query=query,
            documents=(),
            sections=(),
            chunks=(),
            debug={"scoped_document_version_ids": list(filters.document_version_ids)},
        )


class _NoOpMetadataService:
    def answer_signatory_question(self, *, document_version_ids):
        return None

    def answer_order_date_question(self, *, document_version_ids):
        return None

    def answer_legal_provisions_question(self, *, document_version_ids, explain: bool):
        return None

    def answer_active_matter_follow_up(self, *, query: str, document_version_ids, follow_up_intent: str):
        return None

    def answer_exact_fact_question(self, *, query: str, document_version_ids):
        return None

    def answer_numeric_fact_question(self, *, query: str, document_version_ids):
        return None

    def answer_observation_question(self, *, query: str, document_version_ids):
        return None


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("LLM should not be called when scoped retrieval has no support")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
