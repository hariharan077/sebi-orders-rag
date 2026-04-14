from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.retrieval.scoring import ChunkSearchHit, HierarchicalSearchResult, ScoreBreakdown
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord, ExactLookupCandidate


class ClarifySelectionMemoryTests(unittest.TestCase):
    def test_numeric_selection_resolves_last_clarify_context(self) -> None:
        session_repository = _FakeSessionRepository()
        service = _build_service(session_repository=session_repository)
        session_id = uuid4()

        clarify_payload = service.answer_query(
            query="sebi vs ashima limited",
            session_id=session_id,
        )
        selected_payload = service.answer_query(query="1", session_id=session_id)

        self.assertEqual(clarify_payload.route_mode, "clarify")
        self.assertEqual(clarify_payload.answer_status, "clarify")
        self.assertEqual(len(clarify_payload.clarification_candidates), 2)
        self.assertEqual(selected_payload.route_mode, "exact_lookup")
        self.assertEqual(selected_payload.answer_status, "answered")
        self.assertEqual(selected_payload.active_record_keys, ("external:ashima-sat-2015",))
        self.assertIn("SAT appeal", selected_payload.answer_text)
        self.assertEqual(
            selected_payload.debug["clarification_debug"]["resolved_candidate_id"],
            "external:ashima-sat-2015",
        )
        self.assertIsNone(session_repository.get_session_state(session_id=session_id).clarification_context)

    def test_title_fragment_selection_resolves_last_clarify_context(self) -> None:
        session_repository = _FakeSessionRepository()
        service = _build_service(session_repository=session_repository)
        session_id = uuid4()

        clarify_payload = service.answer_query(
            query="prime broking vs nse",
            session_id=session_id,
        )
        selected_payload = service.answer_query(
            query="vs nse",
            session_id=session_id,
        )

        self.assertEqual(clarify_payload.route_mode, "clarify")
        self.assertEqual(selected_payload.route_mode, "exact_lookup")
        self.assertEqual(selected_payload.answer_status, "answered")
        self.assertEqual(selected_payload.active_record_keys, ("external:prime-broking-sat",))
        self.assertEqual(
            selected_payload.debug["clarification_debug"]["match_reason"],
            "selection_alias",
        )

    def test_unrelated_query_does_not_get_trapped_by_old_clarify_context(self) -> None:
        session_repository = _FakeSessionRepository()
        service = _build_service(session_repository=session_repository)
        session_id = uuid4()

        clarify_payload = service.answer_query(
            query="prime broking vs nse",
            session_id=session_id,
        )
        state = session_repository.get_session_state(session_id=session_id)
        selection = service._memory.resolve_clarification_selection(  # noqa: SLF001
            query="yash trading academy",
            state=state,
        )

        self.assertEqual(clarify_payload.route_mode, "clarify")
        self.assertFalse(selection.active_context)
        self.assertEqual(selection.selected_candidates, ())

    def test_new_vs_query_does_not_get_misread_as_old_selection(self) -> None:
        session_repository = _FakeSessionRepository()
        service = _build_service(session_repository=session_repository)
        session_id = uuid4()

        clarify_payload = service.answer_query(
            query="prime broking vs nse",
            session_id=session_id,
        )
        state = session_repository.get_session_state(session_id=session_id)
        selection = service._memory.resolve_clarification_selection(  # noqa: SLF001
            query="umashanker vs sebi",
            state=state,
        )

        self.assertEqual(clarify_payload.route_mode, "clarify")
        self.assertFalse(selection.active_context)
        self.assertEqual(selection.selected_candidates, ())


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        normalized = " ".join(query.lower().split())
        if normalized == "sebi vs ashima limited":
            return [
                ExactLookupCandidate(
                    document_version_id=101,
                    document_id=201,
                    record_key="external:ashima-sat-2015",
                    bucket_name="orders-of-sat",
                    external_record_id="ashima-sat-2015",
                    order_date=date(2015, 8, 19),
                    title="SEBI vs Ashima Limited",
                    match_score=0.74,
                ),
                ExactLookupCandidate(
                    document_version_id=102,
                    document_id=202,
                    record_key="external:ashima-court-2016",
                    bucket_name="orders-of-courts",
                    external_record_id="ashima-court-2016",
                    order_date=date(2016, 2, 3),
                    title="SEBI vs Ashima Limited and Others",
                    match_score=0.73,
                ),
            ]
        if normalized == "prime broking vs nse":
            return [
                ExactLookupCandidate(
                    document_version_id=301,
                    document_id=401,
                    record_key="external:prime-broking-sat",
                    bucket_name="orders-of-sat",
                    external_record_id="prime-broking-sat",
                    order_date=date(2017, 7, 10),
                    title="Prime Broking Company (India) Limited vs NSE",
                    match_score=0.73,
                ),
                ExactLookupCandidate(
                    document_version_id=302,
                    document_id=402,
                    record_key="external:prime-broking-sebi",
                    bucket_name="orders-of-sat",
                    external_record_id="prime-broking-sebi",
                    order_date=date(2018, 1, 16),
                    title="Prime Broking Company (India) Limited vs SEBI",
                    match_score=0.71,
                ),
            ]
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        if record_keys == ("external:ashima-sat-2015",):
            return (101,)
        if record_keys == ("external:prime-broking-sat",):
            return (301,)
        return ()


class _FakeSearchService:
    def search(self, *, query: str, filters=None, top_k_docs=None, top_k_sections=None, top_k_chunks=None, strict_matter_lock=None):
        record_key = getattr(filters, "record_key", None)
        if record_key == "external:ashima-sat-2015":
            return _search_result(
                query=query,
                document_version_id=101,
                document_id=201,
                record_key=record_key,
                bucket_name="orders-of-sat",
                title="SEBI vs Ashima Limited",
                text="The SAT appeal was dismissed and the Tribunal upheld SEBI's findings.",
            )
        if record_key == "external:prime-broking-sat":
            return _search_result(
                query=query,
                document_version_id=301,
                document_id=401,
                record_key=record_key,
                bucket_name="orders-of-sat",
                title="Prime Broking Company (India) Limited vs NSE",
                text="The SAT order records that Prime Broking's appeal was dismissed.",
            )
        raise AssertionError(f"unexpected record_key filter: {record_key}")


class _FakeSessionRepository:
    def __init__(self) -> None:
        self._states: dict[object, ChatSessionStateRecord] = {}
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
                state=self._states.get(session_id),
            ),
        )

    def get_session_snapshot(self, *, session_id):
        snapshot = self._snapshots.get(session_id)
        if snapshot is None:
            return None
        return ChatSessionSnapshot(
            session_id=snapshot.session_id,
            user_name=snapshot.user_name,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            state=self._states.get(session_id),
        )

    def get_session_state(self, *, session_id):
        return self._states.get(session_id)

    def upsert_session_state(self, **kwargs):
        self._states[kwargs["session_id"]] = ChatSessionStateRecord(
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


class _FakeLlmClient:
    def complete_json(self, prompt):
        return {
            "answer_status": "answered",
            "answer_text": "The SAT appeal was dismissed [1].",
            "cited_numbers": [1],
        }


def _build_service(*, session_repository: _FakeSessionRepository) -> AdaptiveRagAnswerService:
    return AdaptiveRagAnswerService(
        settings=SebiOrdersRagSettings(
            db_dsn="postgresql://unused",
            data_root=Path(".").resolve(),
            enable_memory=True,
            low_confidence_threshold=0.35,
        ),
        connection=_FakeConnection(),
        search_service=_FakeSearchService(),
        retrieval_repository=_FakeRetrievalRepository(),
        session_repository=session_repository,
        answer_repository=_FakeAnswerRepository(),
        llm_client=_FakeLlmClient(),
    )


def _search_result(
    *,
    query: str,
    document_version_id: int,
    document_id: int,
    record_key: str,
    bucket_name: str,
    title: str,
    text: str,
) -> HierarchicalSearchResult:
    return HierarchicalSearchResult(
        query=query,
        documents=(),
        sections=(),
        chunks=(
            ChunkSearchHit(
                chunk_id=document_version_id * 10,
                document_version_id=document_version_id,
                document_id=document_id,
                record_key=record_key,
                bucket_name=bucket_name,
                external_record_id=record_key,
                order_date=None,
                title=title,
                chunk_index=0,
                page_start=1,
                page_end=2,
                section_key="section-1",
                section_type="operative_order",
                section_title="ORDER",
                heading_path="ORDER",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                chunk_text=text,
                token_count=32,
                score=ScoreBreakdown(combined_score=0.91, final_score=0.91),
            ),
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
