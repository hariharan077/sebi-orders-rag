from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.control.models import MatterLockCandidate, StrictMatterLock
from app.sebi_orders_rag.retrieval.scoring import ChunkSearchHit, HierarchicalSearchResult, ScoreBreakdown
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, QueryAnalysis, RouteDecision


class SingleCandidateAutoresolveTests(unittest.TestCase):
    def test_single_named_matter_candidate_resolves_without_clarify(self) -> None:
        candidate = MatterLockCandidate(
            record_key="external:yash-garg",
            title="Order in the matter of Yash Garg",
            bucket_name="orders-of-whole-time-member",
            document_version_id=401,
            canonical_entities=("Yash Garg",),
            score=0.47,
            exact_title_match=False,
            matched_entity_terms=("yash garg",),
        )
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_FakeSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            router=_ForcedRouter(candidate=candidate),
            llm_client=_FakeLlmClient(
                {
                    "answer_status": "answered",
                    "answer_text": "SEBI recorded that the matter involved manipulative trading and passed directions.",
                    "cited_numbers": [1],
                }
            ),
        )

        payload = service.answer_query(
            query="Give a brief summary of what happened in the Yash Garg case",
            session_id=uuid4(),
        )

        self.assertNotEqual(payload.route_mode, "clarify")
        self.assertEqual(payload.route_mode, "exact_lookup")
        self.assertEqual(payload.answer_status, "answered")
        self.assertEqual(payload.active_record_keys, ("external:yash-garg",))


class _ForcedRouter:
    def __init__(self, *, candidate: MatterLockCandidate) -> None:
        self._candidate = candidate

    def decide(self, *, query: str, session_state=None) -> RouteDecision:
        strict_lock = StrictMatterLock(
            named_matter_query=True,
            strict_scope_required=True,
            strict_single_matter=False,
            ambiguous=True,
            matched_entities=("yash garg",),
            candidates=(self._candidate,),
            reason_codes=("named_matter_query", "ambiguous_named_matter"),
        )
        analysis = QueryAnalysis(
            raw_query=query,
            normalized_query=" ".join(query.lower().split()),
            query_family="named_order_query",
            strict_scope_required=True,
            strict_single_matter=False,
            strict_lock_ambiguous=True,
            strict_matter_lock=strict_lock,
            asks_brief_summary=True,
            appears_matter_specific=True,
        )
        return RouteDecision(
            route_mode="hierarchical_rag",
            query_intent="matter_specific",
            analysis=analysis,
            reason_codes=("named_matter_query",),
        )


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeSearchService:
    def search(self, *, query: str, filters=None, top_k_docs=None, top_k_sections=None, top_k_chunks=None, strict_matter_lock=None):
        del query, top_k_docs, top_k_sections, top_k_chunks, strict_matter_lock
        return HierarchicalSearchResult(
            query="yash garg",
            documents=(),
            sections=(),
            chunks=(
                ChunkSearchHit(
                    chunk_id=5001,
                    document_version_id=401,
                    document_id=901,
                    record_key="external:yash-garg",
                    bucket_name="orders-of-whole-time-member",
                    external_record_id="401",
                    order_date=None,
                    title="Order in the matter of Yash Garg",
                    chunk_index=0,
                    page_start=2,
                    page_end=2,
                    section_key="findings",
                    section_type="findings",
                    section_title="Findings",
                    heading_path="Findings",
                    detail_url="https://example.com/yash-garg",
                    pdf_url="https://example.com/yash-garg.pdf",
                    chunk_text="SEBI observed manipulative trading activity and passed directions against the noticee.",
                    token_count=24,
                    score=ScoreBreakdown(combined_score=0.91, final_score=0.91),
                ),
            ),
            debug={"forced_single_candidate": True, "filters": getattr(filters, "record_key", None)},
        )


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return (401,)


class _FakeSessionRepository:
    def __init__(self) -> None:
        self._snapshots = {}
        self._states = {}

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
        from app.sebi_orders_rag.schemas import ChatSessionStateRecord

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
    def __init__(self, payload) -> None:
        self._payload = payload

    def complete_json(self, prompt):
        return dict(self._payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
