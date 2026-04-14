from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.retrieval.scoring import ChunkSearchHit, HierarchicalSearchResult, ScoreBreakdown
from app.sebi_orders_rag.schemas import ChatSessionSnapshot
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class MixedRecordGuardrailTests(unittest.TestCase):
    def test_strict_query_prefilters_mixed_retrieval_and_abstains_on_invalid_mixed_citations(self) -> None:
        service = _build_service(
            llm_payload={
                "answer_status": "answered",
                "answer_text": "This mixes matters [1][2].",
                "cited_numbers": [1, 2],
            }
        )

        payload = service.answer_query(
            query="Tell me more about the IPO of Vishvaraj Environment Limited",
            session_id=uuid4(),
        )

        self.assertEqual(payload.route_mode, "abstain")
        self.assertEqual(payload.answer_status, "abstained")
        self.assertEqual(payload.citations, ())
        self.assertTrue(
            payload.debug["mixed_record_guardrail"]["mixed_record_guardrail_fired"]
        )
        self.assertIn(
            "derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1",
            payload.debug["mixed_record_guardrail"]["retrieved_record_keys_before_filter"],
        )
        self.assertIn(
            "external:98714",
            payload.debug["mixed_record_guardrail"]["retrieved_record_keys_before_filter"],
        )


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeSearchService:
    def search(self, *, query: str, filters=None, top_k_docs=None, top_k_sections=None, top_k_chunks=None, strict_matter_lock=None):
        return HierarchicalSearchResult(
            query=query,
            documents=(),
            sections=(),
            chunks=(
                _chunk_hit(
                    chunk_id=1001,
                    record_key="external:98714",
                    title="Revocation Order in the matter of Varyaa Creations Limited",
                    document_version_id=9001,
                    score=0.095,
                ),
                _chunk_hit(
                    chunk_id=1002,
                    record_key="derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1",
                    title="Order in the matter of Vishvaraj Environment Limited",
                    document_version_id=9002,
                    score=0.090,
                ),
            ),
            debug={
                "strict_single_matter": bool(strict_matter_lock and strict_matter_lock.strict_single_matter),
            },
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


def _chunk_hit(
    *,
    chunk_id: int,
    record_key: str,
    title: str,
    document_version_id: int,
    score: float,
) -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        document_version_id=document_version_id,
        document_id=document_version_id,
        record_key=record_key,
        bucket_name="orders-of-chairperson-members",
        external_record_id=record_key.split(":")[-1],
        order_date=date(2026, 3, 20),
        title=title,
        chunk_index=0,
        page_start=1,
        page_end=2,
        section_key="section-1",
        section_type="operative_order",
        section_title="ORDER",
        heading_path="ORDER",
        detail_url=f"https://example.com/detail/{document_version_id}",
        pdf_url=f"https://example.com/pdf/{document_version_id}.pdf",
        chunk_text="SEBI recorded operative findings and directions in this matter.",
        token_count=48,
        score=ScoreBreakdown(
            combined_score=score,
            final_score=score,
        ),
    )


def _build_service(*, llm_payload) -> AdaptiveRagAnswerService:
    settings = SebiOrdersRagSettings(
        db_dsn="postgresql://unused",
        data_root=Path(".").resolve(),
        control_pack_root=REAL_CONTROL_PACK,
        low_confidence_threshold=0.35,
        enable_memory=True,
    )
    return AdaptiveRagAnswerService(
        settings=settings,
        connection=_FakeConnection(),
        search_service=_FakeSearchService(),
        retrieval_repository=_FakeRetrievalRepository(),
        session_repository=_FakeSessionRepository(),
        answer_repository=_FakeAnswerRepository(),
        llm_client=_FakeLlmClient(llm_payload),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
