from __future__ import annotations

import unittest
from pathlib import Path

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.answering.confidence import assess_retrieval_confidence
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.retrieval.query_intent import QueryIntent, QueryIntentResult
from app.sebi_orders_rag.retrieval.scoring import ChunkSearchHit, HierarchicalSearchResult, ScoreBreakdown
from app.sebi_orders_rag.schemas import PromptContextChunk


class SettlementQueryTests(unittest.TestCase):
    def test_select_context_chunks_prefers_matching_settlement_body_material(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                max_context_chunks=4,
            ),
            connection=_FakeConnection(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            llm_client=_FakeLlmClient(),
        )
        search_result = HierarchicalSearchResult(
            query="What was the settlement amount in the JP Morgan settlement order?",
            documents=(),
            sections=(),
            chunks=(
                _chunk_hit(
                    chunk_id=1,
                    title="Settlement Order in the matter of Sixteenth Street Asian GEMS Fund",
                    score=0.0660,
                    chunk_text="SETTLEMENT ORDER IN RESPECT OF Sixteenth Street Asian GEMS Fund",
                    token_count=18,
                ),
                _chunk_hit(
                    chunk_id=2,
                    title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                    score=0.0750,
                    chunk_text=(
                        "The Applicant remitted the said settlement amount after Notice of Demand "
                        "and it is hereby ordered that no enforcement action shall be initiated."
                    ),
                    token_count=52,
                ),
                _chunk_hit(
                    chunk_id=3,
                    title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                    score=0.0610,
                    chunk_text="SETTLEMENT ORDER IN RESPECT OF JP Morgan Chase Bank N.A.",
                    token_count=17,
                ),
            ),
            query_intent=QueryIntentResult(
                intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
                matched_terms=("settlement",),
                settlement_terms=("settlement",),
                entity_terms=("jp", "morgan"),
                settlement_focused=True,
            ),
        )

        context_chunks = service._select_context_chunks(search_result)

        self.assertEqual(context_chunks[0].chunk_id, 2)
        self.assertEqual(context_chunks[0].record_key, "external:101")

    def test_confidence_tuning_accepts_strong_settlement_support(self) -> None:
        context = (
            PromptContextChunk(
                citation_number=1,
                chunk_id=11,
                document_version_id=101,
                document_id=201,
                record_key="external:100486",
                bucket_name="settlement-orders",
                title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                page_start=2,
                page_end=3,
                section_type="operative_order",
                section_title="SETTLEMENT ORDER",
                detail_url="https://example.com/detail/jpmorgan",
                pdf_url="https://example.com/pdf/jpmorgan.pdf",
                chunk_text="Notice of Demand was issued and the Applicant remitted the said settlement amount.",
                token_count=42,
                score=0.046,
            ),
            PromptContextChunk(
                citation_number=2,
                chunk_id=12,
                document_version_id=101,
                document_id=201,
                record_key="external:100486",
                bucket_name="settlement-orders",
                title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                page_start=2,
                page_end=3,
                section_type="operative_order",
                section_title="SETTLEMENT ORDER",
                detail_url="https://example.com/detail/jpmorgan",
                pdf_url="https://example.com/pdf/jpmorgan.pdf",
                chunk_text="It is hereby ordered that no enforcement action shall be initiated.",
                token_count=38,
                score=0.041,
            ),
        )

        assessment = assess_retrieval_confidence(
            context_chunks=context,
            cited_context_chunks=context,
            answer_status="answered",
            threshold=0.35,
        )

        self.assertFalse(assessment.should_abstain)
        self.assertGreaterEqual(assessment.confidence, 0.30)


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
    def create_session_if_missing(self, *, session_id, user_name):
        return None

    def get_session_snapshot(self, *, session_id):
        return None

    def get_session_state(self, *, session_id):
        return None


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _FakeLlmClient:
    def complete_json(self, prompt):
        return {}


def _chunk_hit(
    *,
    chunk_id: int,
    title: str,
    score: float,
    chunk_text: str,
    token_count: int,
) -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        document_version_id=101 if "JP Morgan" in title else 102,
        document_id=201 if "JP Morgan" in title else 202,
        record_key="external:101" if "JP Morgan" in title else "external:102",
        bucket_name="settlement-orders",
        external_record_id="100486" if "JP Morgan" in title else "100484",
        order_date=None,
        title=title,
        chunk_index=chunk_id,
        page_start=1,
        page_end=2,
        section_key="section-operative-order",
        section_type="operative_order",
        section_title="SETTLEMENT ORDER",
        heading_path="SETTLEMENT ORDER",
        detail_url=f"https://example.com/detail/{chunk_id}",
        pdf_url=f"https://example.com/pdf/{chunk_id}.pdf",
        chunk_text=chunk_text,
        token_count=token_count,
        score=ScoreBreakdown(final_score=score, combined_score=score),
    )
