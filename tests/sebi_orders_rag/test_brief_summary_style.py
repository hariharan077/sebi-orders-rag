from __future__ import annotations

import unittest

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.answering.style import apply_grounded_wording_caution
from app.sebi_orders_rag.retrieval.scoring import ChunkSearchHit, ScoreBreakdown
from app.sebi_orders_rag.router.query_analyzer import analyze_query
from app.sebi_orders_rag.schemas import PromptContextChunk


class BriefSummaryStyleTests(unittest.TestCase):
    def test_brief_summary_uses_multi_sentence_case_summary(self) -> None:
        context_chunks = (
            PromptContextChunk(
                citation_number=1,
                chunk_id=11,
                document_version_id=601,
                document_id=601,
                record_key="external:yash-garg",
                bucket_name="orders-of-whole-time-member",
                title="Order in the matter of Yash Garg",
                page_start=2,
                page_end=2,
                section_type="facts",
                section_title="Facts",
                detail_url=None,
                pdf_url=None,
                chunk_text=(
                    "The proceedings arose from suspicious trading activity in the scrip and the investigation examined the noticee's trades."
                ),
                token_count=20,
                score=0.88,
            ),
            PromptContextChunk(
                citation_number=2,
                chunk_id=12,
                document_version_id=601,
                document_id=601,
                record_key="external:yash-garg",
                bucket_name="orders-of-whole-time-member",
                title="Order in the matter of Yash Garg",
                page_start=5,
                page_end=5,
                section_type="findings",
                section_title="Findings",
                detail_url=None,
                pdf_url=None,
                chunk_text=(
                    "SEBI observed manipulative and misleading trading patterns and concluded that the noticee violated the PFUTP Regulations."
                ),
                token_count=22,
                score=0.92,
            ),
            PromptContextChunk(
                citation_number=3,
                chunk_id=13,
                document_version_id=601,
                document_id=601,
                record_key="external:yash-garg",
                bucket_name="orders-of-whole-time-member",
                title="Order in the matter of Yash Garg",
                page_start=8,
                page_end=8,
                section_type="operative_order",
                section_title="Order",
                detail_url=None,
                pdf_url=None,
                chunk_text=(
                    "Accordingly, the noticee was restrained from accessing the securities market and directions were issued."
                ),
                token_count=18,
                score=0.94,
            ),
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="SEBI observed manipulative trading patterns.",
            context_chunks=context_chunks,
            analysis=analyze_query("Give a brief summary of what happened in the Yash Garg case"),
        )

        self.assertIn("This matter concerns Order in the matter of Yash Garg.", answer_text)
        self.assertIn("suspicious trading activity", answer_text)
        self.assertIn("violated the PFUTP Regulations", answer_text)
        self.assertIn("accessing the securities market", answer_text)
        self.assertNotIn("Whether the acts", answer_text)
        self.assertGreaterEqual(answer_text.count("."), 3)
        self.assertTrue(debug["brief_summary_requested"])

    def test_brief_summary_ignores_uncited_chunk_boundary_fragments(self) -> None:
        context_chunks = (
            PromptContextChunk(
                citation_number=1,
                chunk_id=21,
                document_version_id=99695,
                document_id=99695,
                record_key="external:99695",
                bucket_name="orders-of-whole-time-member",
                title="CONFIRMATORY ORDER IN THE MATTER OF M/S PATEL WEALTH ADVISORS PRIVATE LIMITED",
                page_start=1,
                page_end=2,
                section_type="facts",
                section_title="Facts",
                detail_url=None,
                pdf_url=None,
                chunk_text=(
                    "The matter concerns allegations that the noticees repeatedly placed and cancelled large orders in multiple scrips after opposite-side trades were executed."
                ),
                token_count=25,
                score=0.93,
            ),
            PromptContextChunk(
                citation_number=2,
                chunk_id=22,
                document_version_id=99695,
                document_id=99695,
                record_key="external:99695",
                bucket_name="orders-of-whole-time-member",
                title="CONFIRMATORY ORDER IN THE MATTER OF M/S PATEL WEALTH ADVISORS PRIVATE LIMITED",
                page_start=18,
                page_end=19,
                section_type="operative_order",
                section_title="Order",
                detail_url=None,
                pdf_url=None,
                chunk_text=(
                    "SEBI confirmed the interim directions against Noticee No. 1, revoked the interim directions against Noticee Nos. 2 to 5, and allowed the restraint on those noticees to cease."
                ),
                token_count=28,
                score=0.96,
            ),
        )
        retrieved_chunks = (
            _chunk_search_hit(
                chunk_id=1371,
                page_start=4,
                page_end=6,
                section_type="operative_order",
                chunk_text="Further, this direction shall not apply.",
            ),
            _chunk_search_hit(
                chunk_id=1389,
                page_start=16,
                page_end=17,
                section_type="findings",
                chunk_text="facie findings recorded in the Interim Order.",
            ),
        )
        analysis = analyze_query("summarise the patel wealth advisors case")

        style_context = AdaptiveRagAnswerService._build_style_context_chunks(
            object.__new__(AdaptiveRagAnswerService),
            context_chunks=context_chunks,
            cited_context_chunks=context_chunks,
            retrieved_chunks=retrieved_chunks,
            analysis=analysis,
        )
        answer_text, debug = apply_grounded_wording_caution(
            answer_text="The matter concerns spoofing allegations.",
            context_chunks=style_context,
            analysis=analysis,
        )

        self.assertIn("repeatedly placed and cancelled large orders", answer_text)
        self.assertIn("confirmed the interim directions", answer_text.lower())
        self.assertNotIn("Further, this direction shall not apply.", answer_text)
        self.assertNotIn("facie findings recorded in the Interim Order.", answer_text)
        self.assertEqual({chunk.chunk_id for chunk in style_context}, {21, 22})
        self.assertTrue(debug["brief_summary_requested"])


def _chunk_search_hit(
    *,
    chunk_id: int,
    page_start: int,
    page_end: int,
    section_type: str,
    chunk_text: str,
) -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        document_version_id=99695,
        document_id=99695,
        record_key="external:99695",
        bucket_name="orders-of-whole-time-member",
        external_record_id=None,
        order_date=None,
        title="CONFIRMATORY ORDER IN THE MATTER OF M/S PATEL WEALTH ADVISORS PRIVATE LIMITED",
        chunk_index=chunk_id,
        page_start=page_start,
        page_end=page_end,
        section_key=None,
        section_type=section_type,
        section_title=section_type.title(),
        heading_path=None,
        detail_url=None,
        pdf_url=None,
        chunk_text=chunk_text,
        token_count=max(len(chunk_text.split()), 1),
        score=ScoreBreakdown(final_score=0.5),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
