from __future__ import annotations

import unittest

from app.sebi_orders_rag.answering.style import apply_grounded_wording_caution
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
