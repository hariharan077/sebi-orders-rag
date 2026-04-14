from __future__ import annotations

import unittest

from app.sebi_orders_rag.answering.style import apply_grounded_wording_caution
from app.sebi_orders_rag.router.query_analyzer import analyze_query
from app.sebi_orders_rag.schemas import PromptContextChunk


class SettlementAnswerShapingTests(unittest.TestCase):
    def test_settlement_findings_query_is_rewritten_with_allegation_and_disposal_caution(self) -> None:
        chunk = PromptContextChunk(
            citation_number=1,
            chunk_id=1,
            document_version_id=501,
            document_id=501,
            record_key="external:hemant-ghai",
            bucket_name="settlement-orders",
            title="Settlement Order in the matter of Hemant Ghai",
            page_start=4,
            page_end=5,
            section_type="operative_order",
            section_title="ORDER",
            detail_url=None,
            pdf_url=None,
            chunk_text=(
                "The show cause notice alleged that Hemant Ghai communicated material non-public information "
                "about stock recommendations to certain persons who traded and made unlawful gains. "
                "The applicant remitted the settlement amount of Rs. 1,45,60,000 pursuant to the notice of demand. "
                "In view of the settlement, the instant proceedings are disposed of."
            ),
            token_count=60,
            score=0.95,
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="SEBI found that Hemant Ghai communicated material non-public information.",
            context_chunks=(chunk,),
            analysis=analyze_query("what did sebi find in hemant ghai order?"),
        )

        self.assertIn("alleged", answer_text.lower())
        self.assertIn("Rs. 1,45,60,000", answer_text)
        self.assertIn("disposed of", answer_text.lower())
        self.assertNotIn("SEBI found", answer_text)
        self.assertEqual(debug["matter_type"], "settlement_order")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
