from __future__ import annotations

import unittest

from app.sebi_orders_rag.answering.style import apply_grounded_wording_caution
from app.sebi_orders_rag.router.query_analyzer import analyze_query
from app.sebi_orders_rag.schemas import PromptContextChunk


class WordingCautionTests(unittest.TestCase):
    def test_individual_holding_query_prefers_person_over_family_trust(self) -> None:
        chunk = PromptContextChunk(
            citation_number=1,
            chunk_id=1,
            document_version_id=10,
            document_id=10,
            record_key="external:mint",
            bucket_name="orders",
            title="Mint Investment Limited",
            page_start=3,
            page_end=3,
            section_type="findings",
            section_title=None,
            detail_url=None,
            pdf_url=None,
            chunk_text=(
                "Mrs. Aruna Dhanuka individually held 5,65,818 shares = 10.21%. "
                "Aruna Dhanuka Family Trust was the proposed acquirer proposed to hold 13,25,880 shares = 23.93% "
                "under an exemption from open offer."
            ),
            token_count=40,
            score=0.9,
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="The order says Aruna Dhanuka is the owner of 23.93% shares.",
            context_chunks=(chunk,),
            analysis=analyze_query("Mint investment limited how much shares does aruna dhanuka own"),
        )

        self.assertIn("Mrs. Aruna Dhanuka individually held 5,65,818 shares (10.21%).", answer_text)
        self.assertIn("Aruna Dhanuka Family Trust", answer_text)
        self.assertNotIn("owner", answer_text.lower())
        self.assertTrue(debug["used"])
        self.assertEqual(debug["holding_selected_subject"], "Mrs. Aruna Dhanuka")

    def test_trust_holding_query_keeps_proposed_acquirer_wording(self) -> None:
        chunk = PromptContextChunk(
            citation_number=1,
            chunk_id=1,
            document_version_id=10,
            document_id=10,
            record_key="external:mint",
            bucket_name="orders",
            title="Mint Investment Limited",
            page_start=3,
            page_end=3,
            section_type="findings",
            section_title=None,
            detail_url=None,
            pdf_url=None,
            chunk_text=(
                "Mrs. Aruna Dhanuka individually held 5,65,818 shares = 10.21%. "
                "Aruna Dhanuka Family Trust was the proposed acquirer proposed to hold "
                "13,25,880 shares = 23.93% under an exemption from open offer."
            ),
            token_count=40,
            score=0.9,
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="The trust holds 23.93% shares.",
            context_chunks=(chunk,),
            analysis=analyze_query(
                "how much shares does aruna dhanuka family trust hold in mint investment limited"
            ),
        )

        self.assertIn(
            "Aruna Dhanuka Family Trust was the proposed acquirer proposed to hold 13,25,880 shares (23.93%).",
            answer_text,
        )
        self.assertTrue(debug["used"])
        self.assertEqual(debug["holding_selected_subject"], "Aruna Dhanuka Family Trust")
        self.assertTrue(debug["selected_holding_proposed"])

    def test_trust_holding_query_infers_proposed_from_tabular_context(self) -> None:
        chunk = PromptContextChunk(
            citation_number=1,
            chunk_id=3,
            document_version_id=12,
            document_id=12,
            record_key="external:mint-table",
            bucket_name="orders",
            title="Mint Investment Limited",
            page_start=5,
            page_end=6,
            section_type="operative_order",
            section_title=None,
            detail_url=None,
            pdf_url=None,
            chunk_text=(
                "Acquirer and PACs (Also part of the Promoter and Promoter Group) "
                "Aruna Dhanuka Family Trust - - 13,25,880 23.93 "
                "Name Shareholding Before The Proposed Transaction Proposed Transaction "
                "Shareholding After The Proposed Transaction. "
                "The Acquirer Trust would be the legal owner of 23.93% of the total equity share capital of MIL."
            ),
            token_count=70,
            score=0.91,
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="The trust holds 23.93% shares.",
            context_chunks=(chunk,),
            analysis=analyze_query(
                "how much shares does aruna dhanuka family trust hold in mint investment limited"
            ),
        )

        self.assertIn("was the proposed acquirer proposed to hold 13,25,880 shares (23.93%)", answer_text)
        self.assertTrue(debug["selected_holding_proposed"])

    def test_interim_context_prefixes_prima_facie_caution(self) -> None:
        chunk = PromptContextChunk(
            citation_number=1,
            chunk_id=2,
            document_version_id=11,
            document_id=11,
            record_key="external:interim",
            bucket_name="orders",
            title="Interim Matter",
            page_start=5,
            page_end=5,
            section_type="findings",
            section_title=None,
            detail_url=None,
            pdf_url=None,
            chunk_text="The whole time member recorded prima facie findings in the interim order.",
            token_count=20,
            score=0.88,
        )

        answer_text, debug = apply_grounded_wording_caution(
            answer_text="The order finally held that the parties violated the regulations.",
            context_chunks=(chunk,),
            analysis=analyze_query("what happened in this case"),
        )

        self.assertIn("prima facie or interim observations", answer_text)
        self.assertTrue(debug["used"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
