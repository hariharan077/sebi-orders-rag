from __future__ import annotations

import unittest

from app.sebi_orders_rag.ingestion.structure_parser import detect_heading, parse_document_structure
from app.sebi_orders_rag.ingestion.token_count import token_count
from app.sebi_orders_rag.schemas import ExtractedPage
from app.sebi_orders_rag.utils.strings import sha256_hexdigest


class StructureParserTests(unittest.TestCase):
    def test_detect_heading_classifies_representative_legal_headings(self) -> None:
        findings = detect_heading(
            "CONSIDERATION OF ISSUES AND FINDINGS",
            min_heading_caps_ratio=0.60,
        )
        issue = detect_heading(
            "Issue II: Whether the notice was served",
            min_heading_caps_ratio=0.60,
        )
        directions = detect_heading(
            "Directions",
            min_heading_caps_ratio=0.60,
        )
        settlement_order = detect_heading(
            "SETTLEMENT ORDER",
            min_heading_caps_ratio=0.60,
        )

        self.assertIsNotNone(findings)
        self.assertIsNotNone(issue)
        self.assertIsNotNone(directions)
        self.assertIsNotNone(settlement_order)
        assert findings is not None
        assert issue is not None
        assert directions is not None
        assert settlement_order is not None
        self.assertEqual(findings.section_type, "findings")
        self.assertEqual(findings.level, 1)
        self.assertEqual(issue.section_type, "issues")
        self.assertEqual(issue.level, 2)
        self.assertEqual(directions.section_type, "directions")
        self.assertEqual(settlement_order.section_type, "operative_order")

    def test_parse_document_structure_assigns_heading_paths(self) -> None:
        pages = (
            self._page(
                1,
                "BACKGROUND\n\n"
                "1. Brief facts\n\n"
                "The notice was issued to the intermediary.\n\n"
                "Issue I\n\n"
                "Whether the conduct violated the regulations.\n",
            ),
        )

        parsed = parse_document_structure(
            pages,
            min_heading_caps_ratio=0.60,
            model_name="text-embedding-3-large",
        )

        headings = [block for block in parsed.blocks if block.block_type == "heading"]
        paragraphs = [block for block in parsed.blocks if block.block_type == "paragraph"]

        self.assertEqual(len(headings), 3)
        self.assertEqual(headings[0].section_title, "BACKGROUND")
        self.assertEqual(headings[1].heading_path, ("BACKGROUND", "1. Brief facts"))
        self.assertEqual(headings[2].heading_path, ("BACKGROUND", "Issue I"))
        self.assertEqual(paragraphs[-1].section_title, "Issue I")
        self.assertEqual(paragraphs[-1].section_type, "issues")

    @staticmethod
    def _page(page_no: int, text: str) -> ExtractedPage:
        return ExtractedPage(
            page_no=page_no,
            extracted_text=text,
            ocr_text=None,
            final_text=text,
            char_count=len(text),
            token_count=token_count(text, model_name="text-embedding-3-large"),
            low_text=False,
            page_sha256=sha256_hexdigest(text),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
