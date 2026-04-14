from __future__ import annotations

import unittest

from app.sebi_orders_rag.ingestion.text_normalizer import normalize_extracted_text


class TextNormalizerTests(unittest.TestCase):
    def test_normalizer_preserves_legal_numbering_and_headings(self) -> None:
        raw_text = (
            "ORDER\n\n"
            "1.   This is the first paragraph of the order.\n"
            "continued on the next line.\n\n"
            "(a)   The first sub-clause remains intact.\n"
            "and continues without losing numbering.\n\n"
            "Issue I\n"
            "Whether the notice was adequate.\n"
        )

        normalized = normalize_extracted_text(raw_text)

        self.assertIn("ORDER", normalized)
        self.assertIn(
            "1. This is the first paragraph of the order. continued on the next line.",
            normalized,
        )
        self.assertIn(
            "(a) The first sub-clause remains intact. and continues without losing numbering.",
            normalized,
        )
        self.assertIn("Issue I", normalized)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
