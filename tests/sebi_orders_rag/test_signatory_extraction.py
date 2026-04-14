from __future__ import annotations

import unittest
from datetime import date

from app.sebi_orders_rag.metadata.models import MetadataPageText
from app.sebi_orders_rag.metadata.signatory import extract_signatory_metadata


class SignatoryExtractionTests(unittest.TestCase):
    def test_extracts_signatory_footer_metadata(self) -> None:
        pages = (
            MetadataPageText(page_no=1, text="Order text\nDate: February 12, 2026\n"),
            MetadataPageText(
                page_no=12,
                text="""
                    Place: Mumbai
                    Sd/-
                    Ananth Narayan G.
                    Whole Time Member
                    Securities and Exchange Board of India
                """,
            ),
        )

        extracted = extract_signatory_metadata(
            document_version_id=99753,
            pages=pages,
            fallback_order_date=date(2026, 2, 12),
        )

        self.assertEqual(extracted.signatory_name, "Ananth Narayan G.")
        self.assertEqual(extracted.signatory_designation, "Whole Time Member")
        self.assertEqual(extracted.order_date, date(2026, 2, 12))
        self.assertEqual(extracted.place, "Mumbai")
        self.assertEqual(extracted.issuing_authority_type, "board_member")
        self.assertEqual(extracted.signatory_page_start, 12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
