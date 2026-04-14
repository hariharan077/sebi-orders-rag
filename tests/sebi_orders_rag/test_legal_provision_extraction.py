from __future__ import annotations

import unittest

from app.sebi_orders_rag.metadata.legal_provisions import extract_legal_provisions
from app.sebi_orders_rag.metadata.models import MetadataChunkText


class LegalProvisionExtractionTests(unittest.TestCase):
    def test_extracts_sections_and_regulations_from_representative_text(self) -> None:
        rows = extract_legal_provisions(
            document_version_id=99753,
            chunks=(
                MetadataChunkText(
                    chunk_id=1,
                    page_start=4,
                    page_end=5,
                    text=(
                        "The Noticee is alleged to have violated Sections 12A(a), 12A(b) and 12A(c) "
                        "of the SEBI Act and Regulations 3(1), 4(1) and 4(2)(a) of the PFUTP Regulations."
                    ),
                ),
            ),
        )

        extracted_refs = {(row.statute_name, row.section_or_regulation) for row in rows}
        self.assertIn(("SEBI Act, 1992", "Section 12A(a)"), extracted_refs)
        self.assertIn(("SEBI Act, 1992", "Section 12A(b)"), extracted_refs)
        self.assertIn(("PFUTP Regulations", "Regulation 3(1)"), extracted_refs)
        self.assertIn(("PFUTP Regulations", "Regulation 4(2)(a)"), extracted_refs)
        self.assertTrue(all(row.page_start == 4 for row in rows))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
