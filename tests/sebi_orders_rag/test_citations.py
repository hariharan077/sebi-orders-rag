from __future__ import annotations

import unittest

from app.sebi_orders_rag.answering.citations import (
    build_citations,
    extract_citation_numbers,
    filter_citations,
)
from app.sebi_orders_rag.schemas import PromptContextChunk


class CitationTests(unittest.TestCase):
    def test_builds_and_filters_citations_from_context_chunks(self) -> None:
        context_chunks = (
            PromptContextChunk(
                citation_number=1,
                chunk_id=11,
                document_version_id=101,
                document_id=201,
                record_key="external:100725",
                bucket_name="orders-of-aa-under-rti-act",
                title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                page_start=1,
                page_end=2,
                section_type="operative_order",
                section_title="ORDER",
                detail_url="https://example.com/detail/100725",
                pdf_url="https://example.com/pdf/100725.pdf",
                chunk_text="The appeal is disposed of with directions.",
                token_count=42,
                score=0.08,
            ),
            PromptContextChunk(
                citation_number=2,
                chunk_id=12,
                document_version_id=101,
                document_id=201,
                record_key="external:100725",
                bucket_name="orders-of-aa-under-rti-act",
                title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                page_start=2,
                page_end=3,
                section_type="findings",
                section_title="FINDINGS",
                detail_url="https://example.com/detail/100725",
                pdf_url="https://example.com/pdf/100725.pdf",
                chunk_text="SEBI found no further violation.",
                token_count=37,
                score=0.07,
            ),
        )

        citations = build_citations(context_chunks)
        cited_numbers = extract_citation_numbers("The appeal was disposed of [1].")
        filtered = filter_citations(citations, cited_numbers=cited_numbers)

        self.assertEqual(len(citations), 2)
        self.assertEqual(cited_numbers, (1,))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk_id, 11)
        self.assertEqual(filtered[0].detail_url, "https://example.com/detail/100725")
        self.assertEqual(filtered[0].pdf_url, "https://example.com/pdf/100725.pdf")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
