from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

from app.sebi_orders_rag.qa.chunk_audit import (
    HEADING_ONLY_CHUNK_FLAG,
    SHORT_DOC_OVERFRAGMENTED_FLAG,
    SUSPICIOUS_SECTION_JUMP_FLAG,
    TINY_CHUNK_FLAG,
    ChunkAuditAnalyzer,
    calculate_severity_score,
    is_heading_only_chunk_text,
)
from app.sebi_orders_rag.repositories.qa import ChunkRow, ProcessedDocumentVersionRow


class ChunkAuditTests(unittest.TestCase):
    def test_tiny_chunk_flagging_skips_exempt_section_types(self) -> None:
        analyzer = ChunkAuditAnalyzer(oversized_chunk_tokens=1000)
        document = self._document(page_count=4)

        result = analyzer.audit_document(
            document=document,
            chunks=(
                self._chunk(
                    chunk_index=0,
                    section_type="other",
                    text="Brief procedural note.",
                    token_count=18,
                ),
                self._chunk(
                    chunk_index=1,
                    section_type="header",
                    text="Appeal No. 6798 of 2026",
                    token_count=11,
                ),
            ),
        )

        self.assertEqual(result.chunk_flag_counts[TINY_CHUNK_FLAG], 1)
        self.assertIn(TINY_CHUNK_FLAG, result.chunks[0].flags)
        self.assertNotIn(TINY_CHUNK_FLAG, result.chunks[1].flags)

    def test_heading_only_chunk_detection(self) -> None:
        self.assertTrue(is_heading_only_chunk_text("ORDER", token_count=1))
        self.assertTrue(
            is_heading_only_chunk_text(
                "APPELLATE AUTHORITY UNDER THE RTI ACT",
                token_count=10,
            )
        )
        self.assertFalse(
            is_heading_only_chunk_text(
                "CPIO, SEBI, Mumbai : Respondent",
                token_count=11,
            )
        )

    def test_short_doc_overfragmented_detection(self) -> None:
        analyzer = ChunkAuditAnalyzer(oversized_chunk_tokens=1000)
        document = self._document(page_count=3)
        chunks = tuple(
            self._chunk(
                chunk_index=index,
                section_type="operative_order",
                text=f"Chunk {index} contains a stable operative passage.",
                token_count=140,
            )
            for index in range(7)
        )

        result = analyzer.audit_document(document=document, chunks=chunks)

        self.assertIn(SHORT_DOC_OVERFRAGMENTED_FLAG, result.document_flags)

    def test_severity_scoring(self) -> None:
        score = calculate_severity_score(
            chunk_flag_counts={
                "oversized_chunk": 2,
                "heading_only_chunk": 1,
                "tiny_chunk": 3,
            },
            document_flag_counts={
                "duplicate_chunk_text_in_doc": 1,
                "short_doc_overfragmented": 1,
                "chunk_density_high": 1,
                "suspicious_section_jump": 2,
            },
        )

        self.assertEqual(score, 23)

    def test_ordered_chunk_inspection_marks_section_jump(self) -> None:
        analyzer = ChunkAuditAnalyzer(oversized_chunk_tokens=1000)
        document = self._document(page_count=2)
        findings_text = " ".join(["The findings paragraph is materially detailed."] * 10)

        result = analyzer.audit_document(
            document=document,
            chunks=(
                self._chunk(
                    chunk_index=0,
                    section_type="findings",
                    text=findings_text,
                    token_count=200,
                    section_title="FINDINGS",
                ),
                self._chunk(
                    chunk_index=1,
                    section_type="other",
                    text="ANNEXURE A",
                    token_count=2,
                    section_title="ANNEXURE A",
                ),
                self._chunk(
                    chunk_index=2,
                    section_type="findings",
                    text=findings_text,
                    token_count=190,
                    section_title="FINDINGS",
                ),
            ),
        )

        self.assertEqual([chunk.chunk_index for chunk in result.chunks], [0, 1, 2])
        self.assertIn(SUSPICIOUS_SECTION_JUMP_FLAG, result.document_flags)
        self.assertEqual(result.document_flag_counts[SUSPICIOUS_SECTION_JUMP_FLAG], 1)
        self.assertIn(TINY_CHUNK_FLAG, result.chunks[1].flags)
        self.assertIn(HEADING_ONLY_CHUNK_FLAG, result.chunks[1].flags)
        self.assertIn(SUSPICIOUS_SECTION_JUMP_FLAG, result.chunks[1].flags)
        self.assertTrue(result.chunks[0].first_text_preview.startswith("The findings paragraph"))

    @staticmethod
    def _document(*, page_count: int) -> ProcessedDocumentVersionRow:
        return ProcessedDocumentVersionRow(
            document_version_id=1,
            document_id=1,
            record_key="external:100728",
            bucket_name="orders-of-aa-under-rti-act",
            order_date=date(2026, 4, 2),
            title="Appeal No. 6800 & 6801 of 2026 filed by Ganga Prasad",
            page_count=page_count,
            chunk_count=0,
            ingested_at=datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
            created_at=datetime(2026, 4, 10, 11, 0, tzinfo=timezone.utc),
        )

    @staticmethod
    def _chunk(
        *,
        chunk_index: int,
        section_type: str,
        text: str,
        token_count: int,
        section_title: str | None = None,
    ) -> ChunkRow:
        heading_path = (section_title,) if section_title else ()
        return ChunkRow(
            chunk_index=chunk_index,
            page_start=1,
            page_end=1,
            section_type=section_type,
            section_title=section_title,
            heading_path=heading_path,
            chunk_text=text,
            chunk_sha256=f"sha-{chunk_index}",
            token_count=token_count,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
