from __future__ import annotations

import unittest

from app.sebi_orders_rag.ingestion.chunker import build_chunks
from app.sebi_orders_rag.ingestion.chunk_postprocess import postprocess_chunks
from app.sebi_orders_rag.ingestion.token_count import token_count
from app.sebi_orders_rag.schemas import ChunkRecord, ParsedDocument, StructuredBlock

MODEL_NAME = "text-embedding-3-large"


class ChunkerTests(unittest.TestCase):
    def test_chunker_does_not_exceed_hard_max(self) -> None:
        long_paragraph = " ".join(
            f"Sentence {index} describes the securities market conduct in detail."
            for index in range(1, 81)
        )
        parsed = ParsedDocument(
            blocks=(
                self._block(0, "heading", "BACKGROUND", "background", "BACKGROUND"),
                self._block(1, "paragraph", long_paragraph, "background", "BACKGROUND"),
            )
        )

        chunks = build_chunks(
            parsed,
            model_name=MODEL_NAME,
            target_chunk_tokens=100,
            max_chunk_tokens=120,
            overlap_tokens=20,
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 120)

    def test_chunker_preserves_major_section_boundaries(self) -> None:
        parsed = ParsedDocument(
            blocks=(
                self._block(0, "heading", "BACKGROUND", "background", "BACKGROUND", heading_level=1),
                self._block(
                    1,
                    "paragraph",
                    " ".join(["Background facts are set out here."] * 30),
                    "background",
                    "BACKGROUND",
                ),
                self._block(2, "heading", "DIRECTIONS", "directions", "DIRECTIONS", heading_level=1),
                self._block(
                    3,
                    "paragraph",
                    "The noticee shall comply within forty five days.",
                    "directions",
                    "DIRECTIONS",
                ),
            )
        )

        chunks = build_chunks(
            parsed,
            model_name=MODEL_NAME,
            target_chunk_tokens=80,
            max_chunk_tokens=120,
            overlap_tokens=20,
        )

        background_chunks = [chunk for chunk in chunks if chunk.section_type == "background"]
        directions_chunks = [chunk for chunk in chunks if chunk.section_type == "directions"]
        self.assertTrue(background_chunks)
        self.assertTrue(directions_chunks)
        for chunk in background_chunks:
            self.assertNotIn("DIRECTIONS", chunk.chunk_text)

    def test_chunker_is_deterministic(self) -> None:
        parsed = ParsedDocument(
            blocks=(
                self._block(0, "heading", "FINDINGS", "findings", "FINDINGS", heading_level=1),
                self._block(
                    1,
                    "paragraph",
                    " ".join(["The findings paragraph is detailed and repeatable."] * 40),
                    "findings",
                    "FINDINGS",
                ),
            )
        )

        first = build_chunks(
            parsed,
            model_name=MODEL_NAME,
            target_chunk_tokens=90,
            max_chunk_tokens=120,
            overlap_tokens=10,
        )
        second = build_chunks(
            parsed,
            model_name=MODEL_NAME,
            target_chunk_tokens=90,
            max_chunk_tokens=120,
            overlap_tokens=10,
        )

        self.assertEqual(
            [(chunk.chunk_index, chunk.chunk_text) for chunk in first],
            [(chunk.chunk_index, chunk.chunk_text) for chunk in second],
        )

    def test_short_operative_section_stays_intact(self) -> None:
        parsed = ParsedDocument(
            blocks=(
                self._block(0, "heading", "ORDER", "operative_order", "ORDER", heading_level=1),
                self._block(
                    1,
                    "paragraph",
                    "The proceedings are disposed of with the above directions.",
                    "operative_order",
                    "ORDER",
                ),
            )
        )

        chunks = build_chunks(
            parsed,
            model_name=MODEL_NAME,
            target_chunk_tokens=20,
            max_chunk_tokens=60,
            overlap_tokens=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_type, "operative_order")

    @staticmethod
    def _block(
        block_index: int,
        block_type: str,
        text: str,
        section_type: str,
        section_title: str | None,
        *,
        heading_level: int | None = None,
    ) -> StructuredBlock:
        heading_path = (section_title,) if section_title else ()
        return StructuredBlock(
            block_index=block_index,
            page_no=1,
            block_type=block_type,
            text=text,
            token_count=token_count(text, model_name=MODEL_NAME),
            section_type=section_type,
            section_title=section_title,
            heading_path=heading_path,
            heading_level=heading_level,
        )


class ChunkPostprocessTests(unittest.TestCase):
    def test_rti_caption_fragments_merge_into_one_chunk(self) -> None:
        chunks = (
            self._chunk(0, "Appeal No. 6798 of 2026", "header", page_start=1, page_end=1),
            self._chunk(
                1,
                "BEFORE THE APPELLATE AUTHORITY\n(Under the Right to Information Act, 2005)",
                "other",
                page_start=1,
                page_end=1,
                section_title="BEFORE THE APPELLATE AUTHORITY",
                heading_path=("BEFORE THE APPELLATE AUTHORITY",),
            ),
            self._chunk(
                2,
                "SECURITIES AND EXCHANGE BOARD OF INDIA\nAppeal No. 6798 of 2026 : Appellant Hariom Yadav Vs",
                "other",
                page_start=1,
                page_end=1,
                section_title="SECURITIES AND EXCHANGE BOARD OF INDIA",
                heading_path=("SECURITIES AND EXCHANGE BOARD OF INDIA",),
            ),
            self._chunk(
                3,
                "CPIO, SEBI, Mumbai : Respondent",
                "other",
                page_start=1,
                page_end=1,
                section_title="CPIO, SEBI, Mumbai",
                heading_path=("CPIO, SEBI, Mumbai",),
            ),
            self._chunk(
                4,
                "ORDER\n\nThe appeal is accordingly dismissed.",
                "operative_order",
                page_start=1,
                page_end=2,
                section_title="ORDER",
                heading_path=("ORDER",),
            ),
        )

        result = postprocess_chunks(
            chunks,
            page_count=2,
            model_name=MODEL_NAME,
            max_chunk_tokens=1000,
        )

        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.summary.merges_applied, 3)
        self.assertEqual(result.chunks[0].section_type, "header")
        self.assertIn("BEFORE THE APPELLATE AUTHORITY", result.chunks[0].chunk_text)
        self.assertIn("CPIO, SEBI, Mumbai : Respondent", result.chunks[0].chunk_text)
        self.assertEqual(result.chunks[1].section_type, "operative_order")

    def test_heading_only_chunk_merges_into_following_substantive_chunk(self) -> None:
        chunks = (
            self._chunk(
                0,
                "ORDER",
                "operative_order",
                page_start=1,
                page_end=1,
                section_title="ORDER",
                heading_path=("ORDER",),
            ),
            self._chunk(
                1,
                "The proceedings are disposed of with the above directions.",
                "operative_order",
                page_start=1,
                page_end=1,
                section_title="ORDER",
                heading_path=("ORDER",),
            ),
        )

        result = postprocess_chunks(
            chunks,
            page_count=1,
            model_name=MODEL_NAME,
            max_chunk_tokens=1000,
        )

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.summary.merges_applied, 1)
        self.assertTrue(result.chunks[0].chunk_text.startswith("ORDER\n\nThe proceedings"))

    def test_footer_signature_is_merged_and_boilerplate_is_suppressed(self) -> None:
        chunks = (
            self._chunk(
                0,
                "ORDER\n\nThe appeal is accordingly dismissed.",
                "operative_order",
                page_start=1,
                page_end=2,
                section_title="ORDER",
                heading_path=("ORDER",),
            ),
            self._chunk(
                1,
                "RUCHI CHOJER\nDate: April 02, 2026",
                "other",
                page_start=2,
                page_end=2,
                section_title="RUCHI CHOJER",
                heading_path=("RUCHI CHOJER",),
            ),
            self._chunk(
                2,
                "APPELLATE AUTHORITY UNDER THE RTI ACT",
                "other",
                page_start=2,
                page_end=2,
                section_title="APPELLATE AUTHORITY UNDER THE RTI ACT",
                heading_path=("APPELLATE AUTHORITY UNDER THE RTI ACT",),
            ),
            self._chunk(
                3,
                "SECURITIES AND EXCHANGE BOARD OF INDIA",
                "other",
                page_start=2,
                page_end=2,
                section_title="SECURITIES AND EXCHANGE BOARD OF INDIA",
                heading_path=("SECURITIES AND EXCHANGE BOARD OF INDIA",),
            ),
        )

        result = postprocess_chunks(
            chunks,
            page_count=2,
            model_name=MODEL_NAME,
            max_chunk_tokens=1000,
        )

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.summary.merges_applied, 1)
        self.assertEqual(result.summary.suppressed_chunks, 2)
        self.assertIn("RUCHI CHOJER", result.chunks[0].chunk_text)
        self.assertNotIn(
            "APPELLATE AUTHORITY UNDER THE RTI ACT",
            result.chunks[0].chunk_text,
        )

    def test_minimum_chunk_merge_rule_cleans_up_tiny_chunk(self) -> None:
        chunks = (
            self._chunk(
                0,
                "Background facts are set out here in detail for the adjudication record.",
                "background",
                page_start=1,
                page_end=1,
                section_title="BACKGROUND",
                heading_path=("BACKGROUND",),
            ),
            self._chunk(
                1,
                "A short continuation clarifies the same facts.",
                "background",
                page_start=1,
                page_end=1,
                section_title="BACKGROUND",
                heading_path=("BACKGROUND",),
            ),
        )

        result = postprocess_chunks(
            chunks,
            page_count=1,
            model_name=MODEL_NAME,
            max_chunk_tokens=1000,
        )

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.summary.merges_applied, 1)
        self.assertIn("short continuation", result.chunks[0].chunk_text)

    def test_short_document_guardrail_runs_extra_cleanup_pass(self) -> None:
        chunks = (
            self._chunk(
                0,
                self._repeated_sentence("Background context remains material to the appeal.", 40),
                "background",
                page_start=1,
                page_end=1,
                section_title="BACKGROUND",
                heading_path=("BACKGROUND",),
            ),
            self._chunk(
                1,
                self._repeated_sentence("This brief connector should not stay alone.", 12),
                "other",
                page_start=1,
                page_end=1,
                section_title="Connector 1",
                heading_path=("Connector 1",),
            ),
            self._chunk(
                2,
                self._repeated_sentence("Findings text remains substantial and self-contained.", 40),
                "findings",
                page_start=2,
                page_end=2,
                section_title="FINDINGS",
                heading_path=("FINDINGS",),
            ),
            self._chunk(
                3,
                self._repeated_sentence("This connector also adds little retrieval value.", 12),
                "other",
                page_start=2,
                page_end=2,
                section_title="Connector 2",
                heading_path=("Connector 2",),
            ),
            self._chunk(
                4,
                self._repeated_sentence("Directions text remains substantive for compliance.", 40),
                "directions",
                page_start=3,
                page_end=3,
                section_title="DIRECTIONS",
                heading_path=("DIRECTIONS",),
            ),
            self._chunk(
                5,
                self._repeated_sentence("This detached note should merge away in a short doc.", 10),
                "other",
                page_start=3,
                page_end=3,
                section_title="Connector 3",
                heading_path=("Connector 3",),
            ),
            self._chunk(
                6,
                self._repeated_sentence("A final detached note should not remain separate.", 12),
                "other",
                page_start=3,
                page_end=3,
                section_title="Connector 4",
                heading_path=("Connector 4",),
            ),
        )

        self.assertGreater(chunks[1].token_count, 80)
        self.assertGreater(chunks[3].token_count, 80)
        self.assertGreater(chunks[5].token_count, 80)

        result = postprocess_chunks(
            chunks,
            page_count=3,
            model_name=MODEL_NAME,
            max_chunk_tokens=1000,
        )

        self.assertLessEqual(len(result.chunks), 5)

    def test_large_operative_chunk_remains_intact(self) -> None:
        large_operative_text = self._repeated_sentence(
            "The operative order remains a single high-value chunk for retrieval.",
            70,
        )
        chunks = (
            self._chunk(
                0,
                large_operative_text,
                "operative_order",
                page_start=1,
                page_end=3,
                section_title="ORDER",
                heading_path=("ORDER",),
            ),
        )

        result = postprocess_chunks(
            chunks,
            page_count=3,
            model_name=MODEL_NAME,
            max_chunk_tokens=1000,
        )

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.summary.merges_applied, 0)
        self.assertEqual(result.chunks[0].chunk_text, large_operative_text)

    @staticmethod
    def _chunk(
        chunk_index: int,
        chunk_text: str,
        section_type: str,
        *,
        page_start: int,
        page_end: int,
        section_title: str | None = None,
        heading_path: tuple[str, ...] = (),
    ) -> ChunkRecord:
        return ChunkRecord(
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
            section_type=section_type,
            section_title=section_title,
            heading_path=heading_path,
            chunk_text=chunk_text,
            chunk_sha256="unused",
            token_count=token_count(chunk_text, model_name=MODEL_NAME),
        )

    @staticmethod
    def _repeated_sentence(sentence: str, count: int) -> str:
        return " ".join([sentence] * count)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
