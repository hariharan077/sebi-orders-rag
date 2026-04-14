from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from app.sebi_orders_rag.embeddings.payload_builder import (
    build_chunk_embedding_text,
    build_document_node_payload,
    build_section_node_payload,
)
from app.sebi_orders_rag.ingestion.token_count import token_count
from app.sebi_orders_rag.schemas import EmbeddingCandidate, SectionGroupInput, StoredChunk

MODEL_NAME = "text-embedding-3-large"


class PayloadBuilderTests(unittest.TestCase):
    def test_build_document_node_payload_is_deterministic_and_structured(self) -> None:
        candidate = self._candidate()
        chunks = (
            self._chunk(
                chunk_id=1,
                chunk_index=0,
                section_type="header",
                section_title="Appeal No. 6799 of 2026",
                chunk_text="Hariom Yadav filed the RTI appeal before the appellate authority.",
            ),
            self._chunk(
                chunk_id=2,
                chunk_index=1,
                section_type="operative_order",
                section_title="ORDER",
                heading_path="ORDER",
                chunk_text="The appeal is dismissed under the RTI Act after examining the record.",
                section_key="section-0002-operative_order-order",
            ),
        )
        sections = (
            self._section(
                section_key="section-0001-header-appeal",
                section_type="header",
                section_title="Appeal No. 6799 of 2026",
                chunks=(chunks[0],),
            ),
            self._section(
                section_key="section-0002-operative_order-order",
                section_type="operative_order",
                section_title="ORDER",
                heading_path="ORDER",
                page_start=1,
                page_end=2,
                chunks=(chunks[1],),
            ),
        )

        payload = build_document_node_payload(
            candidate,
            sections=sections,
            chunks=chunks,
            model_name=MODEL_NAME,
        )

        self.assertIn("Document title: Appeal No. 6799 of 2026 filed by Hariom Yadav", payload.node_text)
        self.assertIn("Bucket: orders-of-aa-under-rti-act", payload.node_text)
        self.assertIn("External record id: 100725", payload.node_text)
        self.assertIn("Procedural type: rti appeal order", payload.node_text)
        self.assertIn("Major headings: Appeal No. 6799 of 2026; ORDER", payload.node_text)
        self.assertIn("Opening lines:", payload.node_text)
        self.assertGreater(payload.token_count, 0)
        self.assertEqual(payload.metadata["record_key"], "external:100725")

    def test_build_section_node_payload_preserves_identity_and_text(self) -> None:
        candidate = self._candidate()
        section = self._section(
            section_key="section-0002-operative_order-order",
            section_type="operative_order",
            section_title="ORDER",
            heading_path="ORDER",
            page_start=1,
            page_end=2,
            chunks=(
                self._chunk(
                    chunk_id=2,
                    chunk_index=1,
                    section_type="operative_order",
                    section_title="ORDER",
                    heading_path="ORDER",
                    chunk_text="The appeal is dismissed under the RTI Act after examining the record.",
                    section_key="section-0002-operative_order-order",
                ),
            ),
        )

        payload = build_section_node_payload(
            candidate,
            section,
            model_name=MODEL_NAME,
        )

        self.assertIn("Section key: section-0002-operative_order-order", payload.node_text)
        self.assertIn("Section type: operative_order", payload.node_text)
        self.assertIn("Page range: 1-2", payload.node_text)
        self.assertIn("Section text:", payload.node_text)
        self.assertIn("The appeal is dismissed under the RTI Act", payload.node_text)
        self.assertEqual(payload.metadata["chunk_ids"], [2])

    def test_build_chunk_embedding_text_adds_light_section_prefix(self) -> None:
        chunk = self._chunk(
            chunk_id=2,
            chunk_index=1,
            section_type="operative_order",
            section_title="ORDER",
            heading_path="ORDER",
            chunk_text="The appeal is dismissed under the RTI Act after examining the record.",
            section_key="section-0002-operative_order-order",
        )

        text = build_chunk_embedding_text(chunk)

        self.assertIn("Section type: operative_order", text)
        self.assertIn("Section title: ORDER", text)
        self.assertTrue(text.endswith("The appeal is dismissed under the RTI Act after examining the record."))

    def test_build_section_node_payload_caps_oversized_text(self) -> None:
        candidate = self._candidate()
        long_text = " ".join(f"token{i}" for i in range(12000))
        section = self._section(
            section_key="section-oversized",
            section_type="findings",
            section_title="FINDINGS",
            heading_path="FINDINGS",
            page_start=1,
            page_end=40,
            chunks=(
                self._chunk(
                    chunk_id=99,
                    chunk_index=0,
                    section_type="findings",
                    section_title="FINDINGS",
                    heading_path="FINDINGS",
                    chunk_text=long_text,
                    section_key="section-oversized",
                ),
            ),
        )

        payload = build_section_node_payload(
            candidate,
            section,
            model_name=MODEL_NAME,
        )

        self.assertIn("Section key: section-oversized", payload.node_text)
        self.assertTrue(payload.metadata["text_truncated"])
        self.assertLessEqual(token_count(payload.node_text, model_name=MODEL_NAME), 7800)

    @staticmethod
    def _candidate() -> EmbeddingCandidate:
        return EmbeddingCandidate(
            document_version_id=101,
            document_id=55,
            record_key="external:100725",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id="100725",
            order_date=date(2026, 4, 2),
            title="Appeal No. 6799 of 2026 filed by Hariom Yadav",
            detail_url="https://example.com/detail",
            pdf_url="https://example.com/pdf",
            local_filename="appeal.pdf",
            local_path="/tmp/appeal.pdf",
            ingest_status="done",
            chunking_version="v2.1",
            chunk_count=2,
            embedding_status=None,
            embedding_error=None,
            embedding_model=None,
            embedding_dim=None,
            embedded_at=None,
            created_at=datetime(2026, 4, 10, 9, 30, tzinfo=timezone.utc),
        )

    @staticmethod
    def _chunk(
        *,
        chunk_id: int,
        chunk_index: int,
        section_type: str,
        section_title: str | None,
        chunk_text: str,
        heading_path: str | None = None,
        section_key: str | None = None,
    ) -> StoredChunk:
        return StoredChunk(
            chunk_id=chunk_id,
            document_version_id=101,
            document_id=55,
            record_key="external:100725",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id="100725",
            order_date=date(2026, 4, 2),
            title="Appeal No. 6799 of 2026 filed by Hariom Yadav",
            chunk_index=chunk_index,
            page_start=1,
            page_end=2,
            section_type=section_type,
            section_title=section_title,
            heading_path=heading_path,
            section_key=section_key,
            chunk_text=chunk_text,
            chunk_sha256=f"sha-{chunk_id}",
            token_count=32,
            chunk_metadata={},
            embedding_model=None,
            embedding_created_at=None,
        )

    @staticmethod
    def _section(
        *,
        section_key: str,
        section_type: str,
        section_title: str | None,
        chunks: tuple[StoredChunk, ...],
        heading_path: str | None = None,
        page_start: int = 1,
        page_end: int = 1,
    ) -> SectionGroupInput:
        return SectionGroupInput(
            document_version_id=101,
            document_id=55,
            record_key="external:100725",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id="100725",
            order_date=date(2026, 4, 2),
            title="Appeal No. 6799 of 2026 filed by Hariom Yadav",
            section_key=section_key,
            section_type=section_type,
            section_title=section_title,
            heading_path=heading_path,
            page_start=page_start,
            page_end=page_end,
            chunks=chunks,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
