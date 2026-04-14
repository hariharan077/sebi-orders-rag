from __future__ import annotations

import unittest
from datetime import date

from app.sebi_orders_rag.retrieval.hierarchical_search import derive_hierarchical_filters
from app.sebi_orders_rag.retrieval.scoring import (
    ChunkSearchHit,
    DocumentSearchHit,
    ScoreBreakdown,
    SectionSearchHit,
    merge_document_hits,
)
from app.sebi_orders_rag.schemas import MetadataFilterInput


class HierarchicalSearchTests(unittest.TestCase):
    def test_derive_hierarchical_filters_narrows_documents_then_sections(self) -> None:
        base_filters = MetadataFilterInput(
            record_key="external:100725",
            bucket_name="orders-of-aa-under-rti-act",
        )
        documents = (
            self._document_hit(document_version_id=101, combined_score=0.07),
            self._document_hit(document_version_id=202, combined_score=0.05),
        )
        sections = (
            self._section_hit(
                section_node_id=1,
                document_version_id=101,
                section_key="section-0002-operative_order-order",
                combined_score=0.08,
            ),
        )

        section_filters, chunk_filters = derive_hierarchical_filters(
            base_filters,
            document_hits=documents,
            section_hits=sections,
        )

        self.assertEqual(section_filters.record_key, "external:100725")
        self.assertEqual(section_filters.bucket_name, "orders-of-aa-under-rti-act")
        self.assertEqual(section_filters.document_version_ids, (101, 202))
        self.assertEqual(chunk_filters.document_version_ids, (101, 202))
        self.assertEqual(
            chunk_filters.section_keys,
            ("section-0002-operative_order-order",),
        )

    def test_merge_document_hits_is_deterministic(self) -> None:
        lexical_hits = [
            self._document_hit(document_version_id=101, lexical_score=0.60),
            self._document_hit(document_version_id=202, lexical_score=0.59),
        ]
        vector_hits = [
            self._document_hit(document_version_id=202, vector_score=0.90, vector_distance=0.10),
            self._document_hit(document_version_id=101, vector_score=0.70, vector_distance=0.30),
        ]

        first = merge_document_hits(lexical_hits, vector_hits)
        second = merge_document_hits(lexical_hits, vector_hits)

        self.assertEqual(
            [hit.document_version_id for hit in first],
            [hit.document_version_id for hit in second],
        )
        self.assertEqual(first[0].document_version_id, 202)
        self.assertGreater(first[0].score.combined_score, first[1].score.combined_score)
        self.assertEqual(first[0].score.lexical_rank, 2)
        self.assertEqual(first[0].score.vector_rank, 1)

    @staticmethod
    def _document_hit(
        *,
        document_version_id: int,
        combined_score: float = 0.0,
        lexical_score: float = 0.0,
        vector_score: float = 0.0,
        vector_distance: float | None = None,
    ) -> DocumentSearchHit:
        return DocumentSearchHit(
            document_version_id=document_version_id,
            document_id=document_version_id,
            record_key=f"external:{document_version_id}",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id=str(document_version_id),
            order_date=date(2026, 4, 2),
            title=f"Document {document_version_id}",
            document_node_text="node text",
            score=ScoreBreakdown(
                combined_score=combined_score,
                lexical_score=lexical_score,
                vector_score=vector_score,
                vector_distance=vector_distance,
            ),
        )

    @staticmethod
    def _section_hit(
        *,
        section_node_id: int,
        document_version_id: int,
        section_key: str,
        combined_score: float,
    ) -> SectionSearchHit:
        return SectionSearchHit(
            section_node_id=section_node_id,
            document_version_id=document_version_id,
            document_id=document_version_id,
            record_key=f"external:{document_version_id}",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id=str(document_version_id),
            order_date=date(2026, 4, 2),
            title=f"Document {document_version_id}",
            section_key=section_key,
            section_type="operative_order",
            section_title="ORDER",
            heading_path="ORDER",
            page_start=1,
            page_end=2,
            section_node_text="section text",
            score=ScoreBreakdown(combined_score=combined_score),
        )

    @staticmethod
    def _chunk_hit(
        *,
        chunk_id: int,
        document_version_id: int,
        section_key: str,
        combined_score: float,
    ) -> ChunkSearchHit:
        return ChunkSearchHit(
            chunk_id=chunk_id,
            document_version_id=document_version_id,
            document_id=document_version_id,
            record_key=f"external:{document_version_id}",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id=str(document_version_id),
            order_date=date(2026, 4, 2),
            title=f"Document {document_version_id}",
            chunk_index=chunk_id,
            page_start=1,
            page_end=2,
            section_key=section_key,
            section_type="operative_order",
            section_title="ORDER",
            heading_path="ORDER",
            detail_url=f"https://example.com/detail/{document_version_id}",
            pdf_url=f"https://example.com/pdf/{document_version_id}.pdf",
            chunk_text="chunk text",
            token_count=120,
            score=ScoreBreakdown(combined_score=combined_score),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
