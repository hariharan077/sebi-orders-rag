from __future__ import annotations

import unittest
from datetime import date

from app.sebi_orders_rag.retrieval.query_intent import QueryIntent, QueryIntentResult, detect_query_intent
from app.sebi_orders_rag.retrieval.scoring import (
    ChunkSearchHit,
    DocumentSearchHit,
    ScoreBreakdown,
    SectionSearchHit,
    merge_document_hits,
    rerank_chunk_hits,
    rerank_section_hits,
    resolve_section_prior,
)


class QueryIntentTests(unittest.TestCase):
    def test_detect_query_intent_marks_substantive_outcome_query(self) -> None:
        result = detect_query_intent("Appeal dismissed under RTI Act")

        self.assertEqual(result.intent, QueryIntent.SUBSTANTIVE_OUTCOME_QUERY)
        self.assertIn("dismissed", result.matched_terms)

    def test_detect_query_intent_marks_party_or_title_lookup(self) -> None:
        result = detect_query_intent("Appeal No. 6798 of 2026 filed by Hariom Yadav")

        self.assertEqual(result.intent, QueryIntent.PARTY_OR_TITLE_LOOKUP)
        self.assertIn("appeal_no", result.matched_terms)
        self.assertIn("filed_by", result.matched_terms)

    def test_detect_query_intent_keeps_generic_settlement_explanation_generic(self) -> None:
        result = detect_query_intent("What is a settlement order?")

        self.assertEqual(result.intent, QueryIntent.GENERIC_LOOKUP)
        self.assertIn("settlement order", result.settlement_terms)
        self.assertFalse(result.settlement_focused)

    def test_detect_query_intent_marks_settlement_query_as_focused(self) -> None:
        result = detect_query_intent(
            "What was the settlement amount in the JP Morgan settlement order?"
        )

        self.assertEqual(result.intent, QueryIntent.SUBSTANTIVE_OUTCOME_QUERY)
        self.assertTrue(result.settlement_focused)
        self.assertIn("settlement", result.settlement_terms)
        self.assertIn("jp", result.entity_terms)
        self.assertIn("morgan", result.entity_terms)


class RetrievalScoringTests(unittest.TestCase):
    def test_section_prior_application_prefers_operative_order_over_header(self) -> None:
        hits = rerank_section_hits(
            [
                self._section_hit(section_node_id=1, document_version_id=101, section_type="header", base_score=0.0500),
                self._section_hit(
                    section_node_id=2,
                    document_version_id=202,
                    section_type="operative_order",
                    base_score=0.0400,
                ),
            ],
            query_intent=QueryIntent.GENERIC_LOOKUP,
        )

        self.assertEqual(hits[0].section_type, "operative_order")
        self.assertEqual(hits[0].score.section_prior, resolve_section_prior("operative_order"))
        self.assertEqual(hits[1].score.section_prior, resolve_section_prior("header"))

    def test_header_suppression_keeps_strong_substantive_section_ahead(self) -> None:
        hits = rerank_section_hits(
            [
                self._section_hit(section_node_id=1, document_version_id=101, section_type="header", base_score=0.1800),
                self._section_hit(
                    section_node_id=2,
                    document_version_id=101,
                    section_type="operative_order",
                    base_score=0.0500,
                ),
            ],
            query_intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
        )

        self.assertEqual(hits[0].section_type, "operative_order")
        self.assertEqual(hits[1].section_type, "header")
        self.assertLess(hits[1].score.query_intent_adjustment, 0.75)
        self.assertGreater(hits[0].score.final_score, hits[1].score.final_score)

    def test_party_or_title_lookup_does_not_over_demote_header(self) -> None:
        hits = rerank_section_hits(
            [
                self._section_hit(section_node_id=1, document_version_id=101, section_type="header", base_score=0.0500),
                self._section_hit(
                    section_node_id=2,
                    document_version_id=101,
                    section_type="operative_order",
                    base_score=0.0500,
                ),
            ],
            query_intent=QueryIntent.PARTY_OR_TITLE_LOOKUP,
        )

        self.assertEqual(hits[0].section_type, "header")
        self.assertGreater(hits[0].score.final_score, hits[1].score.final_score)

    def test_chunk_diversity_prefers_near_competitive_substantive_chunk(self) -> None:
        hits = rerank_chunk_hits(
            [
                self._chunk_hit(chunk_id=1, document_version_id=101, section_type="header", base_score=0.1300),
                self._chunk_hit(chunk_id=2, document_version_id=101, section_type="header", base_score=0.1100),
                self._chunk_hit(
                    chunk_id=3,
                    document_version_id=101,
                    section_type="background",
                    base_score=0.0685,
                ),
            ],
            query_intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
        )

        self.assertEqual(hits[0].chunk_id, 3)
        self.assertGreater(hits[0].score.diversity_adjustment, 0.0)
        repeated_header = next(hit for hit in hits if hit.chunk_id == 2)
        self.assertLess(repeated_header.score.diversity_adjustment, 0.0)

    def test_rerank_chunk_hits_is_deterministic(self) -> None:
        candidate_hits = [
            self._chunk_hit(chunk_id=1, document_version_id=101, section_type="header", base_score=0.1200),
            self._chunk_hit(chunk_id=2, document_version_id=102, section_type="findings", base_score=0.0500),
            self._chunk_hit(chunk_id=3, document_version_id=101, section_type="operative_order", base_score=0.0400),
        ]

        first = rerank_chunk_hits(
            candidate_hits,
            query_intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
        )
        second = rerank_chunk_hits(
            candidate_hits,
            query_intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
        )

        self.assertEqual([hit.chunk_id for hit in first], [hit.chunk_id for hit in second])

    def test_settlement_document_prior_prefers_matching_party_title(self) -> None:
        query_intent = QueryIntentResult(
            intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
            matched_terms=("settlement",),
            settlement_terms=("settlement",),
            entity_terms=("jp", "morgan"),
            settlement_focused=True,
        )

        hits = merge_document_hits(
            [
                self._document_hit(
                    document_version_id=195,
                    title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                    combined_score=0.0340,
                ),
                self._document_hit(
                    document_version_id=194,
                    title="Settlement Order in the matter of Sixteenth Street Asian GEMS Fund",
                    combined_score=0.0345,
                ),
            ],
            [],
            query_intent=query_intent,
        )

        self.assertEqual(hits[0].document_version_id, 195)
        self.assertGreater(hits[0].score.bucket_adjustment, 1.0)
        self.assertGreater(hits[0].score.query_alignment_adjustment, 1.0)
        self.assertLess(hits[1].score.query_alignment_adjustment, 1.0)

    def test_settlement_body_chunk_outranks_header_only_chunk(self) -> None:
        query_intent = QueryIntentResult(
            intent=QueryIntent.SUBSTANTIVE_OUTCOME_QUERY,
            matched_terms=("settlement",),
            settlement_terms=("settlement",),
            entity_terms=("jp", "morgan"),
            settlement_focused=True,
        )
        hits = rerank_chunk_hits(
            [
                self._chunk_hit(
                    chunk_id=1,
                    document_version_id=195,
                    title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                    section_type="operative_order",
                    base_score=0.0430,
                    chunk_text="SECURITIES AND EXCHANGE BOARD OF INDIA SETTLEMENT ORDER IN RESPECT OF JP Morgan Chase Bank N.A.",
                    token_count=18,
                ),
                self._chunk_hit(
                    chunk_id=2,
                    document_version_id=195,
                    title="Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                    section_type="operative_order",
                    base_score=0.0400,
                    chunk_text=(
                        "The High Powered Advisory Committee considered the settlement terms "
                        "proposed by the Applicant. Notice of Demand was issued, the Applicant "
                        "remitted the said settlement amount, and it is hereby ordered that no "
                        "enforcement action shall be initiated."
                    ),
                    token_count=64,
                ),
            ],
            query_intent=query_intent,
        )

        self.assertEqual(hits[0].chunk_id, 2)
        self.assertLess(hits[1].score.content_adjustment, 1.0)
        self.assertGreater(hits[0].score.content_adjustment, 1.0)

    @staticmethod
    def _section_hit(
        *,
        section_node_id: int,
        document_version_id: int,
        section_type: str,
        base_score: float,
        title: str | None = None,
    ) -> SectionSearchHit:
        return SectionSearchHit(
            section_node_id=section_node_id,
            document_version_id=document_version_id,
            document_id=document_version_id,
            record_key=f"external:{document_version_id}",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id=str(document_version_id),
            order_date=date(2026, 4, 2),
            title=title or f"Document {document_version_id}",
            section_key=f"section-{section_node_id}",
            section_type=section_type,
            section_title=section_type.upper(),
            heading_path=section_type.upper(),
            page_start=1,
            page_end=1,
            section_node_text="section text",
            score=ScoreBreakdown(combined_score=base_score),
        )

    @staticmethod
    def _chunk_hit(
        *,
        chunk_id: int,
        document_version_id: int,
        section_type: str,
        base_score: float,
        title: str | None = None,
        chunk_text: str = "chunk text",
        token_count: int = 120,
    ) -> ChunkSearchHit:
        section_key = None if section_type == "header" else f"section-{section_type}"
        return ChunkSearchHit(
            chunk_id=chunk_id,
            document_version_id=document_version_id,
            document_id=document_version_id,
            record_key=f"external:{document_version_id}",
            bucket_name="orders-of-aa-under-rti-act",
            external_record_id=str(document_version_id),
            order_date=date(2026, 4, 2),
            title=title or f"Document {document_version_id}",
            chunk_index=chunk_id,
            page_start=1,
            page_end=1,
            section_key=section_key,
            section_type=section_type,
            section_title=section_type.upper(),
            heading_path=section_type.upper(),
            detail_url=f"https://example.com/detail/{document_version_id}",
            pdf_url=f"https://example.com/pdf/{document_version_id}.pdf",
            chunk_text=chunk_text,
            token_count=token_count,
            score=ScoreBreakdown(combined_score=base_score),
        )

    @staticmethod
    def _document_hit(
        *,
        document_version_id: int,
        title: str,
        combined_score: float,
    ) -> DocumentSearchHit:
        return DocumentSearchHit(
            document_version_id=document_version_id,
            document_id=document_version_id,
            record_key=f"external:{document_version_id}",
            bucket_name="settlement-orders",
            external_record_id=str(document_version_id),
            order_date=date(2026, 4, 2),
            title=title,
            document_node_text="document node text",
            score=ScoreBreakdown(combined_score=combined_score),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
