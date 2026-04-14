from __future__ import annotations

import unittest
from datetime import date

from app.sebi_orders_rag.metadata.models import StoredNumericFact, StoredPriceMovement
from app.sebi_orders_rag.metadata.service import OrderMetadataService


class PriceMovementAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = OrderMetadataService(repository=_DuDigitalMetadataRepository())

    def test_answers_share_price_increase_from_metadata(self) -> None:
        answer = self.service.answer_numeric_fact_question(
            query="how much did DU Digital share price increase",
            document_version_ids=(118,),
        )

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.metadata_type, "price_increase")
        self.assertIn("1392.5%", answer.answer_text)
        self.assertIn("Rs.12/share", answer.answer_text)
        self.assertIn("Rs.179.10", answer.answer_text)
        self.assertIn("Rs.296.05", answer.answer_text)

    def test_answers_before_and_after_price_from_metadata(self) -> None:
        answer = self.service.answer_numeric_fact_question(
            query="what was the price before and after the increase",
            document_version_ids=(118,),
        )

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.metadata_type, "before_after_price")
        self.assertIn("Rs.12/share", answer.answer_text)
        self.assertIn("Rs.179.10", answer.answer_text)
        self.assertIn("highest price", answer.answer_text.lower())

    def test_answers_period_wise_price_movement_from_metadata(self) -> None:
        answer = self.service.answer_numeric_fact_question(
            query="give the price movement of DU Digital for each period",
            document_version_ids=(118,),
        )

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.metadata_type, "price_movements_by_period")
        self.assertIn("Patch 1", answer.answer_text)
        self.assertIn("+142.93%".replace("+", ""), answer.answer_text.replace("+", ""))
        self.assertIn("Patch 4", answer.answer_text)
        self.assertIn("-58.9%", answer.answer_text)


class _DuDigitalMetadataRepository:
    def fetch_numeric_facts(self, *, document_version_ids, fact_types=None):
        facts = (
            StoredNumericFact(
                numeric_fact_id=1,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                fact_type="listing_price",
                value_text="Rs.12/share",
                value_numeric=12.0,
                unit="INR/share",
                context_label="listed on August 26, 2021",
                page_start=2,
                page_end=2,
            ),
            StoredNumericFact(
                numeric_fact_id=2,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                fact_type="closing_price",
                value_text="Rs.179.10",
                value_numeric=179.10,
                unit="INR",
                context_label="August 26, 2021 to March 31, 2023",
                page_start=2,
                page_end=2,
            ),
            StoredNumericFact(
                numeric_fact_id=3,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                fact_type="percentage_change",
                value_text="1392.5%",
                value_numeric=1392.5,
                unit="percent",
                context_label="August 26, 2021 to March 31, 2023",
                page_start=2,
                page_end=2,
            ),
            StoredNumericFact(
                numeric_fact_id=4,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                fact_type="highest_price",
                value_text="Rs.296.05",
                value_numeric=296.05,
                unit="INR",
                context_label="on November 11, 2022",
                page_start=2,
                page_end=2,
            ),
            StoredNumericFact(
                numeric_fact_id=5,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                fact_type="percentage_change_from_listing",
                value_text="2467%",
                value_numeric=2467.0,
                unit="percent",
                context_label="highest price on November 11, 2022",
                page_start=2,
                page_end=2,
            ),
        )
        if not fact_types:
            return facts
        return tuple(fact for fact in facts if fact.fact_type in set(fact_types))

    def fetch_price_movements(self, *, document_version_ids):
        return (
            StoredPriceMovement(
                price_movement_id=1,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                period_label="Patch 1",
                period_start_text="August 26, 2021",
                period_end_text="November 23, 2021",
                start_price=63.0,
                high_price=153.05,
                end_price=132.5,
                pct_change=142.93,
                page_start=3,
                page_end=3,
            ),
            StoredPriceMovement(
                price_movement_id=2,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                period_label="Patch 2",
                period_start_text="November 24, 2021",
                period_end_text="June 29, 2022",
                start_price=139.0,
                high_price=489.0,
                end_price=455.0,
                pct_change=251.7,
                page_start=3,
                page_end=3,
            ),
            StoredPriceMovement(
                price_movement_id=3,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                period_label="Patch 3",
                period_start_text="June 30, 2022",
                period_end_text="November 11, 2022",
                start_price=93.15,
                high_price=296.05,
                end_price=296.05,
                pct_change=217.8,
                page_start=3,
                page_end=3,
            ),
            StoredPriceMovement(
                price_movement_id=4,
                document_version_id=118,
                document_id=137,
                record_key="external:98774",
                title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                period_label="Patch 4",
                period_start_text="November 12, 2022",
                period_end_text="March 31, 2023",
                start_price=289.8,
                low_price=119.05,
                end_price=179.1,
                pct_change=-58.9,
                page_start=3,
                page_end=3,
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
