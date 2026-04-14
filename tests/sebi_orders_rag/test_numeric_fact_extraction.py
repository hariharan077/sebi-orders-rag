from __future__ import annotations

import unittest

from app.sebi_orders_rag.metadata.numeric_facts import extract_numeric_facts
from app.sebi_orders_rag.metadata.models import MetadataChunkText, MetadataPageText
from app.sebi_orders_rag.metadata.tables import extract_price_movements


_DU_PAGE_2 = """Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)

1. Securities and Exchange Board of India initiated investigation in the scrip of DU Digital Technologies Limited. It was observed that post listing on August 26, 2021 at the price of Rs.12/share, the price of the shares of the company increased by 1392.5% during the period August 26, 2021 to March 31, 2023 and closed at Rs.179.10. During this period, the scrip also closed at the highest price of Rs.296.05, which is 2467% of the listing price, on November 11, 2022.
"""

_DU_PAGE_3 = """Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)

4. The basis of the above division of IP into patches and the corresponding price movement in the scrip of DU Digital in each patch as observed during the investigation: *price referred is closing price of the scrip on the day.

S.No. Patch/Period Rationale Price* % (increase/decrease from high/low) 1

August 26, 2021- Nov 23, 2021

No manipulative pattern could be ascertained. The price of the scrip closed at Rs.63 and reached a high price of Rs. 153.05 and closed at Rs.132.5 on November 23, 2021.

+142.93% 2

Nov 24, 2021 – June 29, 2022

Share Face value split from Rs.10 to Rs.5, price got adjusted due to corporate action from June 30, 2022. Post adjusted price from June 30, 2022, the scrip reached high price on Nov 11, 2022. The price of the scrip closed at Rs. 139 on Nov 24, 2021 and reached a high price of Rs. 489 and closed at 455 on June 29, 2022.

+251.7 % 3

June 30, 2022- Nov 11, 2022 The price of the scrip closed at Rs. 93.15 on June 30, 2022 and a reached a high price of Rs. 296.05 and closed at Rs.

296.05 on Nov 11, 2022

+217.8% 4

Nov

12,

2022-

March 31, 2023 Post-split of face value, the share price reached a highest price on Nov 11, 2022 and post that the price of the scrip has fallen. The price of scrip closed at Rs. 289.8 on Nov 14, 2022 and reached a low price of Rs.

119.05 and closed at 179.1 on March 31, 2023.

-58.9 %
"""


class NumericFactExtractionTests(unittest.TestCase):
    def test_extracts_du_digital_overall_price_facts_and_patch_rows(self) -> None:
        pages = (
            MetadataPageText(page_no=2, text=_DU_PAGE_2),
            MetadataPageText(page_no=3, text=_DU_PAGE_3),
        )
        price_movements = extract_price_movements(document_version_id=118, pages=pages)
        facts = extract_numeric_facts(
            document_version_id=118,
            pages=pages,
            chunks=(),
            price_movements=price_movements,
            title="Final Order in the matter of trading activities of certain entities in the scrip of DU Digital Technologies Limited (now DU Digital Global Limited)",
        )

        fact_map = {fact.fact_type: fact for fact in facts if fact.fact_type in {"listing_price", "closing_price", "percentage_change", "highest_price", "percentage_change_from_listing"}}

        self.assertEqual(len(price_movements), 4)
        self.assertEqual(fact_map["listing_price"].value_numeric, 12.0)
        self.assertEqual(fact_map["closing_price"].value_numeric, 179.10)
        self.assertEqual(fact_map["percentage_change"].value_numeric, 1392.5)
        self.assertEqual(fact_map["highest_price"].value_numeric, 296.05)
        self.assertEqual(fact_map["percentage_change_from_listing"].value_numeric, 2467.0)
        self.assertEqual(price_movements[0].period_label, "Patch 1")
        self.assertEqual(price_movements[0].start_price, 63.0)
        self.assertEqual(price_movements[1].high_price, 489.0)
        self.assertEqual(price_movements[2].end_price, 296.05)
        self.assertEqual(price_movements[3].low_price, 119.05)
        self.assertEqual(price_movements[3].pct_change, -58.9)

    def test_extracts_settlement_and_penalty_amounts_from_structured_sentences(self) -> None:
        chunks = (
            MetadataChunkText(
                chunk_id=1,
                page_start=5,
                page_end=5,
                text=(
                    "The applicant remitted the settlement amount of Rs. 25,00,000. "
                    "A penalty of Rs. 10,00,000 was imposed."
                ),
            ),
        )

        facts = extract_numeric_facts(
            document_version_id=501,
            pages=(),
            chunks=chunks,
            title="Settlement Order in the matter of Example Limited",
        )

        settlement = next(fact for fact in facts if fact.fact_type == "settlement_amount")
        penalty = next(fact for fact in facts if fact.fact_type == "penalty_amount")

        self.assertEqual(settlement.value_text, "Rs. 25,00,000")
        self.assertEqual(settlement.value_numeric, 2500000.0)
        self.assertEqual(penalty.value_text, "Rs. 10,00,000")
        self.assertEqual(penalty.value_numeric, 1000000.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
