from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.metrics_numeric import evaluate_numeric_metrics
from app.sebi_orders_rag.evaluation.schemas import EvaluationCase, GoldNumericFact


class EvalNumericMetricTests(unittest.TestCase):
    def test_numeric_scoring_matches_du_digital_anchor(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="How much did DU Digital share price increase?",
            gold_numeric_facts=(
                GoldNumericFact(
                    fact_type="percentage_change",
                    value_numeric=1392.5,
                    value_text="1392.5%",
                    unit="percent",
                ),
                GoldNumericFact(
                    fact_type="highest_price",
                    value_numeric=296.05,
                    value_text="Rs.296.05",
                    unit="INR",
                ),
            ),
        )
        answer_text = (
            "The order metadata records that DU Digital increased by 1392.5% and reached "
            "a highest price of Rs.296.05."
        )

        metrics = evaluate_numeric_metrics(case=case, answer_text=answer_text)

        self.assertEqual(metrics.matched_fact_count, 2)
        self.assertEqual(metrics.numeric_accuracy, 1.0)

    def test_numeric_scoring_flags_missing_order_date(self) -> None:
        case = EvaluationCase(
            case_id="case-2",
            query="When was this order passed?",
            gold_numeric_facts=(
                GoldNumericFact(
                    fact_type="order_date",
                    value_text="2026-03-20",
                ),
            ),
        )

        metrics = evaluate_numeric_metrics(case=case, answer_text="The order date is not available.")

        self.assertEqual(metrics.matched_fact_count, 0)
        self.assertIn("order_date", metrics.missing_fact_types)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
