from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.dataset import case_from_dict
from app.sebi_orders_rag.evaluation.metrics_grounding import evaluate_grounding_metrics
from app.sebi_orders_rag.evaluation.schemas import EvaluationCase, NumericMetrics, RetrievalMetrics


class EvalGroundingMetricTests(unittest.TestCase):
    def test_grounding_detects_missing_metadata_support(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Who signed this case?",
            expected_record_keys=("external:100486",),
            must_use_metadata=True,
        )

        metrics = evaluate_grounding_metrics(
            case=case,
            route_mode="hierarchical_rag",
            answer_status="answered",
            answer_text="The order was signed by someone.",
            actual_record_keys=("external:100486",),
            citations=({"record_key": "external:100486"},),
            debug={"metadata_debug": {"used": False}, "web_fallback_debug": {}},
            retrieval=RetrievalMetrics(context_precision=1.0, mixed_record_contamination=False),
            numeric=NumericMetrics(),
            judge=None,
        )

        self.assertFalse(metrics.used_metadata_correctly)
        self.assertGreaterEqual(metrics.missing_critical_info_count, 1)

    def test_grounding_handles_expected_abstain(self) -> None:
        case = EvaluationCase(
            case_id="case-2",
            query="Imaginary Capital settlement amount",
            must_abstain=True,
        )

        metrics = evaluate_grounding_metrics(
            case=case,
            route_mode="abstain",
            answer_status="abstained",
            answer_text="I cannot safely answer that from the available grounded evidence.",
            actual_record_keys=(),
            citations=(),
            debug={"metadata_debug": {"used": False}, "web_fallback_debug": {}},
            retrieval=RetrievalMetrics(mixed_record_contamination=False),
            numeric=NumericMetrics(),
            judge=None,
        )

        self.assertTrue(metrics.abstain_correct)
        self.assertEqual(metrics.hallucination_rate, 0.0)

    def test_regression_guidance_accepts_explicit_exemption_scope_denial(self) -> None:
        case = EvaluationCase(
            case_id="case-3",
            query="What exemption did SEBI grant in the JP Morgan Chase Bank N.A. matter?",
            issue_class="regression",
            metadata={
                "answer_guidance": (
                    "This is a settlement order, not an exemption order; answer should abstain "
                    "or explicitly say no exemption order is in scope."
                )
            },
        )

        metrics = evaluate_grounding_metrics(
            case=case,
            route_mode="exact_lookup",
            answer_status="answered",
            answer_text="This is a settlement order, not an exemption order.",
            actual_record_keys=("external:100486",),
            citations=({"record_key": "external:100486"},),
            debug={"metadata_debug": {"used": False}, "web_fallback_debug": {}},
            retrieval=RetrievalMetrics(context_precision=1.0, mixed_record_contamination=False),
            numeric=NumericMetrics(),
            judge=None,
        )

        self.assertEqual(metrics.answer_correctness, 1.0)

    def test_regression_guidance_accepts_negative_ipo_proceeds_answer(self) -> None:
        case = EvaluationCase(
            case_id="case-4",
            query="What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
            issue_class="regression",
            metadata={
                "answer_guidance": (
                    "The answer should not relabel preferential allotment financing as IPO proceeds."
                )
            },
        )

        metrics = evaluate_grounding_metrics(
            case=case,
            route_mode="exact_lookup",
            answer_status="answered",
            answer_text=(
                "The cited order does not describe IPO proceeds. It instead discusses a "
                "preferential allotment."
            ),
            actual_record_keys=("derived:pacheli",),
            citations=({"record_key": "derived:pacheli"},),
            debug={"metadata_debug": {"used": False}, "web_fallback_debug": {}},
            retrieval=RetrievalMetrics(context_precision=1.0, mixed_record_contamination=False),
            numeric=NumericMetrics(),
            judge=None,
        )

        self.assertEqual(metrics.answer_correctness, 1.0)

    def test_scaffold_gold_answer_short_does_not_zero_out_correct_exact_lookup(self) -> None:
        case = case_from_dict(
            {
                "case_id": "case-5",
                "query": "Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                "issue_class": "gold_fact",
                "gold_answer_short": "Session-seeding exact lookup.",
                "expected_record_keys": ["external:100486"],
            }
        )

        metrics = evaluate_grounding_metrics(
            case=case,
            route_mode="exact_lookup",
            answer_status="answered",
            answer_text="This settlement order disposed of the enforcement proceedings on agreed terms.",
            actual_record_keys=("external:100486",),
            citations=({"record_key": "external:100486"},),
            debug={"metadata_debug": {"used": False}, "web_fallback_debug": {}},
            retrieval=RetrievalMetrics(context_precision=1.0, mixed_record_contamination=False),
            numeric=NumericMetrics(),
            judge=None,
        )

        self.assertEqual(metrics.answer_correctness, 1.0)

    def test_regression_expected_anchor_guidance_accepts_grounded_answer(self) -> None:
        case = case_from_dict(
            {
                "case_id": "case-6",
                "query": "Tell me more about Hardcastle and Waud Manufacturing Ltd.",
                "issue_class": "regression",
                "gold_answer_short": (
                    "Stay on the base Hardcastle exemption order unless the user asks for the "
                    "corrigendum. Expected anchor: Exemption order in the matter of Hardcastle "
                    "and Waud Manufacturing Ltd. (derived:hardcastle). Summary anchor: "
                    "Exemption Order in the matter of Hardcastle and Waud Manufacturing Limited"
                ),
                "expected_record_keys": ["derived:hardcastle"],
            }
        )

        metrics = evaluate_grounding_metrics(
            case=case,
            route_mode="exact_lookup",
            answer_status="answered",
            answer_text=(
                "SEBI considered the exemption request for the proposed acquisition and granted "
                "relief subject to disclosure and compliance conditions."
            ),
            actual_record_keys=("derived:hardcastle",),
            citations=({"record_key": "derived:hardcastle"},),
            debug={"metadata_debug": {"used": False}, "web_fallback_debug": {}},
            retrieval=RetrievalMetrics(context_precision=1.0, mixed_record_contamination=False),
            numeric=NumericMetrics(),
            judge=None,
        )

        self.assertEqual(metrics.answer_correctness, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
