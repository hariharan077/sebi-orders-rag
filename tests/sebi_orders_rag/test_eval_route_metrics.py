from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.metrics_route import evaluate_route_metrics
from app.sebi_orders_rag.evaluation.schemas import EvaluationCase


class EvalRouteMetricTests(unittest.TestCase):
    def test_named_single_matter_routes_are_equivalent(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Tell me more about Vishvaraj Environment Limited",
            route_family_expected="hierarchical_rag",
            expected_record_keys=("derived:vishvaraj",),
        )

        metrics = evaluate_route_metrics(
            case=case,
            route_mode="exact_lookup",
            answer_status="answered",
            debug={
                "route_debug": {
                    "strict_scope_required": True,
                    "strict_single_matter": True,
                }
            },
            actual_record_keys=("derived:vishvaraj",),
        )

        self.assertFalse(metrics.strict_route_match)
        self.assertTrue(metrics.equivalent_route_match)
        self.assertEqual(metrics.equivalent_route_reason, "named_single_matter_equivalent")

    def test_current_info_routes_are_equivalent(self) -> None:
        case = EvaluationCase(
            case_id="case-2",
            query="Who is the previous chairman of SEBI?",
            route_family_expected="historical_official_lookup",
            must_use_official_web=True,
        )

        metrics = evaluate_route_metrics(
            case=case,
            route_mode="current_official_lookup",
            answer_status="answered",
            debug={
                "route_debug": {
                    "appears_historical_official_lookup": True,
                },
                "planner_debug": {
                    "used": True,
                    "execution_route_mode": "current_official_lookup",
                },
                "web_fallback_debug": {"official_web_attempted": True},
            },
            actual_record_keys=(),
        )

        self.assertTrue(metrics.equivalent_route_match)
        self.assertEqual(metrics.equivalent_route_reason, "current_info_equivalent")
        self.assertTrue(metrics.web_fallback_correct)

    def test_ambiguous_named_matter_clarify_counts_as_equivalent_when_expected_record_is_listed(self) -> None:
        case = EvaluationCase(
            case_id="case-3",
            query="Prime Broking Company (India) Limited",
            route_family_expected="exact_lookup",
            expected_record_keys=("external:30222",),
        )

        metrics = evaluate_route_metrics(
            case=case,
            route_mode="clarify",
            answer_status="clarify",
            debug={
                "route_debug": {
                    "strict_scope_required": True,
                    "strict_single_matter": False,
                },
                "candidate_list_debug": {
                    "used": True,
                    "candidate_source": "strict_matter_lock",
                    "record_keys": [
                        "external:30222",
                        "external:30223",
                        "external:30189",
                    ],
                },
            },
            actual_record_keys=(),
        )

        self.assertTrue(metrics.equivalent_route_match)
        self.assertEqual(metrics.equivalent_route_reason, "ambiguous_named_matter_clarify")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
