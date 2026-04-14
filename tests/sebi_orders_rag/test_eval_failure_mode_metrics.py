from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.metrics_failure_modes import classify_failure_modes
from app.sebi_orders_rag.evaluation.schemas import (
    EvaluationCase,
    FailureModeMetrics,
    GroundingMetrics,
    NumericMetrics,
    RetrievalMetrics,
    RouteMetrics,
)


class EvalFailureModeMetricTests(unittest.TestCase):
    def test_missing_clarify_is_bucketed(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Prime Broking Company (India) Limited",
            must_clarify=True,
        )

        failure = classify_failure_modes(
            case=case,
            route_mode="exact_lookup",
            route=RouteMetrics(strict_route_match=False, equivalent_route_match=False),
            retrieval=RetrievalMetrics(candidate_list_correctness=0.0),
            grounding=GroundingMetrics(answer_correctness=0.0),
            numeric=NumericMetrics(),
            answer_status="answered",
            answer_text="Merged answer",
            debug={"candidate_list_debug": {"used": False}},
            judge=None,
        )

        self.assertEqual(failure.primary_bucket, "missing clarify")

    def test_numeric_miss_and_person_trust_confusion_are_detected(self) -> None:
        case = EvaluationCase(
            case_id="case-2",
            query="How many shares did Aruna Dhanuka hold?",
            must_use_metadata=True,
            tags=("person_vs_trust",),
        )

        failure = classify_failure_modes(
            case=case,
            route_mode="hierarchical_rag",
            route=RouteMetrics(strict_route_match=True, equivalent_route_match=True),
            retrieval=RetrievalMetrics(),
            grounding=GroundingMetrics(answer_correctness=0.6),
            numeric=NumericMetrics(expected_fact_count=2, matched_fact_count=1, numeric_accuracy=0.5),
            answer_status="answered",
            answer_text="The trust held 23.93%.",
            debug={"metadata_debug": {"used": True}},
            judge=None,
        )

        self.assertIn("numeric fact extraction miss", failure.buckets)
        self.assertIn("person-vs-trust confusion", failure.buckets)

    def test_correct_abstain_case_is_not_marked_as_weak_metadata_support(self) -> None:
        case = EvaluationCase(
            case_id="case-3",
            query="What was the settlement amount in the Imaginary Capital Limited settlement order?",
            must_use_metadata=True,
            must_abstain=True,
        )

        failure = classify_failure_modes(
            case=case,
            route_mode="abstain",
            route=RouteMetrics(strict_route_match=True, equivalent_route_match=True),
            retrieval=RetrievalMetrics(),
            grounding=GroundingMetrics(answer_correctness=1.0, abstain_correct=True),
            numeric=NumericMetrics(),
            answer_status="abstained",
            answer_text="I cannot safely answer that from a real SEBI matter.",
            debug={"metadata_debug": {"used": False}},
            judge=None,
        )

        self.assertEqual(failure.primary_bucket, "pass")
        self.assertNotIn("weak metadata support", failure.buckets)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
