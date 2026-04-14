from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.metrics_retrieval import evaluate_retrieval_metrics
from app.sebi_orders_rag.evaluation.schemas import EvaluationCase, RetrievedContext


class EvalRetrievalMetricTests(unittest.TestCase):
    def test_named_matter_retrieval_penalizes_mixed_record_contamination(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Tell me more about Vishvaraj Environment Limited",
            expected_record_keys=("derived:vishvaraj",),
            expected_bucket_names=("orders-of-chairperson-members",),
        )
        retrieved = (
            RetrievedContext(rank=1, record_key="derived:vishvaraj", bucket_name="orders-of-chairperson-members", section_type="operative_order", chunk_text="Vishvaraj chunk."),
            RetrievedContext(rank=2, record_key="external:varyaa", bucket_name="orders-of-chairperson-members", section_type="operative_order", chunk_text="Wrong chunk."),
        )

        metrics = evaluate_retrieval_metrics(case=case, retrieved_context=retrieved, debug={})

        self.assertAlmostEqual(metrics.context_precision or 0.0, 0.5)
        self.assertAlmostEqual(metrics.context_recall or 0.0, 1.0)
        self.assertTrue(metrics.mixed_record_contamination)
        self.assertAlmostEqual(metrics.single_matter_purity or 0.0, 0.5)

    def test_candidate_list_correctness_uses_expected_record_keys(self) -> None:
        case = EvaluationCase(
            case_id="case-2",
            query="Prime Broking Company (India) Limited",
            expected_record_keys=("external:30189",),
            must_clarify=True,
        )

        metrics = evaluate_retrieval_metrics(
            case=case,
            retrieved_context=(),
            debug={
                "candidate_list_debug": {
                    "used": True,
                    "record_keys": ["external:30189", "external:30161"],
                    "bucket_names": ["orders-of-ao"],
                }
            },
        )

        self.assertEqual(metrics.candidate_list_correctness, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
