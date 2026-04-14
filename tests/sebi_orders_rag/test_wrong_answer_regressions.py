from __future__ import annotations

import unittest
from uuid import uuid4

from app.sebi_orders_rag.eval.evaluator import _evaluate_payload
from app.sebi_orders_rag.schemas import ChatAnswerPayload, Citation


class WrongAnswerRegressionTests(unittest.TestCase):
    def test_grounded_active_matter_follow_up_does_not_fail_only_for_missing_lock_flag(self) -> None:
        expected_record_key = "derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e"
        payload = ChatAnswerPayload(
            session_id=uuid4(),
            route_mode="memory_scoped_rag",
            query_intent="follow_up",
            answer_text=(
                "The cited order text says the acquisition was exempted from the open-offer obligation "
                "under Regulation 11."
            ),
            confidence=0.91,
            citations=(
                Citation(
                    citation_number=1,
                    record_key=expected_record_key,
                    title="Exemption order in the matter of Hardcastle and Waud Manufacturing Ltd.",
                    page_start=7,
                    page_end=7,
                    section_type="operative_order",
                    document_version_id=101,
                    chunk_id=7001,
                    detail_url="https://example.com/detail/hardcastle",
                    pdf_url="https://example.com/pdf/hardcastle.pdf",
                ),
            ),
            active_record_keys=(expected_record_key,),
            answer_status="answered",
            debug={
                "route_debug": {
                    "strict_single_matter": False,
                    "strict_scope_required": False,
                    "active_order_override": True,
                    "active_matter_follow_up_intent": "exemption_granted",
                },
                "mixed_record_guardrail": {
                    "single_matter_rule_respected": True,
                    "guardrail_fired": False,
                },
            },
        )

        result = _evaluate_payload(
            payload=payload,
            case_kind="eval_query",
            query="What exemption was granted?",
            expected_route_mode="memory_scoped_rag",
            expected_record_key=expected_record_key,
            expected_title="Exemption order in the matter of Hardcastle and Waud Manufacturing Ltd.",
            comparison_allowed=False,
            incorrect_record_keys=(),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, ())
        self.assertFalse(result.strict_single_matter_triggered)
        self.assertTrue(result.single_matter_rule_respected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
