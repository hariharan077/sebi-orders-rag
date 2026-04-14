from __future__ import annotations

import unittest

from app.sebi_orders_rag.eval.report import EvalCaseResult
from app.sebi_orders_rag.eval.triage import load_failure_dump_reference, triage_eval_result
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT, EVAL_FAILURE_DUMP_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT
FAILURE_DUMP_ROOT = EVAL_FAILURE_DUMP_FIXTURE_ROOT


class EvalEquivalentRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(REAL_CONTROL_PACK)
        assert cls.pack is not None
        cls.failure_dump = load_failure_dump_reference(FAILURE_DUMP_ROOT)

    def test_named_single_matter_routes_are_treated_as_equivalent(self) -> None:
        result = _result(
            query="What did SEBI order for Vishvaraj Environment Limited?",
            expected_route_mode="hierarchical_rag",
            actual_route_mode="exact_lookup",
            expected_record_key="derived:vishvaraj",
            actual_record_key="derived:vishvaraj",
            reasons=("expected route hierarchical_rag got exact_lookup",),
            route_debug={
                "strict_scope_required": True,
                "strict_single_matter": True,
                "appears_matter_specific": True,
            },
        )

        triage = triage_eval_result(result, reference=self.failure_dump)

        self.assertTrue(triage.equivalent_route_passed)
        self.assertEqual(triage.equivalent_route_reason, "named_single_matter_equivalent")
        self.assertTrue(triage.stale_expectation)

    def test_general_explanatory_routes_accept_direct_llm_alias(self) -> None:
        result = _result(
            query="Explain orders issued under Regulation 30A.",
            expected_route_mode="direct_llm",
            actual_route_mode="general_knowledge",
            reasons=("expected route direct_llm got general_knowledge",),
            route_debug={
                "appears_general_explanatory": True,
                "appears_matter_specific": False,
            },
        )

        triage = triage_eval_result(result, reference=self.failure_dump)

        self.assertTrue(triage.equivalent_route_passed)
        self.assertEqual(triage.equivalent_route_reason, "general_explanatory_equivalent")

    def test_ambiguous_named_query_clarify_is_accepted(self) -> None:
        result = _result(
            query="Prime Broking Company (India) Limited",
            expected_route_mode="exact_lookup",
            actual_route_mode="clarify",
            expected_record_key="external:30222",
            reasons=(
                "expected route exact_lookup got clarify",
                "expected grounded record key but answer cited none",
            ),
            answer_status="clarify",
            debug={
                "route_debug": {
                    "strict_scope_required": True,
                    "strict_single_matter": False,
                },
                "candidate_list_debug": {
                    "used": True,
                    "candidate_source": "strict_matter_lock",
                },
            },
        )

        triage = triage_eval_result(result, reference=self.failure_dump)

        self.assertTrue(triage.equivalent_route_passed)
        self.assertEqual(triage.equivalent_route_reason, "ambiguous_named_matter_clarify")
        self.assertEqual(triage.primary_bucket, "stale expectation")

    def test_generic_order_topic_query_routes_to_general_knowledge(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.pack)

        decision = router.decide(query="What is a settlement order?")

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertTrue(decision.analysis.appears_general_explanatory)
        self.assertFalse(decision.analysis.appears_matter_specific)
        self.assertFalse(decision.analysis.strict_scope_required)


def _result(
    *,
    query: str,
    expected_route_mode: str,
    actual_route_mode: str,
    reasons: tuple[str, ...],
    route_debug: dict[str, object] | None = None,
    expected_record_key: str | None = None,
    actual_record_key: str | None = None,
    answer_status: str = "answered",
    debug: dict[str, object] | None = None,
) -> EvalCaseResult:
    payload_debug = {
        "route_debug": dict(route_debug or {}),
        "candidate_list_debug": {},
        "metadata_debug": {"used": False},
    }
    if debug:
        payload_debug.update(debug)
    actual_cited_record_keys = ((actual_record_key,) if actual_record_key else ())
    return EvalCaseResult(
        case_kind="eval_query",
        query=query,
        expected_route_mode=expected_route_mode,
        actual_route_mode=actual_route_mode,
        expected_record_key=expected_record_key,
        expected_title=None,
        actual_active_record_keys=actual_cited_record_keys,
        actual_cited_record_keys=actual_cited_record_keys,
        strict_single_matter_triggered=bool(payload_debug["route_debug"].get("strict_single_matter")),
        comparison_disabled_lock=False,
        mixed_record_guardrail_fired=False,
        single_matter_rule_respected=True,
        answer_status=answer_status,
        confidence=0.91,
        passed=False,
        reasons=reasons,
        debug=payload_debug,
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
