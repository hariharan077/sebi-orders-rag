from __future__ import annotations

import unittest

from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class SatCourtRoutePriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = AdaptiveQueryRouter()
        self.router_with_pack = AdaptiveQueryRouter(control_pack=load_control_pack(REAL_CONTROL_PACK))

    def test_vs_sebi_queries_do_not_route_to_structured_current_info(self) -> None:
        for query in (
            "suresh bharrat vs sebi",
            "umashanker vs sebi",
            "tushar oil food ltd vs sebi",
            "prime broking vs nse",
        ):
            with self.subTest(query=query):
                decision = self.router.decide(query=query)
                self.assertNotEqual(decision.route_mode, "structured_current_info")
                self.assertTrue(decision.analysis.appears_sat_court_style)
                self.assertIn(decision.route_mode, {"exact_lookup", "hierarchical_rag"})

    def test_umashankar_query_prefers_the_umashankar_sat_candidate(self) -> None:
        decision = self.router_with_pack.decide(query="umashanker vs sebi")

        self.assertEqual(decision.route_mode, "exact_lookup")
        self.assertTrue(decision.analysis.strict_matter_lock.candidates)
        self.assertEqual(
            decision.analysis.strict_matter_lock.candidates[0].record_key,
            "external:30160",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
