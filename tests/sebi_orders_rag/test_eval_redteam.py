from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.case_loader import build_dataset
from app.sebi_orders_rag.evaluation.redteam import build_redteam_cases
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT, EVAL_FAILURE_DUMP_FIXTURE_ROOT


class EvalRedteamTests(unittest.TestCase):
    def test_redteam_generation_returns_first_class_eval_cases(self) -> None:
        base_cases, _, _ = build_dataset(
            name="sebi_eval",
            control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            failure_dump_root=EVAL_FAILURE_DUMP_FIXTURE_ROOT,
        )

        redteam_cases = build_redteam_cases(base_cases)

        self.assertGreaterEqual(len(redteam_cases), 5)
        self.assertTrue(all(case.issue_class == "redteam" for case in redteam_cases))
        self.assertTrue(all("redteam" in case.tags for case in redteam_cases))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
