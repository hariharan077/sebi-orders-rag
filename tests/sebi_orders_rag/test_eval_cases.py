from __future__ import annotations

import unittest

from app.sebi_orders_rag.eval.cases import load_eval_cases, validate_eval_cases


class EvalCaseTests(unittest.TestCase):
    def test_packaged_eval_cases_are_valid_and_settlement_focused(self) -> None:
        cases = load_eval_cases()

        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(validate_eval_cases(cases), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
