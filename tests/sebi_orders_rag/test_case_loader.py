from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.case_loader import (
    _load_control_pack_eval_cases,
    _load_control_pack_wrong_answer_cases,
)
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT


class CaseLoaderTests(unittest.TestCase):
    def test_control_pack_eval_notes_do_not_become_gold_answers(self) -> None:
        cases = _load_control_pack_eval_cases(CONTROL_PACK_FIXTURE_ROOT)
        by_query = {case.query: case for case in cases}

        exact_lookup_case = by_query["Settlement Order in the matter of JP Morgan Chase Bank N.A."]
        follow_up_case = by_query["What did SEBI finally direct?"]

        self.assertIsNone(exact_lookup_case.gold_answer_short)
        self.assertEqual(exact_lookup_case.notes, "Session-seeding exact lookup.")
        self.assertIsNone(follow_up_case.gold_answer_short)
        self.assertEqual(follow_up_case.notes, "Follow-up should stay anchored to prior record.")

    def test_wrong_answer_guidance_only_seeds_gold_text_for_semantic_regressions(self) -> None:
        cases = _load_control_pack_wrong_answer_cases(CONTROL_PACK_FIXTURE_ROOT)
        by_query = {case.query: case for case in cases}

        anchor_case = by_query["Tell me more about Hardcastle and Waud Manufacturing Ltd."]
        semantic_case = by_query["What exemption did SEBI grant in the JP Morgan Chase Bank N.A. matter?"]

        self.assertIsNone(anchor_case.gold_answer_short)
        self.assertIn("Stay on the base Hardcastle exemption order", anchor_case.notes)
        self.assertEqual(
            anchor_case.metadata.get("answer_guidance"),
            anchor_case.notes,
        )

        self.assertIn("answer should abstain", semantic_case.gold_answer_short or "")
        self.assertEqual(
            semantic_case.metadata.get("answer_guidance"),
            semantic_case.notes,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
