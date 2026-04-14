from __future__ import annotations

import unittest

from tests.sebi_orders_rag.structured_info_test_fixture import build_structured_info_service


class StructuredInfoCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = build_structured_info_service()

    def test_assistant_manager_count_is_built_from_canonical_people(self) -> None:
        for query in (
            "how many assistant managers are there in sebi",
            "how many assistant managers are currently serving in sebi",
        ):
            with self.subTest(query=query):
                result = self.service.lookup(query=query)
                self.assertEqual(result.answer_status, "answered")
                self.assertIn('currently lists 2 entries matching the designation "Assistant Manager"', result.answer_text)
                self.assertEqual(result.debug["count_debug"]["count"], 2)
                self.assertIn("Chitra Bhandari", result.debug["count_debug"]["contributing_names"])
                self.assertIn("Chitra M", result.debug["count_debug"]["contributing_names"])

    def test_wtm_and_ed_counts_use_canonical_role_counts(self) -> None:
        wtm_result = self.service.lookup(
            query="how many whole time members are serving in sebi currently and who are they"
        )
        ed_result = self.service.lookup(query="who are the ed")

        self.assertEqual(wtm_result.answer_status, "answered")
        self.assertIn("SEBI currently has 2 Whole-Time Members", wtm_result.answer_text)
        self.assertIn("Amarjeet Singh", wtm_result.answer_text)
        self.assertIn("Kamlesh Chandra Varshney", wtm_result.answer_text)

        self.assertEqual(ed_result.answer_status, "answered")
        self.assertIn("SEBI currently has 2 Executive Directors", ed_result.answer_text)
        self.assertIn("Manoj Kumar", ed_result.answer_text)
        self.assertIn("Nandita Rao", ed_result.answer_text)

    def test_board_member_summary_uses_canonical_board_layer(self) -> None:
        result = self.service.lookup(query="who are the board members of sebi")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("SEBI currently has 5 board members.", result.answer_text)
        self.assertIn("Tuhin Kanta Pandey", result.answer_text)
        self.assertIn("G Anantharaman", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
