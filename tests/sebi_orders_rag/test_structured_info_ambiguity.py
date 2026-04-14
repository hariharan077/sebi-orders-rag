from __future__ import annotations

import unittest

from tests.sebi_orders_rag.structured_info_test_fixture import build_structured_info_service


class StructuredInfoAmbiguityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = build_structured_info_service()

    def test_plain_single_name_query_clarifies_when_multiple_current_people_match(self) -> None:
        result = self.service.lookup(query="who is chitra")

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertIn("Did you mean", result.answer_text)
        self.assertIn("Chitra Bhandari", result.answer_text)
        self.assertIn("Chitra M", result.answer_text)

    def test_unique_single_name_query_answers_directly(self) -> None:
        result = self.service.lookup(query="who is tuhin")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Tuhin Kanta Pandey", result.answer_text)
        self.assertEqual(result.debug["person_match_status"], "exact_alias")

    def test_exact_alias_query_answers_directly(self) -> None:
        result = self.service.lookup(query="who is dron amrit")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Dron Kumar Amrit", result.answer_text)
        self.assertEqual(result.debug["person_match_status"], "exact_alias")

    def test_medium_confidence_variant_clarifies_instead_of_over_committing(self) -> None:
        result = self.service.lookup(query="designation of prasanjith dey")

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertIn("Did you mean Prasenjit Dey", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
