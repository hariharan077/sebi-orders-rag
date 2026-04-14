from __future__ import annotations

import unittest

from tests.sebi_orders_rag.structured_info_test_fixture import build_structured_info_service


class StructuredInfoPeopleLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = build_structured_info_service()

    def test_exact_and_sebi_context_person_queries_answer_from_canonical_people(self) -> None:
        expectations = {
            "chitra bhandari sebi": "Chitra Bhandari is listed as Assistant Manager",
            "who is dron amrit": "Dron Kumar Amrit is listed as Deputy General Manager",
            "rajudeen": "Rajudeen is listed as Deputy General Manager",
            "what is the designation of lenin": "Kandunuri Lenin is listed as General Manager",
            "who is tuhin": "Tuhin Kanta Pandey is SEBI's Chairperson.",
        }

        for query, expected in expectations.items():
            with self.subTest(query=query):
                result = self.service.lookup(query=query)
                self.assertEqual(result.answer_status, "answered")
                self.assertIn(expected, result.answer_text)

    def test_in_sebi_normalization_prefers_canonical_people_lookup(self) -> None:
        result = self.service.lookup(query="who is rajudeen in sebi")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Rajudeen is listed as Deputy General Manager", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
