from __future__ import annotations

import unittest

from tests.sebi_orders_rag.structured_info_test_fixture import build_structured_info_service


class StaffIdLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = build_structured_info_service()

    def test_staff_id_lookup_answers_from_structured_people_rows(self) -> None:
        result = self.service.lookup(query="Whose staff id is 1668")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Staff ID 1668 belongs to Chitra Bhandari", result.answer_text)
        self.assertEqual(result.debug["detected_query_family"], "staff_id_lookup")
        self.assertTrue(result.debug["raw_staff_rows"])

    def test_unknown_staff_id_returns_clear_no_match(self) -> None:
        result = self.service.lookup(query="Whose staff id is 9999")

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertIn("No matching current directory entry was found for staff ID 9999", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
