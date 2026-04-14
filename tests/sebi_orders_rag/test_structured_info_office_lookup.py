from __future__ import annotations

import unittest
from uuid import uuid4

from app.sebi_orders_rag.schemas import ChatSessionStateRecord
from tests.sebi_orders_rag.structured_info_test_fixture import build_structured_info_service


class StructuredInfoOfficeLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = build_structured_info_service()

    def test_city_office_lookup_answers_from_canonical_offices(self) -> None:
        result = self.service.lookup(query="where is sebi office in chennai")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Chennai - 600002", result.answer_text)
        self.assertEqual(result.sources[0].record_key, "official:contact_us")

    def test_generic_mumbai_follow_up_lists_multiple_offices(self) -> None:
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            current_lookup_family="office_contact",
            current_lookup_query="where is sebi office in chennai",
        )

        result = self.service.lookup(query="In mumbai?", session_state=state)

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("multiple official offices in Mumbai", result.answer_text)
        self.assertIn("SEBI Bhavan BKC", result.answer_text)
        self.assertIn("SEBI Bhavan II BKC", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
