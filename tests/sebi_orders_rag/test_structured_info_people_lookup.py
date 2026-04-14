from __future__ import annotations

import unittest

from app.sebi_orders_rag.directory_data.models import DirectoryPersonRecord, DirectoryReferenceDataset
from app.sebi_orders_rag.repositories.structured_info import build_structured_info_snapshot
from app.sebi_orders_rag.structured_info.query_service import StructuredInfoQueryService
from tests.sebi_orders_rag.structured_info_test_fixture import (
    build_structured_info_dataset,
    build_structured_info_service,
)


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

    def test_department_filtered_presence_query_answers_multi_match_directly(self) -> None:
        base_dataset = build_structured_info_dataset()
        dataset = DirectoryReferenceDataset(
            people=base_dataset.people
            + (
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Abhishek Kumar",
                    designation="Assistant Manager",
                    role_group="staff",
                    department_name="Information Technology Department",
                    office_name="SEBI Bhavan BKC",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Abhishek Nareshkumar Yadav",
                    designation="Assistant Manager",
                    role_group="staff",
                    department_name="Information Technology Department",
                    office_name="SEBI Bhavan BKC",
                ).with_hash(),
            ),
            board_members=base_dataset.board_members,
            offices=base_dataset.offices,
            org_structure=base_dataset.org_structure,
        )
        service = StructuredInfoQueryService(
            snapshot_loader=lambda: build_structured_info_snapshot(dataset),
            provider_name="canonical_structured_info",
        )

        result = service.lookup(query="Is there an abhishek in ITD")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("lists 2 matching entries for ITD", result.answer_text)
        self.assertIn("Abhishek Kumar", result.answer_text)
        self.assertIn("Abhishek Nareshkumar Yadav", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
