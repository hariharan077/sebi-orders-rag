from __future__ import annotations

import unittest

from app.sebi_orders_rag.directory_data.models import DirectoryPersonRecord, DirectoryReferenceDataset
from app.sebi_orders_rag.directory_data.service import DirectoryReferenceQueryService


class PersonFuzzyLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Rajudeen",
                    designation="Deputy General Manager",
                    role_group="staff",
                    email="rajudeen@sebi.gov.in",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Prasenjit Dey",
                    designation="Deputy General Manager",
                    role_group="staff",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Rina Patel",
                    designation="Assistant Manager",
                    role_group="staff",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory-alt",
                    canonical_name="Rina Patel",
                    designation="Assistant Manager",
                    role_group="staff",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Arun Kumar",
                    designation="Assistant Manager",
                    role_group="staff",
                ).with_hash(),
            ),
        )
        self.service = DirectoryReferenceQueryService(
            dataset_loader=lambda: self.dataset,
            provider_name="structured_directory",
        )

    def test_plain_name_fragment_matches_structured_person_fuzzily(self) -> None:
        answered_queries = ("prasenjith dey", "prasenjit dey")
        for query in answered_queries:
            with self.subTest(query=query):
                result = self.service.lookup(query=query)
                self.assertEqual(result.answer_status, "answered")
                self.assertIn("Prasenjit Dey is listed as Deputy General Manager", result.answer_text)

        clarification_result = self.service.lookup(query="prasanjith dey")
        self.assertEqual(clarification_result.answer_status, "insufficient_context")
        self.assertIn("Did you mean Prasenjit Dey", clarification_result.answer_text)

    def test_designation_query_handles_current_person_variant(self) -> None:
        result = self.service.lookup(query="What is the designation of Rajudeen?")
        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Rajudeen is listed as Deputy General Manager", result.answer_text)

    def test_missing_person_fails_honestly(self) -> None:
        result = self.service.lookup(query="designation of Bhuvanesh")
        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertIn("No matching current directory entry was found", result.answer_text)

    def test_count_query_counts_canonical_people_not_duplicate_rows(self) -> None:
        result = self.service.lookup(query="how many am are there in sebi")
        self.assertEqual(result.answer_status, "answered")
        self.assertIn('currently lists 2 entries matching the designation "Assistant Manager"', result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
