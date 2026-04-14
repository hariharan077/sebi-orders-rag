from __future__ import annotations

import unittest

from app.sebi_orders_rag.directory_data.models import DirectoryPersonRecord, DirectoryReferenceDataset
from app.sebi_orders_rag.directory_data.service import DirectoryReferenceQueryService


class StructuredPeopleFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Kumar Abhishek",
                    designation="Manager",
                    department_name="ITD-2",
                    office_name="Head Office, Mumbai",
                    role_group="staff",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Rajudeen",
                    designation="Deputy General Manager",
                    department_name="IVD",
                    office_name="Head Office, Mumbai",
                    role_group="staff",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Bhuvanesh",
                    designation="Assistant Manager",
                    department_name="IVD",
                    office_name="Head Office, Mumbai",
                    role_group="staff",
                ).with_hash(),
            ),
        )
        self.service = DirectoryReferenceQueryService(
            dataset_loader=lambda: self.dataset,
            provider_name="structured_directory",
        )

    def test_name_and_department_filter_returns_single_answer(self) -> None:
        result = self.service.lookup(query="Is there an abhishek in ITD")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Kumar Abhishek is listed as Manager", result.answer_text)
        self.assertEqual(result.debug["department_hint"], "ITD")

    def test_designation_called_pattern_returns_answer(self) -> None:
        result = self.service.lookup(query="assistant manager called Bhuvanesh")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Bhuvanesh is listed as Assistant Manager", result.answer_text)

    def test_department_only_people_query_lists_matching_rows(self) -> None:
        result = self.service.lookup(query="who is in IVD")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("matching entries for IVD", result.answer_text)
        self.assertIn("Rajudeen", result.answer_text)
        self.assertIn("Bhuvanesh", result.answer_text)

    def test_unsupported_division_hierarchy_filter_fails_honestly(self) -> None:
        result = self.service.lookup(query="who is the division chief of IVD ID7")

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertIn("does not expose division-level hierarchy", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
