from __future__ import annotations

import unittest

from app.sebi_orders_rag.directory_data.models import DirectoryPersonRecord, DirectoryReferenceDataset
from app.sebi_orders_rag.directory_data.service import DirectoryReferenceQueryService
from app.sebi_orders_rag.normalization import rank_fuzzy_candidates


class FuzzySafetyTests(unittest.TestCase):
    def test_high_medium_and_low_fuzzy_bands_are_deterministic(self) -> None:
        high = rank_fuzzy_candidates(
            "prasenjith dey",
            ("Prasenjit Dey",),
            key=lambda value: value,
            min_score=0.7,
            medium_score=0.8,
            confident_score=0.84,
            ambiguity_gap=0.05,
        )
        medium = rank_fuzzy_candidates(
            "prasenjeet dey",
            ("Prasenjit Dey",),
            key=lambda value: value,
            min_score=0.7,
            medium_score=0.8,
            confident_score=0.84,
            ambiguity_gap=0.05,
        )
        low = rank_fuzzy_candidates(
            "shruti singh",
            ("Ishpreet Singh",),
            key=lambda value: value,
            min_score=0.7,
            medium_score=0.8,
            confident_score=0.84,
            ambiguity_gap=0.05,
        )

        self.assertEqual(high.band, "high")
        self.assertEqual(medium.band, "medium")
        self.assertEqual(low.band, "low")

    def test_medium_confidence_person_match_returns_clarification(self) -> None:
        dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Prasenjit Dey",
                    designation="Deputy General Manager",
                    department_name="IVD",
                    office_name="Head Office, Mumbai",
                    role_group="staff",
                ).with_hash(),
            ),
        )
        service = DirectoryReferenceQueryService(
            dataset_loader=lambda: dataset,
            provider_name="structured_directory",
        )

        result = service.lookup(query="designation of prasenjeet dey")

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertEqual(result.debug["fuzzy_band"], "medium")
        self.assertIn("Did you mean Prasenjit Dey, Deputy General Manager in IVD", result.answer_text)

    def test_low_confidence_person_match_abstains_cleanly(self) -> None:
        dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://example.com/directory",
                    canonical_name="Ishpreet Singh",
                    designation="Assistant General Manager",
                    role_group="staff",
                ).with_hash(),
            ),
        )
        service = DirectoryReferenceQueryService(
            dataset_loader=lambda: dataset,
            provider_name="structured_directory",
        )

        result = service.lookup(query="designation of shruti singh")

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertEqual(result.debug["fuzzy_band"], "low")
        self.assertIn("No matching current directory entry was found", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
