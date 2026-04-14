from __future__ import annotations

import unittest

from app.sebi_orders_rag.directory_data.canonicalize import canonicalize_reference_dataset, match_canonical_offices
from app.sebi_orders_rag.directory_data.models import (
    BoardMemberRecord,
    DirectoryOfficeRecord,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
)


class DirectoryCanonicalizationTests(unittest.TestCase):
    def test_canonical_wtm_deduplication_merges_directory_orgchart_and_board_rows(self) -> None:
        dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Amarjeet Singh",
                    designation="Whole Time Member",
                    role_group="wtm",
                    email="amarjeets@sebi.gov.in",
                    phone="022-40459991",
                    date_of_joining="Sep 01, 2023",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="orgchart",
                    source_url="https://www.sebi.gov.in/orgchart-grid.html",
                    canonical_name="Shri Amarjeet Singh",
                    designation="Whole-Time Member",
                    role_group="wtm",
                ).with_hash(),
            ),
            board_members=(
                BoardMemberRecord(
                    source_url="https://www.sebi.gov.in/sebiweb/boardmember/BoardMemberAction.do?doBoardMember=yes&lang=en",
                    canonical_name="Shri Amarjeet Singh",
                    board_role="Whole-Time Member, SEBI",
                    category="whole_time_member",
                ).with_hash(),
            ),
        )

        canonical = canonicalize_reference_dataset(dataset)
        wtms = [record for record in canonical.people if record.role_group == "wtm"]

        self.assertEqual(len(wtms), 1)
        self.assertEqual(wtms[0].canonical_name, "Amarjeet Singh")
        self.assertEqual(wtms[0].phone, "022-40459991")
        self.assertEqual(wtms[0].date_of_joining, "Sep 01, 2023")
        self.assertTrue(wtms[0].is_board_member)

    def test_canonical_executive_director_deduplication_keeps_department_and_contact_details(self) -> None:
        dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Manoj Kumar",
                    designation="Executive Director",
                    role_group="executive_director",
                    email="manojk@sebi.gov.in",
                    phone="022-40459260",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="orgchart",
                    source_url="https://www.sebi.gov.in/orgchart-grid.html",
                    canonical_name="Shri Manoj Kumar",
                    designation="Executive Director",
                    role_group="executive_director",
                    department_name="Investment Management Department (IMD)",
                ).with_hash(),
            )
        )

        canonical = canonicalize_reference_dataset(dataset)
        directors = [record for record in canonical.people if record.role_group == "executive_director"]

        self.assertEqual(len(directors), 1)
        self.assertEqual(directors[0].department_name, "Investment Management Department (IMD)")
        self.assertEqual(directors[0].email, "manojk@sebi.gov.in")
        self.assertEqual(directors[0].phone, "022-40459260")

    def test_office_consolidation_prefers_contact_us_but_matches_regional_aliases(self) -> None:
        dataset = DirectoryReferenceDataset(
            offices=(
                DirectoryOfficeRecord(
                    source_type="regional_offices",
                    source_url="https://www.sebi.gov.in/department/regional-offices-43/contact.html",
                    office_name="Southern Regional Office (SRO), Chennai",
                    office_type="regional_office",
                    region="south",
                    city="Chennai",
                ).with_hash(),
                DirectoryOfficeRecord(
                    source_type="contact_us",
                    source_url="https://www.sebi.gov.in/contact-us.html",
                    office_name="Southern Regional Office (SRO)",
                    office_type="regional_office",
                    region="south",
                    address="7th Floor, 756-L, Anna Salai, Chennai - 600002, Tamil Nadu",
                    phone="+91-44-28880222 / 28526686",
                    email="sebisro@sebi.gov.in",
                    city="Chennai",
                    state="Tamil Nadu",
                ).with_hash(),
            )
        )

        canonical = canonicalize_reference_dataset(dataset)
        matches = match_canonical_offices(canonical.offices, "where is SEBI office in Chennai")

        self.assertEqual(len(canonical.offices), 1)
        self.assertEqual(matches[0].canonical_name, "Southern Regional Office (SRO), Chennai")
        self.assertIn("Chennai - 600002", matches[0].address or "")
        self.assertIn("sro", matches[0].aliases)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
