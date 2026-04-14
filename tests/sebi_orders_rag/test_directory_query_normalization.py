from __future__ import annotations

import unittest
from uuid import uuid4

from app.sebi_orders_rag.directory_data.models import (
    BoardMemberRecord,
    DirectoryOfficeRecord,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
)
from app.sebi_orders_rag.directory_data.service import DirectoryReferenceQueryService, _classify_query
from app.sebi_orders_rag.schemas import ChatSessionStateRecord


class DirectoryQueryNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Rina Patel",
                    designation="Assistant Manager",
                    role_group="staff",
                    phone="022-40450000",
                    date_of_joining="Jan 12, 2024",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Rajudeen",
                    designation="Deputy General Manager",
                    role_group="staff",
                    phone="044-28880222",
                    office_name="Southern Regional Office (SRO)",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Arun Kumar",
                    designation="Assistant Manager",
                    role_group="staff",
                    phone="033-23023000",
                ).with_hash(),
            ),
            board_members=(
                BoardMemberRecord(
                    source_url="https://www.sebi.gov.in/sebiweb/boardmember/BoardMemberAction.do?doBoardMember=yes&lang=en",
                    canonical_name="Tuhin Kanta Pandey",
                    board_role="Chairman, SEBI",
                    category="chairperson",
                ).with_hash(),
            ),
            offices=(
                DirectoryOfficeRecord(
                    source_type="contact_us",
                    source_url="https://www.sebi.gov.in/contact-us.html",
                    office_name="SEBI Bhavan BKC",
                    office_type="head_office",
                    region="head_office",
                    address="Plot No.C4-A, G Block, Bandra-Kurla Complex, Bandra (East), Mumbai - 400051, Maharashtra",
                    phone="+91-22-26449000 / 40459000",
                    email="sebi@sebi.gov.in",
                    city="Mumbai",
                    state="Maharashtra",
                ).with_hash(),
                DirectoryOfficeRecord(
                    source_type="contact_us",
                    source_url="https://www.sebi.gov.in/contact-us.html",
                    office_name="SEBI Bhavan II BKC",
                    office_type="head_office",
                    region="head_office",
                    address="Plot no. C-7, G Block, Bandra Kurla Complex, Bandra(E), Mumbai - 400051, Maharashtra",
                    phone="91-22-26449000/40459000",
                    email="sebi@sebi.gov.in",
                    city="Mumbai",
                    state="Maharashtra",
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
                DirectoryOfficeRecord(
                    source_type="contact_us",
                    source_url="https://www.sebi.gov.in/contact-us.html",
                    office_name="Northern Regional Office (NRO)",
                    office_type="regional_office",
                    region="north",
                    address="NBCC Complex, New Delhi - 110023",
                    email="sebinro@sebi.gov.in",
                    city="New Delhi",
                    state="Delhi",
                ).with_hash(),
                DirectoryOfficeRecord(
                    source_type="contact_us",
                    source_url="https://www.sebi.gov.in/contact-us.html",
                    office_name="Eastern Regional Office (ERO)",
                    office_type="regional_office",
                    region="east",
                    address="L&T Chambers, 16 Camac Street, Kolkata - 700017, West Bengal",
                    phone="+91-33-23023000",
                    email="sebiero@sebi.gov.in",
                    city="Kolkata",
                    state="West Bengal",
                ).with_hash(),
            ),
        )
        self.query_service = DirectoryReferenceQueryService(
            dataset_loader=lambda: self.dataset,
            provider_name="structured_directory",
        )

    def test_query_normalization_handles_board_wtm_and_office_variants(self) -> None:
        expectations = {
            "who are the current board members of sebi?": "board_members",
            "how many wtm are there?": "wtm_list",
            "who are the executive directors of sebi?": "ed_list",
            "location of Mumbai SEBI office": "office_contact",
            "where is SEBI office in Chennai?": "office_contact",
            "SEBI office address Delhi": "office_contact",
            "where is SEBI located in Kolkata": "office_contact",
            "What is the designation of Rajudeen?": "person_lookup",
            "how may assistant managers are there in SEBI?": "designation_count",
        }

        for query, lookup_type in expectations.items():
            with self.subTest(query=query):
                self.assertEqual(_classify_query(query).lookup_type, lookup_type)

    def test_query_normalization_extracts_person_name_and_requested_details(self) -> None:
        intent = _classify_query("is there an assistant manager called Rina Patel? when did she join whats her number?")

        self.assertEqual(intent.lookup_type, "person_lookup")
        self.assertEqual(intent.person_name, "Rina Patel")
        self.assertEqual(intent.designation_hint, "Assistant Manager")
        self.assertTrue(intent.wants_joining_date)
        self.assertTrue(intent.wants_phone)

    def test_person_lookup_success_returns_joining_date_and_phone(self) -> None:
        result = self.query_service.lookup(
            query="is there an assistant manager called Rina Patel? when did she join whats her number?"
        )

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Rina Patel is listed as Assistant Manager", result.answer_text)
        self.assertIn("date of joining: Jan 12, 2024", result.answer_text)
        self.assertIn("phone: 022-40450000", result.answer_text)

    def test_person_lookup_failure_returns_clear_no_match_message(self) -> None:
        result = self.query_service.lookup(
            query="is there an assistant manager called Bhuvanesh? when did he join whats his number?"
        )

        self.assertEqual(result.answer_status, "insufficient_context")
        self.assertIn("No matching current directory entry was found", result.answer_text)

    def test_generic_mumbai_office_query_lists_multiple_offices(self) -> None:
        result = self.query_service.lookup(query="location of Mumbai SEBI office")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("multiple official offices in Mumbai", result.answer_text)
        self.assertIn("SEBI Bhavan BKC", result.answer_text)
        self.assertIn("SEBI Bhavan II BKC", result.answer_text)

    def test_city_normalization_handles_chennai_delhi_and_kolkata(self) -> None:
        expectations = {
            "where is SEBI office in Chennai?": "Chennai - 600002",
            "SEBI office address Delhi": "New Delhi - 110023",
            "where is SEBI located in Kolkata": "Kolkata - 700017",
        }

        for query, expected in expectations.items():
            with self.subTest(query=query):
                result = self.query_service.lookup(query=query)
                self.assertEqual(result.answer_status, "answered")
                self.assertIn(expected, result.answer_text)

    def test_office_follow_up_uses_narrow_office_context(self) -> None:
        session_state = ChatSessionStateRecord(
            session_id=uuid4(),
            current_lookup_family="office_contact",
            current_lookup_query="where is sebi office in chennai",
        )

        result = self.query_service.lookup(
            query="In Mumbai?",
            session_state=session_state,
        )

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("multiple official offices in Mumbai", result.answer_text)

    def test_person_designation_lookup_returns_structured_match(self) -> None:
        result = self.query_service.lookup(query="What is the designation of Rajudeen?")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Rajudeen is listed as Deputy General Manager", result.answer_text)

    def test_designation_count_and_total_strength_queries_are_cautious(self) -> None:
        count_result = self.query_service.lookup(query="how may assistant managers are there in SEBI?")
        total_strength_result = self.query_service.lookup(query="what is the total strength of SEBI")

        self.assertEqual(count_result.answer_status, "answered")
        self.assertIn('currently lists 2 entries matching the designation "Assistant Manager"', count_result.answer_text)
        self.assertEqual(total_strength_result.answer_status, "answered")
        self.assertIn("does not reliably establish total institutional strength", total_strength_result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
