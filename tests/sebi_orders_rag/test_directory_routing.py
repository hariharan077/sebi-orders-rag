from __future__ import annotations

import unittest
from uuid import uuid4

from app.sebi_orders_rag.directory_data.models import (
    BoardMemberRecord,
    DirectoryOfficeRecord,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
    OrgStructureRecord,
)
from app.sebi_orders_rag.directory_data.service import DirectoryReferenceQueryService
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionStateRecord


class DirectoryRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = AdaptiveQueryRouter()
        self.dataset = DirectoryReferenceDataset(
            people=(
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Tuhin Kanta Pandey",
                    designation="Chairman",
                    role_group="chairperson",
                    email="chairman@sebi.gov.in",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="orgchart",
                    source_url="https://www.sebi.gov.in/orgchart-grid.html",
                    canonical_name="Amarjeet Singh",
                    designation="Whole Time Member",
                    role_group="wtm",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="orgchart",
                    source_url="https://www.sebi.gov.in/orgchart-grid.html",
                    canonical_name="Kamlesh Chandra Varshney",
                    designation="Whole Time Member",
                    role_group="wtm",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="orgchart",
                    source_url="https://www.sebi.gov.in/orgchart-grid.html",
                    canonical_name="Manoj Kumar",
                    designation="Executive Director",
                    role_group="executive_director",
                    department_name="Investment Management Department (IMD)",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="regional_offices",
                    source_url="https://www.sebi.gov.in/department/regional-offices-43/contact.html",
                    canonical_name="Amit Pradhan",
                    designation="ED, Regional Director",
                    role_group="regional_director",
                    office_name="Northern Regional Office (NRO), New Delhi",
                    email="amitp@sebi.gov.in",
                    phone="011-69012960",
                ).with_hash(),
                DirectoryPersonRecord(
                    source_type="directory",
                    source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                    canonical_name="Rajudeen",
                    designation="Assistant General Manager",
                    role_group="staff",
                    phone="022-40450088",
                ).with_hash(),
            ),
            board_members=(
                BoardMemberRecord(
                    source_url="https://www.sebi.gov.in/sebiweb/boardmember/BoardMemberAction.do?doBoardMember=yes&lang=en",
                    canonical_name="Tuhin Kanta Pandey",
                    board_role="Chairman, SEBI",
                    category="chairperson",
                ).with_hash(),
                BoardMemberRecord(
                    source_url="https://www.sebi.gov.in/sebiweb/boardmember/BoardMemberAction.do?doBoardMember=yes&lang=en",
                    canonical_name="Shri Amarjeet Singh",
                    board_role="Whole-Time Member, SEBI",
                    category="whole_time_member",
                ).with_hash(),
                BoardMemberRecord(
                    source_url="https://www.sebi.gov.in/sebiweb/boardmember/BoardMemberAction.do?doBoardMember=yes&lang=en",
                    canonical_name="Shri Kamlesh Chandra Varshney",
                    board_role="Whole-Time Member, SEBI",
                    category="whole_time_member",
                ).with_hash(),
            ),
            offices=(
                DirectoryOfficeRecord(
                    source_type="contact_us",
                    source_url="https://www.sebi.gov.in/contact-us.html",
                    office_name="Southern Regional Office (SRO)",
                    office_type="regional_office",
                    region="south",
                    address="7th Floor, 756-L, Anna Salai, Chennai - 600002, Tamil Nadu",
                    phone="+91-44-28880222 / 28526686",
                    fax="+91-44-28880333",
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
                ).with_hash(),
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
                    office_name="Eastern Regional Office (ERO)",
                    office_type="regional_office",
                    region="east",
                    address="L&T Chambers, 16 Camac Street, Kolkata - 700017, West Bengal",
                    email="sebiero@sebi.gov.in",
                    city="Kolkata",
                    state="West Bengal",
                ).with_hash(),
            ),
            org_structure=(
                OrgStructureRecord(
                    source_type="orgchart",
                    source_url="https://www.sebi.gov.in/orgchart-grid.html",
                    leader_name="Amarjeet Singh",
                    leader_role="Whole Time Member",
                    department_name="Investment Management Department (IMD)",
                    executive_director_name="Manoj Kumar",
                ).with_hash(),
            ),
        )
        self.query_service = DirectoryReferenceQueryService(
            dataset_loader=lambda: self.dataset,
            provider_name="structured_directory",
        )

    def test_router_sends_structured_reference_queries_to_structured_current_info(self) -> None:
        queries = (
            "who is the chairman of sebi?",
            "who are the current board members of sebi?",
            "how many wtm are there?",
            "Who are the WTMs of SEBI?",
            "Who are the EDs of SEBI?",
            "what is the organisation structure of sebi",
            "what is the address of SEBI Chennai office?",
            "SEBI office address Delhi",
            "where is SEBI located in Kolkata",
            "who is the Regional Director of the Northern Regional Office?",
            "What is the designation of Rajudeen?",
            "how may assistant managers are there in SEBI?",
            "is there an assistant manager called Bhuvanesh? when did he join whats his number?",
        )

        for query in queries:
            with self.subTest(query=query):
                decision = self.router.decide(query=query)
                self.assertEqual(decision.route_mode, "structured_current_info")

    def test_query_service_answers_office_address_from_structured_rows(self) -> None:
        result = self.query_service.lookup(query="what is the address of SEBI Chennai office?")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Chennai - 600002", result.answer_text)
        self.assertEqual(result.sources[0].record_key, "official:contact_us")

    def test_query_service_answers_regional_director_from_structured_rows(self) -> None:
        result = self.query_service.lookup(
            query="who is the Regional Director of the Northern Regional Office?"
        )

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Amit Pradhan", result.answer_text)
        self.assertIn("amitp@sebi.gov.in", result.answer_text)

    def test_query_service_answers_board_members_from_board_source(self) -> None:
        result = self.query_service.lookup(query="who are the current board members of sebi?")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("SEBI currently has 3 board members.", result.answer_text)
        self.assertEqual(result.sources[0].record_key, "official:board_members")

    def test_query_service_returns_insufficient_context_for_unknown_match(self) -> None:
        result = self.query_service.lookup(query="who is the regional director of the Pune office?")

        self.assertEqual(result.answer_status, "insufficient_context")

    def test_query_service_answers_wtm_count_with_names(self) -> None:
        result = self.query_service.lookup(
            query="how many whole time members are serving in sebi currently and who are they?"
        )

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("SEBI currently has 2 Whole-Time Members", result.answer_text)
        self.assertIn("Amarjeet Singh", result.answer_text)
        self.assertIn("Kamlesh Chandra Varshney", result.answer_text)

    def test_office_follow_up_memory_routes_and_answers(self) -> None:
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            current_lookup_family="office_contact",
            current_lookup_query="where is sebi office in chennai",
        )

        decision = self.router.decide(query="In mumbai?", session_state=state)
        result = self.query_service.lookup(query="In mumbai?", session_state=state)

        self.assertEqual(decision.route_mode, "structured_current_info")
        self.assertEqual(result.answer_status, "answered")
        self.assertIn("multiple official offices in Mumbai", result.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
