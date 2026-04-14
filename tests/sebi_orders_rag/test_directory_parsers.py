from __future__ import annotations

import unittest

from app.sebi_orders_rag.directory_data.parser_directory import parse_directory_page
from app.sebi_orders_rag.directory_data.parser_offices import (
    parse_contact_us_page,
    parse_regional_offices_page,
)
from app.sebi_orders_rag.directory_data.parser_orgchart import parse_orgchart_page


class DirectoryParserTests(unittest.TestCase):
    def test_directory_parser_extracts_chairperson_and_wtms(self) -> None:
        parsed = parse_directory_page(
            """
            <html><body>
                <div class="portlet1 box1 green">
                    <div class="portlet-title"><h2>HEAD OFFICE, MUMBAI</h2></div>
                    <table class="table1">
                        <thead>
                            <tr><th colspan="5"><h3>Chairman</h3></th></tr>
                            <tr><th>Staff No</th><th>Name</th><th>Date of Joining</th><th>Email ID</th><th>Telephone No</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>2801</td><td>TUHIN KANTA PANDEY</td><td>Mar 01, 2025</td><td>chairman [at] sebi [dot] gov [dot] in</td><td>022-40459999</td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead>
                            <tr><th colspan="5"><h3>Whole Time Member</h3></th></tr>
                            <tr><th>Staff No</th><th>Name of the Staff Member</th><th>Date of Joining</th><th>Email ID</th><th>Telephone No</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>2704</td><td>AMARJEET SINGH</td><td>Sep 01, 2023</td><td>amarjeets [at] sebi [dot] gov [dot] in</td><td>022-40459991</td></tr>
                            <tr><td>2706</td><td>KAMLESH CHANDRA VARSHNEY</td><td>Sep 20, 2023</td><td>kamlesh.varshney [at] sebi [dot] gov [dot] in</td><td>022-40459989</td></tr>
                        </tbody>
                    </table>
                </div>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
        )

        chairpeople = [record for record in parsed.people if record.role_group == "chairperson"]
        wtms = [record for record in parsed.people if record.role_group == "wtm"]

        self.assertEqual(len(chairpeople), 1)
        self.assertEqual(chairpeople[0].canonical_name, "Tuhin Kanta Pandey")
        self.assertEqual(chairpeople[0].email, "chairman@sebi.gov.in")
        self.assertEqual(len(wtms), 2)
        self.assertEqual(wtms[0].office_name, "HEAD OFFICE, MUMBAI")

    def test_directory_parser_extracts_title_only_head_office_tables(self) -> None:
        parsed = parse_directory_page(
            """
            <html><body>
                <div class="portlet1 box1 green">
                    <div class="portlet-title"><h2>HEAD OFFICE, MUMBAI</h2></div>
                    <table class="table1">
                        <thead><tr><th colspan="5"><h3>Executive Director</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1238</td><td>G BABITA RAYUDU</td><td>Jun 03, 1996</td><td>babitar [at] sebi [dot] gov [dot] in</td><td>022-40459332/022-26449332</td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead><tr><th colspan="6"><h3>Chief General Manager</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1306</td><td>SANTOSH KUMAR SHUKLA</td><td>Sep 10, 1996</td><td>QUASI-JUDICIAL CELL-1</td><td>santoshs [at] sebi [dot] gov [dot] in</td><td>022-40459833/022-26449833</td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead><tr><th colspan="6"><h3>General Manager</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1360</td><td>ACHAL SINGH</td><td>Nov 19, 1997</td><td></td><td></td><td></td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead><tr><th colspan="6"><h3>Deputy General Manager</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1368</td><td>AVARJEET SINGH</td><td>Dec 01, 1997</td><td></td><td></td><td></td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead><tr><th colspan="6"><h3>Assistant General Manager</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1615</td><td>P BHAGAVATHI RAJA</td><td>Jan 29, 2004</td><td>TAD</td><td>braja [at] sebi [dot] gov [dot] in</td><td>022-40459529/022-26449529</td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead><tr><th colspan="6"><h3>Manager</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1979</td><td>RAHUL POSWAL</td><td>Jun 15, 2011</td><td>CFID</td><td>rahulp [at] sebi [dot] gov [dot] in</td><td>022-40459117/022-26449117</td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead><tr><th colspan="6"><h3>Assistant Manager</h3></th></tr></thead>
                        <tbody>
                            <tr><td>2614</td><td>YASHWANT MANTHANI</td><td>Aug 01, 2022</td><td>CFD</td><td>yashwantm [at] sebi [dot] gov [dot] in</td><td>022-20752151</td></tr>
                        </tbody>
                    </table>
                </div>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
        )

        people_by_name = {record.canonical_name: record for record in parsed.people}

        self.assertEqual(len(parsed.people), 7)
        self.assertEqual(people_by_name["G Babita Rayudu"].designation, "Executive Director")
        self.assertEqual(people_by_name["G Babita Rayudu"].role_group, "executive_director")
        self.assertEqual(people_by_name["Santosh Kumar Shukla"].designation, "Chief General Manager")
        self.assertEqual(people_by_name["Santosh Kumar Shukla"].department_name, "QUASI-JUDICIAL CELL-1")
        self.assertEqual(people_by_name["Achal Singh"].designation, "General Manager")
        self.assertIsNone(people_by_name["Achal Singh"].department_name)
        self.assertEqual(people_by_name["Avarjeet Singh"].designation, "Deputy General Manager")
        self.assertEqual(people_by_name["P Bhagavathi Raja"].designation, "Assistant General Manager")
        self.assertEqual(people_by_name["Rahul Poswal"].designation, "Manager")
        self.assertEqual(people_by_name["Yashwant Manthani"].designation, "Assistant Manager")

    def test_directory_parser_extracts_indore_local_office_title_only_table(self) -> None:
        parsed = parse_directory_page(
            """
            <html><body>
                <div class="portlet1 box1 green">
                    <div class="portlet-title"><h2>West Zone</h2></div>
                    <table class="table1">
                        <thead><tr><th colspan="7"><h3>Indore Local Office</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1842</td><td>SHAILESH DASHARATH PINGALE</td><td>Feb 04, 2008</td><td>Deputy General Manager</td><td>LO-INDORE</td><td>shaileshp [at] sebi [dot] gov [dot] in</td><td>022-40452264/022-26442264</td></tr>
                            <tr><td>2376</td><td>VISHAL ASHOKRAO GAWANDE</td><td>May 28, 2019</td><td>Manager</td><td>LO-INDORE</td><td>vishala [at] sebi [dot] gov [dot] in</td><td>079-27467502</td></tr>
                        </tbody>
                    </table>
                </div>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
        )

        people_by_name = {record.canonical_name: record for record in parsed.people}

        self.assertEqual(len(parsed.people), 2)
        self.assertEqual(people_by_name["Shailesh Dasharath Pingale"].designation, "Deputy General Manager")
        self.assertEqual(people_by_name["Shailesh Dasharath Pingale"].department_name, "LO-INDORE")
        self.assertEqual(people_by_name["Vishal Ashokrao Gawande"].designation, "Manager")
        self.assertEqual(people_by_name["Vishal Ashokrao Gawande"].phone, "079-27467502")

    def test_directory_parser_extracts_offices_from_tel_fax_block(self) -> None:
        parsed = parse_directory_page(
            """
            <html><body>
                <table class="table1 tel_fax_main">
                    <tbody>
                        <tr>
                            <td>
                                <table class="table1 tel_fax">
                                    <tr><td>SEBI Bhavan BKC: +91-22-26449000 / 40459000</td></tr>
                                    <tr><td>Northern Regional Office (NRO): +91-011-69012998</td></tr>
                                    <tr><td>Indore Local Office: +91-0731-2557002</td></tr>
                                </table>
                            </td>
                            <td>
                                <table class="table1 tel_fax">
                                    <tr><td>SEBI Bhavan BKC: +91-22-40459019 / +91-22-40459021</td></tr>
                                    <tr><td>Northern Regional Office (NRO): +91-011-69012998</td></tr>
                                </table>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
        )

        offices_by_name = {record.office_name: record for record in parsed.offices}

        self.assertEqual(len(parsed.offices), 3)
        self.assertEqual(offices_by_name["SEBI Bhavan BKC"].phone, "+91-22-26449000/40459000")
        self.assertEqual(offices_by_name["SEBI Bhavan BKC"].fax, "+91-22-40459019/+91-22-40459021")
        self.assertEqual(offices_by_name["SEBI Bhavan BKC"].office_type, "head_office")
        self.assertEqual(offices_by_name["Northern Regional Office (NRO)"].region, "north")
        self.assertEqual(offices_by_name["Northern Regional Office (NRO)"].fax, "+91-011-69012998")
        self.assertEqual(offices_by_name["Indore Local Office"].city, "Indore")
        self.assertIsNone(offices_by_name["Indore Local Office"].fax)

    def test_directory_parser_handles_mixed_table_shapes_in_one_page(self) -> None:
        parsed = parse_directory_page(
            """
            <html><body>
                <table class="table1 tel_fax_main">
                    <tbody>
                        <tr>
                            <td>
                                <table class="table1 tel_fax">
                                    <tr><td>SEBI Bhavan BKC: +91-22-26449000 / 40459000</td></tr>
                                    <tr><td>Indore Local Office: +91-0731-2557002</td></tr>
                                </table>
                            </td>
                            <td>
                                <table class="table1 tel_fax">
                                    <tr><td>SEBI Bhavan BKC: +91-22-40459019 / +91-22-40459021</td></tr>
                                </table>
                            </td>
                        </tr>
                    </tbody>
                </table>
                <div class="portlet1 box1 green">
                    <div class="portlet-title"><h2>HEAD OFFICE, MUMBAI</h2></div>
                    <table class="table1">
                        <thead>
                            <tr><th colspan="5"><h3>Chairman</h3></th></tr>
                            <tr><th>Staff No</th><th>Name</th><th>Date of Joining</th><th>Email ID</th><th>Telephone No</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>2801</td><td>TUHIN KANTA PANDEY</td><td>Mar 01, 2025</td><td>chairman [at] sebi [dot] gov [dot] in</td><td>022-40459999</td></tr>
                        </tbody>
                    </table>
                    <table class="table1">
                        <thead><tr><th colspan="6"><h3>Manager</h3></th></tr></thead>
                        <tbody>
                            <tr><td>1979</td><td>RAHUL POSWAL</td><td>Jun 15, 2011</td><td>CFID</td><td>rahulp [at] sebi [dot] gov [dot] in</td><td>022-40459117/022-26449117</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="portlet1 box1 green">
                    <div class="portlet-title"><h2>West Zone</h2></div>
                    <table class="table1">
                        <thead><tr><th colspan="7"><h3>Indore Local Office</h3></th></tr></thead>
                        <tbody>
                            <tr><td>2376</td><td>VISHAL ASHOKRAO GAWANDE</td><td>May 28, 2019</td><td>Manager</td><td>LO-INDORE</td><td>vishala [at] sebi [dot] gov [dot] in</td><td>079-27467502</td></tr>
                        </tbody>
                    </table>
                </div>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
        )

        people_by_name = {record.canonical_name: record for record in parsed.people}
        offices_by_name = {record.office_name: record for record in parsed.offices}

        self.assertEqual(len(parsed.people), 3)
        self.assertEqual(people_by_name["Tuhin Kanta Pandey"].role_group, "chairperson")
        self.assertEqual(people_by_name["Rahul Poswal"].designation, "Manager")
        self.assertEqual(people_by_name["Vishal Ashokrao Gawande"].designation, "Manager")
        self.assertEqual(len(parsed.offices), 2)
        self.assertEqual(offices_by_name["SEBI Bhavan BKC"].fax, "+91-22-40459019/+91-22-40459021")
        self.assertEqual(offices_by_name["Indore Local Office"].phone, "+91-0731-2557002")

    def test_orgchart_parser_extracts_leaders_and_department_mappings(self) -> None:
        parsed = parse_orgchart_page(
            """
            <html><body>
                <div class="orgchart">
                    <div class="tree-m-info">
                        <h3>Shri. Tuhin Kanta Pandey</h3>
                        <h4>Chairman</h4>
                        <h5>chairman@sebi.gov.in</h5>
                    </div>
                    <div class="tree-m-info">
                        <h3>Shri Amarjeet Singh</h3>
                        <h4>Whole-Time Member</h4>
                        <h5>amarjeets@sebi.gov.in</h5>
                    </div>
                </div>
                <div class="details_1">
                    <h2>Shri Amarjeet Singh, Whole Time Member</h2>
                    <table>
                        <thead>
                            <tr><th>Department / Division</th><th>Executive Director</th><th>Phone No. (+91) 022 / E-mail ID</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Investment Management Department (IMD)</td>
                                <td>Shri Manoj Kumar</td>
                                <td>40459260/26449260 manojk@sebi.gov.in</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/orgchart-grid.html",
        )

        leaders = [record for record in parsed.people if record.role_group in {"chairperson", "wtm"}]
        eds = [record for record in parsed.people if record.role_group == "executive_director"]

        self.assertEqual(len(leaders), 2)
        self.assertEqual(len(parsed.org_structure), 1)
        self.assertEqual(parsed.org_structure[0].department_name, "Investment Management Department (IMD)")
        self.assertEqual(eds[0].canonical_name, "Shri Manoj Kumar")
        self.assertEqual(eds[0].email, "manojk@sebi.gov.in")

    def test_regional_offices_parser_extracts_regional_director(self) -> None:
        parsed = parse_regional_offices_page(
            """
            <html><body>
                <table class="table">
                    <tr>
                        <td>Southern Regional Office (SRO), Chennai</td>
                        <td>Shri Suraj Mohan M, GM, Regional Director</td>
                        <td>044-28880141</td>
                        <td>surajmohanm@sebi.gov.in</td>
                    </tr>
                    <tr>
                        <td>Shri Salmanu KK, DGM</td>
                        <td>044-28884140</td>
                        <td>salmanuk@sebi.gov.in</td>
                    </tr>
                </table>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/department/regional-offices-43/contact.html",
        )

        directors = [record for record in parsed.people if record.role_group == "regional_director"]

        self.assertEqual(len(parsed.offices), 1)
        self.assertEqual(directors[0].canonical_name, "Shri Suraj Mohan M")
        self.assertIn("Regional Director", directors[0].designation or "")

    def test_contact_us_parser_extracts_office_address_phone_fax_and_email(self) -> None:
        parsed = parse_contact_us_page(
            """
            <html><body><script>
                var locations = [
                    ['<div><span><h2><span>Southern Regional Office (SRO)</span></h2><span style="font-weight:bold;">Address : </span><dl><dt>7th Floor, 756-L,</dt><dt>Anna Salai,</dt><dt>Chennai - 600002, Tamil Nadu</dt><dt><b>Tel:</b> +91-44-28880222 / 28526686</dt><dt><b>Fax:</b> +91-44-28880333</dt><dt><b>Email:</b> sebisro@sebi.gov.in</dt></dl></span></div>', 13.0689, 80.2671, "Southern Regional Office (SRO)"]
                ];
                // Setup the different icons
            </script></body></html>
            """,
            source_url="https://www.sebi.gov.in/contact-us.html",
        )

        self.assertEqual(len(parsed.offices), 1)
        office = parsed.offices[0]
        self.assertEqual(office.office_name, "Southern Regional Office (SRO)")
        self.assertIn("Chennai - 600002", office.address or "")
        self.assertEqual(office.email, "sebisro@sebi.gov.in")
        self.assertEqual(office.city, "Chennai")
        self.assertEqual(office.state, "Tamil Nadu")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
