from __future__ import annotations

import unittest

from app.sebi_orders_rag.directory_data.parser_board import parse_board_page


class BoardParserTests(unittest.TestCase):
    def test_board_parser_extracts_chairperson_and_member_categories(self) -> None:
        parsed = parse_board_page(
            """
            <html><body>
                <div class="member-first">
                    <div class="member-info member-right">
                        <h3 id="nameId80">Tuhin Kanta Pandey</h3>
                        <h4 id="typeId80">Chairman, SEBI</h4>
                    </div>
                </div>
                <div class="member-list">
                    <ul>
                        <h2>Whole-Time Members</h2>
                        <li>
                            <h3 id="nameId73">Shri Amarjeet Singh</h3>
                            <h4 id="typeId73">Whole-Time Member, SEBI</h4>
                        </li>
                    </ul>
                </div>
                <div class="member-list">
                    <ul>
                        <h2>Part-Time Members</h2>
                        <li>
                            <h3>Ms. Deepti Gaur Mukerjee</h3>
                            <h4>Part-Time Member, SEBI</h4>
                            <h5>Secretary, Ministry of Corporate Affairs, Government of India</h5>
                        </li>
                        <li>
                            <h3>Shri Shirish Chandra Murmu</h3>
                            <h4>Part-Time Member, SEBI</h4>
                            <h5>Deputy Governor, Reserve Bank of India</h5>
                        </li>
                    </ul>
                </div>
            </body></html>
            """,
            source_url="https://www.sebi.gov.in/sebiweb/boardmember/BoardMemberAction.do?doBoardMember=yes&lang=en",
        )

        self.assertEqual(len(parsed.board_members), 4)
        self.assertEqual(parsed.board_members[0].category, "chairperson")
        self.assertEqual(parsed.board_members[1].category, "whole_time_member")
        self.assertEqual(parsed.board_members[2].category, "government_nominee")
        self.assertEqual(parsed.board_members[3].category, "rbi_nominee")
        self.assertIn("Government of India", parsed.board_members[2].board_role)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
