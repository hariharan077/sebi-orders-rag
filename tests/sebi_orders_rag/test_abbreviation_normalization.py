from __future__ import annotations

import unittest

from app.sebi_orders_rag.current_info.query_normalization import normalize_current_info_query
from app.sebi_orders_rag.normalization import expand_query


class AbbreviationNormalizationTests(unittest.TestCase):
    def test_structured_designation_abbreviations_expand_by_context(self) -> None:
        intent = normalize_current_info_query("who are the ed")
        self.assertEqual(intent.lookup_type, "ed_list")
        self.assertIn("executive director", intent.normalized_query)
        self.assertEqual(intent.matched_abbreviations, ("ed",))

        rd_intent = normalize_current_info_query("who is rd of nro")
        self.assertEqual(rd_intent.lookup_type, "regional_director")
        self.assertIn("regional director", rd_intent.normalized_query)
        self.assertIn("northern regional office", rd_intent.normalized_query)

        am_intent = normalize_current_info_query("how many am are there in sebi")
        self.assertEqual(am_intent.lookup_type, "designation_count")
        self.assertEqual(am_intent.designation_hint, "Assistant Manager")

    def test_glued_current_info_terms_are_split_before_routing(self) -> None:
        am_intent = normalize_current_info_query("how many assistantmanagers are there in sebi")
        self.assertEqual(am_intent.lookup_type, "designation_count")
        self.assertEqual(am_intent.designation_hint, "Assistant Manager")

        wtm_intent = normalize_current_info_query("who are the wholetimemembers of sebi")
        self.assertEqual(wtm_intent.lookup_type, "wtm_list")

        board_intent = normalize_current_info_query("who are the boardmembers of sebi")
        self.assertEqual(board_intent.lookup_type, "board_members")

    def test_office_abbreviations_expand_without_global_rewrites(self) -> None:
        intent = normalize_current_info_query("location of wro")
        self.assertEqual(intent.lookup_type, "office_contact")
        self.assertIn("western regional office", intent.normalized_query)

        expansion = expand_query("dam capital advisors", contexts=("current_people", "order_lookup"))
        self.assertEqual(expansion.normalized_query, "dam capital advisors")
        self.assertEqual(expansion.matched_abbreviations, ())

    def test_legal_abbreviations_expand_in_order_context_only(self) -> None:
        expansion = expand_query(
            "could you explain pfutp and pit violations",
            contexts=("order_lookup", "order_legal"),
        )
        self.assertTrue(any("pfutp regulations" in value for value in expansion.expansions))
        self.assertTrue(any("pit regulations" in value for value in expansion.expansions))
        self.assertEqual(expansion.matched_abbreviations[:2], ("pfutp", "pit"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
