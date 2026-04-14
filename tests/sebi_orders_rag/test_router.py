from __future__ import annotations

import unittest
from uuid import uuid4

from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionStateRecord
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = AdaptiveQueryRouter()
        self.router_with_pack = AdaptiveQueryRouter(control_pack=load_control_pack(REAL_CONTROL_PACK))

    def test_routes_smalltalk_without_corpus(self) -> None:
        decision = self.router.decide(query="hello")

        self.assertEqual(decision.route_mode, "smalltalk")
        self.assertEqual(decision.query_intent, "smalltalk")

    def test_routes_capability_prompt_to_smalltalk(self) -> None:
        decision = self.router.decide(query="what can you do")

        self.assertEqual(decision.route_mode, "smalltalk")
        self.assertEqual(decision.query_intent, "smalltalk")

    def test_selects_general_knowledge_for_general_explanatory_question(self) -> None:
        decision = self.router.decide(query="What is a settlement order?")

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertEqual(decision.query_intent, "general_knowledge")

    def test_routes_non_corpus_economics_question_to_general_knowledge(self) -> None:
        decision = self.router.decide(query="What is aggregate demand?")

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertEqual(decision.query_intent, "general_knowledge")

    def test_routes_topic_fragment_to_general_knowledge(self) -> None:
        decision = self.router.decide(query="on macroeconomics")

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertEqual(decision.query_intent, "general_knowledge")

    def test_routes_current_official_query_away_from_corpus(self) -> None:
        decision = self.router.decide(query="Who is the chairperson of SEBI?")

        self.assertEqual(decision.route_mode, "structured_current_info")
        self.assertEqual(decision.query_intent, "structured_current_info")

    def test_routes_ministry_query_to_current_official_lookup(self) -> None:
        decision = self.router.decide(query="Does SEBI come under Ministry of Finance?")

        self.assertEqual(decision.route_mode, "current_official_lookup")
        self.assertEqual(decision.query_intent, "current_official_lookup")

    def test_routes_person_and_designation_count_queries_to_current_official_lookup(self) -> None:
        for query in (
            "What is the designation of Rajudeen?",
            "how may assistant managers are there in SEBI?",
        ):
            with self.subTest(query=query):
                decision = self.router.decide(query=query)
                self.assertEqual(decision.route_mode, "structured_current_info")
                self.assertEqual(decision.query_intent, "structured_current_info")

    def test_routes_sebi_definition_query_to_general_knowledge(self) -> None:
        decision = self.router.decide(query="Who are SEBI")

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertEqual(decision.query_intent, "general_knowledge")

    def test_selects_exact_lookup_for_document_identity_query(self) -> None:
        decision = self.router.decide(
            query="Appeal No. 6798 of 2026 filed by Hariom Yadav"
        )

        self.assertEqual(decision.route_mode, "exact_lookup")
        self.assertEqual(decision.query_intent, "document_lookup")

    def test_selects_memory_scoped_rag_for_follow_up_with_session_scope(self) -> None:
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_document_ids=(123,),
            active_record_keys=("external:100725",),
        )

        decision = self.router.decide(
            query="What did SEBI finally direct?",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "memory_scoped_rag")
        self.assertEqual(decision.query_intent, "follow_up")

    def test_named_single_matter_lookup_wins_over_existing_session_scope(self) -> None:
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_document_ids=(123,),
            active_record_keys=("external:100663",),
        )

        decision = self.router_with_pack.decide(
            query="What was the Paresh Nathanlal case",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "exact_lookup")
        self.assertEqual(decision.query_intent, "matter_specific")
        self.assertTrue(decision.analysis.strict_single_matter)
        self.assertEqual(decision.analysis.strict_lock_record_keys, ("external:98776",))

    def test_named_substantive_settlement_query_locks_to_jp_morgan_record(self) -> None:
        decision = self.router_with_pack.decide(
            query="What did SEBI finally direct in the JP Morgan settlement?"
        )

        self.assertEqual(decision.route_mode, "exact_lookup")
        self.assertTrue(decision.analysis.strict_single_matter)
        self.assertEqual(decision.analysis.strict_lock_record_keys, ("external:100486",))

    def test_routes_office_follow_up_with_current_lookup_context(self) -> None:
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            current_lookup_family="office_contact",
            current_lookup_query="where is sebi office in chennai",
        )

        decision = self.router.decide(
            query="In mumbai?",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "structured_current_info")
        self.assertEqual(decision.query_intent, "structured_current_info")

    def test_selects_hierarchical_rag_when_query_is_matter_specific_but_not_exact_lookup(self) -> None:
        decision = self.router.decide(
            query="What penalty was imposed in the 2026 order?"
        )

        self.assertEqual(decision.route_mode, "hierarchical_rag")
        self.assertEqual(decision.query_intent, "substantive_outcome")

    def test_selects_hierarchical_rag_for_free_form_settlement_query(self) -> None:
        decision = self.router.decide(
            query="What was the settlement amount in the JP Morgan settlement order?"
        )

        self.assertEqual(decision.route_mode, "hierarchical_rag")
        self.assertEqual(decision.query_intent, "substantive_outcome")
        self.assertIn("settlement_specific", decision.reason_codes)

    def test_keeps_generic_settlement_explanation_on_direct_llm(self) -> None:
        decision = self.router.decide(
            query="Explain settlement proceedings"
        )

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertEqual(decision.query_intent, "general_knowledge")

    def test_general_legal_definition_ignores_active_matter_scope(self) -> None:
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_document_ids=(123,),
            active_record_keys=("external:100663",),
        )

        decision = self.router_with_pack.decide(
            query="What is an exemption order under the takeover regulations?",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertEqual(decision.query_intent, "general_knowledge")

    def test_regulation_30a_query_stays_on_general_knowledge(self) -> None:
        decision = self.router_with_pack.decide(
            query="Explain orders issued under Regulation 30A."
        )

        self.assertEqual(decision.route_mode, "general_knowledge")
        self.assertEqual(decision.query_intent, "legal_explanation")

    def test_neelgiri_comparison_query_builds_expected_special_court_candidates(self) -> None:
        decision = self.router_with_pack.decide(
            query="Compare the Neelgiri Forest Ltd judgment and sentencing order."
        )

        self.assertEqual(decision.route_mode, "hierarchical_rag")
        self.assertTrue(decision.analysis.comparison_intent)
        candidate_record_keys = {
            candidate.record_key for candidate in decision.analysis.strict_matter_lock.candidates
        }
        self.assertIn("external:87947", candidate_record_keys)
        self.assertIn("external:87948", candidate_record_keys)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
