from __future__ import annotations

import unittest

from app.sebi_orders_rag.evaluation.dataset import case_from_dict, validate_dataset


class EvalDatasetSchemaTests(unittest.TestCase):
    def test_case_from_dict_parses_jsonl_friendly_schema(self) -> None:
        case = case_from_dict(
            {
                "case_id": "case-1",
                "query": "What was the listing price of DU Digital?",
                "route_family_expected": "hierarchical_rag",
                "allowed_routes": ["hierarchical_rag", "memory_scoped_rag"],
                "expected_record_keys": ["external:98774"],
                "expected_bucket_names": ["orders-of-ed-cgm"],
                "gold_numeric_facts": [
                    {
                        "fact_type": "listing_price",
                        "value_numeric": 12.0,
                        "value_text": "Rs.12/share",
                    }
                ],
                "must_use_metadata": True,
                "tags": ["numeric", "du_digital"],
                "issue_class": "gold_fact",
            }
        )

        self.assertEqual(case.case_id, "case-1")
        self.assertEqual(case.expected_record_keys, ("external:98774",))
        self.assertEqual(case.gold_numeric_facts[0].fact_type, "listing_price")
        self.assertEqual(validate_dataset((case,)), [])

    def test_validate_dataset_rejects_duplicate_ids_and_conflicting_flags(self) -> None:
        first = case_from_dict(
            {
                "case_id": "case-dup",
                "query": "One query",
            }
        )
        second = case_from_dict(
            {
                "case_id": "case-dup",
                "query": "Second query",
                "must_abstain": True,
                "must_clarify": True,
            }
        )

        errors = validate_dataset((first, second))

        self.assertIn("duplicate case_id: case-dup", errors)
        self.assertTrue(any("cannot require both abstain and clarify" in item for item in errors))

    def test_case_from_dict_splits_semicolon_joined_expected_record_keys(self) -> None:
        case = case_from_dict(
            {
                "case_id": "case-multi",
                "query": "Compare two matters",
                "expected_record_keys": ["external:100486;external:100429"],
            }
        )

        self.assertEqual(case.expected_record_keys, ("external:100486", "external:100429"))

    def test_case_from_dict_strips_scaffold_gold_answer_short(self) -> None:
        case = case_from_dict(
            {
                "case_id": "case-scaffold",
                "query": "Settlement Order in the matter of JP Morgan Chase Bank N.A.",
                "gold_answer_short": "Session-seeding exact lookup.",
                "notes": "Session-seeding exact lookup.",
            }
        )

        self.assertIsNone(case.gold_answer_short)
        self.assertEqual(case.notes, "Session-seeding exact lookup.")

    def test_case_from_dict_preserves_regression_answer_guidance(self) -> None:
        guidance = (
            "This is a settlement order, not an exemption order; answer should abstain "
            "or explicitly say no exemption order is in scope."
        )
        case = case_from_dict(
            {
                "case_id": "case-regression",
                "query": "What exemption did SEBI grant in the JP Morgan Chase Bank N.A. matter?",
                "issue_class": "regression",
                "gold_answer_short": guidance,
            }
        )

        self.assertEqual(case.gold_answer_short, guidance)
        self.assertEqual(case.metadata["answer_guidance"], guidance)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
