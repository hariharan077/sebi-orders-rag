from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.sebi_orders_rag.evaluation.stats import compare_result_frames
from app.sebi_orders_rag.evaluation.triage import build_true_bug_queue
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT


class BugQueueGenerationTests(unittest.TestCase):
    def test_bug_queue_marks_route_only_single_matter_case_as_stale_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-stale",
                query="What did SEBI finally direct in the JP Morgan settlement?",
                route_expected="hierarchical_rag",
                route_actual="exact_lookup",
                answer_status="answered",
                active_record_keys=("external:100486",),
                grounding={"answer_correctness": 1.0, "faithfulness": 1.0, "hallucination_rate": 0.0},
                failure_modes={"primary_bucket": "wrong route", "buckets": ["wrong route"]},
            )
            real_bug = _result_row(
                case_id="case-bug",
                query="What was the settlement amount in the JP Morgan settlement order?",
                route_expected="hierarchical_rag",
                route_actual="clarify",
                answer_status="clarify",
                failure_modes={"primary_bucket": "wrong route", "buckets": ["wrong route", "weak metadata support"]},
                debug={"candidate_list_debug": {"used": True}},
            )
            _write_run_fixture(run_dir, [stale_case, real_bug])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 1)
            self.assertEqual(payload["summary"]["bucket_counts"]["stale expectation"], 1)
            self.assertEqual(payload["summary"]["bucket_counts"]["wrong candidate ranking"], 1)
            cluster = payload["clusters"][0]
            self.assertEqual(cluster["cluster"], "candidate ranking bugs")
            self.assertEqual(cluster["affected_case_ids"], ("case-bug",))

    def test_compare_result_frames_handles_none_route_family_expected(self) -> None:
        base = pd.DataFrame(
            [
                {
                    "route_family_expected": None,
                    "case_id": "base-none",
                    "passed": True,
                    "strict_route_match": True,
                    "equivalent_route_match": True,
                    "hallucination_rate": 0.0,
                },
                {
                    "route_family_expected": "hierarchical_rag",
                    "case_id": "base-rag",
                    "passed": False,
                    "strict_route_match": False,
                    "equivalent_route_match": False,
                    "hallucination_rate": 1.0,
                },
            ]
        )
        head = pd.DataFrame(
            [
                {
                    "route_family_expected": "hierarchical_rag",
                    "case_id": "head-rag",
                    "passed": True,
                    "strict_route_match": True,
                    "equivalent_route_match": True,
                    "hallucination_rate": 0.0,
                }
            ]
        )

        comparison = compare_result_frames(base, head)

        self.assertTrue(pd.isna(comparison.iloc[0]["route_family_expected"]))
        self.assertEqual(comparison.iloc[1]["route_family_expected"], "hierarchical_rag")

    def test_bug_queue_marks_safe_clarify_for_abstain_case_as_stale_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-safe-clarify",
                query="Tell me more about Adani Green Energy Limited.",
                route_expected="abstain",
                route_actual="clarify",
                answer_status="clarify",
                grounding={
                    "answer_correctness": 1.0,
                    "faithfulness": 1.0,
                    "hallucination_rate": 0.0,
                    "abstain_correct": True,
                },
                retrieval={"candidate_list_correctness": 1.0},
                debug={"candidate_list_debug": {"used": True}},
                case_overrides={"must_abstain": True},
                failure_modes={"primary_bucket": "wrong route", "buckets": ["wrong route"]},
            )
            _write_run_fixture(run_dir, [stale_case])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 0)
            self.assertEqual(payload["entries"][0]["primary_bucket"], "stale expectation")

    def test_bug_queue_marks_weak_comparison_support_abstain_as_stale_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-weak-compare",
                query="Compare the JP Morgan Chase Bank N.A. and DDP- Standard Chartered Bank settlement orders.",
                route_expected="hierarchical_rag",
                route_actual="abstain",
                answer_status="abstained",
                retrieval={
                    "mixed_record_contamination": False,
                    "retrieved_record_keys": ["external:100486", "external:100429"],
                },
                case_overrides={
                    "expected_record_keys": ["external:100486", "external:100429"],
                },
                debug={"route_debug": {"comparison_intent": True}},
                execution_overrides={
                    "retrieved_context": [
                        {"record_key": "external:100486", "section_type": "operative_order"},
                        {"record_key": "external:100429", "section_type": "header"},
                    ]
                },
                failure_modes={"primary_bucket": "over-abstain", "buckets": ["over-abstain"]},
            )
            _write_run_fixture(run_dir, [stale_case])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 0)
            self.assertEqual(
                payload["entries"][0]["stale_expectation_annotation_reason"],
                "better grounded answer than prior expectation",
            )

    def test_bug_queue_marks_sessionless_follow_up_clarify_as_stale_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-follow-up-clarify",
                query="What did the appellate authority decide?",
                route_expected="",
                route_actual="clarify",
                answer_status="clarify",
                case_overrides={
                    "issue_class": "regression",
                    "expected_record_keys": ["external:100722"],
                    "metadata": {
                        "expected_titles": ["Appeal No. 6795 of 2026 filed by Rajat Kumar"],
                    },
                },
                failure_modes={"primary_bucket": "over-abstain", "buckets": ["over-abstain"]},
            )
            _write_run_fixture(run_dir, [stale_case])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 0)
            self.assertEqual(payload["entries"][0]["primary_bucket"], "stale expectation")
            self.assertEqual(
                payload["entries"][0]["stale_expectation_annotation_reason"],
                "safer clarify behavior",
            )

    def test_bug_queue_honors_seeded_stale_expectation_bucket_for_clarify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-seeded-stale",
                query="Tell me more about Appeal No. 9999 of 2026 filed by Nonexistent Person.",
                route_expected="",
                route_actual="clarify",
                answer_status="clarify",
                case_overrides={
                    "issue_class": "regression",
                    "expected_failure_buckets": ["stale expectation"],
                },
                failure_modes={"primary_bucket": "wrong route", "buckets": ["wrong route"]},
            )
            _write_run_fixture(run_dir, [stale_case])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 0)
            self.assertEqual(payload["entries"][0]["primary_bucket"], "stale expectation")

    def test_bug_queue_marks_grounded_named_matter_metadata_only_failure_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-stale-metadata",
                query="In the matter of Prime Broking Company Limited",
                route_expected="exact_lookup",
                route_actual="exact_lookup",
                answer_status="answered",
                active_record_keys=("external:30161",),
                grounding={
                    "answer_correctness": 1.0,
                    "faithfulness": 1.0,
                    "hallucination_rate": 0.0,
                },
                case_overrides={
                    "expected_record_keys": ["external:30161"],
                    "must_use_metadata": True,
                    "metadata": {"expected_titles": ["In the matter of Prime Broking Company Limited"]},
                },
                failure_modes={"primary_bucket": "weak metadata support", "buckets": ["weak metadata support"]},
            )
            _write_run_fixture(run_dir, [stale_case])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 0)
            self.assertEqual(payload["entries"][0]["primary_bucket"], "stale expectation")
            self.assertEqual(
                payload["entries"][0]["stale_expectation_annotation_reason"],
                "better grounded answer than prior expectation",
            )

    def test_bug_queue_marks_failure_dump_abstain_note_as_stale_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-failure-dump-abstain",
                query="Tell me more about Cochin Stock Exchange Limited.",
                route_expected="",
                route_actual="abstain",
                answer_status="abstained",
                case_overrides={
                    "notes": "expected route abstain got exact_lookup",
                    "source_case_refs": ["failure_dump:failed_cases.json"],
                    "metadata": {
                        "reference_failure_dump": {
                            "primary_bucket": "wrong route",
                        }
                    },
                },
                failure_modes={"primary_bucket": "over-abstain", "buckets": ["over-abstain"]},
            )
            _write_run_fixture(run_dir, [stale_case])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 0)
            self.assertEqual(payload["entries"][0]["primary_bucket"], "stale expectation")
            self.assertEqual(
                payload["entries"][0]["stale_expectation_annotation_reason"],
                "better grounded answer than prior expectation",
            )

    def test_bug_queue_marks_grounded_negative_answer_for_abstain_case_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "artifacts" / "sebi_eval_runs" / "sebi_eval_run_fixture"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_case = _result_row(
                case_id="case-negative-answer-stale",
                query="What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
                route_expected="abstain",
                route_actual="hierarchical_rag",
                answer_status="answered",
                active_record_keys=("derived:pacheli",),
                grounding={"faithfulness": 1.0, "hallucination_rate": 0.0},
                case_overrides={
                    "issue_class": "abstain",
                    "must_abstain": True,
                },
                execution_overrides={
                    "answer_text": (
                        "The cited order does not describe IPO proceeds. It instead discusses "
                        "a preferential allotment."
                    )
                },
                failure_modes={"primary_bucket": "missing abstain", "buckets": ["missing abstain", "wrong route"]},
            )
            _write_run_fixture(run_dir, [stale_case])

            payload = build_true_bug_queue(
                run_dir=run_dir,
                control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
            )

            self.assertEqual(payload["summary"]["stale_expectation_count"], 1)
            self.assertEqual(payload["summary"]["true_bug_count"], 0)
            self.assertEqual(payload["entries"][0]["primary_bucket"], "stale expectation")
            self.assertEqual(
                payload["entries"][0]["stale_expectation_annotation_reason"],
                "better grounded answer than prior expectation",
            )


def _result_row(
    *,
    case_id: str,
    query: str,
    route_expected: str,
    route_actual: str,
    answer_status: str,
    active_record_keys: tuple[str, ...] = (),
    grounding: dict[str, float] | None = None,
    failure_modes: dict[str, object] | None = None,
    debug: dict[str, object] | None = None,
    retrieval: dict[str, object] | None = None,
    case_overrides: dict[str, object] | None = None,
    execution_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "case": {
            "case_id": case_id,
            "query": query,
            "route_family_expected": route_expected,
            "expected_record_keys": [],
            "gold_numeric_facts": [],
            "tags": [],
            "metadata": {},
            "issue_class": "open_ended",
            "must_abstain": False,
            "must_clarify": False,
            "must_use_metadata": False,
            "must_use_active_matter": False,
            "must_use_structured_current_info": False,
            "must_use_official_web": False,
            **dict(case_overrides or {}),
        },
        "execution": {
            "route_mode": route_actual,
            "answer_status": answer_status,
            "answer_text": "fixture answer",
            "active_record_keys": list(active_record_keys),
            "debug": {
                "route_debug": {
                    "strict_scope_required": True,
                },
                **dict(debug or {}),
            },
            **dict(execution_overrides or {}),
        },
        "route": {
            "strict_route_match": route_expected == route_actual,
            "equivalent_route_match": False,
        },
        "retrieval": {
            "mixed_record_contamination": False,
            **dict(retrieval or {}),
        },
        "grounding": {
            "answer_correctness": 0.0,
            "faithfulness": 0.0,
            "hallucination_rate": 1.0,
            **dict(grounding or {}),
        },
        "numeric": {
            "expected_fact_count": 0,
            "missing_fact_types": [],
            "mismatched_fact_types": [],
        },
        "failure_modes": {
            "primary_bucket": "wrong route",
            "buckets": ["wrong route"],
            "stale_expectation": False,
            **dict(failure_modes or {}),
        },
    }


def _write_run_fixture(run_dir: Path, rows: list[dict[str, object]]) -> None:
    (run_dir / "summary.json").write_text(json.dumps({"failed_cases": len(rows)}), encoding="utf-8")
    serialized = "\n".join(json.dumps(row) for row in rows) + "\n"
    (run_dir / "per_case_results.jsonl").write_text(serialized, encoding="utf-8")
    (run_dir / "failures.jsonl").write_text(serialized, encoding="utf-8")
    (run_dir / "failures.md").write_text("# Failures\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
