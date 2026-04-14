from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.sebi_orders_rag.evaluation.benchmark import compare_run_summaries
from app.sebi_orders_rag.evaluation.case_loader import build_dataset
from app.sebi_orders_rag.evaluation.dataset import merge_datasets
from app.sebi_orders_rag.evaluation.redteam import build_redteam_cases
from app.sebi_orders_rag.evaluation.runner import EvaluationRunner, ReplayExecutor, build_run_metadata
from app.sebi_orders_rag.evaluation.schemas import AssistantExecution, EvaluationCase, EvaluationDataset
from app.sebi_orders_rag.evaluation.stats import compare_result_frames, results_to_frame, summarize_case_results
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT, EVAL_FAILURE_DUMP_FIXTURE_ROOT


class EvalStatsTests(unittest.TestCase):
    def test_summary_and_compare_frames(self) -> None:
        dataset = _mini_dataset()
        runner = EvaluationRunner(executor=ReplayExecutor())
        metadata = build_run_metadata(
            dataset=dataset,
            executor_mode="replay",
            output_root=Path(tempfile.mkdtemp()),
        )
        run_result = runner.run(dataset=dataset, metadata=metadata)
        frame = results_to_frame(run_result.case_results)
        summary = summarize_case_results(run_result.case_results)
        comparison = compare_result_frames(frame, frame)

        self.assertEqual(int(summary["total_cases"]), len(dataset.cases))
        self.assertFalse(frame.empty)
        self.assertFalse(comparison.empty)

    def test_benchmark_artifacts_and_compare_runs(self) -> None:
        dataset = _mini_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            runner = EvaluationRunner(executor=ReplayExecutor())

            first_metadata = build_run_metadata(
                dataset=dataset,
                executor_mode="replay",
                output_root=output_root,
            )
            first_run = runner.run(
                dataset=dataset,
                metadata=first_metadata,
                output_root=output_root,
            )
            second_dataset = merge_datasets(
                name=dataset.name,
                cases=(*dataset.cases, *build_redteam_cases(dataset.cases)),
                source_files=dataset.source_files,
            )
            second_metadata = build_run_metadata(
                dataset=second_dataset,
                executor_mode="replay",
                output_root=output_root,
            )
            second_run = runner.run(
                dataset=second_dataset,
                metadata=second_metadata,
                output_root=output_root,
            )

            first_dir = output_root / first_metadata.run_id
            second_dir = output_root / second_metadata.run_id
            for required in (
                "run_config.json",
                "summary.json",
                "per_case_results.jsonl",
                "failures.jsonl",
                "failures.md",
                "metrics_by_tag.csv",
                "metrics_by_route.csv",
                "confusion_matrix.csv",
                "redteam_summary.json",
            ):
                self.assertTrue((second_dir / required).exists(), required)
            comparison = compare_run_summaries(base_run_dir=first_dir, head_run_dir=second_dir)
            self.assertIn("metric_deltas", comparison)
            self.assertEqual(
                json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))["total_cases"],
                len(first_run.case_results),
            )
            self.assertEqual(
                json.loads((second_dir / "summary.json").read_text(encoding="utf-8"))["total_cases"],
                len(second_run.case_results),
            )

    def test_approved_safer_clarify_expectation_counts_as_pass(self) -> None:
        case = EvaluationCase(
            case_id="case-safe-clarify",
            query="What did SEBI finally direct?",
            issue_class="regression",
            expected_record_keys=("external:100669",),
            metadata={
                "expectation_update": {
                    "approved": True,
                    "reason": "safer clarify behavior",
                    "comment": "approved from burn-down",
                }
            },
        )
        dataset = EvaluationDataset(
            name="expectation-updates",
            version="v1",
            cases=(case,),
            source_files=(),
        )
        execution = AssistantExecution(
            case_id=case.case_id,
            query=case.query,
            route_mode="clarify",
            query_intent="follow_up",
            answer_status="clarify",
            answer_text="Please clarify which SEBI matter you mean.",
            confidence=0.2,
            session_id=None,
            debug={"planner_debug": {"used": True, "execution_route_mode": "clarify"}},
            prompt_family="clarify",
            run_metadata={},
        )
        runner = EvaluationRunner(executor=_SingleExecutionExecutor(execution))
        metadata = build_run_metadata(
            dataset=dataset,
            executor_mode="stub",
            output_root=Path(tempfile.mkdtemp()),
        )

        run_result = runner.run(dataset=dataset, metadata=metadata)

        case_result = run_result.case_results[0]
        self.assertTrue(case_result.passed)
        self.assertEqual(case_result.failure_modes.primary_bucket, "stale expectation")
        self.assertEqual(run_result.summary["wrong_answer_regression_pass_count"], 1)
        self.assertEqual(run_result.summary["failed_cases"], 0)

    def test_approved_grounded_answer_expectation_counts_as_pass(self) -> None:
        case = EvaluationCase(
            case_id="case-grounded-answer",
            query="What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
            route_family_expected="abstain",
            allowed_routes=("abstain", "hierarchical_rag"),
            expected_record_keys=("derived:69093",),
            must_abstain=True,
            issue_class="regression",
            metadata={
                "expectation_update": {
                    "approved": True,
                    "reason": "better grounded answer than prior expectation",
                    "comment": "approved from burn-down",
                }
            },
        )
        dataset = EvaluationDataset(
            name="expectation-updates",
            version="v1",
            cases=(case,),
            source_files=(),
        )
        execution = AssistantExecution(
            case_id=case.case_id,
            query=case.query,
            route_mode="hierarchical_rag",
            query_intent="named_order_query",
            answer_status="answered",
            answer_text="The order does not state any IPO proceeds amount for this matter.",
            confidence=0.91,
            session_id=None,
            citations=(
                {
                    "citation_number": 1,
                    "record_key": "derived:69093",
                    "title": "Pacheli Industrial Finance Limited",
                    "source_type": "corpus_chunk",
                },
            ),
            active_record_keys=("derived:69093",),
            debug={"route_debug": {"strict_scope_required": True}},
            prompt_family="metadata-first fact",
            run_metadata={},
        )
        runner = EvaluationRunner(executor=_SingleExecutionExecutor(execution))
        metadata = build_run_metadata(
            dataset=dataset,
            executor_mode="stub",
            output_root=Path(tempfile.mkdtemp()),
        )

        run_result = runner.run(dataset=dataset, metadata=metadata)

        case_result = run_result.case_results[0]
        self.assertTrue(case_result.passed)
        self.assertEqual(case_result.failure_modes.primary_bucket, "stale expectation")
        self.assertEqual(run_result.summary["wrong_answer_regression_pass_count"], 1)
        self.assertEqual(run_result.summary["failed_cases"], 0)


def _mini_dataset():
    cases, source_files, _ = build_dataset(
        name="mini_eval",
        control_pack_root=CONTROL_PACK_FIXTURE_ROOT,
        failure_dump_root=EVAL_FAILURE_DUMP_FIXTURE_ROOT,
    )
    return merge_datasets(
        name="mini_eval",
        cases=cases[:5],
        source_files=source_files,
    )


class _SingleExecutionExecutor:
    mode = "stub"

    def __init__(self, execution: AssistantExecution) -> None:
        self._execution = execution

    def execute(self, *, case, session_id):
        return AssistantExecution(
            case_id=case.case_id,
            query=case.query,
            route_mode=self._execution.route_mode,
            query_intent=self._execution.query_intent,
            answer_status=self._execution.answer_status,
            answer_text=self._execution.answer_text,
            confidence=self._execution.confidence,
            session_id=self._execution.session_id,
            citations=self._execution.citations,
            retrieved_chunk_ids=self._execution.retrieved_chunk_ids,
            active_record_keys=self._execution.active_record_keys,
            retrieved_context=self._execution.retrieved_context,
            debug=dict(self._execution.debug),
            prompt_family=self._execution.prompt_family,
            run_metadata=dict(self._execution.run_metadata),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
