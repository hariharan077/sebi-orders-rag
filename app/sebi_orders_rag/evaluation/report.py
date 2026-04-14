"""Human-readable evaluation reporting."""

from __future__ import annotations

from collections import Counter

from .schemas import CaseEvaluationResult
from .triage import render_compare_summary_markdown, render_true_bug_queue_markdown


def render_run_summary(summary: dict[str, object]) -> str:
    """Render a compact terminal summary."""

    lines = [
        f"total cases: {summary.get('total_cases', 0)}",
        f"passed / failed: {summary.get('passed_cases', 0)} / {summary.get('failed_cases', 0)}",
        f"strict route accuracy: {float(summary.get('strict_route_accuracy', 0.0)):.4f}",
        f"equivalent route accuracy: {float(summary.get('equivalent_route_accuracy', 0.0)):.4f}",
        f"faithfulness average: {float(summary.get('faithfulness_average', 0.0)):.4f}",
        f"hallucination rate: {float(summary.get('hallucination_rate', 0.0)):.4f}",
        f"candidate-list correctness: {float(summary.get('candidate_list_correctness', 0.0)):.4f}",
        f"numeric-fact accuracy: {float(summary.get('numeric_fact_accuracy', 0.0)):.4f}",
        f"structured-current-info accuracy: {float(summary.get('structured_current_info_accuracy', 0.0)):.4f}",
        f"current-fact routing accuracy: {float(summary.get('current_fact_routing_accuracy', 0.0)):.4f}",
        f"wrong-answer regressions: {summary.get('wrong_answer_regression_pass_count', 0)}/{summary.get('wrong_answer_regression_total', 0)}",
        f"stale expectations: {summary.get('stale_expectation_count', 0)}",
        f"true bugs: {summary.get('true_bug_count', 0)}",
        f"confidence ECE: {float(summary.get('confidence_ece', 0.0)):.4f}",
        f"confidence Brier: {float(summary.get('confidence_brier', 0.0)):.4f}",
    ]
    bucket_counts = summary.get("failure_bucket_counts", {}) or {}
    if bucket_counts:
        lines.append(
            "failure buckets: "
            + ", ".join(f"{key}={value}" for key, value in sorted(bucket_counts.items()))
        )
    status_counts = summary.get("answer_status_counts", {}) or {}
    if status_counts:
        lines.append(
            "answer statuses: "
            + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
        )
    return "\n".join(lines)


def render_failures_markdown(results: tuple[CaseEvaluationResult, ...] | list[CaseEvaluationResult]) -> str:
    """Render failed cases to markdown."""

    failed = [item for item in results if not item.passed]
    if not failed:
        return "# Failures\n\nNo failed cases.\n"
    lines = ["# Failures", ""]
    bucket_counts = Counter(item.failure_modes.primary_bucket for item in failed)
    lines.append(
        "Buckets: " + ", ".join(f"{bucket}={count}" for bucket, count in sorted(bucket_counts.items()))
    )
    lines.append("")
    for index, item in enumerate(failed, start=1):
        lines.append(f"## {index}. {item.case.query}")
        lines.append("")
        lines.append(f"- Case ID: `{item.case.case_id}`")
        lines.append(f"- Expected route: `{item.case.route_family_expected}`")
        lines.append(f"- Actual route: `{item.execution.route_mode}`")
        lines.append(f"- Answer status: `{item.execution.answer_status}`")
        lines.append(f"- Primary bucket: `{item.failure_modes.primary_bucket}`")
        lines.append(f"- Failure buckets: {', '.join(item.failure_modes.buckets)}")
        lines.append(f"- Confidence: `{item.execution.confidence:.4f}`")
        if item.case.expected_record_keys:
            lines.append(
                f"- Expected record keys: {', '.join(item.case.expected_record_keys)}"
            )
        if item.execution.active_record_keys:
            lines.append(
                f"- Active record keys: {', '.join(item.execution.active_record_keys)}"
            )
        lines.append(
            f"- Unsupported claims: `{item.grounding.unsupported_claim_count}`"
        )
        lines.append(
            f"- Missing critical info: `{item.grounding.missing_critical_info_count}`"
        )
        if item.numeric.expected_fact_count:
            lines.append(
                f"- Numeric accuracy: `{item.numeric.numeric_accuracy}` "
                f"(matched {item.numeric.matched_fact_count}/{item.numeric.expected_fact_count})"
            )
        lines.append("")
        lines.append("```text")
        lines.append(item.execution.answer_text.strip() or "<empty>")
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = (
    "render_compare_summary_markdown",
    "render_failures_markdown",
    "render_run_summary",
    "render_true_bug_queue_markdown",
)
