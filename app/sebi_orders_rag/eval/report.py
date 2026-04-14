"""Structured evaluation reporting for SEBI Orders hardening checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCaseResult:
    """One evaluated query or regression result."""

    case_kind: str
    query: str
    expected_route_mode: str | None
    actual_route_mode: str
    expected_record_key: str | None
    expected_title: str | None
    actual_active_record_keys: tuple[str, ...]
    actual_cited_record_keys: tuple[str, ...]
    strict_single_matter_triggered: bool
    comparison_disabled_lock: bool
    mixed_record_guardrail_fired: bool
    single_matter_rule_respected: bool
    answer_status: str
    confidence: float
    passed: bool
    equivalent_route_passed: bool = False
    equivalent_route_reason: str | None = None
    triage_bucket: str | None = None
    triage_tags: tuple[str, ...] = ()
    stale_expectation: bool = False
    true_bug: bool = False
    reasons: tuple[str, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate evaluation output across eval queries and regressions."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    answered_count: int
    clarify_count: int
    abstain_count: int
    route_accuracy: float
    route_accuracy_equivalent: float
    candidate_list_correctness: float
    stale_expectation_count: int
    true_bug_count: int
    single_matter_accuracy: float
    structured_current_info_accuracy: float
    count_parity_pass: bool | None
    request_failure_surface_count: int
    web_fallback_route_usage_count: int
    mixed_record_failure_count: int
    mixed_record_contamination_count: int
    candidate_list_usage_count: int
    wrong_example_regression_pass_count: int
    wrong_example_regression_total: int
    results: tuple[EvalCaseResult, ...]
    metadata_backfill_coverage: dict[str, int] = field(default_factory=dict)
    numeric_fact_coverage: dict[str, int] = field(default_factory=dict)
    triage_bucket_counts: dict[str, int] = field(default_factory=dict)


def render_summary(summary: EvalSummary) -> str:
    """Render a compact terminal summary."""

    lines = [
        f"cases: {summary.passed_cases}/{summary.total_cases} passed",
        f"answered: {summary.answered_count}",
        f"clarify: {summary.clarify_count}",
        f"abstain: {summary.abstain_count}",
        f"route accuracy (strict): {summary.route_accuracy:.4f}",
        f"route accuracy (equivalent): {summary.route_accuracy_equivalent:.4f}",
        f"candidate-list correctness: {summary.candidate_list_correctness:.4f}",
        f"stale expectations: {summary.stale_expectation_count}",
        f"true bugs: {summary.true_bug_count}",
        f"single-matter accuracy: {summary.single_matter_accuracy:.4f}",
        f"structured current-info accuracy: {summary.structured_current_info_accuracy:.4f}",
        f"count parity pass: {summary.count_parity_pass}",
        f"request-failure surface count: {summary.request_failure_surface_count}",
        f"web fallback route usage: {summary.web_fallback_route_usage_count}",
        f"mixed-record failures: {summary.mixed_record_failure_count}",
        f"mixed-record contamination count: {summary.mixed_record_contamination_count}",
        f"candidate-list usage: {summary.candidate_list_usage_count}",
        (
            "triage buckets: "
            + ", ".join(
                f"{key}={value}" for key, value in sorted(summary.triage_bucket_counts.items())
            )
        ),
        (
            "metadata-backfill coverage: "
            + ", ".join(
                f"{key}={value}" for key, value in sorted(summary.metadata_backfill_coverage.items())
            )
        ),
        (
            "numeric-fact coverage: "
            + ", ".join(
                f"{key}={value}" for key, value in sorted(summary.numeric_fact_coverage.items())
            )
        ),
        (
            "wrong-example regressions: "
            f"{summary.wrong_example_regression_pass_count}/{summary.wrong_example_regression_total} passed"
        ),
    ]
    return "\n".join(lines)
