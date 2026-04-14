"""Control-pack evaluator for the SEBI Orders retrieval hardening pass."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import UUID

from ..control import ControlPack, dataclass_asdict
from ..schemas import ChatAnswerPayload
from .report import EvalCaseResult, EvalSummary
from .triage import (
    load_failure_dump_reference,
    triage_bucket_counts,
    triage_eval_result,
)


class ControlPackEvaluator:
    """Run eval queries and wrong-answer regressions against one answer service."""

    def __init__(
        self,
        *,
        service: Any,
        control_pack: ControlPack,
        failure_dump_root: Path | None = None,
    ) -> None:
        self._service = service
        self._control_pack = control_pack
        self._failure_dump_reference = load_failure_dump_reference(failure_dump_root)

    def run(
        self,
        *,
        run_eval_queries: bool,
        run_regressions: bool,
    ) -> EvalSummary:
        results: list[EvalCaseResult] = []
        if run_eval_queries:
            results.extend(self._run_eval_queries())
        if run_regressions:
            results.extend(self._run_regressions())
        annotated_results = _annotate_results(
            tuple(results),
            failure_dump_reference=self._failure_dump_reference,
        )
        return _summarize_results(
            annotated_results,
            count_parity_pass=self._count_parity_pass(),
            metadata_backfill_coverage=self._metadata_backfill_coverage(),
        )

    def _count_parity_pass(self) -> bool | None:
        connection = getattr(self._service, "_connection", None)
        if connection is None:
            return None
        try:
            from ..repositories.structured_info import StructuredInfoRepository
            from ..structured_info.audit import build_audit_report

            report = build_audit_report(StructuredInfoRepository(connection))
        except Exception:
            return None
        return not report.failures

    def _metadata_backfill_coverage(self) -> dict[str, int]:
        connection = getattr(self._service, "_connection", None)
        if connection is None:
            return {}
        try:
            from ..repositories.metadata import OrderMetadataRepository

            return OrderMetadataRepository(connection).fetch_backfill_coverage()
        except Exception:
            return {}

    def _run_eval_queries(self) -> list[EvalCaseResult]:
        session_by_group: dict[str, UUID] = {}
        previous_session_id: UUID | None = None
        results: list[EvalCaseResult] = []
        for case in self._control_pack.eval_queries:
            session_id = None
            if case.session_group and case.session_group in session_by_group:
                session_id = session_by_group[case.session_group]
            elif case.reuse_previous_session:
                session_id = previous_session_id

            payload = self._service.answer_query(query=case.query, session_id=session_id)
            previous_session_id = payload.session_id
            if case.session_group:
                session_by_group[case.session_group] = payload.session_id
            results.append(
                _evaluate_payload(
                    payload=payload,
                    case_kind="eval_query",
                    query=case.query,
                    expected_route_mode=case.expected_route_mode or None,
                    expected_record_key=case.expected_record_key,
                    expected_title=case.expected_title,
                    comparison_allowed=case.comparison_allowed,
                    incorrect_record_keys=(),
                )
            )
        return results

    def _run_regressions(self) -> list[EvalCaseResult]:
        results: list[EvalCaseResult] = []
        for example in self._control_pack.wrong_answer_examples:
            payload = self._service.answer_query(query=example.user_query)
            results.append(
                _evaluate_payload(
                    payload=payload,
                    case_kind="wrong_answer_regression",
                    query=example.user_query,
                    expected_route_mode=None,
                    expected_record_key=example.expected_record_key,
                    expected_title=example.expected_title,
                    comparison_allowed=False,
                    incorrect_record_keys=example.incorrectly_pulled_record_keys,
                )
            )
        return results


def summary_as_dict(summary: EvalSummary) -> dict[str, object]:
    """Return a JSON-serializable representation of one evaluation summary."""

    return asdict(summary)


def _evaluate_payload(
    *,
    payload: ChatAnswerPayload,
    case_kind: str,
    query: str,
    expected_route_mode: str | None,
    expected_record_key: str | None,
    expected_title: str | None,
    comparison_allowed: bool,
    incorrect_record_keys: tuple[str, ...],
) -> EvalCaseResult:
    route_debug = dict(payload.debug.get("route_debug", {}))
    guardrail_debug = dict(payload.debug.get("mixed_record_guardrail", {}))
    current_lookup_debug = dict(payload.debug.get("current_lookup_debug", {}))
    news_lookup_debug = dict(payload.debug.get("news_lookup_debug", {}))
    historical_lookup_debug = dict(payload.debug.get("historical_lookup_debug", {}))
    metadata_debug = dict(payload.debug.get("metadata_debug", {}))
    candidate_list_debug = dict(payload.debug.get("candidate_list_debug", {}))
    web_fallback_debug = dict(payload.debug.get("web_fallback_debug", {}))
    citation_debug = dict(payload.debug.get("citation_debug", {}))
    actual_cited_record_keys = tuple(citation.record_key for citation in payload.citations)
    actual_active_record_keys = tuple(payload.active_record_keys)
    actual_record_key_set = set(actual_cited_record_keys or actual_active_record_keys)
    reasons: list[str] = []

    if expected_route_mode:
        if payload.route_mode != expected_route_mode:
            reasons.append(f"expected route {expected_route_mode} got {payload.route_mode}")

    strict_single_matter_triggered = bool(route_debug.get("strict_single_matter", False))
    comparison_disabled_lock = bool(
        payload.debug.get("search_debug", {}).get("comparison_intent_disabled_lock", False)
    )
    single_matter_rule_respected = bool(
        guardrail_debug.get("single_matter_rule_respected", True)
    )
    mixed_record_guardrail_fired = bool(guardrail_debug.get("guardrail_fired", False)) or bool(
        guardrail_debug.get("mixed_record_guardrail_fired", False)
    )

    if expected_record_key and not comparison_allowed:
        if payload.answer_status != "abstained":
            if not actual_record_key_set:
                reasons.append("expected grounded record key but answer cited none")
            elif actual_record_key_set != {expected_record_key}:
                reasons.append(
                    "expected single record "
                    f"{expected_record_key} got {sorted(actual_record_key_set)}"
                )
        grounded_single_record = (
            actual_record_key_set == {expected_record_key}
            and single_matter_rule_respected
        )
        if (
            not strict_single_matter_triggered
            and payload.answer_status != "abstained"
            and not grounded_single_record
        ):
            reasons.append("strict single-matter lock did not trigger")
    if not expected_record_key and case_kind == "wrong_answer_regression":
        if payload.answer_status not in {"abstained", "clarify"}:
            reasons.append("expected abstain or clarify for ambiguous or unsupported named matter")

    wrong_record_key_hits = sorted(set(incorrect_record_keys) & actual_record_key_set)
    if wrong_record_key_hits:
        reasons.append(f"cited contaminated record keys {wrong_record_key_hits}")
    if not comparison_allowed and not single_matter_rule_respected:
        reasons.append("single-matter rule not respected")

    passed = not reasons
    return EvalCaseResult(
        case_kind=case_kind,
        query=query,
        expected_route_mode=expected_route_mode,
        actual_route_mode=payload.route_mode,
        expected_record_key=expected_record_key,
        expected_title=expected_title,
        actual_active_record_keys=actual_active_record_keys,
        actual_cited_record_keys=actual_cited_record_keys,
        strict_single_matter_triggered=strict_single_matter_triggered,
        comparison_disabled_lock=comparison_disabled_lock,
        mixed_record_guardrail_fired=mixed_record_guardrail_fired,
        single_matter_rule_respected=single_matter_rule_respected,
        answer_status=payload.answer_status,
        confidence=payload.confidence,
        passed=passed,
        reasons=tuple(reasons),
        debug=dataclass_asdict(
            {
                "route_debug": route_debug,
                "guardrail_debug": guardrail_debug,
                "current_lookup_debug": current_lookup_debug,
                "news_lookup_debug": news_lookup_debug,
                "historical_lookup_debug": historical_lookup_debug,
                "metadata_debug": metadata_debug,
                "candidate_list_debug": candidate_list_debug,
                "web_fallback_debug": web_fallback_debug,
                "citation_debug": citation_debug,
                "answer_text_contains_request_failed": "request failed" in payload.answer_text.lower(),
            }
        ),
    )


def _summarize_results(
    results: tuple[EvalCaseResult, ...],
    *,
    count_parity_pass: bool | None,
    metadata_backfill_coverage: dict[str, int],
) -> EvalSummary:
    numeric_fact_coverage = {
        key: value
        for key, value in metadata_backfill_coverage.items()
        if key
        in {
            "numeric_fact_docs",
            "numeric_fact_rows",
            "listing_price_docs",
            "highest_price_docs",
            "settlement_amount_docs",
            "price_movement_docs",
            "price_movement_rows",
        }
    }
    metadata_coverage = {
        key: value
        for key, value in metadata_backfill_coverage.items()
        if key not in numeric_fact_coverage
    }
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    failed_cases = total_cases - passed_cases
    answered_count = sum(1 for result in results if result.answer_status in {"answered", "cautious"})
    clarify_count = sum(1 for result in results if result.answer_status == "clarify")
    abstain_count = sum(1 for result in results if result.answer_status == "abstained")

    route_cases = [result for result in results if result.expected_route_mode]
    route_accuracy = (
        sum(1 for result in route_cases if result.actual_route_mode == result.expected_route_mode)
        / len(route_cases)
        if route_cases
        else 1.0
    )
    route_accuracy_equivalent = (
        sum(
            1
            for result in route_cases
            if (
                result.actual_route_mode == result.expected_route_mode
                or result.equivalent_route_passed
            )
        )
        / len(route_cases)
        if route_cases
        else 1.0
    )

    strict_cases = [
        result
        for result in results
        if result.expected_record_key and result.case_kind in {"eval_query", "wrong_answer_regression"}
    ]
    single_matter_accuracy = (
        sum(1 for result in strict_cases if result.single_matter_rule_respected) / len(strict_cases)
        if strict_cases
        else 1.0
    )
    structured_cases = [
        result
        for result in results
        if result.expected_route_mode == "structured_current_info"
    ]
    structured_current_info_accuracy = (
        sum(1 for result in structured_cases if result.actual_route_mode == "structured_current_info")
        / len(structured_cases)
        if structured_cases
        else 1.0
    )
    mixed_record_failure_count = sum(
        1 for result in results if not result.single_matter_rule_respected
    )
    request_failed_surface_count = sum(
        1 for result in results if result.debug.get("answer_text_contains_request_failed")
    )
    web_fallback_route_usage_count = sum(
        1
        for result in results
        if (
            result.debug.get("news_lookup_debug", {}).get("used")
            or result.debug.get("historical_lookup_debug", {}).get("used")
            or result.debug.get("web_fallback_debug", {}).get("official_web_attempted")
            or result.debug.get("web_fallback_debug", {}).get("general_web_attempted")
        )
    )
    candidate_list_usage_count = sum(
        1 for result in results if result.debug.get("candidate_list_debug", {}).get("used")
    )
    candidate_list_results = [
        result for result in results if result.debug.get("candidate_list_debug", {}).get("used")
    ]
    candidate_list_correctness = (
        sum(
            1
            for result in candidate_list_results
            if result.passed or result.equivalent_route_passed or result.stale_expectation
        )
        / len(candidate_list_results)
        if candidate_list_results
        else 1.0
    )
    mixed_record_contamination_count = sum(
        1
        for result in results
        if (
            "contaminated" in " ".join(result.reasons).lower()
            or not result.single_matter_rule_respected
        )
    )

    regression_results = [result for result in results if result.case_kind == "wrong_answer_regression"]
    wrong_example_regression_pass_count = sum(
        1 for result in regression_results if result.passed or result.equivalent_route_passed
    )
    stale_expectation_count = sum(1 for result in results if result.stale_expectation)
    true_bug_count = sum(
        1
        for result in results
        if not result.passed and not result.stale_expectation
    )

    return EvalSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        answered_count=answered_count,
        clarify_count=clarify_count,
        abstain_count=abstain_count,
        route_accuracy=round(route_accuracy, 4),
        route_accuracy_equivalent=round(route_accuracy_equivalent, 4),
        candidate_list_correctness=round(candidate_list_correctness, 4),
        stale_expectation_count=stale_expectation_count,
        true_bug_count=true_bug_count,
        single_matter_accuracy=round(single_matter_accuracy, 4),
        structured_current_info_accuracy=round(structured_current_info_accuracy, 4),
        count_parity_pass=count_parity_pass,
        request_failure_surface_count=request_failed_surface_count,
        web_fallback_route_usage_count=web_fallback_route_usage_count,
        mixed_record_failure_count=mixed_record_failure_count,
        mixed_record_contamination_count=mixed_record_contamination_count,
        candidate_list_usage_count=candidate_list_usage_count,
        wrong_example_regression_pass_count=wrong_example_regression_pass_count,
        wrong_example_regression_total=len(regression_results),
        results=results,
        metadata_backfill_coverage=metadata_coverage,
        numeric_fact_coverage=numeric_fact_coverage,
        triage_bucket_counts=triage_bucket_counts(results),
    )


def _annotate_results(
    results: tuple[EvalCaseResult, ...],
    *,
    failure_dump_reference,
) -> tuple[EvalCaseResult, ...]:
    annotated: list[EvalCaseResult] = []
    for result in results:
        triage = triage_eval_result(
            result,
            reference=failure_dump_reference,
        )
        annotated.append(
            replace(
                result,
                equivalent_route_passed=triage.equivalent_route_passed,
                equivalent_route_reason=triage.equivalent_route_reason,
                triage_bucket=triage.primary_bucket,
                triage_tags=triage.bucket_tags,
                stale_expectation=triage.stale_expectation,
                true_bug=triage.true_bug,
            )
        )
    return tuple(annotated)
