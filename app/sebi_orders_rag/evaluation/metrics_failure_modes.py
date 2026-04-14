"""Failure-mode taxonomy and classification."""

from __future__ import annotations

from .schemas import (
    EvaluationCase,
    FailureModeMetrics,
    GroundingMetrics,
    JudgeScores,
    NumericMetrics,
    RetrievalMetrics,
    RouteMetrics,
)

_BUCKET_PRIORITY = (
    "contamination",
    "missing clarify",
    "missing abstain",
    "wrong route",
    "wrong candidate ranking",
    "weak metadata support",
    "current-fact-vs-order confusion",
    "person-vs-trust confusion",
    "settlement/final-order phrasing issue",
    "numeric fact extraction miss",
    "stale session context leak",
    "weak official-web fallback",
    "structured-current-info mismatch",
    "over-abstain",
    "wrong answer despite correct route",
    "stale expectation",
)


def classify_failure_modes(
    *,
    case: EvaluationCase,
    route_mode: str,
    route: RouteMetrics,
    retrieval: RetrievalMetrics,
    grounding: GroundingMetrics,
    numeric: NumericMetrics,
    answer_status: str,
    answer_text: str,
    debug: dict[str, object],
    judge: JudgeScores | None,
) -> FailureModeMetrics:
    """Map result signals into auditable failure buckets."""

    buckets: list[str] = []
    route_debug = dict(debug.get("route_debug", {}) or {})
    metadata_debug = dict(debug.get("metadata_debug", {}) or {})
    candidate_list_debug = dict(debug.get("candidate_list_debug", {}) or {})
    web_debug = dict(debug.get("web_fallback_debug", {}) or {})

    if retrieval.mixed_record_contamination or grounding.hallucination_detected and case.expected_record_keys:
        buckets.append("contamination")
    if case.must_clarify and answer_status != "clarify":
        buckets.append("missing clarify")
    if case.must_abstain and answer_status not in {"abstained", "clarify"}:
        buckets.append("missing abstain")
    if not route.strict_route_match and not route.equivalent_route_match:
        buckets.append("wrong route")
    if case.must_clarify and candidate_list_debug.get("used") and retrieval.candidate_list_correctness == 0.0:
        buckets.append("wrong candidate ranking")
    if (
        case.must_use_metadata
        and not metadata_debug.get("used")
        and not (
            case.must_abstain
            and grounding.abstain_correct
            and answer_status in {"abstained", "clarify"}
        )
    ):
        buckets.append("weak metadata support")
    if (
        (case.must_use_structured_current_info or case.must_use_official_web)
        and route_mode in {"hierarchical_rag", "memory_scoped_rag", "exact_lookup"}
    ):
        buckets.append("current-fact-vs-order confusion")
    if "person_vs_trust" in case.tags and numeric.numeric_accuracy is not None and numeric.numeric_accuracy < 1.0:
        buckets.append("person-vs-trust confusion")
    if "settlement" in " ".join(case.tags) and "final order" in answer_text.lower():
        buckets.append("settlement/final-order phrasing issue")
    if numeric.expected_fact_count and numeric.numeric_accuracy is not None and numeric.numeric_accuracy < 1.0:
        buckets.append("numeric fact extraction miss")
    if case.must_use_active_matter and not route_debug.get("active_order_override"):
        buckets.append("stale session context leak")
    if case.must_use_official_web and not (
        web_debug.get("official_web_attempted") or route_mode in {
            "current_official_lookup",
            "historical_official_lookup",
            "current_news_lookup",
        }
    ):
        buckets.append("weak official-web fallback")
    if case.must_use_structured_current_info and route_mode != "structured_current_info":
        buckets.append("structured-current-info mismatch")
    if not case.must_abstain and not case.must_clarify and answer_status == "abstained":
        buckets.append("over-abstain")
    if judge is not None:
        for item in judge.failure_modes:
            if item == "current-fact-vs-order confusion" and item not in buckets:
                buckets.append(item)
    if route.equivalent_route_match and not route.strict_route_match:
        buckets.append("stale expectation")
    if not buckets and (
        grounding.answer_correctness is not None and grounding.answer_correctness < 0.99
    ):
        buckets.append("wrong answer despite correct route")
    if not buckets:
        buckets.append("pass")

    primary_bucket = next(
        (bucket for bucket in _BUCKET_PRIORITY if bucket in buckets),
        buckets[0],
    )
    stale_expectation = "stale expectation" in buckets
    return FailureModeMetrics(
        primary_bucket=primary_bucket,
        buckets=tuple(dict.fromkeys(buckets)),
        stale_expectation=stale_expectation,
        true_bug=not stale_expectation and primary_bucket != "pass",
    )
