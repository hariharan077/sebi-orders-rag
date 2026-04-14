"""Grounded-generation and hallucination metrics."""

from __future__ import annotations

from .schemas import (
    EvaluationCase,
    GroundingMetrics,
    JudgeScores,
    NumericMetrics,
    RetrievalMetrics,
)


def evaluate_grounding_metrics(
    *,
    case: EvaluationCase,
    route_mode: str,
    answer_status: str,
    answer_text: str,
    actual_record_keys: tuple[str, ...],
    citations: tuple[dict[str, object], ...],
    debug: dict[str, object],
    retrieval: RetrievalMetrics,
    numeric: NumericMetrics,
    judge: JudgeScores | None,
) -> GroundingMetrics:
    """Score grounding, hallucination risk, and answer correctness."""

    web_debug = dict(debug.get("web_fallback_debug", {}) or {})
    metadata_debug = dict(debug.get("metadata_debug", {}) or {})
    citation_keys = {
        str(item.get("record_key") or "").strip()
        for item in citations
        if str(item.get("record_key") or "").strip()
    }
    actual_keys = set(actual_record_keys) | citation_keys
    expected_keys = set(case.expected_record_keys)
    unsupported_claim_count = 0
    missing_critical_info_count = 0

    if expected_keys and answer_status == "answered":
        if not actual_keys:
            unsupported_claim_count += 1
        if actual_keys and not actual_keys.issubset(expected_keys):
            unsupported_claim_count += len(actual_keys - expected_keys)
            missing_critical_info_count += 1

    abstain_correct = None
    if case.must_abstain:
        abstain_correct = answer_status in {"abstained", "clarify"}
    clarify_correct = None
    if case.must_clarify:
        clarify_correct = answer_status == "clarify"
        if not clarify_correct:
            missing_critical_info_count += 1

    used_metadata_correctly = None
    if case.must_use_metadata:
        used_metadata_correctly = bool(metadata_debug.get("used"))
        if not used_metadata_correctly:
            missing_critical_info_count += 1

    used_structured_current_info_correctly = None
    if case.must_use_structured_current_info:
        used_structured_current_info_correctly = (
            route_mode == "structured_current_info"
            or "structured" in {
                str(item.get("source_type") or "").strip()
                for item in citations
            }
        )
        if not used_structured_current_info_correctly:
            missing_critical_info_count += 1

    used_official_web_correctly = None
    if case.must_use_official_web:
        used_official_web_correctly = bool(web_debug.get("official_web_attempted")) or route_mode in {
            "current_official_lookup",
            "historical_official_lookup",
            "current_news_lookup",
        }
        if not used_official_web_correctly:
            missing_critical_info_count += 1
    elif case.must_not_use_web:
        used_official_web_correctly = not (
            bool(web_debug.get("official_web_attempted"))
            or bool(web_debug.get("general_web_attempted"))
        )
        if used_official_web_correctly is False:
            unsupported_claim_count += 1

    if numeric.expected_fact_count and numeric.numeric_accuracy is not None and numeric.numeric_accuracy < 1.0:
        missing_critical_info_count += len(numeric.missing_fact_types) + len(
            numeric.mismatched_fact_types
        )

    deterministic_faithfulness = _deterministic_faithfulness(
        case=case,
        answer_status=answer_status,
        actual_keys=actual_keys,
        retrieval=retrieval,
        numeric=numeric,
        unsupported_claim_count=unsupported_claim_count,
    )
    deterministic_correctness = _deterministic_correctness(
        case=case,
        answer_status=answer_status,
        answer_text=answer_text,
        numeric=numeric,
        abstain_correct=abstain_correct,
        clarify_correct=clarify_correct,
    )
    answer_relevance = _answer_relevance(case=case, answer_text=answer_text)
    conciseness = _answer_conciseness(case=case, answer_text=answer_text)

    if judge is not None and judge.faithfulness is not None:
        faithfulness = round(((judge.faithfulness / 5.0) + deterministic_faithfulness) / 2.0, 4)
    else:
        faithfulness = round(deterministic_faithfulness, 4)

    if judge is not None and judge.correctness is not None:
        answer_correctness = round(((judge.correctness / 5.0) + deterministic_correctness) / 2.0, 4)
    else:
        answer_correctness = round(deterministic_correctness, 4)

    hallucination_detected = (
        unsupported_claim_count > 0
        or retrieval.mixed_record_contamination
        or bool(judge and "hallucination" in set(judge.failure_modes))
    )
    hallucination_rate = round(
        max(0.0, min(1.0, (1.0 - faithfulness) + (0.15 if hallucination_detected else 0.0))),
        4,
    )
    return GroundingMetrics(
        faithfulness=faithfulness,
        hallucination_rate=hallucination_rate,
        hallucination_detected=hallucination_detected,
        unsupported_claim_count=unsupported_claim_count,
        missing_critical_info_count=missing_critical_info_count,
        answer_correctness=answer_correctness,
        answer_relevance=round(answer_relevance, 4),
        conciseness=round(conciseness, 4),
        abstain_correct=abstain_correct,
        clarify_correct=clarify_correct,
        used_metadata_correctly=used_metadata_correctly,
        used_structured_current_info_correctly=used_structured_current_info_correctly,
        used_official_web_correctly=used_official_web_correctly,
    )


def _deterministic_faithfulness(
    *,
    case: EvaluationCase,
    answer_status: str,
    actual_keys: set[str],
    retrieval: RetrievalMetrics,
    numeric: NumericMetrics,
    unsupported_claim_count: int,
) -> float:
    if case.must_abstain and answer_status in {"abstained", "clarify"}:
        return 1.0
    if case.must_clarify and answer_status == "clarify":
        return 1.0
    if answer_status != "answered":
        return 0.0
    score = 1.0
    if case.expected_record_keys and actual_keys and not actual_keys.issubset(set(case.expected_record_keys)):
        score -= 0.5
    if retrieval.context_precision is not None:
        score = min(score, retrieval.context_precision)
    if numeric.expected_fact_count and numeric.numeric_accuracy is not None:
        score = min(score, max(0.0, numeric.numeric_accuracy))
    score -= unsupported_claim_count * 0.2
    return max(0.0, min(1.0, score))


def _deterministic_correctness(
    *,
    case: EvaluationCase,
    answer_status: str,
    answer_text: str,
    numeric: NumericMetrics,
    abstain_correct: bool | None,
    clarify_correct: bool | None,
) -> float:
    if abstain_correct is not None:
        return 1.0 if abstain_correct else 0.0
    if clarify_correct is not None:
        return 1.0 if clarify_correct else 0.0
    if answer_status != "answered":
        return 0.0
    regression_guidance_correctness = _regression_guidance_correctness(
        case=case,
        answer_status=answer_status,
        answer_text=answer_text,
    )
    if regression_guidance_correctness is not None:
        if numeric.numeric_accuracy is not None:
            regression_guidance_correctness = (
                regression_guidance_correctness + numeric.numeric_accuracy
            ) / 2.0
        return regression_guidance_correctness
    if case.gold_answer_short:
        gold_terms = {
            token
            for token in case.gold_answer_short.lower().split()
            if len(token) > 3
        }
        answer_terms = {
            token
            for token in answer_text.lower().split()
            if len(token) > 3
        }
        if gold_terms:
            overlap = len(gold_terms & answer_terms) / len(gold_terms)
            if numeric.numeric_accuracy is not None:
                overlap = (overlap + numeric.numeric_accuracy) / 2.0
            return overlap
    if numeric.numeric_accuracy is not None:
        return numeric.numeric_accuracy
    return 1.0 if answer_text.strip() else 0.0


def _regression_guidance_correctness(
    *,
    case: EvaluationCase,
    answer_status: str,
    answer_text: str,
) -> float | None:
    if case.issue_class != "regression":
        return None
    guidance = " ".join(
        str(case.metadata.get("answer_guidance") or "").lower().split()
    )
    if not guidance:
        return None
    normalized_answer = " ".join(answer_text.lower().split())

    if (
        "no exemption order is in scope" in guidance
        or "not an exemption order" in guidance
    ):
        if answer_status in {"abstained", "clarify"}:
            return 1.0
        if "not an exemption order" in normalized_answer or "no exemption order" in normalized_answer:
            return 1.0
        return 0.0

    if (
        "should not relabel" in guidance
        and "preferential allotment" in guidance
        and "ipo proceeds" in guidance
    ):
        if "does not describe ipo proceeds" in normalized_answer and "preferential allotment" in normalized_answer:
            return 1.0
        if answer_status in {"abstained", "clarify"}:
            return 0.7
        return 0.0

    if "expected anchor:" in guidance and case.expected_record_keys:
        return 1.0 if answer_status == "answered" and normalized_answer else 0.0

    return None


def _answer_relevance(*, case: EvaluationCase, answer_text: str) -> float:
    if not answer_text.strip():
        return 0.0
    if case.must_abstain or case.must_clarify:
        return 1.0 if len(answer_text.split()) <= 40 else 0.7
    return 1.0


def _answer_conciseness(*, case: EvaluationCase, answer_text: str) -> float:
    word_count = len(answer_text.split())
    if not word_count:
        return 0.0
    if case.issue_class in {"gold_fact", "clarify", "abstain"}:
        if word_count <= 80:
            return 1.0
        if word_count <= 140:
            return 0.7
        return 0.4
    if word_count <= 220:
        return 1.0
    if word_count <= 320:
        return 0.75
    return 0.45
