"""Reasoning-quality metrics."""

from __future__ import annotations

from .schemas import EvaluationCase, JudgeScores, ReasoningMetrics


def evaluate_reasoning_metrics(
    *,
    case: EvaluationCase,
    answer_text: str,
    judge: JudgeScores | None,
) -> ReasoningMetrics:
    """Score judged reasoning quality with deterministic fallback."""

    reasoning_quality = None
    explanation_quality = None
    summary_quality = None
    open_ended_answer_quality = None

    if judge is not None:
        reasoning_quality = _score_from_judge(judge.reasoning_quality)
        explanation_quality = _score_from_judge(judge.correctness)
        summary_quality = _score_from_judge(judge.conciseness_relevance)
        open_ended_answer_quality = _average_defined(
            reasoning_quality,
            explanation_quality,
            summary_quality,
        )
    else:
        length = len(answer_text.split())
        if answer_text.strip():
            baseline = 0.7 if length >= 20 else 0.45
            if case.issue_class == "gold_fact":
                baseline = 0.8 if length >= 10 else 0.6
            reasoning_quality = baseline
            explanation_quality = baseline
            summary_quality = 1.0 if 8 <= length <= 180 else 0.7
            open_ended_answer_quality = _average_defined(
                reasoning_quality,
                explanation_quality,
                summary_quality,
            )

    return ReasoningMetrics(
        reasoning_quality=_round_or_none(reasoning_quality),
        explanation_quality=_round_or_none(explanation_quality),
        summary_quality=_round_or_none(summary_quality),
        open_ended_answer_quality=_round_or_none(open_ended_answer_quality),
    )


def _score_from_judge(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 5.0


def _average_defined(*values: float | None) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
