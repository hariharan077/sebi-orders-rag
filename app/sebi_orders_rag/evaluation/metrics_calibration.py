"""Confidence calibration metrics."""

from __future__ import annotations

from .schemas import CalibrationMetrics, EvaluationCase


def build_calibration_metrics(
    *,
    case: EvaluationCase,
    confidence: float,
    passed: bool,
    answer_status: str,
) -> CalibrationMetrics:
    """Build one calibration observation."""

    bounded = max(0.0, min(1.0, float(confidence)))
    bucket_floor = int(bounded * 10) / 10
    bucket_ceiling = min(1.0, bucket_floor + 0.1)
    confidence_bin = f"{bucket_floor:.1f}-{bucket_ceiling:.1f}"
    expected_abstain_or_clarify = case.must_abstain or case.must_clarify
    predicted_abstain_or_clarify = answer_status in {"abstained", "clarify"}
    return CalibrationMetrics(
        confidence=round(bounded, 4),
        confidence_bin=confidence_bin,
        correct=passed,
        expected_abstain_or_clarify=expected_abstain_or_clarify,
        predicted_abstain_or_clarify=predicted_abstain_or_clarify,
    )
