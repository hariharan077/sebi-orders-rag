"""Deterministic numeric fact evaluation."""

from __future__ import annotations

import re
from datetime import date

from .schemas import EvaluationCase, GoldNumericFact, NumericMetrics

_NUMBER_RE = re.compile(r"[-+]?[0-9][0-9,]*(?:\.\d+)?")
_PERCENT_RE = re.compile(r"[-+]?[0-9][0-9,]*(?:\.\d+)?\s*%")
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_ISO_DATE_RE = re.compile(r"\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b")
_TEXT_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)


def evaluate_numeric_metrics(
    *,
    case: EvaluationCase,
    answer_text: str,
) -> NumericMetrics:
    """Compare expected numeric facts against the answer text."""

    if not case.gold_numeric_facts:
        return NumericMetrics(expected_fact_count=0, matched_fact_count=0, numeric_accuracy=None)

    matched: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for fact in case.gold_numeric_facts:
        if _fact_matches_answer(fact=fact, answer_text=answer_text):
            matched.append(fact.fact_type)
        elif _fact_present_with_wrong_value(fact=fact, answer_text=answer_text):
            mismatched.append(fact.fact_type)
        else:
            missing.append(fact.fact_type)

    expected_fact_count = len(case.gold_numeric_facts)
    matched_fact_count = len(matched)
    accuracy = matched_fact_count / expected_fact_count if expected_fact_count else None
    return NumericMetrics(
        expected_fact_count=expected_fact_count,
        matched_fact_count=matched_fact_count,
        numeric_accuracy=(round(accuracy, 4) if accuracy is not None else None),
        missing_fact_types=tuple(missing),
        mismatched_fact_types=tuple(mismatched),
        matched_fact_types=tuple(matched),
    )


def _fact_matches_answer(*, fact: GoldNumericFact, answer_text: str) -> bool:
    normalized_answer = " ".join(answer_text.split())
    if not normalized_answer:
        return False
    if fact.fact_type == "pan":
        expected_pan = str(fact.value_text or "").upper().strip()
        return bool(expected_pan and expected_pan in normalized_answer.upper())
    if fact.fact_type == "order_date":
        return _date_in_answer(fact=fact, answer_text=normalized_answer)
    if fact.value_text and fact.value_text in normalized_answer:
        return True
    if fact.value_numeric is None:
        return False
    numeric_matches = _numeric_candidates(fact=fact, answer_text=normalized_answer)
    tolerance_abs = fact.resolved_tolerance_abs()
    tolerance_pct = fact.tolerance_pct
    for candidate in numeric_matches:
        if tolerance_abs is not None and abs(candidate - fact.value_numeric) <= tolerance_abs:
            return True
        if tolerance_pct is not None and fact.value_numeric != 0:
            pct_delta = abs(candidate - fact.value_numeric) / abs(fact.value_numeric)
            if pct_delta <= tolerance_pct:
                return True
        if tolerance_abs is None and candidate == fact.value_numeric:
            return True
    return False


def _fact_present_with_wrong_value(*, fact: GoldNumericFact, answer_text: str) -> bool:
    lowered = answer_text.lower()
    if fact.fact_type == "pan":
        return bool(_PAN_RE.search(answer_text.upper()))
    if fact.fact_type == "order_date":
        return bool(_ISO_DATE_RE.search(answer_text) or _TEXT_DATE_RE.search(answer_text))
    if fact.fact_type in {"holding_percentage", "percentage_change", "percentage_change_from_listing"}:
        return "%" in answer_text
    if fact.fact_type in {
        "listing_price",
        "closing_price",
        "highest_price",
        "lowest_price",
        "settlement_amount",
        "penalty_amount",
    }:
        return "rs" in lowered or "inr" in lowered
    return bool(_NUMBER_RE.search(answer_text))


def _numeric_candidates(*, fact: GoldNumericFact, answer_text: str) -> tuple[float, ...]:
    if fact.fact_type in {"pan", "order_date"}:
        return ()
    matches = _PERCENT_RE.findall(answer_text) if "percent" in str(fact.unit or "").lower() else _NUMBER_RE.findall(answer_text)
    values: list[float] = []
    for match in matches:
        cleaned = match.replace("%", "").replace(",", "").strip()
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return tuple(values)


def _date_in_answer(*, fact: GoldNumericFact, answer_text: str) -> bool:
    expected_text = str(fact.value_text or "").strip()
    if expected_text and expected_text in answer_text:
        return True
    if fact.value_numeric is not None:
        try:
            ordinal = int(fact.value_numeric)
            expected_date = date.fromordinal(ordinal)
        except (ValueError, OverflowError):
            return False
        return expected_date.isoformat() in answer_text
    return False
