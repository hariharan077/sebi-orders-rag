"""Retrieval quality metrics for evaluation cases."""

from __future__ import annotations

from collections import Counter

from .schemas import EvaluationCase, RetrievalMetrics, RetrievedContext


def evaluate_retrieval_metrics(
    *,
    case: EvaluationCase,
    retrieved_context: tuple[RetrievedContext, ...],
    debug: dict[str, object],
) -> RetrievalMetrics:
    """Measure retrieval-side precision, recall, purity, and candidate quality."""

    candidate_list_debug = dict(debug.get("candidate_list_debug", {}) or {})
    retrieved_record_keys = tuple(
        _ordered_unique(item.record_key for item in retrieved_context if item.record_key)
    )
    retrieved_bucket_names = tuple(
        _ordered_unique(item.bucket_name for item in retrieved_context if item.bucket_name)
    )

    expected_records = set(case.expected_record_keys)
    expected_buckets = set(case.expected_bucket_names)
    retrieved_records = [item.record_key for item in retrieved_context if item.record_key]
    retrieved_buckets = [item.bucket_name for item in retrieved_context if item.bucket_name]

    context_precision = None
    context_recall = None
    if retrieved_context:
        if expected_records:
            match_count = sum(1 for value in retrieved_records if value in expected_records)
            context_precision = match_count / len(retrieved_records) if retrieved_records else 0.0
            context_recall = (
                len(expected_records & set(retrieved_records)) / len(expected_records)
            )
        elif expected_buckets:
            match_count = sum(1 for value in retrieved_buckets if value in expected_buckets)
            context_precision = match_count / len(retrieved_buckets) if retrieved_buckets else 0.0
            context_recall = (
                len(expected_buckets & set(retrieved_buckets)) / len(expected_buckets)
            )

    context_relevance = _weighted_context_relevance(
        case=case,
        retrieved_context=retrieved_context,
    )
    redundancy_ratio = _redundancy_ratio(retrieved_context)
    duplicate_context_ratio = _duplicate_ratio(retrieved_context)
    single_matter_purity = _single_matter_purity(retrieved_context)
    mixed_record_contamination = bool(
        expected_records
        and any(value not in expected_records for value in retrieved_records)
    )
    candidate_list_correctness = _candidate_list_correctness(case=case, debug=candidate_list_debug)

    return RetrievalMetrics(
        context_precision=_round_or_none(context_precision),
        context_recall=_round_or_none(context_recall),
        context_relevance=_round_or_none(context_relevance),
        redundancy_ratio=_round_or_none(redundancy_ratio),
        duplicate_context_ratio=_round_or_none(duplicate_context_ratio),
        single_matter_purity=_round_or_none(single_matter_purity),
        mixed_record_contamination=mixed_record_contamination,
        candidate_list_correctness=_round_or_none(candidate_list_correctness),
        expected_record_retrieved=bool(expected_records & set(retrieved_records)),
        expected_bucket_retrieved=bool(expected_buckets & set(retrieved_buckets)),
        retrieved_record_keys=retrieved_record_keys,
        retrieved_bucket_names=retrieved_bucket_names,
    )


def _weighted_context_relevance(
    *,
    case: EvaluationCase,
    retrieved_context: tuple[RetrievedContext, ...],
) -> float | None:
    if not retrieved_context:
        return 0.0 if (case.expected_record_keys or case.expected_bucket_names) else None
    expected_records = set(case.expected_record_keys)
    expected_buckets = set(case.expected_bucket_names)
    total_weight = 0.0
    matched_weight = 0.0
    for item in retrieved_context:
        weight = 1.0 / max(item.rank, 1)
        total_weight += weight
        record_match = bool(item.record_key and item.record_key in expected_records)
        bucket_match = bool(item.bucket_name and item.bucket_name in expected_buckets)
        if record_match or bucket_match:
            matched_weight += weight
    if total_weight == 0.0:
        return None
    if not expected_records and not expected_buckets:
        return 1.0
    return matched_weight / total_weight


def _redundancy_ratio(retrieved_context: tuple[RetrievedContext, ...]) -> float | None:
    if not retrieved_context:
        return None
    fingerprints = [
        (
            item.record_key,
            item.section_type,
            " ".join((item.chunk_text or "").lower().split())[:80],
        )
        for item in retrieved_context
    ]
    duplicates = len(fingerprints) - len(set(fingerprints))
    return duplicates / len(fingerprints)


def _duplicate_ratio(retrieved_context: tuple[RetrievedContext, ...]) -> float | None:
    if not retrieved_context:
        return None
    chunk_ids = [item.chunk_id for item in retrieved_context if item.chunk_id is not None]
    if not chunk_ids:
        return 0.0
    duplicates = len(chunk_ids) - len(set(chunk_ids))
    return duplicates / len(chunk_ids)


def _single_matter_purity(retrieved_context: tuple[RetrievedContext, ...]) -> float | None:
    record_keys = [item.record_key for item in retrieved_context if item.record_key]
    if not record_keys:
        return None
    counts = Counter(record_keys)
    return max(counts.values()) / sum(counts.values())


def _candidate_list_correctness(case: EvaluationCase, *, debug: dict[str, object]) -> float | None:
    if not debug.get("used"):
        return 1.0 if not case.must_clarify else 0.0
    candidate_record_keys = {
        str(item).strip() for item in debug.get("record_keys", []) if str(item).strip()
    }
    candidate_bucket_names = {
        str(item).strip() for item in debug.get("bucket_names", []) if str(item).strip()
    }
    if case.expected_record_keys:
        expected = set(case.expected_record_keys)
        return len(expected & candidate_record_keys) / len(expected)
    if case.expected_bucket_names:
        expected = set(case.expected_bucket_names)
        return len(expected & candidate_bucket_names) / len(expected)
    return 1.0


def _ordered_unique(values: object) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)
