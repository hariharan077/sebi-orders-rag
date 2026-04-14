"""Typed schemas for the SEBI evaluation engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_NUMERIC_TOLERANCE_BY_FACT_TYPE = {
    "listing_price": 0.01,
    "closing_price": 0.01,
    "highest_price": 0.01,
    "lowest_price": 0.01,
    "period_start_price": 0.01,
    "period_end_price": 0.01,
    "period_high_price": 0.01,
    "period_low_price": 0.01,
    "percentage_change": 0.05,
    "percentage_change_from_listing": 0.05,
    "holding_percentage": 0.05,
    "share_count": 0.5,
    "settlement_amount": 1.0,
    "penalty_amount": 1.0,
}


@dataclass(frozen=True)
class GoldNumericFact:
    """One deterministic numeric fact expectation for a case."""

    fact_type: str
    value_text: str | None = None
    value_numeric: float | None = None
    unit: str | None = None
    subject: str | None = None
    context_label: str | None = None
    tolerance_abs: float | None = None
    tolerance_pct: float | None = None

    def resolved_tolerance_abs(self) -> float | None:
        if self.tolerance_abs is not None:
            return self.tolerance_abs
        return _DEFAULT_NUMERIC_TOLERANCE_BY_FACT_TYPE.get(self.fact_type)


@dataclass(frozen=True)
class EvaluationCase:
    """Unified JSONL-friendly evaluation case."""

    case_id: str
    query: str
    session_id: str | None = None
    session_group: str | None = None
    reuse_previous_session: bool = False
    route_family_expected: str | None = None
    allowed_routes: tuple[str, ...] = ()
    expected_record_keys: tuple[str, ...] = ()
    expected_bucket_names: tuple[str, ...] = ()
    gold_answer_short: str | None = None
    gold_answer_long: str | None = None
    gold_numeric_facts: tuple[GoldNumericFact, ...] = ()
    must_abstain: bool = False
    must_clarify: bool = False
    must_use_active_matter: bool = False
    must_use_metadata: bool = False
    must_use_structured_current_info: bool = False
    must_use_official_web: bool = False
    must_not_use_web: bool = False
    tags: tuple[str, ...] = ()
    issue_class: str = "open_ended"
    difficulty: str = "medium"
    notes: str = ""
    prompt_family_expected: str | None = None
    expected_failure_buckets: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    source_case_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_numeric_facts"] = [asdict(item) for item in self.gold_numeric_facts]
        return payload


@dataclass(frozen=True)
class EvaluationDataset:
    """One loaded evaluation dataset."""

    name: str
    version: str
    cases: tuple[EvaluationCase, ...]
    source_files: tuple[str, ...] = ()
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "source_files": list(self.source_files),
            "metadata": dict(self.metadata),
            "case_count": len(self.cases),
        }


@dataclass(frozen=True)
class RetrievedContext:
    """One retrieved chunk or source used during evaluation."""

    rank: int
    chunk_id: int | None = None
    document_version_id: int | None = None
    record_key: str | None = None
    bucket_name: str | None = None
    title: str | None = None
    section_type: str | None = None
    score: float | None = None
    chunk_text: str | None = None
    source_url: str | None = None
    source_type: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class AssistantExecution:
    """Runtime execution output for one case."""

    case_id: str
    query: str
    route_mode: str
    query_intent: str
    answer_status: str
    answer_text: str
    confidence: float
    session_id: str | None
    citations: tuple[dict[str, Any], ...] = ()
    retrieved_chunk_ids: tuple[int, ...] = ()
    active_record_keys: tuple[str, ...] = ()
    retrieved_context: tuple[RetrievedContext, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)
    prompt_family: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "route_mode": self.route_mode,
            "query_intent": self.query_intent,
            "answer_status": self.answer_status,
            "answer_text": self.answer_text,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "citations": list(self.citations),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "active_record_keys": list(self.active_record_keys),
            "retrieved_context": [asdict(item) for item in self.retrieved_context],
            "debug": dict(self.debug),
            "prompt_family": self.prompt_family,
            "run_metadata": dict(self.run_metadata),
        }


@dataclass(frozen=True)
class JudgeScores:
    """Optional LLM-judge output."""

    context_relevance: float | None = None
    context_coverage: float | None = None
    faithfulness: float | None = None
    correctness: float | None = None
    reasoning_quality: float | None = None
    conciseness_relevance: float | None = None
    failure_modes: tuple[str, ...] = ()
    rationale: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_modes"] = list(self.failure_modes)
        return payload


@dataclass(frozen=True)
class RetrievalMetrics:
    """Retrieval-side deterministic metrics."""

    context_precision: float | None = None
    context_recall: float | None = None
    context_relevance: float | None = None
    redundancy_ratio: float | None = None
    duplicate_context_ratio: float | None = None
    single_matter_purity: float | None = None
    mixed_record_contamination: bool = False
    candidate_list_correctness: float | None = None
    expected_record_retrieved: bool = False
    expected_bucket_retrieved: bool = False
    retrieved_record_keys: tuple[str, ...] = ()
    retrieved_bucket_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrieved_record_keys"] = list(self.retrieved_record_keys)
        payload["retrieved_bucket_names"] = list(self.retrieved_bucket_names)
        return payload


@dataclass(frozen=True)
class GroundingMetrics:
    """Grounding, hallucination, and answer-quality metrics."""

    faithfulness: float | None = None
    hallucination_rate: float | None = None
    hallucination_detected: bool = False
    unsupported_claim_count: int = 0
    missing_critical_info_count: int = 0
    answer_correctness: float | None = None
    answer_relevance: float | None = None
    conciseness: float | None = None
    abstain_correct: bool | None = None
    clarify_correct: bool | None = None
    used_metadata_correctly: bool | None = None
    used_structured_current_info_correctly: bool | None = None
    used_official_web_correctly: bool | None = None


@dataclass(frozen=True)
class ReasoningMetrics:
    """Reasoning and judged answer quality metrics."""

    reasoning_quality: float | None = None
    explanation_quality: float | None = None
    summary_quality: float | None = None
    open_ended_answer_quality: float | None = None


@dataclass(frozen=True)
class RouteMetrics:
    """Planner and route-evaluation metrics."""

    strict_route_match: bool
    equivalent_route_match: bool
    equivalent_route_reason: str | None = None
    planner_choice_correct: bool | None = None
    internal_first_policy_correct: bool | None = None
    web_fallback_correct: bool | None = None
    active_matter_follow_up_correct: bool | None = None
    company_role_routing_correct: bool | None = None


@dataclass(frozen=True)
class NumericMetrics:
    """Deterministic numeric fact scoring."""

    expected_fact_count: int = 0
    matched_fact_count: int = 0
    numeric_accuracy: float | None = None
    missing_fact_types: tuple[str, ...] = ()
    mismatched_fact_types: tuple[str, ...] = ()
    matched_fact_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_fact_types"] = list(self.missing_fact_types)
        payload["mismatched_fact_types"] = list(self.mismatched_fact_types)
        payload["matched_fact_types"] = list(self.matched_fact_types)
        return payload


@dataclass(frozen=True)
class FailureModeMetrics:
    """Primary and secondary failure bucket assignments."""

    primary_bucket: str
    buckets: tuple[str, ...] = ()
    stale_expectation: bool = False
    true_bug: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["buckets"] = list(self.buckets)
        return payload


@dataclass(frozen=True)
class CalibrationMetrics:
    """Per-case calibration observations."""

    confidence: float
    confidence_bin: str
    correct: bool
    expected_abstain_or_clarify: bool
    predicted_abstain_or_clarify: bool


@dataclass(frozen=True)
class CaseEvaluationResult:
    """Complete evaluated case output."""

    case: EvaluationCase
    execution: AssistantExecution
    retrieval: RetrievalMetrics
    grounding: GroundingMetrics
    reasoning: ReasoningMetrics
    route: RouteMetrics
    numeric: NumericMetrics
    failure_modes: FailureModeMetrics
    calibration: CalibrationMetrics
    judge: JudgeScores | None = None
    passed: bool = False
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "execution": self.execution.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "grounding": asdict(self.grounding),
            "reasoning": asdict(self.reasoning),
            "route": asdict(self.route),
            "numeric": self.numeric.to_dict(),
            "failure_modes": self.failure_modes.to_dict(),
            "calibration": asdict(self.calibration),
            "judge": (self.judge.to_dict() if self.judge is not None else None),
            "passed": self.passed,
            "recorded_at": self.recorded_at,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case.case_id,
            "query": self.case.query,
            "issue_class": self.case.issue_class,
            "difficulty": self.case.difficulty,
            "route_family_expected": self.case.route_family_expected,
            "route_mode": self.execution.route_mode,
            "answer_status": self.execution.answer_status,
            "confidence": self.execution.confidence,
            "prompt_family": self.execution.prompt_family,
            "passed": self.passed,
            "primary_failure_bucket": self.failure_modes.primary_bucket,
            "stale_expectation": self.failure_modes.stale_expectation,
            "true_bug": self.failure_modes.true_bug,
            "strict_route_match": self.route.strict_route_match,
            "equivalent_route_match": self.route.equivalent_route_match,
            "equivalent_route_reason": self.route.equivalent_route_reason,
            "context_precision": self.retrieval.context_precision,
            "context_recall": self.retrieval.context_recall,
            "context_relevance": self.retrieval.context_relevance,
            "single_matter_purity": self.retrieval.single_matter_purity,
            "duplicate_context_ratio": self.retrieval.duplicate_context_ratio,
            "candidate_list_correctness": self.retrieval.candidate_list_correctness,
            "faithfulness": self.grounding.faithfulness,
            "hallucination_rate": self.grounding.hallucination_rate,
            "hallucination_detected": self.grounding.hallucination_detected,
            "unsupported_claim_count": self.grounding.unsupported_claim_count,
            "missing_critical_info_count": self.grounding.missing_critical_info_count,
            "answer_correctness": self.grounding.answer_correctness,
            "answer_relevance": self.grounding.answer_relevance,
            "conciseness": self.grounding.conciseness,
            "reasoning_quality": self.reasoning.reasoning_quality,
            "numeric_accuracy": self.numeric.numeric_accuracy,
            "expected_fact_count": self.numeric.expected_fact_count,
            "matched_fact_count": self.numeric.matched_fact_count,
            "confidence_bin": self.calibration.confidence_bin,
        }
        payload["tags"] = list(self.case.tags)
        payload["failure_buckets"] = list(self.failure_modes.buckets)
        payload["expected_record_keys"] = list(self.case.expected_record_keys)
        payload["active_record_keys"] = list(self.execution.active_record_keys)
        payload["retrieved_record_keys"] = list(self.retrieval.retrieved_record_keys)
        payload["retrieved_bucket_names"] = list(self.retrieval.retrieved_bucket_names)
        return payload


@dataclass(frozen=True)
class RunMetadata:
    """Metadata persisted alongside one benchmark run."""

    run_id: str
    timestamp: str
    dataset_name: str
    dataset_version: str
    dataset_files: tuple[str, ...]
    output_dir: str
    executor_mode: str
    assistant_model: str | None = None
    judge_model: str | None = None
    prompt_version: str | None = None
    retrieval_settings: dict[str, Any] = field(default_factory=dict)
    planner_settings: dict[str, Any] = field(default_factory=dict)
    git_commit_hash: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_files": list(self.dataset_files),
            "output_dir": self.output_dir,
            "executor_mode": self.executor_mode,
            "assistant_model": self.assistant_model,
            "judge_model": self.judge_model,
            "prompt_version": self.prompt_version,
            "retrieval_settings": dict(self.retrieval_settings),
            "planner_settings": dict(self.planner_settings),
            "git_commit_hash": self.git_commit_hash,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class EvaluationRunResult:
    """Summary plus detailed results for one run."""

    metadata: RunMetadata
    dataset: EvaluationDataset
    case_results: tuple[CaseEvaluationResult, ...]
    summary: dict[str, Any]
    comparison_summary: dict[str, Any] = field(default_factory=dict)


def normalize_path_strings(values: tuple[str | Path, ...]) -> tuple[str, ...]:
    """Return absolute string paths for JSON serialization."""

    normalized: list[str] = []
    for value in values:
        normalized.append(str(Path(value).expanduser().resolve(strict=False)))
    return tuple(normalized)
