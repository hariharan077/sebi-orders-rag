"""Execution runner for SEBI evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from ..answering.answer_service import AdaptiveRagAnswerService
from ..schemas import ChatAnswerPayload
from .benchmark import persist_run_artifacts
from .judge_llm import JudgeClient, NoopJudgeClient
from .metrics_calibration import build_calibration_metrics
from .metrics_failure_modes import classify_failure_modes
from .metrics_grounding import evaluate_grounding_metrics
from .metrics_numeric import evaluate_numeric_metrics
from .metrics_reasoning import evaluate_reasoning_metrics
from .metrics_retrieval import evaluate_retrieval_metrics
from .metrics_route import evaluate_route_metrics
from .report import render_run_summary
from .schemas import (
    AssistantExecution,
    CaseEvaluationResult,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRunResult,
    JudgeScores,
    RetrievedContext,
    RunMetadata,
)
from .stats import summarize_case_results


class EvaluationExecutor(Protocol):
    """Executor protocol used by the runner."""

    mode: str

    def execute(
        self,
        *,
        case: EvaluationCase,
        session_id: UUID | None,
    ) -> AssistantExecution:
        raise NotImplementedError


class LiveAssistantExecutor:
    """Execute cases against the live adaptive assistant."""

    mode = "live"

    def __init__(
        self,
        *,
        service: AdaptiveRagAnswerService,
        connection: Any | None = None,
    ) -> None:
        self._service = service
        self._connection = connection or getattr(service, "_connection", None)

    def execute(
        self,
        *,
        case: EvaluationCase,
        session_id: UUID | None,
    ) -> AssistantExecution:
        payload = self._service.answer_query(query=case.query, session_id=session_id)
        citations = tuple(_citation_dict(item) for item in payload.citations)
        retrieved_context = self._load_retrieved_context(payload)
        prompt_family = infer_prompt_family(case=case, payload=payload)
        return AssistantExecution(
            case_id=case.case_id,
            query=case.query,
            route_mode=payload.route_mode,
            query_intent=payload.query_intent,
            answer_status=payload.answer_status,
            answer_text=payload.answer_text,
            confidence=payload.confidence,
            session_id=str(payload.session_id),
            citations=citations,
            retrieved_chunk_ids=tuple(payload.retrieved_chunk_ids),
            active_record_keys=tuple(payload.active_record_keys),
            retrieved_context=retrieved_context,
            debug=dict(payload.debug),
            prompt_family=prompt_family,
            run_metadata={},
        )

    def _load_retrieved_context(self, payload: ChatAnswerPayload) -> tuple[RetrievedContext, ...]:
        if self._connection is not None and payload.retrieved_chunk_ids:
            rows = _query_context_rows(self._connection, payload.retrieved_chunk_ids)
            if rows:
                return tuple(
                    RetrievedContext(
                        rank=index,
                        chunk_id=row["chunk_id"],
                        document_version_id=row["document_version_id"],
                        record_key=row["record_key"],
                        bucket_name=row["bucket_name"],
                        title=row["title"],
                        section_type=row["section_type"],
                        score=row["score"],
                        chunk_text=row["chunk_text"],
                        source_url=row["detail_url"] or row["pdf_url"],
                        source_type="corpus_chunk",
                        domain=None,
                    )
                    for index, row in enumerate(rows, start=1)
                )

        search_debug = dict(payload.debug.get("search_debug", {}) or {})
        top_chunks = search_debug.get("top_chunks", [])
        if isinstance(top_chunks, list):
            return tuple(
                RetrievedContext(
                    rank=index,
                    chunk_id=_as_int(item.get("chunk_id")),
                    document_version_id=_as_int(item.get("document_version_id")),
                    record_key=_clean_text(item.get("record_key")),
                    bucket_name=_clean_text(item.get("bucket_name")),
                    title=_clean_text(item.get("title")),
                    section_type=_clean_text(item.get("section_type")),
                    score=_as_float(item.get("final_score")),
                    chunk_text=None,
                    source_url=_clean_text(item.get("detail_url")) or _clean_text(item.get("pdf_url")),
                    source_type="corpus_chunk",
                )
                for index, item in enumerate(top_chunks, start=1)
                if isinstance(item, dict)
            )
        return ()


class ReplayExecutor:
    """Deterministic executor for tests and local smoke runs."""

    mode = "replay"

    def execute(
        self,
        *,
        case: EvaluationCase,
        session_id: UUID | None,
    ) -> AssistantExecution:
        route_mode = _resolve_replay_route(case)
        answer_status = _resolve_replay_answer_status(case)
        answer_text = _build_replay_answer(case)
        citations = _build_replay_citations(case)
        prompt_family = _resolve_replay_prompt_family(case, route_mode=route_mode)
        debug = {
            "route_debug": {
                "strict_scope_required": bool(case.expected_record_keys),
                "strict_single_matter": bool(case.expected_record_keys and len(case.expected_record_keys) == 1),
                "appears_structured_current_info": case.must_use_structured_current_info,
                "appears_current_official_lookup": case.must_use_official_web,
                "appears_current_news_lookup": "current_news" in case.tags,
                "active_order_override": case.must_use_active_matter,
                "asks_brief_summary": prompt_family == "brief summary",
                "appears_sat_court_style": "sat" in " ".join(case.tags).lower(),
            },
            "planner_debug": {
                "used": True,
                "execution_route_mode": route_mode,
            },
            "metadata_debug": {
                "used": case.must_use_metadata,
                "metadata_type": "replay",
            },
            "candidate_list_debug": {
                "used": case.must_clarify,
                "record_keys": list(case.expected_record_keys),
                "bucket_names": list(case.expected_bucket_names),
            },
            "web_fallback_debug": {
                "official_web_attempted": case.must_use_official_web,
                "general_web_attempted": False,
            },
        }
        retrieved_context = tuple(
            RetrievedContext(
                rank=index,
                record_key=record_key,
                bucket_name=(case.expected_bucket_names[0] if case.expected_bucket_names else None),
                title=(case.metadata.get("expected_titles", [None])[0] if case.metadata else None),
                section_type="operative_order",
                score=1.0 / index,
                chunk_text=answer_text,
                source_type="corpus_chunk",
            )
            for index, record_key in enumerate(case.expected_record_keys or ("replay:unknown",), start=1)
        )
        return AssistantExecution(
            case_id=case.case_id,
            query=case.query,
            route_mode=route_mode,
            query_intent=case.issue_class,
            answer_status=answer_status,
            answer_text=answer_text,
            confidence=0.95 if answer_status == "answered" else 0.15,
            session_id=str(session_id or uuid4()),
            citations=tuple(citations),
            retrieved_chunk_ids=(),
            active_record_keys=tuple(case.expected_record_keys),
            retrieved_context=retrieved_context,
            debug=debug,
            prompt_family=prompt_family,
            run_metadata={"replay": True},
        )


class EvaluationRunner:
    """Run one dataset through an executor and optional judge."""

    def __init__(
        self,
        *,
        executor: EvaluationExecutor,
        judge_client: JudgeClient | None = None,
    ) -> None:
        self._executor = executor
        self._judge = judge_client or NoopJudgeClient()

    def run(
        self,
        *,
        dataset: EvaluationDataset,
        metadata: RunMetadata,
        output_root: str | Path | None = None,
    ) -> EvaluationRunResult:
        session_by_group: dict[str, UUID] = {}
        previous_session_id: UUID | None = None
        case_results: list[CaseEvaluationResult] = []

        for case in dataset.cases:
            session_id = _resolve_session_id(
                case=case,
                session_by_group=session_by_group,
                previous_session_id=previous_session_id,
            )
            execution = self._executor.execute(case=case, session_id=session_id)
            resolved_session_id = (
                UUID(execution.session_id) if execution.session_id else session_id or uuid4()
            )
            previous_session_id = resolved_session_id
            if case.session_group:
                session_by_group[case.session_group] = resolved_session_id

            case_results.append(self._evaluate_case(case=case, execution=execution))

        summary = summarize_case_results(case_results, metadata=metadata.to_dict())
        run_result = EvaluationRunResult(
            metadata=metadata,
            dataset=dataset,
            case_results=tuple(case_results),
            summary=summary,
        )
        if output_root is not None:
            persist_run_artifacts(run_result=run_result, output_root=output_root)
        return run_result

    def _evaluate_case(
        self,
        *,
        case: EvaluationCase,
        execution: AssistantExecution,
    ) -> CaseEvaluationResult:
        judge = self._judge.evaluate(
            case=case,
            retrieved_context=execution.retrieved_context,
            route_mode=execution.route_mode,
            answer_status=execution.answer_status,
            answer_text=execution.answer_text,
        )
        actual_record_keys = tuple(
            dict.fromkeys(
                [
                    *execution.active_record_keys,
                    *[
                        _clean_text(item.get("record_key"))
                        for item in execution.citations
                        if _clean_text(item.get("record_key"))
                    ],
                ]
            )
        )
        retrieval = evaluate_retrieval_metrics(
            case=case,
            retrieved_context=execution.retrieved_context,
            debug=execution.debug,
        )
        numeric = evaluate_numeric_metrics(
            case=case,
            answer_text=execution.answer_text,
        )
        grounding = evaluate_grounding_metrics(
            case=case,
            route_mode=execution.route_mode,
            answer_status=execution.answer_status,
            answer_text=execution.answer_text,
            actual_record_keys=actual_record_keys,
            citations=execution.citations,
            debug=execution.debug,
            retrieval=retrieval,
            numeric=numeric,
            judge=judge,
        )
        route = evaluate_route_metrics(
            case=case,
            route_mode=execution.route_mode,
            answer_status=execution.answer_status,
            debug=execution.debug,
            actual_record_keys=actual_record_keys,
        )
        reasoning = evaluate_reasoning_metrics(
            case=case,
            answer_text=execution.answer_text,
            judge=judge,
        )
        provisional_pass = _case_passes(
            case=case,
            route=route,
            grounding=grounding,
            numeric=numeric,
            answer_status=execution.answer_status,
        )
        calibration = build_calibration_metrics(
            case=case,
            confidence=execution.confidence,
            passed=provisional_pass,
            answer_status=execution.answer_status,
        )
        failure_modes = classify_failure_modes(
            case=case,
            route_mode=execution.route_mode,
            route=route,
            retrieval=retrieval,
            grounding=grounding,
            numeric=numeric,
            answer_status=execution.answer_status,
            answer_text=execution.answer_text,
            debug=execution.debug,
            judge=judge,
        )
        approved_expectation_match = _matches_approved_expectation_update(
            case=case,
            route=route,
            retrieval=retrieval,
            grounding=grounding,
            answer_status=execution.answer_status,
            actual_record_keys=actual_record_keys,
        )
        if approved_expectation_match:
            failure_modes = replace(
                failure_modes,
                primary_bucket="stale expectation",
                buckets=tuple(dict.fromkeys(("stale expectation", *failure_modes.buckets))),
                stale_expectation=True,
                true_bug=False,
            )
        passed = approved_expectation_match or (
            provisional_pass and failure_modes.primary_bucket in {"pass", "stale expectation"}
        )
        return CaseEvaluationResult(
            case=case,
            execution=execution,
            retrieval=retrieval,
            grounding=grounding,
            reasoning=reasoning,
            route=route,
            numeric=numeric,
            failure_modes=replace(
                failure_modes,
                true_bug=(failure_modes.true_bug and not passed),
            ),
            calibration=replace(calibration, correct=passed),
            judge=judge,
            passed=passed,
        )


def build_run_metadata(
    *,
    dataset: EvaluationDataset,
    executor_mode: str,
    output_root: str | Path,
    assistant_model: str | None = None,
    judge_model: str | None = None,
    prompt_version: str | None = None,
    retrieval_settings: dict[str, Any] | None = None,
    planner_settings: dict[str, Any] | None = None,
    git_commit_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RunMetadata:
    """Build stable run metadata."""

    timestamp = _timestamp()
    run_id = f"sebi_eval_run_{timestamp}"
    return RunMetadata(
        run_id=run_id,
        timestamp=timestamp,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_files=tuple(dataset.source_files),
        output_dir=str(Path(output_root).expanduser().resolve(strict=False) / run_id),
        executor_mode=executor_mode,
        assistant_model=assistant_model,
        judge_model=judge_model,
        prompt_version=prompt_version,
        retrieval_settings=dict(retrieval_settings or {}),
        planner_settings=dict(planner_settings or {}),
        git_commit_hash=git_commit_hash,
        extra=dict(extra or {}),
    )


def infer_prompt_family(*, case: EvaluationCase, payload: ChatAnswerPayload) -> str:
    """Infer the prompt family used for one payload."""

    debug = dict(payload.debug)
    route_debug = dict(debug.get("route_debug", {}) or {})
    metadata_debug = dict(debug.get("metadata_debug", {}) or {})
    if payload.answer_status == "clarify":
        return "clarify"
    if payload.route_mode == "structured_current_info":
        return "structured current-info"
    if metadata_debug.get("used"):
        if route_debug.get("asks_provision_explanation"):
            return "explain provision"
        return "metadata-first fact"
    if payload.route_mode in {"current_official_lookup", "historical_official_lookup", "current_news_lookup"}:
        return "current fact"
    if route_debug.get("appears_sat_court_style"):
        return "SAT/court summary"
    if "settlement" in " ".join(case.tags).lower() or any(
        "settlement" in value.lower() for value in case.expected_bucket_names
    ):
        return "settlement-shaped answer"
    if route_debug.get("asks_brief_summary"):
        return "brief summary"
    return "brief summary"


def _case_passes(
    *,
    case: EvaluationCase,
    route,
    grounding,
    numeric,
    answer_status: str,
) -> bool:
    if case.must_abstain:
        return answer_status in {"abstained", "clarify"}
    if case.must_clarify:
        return answer_status == "clarify"
    if not route.equivalent_route_match:
        return False
    if grounding.answer_correctness is not None and grounding.answer_correctness < 0.7:
        return False
    if grounding.faithfulness is not None and grounding.faithfulness < 0.7:
        return False
    if numeric.expected_fact_count and (numeric.numeric_accuracy or 0.0) < 1.0:
        return False
    return answer_status == "answered"


def _matches_approved_expectation_update(
    *,
    case: EvaluationCase,
    route,
    retrieval,
    grounding,
    answer_status: str,
    actual_record_keys: tuple[str, ...],
) -> bool:
    expectation_update = dict(case.metadata.get("expectation_update", {}) or {})
    if not expectation_update.get("approved"):
        return False
    if retrieval.mixed_record_contamination or grounding.hallucination_detected:
        return False
    reason = str(expectation_update.get("reason") or "").strip()
    expected_record_keys = set(case.expected_record_keys)
    actual_record_key_set = set(actual_record_keys)

    if reason == "equivalent acceptable route":
        if answer_status != "answered" or not route.equivalent_route_match:
            return False
        return not expected_record_keys or actual_record_key_set == expected_record_keys

    if reason == "safer clarify behavior":
        return answer_status in {"clarify", "abstained"}

    if reason == "better grounded answer than prior expectation":
        if answer_status == "abstained":
            return True
        if answer_status != "answered":
            return False
        if expected_record_keys and actual_record_key_set != expected_record_keys:
            return False
        if expected_record_keys and not actual_record_key_set:
            return False
        if grounding.faithfulness is not None and grounding.faithfulness < 0.7:
            return False
        return bool(actual_record_key_set) or route.equivalent_route_match

    return False


def _resolve_session_id(
    *,
    case: EvaluationCase,
    session_by_group: dict[str, UUID],
    previous_session_id: UUID | None,
) -> UUID | None:
    if case.session_id:
        try:
            return UUID(case.session_id)
        except ValueError:
            return uuid4()
    if case.session_group and case.session_group in session_by_group:
        return session_by_group[case.session_group]
    if case.reuse_previous_session:
        return previous_session_id
    return None


def _build_replay_answer(case: EvaluationCase) -> str:
    if case.must_abstain:
        return "I cannot safely answer that from the available grounded evidence."
    if case.must_clarify:
        return "I found multiple plausible matches. Please clarify the exact matter."
    if case.gold_answer_short:
        return case.gold_answer_short
    if case.gold_numeric_facts:
        parts = [
            str(item.value_text or item.value_numeric)
            for item in case.gold_numeric_facts
            if item.value_text or item.value_numeric is not None
        ]
        return "The relevant values are " + ", ".join(parts) + "."
    return f"Replay answer for {case.query}"


def _build_replay_citations(case: EvaluationCase) -> list[dict[str, Any]]:
    if case.must_abstain or case.must_clarify:
        return []
    citations: list[dict[str, Any]] = []
    for index, record_key in enumerate(case.expected_record_keys, start=1):
        citations.append(
            {
                "citation_number": index,
                "record_key": record_key,
                "title": (case.metadata.get("expected_titles", [None])[0] if case.metadata else None),
                "source_type": "corpus_chunk",
            }
        )
    return citations


def _resolve_replay_route(case: EvaluationCase) -> str:
    if case.route_family_expected:
        return case.route_family_expected
    if case.must_clarify:
        return "clarify"
    if case.must_abstain:
        return "abstain"
    return "hierarchical_rag"


def _resolve_replay_answer_status(case: EvaluationCase) -> str:
    if case.must_clarify:
        return "clarify"
    if case.must_abstain:
        return "abstained"
    return "answered"


def _resolve_replay_prompt_family(case: EvaluationCase, *, route_mode: str) -> str:
    if case.prompt_family_expected:
        return case.prompt_family_expected
    if route_mode == "structured_current_info":
        return "structured current-info"
    if case.must_use_metadata:
        return "metadata-first fact"
    if route_mode in {"current_official_lookup", "historical_official_lookup", "current_news_lookup"}:
        return "current fact"
    return "brief summary"


def _citation_dict(citation: object) -> dict[str, Any]:
    return {
        "citation_number": getattr(citation, "citation_number", None),
        "record_key": getattr(citation, "record_key", None),
        "title": getattr(citation, "title", None),
        "page_start": getattr(citation, "page_start", None),
        "page_end": getattr(citation, "page_end", None),
        "section_type": getattr(citation, "section_type", None),
        "document_version_id": getattr(citation, "document_version_id", None),
        "chunk_id": getattr(citation, "chunk_id", None),
        "detail_url": getattr(citation, "detail_url", None),
        "pdf_url": getattr(citation, "pdf_url", None),
        "source_url": getattr(citation, "source_url", None),
        "domain": getattr(citation, "domain", None),
        "source_type": getattr(citation, "source_type", None),
        "snippet": getattr(citation, "snippet", None),
    }


def _query_context_rows(connection: Any, chunk_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    unique_chunk_ids = list(dict.fromkeys(int(chunk_id) for chunk_id in chunk_ids))
    sql = """
        SELECT
            dc.chunk_id,
            dc.document_version_id,
            sd.record_key,
            sd.bucket_name,
            dv.title,
            dc.section_type,
            dc.chunk_text,
            dv.detail_url,
            dv.pdf_url
        FROM document_chunks dc
        INNER JOIN document_versions dv
            ON dv.document_version_id = dc.document_version_id
        INNER JOIN source_documents sd
            ON sd.document_id = dv.document_id
        WHERE dc.chunk_id = ANY(%s)
        ORDER BY array_position(%s::bigint[], dc.chunk_id)
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, (unique_chunk_ids, unique_chunk_ids))
        rows = cursor.fetchall()
    return [
        {
            "chunk_id": int(row[0]),
            "document_version_id": int(row[1]),
            "record_key": str(row[2]),
            "bucket_name": str(row[3]),
            "title": str(row[4]),
            "section_type": str(row[5]),
            "chunk_text": str(row[6] or ""),
            "detail_url": row[7],
            "pdf_url": row[8],
            "score": None,
        }
        for row in rows
    ]


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S_%f")


def _as_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: object) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None
