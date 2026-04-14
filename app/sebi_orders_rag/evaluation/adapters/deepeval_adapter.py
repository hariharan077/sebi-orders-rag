"""Optional DeepEval export adapter."""

from __future__ import annotations

from typing import Any

from ..schemas import CaseEvaluationResult, EvaluationCase


def export_results_to_deepeval(
    *,
    cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
    results: tuple[CaseEvaluationResult, ...] | list[CaseEvaluationResult] | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Export the internal schema to a DeepEval-friendly list of dicts."""

    result_by_id = {_result_case_id(item): item for item in (results or [])}
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = result_by_id.get(case.case_id)
        rows.append(
            {
                "input": case.query,
                "expected_output": case.gold_answer_long or case.gold_answer_short or "",
                "actual_output": _result_answer(result),
                "context": [
                    _context_text(item)
                    for item in _result_contexts(result)
                ],
                "metadata": case.to_dict(),
            }
        )
    return rows


def run_deepeval_if_available(
    *,
    rows: list[dict[str, Any]],
) -> Any:
    """Run DeepEval only if the package is installed."""

    try:
        import deepeval  # type: ignore  # pragma: no cover - optional dependency
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("deepeval is not installed.") from exc
    return {"deepeval_version": getattr(deepeval, "__version__", "unknown"), "rows": rows}


def _result_case_id(result: CaseEvaluationResult | dict[str, Any]) -> str:
    if isinstance(result, dict):
        return str(result.get("case", {}).get("case_id", "")).strip()
    return result.case.case_id


def _result_contexts(result: CaseEvaluationResult | dict[str, Any] | None) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, dict):
        return list(result.get("execution", {}).get("retrieved_context", []))
    return list(result.execution.retrieved_context)


def _context_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("chunk_text") or "")
    return str(getattr(item, "chunk_text", "") or "")


def _result_answer(result: CaseEvaluationResult | dict[str, Any] | None) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        return str(result.get("execution", {}).get("answer_text") or "")
    return result.execution.answer_text
