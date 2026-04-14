"""Artifact persistence, bug-queue generation, and run comparison."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .report import (
    render_compare_summary_markdown,
    render_failures_markdown,
    render_true_bug_queue_markdown,
)
from .schemas import EvaluationRunResult
from .stats import (
    compare_result_frames,
    confidence_bin_frame,
    confusion_matrix_frame,
    metrics_by_prompt_family_frame,
    metrics_by_route_frame,
    metrics_by_tag_frame,
    results_to_frame,
    summarize_case_results,
)
from .triage import build_true_bug_queue


def persist_run_artifacts(
    *,
    run_result: EvaluationRunResult,
    output_root: str | Path,
    failure_dump_root: str | Path | None = None,
    control_pack_root: str | Path | None = None,
) -> Path:
    """Persist one benchmark run to a timestamped artifact directory."""

    root = Path(output_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_result.metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = results_to_frame(run_result.case_results)
    _write_json(run_dir / "run_config.json", run_result.metadata.to_dict())
    _write_json(run_dir / "summary.json", run_result.summary)
    _write_jsonl(run_dir / "per_case_results.jsonl", [item.to_dict() for item in run_result.case_results])
    _write_jsonl(
        run_dir / "failures.jsonl",
        [item.to_dict() for item in run_result.case_results if not item.passed],
    )
    (run_dir / "failures.md").write_text(
        render_failures_markdown(run_result.case_results),
        encoding="utf-8",
    )
    metrics_by_tag_frame(frame).to_csv(run_dir / "metrics_by_tag.csv", index=False)
    metrics_by_route_frame(frame).to_csv(run_dir / "metrics_by_route.csv", index=False)
    metrics_by_prompt_family_frame(frame).to_csv(
        run_dir / "metrics_by_prompt_family.csv",
        index=False,
    )
    confusion = confusion_matrix_frame(frame)
    if confusion.empty:
        pd.DataFrame().to_csv(run_dir / "confusion_matrix.csv", index=False)
    else:
        confusion.to_csv(run_dir / "confusion_matrix.csv")
    confidence_bin_frame(frame).to_csv(run_dir / "confidence_bins.csv", index=False)

    redteam_results = tuple(
        item for item in run_result.case_results if item.case.issue_class == "redteam"
    )
    redteam_summary = summarize_case_results(
        redteam_results,
        metadata={"subset": "redteam"},
    )
    _write_json(run_dir / "redteam_summary.json", redteam_summary)
    true_bug_queue = build_true_bug_queue(
        run_dir=run_dir,
        failure_dump_root=failure_dump_root,
        control_pack_root=control_pack_root,
    )
    _write_json(run_dir / "true_bug_queue.json", true_bug_queue)
    (run_dir / "true_bug_queue.md").write_text(
        render_true_bug_queue_markdown(true_bug_queue),
        encoding="utf-8",
    )
    return run_dir


def compare_run_summaries(
    *,
    base_run_dir: str | Path,
    head_run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare two persisted runs."""

    base_dir = Path(base_run_dir).expanduser().resolve(strict=False)
    head_dir = Path(head_run_dir).expanduser().resolve(strict=False)
    base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    head_summary = json.loads((head_dir / "summary.json").read_text(encoding="utf-8"))
    base_frame = _load_per_case_frame(base_dir / "per_case_results.jsonl")
    head_frame = _load_per_case_frame(head_dir / "per_case_results.jsonl")
    route_comparison = compare_result_frames(base_frame, head_frame)
    base_bug_queue = _load_bug_queue(base_dir)
    head_bug_queue = _load_bug_queue(head_dir)
    keys = (
        "passed_cases",
        "failed_cases",
        "strict_route_accuracy",
        "equivalent_route_accuracy",
        "faithfulness_average",
        "hallucination_rate",
        "numeric_fact_accuracy",
        "structured_current_info_accuracy",
        "candidate_list_correctness",
        "wrong_answer_regression_pass_count",
        "wrong_answer_regression_total",
        "true_bug_count",
    )
    delta = {
        key: round(float(head_summary.get(key, 0.0)) - float(base_summary.get(key, 0.0)), 4)
        for key in keys
    }
    wrong_answer_delta = _ratio_delta(
        numerator_key="wrong_answer_regression_pass_count",
        denominator_key="wrong_answer_regression_total",
        base_summary=base_summary,
        head_summary=head_summary,
    )
    comparison = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_run_dir": str(base_dir),
        "head_run_dir": str(head_dir),
        "metric_deltas": {
            **delta,
            "true_bug_count_delta": _bug_queue_delta(
                base_bug_queue=base_bug_queue,
                head_bug_queue=head_bug_queue,
                field="true_bug_count",
                fallback_base=int(base_summary.get("true_bug_count", 0)),
                fallback_head=int(head_summary.get("true_bug_count", 0)),
            ),
            "stale_expectation_delta": _bug_queue_delta(
                base_bug_queue=base_bug_queue,
                head_bug_queue=head_bug_queue,
                field="stale_expectation_count",
                fallback_base=int(base_summary.get("stale_expectation_count", 0)),
                fallback_head=int(head_summary.get("stale_expectation_count", 0)),
            ),
            "contamination_delta": round(
                float(head_summary.get("contamination_count", 0.0))
                - float(base_summary.get("contamination_count", 0.0)),
                4,
            ),
            "route_accuracy_delta": round(
                float(head_summary.get("strict_route_accuracy", 0.0))
                - float(base_summary.get("strict_route_accuracy", 0.0)),
                4,
            ),
            "equivalent_route_aware_accuracy_delta": round(
                float(head_summary.get("equivalent_route_accuracy", 0.0))
                - float(base_summary.get("equivalent_route_accuracy", 0.0)),
                4,
            ),
            "numeric_fact_accuracy_delta": round(
                float(head_summary.get("numeric_fact_accuracy", 0.0))
                - float(base_summary.get("numeric_fact_accuracy", 0.0)),
                4,
            ),
            "wrong_answer_regression_delta": wrong_answer_delta,
            "faithfulness_delta": round(
                float(head_summary.get("faithfulness_average", 0.0))
                - float(base_summary.get("faithfulness_average", 0.0)),
                4,
            ),
            "hallucination_delta": round(
                float(head_summary.get("hallucination_rate", 0.0))
                - float(base_summary.get("hallucination_rate", 0.0)),
                4,
            ),
        },
        "route_deltas": route_comparison.to_dict(orient="records"),
    }
    if output_dir is not None:
        target_dir = Path(output_dir).expanduser().resolve(strict=False)
    else:
        target_dir = head_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target_dir / "compare_summary.json", comparison)
    (target_dir / "compare_summary.md").write_text(
        render_compare_summary_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(_json_ready(row), sort_keys=True, allow_nan=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_per_case_frame(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        case = payload["case"]
        execution = payload["execution"]
        route = payload["route"]
        grounding = payload["grounding"]
        numeric = payload["numeric"]
        failure_modes = payload["failure_modes"]
        calibration = payload["calibration"]
        retrieval = payload["retrieval"]
        rows.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "issue_class": case["issue_class"],
                "difficulty": case["difficulty"],
                "route_family_expected": case.get("route_family_expected"),
                "route_mode": execution["route_mode"],
                "answer_status": execution["answer_status"],
                "confidence": calibration["confidence"],
                "passed": payload["passed"],
                "strict_route_match": route["strict_route_match"],
                "equivalent_route_match": route["equivalent_route_match"],
                "faithfulness": grounding["faithfulness"],
                "hallucination_rate": grounding["hallucination_rate"],
                "numeric_accuracy": numeric["numeric_accuracy"],
                "candidate_list_correctness": retrieval["candidate_list_correctness"],
                "primary_failure_bucket": failure_modes["primary_bucket"],
                "stale_expectation": failure_modes["stale_expectation"],
                "true_bug": failure_modes["true_bug"],
                "confidence_bin": calibration["confidence_bin"],
                "prompt_family": execution.get("prompt_family"),
                "expected_fact_count": numeric["expected_fact_count"],
                "tags": case.get("tags", []),
                "failure_buckets": failure_modes.get("buckets", []),
            }
        )
    return pd.DataFrame(rows)


def _load_bug_queue(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "true_bug_queue.json"
    if not path.exists():
        return None
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bug_queue_delta(
    *,
    base_bug_queue: dict[str, Any] | None,
    head_bug_queue: dict[str, Any] | None,
    field: str,
    fallback_base: int,
    fallback_head: int,
) -> int:
    base_value = int(base_bug_queue["summary"].get(field, fallback_base)) if base_bug_queue else fallback_base
    head_value = int(head_bug_queue["summary"].get(field, fallback_head)) if head_bug_queue else fallback_head
    return head_value - base_value


def _ratio_delta(
    *,
    numerator_key: str,
    denominator_key: str,
    base_summary: dict[str, Any],
    head_summary: dict[str, Any],
) -> float:
    base_denom = float(base_summary.get(denominator_key, 0) or 0.0)
    head_denom = float(head_summary.get(denominator_key, 0) or 0.0)
    base_ratio = (
        float(base_summary.get(numerator_key, 0) or 0.0) / base_denom
        if base_denom > 0
        else 0.0
    )
    head_ratio = (
        float(head_summary.get(numerator_key, 0) or 0.0) / head_denom
        if head_denom > 0
        else 0.0
    )
    return round(head_ratio - base_ratio, 4)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
