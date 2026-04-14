#!/usr/bin/env python3
"""Fail a SEBI eval run when configured quality thresholds are not met."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.evaluation.triage import build_true_bug_queue, resolve_latest_run_dir

DEFAULT_GATE_CONFIG: dict[str, float | int] = {
    "mixed_record_contamination_count_max": 0,
    "wrong_answer_regression_pass_rate_min": 1.0,
    "structured_current_info_accuracy_min": 0.9,
    "numeric_fact_accuracy_min": 0.9,
    "equivalent_route_accuracy_min": 0.85,
    "candidate_list_correctness_min": 0.95,
    "request_failure_surface_count_max": 0,
    "true_bug_count_max": 0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sebi_eval_runs",
        help="Artifact root containing eval runs.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Optional JSON file overriding gate thresholds.",
    )
    parser.add_argument(
        "--failure-dump-root",
        type=Path,
        help="Optional failure dump root for bug-queue generation.",
    )
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "control_pack",
    )
    return parser


def evaluate_release_gate(
    *,
    run_dir: str | Path,
    config: dict[str, float | int] | None = None,
    failure_dump_root: str | Path | None = None,
    control_pack_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate all release-gate checks for one persisted run."""

    run_path = Path(run_dir).expanduser().resolve(strict=False)
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    bug_queue_path = run_path / "true_bug_queue.json"
    bug_queue = (
        json.loads(bug_queue_path.read_text(encoding="utf-8"))
        if bug_queue_path.exists()
        else build_true_bug_queue(
            run_dir=run_path,
            failure_dump_root=failure_dump_root,
            control_pack_root=control_pack_root,
        )
    )
    per_case_results = _load_jsonl(run_path / "per_case_results.jsonl")
    thresholds = dict(DEFAULT_GATE_CONFIG)
    thresholds.update(config or {})

    wrong_answer_pass_rate = _ratio(
        numerator=float(summary.get("wrong_answer_regression_pass_count", 0) or 0.0),
        denominator=float(summary.get("wrong_answer_regression_total", 0) or 0.0),
        empty_value=1.0,
    )
    request_failure_surface_count = sum(
        1
        for row in per_case_results
        if str(row.get("execution", {}).get("answer_status") or "") == "request_failed"
        or bool(row.get("execution", {}).get("debug", {}).get("failure_safe", {}).get("used"))
    )
    observed = {
        "mixed_record_contamination_count": int(summary.get("contamination_count", 0) or 0),
        "wrong_answer_regression_pass_rate": wrong_answer_pass_rate,
        "structured_current_info_accuracy": float(summary.get("structured_current_info_accuracy", 0.0) or 0.0),
        "numeric_fact_accuracy": float(summary.get("numeric_fact_accuracy", 0.0) or 0.0),
        "equivalent_route_accuracy": float(summary.get("equivalent_route_accuracy", 0.0) or 0.0),
        "candidate_list_correctness": float(summary.get("candidate_list_correctness", 0.0) or 0.0),
        "request_failure_surface_count": request_failure_surface_count,
        "true_bug_count": int(
            bug_queue.get("summary", {}).get("true_bug_count", summary.get("true_bug_count", 0))
        ),
    }
    checks = [
        _check_max(
            "mixed_record_contamination_count",
            observed["mixed_record_contamination_count"],
            int(thresholds["mixed_record_contamination_count_max"]),
        ),
        _check_min(
            "wrong_answer_regression_pass_rate",
            observed["wrong_answer_regression_pass_rate"],
            float(thresholds["wrong_answer_regression_pass_rate_min"]),
        ),
        _check_min(
            "structured_current_info_accuracy",
            observed["structured_current_info_accuracy"],
            float(thresholds["structured_current_info_accuracy_min"]),
        ),
        _check_min(
            "numeric_fact_accuracy",
            observed["numeric_fact_accuracy"],
            float(thresholds["numeric_fact_accuracy_min"]),
        ),
        _check_min(
            "equivalent_route_accuracy",
            observed["equivalent_route_accuracy"],
            float(thresholds["equivalent_route_accuracy_min"]),
        ),
        _check_min(
            "candidate_list_correctness",
            observed["candidate_list_correctness"],
            float(thresholds["candidate_list_correctness_min"]),
        ),
        _check_max(
            "request_failure_surface_count",
            observed["request_failure_surface_count"],
            int(thresholds["request_failure_surface_count_max"]),
        ),
        _check_max(
            "true_bug_count",
            observed["true_bug_count"],
            int(thresholds["true_bug_count_max"]),
        ),
    ]
    passed = all(check["passed"] for check in checks)
    payload = {
        "run_dir": str(run_path),
        "passed": passed,
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "source_refs": {
            "summary_json": str(run_path / "summary.json"),
            "per_case_results_jsonl": str(run_path / "per_case_results.jsonl"),
            "true_bug_queue_json": str(run_path / "true_bug_queue.json"),
        },
    }
    (run_path / "release_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_path / "release_gate.md").write_text(
        _render_release_gate_markdown(payload),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = build_parser().parse_args()
    run_dir = (
        args.run_dir.expanduser().resolve(strict=False)
        if args.run_dir is not None
        else resolve_latest_run_dir(args.runs_root)
    )
    config = (
        json.loads(args.config_json.read_text(encoding="utf-8"))
        if args.config_json is not None
        else None
    )
    payload = evaluate_release_gate(
        run_dir=run_dir,
        config=config,
        failure_dump_root=args.failure_dump_root,
        control_pack_root=args.control_pack_root,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


def _check_min(name: str, observed: float, threshold: float) -> dict[str, Any]:
    return {
        "metric": name,
        "operator": ">=",
        "threshold": threshold,
        "observed": observed,
        "passed": observed >= threshold,
    }


def _check_max(name: str, observed: int, threshold: int) -> dict[str, Any]:
    return {
        "metric": name,
        "operator": "<=",
        "threshold": threshold,
        "observed": observed,
        "passed": observed <= threshold,
    }


def _ratio(*, numerator: float, denominator: float, empty_value: float) -> float:
    if denominator <= 0:
        return empty_value
    return numerator / denominator


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _render_release_gate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Release Gate",
        "",
        f"- Run dir: `{payload['run_dir']}`",
        f"- Passed: `{payload['passed']}`",
        "",
        "## Checks",
        "",
    ]
    for check in payload["checks"]:
        lines.append(
            f"- {check['metric']}: observed `{check['observed']}` {check['operator']} threshold `{check['threshold']}` -> `{check['passed']}`"
        )
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
