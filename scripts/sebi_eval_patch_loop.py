#!/usr/bin/env python3
"""Run a focused burn-down loop for selected SEBI eval bug clusters or case ids."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.evaluation.benchmark import compare_run_summaries
from app.sebi_orders_rag.evaluation.dataset import load_cases, write_cases
from app.sebi_orders_rag.evaluation.triage import build_true_bug_queue, resolve_latest_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", action="append", default=[], help="Bug-cluster name to burn down.")
    parser.add_argument("--case-id", action="append", default=[], help="Specific case id to include.")
    parser.add_argument("--base-run-dir", type=Path, help="Existing eval run directory to compare against.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sebi_eval_runs",
        help="Root containing eval run directories.",
    )
    parser.add_argument("--dataset-jsonl", type=Path, help="Explicit dataset JSONL override.")
    parser.add_argument(
        "--executor",
        choices=("auto", "live", "replay"),
        default="auto",
        help="Execution backend for the focused subset eval.",
    )
    parser.add_argument(
        "--judge",
        choices=("none", "openai", "auto"),
        default="none",
    )
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--failure-dump-root",
        type=Path,
        help="Optional failure dump root for refreshed bug-queue generation.",
    )
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "control_pack",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip focused regression tests and only run the focused eval subset.",
    )
    parser.add_argument(
        "--pytest-target",
        action="append",
        default=[],
        help="Optional explicit pytest target(s). Overrides auto-selected targets when provided.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_run_dir = (
        args.base_run_dir.expanduser().resolve(strict=False)
        if args.base_run_dir is not None
        else resolve_latest_run_dir(args.runs_root)
    )
    bug_queue = _load_or_build_bug_queue(
        run_dir=base_run_dir,
        failure_dump_root=args.failure_dump_root,
        control_pack_root=args.control_pack_root,
    )
    selected_case_ids = _resolve_case_ids(
        bug_queue=bug_queue,
        clusters=args.cluster,
        explicit_case_ids=args.case_id,
    )
    if not selected_case_ids:
        raise SystemExit("No case ids resolved for the requested patch loop.")

    dataset_path = _resolve_dataset_path(
        base_run_dir=base_run_dir,
        dataset_override=args.dataset_jsonl,
    )
    cases = load_cases(dataset_path)
    selected_cases = [case for case in cases if case.case_id in selected_case_ids]
    if not selected_cases:
        raise SystemExit("Resolved case ids are not present in the selected dataset.")

    loop_dir = _make_loop_dir()
    loop_dir.mkdir(parents=True, exist_ok=True)
    subset_dataset_path = loop_dir / "focused_subset.jsonl"
    write_cases(subset_dataset_path, selected_cases)

    test_targets = (
        tuple(args.pytest_target)
        if args.pytest_target
        else _resolve_pytest_targets(bug_queue=bug_queue, case_ids=selected_case_ids, clusters=args.cluster)
    )
    test_result = None
    if not args.skip_tests and test_targets:
        test_result = _run_pytest(test_targets)

    run_result = _run_focused_eval(
        subset_dataset_path=subset_dataset_path,
        runs_root=args.runs_root,
        executor=args.executor,
        judge=args.judge,
        judge_model=args.judge_model,
    )
    head_run_dir = Path(run_result["artifact_dir"]).expanduser().resolve(strict=False)
    head_bug_queue = _load_or_build_bug_queue(
        run_dir=head_run_dir,
        failure_dump_root=args.failure_dump_root,
        control_pack_root=args.control_pack_root,
    )
    baseline_subset_dir = _materialize_base_subset_artifact(
        base_run_dir=base_run_dir,
        loop_dir=loop_dir,
        selected_case_ids=selected_case_ids,
        bug_queue=bug_queue,
    )
    compare_payload = compare_run_summaries(
        base_run_dir=baseline_subset_dir,
        head_run_dir=head_run_dir,
        output_dir=loop_dir,
    )
    summary = _build_patch_loop_summary(
        base_bug_queue=bug_queue,
        head_bug_queue=head_bug_queue,
        selected_case_ids=selected_case_ids,
        base_run_dir=base_run_dir,
        head_run_dir=head_run_dir,
        loop_dir=loop_dir,
        test_result=test_result,
        compare_payload=compare_payload,
    )
    (loop_dir / "patch_loop_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (loop_dir / "patch_loop_summary.md").write_text(
        _render_patch_loop_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["cases_regressed"] == [] else 1


def _load_or_build_bug_queue(
    *,
    run_dir: Path,
    failure_dump_root: str | Path | None,
    control_pack_root: str | Path | None,
) -> dict[str, Any]:
    path = run_dir / "true_bug_queue.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = build_true_bug_queue(
        run_dir=run_dir,
        failure_dump_root=failure_dump_root,
        control_pack_root=control_pack_root,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _resolve_case_ids(
    *,
    bug_queue: dict[str, Any],
    clusters: list[str],
    explicit_case_ids: list[str],
) -> tuple[str, ...]:
    selected: list[str] = list(explicit_case_ids)
    cluster_lookup = {
        cluster["cluster"]: tuple(cluster["affected_case_ids"])
        for cluster in bug_queue.get("clusters", [])
    }
    for cluster_name in clusters:
        selected.extend(cluster_lookup.get(cluster_name, ()))
    return tuple(dict.fromkeys(selected))


def _resolve_dataset_path(*, base_run_dir: Path, dataset_override: Path | None) -> Path:
    if dataset_override is not None:
        return dataset_override.expanduser().resolve(strict=False)
    run_config_path = base_run_dir / "run_config.json"
    if run_config_path.exists():
        config = json.loads(run_config_path.read_text(encoding="utf-8"))
        dataset_files = config.get("dataset_files", []) or []
        if dataset_files:
            return Path(dataset_files[0]).expanduser().resolve(strict=False)
    raise FileNotFoundError("Could not resolve a dataset JSONL for the focused patch loop.")


def _resolve_pytest_targets(
    *,
    bug_queue: dict[str, Any],
    case_ids: tuple[str, ...],
    clusters: list[str],
) -> tuple[str, ...]:
    targets: list[str] = []
    cluster_targets = {
        cluster["cluster"]: tuple(cluster["expected_regression_tests"])
        for cluster in bug_queue.get("clusters", [])
    }
    for cluster_name in clusters:
        targets.extend(cluster_targets.get(cluster_name, ()))
    entry_targets = {
        entry["case_id"]: tuple(entry["expected_regression_tests"])
        for entry in bug_queue.get("entries", [])
    }
    for case_id in case_ids:
        targets.extend(entry_targets.get(case_id, ()))
    if not targets:
        targets.append("tests/sebi_orders_rag/test_true_bug_regressions.py")
    return tuple(dict.fromkeys(targets))


def _run_pytest(targets: tuple[str, ...]) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *targets]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_focused_eval(
    *,
    subset_dataset_path: Path,
    runs_root: Path,
    executor: str,
    judge: str,
    judge_model: str | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/sebi_eval_run.py",
        "--dataset-jsonl",
        str(subset_dataset_path),
        "--output-root",
        str(runs_root),
        "--executor",
        executor,
        "--judge",
        judge,
        "--json",
    ]
    if judge_model:
        command.extend(["--judge-model", judge_model])
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"Focused eval command failed with exit code {completed.returncode}: {completed.stderr}"
        )
    return json.loads(completed.stdout)


def _build_patch_loop_summary(
    *,
    base_bug_queue: dict[str, Any],
    head_bug_queue: dict[str, Any],
    selected_case_ids: tuple[str, ...],
    base_run_dir: Path,
    head_run_dir: Path,
    loop_dir: Path,
    test_result: dict[str, Any] | None,
    compare_payload: dict[str, Any],
) -> dict[str, Any]:
    base_entries = {
        entry["case_id"]: entry
        for entry in base_bug_queue.get("entries", [])
        if entry["case_id"] in selected_case_ids
    }
    head_entries = {
        entry["case_id"]: entry
        for entry in head_bug_queue.get("entries", [])
        if entry["case_id"] in selected_case_ids
    }
    cases_fixed = sorted(
        case_id
        for case_id in selected_case_ids
        if base_entries.get(case_id, {}).get("true_bug")
        and not head_entries.get(case_id, {}).get("true_bug")
    )
    cases_regressed = sorted(
        case_id
        for case_id in selected_case_ids
        if not base_entries.get(case_id, {}).get("true_bug")
        and head_entries.get(case_id, {}).get("true_bug")
    )
    return {
        "base_run_dir": str(base_run_dir),
        "head_run_dir": str(head_run_dir),
        "loop_dir": str(loop_dir),
        "baseline_subset_dir": str(loop_dir / "base_subset"),
        "selected_case_ids": list(selected_case_ids),
        "bug_count_before": sum(1 for entry in base_entries.values() if entry.get("true_bug")),
        "bug_count_after": sum(1 for entry in head_entries.values() if entry.get("true_bug")),
        "cases_fixed": cases_fixed,
        "cases_regressed": cases_regressed,
        "notes": [
            f"focused_case_count={len(selected_case_ids)}",
            f"compare_summary_json={loop_dir / 'compare_summary.json'}",
            f"focused_bug_queue_json={head_run_dir / 'true_bug_queue.json'}",
        ],
        "test_result": test_result,
        "compare_summary": compare_payload,
    }


def _render_patch_loop_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Patch Loop",
        "",
        f"- Base run: `{payload['base_run_dir']}`",
        f"- Head run: `{payload['head_run_dir']}`",
        f"- Selected cases: `{len(payload['selected_case_ids'])}`",
        f"- Bug count before: `{payload['bug_count_before']}`",
        f"- Bug count after: `{payload['bug_count_after']}`",
        f"- Cases fixed: {', '.join(payload['cases_fixed']) if payload['cases_fixed'] else 'None'}",
        f"- Cases regressed: {', '.join(payload['cases_regressed']) if payload['cases_regressed'] else 'None'}",
        "",
        "## Notes",
        "",
    ]
    for note in payload["notes"]:
        lines.append(f"- {note}")
    if payload.get("test_result") is not None:
        lines.extend(
            [
                "",
                "## Tests",
                "",
                f"- Return code: `{payload['test_result']['returncode']}`",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _make_loop_dir() -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return PROJECT_ROOT / "artifacts" / "sebi_eval_patch_loops" / timestamp


def _materialize_base_subset_artifact(
    *,
    base_run_dir: Path,
    loop_dir: Path,
    selected_case_ids: tuple[str, ...],
    bug_queue: dict[str, Any],
) -> Path:
    subset_dir = loop_dir / "base_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in _load_jsonl(base_run_dir / "per_case_results.jsonl")
        if str(row.get("case", {}).get("case_id") or "") in selected_case_ids
    ]
    failures = [row for row in rows if not bool(row.get("passed", False))]
    summary = _summarize_rows(rows)
    summary["true_bug_count"] = sum(
        1
        for entry in bug_queue.get("entries", [])
        if entry["case_id"] in selected_case_ids and entry["true_bug"]
    )
    summary["stale_expectation_count"] = sum(
        1
        for entry in bug_queue.get("entries", [])
        if entry["case_id"] in selected_case_ids and entry["stale_expectation"]
    )
    (subset_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_jsonl(subset_dir / "per_case_results.jsonl", rows)
    _write_jsonl(subset_dir / "failures.jsonl", failures)
    (subset_dir / "failures.md").write_text("# Failures\n", encoding="utf-8")
    subset_bug_queue = {
        **bug_queue,
        "run_dir": str(subset_dir),
        "summary": {
            **bug_queue.get("summary", {}),
            "failed_case_count": len(failures),
            "true_bug_count": summary["true_bug_count"],
            "stale_expectation_count": summary["stale_expectation_count"],
        },
        "entries": [
            entry
            for entry in bug_queue.get("entries", [])
            if entry["case_id"] in selected_case_ids
        ],
        "clusters": [
            cluster
            for cluster in bug_queue.get("clusters", [])
            if any(case_id in selected_case_ids for case_id in cluster["affected_case_ids"])
        ],
    }
    (subset_dir / "true_bug_queue.json").write_text(
        json.dumps(subset_bug_queue, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return subset_dir


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "passed_cases": 0,
            "failed_cases": 0,
            "strict_route_accuracy": 0.0,
            "equivalent_route_accuracy": 0.0,
            "faithfulness_average": 0.0,
            "hallucination_rate": 0.0,
            "numeric_fact_accuracy": 0.0,
            "structured_current_info_accuracy": 0.0,
            "candidate_list_correctness": 0.0,
            "wrong_answer_regression_pass_count": 0,
            "wrong_answer_regression_total": 0,
            "true_bug_count": 0,
            "stale_expectation_count": 0,
            "contamination_count": 0,
        }
    passed_cases = sum(1 for row in rows if bool(row.get("passed", False)))
    failed_cases = total - passed_cases
    strict_route_accuracy = _mean(float(row.get("route", {}).get("strict_route_match", False)) for row in rows)
    equivalent_route_accuracy = _mean(float(row.get("route", {}).get("equivalent_route_match", False)) for row in rows)
    faithfulness_average = _mean(
        float(row.get("grounding", {}).get("faithfulness", 0.0) or 0.0)
        for row in rows
    )
    hallucination_rate = _mean(
        float(row.get("grounding", {}).get("hallucination_rate", 0.0) or 0.0)
        for row in rows
    )
    numeric_rows = [row for row in rows if int(row.get("numeric", {}).get("expected_fact_count", 0) or 0) > 0]
    numeric_fact_accuracy = _mean(
        float(row.get("numeric", {}).get("numeric_accuracy", 0.0) or 0.0)
        for row in numeric_rows
    )
    candidate_rows = [
        row
        for row in rows
        if row.get("execution", {}).get("answer_status") == "clarify"
        or bool(row.get("execution", {}).get("debug", {}).get("candidate_list_debug", {}).get("used"))
    ]
    candidate_list_correctness = _mean(
        float(row.get("retrieval", {}).get("candidate_list_correctness", 0.0) or 0.0)
        for row in candidate_rows
    )
    regression_rows = [
        row
        for row in rows
        if str(row.get("case", {}).get("issue_class") or "") == "regression"
        or "wrong_answer_regression" in set(row.get("case", {}).get("tags", []) or ())
    ]
    return {
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "strict_route_accuracy": round(strict_route_accuracy, 4),
        "equivalent_route_accuracy": round(equivalent_route_accuracy, 4),
        "faithfulness_average": round(faithfulness_average, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "numeric_fact_accuracy": round(numeric_fact_accuracy, 4),
        "structured_current_info_accuracy": 0.0,
        "candidate_list_correctness": round(candidate_list_correctness, 4),
        "wrong_answer_regression_pass_count": sum(1 for row in regression_rows if bool(row.get("passed", False))),
        "wrong_answer_regression_total": len(regression_rows),
        "contamination_count": sum(
            1
            for row in rows
            if bool(row.get("retrieval", {}).get("mixed_record_contamination"))
        ),
    }


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
