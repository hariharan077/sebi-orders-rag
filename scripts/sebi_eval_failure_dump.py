#!/usr/bin/env python3
"""Generate a bucketed failure dump for the SEBI Orders eval suite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.constants import DEFAULT_CONTROL_PACK_DIR_PREFIX, ENV_PREFIX
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.eval import ControlPackEvaluator, summary_as_dict
from app.sebi_orders_rag.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional existing eval summary JSON from scripts/sebi_orders_phase4_eval.py --json.",
    )
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        help="Optional override for the control pack root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to artifacts/sebi_eval_failure_dump_<timestamp>.",
    )
    parser.add_argument(
        "--run-regressions",
        action="store_true",
        help="Run wrong-answer regression checks when no summary JSON is supplied.",
    )
    parser.add_argument(
        "--run-eval-queries",
        action="store_true",
        help="Run eval_queries.jsonl when no summary JSON is supplied.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(PROJECT_ROOT / ".env")

    control_pack_root = _resolve_control_pack_root(args.control_pack_root)
    control_pack = load_control_pack(control_pack_root)
    if control_pack is None:
        print("Control pack is not configured.", file=sys.stderr)
        return 1

    if args.summary_json:
        summary = _load_json(args.summary_json)
    else:
        run_eval_queries = args.run_eval_queries or (not args.run_eval_queries and not args.run_regressions)
        run_regressions = args.run_regressions or (not args.run_eval_queries and not args.run_regressions)
        summary = _run_eval(
            control_pack_root=args.control_pack_root,
            run_eval_queries=run_eval_queries,
            run_regressions=run_regressions,
        )

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched = _build_failure_dump(summary=summary, control_pack=control_pack)
    summary_path = output_dir / "summary.json"
    failures_path = output_dir / "failed_cases.json"
    report_path = output_dir / "failed_cases.md"

    _write_json(summary_path, summary)
    _write_json(failures_path, enriched)
    report_path.write_text(_render_markdown_report(enriched), encoding="utf-8")

    print(f"summary_json: {summary_path}")
    print(f"failed_cases_json: {failures_path}")
    print(f"failed_cases_md: {report_path}")
    print(
        "bucket_counts: "
        + ", ".join(
            f"{bucket}={count}"
            for bucket, count in sorted(enriched["bucket_counts"].items())
        )
    )
    return 0


def _run_eval(
    *,
    control_pack_root: Path | None,
    run_eval_queries: bool,
    run_regressions: bool,
) -> dict[str, Any]:
    settings = SebiOrdersRagSettings.from_env(
        control_pack_root_override=control_pack_root,
    )
    configure_logging(settings.log_level)
    control_pack = load_control_pack(settings.control_pack_root)
    if control_pack is None:
        raise RuntimeError("Control pack is not configured.")
    with get_connection(settings) as connection:
        initialize_phase4_schema(connection, settings)
        connection.commit()
        service = AdaptiveRagAnswerService(settings=settings, connection=connection)
        evaluator = ControlPackEvaluator(service=service, control_pack=control_pack)
        summary = evaluator.run(
            run_eval_queries=run_eval_queries,
            run_regressions=run_regressions,
        )
    return summary_as_dict(summary)


def _resolve_control_pack_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser().resolve(strict=False)
    env_value = os.environ.get(f"{ENV_PREFIX}CONTROL_PACK_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve(strict=False)
    artifacts_dir = PROJECT_ROOT / "artifacts"
    candidates = sorted(
        (
            path
            for path in artifacts_dir.glob(f"{DEFAULT_CONTROL_PACK_DIR_PREFIX}*")
            if path.is_dir()
        ),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Could not resolve a control pack root.")
    return candidates[0]


def _resolve_output_dir(cli_output_dir: Path | None) -> Path:
    if cli_output_dir is not None:
        return cli_output_dir.expanduser().resolve(strict=False)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return PROJECT_ROOT / "artifacts" / f"sebi_eval_failure_dump_{timestamp}"


def _build_failure_dump(*, summary: dict[str, Any], control_pack) -> dict[str, Any]:
    eval_case_meta = {case.query: case.notes for case in control_pack.eval_queries}
    regression_meta = {
        case.user_query: case.what_it_should_have_answered
        for case in control_pack.wrong_answer_examples
    }
    failures: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    for index, result in enumerate(summary.get("results", []), start=1):
        if result.get("passed", False):
            continue
        failed_case = _build_failed_case(
            index=index,
            result=result,
            control_pack=control_pack,
            eval_case_meta=eval_case_meta,
            regression_meta=regression_meta,
        )
        bucket_counts[failed_case["primary_bucket"]] += 1
        failures.append(failed_case)

    grouped_queries: dict[str, list[str]] = defaultdict(list)
    for item in failures:
        grouped_queries[item["primary_bucket"]].append(item["query"])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            key: value
            for key, value in summary.items()
            if key != "results"
        },
        "failed_case_count": len(failures),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "bucket_queries": {
            bucket: queries for bucket, queries in sorted(grouped_queries.items())
        },
        "failed_cases": failures,
    }


def _build_failed_case(
    *,
    index: int,
    result: dict[str, Any],
    control_pack,
    eval_case_meta: dict[str, str],
    regression_meta: dict[str, str],
) -> dict[str, Any]:
    route_debug = dict(result.get("debug", {}).get("route_debug", {}) or {})
    metadata_debug = dict(result.get("debug", {}).get("metadata_debug", {}) or {})
    candidate_list_debug = dict(result.get("debug", {}).get("candidate_list_debug", {}) or {})
    clarification_debug = dict(result.get("debug", {}).get("clarification_debug", {}) or {})

    clarify_fired = bool(
        result.get("answer_status") == "clarify"
        or result.get("actual_route_mode") == "clarify"
        or clarification_debug.get("used")
    )
    candidate_list_used = bool(candidate_list_debug.get("used"))
    actual_cited_record_keys = list(result.get("actual_cited_record_keys") or [])
    actual_active_record_keys = list(result.get("actual_active_record_keys") or [])

    tags = list(result.get("triage_tags") or ())
    primary_bucket = str(result.get("triage_bucket") or "")
    if not tags or not primary_bucket:
        tags = _classify_failure_tags(
            result=result,
            route_debug=route_debug,
            metadata_debug=metadata_debug,
            clarify_fired=clarify_fired,
            candidate_list_used=candidate_list_used,
        )
        primary_bucket = _pick_primary_bucket(tags)

    query = str(result.get("query") or "")
    expectation_note = (
        eval_case_meta.get(query)
        if result.get("case_kind") == "eval_query"
        else regression_meta.get(query)
    )

    return {
        "index": index,
        "case_kind": result.get("case_kind"),
        "primary_bucket": primary_bucket,
        "bucket_tags": tags,
        "query": query,
        "expected_route": result.get("expected_route_mode"),
        "actual_route": result.get("actual_route_mode"),
        "expected_record_key": result.get("expected_record_key"),
        "expected_record_title": result.get("expected_title"),
        "actual_cited_record_keys": actual_cited_record_keys,
        "actual_cited_records": _resolve_records(control_pack, actual_cited_record_keys),
        "actual_active_record_keys": actual_active_record_keys,
        "actual_active_records": _resolve_records(control_pack, actual_active_record_keys),
        "clarify_fired": clarify_fired,
        "candidate_list_used": candidate_list_used,
        "mixed_record_guardrail_fired": bool(result.get("mixed_record_guardrail_fired")),
        "single_matter_rule_respected": bool(result.get("single_matter_rule_respected")),
        "metadata_used": bool(metadata_debug.get("used")),
        "metadata_type": metadata_debug.get("metadata_type"),
        "stale_expectation": bool(result.get("stale_expectation")),
        "equivalent_route_passed": bool(result.get("equivalent_route_passed")),
        "equivalent_route_reason": result.get("equivalent_route_reason"),
        "candidate_source": candidate_list_debug.get("candidate_source"),
        "candidate_record_keys": candidate_list_debug.get("record_keys") or [],
        "answer_status": result.get("answer_status"),
        "confidence": result.get("confidence"),
        "reasons": list(result.get("reasons") or []),
        "expectation_note": expectation_note,
    }


def _classify_failure_tags(
    *,
    result: dict[str, Any],
    route_debug: dict[str, Any],
    metadata_debug: dict[str, Any],
    clarify_fired: bool,
    candidate_list_used: bool,
) -> list[str]:
    reason_text = " ".join(str(item) for item in (result.get("reasons") or [])).lower()
    expected_route = str(result.get("expected_route_mode") or "")
    actual_route = str(result.get("actual_route_mode") or "")
    expected_record_key = str(result.get("expected_record_key") or "")

    contamination = bool(
        not result.get("single_matter_rule_respected", True)
        or "contaminated" in reason_text
    )
    stale_expectation = bool(
        expected_route
        and actual_route != expected_route
        and clarify_fired
        and candidate_list_used
    )
    weak_metadata_extraction = bool(
        metadata_debug.get("used")
        or (
            bool(expected_record_key)
            and (
                route_debug.get("asks_order_signatory")
                or route_debug.get("asks_order_date")
                or route_debug.get("asks_legal_provisions")
                or route_debug.get("asks_order_observations")
            )
        )
    )
    wrong_candidate_ranking = bool(
        candidate_list_used
        and expected_record_key
        and not stale_expectation
    )
    wrong_route = bool(
        expected_route
        and actual_route != expected_route
        and not stale_expectation
    )
    wrong_answer_despite_correct_route = bool(
        not contamination
        and not wrong_route
        and not wrong_candidate_ranking
        and (
            (expected_route and actual_route == expected_route)
            or result.get("case_kind") == "wrong_answer_regression"
        )
    )

    tags: list[str] = []
    if contamination:
        tags.append("contamination")
    if stale_expectation:
        tags.append("stale expectation")
    if weak_metadata_extraction:
        tags.append("weak metadata extraction")
    if wrong_candidate_ranking:
        tags.append("wrong candidate ranking")
    if wrong_route:
        tags.append("wrong route")
    if wrong_answer_despite_correct_route:
        tags.append("wrong answer despite correct route")
    if not tags:
        tags.append("wrong answer despite correct route")
    return tags


def _pick_primary_bucket(tags: list[str]) -> str:
    priority = (
        "contamination",
        "stale expectation",
        "weak metadata extraction",
        "wrong candidate ranking",
        "wrong route",
        "wrong answer despite correct route",
    )
    for bucket in priority:
        if bucket in tags:
            return bucket
    return tags[0]


def _resolve_records(control_pack, record_keys: list[str]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record_key in record_keys:
        if not record_key or record_key in seen:
            continue
        seen.add(record_key)
        row = control_pack.documents_by_record_key.get(record_key)
        resolved.append(
            {
                "record_key": record_key,
                "title": row.exact_title if row is not None else None,
                "bucket_category": row.bucket_category if row is not None else None,
                "order_date": row.order_date.isoformat() if row and row.order_date else None,
            }
        )
    return resolved


def _render_markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = payload["summary"]
    lines.append("# SEBI Eval Failure Dump")
    lines.append("")
    lines.append(f"- Generated at: `{payload['generated_at']}`")
    lines.append(
        f"- Cases: `{summary['passed_cases']}/{summary['total_cases']}` passed"
    )
    lines.append(f"- Failed cases: `{payload['failed_case_count']}`")
    lines.append(f"- Route accuracy (strict): `{summary['route_accuracy']}`")
    if "route_accuracy_equivalent" in summary:
        lines.append(
            f"- Route accuracy (equivalent): `{summary['route_accuracy_equivalent']}`"
        )
    lines.append(
        f"- Candidate-list correctness: `{summary['candidate_list_correctness']}`"
    )
    if "stale_expectation_count" in summary:
        lines.append(f"- Stale expectations: `{summary['stale_expectation_count']}`")
    if "true_bug_count" in summary:
        lines.append(f"- True bugs: `{summary['true_bug_count']}`")
    lines.append(
        f"- Wrong-example regressions: `{summary['wrong_example_regression_pass_count']}/{summary['wrong_example_regression_total']}` passed"
    )
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    for bucket, count in sorted(payload["bucket_counts"].items()):
        lines.append(f"- {bucket}: {count}")
    for bucket, cases in _group_by_bucket(payload["failed_cases"]).items():
        lines.append("")
        lines.append(f"## {bucket.title()}")
        lines.append("")
        for case in cases:
            lines.extend(_render_case_block(case))
    return "\n".join(lines) + "\n"


def _group_by_bucket(failed_cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in failed_cases:
        grouped[item["primary_bucket"]].append(item)
    return dict(grouped)


def _render_case_block(case: dict[str, Any]) -> list[str]:
    lines = [
        f"### {case['index']}. {case['query']}",
        "",
        f"- Case kind: `{case['case_kind']}`",
        f"- Expected route: `{case['expected_route'] or '-'}`",
        f"- Actual route: `{case['actual_route'] or '-'}`",
        f"- Expected record: `{case['expected_record_key'] or '-'}`",
        f"- Actual cited record keys: `{', '.join(case['actual_cited_record_keys']) or '-'}`",
        f"- Clarify fired: `{case['clarify_fired']}`",
        f"- Mixed-record guardrail fired: `{case['mixed_record_guardrail_fired']}`",
        f"- Stale expectation: `{case['stale_expectation']}`",
        f"- Equivalent route passed: `{case['equivalent_route_passed']}`",
        f"- Reasons: `{'; '.join(case['reasons']) or '-'}`",
    ]
    if case.get("equivalent_route_reason"):
        lines.append(f"- Equivalent route reason: `{case['equivalent_route_reason']}`")
    if case.get("expectation_note"):
        lines.append(f"- Expected behavior note: {case['expectation_note']}")
    if case["actual_cited_records"]:
        titles = ", ".join(
            f"{item['record_key']} ({item['title'] or 'unknown title'})"
            for item in case["actual_cited_records"]
        )
        lines.append(f"- Actual cited records: {titles}")
    lines.append("")
    return lines


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
