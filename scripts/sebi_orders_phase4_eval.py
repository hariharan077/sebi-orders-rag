#!/usr/bin/env python3
"""Control-pack evaluator for Phase 4 adaptive chat hardening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.eval import (
    ControlPackEvaluator,
    render_summary,
    resolve_failure_dump_root,
    summary_as_dict,
)
from app.sebi_orders_rag.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SEBI Orders control-pack evaluation and regression suite.",
    )
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_CONTROL_PACK_ROOT.",
    )
    parser.add_argument(
        "--run-regressions",
        action="store_true",
        help="Run wrong-answer regression checks from wrong_answer_examples.jsonl.",
    )
    parser.add_argument(
        "--run-eval-queries",
        action="store_true",
        help="Run eval_queries.jsonl through the Phase 4 answer pipeline.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of terminal text.",
    )
    parser.add_argument(
        "--failure-dump-root",
        type=Path,
        help="Optional eval-triage reference dump. Defaults to the latest artifacts/sebi_eval_failure_dump_* directory.",
    )
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        run_eval_queries = args.run_eval_queries or (not args.run_eval_queries and not args.run_regressions)
        run_regressions = args.run_regressions or (not args.run_eval_queries and not args.run_regressions)

        load_env_file(PROJECT_ROOT / ".env")
        settings = SebiOrdersRagSettings.from_env(
            control_pack_root_override=args.control_pack_root,
        )
        configure_logging(settings.log_level)
        control_pack = load_control_pack(settings.control_pack_root)
        if control_pack is None:
            print("Control pack is not configured.", file=sys.stderr)
            return 1
        failure_dump_root = resolve_failure_dump_root(
            args.failure_dump_root,
            search_root=PROJECT_ROOT,
        )

        with get_connection(settings) as connection:
            initialize_phase4_schema(connection, settings)
            connection.commit()
            service = AdaptiveRagAnswerService(settings=settings, connection=connection)
            evaluator = ControlPackEvaluator(
                service=service,
                control_pack=control_pack,
                failure_dump_root=failure_dump_root,
            )
            summary = evaluator.run(
                run_eval_queries=run_eval_queries,
                run_regressions=run_regressions,
            )

        if args.json:
            print(json.dumps(summary_as_dict(summary), indent=2))
        else:
            for index, result in enumerate(summary.results, start=1):
                print(
                    f"{index}. {'PASS' if result.passed else 'FAIL'} "
                    f"[{result.case_kind}] route={result.actual_route_mode} "
                    f"status={result.answer_status} confidence={result.confidence:.4f}"
                )
                print(f"   query={result.query}")
                print(
                    "   triage="
                    f"{result.triage_bucket or '-'} "
                    f"stale_expectation={result.stale_expectation} "
                    f"equivalent_route={result.equivalent_route_passed}"
                )
                print(
                    "   strict="
                    f"{result.strict_single_matter_triggered} "
                    f"comparison_disabled_lock={result.comparison_disabled_lock} "
                    f"mixed_guardrail={result.mixed_record_guardrail_fired}"
                )
                if result.expected_record_key or result.actual_cited_record_keys or result.actual_active_record_keys:
                    print(
                        "   record_keys="
                        f"expected={result.expected_record_key or '-'} "
                        f"cited={list(result.actual_cited_record_keys)} "
                        f"active={list(result.actual_active_record_keys)}"
                    )
                if result.reasons:
                    print(f"   reasons={'; '.join(result.reasons)}")
                if result.equivalent_route_reason:
                    print(f"   equivalent_route_reason={result.equivalent_route_reason}")
            print(render_summary(summary))

        return 0 if summary.failed_cases == 0 else 1
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(
            "Phase 4 eval finished with a safe failure summary: "
            f"{type(exc).__name__}: {' '.join(str(exc).split()) or 'no details available'}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
