#!/usr/bin/env python3
"""Generate and run the red-team SEBI evaluation suite."""

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
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.evaluation.case_loader import build_dataset
from app.sebi_orders_rag.evaluation.dataset import merge_datasets, write_cases
from app.sebi_orders_rag.evaluation.redteam import build_redteam_cases
from app.sebi_orders_rag.evaluation.report import render_run_summary
from app.sebi_orders_rag.evaluation.runner import (
    EvaluationRunner,
    LiveAssistantExecutor,
    ReplayExecutor,
    build_run_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or run red-team SEBI eval cases.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional output path for generated red-team JSONL.",
    )
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "control_pack",
    )
    parser.add_argument(
        "--failure-dump-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "eval_failure_dump",
    )
    parser.add_argument(
        "--executor",
        choices=("auto", "live", "replay"),
        default="replay",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sebi_eval_runs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(PROJECT_ROOT / ".env")
    base_cases, source_files, _ = build_dataset(
        name="sebi_eval_dataset",
        control_pack_root=args.control_pack_root,
        failure_dump_root=args.failure_dump_root,
    )
    redteam_dataset = merge_datasets(
        name="sebi_eval_redteam",
        cases=build_redteam_cases(base_cases),
        source_files=source_files,
    )
    output_path = args.output_jsonl or (
        PROJECT_ROOT / "artifacts" / f"sebi_eval_redteam_{redteam_dataset.version}" / "redteam.jsonl"
    )
    write_cases(output_path, redteam_dataset.cases)
    settings = _load_settings(args)
    executor = _build_executor(args, settings)
    runner = EvaluationRunner(executor=executor)
    metadata = build_run_metadata(
        dataset=redteam_dataset,
        executor_mode=executor.mode,
        output_root=args.output_root,
        assistant_model=(settings.chat_model if settings is not None else None),
        prompt_version="sebi_eval_redteam_v1",
        retrieval_settings=_retrieval_settings(settings),
        planner_settings={"planner": "deterministic"},
    )
    try:
        run_result = runner.run(
            dataset=redteam_dataset,
            metadata=metadata,
            output_root=args.output_root,
        )
    finally:
        _close_executor(executor)
    artifact_dir = args.output_root.resolve(strict=False) / metadata.run_id
    if args.json:
        payload = dict(run_result.summary)
        payload["artifact_dir"] = str(artifact_dir)
        payload["dataset_jsonl"] = str(output_path.resolve(strict=False))
        print(json.dumps(payload, indent=2))
    else:
        print(render_run_summary(run_result.summary))
        print(f"dataset_jsonl: {output_path.resolve(strict=False)}")
        print(f"artifact_dir: {artifact_dir}")
    return 0 if run_result.summary.get("failed_cases", 0) == 0 else 1


def _load_settings(args) -> SebiOrdersRagSettings | None:
    try:
        return SebiOrdersRagSettings.from_env(
            control_pack_root_override=args.control_pack_root,
        )
    except Exception:
        return None


def _build_executor(args, settings: SebiOrdersRagSettings | None):
    if args.executor == "replay":
        return ReplayExecutor()
    if args.executor == "auto" and settings is None:
        return ReplayExecutor()
    if settings is None:
        raise RuntimeError("Live executor requires a valid SEBI runtime environment.")
    connection_cm = get_connection(settings)
    connection = connection_cm.__enter__()
    initialize_phase4_schema(connection, settings)
    service = AdaptiveRagAnswerService(settings=settings, connection=connection)
    executor = LiveAssistantExecutor(service=service, connection=connection)
    setattr(executor, "_connection_cm", connection_cm)
    return executor


def _retrieval_settings(settings: SebiOrdersRagSettings | None) -> dict[str, object]:
    if settings is None:
        return {}
    return {
        "retrieval_top_k_docs": settings.retrieval_top_k_docs,
        "retrieval_top_k_sections": settings.retrieval_top_k_sections,
        "retrieval_top_k_chunks": settings.retrieval_top_k_chunks,
    }


def _close_executor(executor) -> None:
    connection_cm = getattr(executor, "_connection_cm", None)
    if connection_cm is None:
        return
    try:
        connection_cm.__exit__(None, None, None)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
