#!/usr/bin/env python3
"""Run the SEBI evaluation engine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.evaluation.case_loader import build_dataset
from app.sebi_orders_rag.evaluation.dataset import filter_cases, load_dataset, merge_datasets
from app.sebi_orders_rag.evaluation.judge_llm import NoopJudgeClient, OpenAIJudgeClient
from app.sebi_orders_rag.evaluation.redteam import build_redteam_cases
from app.sebi_orders_rag.evaluation.report import render_run_summary
from app.sebi_orders_rag.evaluation.runner import (
    EvaluationRunner,
    LiveAssistantExecutor,
    ReplayExecutor,
    build_run_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SEBI evaluation engine.")
    parser.add_argument("--dataset-jsonl", type=Path, help="Existing dataset JSONL path.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sebi_eval_runs",
        help="Artifact root for benchmark outputs.",
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
        help="Execution backend.",
    )
    parser.add_argument(
        "--judge",
        choices=("none", "openai", "auto"),
        default="none",
        help="Optional judge backend.",
    )
    parser.add_argument("--judge-model", help="Optional judge model override.")
    parser.add_argument("--limit", type=int, help="Optional case limit.")
    parser.add_argument(
        "--include-tag",
        action="append",
        default=[],
        help="Only include cases that contain all requested tags.",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="Exclude cases containing any requested tags.",
    )
    parser.add_argument(
        "--include-redteam",
        action="store_true",
        help="Merge generated red-team cases into the run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary instead of text.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(PROJECT_ROOT / ".env")

    dataset = (
        load_dataset(args.dataset_jsonl, name="sebi_eval_dataset")
        if args.dataset_jsonl
        else _build_default_dataset(args)
    )
    if args.include_redteam:
        dataset = merge_datasets(
            name=dataset.name,
            cases=(*dataset.cases, *build_redteam_cases(dataset.cases)),
            source_files=dataset.source_files,
            metadata=dataset.metadata,
        )
    dataset = merge_datasets(
        name=dataset.name,
        cases=filter_cases(
            dataset.cases,
            include_tags=args.include_tag,
            exclude_tags=args.exclude_tag,
            limit=args.limit,
        ),
        source_files=dataset.source_files,
        metadata=dataset.metadata,
    )

    settings = _load_settings(args)
    executor = _build_executor(args, settings)
    judge_client = _build_judge(args, settings)
    runner = EvaluationRunner(executor=executor, judge_client=judge_client)
    metadata = build_run_metadata(
        dataset=dataset,
        executor_mode=executor.mode,
        output_root=args.output_root,
        assistant_model=(settings.chat_model if settings is not None else None),
        judge_model=(args.judge_model or (settings.chat_model if args.judge != "none" and settings is not None else None)),
        prompt_version="sebi_eval_runner_v1",
        retrieval_settings=_retrieval_settings(settings),
        planner_settings={"planner": "deterministic"},
        git_commit_hash=_git_commit_hash(),
    )
    try:
        run_result = runner.run(
            dataset=dataset,
            metadata=metadata,
            output_root=args.output_root,
        )
    finally:
        _close_executor(executor)
    artifact_dir = args.output_root.resolve(strict=False) / metadata.run_id
    if args.json:
        payload = dict(run_result.summary)
        payload["artifact_dir"] = str(artifact_dir)
        print(json.dumps(payload, indent=2))
    else:
        print(render_run_summary(run_result.summary))
        print(f"artifact_dir: {artifact_dir}")
    return 0 if run_result.summary.get("failed_cases", 0) == 0 else 1


def _build_default_dataset(args) -> object:
    cases, source_files, _ = build_dataset(
        name="sebi_eval_dataset",
        control_pack_root=args.control_pack_root,
        failure_dump_root=args.failure_dump_root,
    )
    return merge_datasets(
        name="sebi_eval_dataset",
        cases=cases,
        source_files=source_files,
    )


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


def _build_judge(args, settings: SebiOrdersRagSettings | None):
    if args.judge == "none":
        return NoopJudgeClient()
    if args.judge == "auto" and settings is None:
        return NoopJudgeClient()
    if settings is None:
        raise RuntimeError("Judge execution requires a valid SEBI runtime environment.")
    return OpenAIJudgeClient(settings=settings, model_name=args.judge_model)


def _retrieval_settings(settings: SebiOrdersRagSettings | None) -> dict[str, object]:
    if settings is None:
        return {}
    return {
        "retrieval_top_k_docs": settings.retrieval_top_k_docs,
        "retrieval_top_k_sections": settings.retrieval_top_k_sections,
        "retrieval_top_k_chunks": settings.retrieval_top_k_chunks,
        "max_context_chunks": settings.max_context_chunks,
        "max_context_tokens": settings.max_context_tokens,
    }


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or None
    except Exception:
        return None


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
