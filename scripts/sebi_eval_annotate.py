#!/usr/bin/env python3
"""Annotate or enrich an existing SEBI evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.evaluation.annotation import annotate_cases, apply_expectation_updates
from app.sebi_orders_rag.evaluation.dataset import load_cases, validate_dataset, write_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate an existing SEBI eval dataset.")
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
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
        "--expectation-updates-json",
        type=Path,
        help="Optional JSON mapping case_id to approved expectation updates with annotation_reason.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = load_cases(args.input_jsonl)
    annotated = annotate_cases(
        cases,
        control_pack_root=args.control_pack_root,
        failure_dump_root=args.failure_dump_root,
    )
    if args.expectation_updates_json is not None:
        updates = json.loads(args.expectation_updates_json.read_text(encoding="utf-8"))
        annotated = apply_expectation_updates(annotated, updates=updates)
    errors = validate_dataset(annotated)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    write_cases(args.output_jsonl, annotated)
    print(f"annotated_cases={len(annotated)} output={args.output_jsonl.resolve(strict=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
