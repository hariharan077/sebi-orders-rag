#!/usr/bin/env python3
"""Build the unified SEBI evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.evaluation.annotation import annotate_cases
from app.sebi_orders_rag.evaluation.case_loader import build_dataset
from app.sebi_orders_rag.evaluation.dataset import (
    case_from_dict,
    load_cases,
    merge_datasets,
    validate_dataset,
    write_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the SEBI evaluation dataset.")
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "control_pack",
        help="Control-pack root used to seed eval cases.",
    )
    parser.add_argument(
        "--failure-dump-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "eval_failure_dump",
        help="Failure-dump root used for regression seed enrichment.",
    )
    parser.add_argument(
        "--sample-eval-cases",
        type=Path,
        default=PROJECT_ROOT / "app" / "sebi_orders_rag" / "eval" / "sample_eval_cases.jsonl",
        help="Packaged phase4 eval JSONL path.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL files to merge into the dataset.",
    )
    parser.add_argument(
        "--name",
        default="sebi_eval_dataset",
        help="Dataset name.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional explicit output dataset JSONL path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seed_cases, source_files, version = build_dataset(
        name=args.name,
        control_pack_root=args.control_pack_root,
        failure_dump_root=args.failure_dump_root,
        sample_eval_cases_path=args.sample_eval_cases,
    )

    extra_cases = []
    for input_path in args.input_jsonl:
        extra_cases.extend(load_cases(input_path))
        source_files.append(str(input_path.resolve(strict=False)))

    dataset = merge_datasets(
        name=args.name,
        cases=(*seed_cases, *extra_cases),
        source_files=source_files,
        metadata={"seed_version": version},
    )
    cases = annotate_cases(
        dataset.cases,
        control_pack_root=args.control_pack_root,
        failure_dump_root=args.failure_dump_root,
    )
    errors = validate_dataset(cases)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    output_path = args.output_jsonl or (
        PROJECT_ROOT / "artifacts" / f"{args.name}_{dataset.version}" / "dataset.jsonl"
    )
    write_cases(output_path, cases)
    manifest = {
        "name": dataset.name,
        "version": dataset.version,
        "case_count": len(cases),
        "output_jsonl": str(output_path.resolve(strict=False)),
        "source_files": list(dict.fromkeys(source_files)),
    }
    manifest_path = output_path.with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
