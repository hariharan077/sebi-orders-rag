#!/usr/bin/env python3
"""Export SEBI evaluation data/results to optional external formats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.evaluation.adapters import (
    export_results_to_deepeval,
    export_results_to_ragas,
)
from app.sebi_orders_rag.evaluation.dataset import load_dataset

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export SEBI eval data to optional formats.")
    parser.add_argument("--format", choices=("ragas", "deepeval"), required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="Optional run directory with per_case_results.jsonl.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_dataset(args.dataset_jsonl)
    results = _load_results(args.run_dir) if args.run_dir else []
    if args.format == "ragas":
        payload = export_results_to_ragas(cases=dataset.cases, results=results)
    else:
        payload = export_results_to_deepeval(cases=dataset.cases, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"exported_rows={len(payload)} output={args.output.resolve(strict=False)}")
    return 0


def _load_results(run_dir: Path) -> list[CaseEvaluationResult]:
    results_path = run_dir / "per_case_results.jsonl"
    rows = []
    if not results_path.exists():
        return rows
    for raw_line in results_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
