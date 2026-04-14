#!/usr/bin/env python3
"""Compare two persisted SEBI evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.evaluation.benchmark import compare_run_summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two SEBI evaluation runs.")
    parser.add_argument("base_run_dir", type=Path, nargs="?")
    parser.add_argument("head_run_dir", type=Path, nargs="?")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sebi_eval_runs",
        help="Artifact root containing run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for compare_summary.json and compare_summary.md. Defaults to the head run dir.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_dir, head_dir = _resolve_run_dirs(args.base_run_dir, args.head_run_dir, args.runs_root)
    comparison = compare_run_summaries(
        base_run_dir=base_dir,
        head_run_dir=head_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_ready(comparison), indent=2, allow_nan=False))
    return 0


def _resolve_run_dirs(base: Path | None, head: Path | None, runs_root: Path) -> tuple[Path, Path]:
    if base is not None and head is not None:
        return (
            base.expanduser().resolve(strict=False),
            head.expanduser().resolve(strict=False),
        )
    candidates = sorted(
        [path for path in runs_root.expanduser().resolve(strict=False).glob("sebi_eval_run_*") if path.is_dir()]
    )
    if len(candidates) < 2:
        raise RuntimeError("Need at least two run directories to compare.")
    return candidates[-2], candidates[-1]


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
