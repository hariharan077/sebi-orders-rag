#!/usr/bin/env python3
"""Generate a true-bug queue from persisted SEBI eval artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.evaluation.triage import (
    build_true_bug_queue,
    render_true_bug_queue_markdown,
    resolve_latest_run_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Existing eval run directory. Defaults to the latest run.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sebi_eval_runs",
        help="Root containing eval run directories.",
    )
    parser.add_argument(
        "--failure-dump-root",
        type=Path,
        help="Optional failure dump root used to preserve stale-expectation references.",
    )
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "control_pack",
        help="Optional control-pack root for richer touchpoint hints.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write true_bug_queue.json and true_bug_queue.md into the run directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = (
        args.run_dir.expanduser().resolve(strict=False)
        if args.run_dir is not None
        else resolve_latest_run_dir(args.runs_root)
    )
    payload = build_true_bug_queue(
        run_dir=run_dir,
        failure_dump_root=args.failure_dump_root,
        control_pack_root=args.control_pack_root,
    )
    if args.write:
        (run_dir / "true_bug_queue.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "true_bug_queue.md").write_text(
            render_true_bug_queue_markdown(payload),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
