#!/usr/bin/env python3
"""Audit canonical structured SEBI current-info coverage and counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection
from app.sebi_orders_rag.repositories.structured_info import StructuredInfoRepository
from app.sebi_orders_rag.structured_info.audit import build_audit_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print an audit report for canonical structured SEBI current-info data.",
    )
    parser.add_argument("--data-root", type=Path, help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero on warnings as well as failures.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
    with get_connection(settings) as connection:
        repository = StructuredInfoRepository(connection)
        report = build_audit_report(repository)

    for line in report.as_lines():
        print(line)
    if report.failures:
        return 2
    if args.fail_on_warning and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
