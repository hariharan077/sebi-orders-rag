#!/usr/bin/env python3
"""CLI entrypoint for structured SEBI directory/reference ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection, initialize_directory_reference_schema
from app.sebi_orders_rag.directory_data.service import DirectoryIngestionService
from app.sebi_orders_rag.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest structured SEBI directory, organisation, and office-reference pages.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.",
    )
    parser.add_argument(
        "--run-migration",
        action="store_true",
        help="Apply the structured directory-reference SQL migrations before ingestion.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="Fetch and parse without persisting database changes. This is the default.",
    )
    mode_group.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="Persist snapshots and structured rows to Postgres.",
    )
    parser.set_defaults(apply=False)
    parser.add_argument(
        "--source",
        choices=("directory", "orgchart", "regional_offices", "contact_us", "board_members", "all"),
        default="all",
        help="Limit ingestion to one official source family.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
    configure_logging(settings.log_level)
    LOGGER.info(
        "Starting structured SEBI directory ingest in %s mode for source=%s",
        "apply" if args.apply else "dry-run",
        args.source,
    )

    with get_connection(settings) as connection:
        if args.run_migration:
            initialize_directory_reference_schema(connection, settings)
            connection.commit()
            print(
                "applied migrations: "
                f"{settings.sql_directory_reference_path}, "
                f"{settings.sql_directory_reference_hardening_path}"
            )

        try:
            service = DirectoryIngestionService(settings=settings, connection=connection)
            summary = service.run(apply=args.apply, source=args.source)
            if args.apply:
                connection.commit()
            else:
                connection.rollback()
        except Exception:
            connection.rollback()
            raise

    print("Structured directory ingest summary")
    for line in summary.as_lines():
        print(f"  - {line}")
    for item in summary.source_summaries:
        print(
            "  - "
            f"{item.source_type}: fetch={item.fetch_status} parse={item.parse_status} "
            f"people={item.people_rows} board={item.board_rows} "
            f"offices={item.office_rows} org={item.org_rows}"
            + (f" error={item.error}" if item.error else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
