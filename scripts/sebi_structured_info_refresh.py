#!/usr/bin/env python3
"""Refresh canonical structured SEBI current-info tables and aggregates."""

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
from app.sebi_orders_rag.repositories.structured_info import StructuredInfoRepository

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh canonical structured SEBI current-info rows and counts.",
    )
    parser.add_argument("--data-root", type=Path, help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.")
    parser.add_argument("--run-migration", action="store_true", help="Apply the directory and canonical structured-info migrations first.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse existing raw structured rows instead of fetching official sources again.")
    parser.add_argument(
        "--source",
        choices=("directory", "orgchart", "regional_offices", "contact_us", "board_members", "all"),
        default="all",
        help="Limit raw structured refresh to one official source family.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
    configure_logging(settings.log_level)

    with get_connection(settings) as connection:
        if args.run_migration:
            initialize_directory_reference_schema(connection, settings)
            connection.commit()
            print(
                "applied migrations: "
                f"{settings.sql_directory_reference_path}, "
                f"{settings.sql_directory_reference_hardening_path}, "
                f"{settings.sql_structured_info_canonical_path}"
            )

        if not args.skip_ingest:
            LOGGER.info("Refreshing raw structured sources for source=%s", args.source)
            ingest_service = DirectoryIngestionService(settings=settings, connection=connection)
            summary = ingest_service.run(apply=True, source=args.source)
            print("Raw structured refresh summary")
            for line in summary.as_lines():
                print(f"  - {line}")

        repository = StructuredInfoRepository(connection)
        snapshot = repository.refresh_from_raw_dataset()
        connection.commit()

    print("Canonical structured-info refresh summary")
    print(f"  - canonical people: {len(snapshot.people)}")
    print(f"  - canonical offices: {len(snapshot.offices)}")
    print("  - designation counts:")
    for record in snapshot.designation_counts:
        print(f"      {record.designation_group}: {record.people_count}")
    print("  - office counts:")
    for record in snapshot.office_counts:
        print(f"      {record.office_name}: people={record.people_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
