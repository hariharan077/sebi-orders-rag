#!/usr/bin/env python3
"""CLI entrypoint for SEBI Orders RAG Phase 1."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection, initialize_schema
from app.sebi_orders_rag.ingestion.service import Phase1IngestionService
from app.sebi_orders_rag.logging_utils import configure_logging
from app.sebi_orders_rag.schemas import Phase1Summary

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 1 CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run Phase 1 of the SEBI Orders RAG ingestion workflow.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the retrieval schema before scanning manifests.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="Scan manifests without writing database changes. This is the default.",
    )
    mode_group.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="Persist document and version records to the retrieval database.",
    )
    parser.set_defaults(apply=False)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
    configure_logging(settings.log_level)
    LOGGER.info(
        "Starting SEBI Orders Phase 1 in %s mode for %s",
        "apply" if args.apply else "dry-run",
        settings.data_root,
    )

    with get_connection(settings) as connection:
        try:
            if args.init_db:
                initialize_schema(connection, settings)
                connection.commit()
                print(f"initialized schema from {settings.sql_schema_path}")

            service = Phase1IngestionService(settings=settings, connection=connection)
            summary = service.run(apply=args.apply)

            if args.apply:
                connection.commit()
            else:
                connection.rollback()
        except Exception:
            connection.rollback()
            raise

    print_summary(summary)
    return 0


def print_summary(summary: Phase1Summary) -> None:
    """Print the stable Phase 1 execution summary."""

    print("Phase 1 summary")
    for line in summary.as_lines():
        print(f"  - {line}")


if __name__ == "__main__":
    raise SystemExit(main())
