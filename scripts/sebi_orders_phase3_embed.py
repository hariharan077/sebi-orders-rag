#!/usr/bin/env python3
"""CLI entrypoint for SEBI Orders RAG Phase 3 embeddings."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import execute_sql_file, get_connection
from app.sebi_orders_rag.embeddings.phase3_service import Phase3EmbeddingService
from app.sebi_orders_rag.exceptions import ConfigurationError
from app.sebi_orders_rag.logging_utils import configure_logging
from app.sebi_orders_rag.schemas import Phase3EmbeddingSummary

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 3 CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run Phase 3 of the SEBI Orders RAG hierarchical embedding workflow.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="Select pending versions without calling OpenAI or writing embeddings. This is the default.",
    )
    mode_group.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="Persist document, section, and chunk embeddings.",
    )
    parser.add_argument(
        "--run-migration",
        action="store_true",
        help="Apply the Phase 3 hierarchical retrieval migration before running.",
    )
    parser.add_argument(
        "--document-version-id",
        type=int,
        help="Restrict processing to a single document_versions.document_version_id.",
    )
    parser.add_argument(
        "--record-key",
        help="Restrict processing to a single source_documents.record_key.",
    )
    parser.add_argument(
        "--bucket-name",
        help="Restrict processing to a single source_documents.bucket_name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of pending document versions selected.",
    )
    parser.set_defaults(apply=False)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
        settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
        configure_logging(settings.log_level)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    LOGGER.info(
        "Starting SEBI Orders Phase 3 embeddings in %s mode",
        "apply" if args.apply else "dry-run",
    )

    try:
        with get_connection(settings) as connection:
            if args.run_migration:
                execute_sql_file(connection, settings.sql_hierarchical_retrieval_path)
                connection.commit()
                print(f"applied migration from {settings.sql_hierarchical_retrieval_path}")

            service = Phase3EmbeddingService(settings=settings, connection=connection)
            summary = service.run(
                apply=args.apply,
                record_key=args.record_key,
                bucket_name=args.bucket_name,
                document_version_id=args.document_version_id,
                limit=args.limit,
            )

            if not args.apply:
                connection.rollback()
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_summary(summary)
    return 0


def print_summary(summary: Phase3EmbeddingSummary) -> None:
    """Print the stable Phase 3 execution summary."""

    print("Phase 3 summary")
    for line in summary.as_lines():
        print(f"  - {line}")


if __name__ == "__main__":
    raise SystemExit(main())
