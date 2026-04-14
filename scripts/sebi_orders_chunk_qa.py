#!/usr/bin/env python3
"""CLI entrypoint for SEBI Orders chunk QA and inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.exceptions import ConfigurationError
from app.sebi_orders_rag.logging_utils import configure_logging
from app.sebi_orders_rag.db import get_connection
from app.sebi_orders_rag.qa import (
    ChunkAuditService,
    render_corpus_audit_json,
    render_corpus_audit_report,
    render_document_json,
    render_document_report,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for chunk QA."""

    parser = argparse.ArgumentParser(
        description="Inspect SEBI Orders Phase 2 chunk quality without modifying chunks.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.",
    )
    parser.add_argument(
        "--document-version-id",
        dest="document_version_ids",
        action="append",
        type=int,
        help="Inspect one document_version_id or repeat the flag to audit an explicit set.",
    )
    parser.add_argument(
        "--record-key",
        dest="record_keys",
        action="append",
        help="Inspect one record_key or repeat the flag to audit an explicit set.",
    )
    parser.add_argument(
        "--bucket-name",
        help="Restrict audit scope to a specific source_documents.bucket_name.",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample-per-bucket",
        type=int,
        help="Audit the oldest N processed documents per bucket.",
    )
    sample_group.add_argument(
        "--limit",
        type=int,
        help="Audit the oldest N processed documents in scope.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of the default terminal report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to also write the rendered report to disk.",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Include per-chunk flagged previews in audit mode.",
    )
    parser.add_argument(
        "--only-flagged",
        action="store_true",
        help="Suppress clean documents from the detailed audit output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    try:
        load_env_file(PROJECT_ROOT / ".env")
        settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
        configure_logging(settings.log_level)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        with get_connection(settings) as connection:
            service = ChunkAuditService(settings=settings, connection=connection)

            if _is_single_document_mode(args):
                rendered_output = _render_single_document(args, service)
            else:
                rendered_output = _render_corpus_audit(args, service)
    except (LookupError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(rendered_output)
    if args.output is not None:
        write_report(args.output, rendered_output)
    return 0


def _render_single_document(args: argparse.Namespace, service: ChunkAuditService) -> str:
    document_version_id = args.document_version_ids[0] if args.document_version_ids else None
    record_key = args.record_keys[0] if args.record_keys else None
    document = service.inspect_document(
        document_version_id=document_version_id,
        record_key=record_key,
    )
    if args.json:
        return render_document_json(document)
    return render_document_report(document)


def _render_corpus_audit(args: argparse.Namespace, service: ChunkAuditService) -> str:
    report = service.audit_scope(
        document_version_ids=args.document_version_ids,
        record_keys=args.record_keys,
        bucket_name=args.bucket_name,
        limit=args.limit,
        sample_per_bucket=args.sample_per_bucket,
    )
    scope_label = _build_scope_label(args)
    if args.json:
        return render_corpus_audit_json(
            report,
            scope_label=scope_label,
            show_chunks=args.show_chunks,
            only_flagged=args.only_flagged,
        )
    return render_corpus_audit_report(
        report,
        scope_label=scope_label,
        show_chunks=args.show_chunks,
        only_flagged=args.only_flagged,
    )


def _is_single_document_mode(args: argparse.Namespace) -> bool:
    has_single_document_id = len(args.document_version_ids or []) == 1 and not args.record_keys
    has_single_record_key = len(args.record_keys or []) == 1 and not args.document_version_ids
    return (has_single_document_id or has_single_record_key) and (
        args.limit is None and args.sample_per_bucket is None
    )


def _build_scope_label(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.document_version_ids:
        joined_ids = ", ".join(str(value) for value in args.document_version_ids)
        parts.append(f"document_version_ids=[{joined_ids}]")
    if args.record_keys:
        joined_keys = ", ".join(args.record_keys)
        parts.append(f"record_keys=[{joined_keys}]")
    if args.bucket_name:
        parts.append(f"bucket_name={args.bucket_name}")
    if args.sample_per_bucket is not None:
        parts.append(f"oldest {args.sample_per_bucket} processed docs per bucket")
    elif args.limit is not None:
        parts.append(f"oldest {args.limit} processed docs in scope")
    else:
        parts.append("all processed docs in scope")
    return "; ".join(parts)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.document_version_ids and args.record_keys:
        parser.error("Use either --document-version-id or --record-key, not both.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    if args.sample_per_bucket is not None and args.sample_per_bucket <= 0:
        parser.error("--sample-per-bucket must be greater than zero.")


if __name__ == "__main__":
    raise SystemExit(main())
