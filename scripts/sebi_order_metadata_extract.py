#!/usr/bin/env python3
"""Extract signatory and legal-provision metadata from processed SEBI orders."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.metadata import extract_order_metadata_bundle
from app.sebi_orders_rag.metadata.models import (
    ExtractedLegalProvision,
    ExtractedNumericFact,
    ExtractedOrderMetadata,
    ExtractedPriceMovement,
)
from app.sebi_orders_rag.repositories.metadata import OrderMetadataRepository

@dataclass
class BackfillSummary:
    docs_scanned: int = 0
    docs_updated: int = 0
    docs_skipped: int = 0
    signatory_rows_extracted: int = 0
    legal_provision_rows_extracted: int = 0
    numeric_fact_rows_extracted: int = 0
    price_movement_rows_extracted: int = 0
    failures: int = 0

    def as_lines(self, *, apply: bool) -> list[str]:
        mode = "apply" if apply else "dry-run"
        return [
            f"mode: {mode}",
            f"docs scanned: {self.docs_scanned}",
            f"docs updated: {self.docs_updated}",
            f"docs skipped: {self.docs_skipped}",
            f"signatory rows extracted: {self.signatory_rows_extracted}",
            f"legal provision rows extracted: {self.legal_provision_rows_extracted}",
            f"numeric fact rows extracted: {self.numeric_fact_rows_extracted}",
            f"price movement rows extracted: {self.price_movement_rows_extracted}",
            f"failures: {self.failures}",
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-key", default=None, help="Optional record_key to process.")
    parser.add_argument(
        "--document-version-id",
        type=int,
        default=None,
        help="Optional document_version_id to process.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for batch extraction.")
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Backfill across the full processed corpus, including documents that already have metadata.",
    )
    parser.add_argument("--apply", action="store_true", help="Persist extracted metadata to Postgres.")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include documents that already have extracted metadata rows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
        settings = SebiOrdersRagSettings.from_env(data_root_override=PROJECT_ROOT)
        with get_connection(settings) as connection:
            initialize_phase4_schema(connection, settings)
            repository = OrderMetadataRepository(connection)
            targets = repository.list_extraction_targets(
                record_key=args.record_key,
                document_version_id=args.document_version_id,
                limit=args.limit,
                include_existing=(
                    args.full_corpus
                    or args.include_existing
                    or bool(args.document_version_id or args.record_key)
                ),
                processed_only=True,
            )
            if not targets:
                print("No processed metadata extraction targets matched the request.")
                return 0

            print(
                f"Processing {len(targets)} processed document version(s)"
                f"{' with apply' if args.apply else ' in dry-run mode'}."
            )
            summary = run_backfill(
                connection=connection,
                repository=repository,
                targets=targets,
                apply=args.apply,
            )

            if args.apply:
                coverage = repository.fetch_backfill_coverage()
                print("")
                print("metadata backfill coverage:")
                print(
                    "processed_docs={processed_docs} metadata_docs={metadata_docs} "
                    "signatory_name_docs={signatory_name_docs} "
                    "signatory_designation_docs={signatory_designation_docs} "
                    "order_date_docs={order_date_docs} place_docs={place_docs} "
                    "legal_provision_docs={legal_provision_docs} "
                    "legal_provision_rows={legal_provision_rows} "
                    "numeric_fact_docs={numeric_fact_docs} numeric_fact_rows={numeric_fact_rows} "
                    "listing_price_docs={listing_price_docs} highest_price_docs={highest_price_docs} "
                    "settlement_amount_docs={settlement_amount_docs} "
                    "price_movement_docs={price_movement_docs} price_movement_rows={price_movement_rows}".format(
                        **coverage
                    )
                )

            print("")
            for line in summary.as_lines(apply=args.apply):
                print(line)
            return 0 if summary.failures == 0 else 1
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(
            "Metadata extraction finished with a safe failure summary: "
            f"{type(exc).__name__}: {_safe_error_text(exc)}"
        )
        return 1


def run_backfill(
    *,
    connection,
    repository: OrderMetadataRepository,
    targets: Iterable[object],
    apply: bool,
) -> BackfillSummary:
    """Run one metadata backfill loop and print a stable per-document status line."""

    summary = BackfillSummary()
    for target in targets:
        summary.docs_scanned += 1
        try:
            pages = repository.load_pages(document_version_id=target.document_version_id)
            chunks = repository.load_chunks(document_version_id=target.document_version_id)
            bundle = extract_order_metadata_bundle(
                document_version_id=target.document_version_id,
                pages=pages,
                chunks=chunks,
                fallback_order_date=target.order_date,
                title=target.title,
            )
            metadata = bundle.order_metadata
            provisions = bundle.legal_provisions
            numeric_facts = bundle.numeric_facts
            price_movements = bundle.price_movements
            summary.signatory_rows_extracted += int(_has_metadata_row(metadata))
            summary.legal_provision_rows_extracted += len(provisions)
            summary.numeric_fact_rows_extracted += len(numeric_facts)
            summary.price_movement_rows_extracted += len(price_movements)

            existing_metadata = repository.fetch_order_metadata(
                document_version_ids=(target.document_version_id,)
            )
            existing_provisions = repository.fetch_legal_provisions(
                document_version_ids=(target.document_version_id,)
            )
            existing_numeric_facts = repository.fetch_numeric_facts(
                document_version_ids=(target.document_version_id,)
            )
            existing_price_movements = repository.fetch_price_movements(
                document_version_ids=(target.document_version_id,)
            )
            changed = _metadata_signature(metadata) != _stored_metadata_signature(existing_metadata[:1])
            changed = changed or (
                _provision_signature_set(provisions)
                != _stored_provision_signature_set(existing_provisions)
            )
            changed = changed or (
                _numeric_fact_signature_set(numeric_facts)
                != _stored_numeric_fact_signature_set(existing_numeric_facts)
            )
            changed = changed or (
                _price_movement_signature_set(price_movements)
                != _stored_price_movement_signature_set(existing_price_movements)
            )

            if changed:
                summary.docs_updated += 1
                status = "updated"
                if apply:
                    repository.upsert_order_metadata(metadata)
                    repository.replace_legal_provisions(
                        document_version_id=target.document_version_id,
                        provisions=provisions,
                    )
                    repository.replace_numeric_facts(
                        document_version_id=target.document_version_id,
                        facts=numeric_facts,
                    )
                    repository.replace_price_movements(
                        document_version_id=target.document_version_id,
                        price_movements=price_movements,
                    )
                    if hasattr(connection, "commit"):
                        connection.commit()
            else:
                summary.docs_skipped += 1
                status = "skipped"

            print(
                f"{target.record_key} | dv={target.document_version_id} | {status}"
                f" | signatory={metadata.signatory_name or '-'}"
                f" | designation={metadata.signatory_designation or '-'}"
                f" | order_date={metadata.order_date.isoformat() if metadata.order_date else '-'}"
                f" | place={metadata.place or '-'}"
                f" | provisions={len(provisions)}"
                f" | numeric_facts={len(numeric_facts)}"
                f" | price_movements={len(price_movements)}"
            )
        except Exception as exc:  # pragma: no cover - defensive CLI path
            summary.failures += 1
            summary.docs_skipped += 1
            if apply and hasattr(connection, "rollback"):
                connection.rollback()
            print(
                f"{target.record_key} | dv={target.document_version_id} | failure"
                f" | {type(exc).__name__}: {_safe_error_text(exc)}"
            )
    return summary


def _has_metadata_row(metadata: ExtractedOrderMetadata) -> bool:
    return any(
        value not in (None, "", ())
        for value in (
            metadata.signatory_name,
            metadata.signatory_designation,
            metadata.order_date,
            metadata.place,
            metadata.authority_panel,
        )
    )


def _metadata_signature(metadata: ExtractedOrderMetadata) -> tuple[object, ...]:
    return (
        metadata.signatory_name or None,
        metadata.signatory_designation or None,
        metadata.order_date.isoformat() if metadata.order_date else None,
        metadata.place or None,
    )


def _stored_metadata_signature(existing_rows: Iterable[object]) -> tuple[object, ...]:
    existing = next(iter(existing_rows), None)
    if existing is None:
        return (None, None, None, None)
    return (
        getattr(existing, "signatory_name", None) or None,
        getattr(existing, "signatory_designation", None) or None,
        getattr(existing, "order_date", None).isoformat()
        if getattr(existing, "order_date", None) is not None
        else None,
        getattr(existing, "place", None) or None,
    )


def _provision_signature_set(
    provisions: Iterable[ExtractedLegalProvision],
) -> set[tuple[object, ...]]:
    return {
        (
            row.statute_name,
            row.section_or_regulation,
            row.provision_type,
            row.text_snippet,
            row.page_start,
            row.page_end,
        )
        for row in provisions
    }


def _stored_provision_signature_set(existing_rows: Iterable[object]) -> set[tuple[object, ...]]:
    return {
        (
            getattr(row, "statute_name", None),
            getattr(row, "section_or_regulation", None),
            getattr(row, "provision_type", None),
            getattr(row, "text_snippet", None),
            getattr(row, "page_start", None),
            getattr(row, "page_end", None),
        )
        for row in existing_rows
    }


def _numeric_fact_signature_set(
    facts: Iterable[ExtractedNumericFact],
) -> set[tuple[object, ...]]:
    return {
        (
            row.fact_type,
            row.subject,
            row.value_text,
            row.value_numeric,
            row.unit,
            row.context_label,
            row.page_start,
            row.page_end,
        )
        for row in facts
    }


def _stored_numeric_fact_signature_set(existing_rows: Iterable[object]) -> set[tuple[object, ...]]:
    return {
        (
            getattr(row, "fact_type", None),
            getattr(row, "subject", None),
            getattr(row, "value_text", None),
            getattr(row, "value_numeric", None),
            getattr(row, "unit", None),
            getattr(row, "context_label", None),
            getattr(row, "page_start", None),
            getattr(row, "page_end", None),
        )
        for row in existing_rows
    }


def _price_movement_signature_set(
    rows: Iterable[ExtractedPriceMovement],
) -> set[tuple[object, ...]]:
    return {
        (
            row.period_label,
            row.period_start_text,
            row.period_end_text,
            row.start_price,
            row.high_price,
            row.low_price,
            row.end_price,
            row.pct_change,
            row.rationale,
            row.page_start,
            row.page_end,
        )
        for row in rows
    }


def _stored_price_movement_signature_set(existing_rows: Iterable[object]) -> set[tuple[object, ...]]:
    return {
        (
            getattr(row, "period_label", None),
            getattr(row, "period_start_text", None),
            getattr(row, "period_end_text", None),
            getattr(row, "start_price", None),
            getattr(row, "high_price", None),
            getattr(row, "low_price", None),
            getattr(row, "end_price", None),
            getattr(row, "pct_change", None),
            getattr(row, "rationale", None),
            getattr(row, "page_start", None),
            getattr(row, "page_end", None),
        )
        for row in existing_rows
    }


def _safe_error_text(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if not text:
        return "no details available"
    if len(text) <= 200:
        return text
    return text[:197].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
