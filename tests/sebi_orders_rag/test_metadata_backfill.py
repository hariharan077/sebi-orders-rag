from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from app.sebi_orders_rag.metadata.models import (
    ExtractedMetadataBundle,
    ExtractedLegalProvision,
    ExtractedNumericFact,
    ExtractedOrderMetadata,
    ExtractedPriceMovement,
    MetadataExtractionTarget,
    StoredLegalProvision,
    StoredNumericFact,
    StoredOrderMetadata,
    StoredPriceMovement,
)
from scripts import sebi_order_metadata_extract


class MetadataBackfillTests(unittest.TestCase):
    def test_run_backfill_is_idempotent_on_rerun(self) -> None:
        repository = _FakeMetadataRepository()
        connection = _FakeConnection()
        targets = (
            MetadataExtractionTarget(
                document_version_id=101,
                document_id=201,
                record_key="external:yash",
                title="Order in the matter of Yash Trading Academy",
                order_date=date(2026, 3, 27),
            ),
        )

        with patch.object(
            sebi_order_metadata_extract,
            "extract_order_metadata_bundle",
            return_value=ExtractedMetadataBundle(
                order_metadata=ExtractedOrderMetadata(
                    document_version_id=101,
                    signatory_name="Ananth Narayan G.",
                    signatory_designation="Whole Time Member",
                    order_date=date(2026, 3, 27),
                    place="Mumbai",
                ),
                legal_provisions=(
                    ExtractedLegalProvision(
                        document_version_id=101,
                        statute_name="SEBI Act, 1992",
                        section_or_regulation="Section 12A(a)",
                        provision_type="section",
                        text_snippet="Section 12A(a) of the SEBI Act, 1992",
                        page_start=3,
                        page_end=3,
                    ),
                ),
                numeric_facts=(
                    ExtractedNumericFact(
                        document_version_id=101,
                        fact_type="listing_price",
                        subject="Yash Trading Academy",
                        value_text="Rs.12/share",
                        value_numeric=12.0,
                        unit="INR/share",
                        page_start=2,
                        page_end=2,
                    ),
                ),
                price_movements=(
                    ExtractedPriceMovement(
                        document_version_id=101,
                        period_label="Patch 1",
                        period_start_text="March 1, 2026",
                        period_end_text="March 27, 2026",
                        start_price=12.0,
                        end_price=18.0,
                        pct_change=50.0,
                        page_start=3,
                        page_end=3,
                    ),
                ),
            ),
        ):
            first_summary = sebi_order_metadata_extract.run_backfill(
                connection=connection,
                repository=repository,
                targets=targets,
                apply=True,
            )
            second_summary = sebi_order_metadata_extract.run_backfill(
                connection=connection,
                repository=repository,
                targets=targets,
                apply=True,
            )

        self.assertEqual(first_summary.docs_scanned, 1)
        self.assertEqual(first_summary.docs_updated, 1)
        self.assertEqual(first_summary.docs_skipped, 0)
        self.assertEqual(first_summary.signatory_rows_extracted, 1)
        self.assertEqual(first_summary.legal_provision_rows_extracted, 1)
        self.assertEqual(first_summary.numeric_fact_rows_extracted, 1)
        self.assertEqual(first_summary.price_movement_rows_extracted, 1)
        self.assertEqual(second_summary.docs_scanned, 1)
        self.assertEqual(second_summary.docs_updated, 0)
        self.assertEqual(second_summary.docs_skipped, 1)
        self.assertEqual(connection.commits, 1)


class _FakeConnection:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        return None


class _FakeMetadataRepository:
    def __init__(self) -> None:
        self._metadata: dict[int, StoredOrderMetadata] = {}
        self._provisions: dict[int, tuple[StoredLegalProvision, ...]] = {}
        self._numeric_facts: dict[int, tuple[StoredNumericFact, ...]] = {}
        self._price_movements: dict[int, tuple[StoredPriceMovement, ...]] = {}

    def load_pages(self, *, document_version_id: int):
        return ()

    def load_chunks(self, *, document_version_id: int):
        return ()

    def fetch_order_metadata(self, *, document_version_ids):
        return tuple(
            self._metadata[document_version_id]
            for document_version_id in document_version_ids
            if document_version_id in self._metadata
        )

    def fetch_legal_provisions(self, *, document_version_ids):
        rows: list[StoredLegalProvision] = []
        for document_version_id in document_version_ids:
            rows.extend(self._provisions.get(document_version_id, ()))
        return tuple(rows)

    def fetch_numeric_facts(self, *, document_version_ids, fact_types=None):
        rows: list[StoredNumericFact] = []
        for document_version_id in document_version_ids:
            rows.extend(self._numeric_facts.get(document_version_id, ()))
        if not fact_types:
            return tuple(rows)
        return tuple(row for row in rows if row.fact_type in set(fact_types))

    def fetch_price_movements(self, *, document_version_ids):
        rows: list[StoredPriceMovement] = []
        for document_version_id in document_version_ids:
            rows.extend(self._price_movements.get(document_version_id, ()))
        return tuple(rows)

    def upsert_order_metadata(self, metadata: ExtractedOrderMetadata) -> None:
        self._metadata[metadata.document_version_id] = StoredOrderMetadata(
            document_version_id=metadata.document_version_id,
            document_id=201,
            record_key="external:yash",
            title="Order in the matter of Yash Trading Academy",
            detail_url="https://example.com/detail",
            pdf_url="https://example.com/pdf",
            signatory_name=metadata.signatory_name,
            signatory_designation=metadata.signatory_designation,
            order_date=metadata.order_date,
            place=metadata.place,
        )

    def replace_legal_provisions(self, *, document_version_id: int, provisions):
        self._provisions[document_version_id] = tuple(
            StoredLegalProvision(
                provision_id=index,
                document_version_id=document_version_id,
                document_id=201,
                record_key="external:yash",
                title="Order in the matter of Yash Trading Academy",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                statute_name=row.statute_name,
                section_or_regulation=row.section_or_regulation,
                provision_type=row.provision_type,
                text_snippet=row.text_snippet,
                page_start=row.page_start,
                page_end=row.page_end,
                row_sha256=row.row_sha256,
            )
            for index, row in enumerate(provisions, start=1)
        )

    def replace_numeric_facts(self, *, document_version_id: int, facts):
        self._numeric_facts[document_version_id] = tuple(
            StoredNumericFact(
                numeric_fact_id=index,
                document_version_id=document_version_id,
                document_id=201,
                record_key="external:yash",
                title="Order in the matter of Yash Trading Academy",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                fact_type=row.fact_type,
                subject=row.subject,
                value_text=row.value_text,
                value_numeric=row.value_numeric,
                unit=row.unit,
                context_label=row.context_label,
                page_start=row.page_start,
                page_end=row.page_end,
                row_sha256=row.row_sha256,
            )
            for index, row in enumerate(facts, start=1)
        )

    def replace_price_movements(self, *, document_version_id: int, price_movements):
        self._price_movements[document_version_id] = tuple(
            StoredPriceMovement(
                price_movement_id=index,
                document_version_id=document_version_id,
                document_id=201,
                record_key="external:yash",
                title="Order in the matter of Yash Trading Academy",
                detail_url="https://example.com/detail",
                pdf_url="https://example.com/pdf",
                period_label=row.period_label,
                period_start_text=row.period_start_text,
                period_end_text=row.period_end_text,
                start_price=row.start_price,
                high_price=row.high_price,
                low_price=row.low_price,
                end_price=row.end_price,
                pct_change=row.pct_change,
                rationale=row.rationale,
                page_start=row.page_start,
                page_end=row.page_end,
                row_sha256=row.row_sha256,
            )
            for index, row in enumerate(price_movements, start=1)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
