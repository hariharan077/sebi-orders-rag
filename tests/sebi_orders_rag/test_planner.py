from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import unittest

from app.sebi_orders_rag.ingestion.planner import plan_manifest_row
from app.sebi_orders_rag.schemas import (
    DocumentVersionRecord,
    FileFingerprint,
    LocalFileSnapshot,
    ManifestRow,
    SourceDocumentRecord,
)


class PlannerTests(unittest.TestCase):
    def test_new_record_key_creates_document_and_version(self) -> None:
        decision = plan_manifest_row(
            self._manifest_row(),
            self._file_snapshot("a" * 64),
            existing_document=None,
            existing_version=None,
            parser_name="sebi_orders_pdf_parser",
            parser_version="v1",
        )

        self.assertEqual(decision.action, "create_document_and_version")
        self.assertTrue(decision.create_document)
        self.assertTrue(decision.create_version)
        self.assertFalse(decision.reuse_existing_version)
        self.assertIsNotNone(decision.version_to_create)
        assert decision.version_to_create is not None
        self.assertEqual(decision.version_to_create.file_sha256, "a" * 64)
        self.assertEqual(decision.version_to_create.ingest_status, "pending")

    def test_existing_record_key_with_same_hash_reuses_existing_version(self) -> None:
        existing_document = self._source_document()
        existing_version = self._document_version(existing_document.document_id, "b" * 64)

        decision = plan_manifest_row(
            self._manifest_row(),
            self._file_snapshot("b" * 64),
            existing_document=existing_document,
            existing_version=existing_version,
            parser_name="sebi_orders_pdf_parser",
            parser_version="v1",
        )

        self.assertEqual(decision.action, "reuse_version")
        self.assertFalse(decision.create_document)
        self.assertFalse(decision.create_version)
        self.assertTrue(decision.reuse_existing_version)
        self.assertEqual(decision.current_version_id, existing_version.document_version_id)
        self.assertIsNone(decision.version_to_create)

    def test_existing_record_key_with_new_hash_creates_new_version(self) -> None:
        existing_document = self._source_document()

        decision = plan_manifest_row(
            self._manifest_row(),
            self._file_snapshot("c" * 64),
            existing_document=existing_document,
            existing_version=None,
            parser_name="sebi_orders_pdf_parser",
            parser_version="v1",
        )

        self.assertEqual(decision.action, "create_version")
        self.assertFalse(decision.create_document)
        self.assertTrue(decision.create_version)
        self.assertFalse(decision.reuse_existing_version)
        self.assertIsNotNone(decision.version_to_create)
        assert decision.version_to_create is not None
        self.assertEqual(decision.version_to_create.file_sha256, "c" * 64)

    def test_missing_file_still_creates_logical_document(self) -> None:
        decision = plan_manifest_row(
            self._manifest_row(),
            LocalFileSnapshot(path=Path("/tmp/missing.pdf"), exists=False, fingerprint=None),
            existing_document=None,
            existing_version=None,
            parser_name="sebi_orders_pdf_parser",
            parser_version="v1",
        )

        self.assertEqual(decision.action, "skip_missing_file")
        self.assertTrue(decision.create_document)
        self.assertFalse(decision.create_version)
        self.assertIsNone(decision.current_version_id)

    @staticmethod
    def _manifest_row() -> ManifestRow:
        return ManifestRow(
            record_key="external:100846",
            bucket_name="orders-of-ao",
            order_date=date(2026, 4, 9),
            title="Sample Title",
            external_record_id="100846",
            detail_url="https://example.com/detail",
            pdf_url="https://example.com/file.pdf",
            local_filename="sample.pdf",
            manifest_status="downloaded",
            error=None,
            first_seen_at=datetime(2026, 4, 9, 18, 7, 23, tzinfo=timezone.utc),
            last_seen_at=datetime(2026, 4, 9, 18, 7, 23, tzinfo=timezone.utc),
            manifest_path=Path("/tmp/orders_manifest.csv"),
            local_path=Path("/tmp/sample.pdf"),
            row_number=2,
        )

    @staticmethod
    def _file_snapshot(file_sha256: str) -> LocalFileSnapshot:
        return LocalFileSnapshot(
            path=Path("/tmp/sample.pdf"),
            exists=True,
            fingerprint=FileFingerprint(file_size_bytes=1234, file_sha256=file_sha256),
        )

    @staticmethod
    def _source_document() -> SourceDocumentRecord:
        return SourceDocumentRecord(
            document_id=42,
            record_key="external:100846",
            bucket_name="orders-of-ao",
            external_record_id="100846",
            first_seen_at=datetime(2026, 4, 9, 18, 7, 23, tzinfo=timezone.utc),
            last_seen_at=datetime(2026, 4, 9, 18, 7, 23, tzinfo=timezone.utc),
            current_version_id=7,
            is_active=True,
        )

    @staticmethod
    def _document_version(document_id: int, file_sha256: str) -> DocumentVersionRecord:
        timestamp = datetime(2026, 4, 9, 18, 7, 23, tzinfo=timezone.utc)
        return DocumentVersionRecord(
            document_version_id=7,
            document_id=document_id,
            order_date=date(2026, 4, 9),
            title="Sample Title",
            detail_url="https://example.com/detail",
            pdf_url="https://example.com/file.pdf",
            local_filename="sample.pdf",
            local_path="/tmp/sample.pdf",
            file_size_bytes=1234,
            file_sha256=file_sha256,
            manifest_status="downloaded",
            parser_name="sebi_orders_pdf_parser",
            parser_version="v1",
            extraction_status="pending",
            ocr_used=False,
            page_count=None,
            extracted_char_count=None,
            ingest_status="pending",
            ingest_error=None,
            ingested_at=None,
            created_at=timestamp,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
