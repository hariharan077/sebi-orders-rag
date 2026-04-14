from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.sebi_orders_rag.ingestion.fingerprint import inspect_local_file
from app.sebi_orders_rag.ingestion.manifest_loader import load_manifest


class ManifestLoaderTests(unittest.TestCase):
    def test_manifest_row_parsing_normalizes_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bucket_dir = Path(temp_dir) / "orders-of-ao"
            bucket_dir.mkdir()
            manifest_path = bucket_dir / "orders_manifest.csv"
            manifest_path.write_text(
                "\n".join(
                    [
                        "record_key,bucket_name,order_date,title,external_record_id,detail_url,pdf_url,local_filename,status,error,first_seen_at,last_seen_at",
                        "external:100846,orders-of-ao,2026-04-09,Sample Title,,,"
                        "https://example.com/sample.pdf,sample.pdf,downloaded,,"
                        "2026-04-09T18:07:23.177332+00:00,2026-04-09T18:07:23.177332+00:00",
                    ]
                ),
                encoding="utf-8",
            )

            loaded_manifest = load_manifest(manifest_path)

            self.assertEqual(loaded_manifest.invalid_rows, 0)
            self.assertEqual(len(loaded_manifest.rows), 1)
            row = loaded_manifest.rows[0]
            self.assertEqual(row.record_key, "external:100846")
            self.assertEqual(row.bucket_name, "orders-of-ao")
            self.assertIsNone(row.external_record_id)
            self.assertIsNone(row.detail_url)
            self.assertEqual(row.pdf_url, "https://example.com/sample.pdf")
            self.assertEqual(row.local_path, (bucket_dir / "sample.pdf").resolve(strict=False))

    def test_missing_file_is_reported_as_not_ingestible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bucket_dir = Path(temp_dir) / "orders-of-ao"
            bucket_dir.mkdir()
            manifest_path = bucket_dir / "orders_manifest.csv"
            manifest_path.write_text(
                "\n".join(
                    [
                        "record_key,bucket_name,order_date,title,external_record_id,detail_url,pdf_url,local_filename,status,error,first_seen_at,last_seen_at",
                        "external:100847,orders-of-ao,2026-04-10,Missing File Title,100847,"
                        "https://example.com/detail,https://example.com/missing.pdf,missing.pdf,downloaded,,"
                        "2026-04-09T18:07:23.177332+00:00,2026-04-09T18:07:23.177332+00:00",
                    ]
                ),
                encoding="utf-8",
            )

            loaded_manifest = load_manifest(manifest_path)
            snapshot = inspect_local_file(loaded_manifest.rows[0].local_path)

            self.assertFalse(snapshot.exists)
            self.assertIsNone(snapshot.fingerprint)

    def test_blank_pdf_url_is_retained_for_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bucket_dir = Path(temp_dir) / "orders-of-sat"
            bucket_dir.mkdir()
            manifest_path = bucket_dir / "orders_manifest.csv"
            manifest_path.write_text(
                "\n".join(
                    [
                        "record_key,bucket_name,order_date,title,external_record_id,detail_url,pdf_url,local_filename,status,error,first_seen_at,last_seen_at",
                        "external:29944,orders-of-sat,2015-06-09,Legacy Row,29944,https://example.com/detail,,legacy.pdf,downloaded,,"
                        "2026-04-09T18:07:23.177332+00:00,2026-04-09T18:07:23.177332+00:00",
                    ]
                ),
                encoding="utf-8",
            )

            loaded_manifest = load_manifest(manifest_path)

            self.assertEqual(loaded_manifest.invalid_rows, 0)
            self.assertEqual(loaded_manifest.rows[0].pdf_url, "")

    def test_hash_computation_uses_sha256_and_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.pdf"
            file_bytes = b"%PDF-1.4\nhello world\n"
            file_path.write_bytes(file_bytes)

            snapshot = inspect_local_file(file_path)

            self.assertTrue(snapshot.exists)
            self.assertIsNotNone(snapshot.fingerprint)
            assert snapshot.fingerprint is not None
            self.assertEqual(snapshot.fingerprint.file_size_bytes, len(file_bytes))
            self.assertEqual(
                snapshot.fingerprint.file_sha256,
                hashlib.sha256(file_bytes).hexdigest(),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
