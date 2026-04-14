"""Application service for Phase 1 manifest scanning and registration."""

from __future__ import annotations

import logging
from typing import Any

from ..config import SebiOrdersRagSettings
from ..repositories.documents import DocumentRepository
from ..schemas import ManifestRow, Phase1Summary
from .fingerprint import inspect_local_file
from .manifest_loader import discover_manifest_paths, load_manifest
from .planner import plan_manifest_row

LOGGER = logging.getLogger(__name__)


class Phase1IngestionService:
    """Coordinate manifest loading, fingerprinting, planning, and persistence."""

    def __init__(self, *, settings: SebiOrdersRagSettings, connection: Any) -> None:
        self._settings = settings
        self._documents = DocumentRepository(connection)

    def run(self, *, apply: bool) -> Phase1Summary:
        """Run Phase 1 across all manifests under the configured corpus root."""

        summary = Phase1Summary()
        manifest_paths = discover_manifest_paths(self._settings.data_root)
        summary.manifests_found = len(manifest_paths)

        for manifest_path in manifest_paths:
            loaded_manifest = load_manifest(manifest_path)
            summary.invalid_rows += loaded_manifest.invalid_rows
            for manifest_row in loaded_manifest.rows:
                self._process_manifest_row(manifest_row, summary=summary, apply=apply)

        return summary

    def _process_manifest_row(
        self,
        manifest_row: ManifestRow,
        *,
        summary: Phase1Summary,
        apply: bool,
    ) -> None:
        summary.rows_processed += 1
        file_snapshot = inspect_local_file(manifest_row.local_path)
        existing_document = self._documents.get_document_by_record_key(manifest_row.record_key)
        existing_version = None
        if file_snapshot.exists:
            summary.pdfs_present += 1
        else:
            summary.pdfs_missing += 1

        if existing_document is not None and file_snapshot.fingerprint is not None:
            existing_version = self._documents.get_version_by_document_id_and_file_sha256(
                document_id=existing_document.document_id,
                file_sha256=file_snapshot.fingerprint.file_sha256,
            )

        decision = plan_manifest_row(
            manifest_row,
            file_snapshot,
            existing_document=existing_document,
            existing_version=existing_version,
            parser_name=self._settings.parser_name,
            parser_version=self._settings.parser_version,
        )

        if decision.create_document:
            summary.documents_inserted += 1
            self._log_info(apply, "New document planned for %s", manifest_row.record_key)

        if decision.action == "skip_missing_file":
            summary.rows_skipped_due_to_missing_files += 1
            LOGGER.warning(
                "Missing PDF for %s at %s",
                manifest_row.record_key,
                manifest_row.local_path,
            )
        elif decision.action == "reuse_version":
            summary.existing_versions_reused += 1
            self._log_info(
                apply,
                "Existing version reused for %s",
                manifest_row.record_key,
            )
        elif decision.action == "create_document_and_version":
            summary.document_versions_inserted += 1
            self._log_info(apply, "New version planned for %s", manifest_row.record_key)
        elif decision.action == "create_version":
            summary.document_versions_inserted += 1
            self._log_info(apply, "New version planned for %s", manifest_row.record_key)

        document_record = existing_document
        if not apply:
            return

        if decision.create_document:
            document_record = self._documents.create_document(
                record_key=manifest_row.record_key,
                bucket_name=manifest_row.bucket_name,
                external_record_id=manifest_row.external_record_id,
                first_seen_at=manifest_row.first_seen_at,
                last_seen_at=manifest_row.last_seen_at,
            )
        elif document_record is None:
            raise RuntimeError("Existing document is required when create_document is false")

        current_version_id = decision.current_version_id
        if decision.version_to_create is not None:
            version_record = self._documents.create_document_version(
                document_id=document_record.document_id,
                version=decision.version_to_create,
            )
            current_version_id = version_record.document_version_id

        self._documents.update_document_seen_timestamps_and_current_version(
            document_id=document_record.document_id,
            bucket_name=manifest_row.bucket_name,
            external_record_id=manifest_row.external_record_id,
            first_seen_at=manifest_row.first_seen_at,
            last_seen_at=manifest_row.last_seen_at,
            current_version_id=current_version_id,
        )

    def _log_info(self, apply: bool, message: str, *args: object) -> None:
        prefix = "" if apply else "[dry-run] "
        LOGGER.info(prefix + message, *args)
