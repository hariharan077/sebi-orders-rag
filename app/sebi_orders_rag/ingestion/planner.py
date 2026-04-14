"""Pure planning logic for Phase 1 document and version registration."""

from __future__ import annotations

from ..constants import PENDING_STATUS
from ..schemas import (
    DocumentVersionRecord,
    LocalFileSnapshot,
    ManifestRow,
    PlannedDocumentVersionCreate,
    PlannerDecision,
    SourceDocumentRecord,
)


def plan_manifest_row(
    manifest_row: ManifestRow,
    file_snapshot: LocalFileSnapshot,
    *,
    existing_document: SourceDocumentRecord | None,
    existing_version: DocumentVersionRecord | None,
    parser_name: str,
    parser_version: str,
) -> PlannerDecision:
    """Decide whether a manifest row should create or reuse DB records."""

    if not file_snapshot.exists or file_snapshot.fingerprint is None:
        return PlannerDecision(
            action="skip_missing_file",
            create_document=existing_document is None,
            create_version=False,
            reuse_existing_version=False,
            current_version_id=(
                existing_document.current_version_id
                if existing_document is not None
                else None
            ),
            version_to_create=None,
        )

    if existing_version is not None:
        return PlannerDecision(
            action="reuse_version",
            create_document=False,
            create_version=False,
            reuse_existing_version=True,
            current_version_id=existing_version.document_version_id,
            version_to_create=None,
        )

    version_to_create = PlannedDocumentVersionCreate(
        order_date=manifest_row.order_date,
        title=manifest_row.title,
        detail_url=manifest_row.detail_url,
        pdf_url=manifest_row.pdf_url,
        local_filename=manifest_row.local_filename,
        local_path=str(file_snapshot.path),
        file_size_bytes=file_snapshot.fingerprint.file_size_bytes,
        file_sha256=file_snapshot.fingerprint.file_sha256,
        manifest_status=manifest_row.manifest_status,
        parser_name=parser_name,
        parser_version=parser_version,
        extraction_status=PENDING_STATUS,
        ingest_status=PENDING_STATUS,
    )

    if existing_document is None:
        return PlannerDecision(
            action="create_document_and_version",
            create_document=True,
            create_version=True,
            reuse_existing_version=False,
            current_version_id=None,
            version_to_create=version_to_create,
        )

    return PlannerDecision(
        action="create_version",
        create_document=False,
        create_version=True,
        reuse_existing_version=False,
        current_version_id=None,
        version_to_create=version_to_create,
    )
