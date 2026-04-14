"""Manifest discovery and parsing for SEBI Orders Phase 1."""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from pathlib import Path

from ..constants import MANIFEST_COLUMNS, MANIFEST_FILE_NAME, REQUIRED_MANIFEST_COLUMNS
from ..exceptions import ManifestValidationError
from ..schemas import LoadedManifest, ManifestRow
from ..utils.files import discover_named_files
from ..utils.time import parse_optional_date, parse_required_datetime

LOGGER = logging.getLogger(__name__)


def discover_manifest_paths(data_root: Path) -> list[Path]:
    """Discover all manifest files under the configured corpus root."""

    manifest_paths = discover_named_files(data_root, MANIFEST_FILE_NAME)
    for manifest_path in manifest_paths:
        LOGGER.info("Manifest discovered: %s", manifest_path)
    return manifest_paths


def load_manifest(path: Path) -> LoadedManifest:
    """Load a single manifest file and keep only valid rows."""

    manifest_path = path.resolve(strict=False)
    inferred_bucket_name = manifest_path.parent.name

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != MANIFEST_COLUMNS:
            raise ManifestValidationError(
                f"Manifest columns do not match expected schema: {fieldnames}",
                manifest_path=manifest_path,
            )

        rows: list[ManifestRow] = []
        invalid_rows = 0
        for row_number, raw_row in enumerate(reader, start=2):
            try:
                rows.append(
                    parse_manifest_row(
                        raw_row,
                        manifest_path=manifest_path,
                        inferred_bucket_name=inferred_bucket_name,
                        row_number=row_number,
                    )
                )
            except ManifestValidationError as exc:
                invalid_rows += 1
                LOGGER.error("Invalid row in %s line %s: %s", manifest_path, row_number, exc)

    return LoadedManifest(
        path=manifest_path,
        bucket_name=inferred_bucket_name,
        rows=tuple(rows),
        invalid_rows=invalid_rows,
    )


def parse_manifest_row(
    raw_row: Mapping[str | None, str | None],
    *,
    manifest_path: Path,
    inferred_bucket_name: str,
    row_number: int,
) -> ManifestRow:
    """Parse and validate a single manifest row."""

    if None in raw_row:
        raise ManifestValidationError(
            "Row contains more columns than expected",
            manifest_path=manifest_path,
            row_number=row_number,
        )

    normalized = {
        column_name: _normalize_optional_text(raw_row.get(column_name))
        for column_name in MANIFEST_COLUMNS
    }
    pdf_url = _normalize_required_text(raw_row.get("pdf_url"), allow_empty=True)
    local_filename = _normalize_required_text(
        raw_row.get("local_filename"),
        allow_empty=True,
    )

    for column_name in REQUIRED_MANIFEST_COLUMNS:
        if normalized[column_name] is None:
            raise ManifestValidationError(
                f"Missing required value for {column_name}",
                manifest_path=manifest_path,
                row_number=row_number,
            )

    bucket_name = normalized["bucket_name"]
    if bucket_name != inferred_bucket_name:
        raise ManifestValidationError(
            f"Bucket name {bucket_name!r} does not match parent folder {inferred_bucket_name!r}",
            manifest_path=manifest_path,
            row_number=row_number,
        )

    try:
        order_date = parse_optional_date(normalized["order_date"])
        first_seen_at = parse_required_datetime(normalized["first_seen_at"], "first_seen_at")
        last_seen_at = parse_required_datetime(normalized["last_seen_at"], "last_seen_at")
    except ValueError as exc:
        raise ManifestValidationError(
            str(exc),
            manifest_path=manifest_path,
            row_number=row_number,
        ) from exc

    if first_seen_at > last_seen_at:
        raise ManifestValidationError(
            "first_seen_at cannot be after last_seen_at",
            manifest_path=manifest_path,
            row_number=row_number,
        )

    local_path = (manifest_path.parent / local_filename).resolve(strict=False)

    return ManifestRow(
        record_key=normalized["record_key"],
        bucket_name=inferred_bucket_name,
        order_date=order_date,
        title=normalized["title"],
        external_record_id=normalized["external_record_id"],
        detail_url=normalized["detail_url"],
        pdf_url=pdf_url,
        local_filename=local_filename,
        manifest_status=normalized["status"],
        error=normalized["error"],
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
        manifest_path=manifest_path,
        local_path=local_path,
        row_number=row_number,
    )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_required_text(value: str | None, *, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise ValueError("Required text value is missing")
    stripped = value.strip()
    if stripped or allow_empty:
        return stripped
    raise ValueError("Required text value is empty")
