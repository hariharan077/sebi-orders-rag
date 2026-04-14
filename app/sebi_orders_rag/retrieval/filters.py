"""Metadata filter normalization and SQL clause helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..schemas import MetadataFilterInput


def normalize_metadata_filters(filters: MetadataFilterInput | None) -> MetadataFilterInput:
    """Return a normalized immutable filter object with de-duplicated tuples."""

    if filters is None:
        return MetadataFilterInput()

    document_version_ids = tuple(
        sorted({int(value) for value in filters.document_version_ids if int(value) > 0})
    )
    section_keys = tuple(sorted({value.strip() for value in filters.section_keys if value.strip()}))
    section_types = tuple(
        sorted({value.strip() for value in filters.section_types if value.strip()})
    )

    return replace(
        filters,
        record_key=_clean_optional_value(filters.record_key),
        bucket_name=_clean_optional_value(filters.bucket_name),
        document_version_ids=document_version_ids,
        section_keys=section_keys,
        section_types=section_types,
    )


def build_shared_filter_clauses(
    filters: MetadataFilterInput | None,
    *,
    source_alias: str = "sd",
    version_alias: str = "dv",
    chunk_alias: str | None = None,
    section_alias: str | None = None,
) -> tuple[list[str], list[Any]]:
    """Build SQL `WHERE` fragments and parameters for shared metadata filters."""

    normalized = normalize_metadata_filters(filters)
    clauses: list[str] = []
    params: list[Any] = []

    if normalized.record_key is not None:
        clauses.append(f"{source_alias}.record_key = %s")
        params.append(normalized.record_key)
    if normalized.bucket_name is not None:
        clauses.append(f"{source_alias}.bucket_name = %s")
        params.append(normalized.bucket_name)
    if normalized.document_version_ids:
        clauses.append(f"{version_alias}.document_version_id = ANY(%s)")
        params.append(list(normalized.document_version_ids))
    if normalized.section_keys:
        target_alias = section_alias or chunk_alias
        if target_alias is not None:
            clauses.append(f"{target_alias}.section_key = ANY(%s)")
            params.append(list(normalized.section_keys))
    if normalized.section_types:
        target_alias = section_alias or chunk_alias
        if target_alias is not None:
            clauses.append(f"{target_alias}.section_type = ANY(%s)")
            params.append(list(normalized.section_types))
    return clauses, params


def _clean_optional_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
