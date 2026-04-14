"""Typed records for structured SEBI reference data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def normalize_whitespace(value: str | None) -> str | None:
    """Collapse repeated whitespace while preserving semantic punctuation."""

    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def normalize_email(value: str | None) -> str | None:
    """Normalize SEBI-style obfuscated email text to a standard address."""

    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    cleaned = cleaned.replace("[at]", "@").replace("[dot]", ".")
    cleaned = cleaned.replace("(at)", "@").replace("(dot)", ".")
    cleaned = re.sub(r"\s*\[\s*at\s*\]\s*", "@", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[\s*dot\s*\]\s*", ".", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*@\s*", "@", cleaned)
    cleaned = re.sub(r"\s*\.\s*", ".", cleaned)
    cleaned = cleaned.replace("mailto:", "").strip(" ;,")
    return cleaned.lower()


def normalize_phone(value: str | None) -> str | None:
    """Normalize phone text for storage while keeping official formatting readable."""

    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    cleaned = cleaned.replace(" / ", "/").replace(" ,", ",")
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
    return cleaned


def normalize_person_name(value: str | None) -> str | None:
    """Convert uppercase staff names to title case while preserving initials."""

    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    if any(char.islower() for char in cleaned):
        return cleaned

    parts = []
    for part in cleaned.split(" "):
        if not part:
            continue
        if len(part) <= 3 and "." in part:
            parts.append(part.upper())
            continue
        if len(part) == 1:
            parts.append(part.upper())
            continue
        if "-" in part:
            parts.append("-".join(sub.capitalize() for sub in part.split("-")))
            continue
        if "'" in part:
            parts.append("'".join(sub.capitalize() for sub in part.split("'")))
            continue
        parts.append(part.capitalize())
    return " ".join(parts)


def row_sha256(payload: dict[str, Any]) -> str:
    """Return a stable row hash over normalized field values."""

    normalized_payload = {
        key: normalize_whitespace(str(value)) if isinstance(value, str) else value
        for key, value in payload.items()
        if value is not None and value != ""
    }
    encoded = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DirectorySourceDefinition:
    """One configured official SEBI structured-reference source page."""

    source_type: str
    title: str
    url: str


@dataclass(frozen=True)
class FetchedDirectorySource:
    """One fetched official source page plus content fingerprint metadata."""

    source_type: str
    title: str
    source_url: str
    raw_html: str
    content_sha256: str
    fetched_at: datetime | None = None


@dataclass(frozen=True)
class ReferenceSnapshotRecord:
    """Persisted snapshot metadata for one fetched source page."""

    snapshot_id: int
    source_type: str
    source_url: str
    fetched_at: datetime
    content_sha256: str
    fetch_status: str
    parse_status: str
    raw_html: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DirectoryPersonRecord:
    """Structured person row derived from an official SEBI source page."""

    source_type: str
    source_url: str
    canonical_name: str
    designation: str | None = None
    role_group: str | None = None
    department_name: str | None = None
    office_name: str | None = None
    email: str | None = None
    phone: str | None = None
    date_of_joining: str | None = None
    staff_no: str | None = None
    row_sha256: str | None = None
    snapshot_id: int | None = None
    person_id: int | None = None
    is_active: bool = True

    def with_hash(self) -> "DirectoryPersonRecord":
        payload = {
            "source_type": self.source_type,
            "source_url": self.source_url,
            "canonical_name": self.canonical_name,
            "designation": self.designation,
            "role_group": self.role_group,
            "department_name": self.department_name,
            "office_name": self.office_name,
            "email": normalize_email(self.email),
            "phone": normalize_phone(self.phone),
            "date_of_joining": self.date_of_joining,
            "staff_no": self.staff_no,
        }
        return DirectoryPersonRecord(**{**self.__dict__, "row_sha256": row_sha256(payload)})


@dataclass(frozen=True)
class BoardMemberRecord:
    """Structured board-member row derived from the official SEBI board page."""

    source_url: str
    canonical_name: str
    board_role: str
    category: str | None = None
    row_sha256: str | None = None
    snapshot_id: int | None = None
    board_member_id: int | None = None
    is_active: bool = True
    source_type: str = "board_members"

    def with_hash(self) -> "BoardMemberRecord":
        payload = {
            "source_type": self.source_type,
            "source_url": self.source_url,
            "canonical_name": self.canonical_name,
            "board_role": self.board_role,
            "category": self.category,
            "is_active": self.is_active,
        }
        return BoardMemberRecord(**{**self.__dict__, "row_sha256": row_sha256(payload)})


@dataclass(frozen=True)
class DirectoryOfficeRecord:
    """Structured office row derived from an official SEBI source page."""

    source_type: str
    source_url: str
    office_name: str
    office_type: str | None = None
    region: str | None = None
    address: str | None = None
    email: str | None = None
    phone: str | None = None
    fax: str | None = None
    city: str | None = None
    state: str | None = None
    row_sha256: str | None = None
    snapshot_id: int | None = None
    office_id: int | None = None
    is_active: bool = True

    def with_hash(self) -> "DirectoryOfficeRecord":
        payload = {
            "source_type": self.source_type,
            "source_url": self.source_url,
            "office_name": self.office_name,
            "office_type": self.office_type,
            "region": self.region,
            "address": self.address,
            "email": normalize_email(self.email),
            "phone": normalize_phone(self.phone),
            "fax": normalize_phone(self.fax),
            "city": self.city,
            "state": self.state,
        }
        return DirectoryOfficeRecord(**{**self.__dict__, "row_sha256": row_sha256(payload)})


@dataclass(frozen=True)
class OrgStructureRecord:
    """Structured leader-to-department mapping from the SEBI organisation chart."""

    source_type: str
    source_url: str
    leader_name: str | None = None
    leader_role: str | None = None
    department_name: str | None = None
    executive_director_name: str | None = None
    executive_director_email: str | None = None
    executive_director_phone: str | None = None
    row_sha256: str | None = None
    snapshot_id: int | None = None
    org_id: int | None = None
    is_active: bool = True

    def with_hash(self) -> "OrgStructureRecord":
        payload = {
            "source_type": self.source_type,
            "source_url": self.source_url,
            "leader_name": self.leader_name,
            "leader_role": self.leader_role,
            "department_name": self.department_name,
            "executive_director_name": self.executive_director_name,
            "executive_director_email": normalize_email(self.executive_director_email),
            "executive_director_phone": normalize_phone(self.executive_director_phone),
        }
        return OrgStructureRecord(**{**self.__dict__, "row_sha256": row_sha256(payload)})


@dataclass(frozen=True)
class DirectoryReferenceDataset:
    """Active structured SEBI reference rows used at answer time."""

    people: tuple[DirectoryPersonRecord, ...] = ()
    board_members: tuple[BoardMemberRecord, ...] = ()
    offices: tuple[DirectoryOfficeRecord, ...] = ()
    org_structure: tuple[OrgStructureRecord, ...] = ()

    def merged(self, other: "DirectoryReferenceDataset") -> "DirectoryReferenceDataset":
        return DirectoryReferenceDataset(
            people=self.people + other.people,
            board_members=self.board_members + other.board_members,
            offices=self.offices + other.offices,
            org_structure=self.org_structure + other.org_structure,
        )


@dataclass(frozen=True)
class DirectoryPageParseResult:
    """Structured rows parsed from a single source page."""

    people: tuple[DirectoryPersonRecord, ...] = ()
    board_members: tuple[BoardMemberRecord, ...] = ()
    offices: tuple[DirectoryOfficeRecord, ...] = ()
    org_structure: tuple[OrgStructureRecord, ...] = ()

    def as_dataset(self) -> DirectoryReferenceDataset:
        return DirectoryReferenceDataset(
            people=self.people,
            board_members=self.board_members,
            offices=self.offices,
            org_structure=self.org_structure,
        )


@dataclass
class DirectorySourceRunSummary:
    """One-source ingestion execution summary."""

    source_type: str
    source_url: str
    fetch_status: str = "pending"
    parse_status: str = "pending"
    snapshot_id: int | None = None
    people_rows: int = 0
    board_rows: int = 0
    office_rows: int = 0
    org_rows: int = 0
    error: str | None = None


@dataclass
class DirectoryIngestionSummary:
    """Aggregated ingestion summary for all configured structured sources."""

    source_summaries: list[DirectorySourceRunSummary] = field(default_factory=list)

    @property
    def snapshots_written(self) -> int:
        return sum(1 for item in self.source_summaries if item.snapshot_id is not None)

    @property
    def people_rows(self) -> int:
        return sum(item.people_rows for item in self.source_summaries)

    @property
    def board_rows(self) -> int:
        return sum(item.board_rows for item in self.source_summaries)

    @property
    def office_rows(self) -> int:
        return sum(item.office_rows for item in self.source_summaries)

    @property
    def org_rows(self) -> int:
        return sum(item.org_rows for item in self.source_summaries)

    def as_lines(self) -> list[str]:
        lines = [
            f"sources processed: {len(self.source_summaries)}",
            f"snapshots written: {self.snapshots_written}",
            f"people rows parsed: {self.people_rows}",
            f"board-member rows parsed: {self.board_rows}",
            f"office rows parsed: {self.office_rows}",
            f"org-structure rows parsed: {self.org_rows}",
        ]
        failed_sources = [item.source_type for item in self.source_summaries if item.error]
        if failed_sources:
            lines.append(f"failed sources: {', '.join(failed_sources)}")
        return lines
