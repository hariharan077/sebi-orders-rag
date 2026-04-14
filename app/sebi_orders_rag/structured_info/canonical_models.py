"""Canonical models for structured SEBI current-information lookups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..directory_data.models import (
    BoardMemberRecord,
    DirectoryOfficeRecord,
    DirectoryPersonRecord,
    OrgStructureRecord,
)

DesignationGroup = Literal[
    "chairperson",
    "whole_time_member",
    "board_member",
    "executive_director",
    "chief_general_manager",
    "general_manager",
    "deputy_general_manager",
    "assistant_general_manager",
    "manager",
    "assistant_manager",
    "regional_director",
    "office_contact",
    "staff",
    "other",
]


@dataclass(frozen=True)
class CanonicalPersonRecord:
    """One merged current SEBI person record."""

    canonical_person_id: str
    canonical_name: str
    normalized_name_key: str
    aliases: tuple[str, ...] = ()
    designation: str | None = None
    designation_group: DesignationGroup = "other"
    department_name: str | None = None
    department_aliases: tuple[str, ...] = ()
    office_name: str | None = None
    office_city: str | None = None
    office_region: str | None = None
    office_aliases: tuple[str, ...] = ()
    email: str | None = None
    phone: str | None = None
    date_of_joining: str | None = None
    staff_no: str | None = None
    role_group: str | None = None
    board_role: str | None = None
    board_category: str | None = None
    source_priority: int = 0
    active_status: bool = True
    source_types: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    source_row_count: int = 0
    merge_notes: tuple[str, ...] = ()
    merged_row_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalOfficeRecord:
    """One merged current SEBI office record."""

    canonical_office_id: str
    office_name: str
    normalized_office_key: str
    office_aliases: tuple[str, ...] = ()
    office_type: str | None = None
    city: str | None = None
    state: str | None = None
    region: str | None = None
    address: str | None = None
    email: str | None = None
    phone: str | None = None
    fax: str | None = None
    source_priority: int = 0
    active_status: bool = True
    source_types: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    source_row_count: int = 0


@dataclass(frozen=True)
class DesignationCountRecord:
    """Aggregate count for one canonical designation group."""

    designation_group: DesignationGroup
    designation_label: str
    people_count: int


@dataclass(frozen=True)
class OfficeCountRecord:
    """Aggregate count for one canonical office."""

    canonical_office_id: str
    office_name: str
    city: str | None
    region: str | None
    people_count: int
    office_count: int = 1


@dataclass(frozen=True)
class DepartmentCountRecord:
    """Aggregate count for one department."""

    department_name: str
    people_count: int


@dataclass(frozen=True)
class RoleCountRecord:
    """Aggregate count for one role bucket."""

    role_key: str
    role_label: str
    people_count: int


@dataclass(frozen=True)
class StructuredInfoSnapshot:
    """Canonical current-information snapshot used by the answer layer."""

    people: tuple[CanonicalPersonRecord, ...] = ()
    offices: tuple[CanonicalOfficeRecord, ...] = ()
    org_structure: tuple[OrgStructureRecord, ...] = ()
    raw_people: tuple[DirectoryPersonRecord, ...] = ()
    raw_board_members: tuple[BoardMemberRecord, ...] = ()
    raw_offices: tuple[DirectoryOfficeRecord, ...] = ()
    designation_counts: tuple[DesignationCountRecord, ...] = ()
    office_counts: tuple[OfficeCountRecord, ...] = ()
    department_counts: tuple[DepartmentCountRecord, ...] = ()
    role_counts: tuple[RoleCountRecord, ...] = ()
    raw_people_count: int = 0
    raw_board_member_count: int = 0
    raw_office_count: int = 0
    raw_org_count: int = 0

    def designation_count_by_group(self) -> dict[str, int]:
        return {
            record.designation_group: record.people_count
            for record in self.designation_counts
        }

    def role_count_by_key(self) -> dict[str, int]:
        return {
            record.role_key: record.people_count
            for record in self.role_counts
        }

    def canonical_people_by_staff_no(self) -> dict[str, CanonicalPersonRecord]:
        return {
            person.staff_no: person
            for person in self.people
            if person.staff_no
        }
