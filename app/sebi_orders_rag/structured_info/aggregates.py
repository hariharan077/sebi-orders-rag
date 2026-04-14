"""Aggregate helpers for canonical structured SEBI current-info data."""

from __future__ import annotations

from collections import Counter

from ..normalization import normalize_designation_alias
from ..directory_data.models import DirectoryReferenceDataset, OrgStructureRecord, normalize_whitespace
from ..normalization.aliases import normalize_department_alias
from .canonical_models import (
    CanonicalOfficeRecord,
    CanonicalPersonRecord,
    DepartmentCountRecord,
    DesignationCountRecord,
    DesignationGroup,
    OfficeCountRecord,
    RoleCountRecord,
    StructuredInfoSnapshot,
)

_DESIGNATION_GROUP_LABELS: dict[str, str] = {
    "chairperson": "Chairperson",
    "whole_time_member": "Whole-Time Member",
    "board_member": "Board Member",
    "executive_director": "Executive Director",
    "chief_general_manager": "Chief General Manager",
    "general_manager": "General Manager",
    "deputy_general_manager": "Deputy General Manager",
    "assistant_general_manager": "Assistant General Manager",
    "manager": "Manager",
    "assistant_manager": "Assistant Manager",
    "regional_director": "Regional Director",
    "office_contact": "Office Contact",
    "staff": "Staff",
    "other": "Other",
}

_QUERY_GROUP_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("chairperson", ("chairperson", "chairman")),
    ("whole_time_member", ("whole time member", "whole-time member", "wtm")),
    ("board_member", ("board member",)),
    ("executive_director", ("executive director", "ed")),
    ("chief_general_manager", ("chief general manager", "cgm")),
    ("general_manager", ("general manager", "gm")),
    ("deputy_general_manager", ("deputy general manager", "dgm")),
    ("assistant_general_manager", ("assistant general manager", "agm")),
    ("manager", ("manager",)),
    ("assistant_manager", ("assistant manager", "am")),
    ("regional_director", ("regional director", "rd")),
    ("office_contact", ("office contact",)),
    ("staff", ("staff",)),
)

_ROLE_COUNT_SPECS: tuple[tuple[str, str, callable], ...] = (
    ("whole_time_member", "Whole-Time Members", lambda person: person.designation_group == "whole_time_member"),
    ("executive_director", "Executive Directors", lambda person: person.designation_group == "executive_director"),
    ("board_member", "Board Members", lambda person: person.designation_group in {"chairperson", "whole_time_member", "board_member"}),
)


def designation_group_label(group: str) -> str:
    """Return the user-facing label for a canonical designation group."""

    return _DESIGNATION_GROUP_LABELS.get(group, group.replace("_", " ").title())


def designation_group_from_query(value: str | None) -> str | None:
    """Map a designation/count query to one canonical designation group."""

    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    alias = normalize_designation_alias(cleaned)
    normalized = (alias or cleaned).strip().lower().replace("-", " ")
    if normalized.endswith("s") and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    for group, variants in _QUERY_GROUP_ALIASES:
        if normalized in variants:
            return group
        if any(normalized.startswith(f"{variant} ") for variant in variants):
            return group
    return None


def designation_group_from_person(
    *,
    designation: str | None,
    role_group: str | None,
    board_category: str | None,
    board_role: str | None,
) -> DesignationGroup:
    """Map person data to one stable canonical designation group."""

    normalized_designation = _normalize_designation_text(designation)
    normalized_role_group = (role_group or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized_board_category = (board_category or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized_board_role = _normalize_designation_text(board_role)

    if normalized_board_category == "chairperson" or normalized_role_group == "chairperson":
        return "chairperson"
    if normalized_board_category == "whole_time_member" or normalized_role_group == "wtm":
        return "whole_time_member"
    if normalized_board_category in {"part_time_member", "government_nominee", "rbi_nominee"}:
        return "board_member"
    if normalized_role_group == "regional_director" or "regional director" in normalized_designation:
        return "regional_director"
    if normalized_role_group == "executive_director" or "executive director" in normalized_designation:
        return "executive_director"
    if "chief general manager" in normalized_designation:
        return "chief_general_manager"
    if "deputy general manager" in normalized_designation:
        return "deputy_general_manager"
    if "assistant general manager" in normalized_designation:
        return "assistant_general_manager"
    if normalized_designation == "assistant manager" or normalized_designation.startswith("assistant manager "):
        return "assistant_manager"
    if normalized_designation == "manager" or normalized_designation.startswith("manager "):
        return "manager"
    if normalized_designation == "general manager" or normalized_designation.startswith("general manager "):
        return "general_manager"
    if normalized_role_group == "office_contact":
        return "office_contact"
    if normalized_role_group in {"staff", "department_head", "chief_vigilance_officer"} or normalized_designation:
        return "staff"
    if normalized_board_role:
        return "board_member"
    return "other"


def build_aggregates(
    *,
    people: tuple[CanonicalPersonRecord, ...],
    offices: tuple[CanonicalOfficeRecord, ...],
) -> tuple[
    tuple[DesignationCountRecord, ...],
    tuple[OfficeCountRecord, ...],
    tuple[DepartmentCountRecord, ...],
    tuple[RoleCountRecord, ...],
]:
    """Build deterministic aggregate summaries from canonical people and offices."""

    designation_counter: Counter[str] = Counter(person.designation_group for person in people)
    designation_counts = tuple(
        DesignationCountRecord(
            designation_group=group,  # type: ignore[arg-type]
            designation_label=designation_group_label(group),
            people_count=designation_counter[group],
        )
        for group in sorted(designation_counter)
    )

    people_by_office: Counter[str] = Counter(
        person.office_name or ""
        for person in people
        if person.office_name
    )
    office_counts = tuple(
        OfficeCountRecord(
            canonical_office_id=office.canonical_office_id,
            office_name=office.office_name,
            city=office.city,
            region=office.region,
            people_count=people_by_office.get(office.office_name, 0),
            office_count=1,
        )
        for office in offices
    )

    department_counter: Counter[str] = Counter(
        person.department_name
        for person in people
        if person.department_name
    )
    department_counts = tuple(
        DepartmentCountRecord(
            department_name=department_name,
            people_count=department_counter[department_name],
        )
        for department_name in sorted(department_counter)
    )

    role_counts = tuple(
        RoleCountRecord(
            role_key=role_key,
            role_label=role_label,
            people_count=sum(1 for person in people if predicate(person)),
        )
        for role_key, role_label, predicate in _ROLE_COUNT_SPECS
    )
    return designation_counts, office_counts, department_counts, role_counts


def build_snapshot(
    *,
    raw_dataset: DirectoryReferenceDataset,
    people: tuple[CanonicalPersonRecord, ...],
    offices: tuple[CanonicalOfficeRecord, ...],
    org_structure: tuple[OrgStructureRecord, ...],
) -> StructuredInfoSnapshot:
    """Build the complete canonical structured-info snapshot."""

    designation_counts, office_counts, department_counts, role_counts = build_aggregates(
        people=people,
        offices=offices,
    )
    return StructuredInfoSnapshot(
        people=people,
        offices=offices,
        org_structure=org_structure,
        raw_people=raw_dataset.people,
        raw_board_members=raw_dataset.board_members,
        raw_offices=raw_dataset.offices,
        designation_counts=designation_counts,
        office_counts=office_counts,
        department_counts=department_counts,
        role_counts=role_counts,
        raw_people_count=len(raw_dataset.people),
        raw_board_member_count=len(raw_dataset.board_members),
        raw_office_count=len(raw_dataset.offices),
        raw_org_count=len(raw_dataset.org_structure),
    )


def department_aliases_for_value(value: str | None) -> tuple[str, ...]:
    """Return deterministic department aliases used by matching and audit."""

    cleaned = normalize_whitespace(value)
    if not cleaned:
        return ()
    alias = normalize_department_alias(cleaned)
    values = [cleaned]
    if alias and alias not in values:
        values.append(alias)
    normalized = tuple(dict.fromkeys(item for item in values if item))
    return normalized


def _normalize_designation_text(value: str | None) -> str:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return ""
    alias = normalize_designation_alias(cleaned)
    return (alias or cleaned).strip().lower().replace("-", " ")
