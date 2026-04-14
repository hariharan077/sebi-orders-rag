"""Canonicalize SEBI people rows into one merged current-people layer."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..directory_data.models import (
    BoardMemberRecord,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
    OrgStructureRecord,
    normalize_email,
    normalize_person_name,
    normalize_phone,
    normalize_whitespace,
)
from ..normalization.aliases import normalize_department_alias
from .aggregates import department_aliases_for_value, designation_group_from_person
from .canonical_models import CanonicalOfficeRecord, CanonicalPersonRecord, DesignationGroup
from .canonicalize_offices import match_canonical_offices, normalize_lookup_key

_SOURCE_PRIORITY: dict[str, int] = {
    "directory": 140,
    "orgchart": 120,
    "regional_offices": 110,
    "board_members": 105,
    "contact_us": 60,
}
_ROLE_PRIORITY: dict[str, int] = {
    "chairperson": 120,
    "wtm": 110,
    "executive_director": 100,
    "regional_director": 95,
    "department_head": 80,
    "chief_vigilance_officer": 75,
    "office_contact": 70,
    "staff": 60,
}
_HONORIFIC_RE = re.compile(r"^\s*(?:shri|smt|mrs|mr|ms|dr)\.?\s+", re.IGNORECASE)
_LEADERSHIP_GROUPS = {"chairperson", "whole_time_member", "board_member", "executive_director"}


@dataclass(frozen=True)
class _PersonSourceRow:
    source_type: str
    source_url: str
    source_row_key: str
    canonical_name: str
    designation: str | None = None
    role_group: str | None = None
    department_name: str | None = None
    office_name: str | None = None
    email: str | None = None
    phone: str | None = None
    date_of_joining: str | None = None
    staff_no: str | None = None
    board_role: str | None = None
    board_category: str | None = None


@dataclass(frozen=True)
class CanonicalPersonMergeAudit:
    """One canonical person plus the raw rows and reasons that formed it."""

    person: CanonicalPersonRecord
    source_rows: tuple[_PersonSourceRow, ...]
    merge_reasons: tuple[str, ...]


@dataclass
class _MergeCluster:
    rows: list[_PersonSourceRow] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def canonicalize_people(
    raw_dataset: DirectoryReferenceDataset,
    *,
    offices: tuple[CanonicalOfficeRecord, ...] = (),
) -> tuple[CanonicalPersonRecord, ...]:
    """Merge raw structured rows into canonical current people."""

    return tuple(
        audit_row.person
        for audit_row in canonicalize_people_with_audit(raw_dataset, offices=offices)
    )


def canonicalize_people_with_audit(
    raw_dataset: DirectoryReferenceDataset,
    *,
    offices: tuple[CanonicalOfficeRecord, ...] = (),
) -> tuple[CanonicalPersonMergeAudit, ...]:
    """Return canonical people plus the raw rows and merge reasons used."""

    grouped: dict[str, list[_PersonSourceRow]] = {}
    for source_row in _source_rows(raw_dataset):
        name_key = normalized_name_key(source_row.canonical_name)
        if not name_key:
            continue
        grouped.setdefault(name_key, []).append(source_row)

    merged: list[CanonicalPersonMergeAudit] = []
    for name_key, rows in sorted(grouped.items()):
        clusters = _build_clusters(rows, offices=offices)
        for index, cluster in enumerate(clusters, start=1):
            ordered = sorted(cluster.rows, key=_person_sort_key)
            merged.append(
                CanonicalPersonMergeAudit(
                    person=_build_person_record(
                        name_key=name_key,
                        cluster_index=index,
                        rows=ordered,
                        offices=offices,
                        merge_reasons=tuple(dict.fromkeys(cluster.reasons)),
                    ),
                    source_rows=tuple(ordered),
                    merge_reasons=tuple(dict.fromkeys(cluster.reasons)),
                )
            )
    return tuple(
        sorted(
            merged,
            key=lambda item: (
                item.person.canonical_name.lower(),
                item.person.designation_group,
                item.person.office_name or "",
                item.person.department_name or "",
            ),
        )
    )


def normalized_name_key(value: str | None) -> str:
    """Return the deterministic person key used for canonical grouping."""

    return normalize_lookup_key(_clean_person_name(value))


def _build_clusters(
    rows: list[_PersonSourceRow],
    *,
    offices: tuple[CanonicalOfficeRecord, ...],
) -> list[_MergeCluster]:
    ordered_rows = sorted(rows, key=_person_sort_key)
    clusters: list[_MergeCluster] = []
    for row in ordered_rows:
        chosen_cluster: _MergeCluster | None = None
        chosen_reasons: tuple[str, ...] = ()
        chosen_score = -1
        for cluster in clusters:
            reasons = _merge_reasons(row, cluster.rows, offices=offices)
            if not reasons or not _can_merge(row, cluster.rows, reasons=reasons, offices=offices):
                continue
            score = _merge_reason_score(reasons)
            if score > chosen_score:
                chosen_cluster = cluster
                chosen_reasons = reasons
                chosen_score = score
        if chosen_cluster is None:
            clusters.append(
                _MergeCluster(
                    rows=[row],
                    reasons=[f"seed:{row.source_type}:{row.source_row_key}"],
                )
            )
            continue
        chosen_cluster.rows.append(row)
        chosen_cluster.reasons.extend(chosen_reasons)
    return clusters


def _can_merge(
    row: _PersonSourceRow,
    cluster_rows: list[_PersonSourceRow],
    *,
    reasons: tuple[str, ...],
    offices: tuple[CanonicalOfficeRecord, ...],
) -> bool:
    strong_identity = any(
        reason.startswith(prefix)
        for prefix in ("same_staff_no", "same_email", "same_phone")
        for reason in reasons
    )
    if strong_identity:
        return True

    row_group = _row_designation_group(row)
    cluster_groups = {
        _row_designation_group(cluster_row)
        for cluster_row in cluster_rows
        if _row_designation_group(cluster_row)
    }
    has_group_conflict = bool(
        row_group
        and cluster_groups
        and not any(_designation_groups_compatible(row_group, group) for group in cluster_groups)
    )
    row_department = _department_key(row.department_name)
    cluster_departments = {
        _department_key(cluster_row.department_name)
        for cluster_row in cluster_rows
        if _department_key(cluster_row.department_name)
    }
    has_department_conflict = bool(
        row_department and cluster_departments and row_department not in cluster_departments
    )
    row_office = _office_key(row, offices=offices)
    cluster_offices = {
        _office_key(cluster_row, offices=offices)
        for cluster_row in cluster_rows
        if _office_key(cluster_row, offices=offices)
    }
    has_office_conflict = bool(row_office and cluster_offices and row_office not in cluster_offices)

    if has_group_conflict:
        return False
    if has_department_conflict and has_office_conflict:
        return False
    if any(
        reason.startswith(prefix)
        for prefix in (
            "same_board_role",
            "same_designation_group_and_department",
            "same_designation_group_and_office",
            "same_leadership_designation_group",
            "same_designation_group_with_missing_context",
        )
        for reason in reasons
    ):
        return True
    return False


def _merge_reasons(
    row: _PersonSourceRow,
    cluster_rows: list[_PersonSourceRow],
    *,
    offices: tuple[CanonicalOfficeRecord, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    row_staff_no = normalize_whitespace(row.staff_no)
    cluster_staff_nos = {
        normalize_whitespace(cluster_row.staff_no)
        for cluster_row in cluster_rows
        if normalize_whitespace(cluster_row.staff_no)
    }
    if row_staff_no and row_staff_no in cluster_staff_nos:
        reasons.append(f"same_staff_no:{row_staff_no}")

    row_email = normalize_email(row.email)
    cluster_emails = {
        normalize_email(cluster_row.email)
        for cluster_row in cluster_rows
        if normalize_email(cluster_row.email)
    }
    if row_email and row_email in cluster_emails:
        reasons.append(f"same_email:{row_email}")

    row_phone = normalize_phone(row.phone)
    cluster_phones = {
        normalize_phone(cluster_row.phone)
        for cluster_row in cluster_rows
        if normalize_phone(cluster_row.phone)
    }
    if row_phone and row_phone in cluster_phones:
        reasons.append(f"same_phone:{row_phone}")

    row_group = _row_designation_group(row)
    cluster_groups = {
        _row_designation_group(cluster_row)
        for cluster_row in cluster_rows
        if _row_designation_group(cluster_row)
    }
    row_department = _department_key(row.department_name)
    cluster_departments = {
        _department_key(cluster_row.department_name)
        for cluster_row in cluster_rows
        if _department_key(cluster_row.department_name)
    }
    row_office = _office_key(row, offices=offices)
    cluster_offices = {
        _office_key(cluster_row, offices=offices)
        for cluster_row in cluster_rows
        if _office_key(cluster_row, offices=offices)
    }

    if row_group and row_department and row_group in cluster_groups and row_department in cluster_departments:
        reasons.append(f"same_designation_group_and_department:{row_group}:{row_department}")
    if row_group and row_office and row_group in cluster_groups and row_office in cluster_offices:
        reasons.append(f"same_designation_group_and_office:{row_group}:{row_office}")
    if row_group and row_group in _LEADERSHIP_GROUPS and row_group in cluster_groups:
        reasons.append(f"same_leadership_designation_group:{row_group}")
    if row_group and row_group in cluster_groups and not row_department and not row_office:
        reasons.append(f"same_designation_group_with_missing_context:{row_group}")

    row_board_role = normalize_lookup_key(row.board_role)
    cluster_board_roles = {
        normalize_lookup_key(cluster_row.board_role)
        for cluster_row in cluster_rows
        if normalize_lookup_key(cluster_row.board_role)
    }
    if row_board_role and row_board_role in cluster_board_roles:
        reasons.append(f"same_board_role:{row_board_role}")

    return tuple(dict.fromkeys(reason for reason in reasons if reason))


def _designation_groups_compatible(left: DesignationGroup, right: DesignationGroup) -> bool:
    if left == right:
        return True
    if left in _LEADERSHIP_GROUPS and right in _LEADERSHIP_GROUPS:
        return left == right
    return False


def _build_person_record(
    *,
    name_key: str,
    cluster_index: int,
    rows: list[_PersonSourceRow],
    offices: tuple[CanonicalOfficeRecord, ...],
    merge_reasons: tuple[str, ...],
) -> CanonicalPersonRecord:
    primary = rows[0]
    office = _resolve_office(rows, offices)
    canonical_name = _choose_display_name(rows)
    designation = _choose_designation(rows)
    role_group = _choose_role_group(rows)
    board_role = _choose_board_role(rows)
    board_category = _choose_board_category(rows)
    designation_group = designation_group_from_person(
        designation=designation,
        role_group=role_group,
        board_category=board_category,
        board_role=board_role,
    )
    department_name = _choose_text_field(rows, "department_name")
    office_name = office.office_name if office is not None else _choose_text_field(rows, "office_name")
    office_aliases = office.office_aliases if office is not None else ()
    office_city = office.city if office is not None else None
    office_region = office.region if office is not None else None
    aliases = tuple(
        dict.fromkeys(
            alias
            for alias in (
                canonical_name,
                *(_name_aliases(canonical_name)),
            )
            if alias
        )
    )
    signature = "|".join(
        value
        for value in (
            name_key,
            designation_group,
            _department_key(department_name) or "",
            _office_key(_PersonSourceRow(
                source_type="",
                source_url="",
                source_row_key="",
                canonical_name=canonical_name,
                office_name=office_name,
            ), offices=offices)
            or "",
            normalize_whitespace(_choose_text_field(rows, "staff_no")) or "",
            normalize_email(_choose_text_field(rows, "email")) or "",
            str(cluster_index),
        )
        if value
    )
    return CanonicalPersonRecord(
        canonical_person_id=_canonical_id("person", signature),
        canonical_name=canonical_name,
        normalized_name_key=name_key,
        aliases=aliases,
        designation=designation,
        designation_group=designation_group,
        department_name=department_name,
        department_aliases=department_aliases_for_value(department_name),
        office_name=office_name,
        office_city=office_city,
        office_region=office_region,
        office_aliases=office_aliases,
        email=normalize_email(_choose_text_field(rows, "email")),
        phone=normalize_phone(_choose_text_field(rows, "phone")),
        date_of_joining=_choose_text_field(rows, "date_of_joining"),
        staff_no=normalize_whitespace(_choose_text_field(rows, "staff_no")),
        role_group=role_group,
        board_role=board_role,
        board_category=board_category,
        source_priority=_source_priority(primary.source_type),
        active_status=True,
        source_types=tuple(dict.fromkeys(row.source_type for row in rows)),
        source_urls=tuple(dict.fromkeys(row.source_url for row in rows)),
        source_row_count=len(rows),
        merge_notes=merge_reasons,
        merged_row_keys=tuple(dict.fromkeys(row.source_row_key for row in rows)),
    )


def _source_rows(raw_dataset: DirectoryReferenceDataset) -> tuple[_PersonSourceRow, ...]:
    rows: list[_PersonSourceRow] = []
    for person in raw_dataset.people:
        rows.append(
            _PersonSourceRow(
                source_type=person.source_type,
                source_url=person.source_url,
                source_row_key=_row_key_from_directory_person(person),
                canonical_name=person.canonical_name,
                designation=person.designation,
                role_group=person.role_group,
                department_name=person.department_name,
                office_name=person.office_name,
                email=person.email,
                phone=person.phone,
                date_of_joining=person.date_of_joining,
                staff_no=person.staff_no,
            )
        )
    for board in raw_dataset.board_members:
        rows.append(
            _PersonSourceRow(
                source_type=board.source_type,
                source_url=board.source_url,
                source_row_key=_row_key_from_board_member(board),
                canonical_name=board.canonical_name,
                designation=_designation_from_board_role(board.board_role),
                role_group=_role_group_from_board_category(board.category),
                board_role=board.board_role,
                board_category=board.category,
            )
        )
    for record in raw_dataset.org_structure:
        rows.extend(_org_structure_rows(record))
    return tuple(rows)


def _org_structure_rows(record: OrgStructureRecord) -> tuple[_PersonSourceRow, ...]:
    rows: list[_PersonSourceRow] = []
    if normalize_whitespace(record.leader_name):
        rows.append(
            _PersonSourceRow(
                source_type=record.source_type,
                source_url=record.source_url,
                source_row_key=_org_row_key(record, suffix="leader"),
                canonical_name=normalize_person_name(record.leader_name) or record.leader_name or "",
                designation=record.leader_role,
                role_group=_role_group_from_title(record.leader_role),
                department_name=record.department_name,
            )
        )
    if normalize_whitespace(record.executive_director_name):
        rows.append(
            _PersonSourceRow(
                source_type=record.source_type,
                source_url=record.source_url,
                source_row_key=_org_row_key(record, suffix="executive_director"),
                canonical_name=normalize_person_name(record.executive_director_name) or record.executive_director_name or "",
                designation="Executive Director",
                role_group="executive_director",
                department_name=record.department_name,
                email=record.executive_director_email,
                phone=record.executive_director_phone,
            )
        )
    return tuple(row for row in rows if normalize_whitespace(row.canonical_name))


def _choose_display_name(rows: list[_PersonSourceRow]) -> str:
    ordered = sorted(rows, key=_person_sort_key)
    for row in ordered:
        cleaned = _clean_person_name(row.canonical_name)
        if cleaned:
            return cleaned
    return normalize_person_name(ordered[0].canonical_name) or ordered[0].canonical_name


def _choose_designation(rows: list[_PersonSourceRow]) -> str | None:
    best_value: str | None = None
    best_rank = (-1, -1)
    for row in rows:
        value = normalize_whitespace(row.designation)
        if not value:
            continue
        rank = (_source_priority(row.source_type), _role_priority(row.role_group))
        if rank > best_rank:
            best_value = value
            best_rank = rank
    return best_value


def _choose_role_group(rows: list[_PersonSourceRow]) -> str | None:
    ranked = [
        row.role_group
        for row in sorted(
            rows,
            key=lambda item: (-_role_priority(item.role_group), -_source_priority(item.source_type)),
        )
        if row.role_group
    ]
    return ranked[0] if ranked else None


def _choose_board_role(rows: list[_PersonSourceRow]) -> str | None:
    return _choose_text_field(rows, "board_role")


def _choose_board_category(rows: list[_PersonSourceRow]) -> str | None:
    return _choose_text_field(rows, "board_category")


def _choose_text_field(rows: list[_PersonSourceRow], field_name: str) -> str | None:
    best_value: str | None = None
    best_rank = (-1, -1)
    for row in rows:
        value = normalize_whitespace(getattr(row, field_name))
        if not value:
            continue
        rank = (_source_priority(row.source_type), len(value))
        if rank > best_rank:
            best_value = value
            best_rank = rank
    return best_value


def _resolve_office(
    rows: list[_PersonSourceRow],
    offices: tuple[CanonicalOfficeRecord, ...],
) -> CanonicalOfficeRecord | None:
    hints = [
        value
        for value in (_choose_text_field(rows, "office_name"), _choose_text_field(rows, "department_name"))
        if value
    ]
    for hint in hints:
        matches = match_canonical_offices(offices, hint)
        if matches:
            return matches[0]
    return None


def _name_aliases(value: str) -> tuple[str, ...]:
    normalized = _clean_person_name(value)
    if not normalized:
        return ()
    tokens = normalized.split()
    aliases = [normalized]
    if tokens:
        aliases.append(tokens[0])
        aliases.append(tokens[-1])
    if len(tokens) >= 2:
        aliases.append(" ".join(tokens[:2]))
        aliases.append(" ".join((tokens[0], tokens[-1])))
    return tuple(
        dict.fromkeys(
            alias
            for alias in (normalize_whitespace(item) for item in aliases)
            if alias
        )
    )


def _designation_from_board_role(value: str | None) -> str | None:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    if "," in cleaned:
        return cleaned.split(",", 1)[0]
    if "(" in cleaned:
        return cleaned.split("(", 1)[0].strip()
    return cleaned


def _role_group_from_board_category(value: str | None) -> str | None:
    cleaned = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned == "chairperson":
        return "chairperson"
    if cleaned == "whole_time_member":
        return "wtm"
    if cleaned in {"part_time_member", "government_nominee", "rbi_nominee"}:
        return "board_member"
    return None


def _role_group_from_title(value: str | None) -> str | None:
    cleaned = (value or "").strip().lower()
    if "chair" in cleaned:
        return "chairperson"
    if "whole time member" in cleaned or "whole-time member" in cleaned:
        return "wtm"
    if "executive director" in cleaned:
        return "executive_director"
    if "regional director" in cleaned:
        return "regional_director"
    if "head" in cleaned:
        return "department_head"
    return None


def _source_priority(source_type: str | None) -> int:
    return _SOURCE_PRIORITY.get(source_type or "", 0)


def _role_priority(role_group: str | None) -> int:
    return _ROLE_PRIORITY.get(role_group or "", 0)


def _person_sort_key(row: _PersonSourceRow) -> tuple[int, int, int, str]:
    richness = sum(
        1
        for value in (
            row.designation,
            row.department_name,
            row.office_name,
            row.email,
            row.phone,
            row.date_of_joining,
            row.staff_no,
            row.board_role,
            row.board_category,
        )
        if normalize_whitespace(value)
    )
    return (
        -_source_priority(row.source_type),
        -_role_priority(row.role_group),
        -richness,
        row.canonical_name.lower(),
    )


def _clean_person_name(value: str | None) -> str | None:
    cleaned = normalize_person_name(normalize_whitespace(value))
    if not cleaned:
        return None
    cleaned = _HONORIFIC_RE.sub("", cleaned).strip(" ,.")
    return cleaned or None


def _row_designation_group(row: _PersonSourceRow) -> DesignationGroup:
    return designation_group_from_person(
        designation=row.designation,
        role_group=row.role_group,
        board_category=row.board_category,
        board_role=row.board_role,
    )


def _department_key(value: str | None) -> str | None:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    alias = normalize_department_alias(cleaned)
    return alias or normalize_lookup_key(cleaned)


def _office_key(
    row: _PersonSourceRow,
    *,
    offices: tuple[CanonicalOfficeRecord, ...],
) -> str | None:
    office_name = normalize_whitespace(row.office_name)
    if office_name:
        matches = match_canonical_offices(offices, office_name)
        if matches:
            return matches[0].normalized_office_key
        return normalize_lookup_key(office_name)
    department_name = normalize_whitespace(row.department_name)
    if not department_name:
        return None
    matches = match_canonical_offices(offices, department_name)
    if matches:
        return matches[0].normalized_office_key
    return None


def _row_key_from_directory_person(person: DirectoryPersonRecord) -> str:
    if person.row_sha256:
        return person.row_sha256
    return _hash_key(
        "person",
        person.source_type,
        person.canonical_name,
        person.designation,
        person.department_name,
        person.office_name,
        person.staff_no,
    )


def _row_key_from_board_member(board: BoardMemberRecord) -> str:
    if board.row_sha256:
        return board.row_sha256
    return _hash_key(
        "board",
        board.source_type,
        board.canonical_name,
        board.board_role,
        board.category,
    )


def _org_row_key(record: OrgStructureRecord, *, suffix: str) -> str:
    if record.row_sha256:
        return f"{record.row_sha256}:{suffix}"
    return _hash_key(
        "org",
        record.source_type,
        record.leader_name,
        record.leader_role,
        record.department_name,
        record.executive_director_name,
        suffix,
    )


def _hash_key(prefix: str, *values: str | None) -> str:
    payload = "|".join(str(value or "") for value in values)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _merge_reason_score(reasons: tuple[str, ...]) -> int:
    score = 0
    for reason in reasons:
        if reason.startswith(("same_staff_no", "same_email", "same_phone")):
            score += 100
        elif reason.startswith(("same_board_role", "same_designation_group_and_department")):
            score += 60
        elif reason.startswith(("same_designation_group_and_office", "same_leadership_designation_group")):
            score += 45
        elif reason.startswith("same_designation_group_with_missing_context"):
            score += 25
    return score


def _canonical_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"
