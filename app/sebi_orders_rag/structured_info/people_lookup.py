"""Helpers for canonical SEBI people and staff-id lookup."""

from __future__ import annotations

from dataclasses import dataclass

from ..current_info.query_normalization import StructuredCurrentInfoQuery
from ..directory_data.models import DirectoryPersonRecord, normalize_whitespace
from .aggregates import designation_group_from_query
from .ambiguity import PersonMatchResult, match_people
from .canonical_models import CanonicalPersonRecord, StructuredInfoSnapshot
from .canonicalize_offices import normalize_lookup_key


@dataclass(frozen=True)
class StaffLookupMatch:
    """Structured staff-id match built from raw rows and canonical rows."""

    staff_no: str
    raw_rows: tuple[DirectoryPersonRecord, ...] = ()
    canonical_people: tuple[CanonicalPersonRecord, ...] = ()


def filter_people(
    people: tuple[CanonicalPersonRecord, ...],
    *,
    intent: StructuredCurrentInfoQuery,
) -> tuple[CanonicalPersonRecord, ...]:
    """Filter canonical people by department or designation hints."""

    filtered = list(people)
    if intent.department_hint:
        department_hint = normalize_lookup_key(intent.department_hint)
        filtered = [
            person
            for person in filtered
            if department_hint
            and (
                any(normalize_lookup_key(alias) == department_hint for alias in person.department_aliases)
                or normalize_lookup_key(person.department_name) == department_hint
                or normalize_lookup_key(person.office_name) == department_hint
            )
        ]
    if intent.designation_hint:
        designation_group = designation_group_from_query(intent.designation_hint)
        if designation_group:
            filtered = [
                person
                for person in filtered
                if person.designation_group == designation_group
            ]
        else:
            designation_hint = normalize_lookup_key(intent.designation_hint)
            filtered = [
                person
                for person in filtered
                if designation_hint and normalize_lookup_key(person.designation) == designation_hint
            ]
    return tuple(filtered)


def resolve_person_match(
    snapshot: StructuredInfoSnapshot,
    *,
    intent: StructuredCurrentInfoQuery,
) -> PersonMatchResult:
    """Resolve one current-person query against the canonical people layer."""

    search_people = (
        filter_people(snapshot.people, intent=intent)
        if (intent.department_hint or intent.designation_hint)
        else snapshot.people
    )
    if not intent.person_name:
        return PersonMatchResult(status="no_match", matches=tuple(search_people))
    return match_people(search_people, intent.person_name)


def lookup_staff_no(
    snapshot: StructuredInfoSnapshot,
    *,
    staff_no: str,
) -> StaffLookupMatch | None:
    """Resolve a staff-id directly against raw structured rows first."""

    normalized_staff_no = _normalize_staff_no(staff_no)
    if not normalized_staff_no:
        return None

    raw_rows = tuple(
        row
        for row in snapshot.raw_people
        if _normalize_staff_no(row.staff_no) == normalized_staff_no
    )
    if not raw_rows:
        canonical_people = tuple(
            person
            for person in snapshot.people
            if _normalize_staff_no(person.staff_no) == normalized_staff_no
        )
        if not canonical_people:
            return None
        return StaffLookupMatch(
            staff_no=normalize_whitespace(staff_no) or staff_no,
            raw_rows=(),
            canonical_people=canonical_people,
        )

    raw_row_keys = {
        row.row_sha256
        for row in raw_rows
        if row.row_sha256
    }
    canonical_people = tuple(
        person
        for person in snapshot.people
        if _normalize_staff_no(person.staff_no) == normalized_staff_no
        or raw_row_keys.intersection(person.merged_row_keys)
    )
    return StaffLookupMatch(
        staff_no=normalize_whitespace(raw_rows[0].staff_no) or normalize_whitespace(staff_no) or staff_no,
        raw_rows=raw_rows,
        canonical_people=canonical_people,
    )


def render_person_answer(person: CanonicalPersonRecord) -> str:
    """Render one canonical person answer line."""

    primary_title = _primary_person_title(person)
    if person.designation_group == "chairperson":
        return f"{person.canonical_name} is SEBI's {primary_title}."
    if person.designation_group == "whole_time_member":
        return f"{person.canonical_name} is a {primary_title} of SEBI."
    if person.designation_group == "board_member":
        return f"{person.canonical_name} is a {primary_title} of SEBI."

    answer = f"{person.canonical_name} is listed as {primary_title}"
    location = _person_location_label(person)
    if location:
        answer += f" in {location}"
    office_suffix = _person_office_suffix(person)
    if office_suffix:
        answer += office_suffix
    return answer + "."


def render_person_with_context(person: CanonicalPersonRecord) -> str:
    """Render one canonical person label for clarifications and lists."""

    parts = [person.canonical_name]
    parts.append(_primary_person_title(person))
    rendered = ", ".join(parts)
    location = _person_location_label(person)
    if location:
        rendered += f" in {location}"
    office_suffix = _person_office_suffix(person)
    if office_suffix:
        rendered += office_suffix
    return rendered


def render_clarification(people: tuple[CanonicalPersonRecord, ...]) -> str:
    """Render a short clarification list for ambiguous person matches."""

    rendered = [
        render_person_with_context(person)
        for person in people[:5]
    ]
    if len(rendered) == 1:
        return rendered[0]
    if len(rendered) == 2:
        return f"{rendered[0]} or {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", or {rendered[-1]}"


def render_staff_lookup_answer(match: StaffLookupMatch) -> str:
    """Render one staff-id answer directly from structured rows."""

    if match.canonical_people:
        person = match.canonical_people[0]
        answer = f"Staff ID {match.staff_no} belongs to {person.canonical_name}"
        if person.designation:
            answer += f", {person.designation}"
        if person.department_name:
            answer += f" in {person.department_name}"
        if person.office_name:
            answer += f" ({person.office_name})"
        answer += "."
        return answer

    row = match.raw_rows[0]
    answer = f"Staff ID {match.staff_no} belongs to {row.canonical_name}"
    if row.designation:
        answer += f", {row.designation}"
    if row.department_name:
        answer += f" in {row.department_name}"
    if row.office_name:
        answer += f" ({row.office_name})"
    answer += "."
    return answer


def _normalize_staff_no(value: str | None) -> str:
    normalized = normalize_whitespace(value)
    if not normalized:
        return ""
    return normalized.lower().replace(" ", "")


def _primary_person_title(person: CanonicalPersonRecord) -> str:
    if person.designation_group == "chairperson":
        return "Chairperson"
    if person.designation_group == "whole_time_member":
        return "Whole-Time Member"
    if person.designation_group == "board_member":
        if person.board_role:
            cleaned = person.board_role.replace(", SEBI", "").strip()
            return cleaned or "Board Member"
        return "Board Member"
    return person.designation or person.board_role or "SEBI official"


def _person_location_label(person: CanonicalPersonRecord) -> str | None:
    if person.designation_group in {"chairperson", "whole_time_member", "board_member"}:
        return None
    return person.department_name


def _person_office_suffix(person: CanonicalPersonRecord) -> str:
    if not person.office_name:
        return ""
    if person.designation_group in {"chairperson", "whole_time_member", "board_member"}:
        return ""
    return f" ({person.office_name})"
