"""Deterministic canonicalization for structured SEBI reference rows."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..normalization import normalize_department_alias, normalize_designation_alias
from .models import (
    BoardMemberRecord,
    DirectoryOfficeRecord,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
    OrgStructureRecord,
    normalize_email,
    normalize_phone,
    normalize_whitespace,
)

_HONORIFIC_RE = re.compile(r"\b(?:shri|smt|ms|mrs|mr|dr)\.?\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_ROMAN_TWO_RE = re.compile(r"\bii\b", re.IGNORECASE)
_ABBREV_RE = re.compile(r"\(([^)]+)\)")
_CITY_NAMES = ("mumbai", "new delhi", "delhi", "chennai", "kolkata", "ahmedabad", "indore")
_CITY_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "mumbai": ("mumbai", "bombay"),
    "delhi": ("new delhi", "delhi"),
    "chennai": ("chennai", "madras"),
    "kolkata": ("kolkata", "calcutta"),
    "ahmedabad": ("ahmedabad",),
    "indore": ("indore",),
}


@dataclass(frozen=True)
class CanonicalPersonRecord:
    """Merged current person record across directory, org-chart, and board sources."""

    canonical_name: str
    canonical_key: str
    designation: str | None = None
    designation_key: str | None = None
    role_group: str | None = None
    department_name: str | None = None
    department_key: str | None = None
    office_name: str | None = None
    office_key: str | None = None
    email: str | None = None
    phone: str | None = None
    date_of_joining: str | None = None
    staff_no: str | None = None
    board_role: str | None = None
    board_category: str | None = None
    is_board_member: bool = False
    source_records: tuple[DirectoryPersonRecord | BoardMemberRecord, ...] = ()


@dataclass(frozen=True)
class CanonicalOfficeRecord:
    """Merged current office record across contact and regional-office sources."""

    canonical_key: str
    canonical_name: str
    office_type: str | None = None
    region: str | None = None
    address: str | None = None
    email: str | None = None
    phone: str | None = None
    fax: str | None = None
    city: str | None = None
    state: str | None = None
    aliases: tuple[str, ...] = ()
    source_records: tuple[DirectoryOfficeRecord, ...] = ()


@dataclass(frozen=True)
class CanonicalReferenceDataset:
    """Canonical structured reference view used by the answer layer."""

    people: tuple[CanonicalPersonRecord, ...] = ()
    board_members: tuple[CanonicalPersonRecord, ...] = ()
    offices: tuple[CanonicalOfficeRecord, ...] = ()
    org_structure: tuple[OrgStructureRecord, ...] = ()


def canonicalize_reference_dataset(dataset: DirectoryReferenceDataset) -> CanonicalReferenceDataset:
    """Return a canonical merged view over the active structured dataset."""

    people = canonicalize_people(dataset.people, dataset.board_members)
    offices = canonicalize_offices(dataset.offices)
    board_members = tuple(
        sorted(
            (person for person in people if person.is_board_member),
            key=_board_member_sort_key,
        )
    )
    return CanonicalReferenceDataset(
        people=people,
        board_members=board_members,
        offices=offices,
        org_structure=dataset.org_structure,
    )


def canonicalize_people(
    people: tuple[DirectoryPersonRecord, ...],
    board_members: tuple[BoardMemberRecord, ...],
) -> tuple[CanonicalPersonRecord, ...]:
    """Merge duplicate person rows across structured sources into one current row."""

    grouped_people: dict[str, list[DirectoryPersonRecord]] = {}
    grouped_board: dict[str, list[BoardMemberRecord]] = {}

    for record in people:
        grouped_people.setdefault(normalize_lookup_key(record.canonical_name), []).append(record)
    for record in board_members:
        grouped_board.setdefault(normalize_lookup_key(record.canonical_name), []).append(record)

    canonical_people: list[CanonicalPersonRecord] = []
    for canonical_key in sorted(set(grouped_people) | set(grouped_board)):
        person_rows = grouped_people.get(canonical_key, [])
        board_rows = grouped_board.get(canonical_key, [])
        source_records: tuple[DirectoryPersonRecord | BoardMemberRecord, ...] = tuple(person_rows) + tuple(board_rows)
        display_name = _choose_display_name(person_rows, board_rows)
        role_group = _choose_role_group(person_rows, board_rows)
        designation = _choose_designation(person_rows, board_rows, role_group=role_group)
        canonical_people.append(
            CanonicalPersonRecord(
                canonical_name=display_name,
                canonical_key=canonical_key,
                designation=designation,
                designation_key=normalize_designation_key(designation),
                role_group=role_group,
                department_name=_choose_person_field(person_rows, "department_name"),
                department_key=normalize_department(
                    _choose_person_field(person_rows, "department_name")
                ),
                office_name=_choose_person_field(person_rows, "office_name"),
                office_key=normalize_lookup_key(_choose_person_field(person_rows, "office_name")),
                email=normalize_email(_choose_person_field(person_rows, "email")),
                phone=normalize_phone(_choose_person_field(person_rows, "phone")),
                date_of_joining=_choose_person_field(person_rows, "date_of_joining"),
                staff_no=_choose_person_field(person_rows, "staff_no"),
                board_role=_choose_board_role(board_rows),
                board_category=_choose_board_category(board_rows),
                is_board_member=bool(board_rows),
                source_records=source_records,
            )
        )
    return tuple(canonical_people)


def canonicalize_offices(offices: tuple[DirectoryOfficeRecord, ...]) -> tuple[CanonicalOfficeRecord, ...]:
    """Merge duplicate office rows so richer contact-us records are preferred."""

    grouped: dict[str, list[DirectoryOfficeRecord]] = {}
    for record in offices:
        grouped.setdefault(_office_canonical_key(record), []).append(record)

    canonical_offices: list[CanonicalOfficeRecord] = []
    for canonical_key, records in sorted(grouped.items()):
        aliases = sorted({alias for record in records for alias in _office_aliases(record) if alias})
        canonical_offices.append(
            CanonicalOfficeRecord(
                canonical_key=canonical_key,
                canonical_name=_choose_office_name(records),
                office_type=_choose_office_field(records, "office_type"),
                region=_choose_office_field(records, "region"),
                address=_choose_office_field(records, "address"),
                email=normalize_email(_choose_office_field(records, "email")),
                phone=normalize_phone(_choose_office_field(records, "phone")),
                fax=normalize_phone(_choose_office_field(records, "fax")),
                city=_choose_office_field(records, "city"),
                state=_choose_office_field(records, "state"),
                aliases=tuple(aliases),
                source_records=tuple(_sort_office_records(records)),
            )
        )
    return tuple(sorted(canonical_offices, key=lambda record: (record.city or "", record.canonical_name)))


def normalize_lookup_key(value: str | None) -> str:
    """Normalize free text for deterministic person and office matching."""

    cleaned = (value or "").lower()
    cleaned = _HONORIFIC_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = _ROMAN_TWO_RE.sub("2", cleaned)
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def normalize_designation(value: str | None) -> str | None:
    """Normalize key leadership designations while preserving other titles."""

    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    canonical = normalize_designation_alias(cleaned)
    if canonical is not None:
        return canonical
    lowered = cleaned.lower()
    if "chairman" in lowered or "chairperson" in lowered:
        return "Chairperson"
    if "whole-time member" in lowered or "whole time member" in lowered or lowered == "wtm":
        return "Whole-Time Member"
    if "executive director" in lowered or lowered == "ed":
        return "Executive Director"
    if "regional director" in lowered:
        return "Regional Director"
    if "part-time member" in lowered or "part time member" in lowered:
        return "Part-Time Member"
    return cleaned.title() if cleaned == cleaned.lower() else cleaned


def normalize_designation_key(value: str | None) -> str | None:
    """Return one stable lowercase lookup key for designation filtering."""

    normalized = normalize_designation(value)
    if not normalized:
        return None
    return normalize_lookup_key(normalized)


def normalize_department(value: str | None) -> str | None:
    """Return one stable uppercase department key when it is inferable."""

    alias = normalize_department_alias(value)
    if alias is not None:
        return alias
    normalized = normalize_lookup_key(value)
    return normalized.upper() if normalized and len(normalized) <= 6 else None


def match_canonical_offices(
    offices: tuple[CanonicalOfficeRecord, ...],
    hint: str,
) -> tuple[CanonicalOfficeRecord, ...]:
    """Return offices ordered by deterministic match strength for one hint."""

    normalized_hint = normalize_lookup_key(hint)
    scored: list[tuple[int, CanonicalOfficeRecord]] = []
    for office in offices:
        score = _score_office_match(office, normalized_hint)
        if score >= 20:
            scored.append((score, office))
    scored.sort(key=lambda item: (-item[0], item[1].canonical_name))
    return tuple(record for _, record in scored)


def is_generic_city_office_query(query: str, city: str | None) -> bool:
    """Return true when the office query names only a city-level office location."""

    normalized_query = normalize_lookup_key(query)
    city_keys = _city_aliases(city)
    if not city_keys or not any(city_key in normalized_query for city_key in city_keys):
        return False
    if any(token in normalized_query for token in ("head office", "bkc", "bhavan", "ncl", "nro", "sro", "wro", "ero", "regional office", "local office")):
        return False
    return "office" in normalized_query or "address" in normalized_query or "location" in normalized_query


def _choose_display_name(
    people: list[DirectoryPersonRecord],
    board_members: list[BoardMemberRecord],
) -> str:
    candidates: list[tuple[tuple[int, int, int, str], str]] = []
    for record in people:
        if record.canonical_name:
            name = normalize_whitespace(record.canonical_name) or record.canonical_name
            candidates.append((_display_name_sort_key(name, record.source_type), name))
    for record in board_members:
        if record.canonical_name:
            name = normalize_whitespace(record.canonical_name) or record.canonical_name
            candidates.append((_display_name_sort_key(name, record.source_type), name))
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _choose_role_group(
    people: list[DirectoryPersonRecord],
    board_members: list[BoardMemberRecord],
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for record in board_members:
        mapped = _role_group_from_board_category(record.category)
        if mapped:
            candidates.append((_role_group_priority(mapped), mapped))
    for record in people:
        mapped = _normalize_role_group(record.role_group, record.designation)
        if mapped:
            candidates.append((_role_group_priority(mapped), mapped))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _choose_designation(
    people: list[DirectoryPersonRecord],
    board_members: list[BoardMemberRecord],
    *,
    role_group: str | None,
) -> str | None:
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for record in board_members:
        value = normalize_designation(record.board_role)
        if value:
            candidates.append((_person_field_sort_key(record, "designation"), value))
    for record in people:
        value = normalize_designation(record.designation)
        if value:
            candidates.append((_person_field_sort_key(record, "designation"), value))
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[0][1]
    if role_group == "chairperson":
        return "Chairperson"
    if role_group == "wtm":
        return "Whole-Time Member"
    if role_group == "executive_director":
        return "Executive Director"
    if role_group == "regional_director":
        return "Regional Director"
    if role_group == "board_member":
        return "Board Member"
    return None


def _choose_person_field(records: list[DirectoryPersonRecord], field_name: str) -> str | None:
    candidates: list[tuple[tuple[int, int, str], str]] = []
    for record in records:
        value = normalize_whitespace(getattr(record, field_name, None))
        if value:
            candidates.append((_person_field_sort_key(record, field_name), value))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _choose_board_role(records: list[BoardMemberRecord]) -> str | None:
    candidates = [normalize_whitespace(record.board_role) for record in records if normalize_whitespace(record.board_role)]
    return candidates[0] if candidates else None


def _choose_board_category(records: list[BoardMemberRecord]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for record in records:
        if record.category:
            candidates.append((_board_category_priority(record.category), record.category))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _normalize_role_group(role_group: str | None, designation: str | None) -> str | None:
    lowered_role = (role_group or "").lower()
    lowered_designation = (designation or "").lower()
    if lowered_role in {"chairperson", "wtm", "executive_director", "regional_director", "staff", "board_member"}:
        return lowered_role
    if "chair" in lowered_designation:
        return "chairperson"
    if "whole-time member" in lowered_designation or "whole time member" in lowered_designation:
        return "wtm"
    if "executive director" in lowered_designation:
        return "executive_director"
    if "regional director" in lowered_designation:
        return "regional_director"
    if lowered_designation:
        return "staff"
    return None


def _role_group_from_board_category(category: str | None) -> str | None:
    if category == "chairperson":
        return "chairperson"
    if category == "whole_time_member":
        return "wtm"
    if category in {"part_time_member", "government_nominee", "rbi_nominee"}:
        return "board_member"
    return None


def _role_group_priority(role_group: str) -> int:
    return {
        "chairperson": 0,
        "wtm": 1,
        "executive_director": 2,
        "regional_director": 3,
        "board_member": 4,
        "staff": 5,
        "office_contact": 6,
    }.get(role_group, 99)


def _board_category_priority(category: str) -> int:
    return {
        "chairperson": 0,
        "whole_time_member": 1,
        "government_nominee": 2,
        "rbi_nominee": 3,
        "part_time_member": 4,
    }.get(category, 99)


def _display_name_sort_key(name: str, source_type: str) -> tuple[int, int, int, str]:
    normalized_name = normalize_lookup_key(name)
    has_honorific = 1 if _HONORIFIC_RE.search(name) else 0
    return (_source_priority(source_type, field_name="display_name"), has_honorific, len(normalized_name), normalized_name)


def _person_field_sort_key(record: DirectoryPersonRecord | BoardMemberRecord, field_name: str) -> tuple[int, int, str]:
    normalized_name = normalize_lookup_key(getattr(record, "canonical_name", ""))
    richness = -sum(
        1
        for candidate_field in (
            "designation",
            "role_group",
            "department_name",
            "office_name",
            "email",
            "phone",
            "date_of_joining",
            "staff_no",
        )
        if normalize_whitespace(getattr(record, candidate_field, None))
    )
    return (_source_priority(record.source_type, field_name=field_name), richness, normalized_name)


def _source_priority(source_type: str, *, field_name: str) -> int:
    field_priorities = {
        "display_name": {
            "directory": 0,
            "orgchart": 1,
            "board_members": 2,
            "regional_offices": 3,
            "contact_us": 4,
        },
        "designation": {
            "board_members": 0,
            "directory": 1,
            "orgchart": 2,
            "regional_offices": 3,
            "contact_us": 4,
        },
        "department_name": {
            "orgchart": 0,
            "directory": 1,
            "regional_offices": 2,
            "board_members": 3,
            "contact_us": 4,
        },
        "office_name": {
            "regional_offices": 0,
            "directory": 1,
            "contact_us": 2,
            "orgchart": 3,
            "board_members": 4,
        },
        "email": {
            "directory": 0,
            "orgchart": 1,
            "regional_offices": 2,
            "board_members": 3,
            "contact_us": 4,
        },
        "phone": {
            "directory": 0,
            "regional_offices": 1,
            "orgchart": 2,
            "board_members": 3,
            "contact_us": 4,
        },
        "date_of_joining": {
            "directory": 0,
            "orgchart": 1,
            "regional_offices": 2,
            "board_members": 3,
            "contact_us": 4,
        },
        "staff_no": {
            "directory": 0,
            "orgchart": 1,
            "regional_offices": 2,
            "board_members": 3,
            "contact_us": 4,
        },
    }
    priorities = field_priorities.get(field_name, field_priorities["designation"])
    return priorities.get(source_type, 99)


def _board_member_sort_key(record: CanonicalPersonRecord) -> tuple[int, str]:
    return (_board_category_priority(record.board_category or ""), record.canonical_name)


def _office_canonical_key(record: DirectoryOfficeRecord) -> str:
    office_name = normalize_lookup_key(record.office_name)
    abbreviation = _office_abbreviation(record.office_name)
    if abbreviation in {"nro", "sro", "ero", "wro"}:
        return f"regional:{abbreviation}"
    if "sebi bhavan ii" in office_name or "sebi bhavan 2" in office_name:
        return "mumbai:sebi_bhavan_ii_bkc"
    if "sebi bhavan bkc" in office_name:
        return "mumbai:sebi_bhavan_bkc"
    if "ncl office" in office_name:
        return "mumbai:ncl_office"
    if "nariman point" in office_name:
        return "mumbai:nariman_point_office"
    if record.office_type == "regional_office" and record.city:
        return f"regional:{normalize_lookup_key(record.city)}"
    if record.office_type == "local_office" and record.city:
        return f"local:{normalize_lookup_key(record.city)}"
    if record.city:
        return f"office:{normalize_lookup_key(record.city)}"
    return office_name


def _office_aliases(record: DirectoryOfficeRecord) -> set[str]:
    aliases: set[str] = set()
    office_name = normalize_lookup_key(record.office_name)
    if office_name:
        aliases.add(office_name)

    abbreviation = _office_abbreviation(record.office_name)
    if abbreviation:
        aliases.add(abbreviation)
    if record.city:
        for city_key in _city_aliases(record.city):
            aliases.add(city_key)
            aliases.add(f"{city_key} office")
            aliases.add(f"sebi {city_key} office")
            aliases.add(f"office in {city_key}")
            aliases.add(f"location of {city_key} sebi office")

    if record.region == "north":
        aliases.update({"northern regional office", "north regional office"})
    if record.region == "south":
        aliases.update({"southern regional office", "south regional office"})
    if record.region == "east":
        aliases.update({"eastern regional office", "east regional office"})
    if record.region == "west":
        aliases.update({"western regional office", "west regional office"})

    if "sebi bhavan bkc" in office_name:
        aliases.update({"head office", "sebi head office", "mumbai head office", "bkc office"})
    if "sebi bhavan ii" in office_name or "sebi bhavan 2" in office_name:
        aliases.update({"sebi bhavan ii", "sebi bhavan 2", "bhavan ii", "bhavan 2"})
    if "ncl office" in office_name:
        aliases.add("ncl office")

    return {alias for alias in aliases if alias}


def _office_abbreviation(office_name: str | None) -> str | None:
    cleaned = normalize_whitespace(office_name)
    if not cleaned:
        return None
    match = _ABBREV_RE.search(cleaned)
    if match is None:
        return None
    return normalize_lookup_key(match.group(1))


def _choose_office_name(records: list[DirectoryOfficeRecord]) -> str:
    candidates: list[tuple[tuple[int, int, str], str]] = []
    for record in records:
        name = normalize_whitespace(record.office_name)
        if not name:
            continue
        normalized_name = normalize_lookup_key(name)
        score = 0
        if record.city and normalize_lookup_key(record.city) not in normalized_name:
            score += 2
        if record.office_type == "regional_office" and record.city and normalize_lookup_key(record.city) in normalized_name:
            score -= 1
        if _office_abbreviation(name):
            score -= 1
        candidates.append(((score, -len(name), normalized_name), name))
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _choose_office_field(records: list[DirectoryOfficeRecord], field_name: str) -> str | None:
    candidates: list[tuple[tuple[int, int, str], str]] = []
    for record in records:
        value = normalize_whitespace(getattr(record, field_name, None))
        if value:
            candidates.append((_office_field_sort_key(record, field_name), value))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _office_field_sort_key(record: DirectoryOfficeRecord, field_name: str) -> tuple[int, int, str]:
    priorities = {
        "address": {"contact_us": 0, "regional_offices": 1, "directory": 2, "orgchart": 3},
        "phone": {"contact_us": 0, "regional_offices": 1, "directory": 2, "orgchart": 3},
        "fax": {"contact_us": 0, "regional_offices": 1, "directory": 2, "orgchart": 3},
        "email": {"contact_us": 0, "regional_offices": 1, "directory": 2, "orgchart": 3},
        "office_type": {"contact_us": 0, "regional_offices": 1, "directory": 2, "orgchart": 3},
        "region": {"regional_offices": 0, "contact_us": 1, "directory": 2, "orgchart": 3},
        "city": {"contact_us": 0, "regional_offices": 1, "directory": 2, "orgchart": 3},
        "state": {"contact_us": 0, "regional_offices": 1, "directory": 2, "orgchart": 3},
    }
    source_priority = priorities.get(field_name, priorities["address"]).get(record.source_type, 99)
    completeness = -sum(
        1 for candidate_field in ("address", "email", "phone", "fax", "city", "state") if getattr(record, candidate_field, None)
    )
    return (source_priority, completeness, normalize_lookup_key(record.office_name))


def _sort_office_records(records: list[DirectoryOfficeRecord]) -> list[DirectoryOfficeRecord]:
    return sorted(records, key=lambda record: _office_field_sort_key(record, "address"))


def _score_office_match(record: CanonicalOfficeRecord, normalized_hint: str) -> int:
    score = 0
    ignored_tokens = {
        "address",
        "contact",
        "director",
        "is",
        "location",
        "of",
        "office",
        "regional",
        "sebi",
        "the",
        "what",
        "where",
        "who",
    }
    hint_tokens = {token for token in normalized_hint.split() if token not in ignored_tokens}
    for alias in record.aliases:
        if alias == normalized_hint:
            score += 160
        elif alias and alias in normalized_hint:
            score += 70
        elif normalized_hint and normalized_hint in alias:
            score += 90
        overlap = hint_tokens.intersection(token for token in alias.split() if token not in ignored_tokens)
        score += len(overlap) * 8

    city_key = normalize_lookup_key(record.city)
    if city_key and city_key in normalized_hint:
        score += 35
    if "head office" in normalized_hint and "head office" in record.aliases:
        score += 100
    if any(city in normalized_hint for city in _CITY_NAMES) and record.address:
        score += 10
    return score


def _city_aliases(city: str | None) -> tuple[str, ...]:
    city_key = normalize_lookup_key(city)
    if not city_key:
        return ()
    for aliases in _CITY_ALIAS_MAP.values():
        if city_key in aliases:
            return aliases
    return (city_key,)
