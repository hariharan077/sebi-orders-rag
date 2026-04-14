"""Canonicalize SEBI office/contact rows into one merged office layer."""

from __future__ import annotations

import hashlib

from ..directory_data.models import DirectoryOfficeRecord, normalize_email, normalize_phone, normalize_whitespace
from .canonical_models import CanonicalOfficeRecord

_CITY_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "mumbai": ("mumbai", "bombay"),
    "delhi": ("delhi", "new delhi"),
    "chennai": ("chennai", "madras"),
    "kolkata": ("kolkata", "calcutta"),
    "ahmedabad": ("ahmedabad",),
    "indore": ("indore",),
}
_REGIONAL_OFFICE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("northern regional office", ("northern regional office", "nro")),
    ("southern regional office", ("southern regional office", "sro")),
    ("eastern regional office", ("eastern regional office", "ero")),
    ("western regional office", ("western regional office", "wro")),
)
_HEAD_OFFICE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sebi bhavan ii", ("sebi bhavan ii", "sebi bhavan 2", "bhavan ii", "bhavan 2")),
    ("sebi bhavan", ("sebi bhavan", "bkc", "bandra kurla complex")),
    ("ncl office", ("ncl office",)),
)
_SOURCE_PRIORITY: dict[str, int] = {
    "contact_us": 120,
    "regional_offices": 100,
    "directory": 80,
}


def canonicalize_offices(
    offices: tuple[DirectoryOfficeRecord, ...],
) -> tuple[CanonicalOfficeRecord, ...]:
    """Merge office rows while preferring richer contact-us rows."""

    grouped: dict[str, list[DirectoryOfficeRecord]] = {}
    for record in offices:
        grouped.setdefault(canonical_office_key(record), []).append(record)

    merged: list[CanonicalOfficeRecord] = []
    for office_key, records in sorted(grouped.items()):
        ordered = sorted(records, key=_office_sort_key)
        representative = ordered[0]
        office_name = _choose_office_field(ordered, "office_name") or representative.office_name
        aliases = tuple(
            sorted(
                {
                    alias
                    for record in ordered
                    for alias in office_aliases(record)
                    if alias
                }
            )
        )
        merged.append(
            CanonicalOfficeRecord(
                canonical_office_id=_canonical_id("office", office_key),
                office_name=office_name,
                normalized_office_key=office_key,
                office_aliases=aliases,
                office_type=_choose_office_field(ordered, "office_type"),
                city=_canonical_city(_choose_office_field(ordered, "city")),
                state=_choose_office_field(ordered, "state"),
                region=_choose_office_field(ordered, "region"),
                address=_choose_office_field(ordered, "address"),
                email=normalize_email(_choose_office_field(ordered, "email")),
                phone=normalize_phone(_choose_office_field(ordered, "phone")),
                fax=normalize_phone(_choose_office_field(ordered, "fax")),
                source_priority=max(_source_priority(record.source_type) for record in ordered),
                active_status=True,
                source_types=tuple(dict.fromkeys(record.source_type for record in ordered)),
                source_urls=tuple(dict.fromkeys(record.source_url for record in ordered)),
                source_row_count=len(ordered),
            )
        )
    return tuple(
        sorted(
            merged,
            key=lambda record: ((record.city or "").lower(), record.office_name.lower()),
        )
    )


def canonical_office_key(record: DirectoryOfficeRecord) -> str:
    """Return the canonical grouping key for one office row."""

    normalized_name = normalize_lookup_key(record.office_name)
    for label, variants in (*_REGIONAL_OFFICE_ALIASES, *_HEAD_OFFICE_ALIASES):
        if any(variant in normalized_name for variant in variants):
            return label
    city = _canonical_city(record.city)
    if city and record.office_type:
        return normalize_lookup_key(f"{city} {record.office_type}")
    if city:
        return normalize_lookup_key(city)
    return normalized_name


def office_aliases(record: DirectoryOfficeRecord) -> tuple[str, ...]:
    """Return deterministic office aliases for matching."""

    values: list[str] = []
    office_name = normalize_whitespace(record.office_name)
    if office_name:
        values.append(office_name)
    normalized_name = normalize_lookup_key(record.office_name)
    for label, variants in (*_REGIONAL_OFFICE_ALIASES, *_HEAD_OFFICE_ALIASES):
        if any(variant in normalized_name for variant in variants):
            values.extend(variants)
    city = _canonical_city(record.city)
    if city:
        values.append(city)
        values.append(f"{city} office")
        if record.office_type == "regional_office":
            values.append(f"{city} regional office")
    return tuple(
        dict.fromkeys(
            value
            for value in (normalize_whitespace(item) for item in values)
            if value
        )
    )


def normalize_lookup_key(value: str | None) -> str:
    """Normalize free text for deterministic office matching."""

    cleaned = (value or "").lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = "".join(char if char.isalnum() else " " for char in cleaned)
    return " ".join(cleaned.split())


def office_match_score(office: CanonicalOfficeRecord, hint: str) -> int:
    """Score one office against one location/contact hint."""

    normalized_hint = normalize_lookup_key(hint)
    if not normalized_hint:
        return 0

    score = 0
    if normalized_hint == office.normalized_office_key:
        score += 120
    if office.office_name and normalize_lookup_key(office.office_name) == normalized_hint:
        score += 110
    if office.city and normalize_lookup_key(office.city) == normalized_hint:
        score += 95
    if office.city and normalize_lookup_key(office.city) in normalized_hint:
        score += 75
    for alias in office.office_aliases:
        alias_key = normalize_lookup_key(alias)
        if alias_key == normalized_hint:
            score += 105
        elif alias_key and alias_key in normalized_hint:
            score += 70
    if office.region and normalize_lookup_key(office.region) in normalized_hint:
        score += 30
    return score


def match_canonical_offices(
    offices: tuple[CanonicalOfficeRecord, ...],
    hint: str,
) -> tuple[CanonicalOfficeRecord, ...]:
    """Return canonical offices ordered by deterministic office-match strength."""

    scored = [
        (office_match_score(office, hint), office)
        for office in offices
    ]
    scored = [item for item in scored if item[0] >= 60]
    scored.sort(key=lambda item: (-item[0], item[1].office_name))
    return tuple(record for _, record in scored)


def is_generic_city_office_query(query: str, city: str | None) -> bool:
    """Return true when the user asked for a generic city office."""

    normalized_query = normalize_lookup_key(query)
    normalized_city = normalize_lookup_key(city)
    if not normalized_city or normalized_city not in normalized_query:
        return False
    if any(token in normalized_query for token in ("bhavan", "ncl office", "nro", "sro", "ero", "wro")):
        return False
    return any(token in normalized_query for token in ("office", "address", "location", "where is", "where are"))


def _choose_office_field(records: list[DirectoryOfficeRecord], field_name: str) -> str | None:
    best_value: str | None = None
    best_rank = (-1, -1)
    for record in records:
        value = normalize_whitespace(getattr(record, field_name))
        if not value:
            continue
        rank = (_source_priority(record.source_type), len(value))
        if rank > best_rank:
            best_value = value
            best_rank = rank
    return best_value


def _office_sort_key(record: DirectoryOfficeRecord) -> tuple[int, int, str]:
    richness = sum(
        1
        for value in (record.address, record.email, record.phone, record.fax, record.city, record.state)
        if normalize_whitespace(value)
    )
    return (-_source_priority(record.source_type), -richness, record.office_name.lower())


def _source_priority(source_type: str) -> int:
    return _SOURCE_PRIORITY.get(source_type, 0)


def _canonical_city(value: str | None) -> str | None:
    cleaned = normalize_lookup_key(value)
    if not cleaned:
        return None
    for canonical, aliases in _CITY_ALIAS_MAP.items():
        if cleaned in aliases:
            return canonical.title() if canonical != "delhi" else "Delhi"
    return normalize_whitespace(value)


def _canonical_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"
