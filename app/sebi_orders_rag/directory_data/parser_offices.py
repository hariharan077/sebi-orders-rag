"""Structured parsers for SEBI regional-office and contact-us pages."""

from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

from .models import (
    DirectoryOfficeRecord,
    DirectoryPageParseResult,
    DirectoryPersonRecord,
    normalize_email,
    normalize_person_name,
    normalize_phone,
    normalize_whitespace,
)
from .sources import SOURCE_CONTACT_US, SOURCE_REGIONAL_OFFICES

_LOCATIONS_BLOCK_RE = re.compile(
    r"var\s+locations\s*=\s*\[(.*?)\];\s*// Setup the different icons",
    re.DOTALL,
)
_LOCATION_ENTRY_RE = re.compile(
    r"\['((?:\\'|[^'])*)',\s*[-\d.]+,\s*[-\d.]+,\s*\"([^\"]+)\"\]",
    re.DOTALL,
)


def parse_regional_offices_page(
    raw_html: str,
    *,
    source_url: str,
    source_type: str = SOURCE_REGIONAL_OFFICES,
) -> DirectoryPageParseResult:
    """Parse the official regional offices contact page."""

    soup = BeautifulSoup(raw_html, "html.parser")
    table = soup.select_one("table.table")
    if table is None:
        return DirectoryPageParseResult()

    people: list[DirectoryPersonRecord] = []
    offices: list[DirectoryOfficeRecord] = []
    current_office_name: str | None = None

    for row in table.select("tr"):
        values = [normalize_whitespace(cell.get_text(" ", strip=True)) for cell in row.select("td")]
        values = [value for value in values if value]
        if not values:
            continue
        if len(values) == 4:
            current_office_name = values[0]
            person_text, phone, email = values[1], values[2], values[3]
        elif len(values) == 3 and current_office_name:
            person_text, phone, email = values[0], values[1], values[2]
        else:
            continue

        office_name = current_office_name
        if office_name is None:
            continue
        office_record = DirectoryOfficeRecord(
            source_type=source_type,
            source_url=source_url,
            office_name=office_name,
            office_type=_office_type_from_name(office_name),
            region=_region_from_office_name(office_name),
            city=_city_from_office_name(office_name),
        ).with_hash()
        offices.append(office_record)

        person_name, designation = _split_person_and_designation(person_text)
        if not person_name:
            continue
        people.append(
            DirectoryPersonRecord(
                source_type=source_type,
                source_url=source_url,
                canonical_name=person_name,
                designation=designation,
                role_group=_role_group_from_designation(designation),
                office_name=office_name,
                email=normalize_email(email),
                phone=normalize_phone(phone),
            ).with_hash()
        )

    return DirectoryPageParseResult(
        people=tuple(_dedupe_people(people)),
        offices=tuple(_dedupe_offices(offices)),
    )


def parse_contact_us_page(
    raw_html: str,
    *,
    source_url: str,
    source_type: str = SOURCE_CONTACT_US,
) -> DirectoryPageParseResult:
    """Parse the office-address/contact blocks embedded in the contact-us page."""

    match = _LOCATIONS_BLOCK_RE.search(raw_html)
    if match is None:
        return DirectoryPageParseResult()

    offices: list[DirectoryOfficeRecord] = []
    for raw_fragment, fallback_title in _LOCATION_ENTRY_RE.findall(match.group(1)):
        fragment_html = html.unescape(raw_fragment.replace("\\'", "'"))
        fragment_soup = BeautifulSoup(fragment_html, "html.parser")
        heading = fragment_soup.select_one("h2 span")
        office_name = normalize_whitespace(
            heading.get_text(" ", strip=True) if heading else fallback_title,
        )
        if not office_name:
            continue
        address, phone, fax, email = _parse_location_fragment(fragment_soup)
        city, state = _parse_city_state(address)
        offices.append(
            DirectoryOfficeRecord(
                source_type=source_type,
                source_url=source_url,
                office_name=office_name,
                office_type=_office_type_from_name(office_name),
                region=_region_from_office_name(office_name),
                address=address,
                email=email,
                phone=phone,
                fax=fax,
                city=city,
                state=state,
            ).with_hash()
        )
    return DirectoryPageParseResult(offices=tuple(_dedupe_offices(offices)))


def _parse_location_fragment(fragment_soup: BeautifulSoup) -> tuple[str | None, str | None, str | None, str | None]:
    address_lines: list[str] = []
    phone_parts: list[str] = []
    fax_parts: list[str] = []
    email_value: str | None = None
    inside_address = True

    for dt in fragment_soup.select("dt"):
        text = normalize_whitespace(dt.get_text(" ", strip=True))
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("address"):
            inside_address = True
            remainder = text.split(":", 1)[1].strip() if ":" in text else ""
            if remainder:
                address_lines.append(remainder)
            continue
        if lowered.startswith("tel"):
            inside_address = False
            value = text.split(":", 1)[1].strip() if ":" in text else text
            phone_parts.append(value)
            continue
        if lowered.startswith("fax"):
            inside_address = False
            value = text.split(":", 1)[1].strip() if ":" in text else text
            fax_parts.append(value)
            continue
        if lowered.startswith("email"):
            inside_address = False
            value = text.split(":", 1)[1].strip() if ":" in text else text
            email_value = normalize_email(value)
            continue
        if "toll free investor helpline" in lowered:
            inside_address = False
            value = text.split(":", 1)[1].strip() if ":" in text else text
            phone_parts.append(f"Toll Free Investor Helpline: {value}")
            continue
        if "interactive voice response system" in lowered:
            inside_address = False
            continue
        if inside_address:
            address_lines.append(text.rstrip(","))

    address = ", ".join(line for line in address_lines if line)
    return (
        normalize_whitespace(address),
        normalize_phone("; ".join(phone_parts) if phone_parts else None),
        normalize_phone("; ".join(fax_parts) if fax_parts else None),
        email_value,
    )


def _parse_city_state(address: str | None) -> tuple[str | None, str | None]:
    cleaned = normalize_whitespace(address)
    if not cleaned:
        return None, None
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return None, None
    if len(parts) >= 2 and re.match(r".+?-\s*\d{6}$", parts[-2]):
        city_match = re.match(r"(.+?)\s*-\s*\d{6}$", parts[-2])
        return normalize_whitespace(city_match.group(1) if city_match else parts[-2]), parts[-1]
    tail = parts[-1]
    match = re.match(r"(.+?)\s*-\s*\d{6}(?:,\s*(.+))?$", tail)
    if match:
        return normalize_whitespace(match.group(1)), normalize_whitespace(match.group(2))
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return tail, None


def _split_person_and_designation(value: str | None) -> tuple[str | None, str | None]:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None, None
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return None, None
    return normalize_person_name(parts[0]), normalize_whitespace(", ".join(parts[1:])) if len(parts) > 1 else None


def _role_group_from_designation(value: str | None) -> str | None:
    cleaned = (value or "").lower()
    if "regional director" in cleaned:
        return "regional_director"
    if "executive director" in cleaned or re.search(r"\bed\b", cleaned):
        return "executive_director"
    return "office_contact" if cleaned else None


def _office_type_from_name(office_name: str | None) -> str | None:
    cleaned = (office_name or "").lower()
    if "regional office" in cleaned:
        return "regional_office"
    if "local office" in cleaned:
        return "local_office"
    if "sebi bhavan" in cleaned or "ncl office" in cleaned:
        return "head_office"
    return None


def _region_from_office_name(office_name: str | None) -> str | None:
    cleaned = (office_name or "").lower()
    if "northern" in cleaned:
        return "north"
    if "southern" in cleaned:
        return "south"
    if "eastern" in cleaned:
        return "east"
    if "western" in cleaned:
        return "west"
    if "mumbai" in cleaned or "bhavan" in cleaned:
        return "head_office"
    return None


def _city_from_office_name(office_name: str | None) -> str | None:
    cleaned = normalize_whitespace(office_name)
    if not cleaned:
        return None
    if "," in cleaned:
        return normalize_whitespace(cleaned.split(",")[-1])
    for city in ("Mumbai", "New Delhi", "Chennai", "Kolkata", "Ahmedabad", "Indore"):
        if city.lower() in cleaned.lower():
            return city
    return None


def _dedupe_people(records: list[DirectoryPersonRecord]) -> list[DirectoryPersonRecord]:
    seen: dict[str, DirectoryPersonRecord] = {}
    for record in records:
        if record.row_sha256:
            seen[record.row_sha256] = record
    return list(seen.values())


def _dedupe_offices(records: list[DirectoryOfficeRecord]) -> list[DirectoryOfficeRecord]:
    seen: dict[str, DirectoryOfficeRecord] = {}
    for record in records:
        if record.row_sha256:
            seen[record.row_sha256] = record
    return list(seen.values())
