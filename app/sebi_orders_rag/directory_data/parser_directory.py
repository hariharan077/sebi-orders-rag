"""Structured parser for the official SEBI directory page."""

from __future__ import annotations

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
from .parser_offices import _city_from_office_name, _office_type_from_name, _region_from_office_name
from .sources import SOURCE_DIRECTORY

_HEADER_KEY_BY_LABEL = {
    "staff no": "staff_no",
    "name": "canonical_name",
    "name of the staff member": "canonical_name",
    "date of joining": "date_of_joining",
    "email id": "email",
    "telephone no": "phone",
    "department": "department_name",
    "designation": "designation",
}
_TITLE_ONLY_COLUMNS = {
    "executive director": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "email",
        "phone",
    ),
    "chief general manager": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "department_name",
        "email",
        "phone",
    ),
    "general manager": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "department_name",
        "email",
        "phone",
    ),
    "deputy general manager": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "department_name",
        "email",
        "phone",
    ),
    "assistant general manager": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "department_name",
        "email",
        "phone",
    ),
    "manager": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "department_name",
        "email",
        "phone",
    ),
    "assistant manager": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "department_name",
        "email",
        "phone",
    ),
    "indore local office": (
        "staff_no",
        "canonical_name",
        "date_of_joining",
        "designation",
        "department_name",
        "email",
        "phone",
    ),
}
_INFERRED_DESIGNATION_TITLES = {
    "chairman",
    "whole time member",
    "chief vigilance officer",
    "executive director",
    "chief general manager",
    "general manager",
    "deputy general manager",
    "assistant general manager",
    "manager",
    "assistant manager",
}


def parse_directory_page(
    raw_html: str,
    *,
    source_url: str,
    source_type: str = SOURCE_DIRECTORY,
) -> DirectoryPageParseResult:
    """Parse official SEBI directory staff rows into structured person records."""

    soup = BeautifulSoup(raw_html, "html.parser")
    people: list[DirectoryPersonRecord] = []
    offices = _parse_directory_offices(
        soup,
        source_url=source_url,
        source_type=source_type,
    )

    for portlet in soup.select("div.portlet1.box1.green"):
        office_heading = _clean_office_heading(
            portlet.select_one(".portlet-title h2"),
        )
        for table in portlet.select("table.table1"):
            columns = _table_columns(table)
            if "canonical_name" not in columns:
                continue
            table_title = _table_title(table)
            for row in table.select("tbody tr"):
                values = [cell.get_text(" ", strip=True) for cell in row.select("td")]
                if not values:
                    continue
                if len(values) < len(columns):
                    values += [""] * (len(columns) - len(values))
                data = {
                    key: normalize_whitespace(values[index])
                    for index, key in enumerate(columns)
                    if index < len(values)
                }
                canonical_name = normalize_person_name(data.get("canonical_name"))
                if not canonical_name:
                    continue
                designation = data.get("designation") or _designation_from_table_title(table_title)
                role_group = _role_group_from_values(table_title, designation)
                people.append(
                    DirectoryPersonRecord(
                        source_type=source_type,
                        source_url=source_url,
                        canonical_name=canonical_name,
                        designation=designation,
                        role_group=role_group,
                        department_name=data.get("department_name"),
                        office_name=office_heading,
                        email=normalize_email(data.get("email")),
                        phone=normalize_phone(data.get("phone")),
                        date_of_joining=data.get("date_of_joining"),
                        staff_no=data.get("staff_no"),
                    ).with_hash()
                )

    return DirectoryPageParseResult(
        people=tuple(_dedupe_people(people)),
        offices=tuple(_dedupe_offices(offices)),
    )


def _table_title(table) -> str | None:
    title_row = table.select_one("thead tr th[colspan]")
    if title_row is None:
        return None
    return normalize_whitespace(title_row.get_text(" ", strip=True))


def _table_columns(table) -> list[str]:
    header_rows = table.select("thead tr")
    if not header_rows:
        return []
    last_header = header_rows[-1]
    labels = [normalize_whitespace(th.get_text(" ", strip=True)) or "" for th in last_header.select("th")]
    columns = [_HEADER_KEY_BY_LABEL.get(label.lower(), label.lower()) for label in labels]
    if "canonical_name" in columns:
        return columns
    return list(_TITLE_ONLY_COLUMNS.get((_table_title(table) or "").lower(), ()))


def _designation_from_table_title(table_title: str | None) -> str | None:
    title = normalize_whitespace(table_title)
    if not title:
        return None
    if title.lower() in _INFERRED_DESIGNATION_TITLES:
        return title
    return None


def _role_group_from_values(table_title: str | None, designation: str | None) -> str | None:
    table_lower = (table_title or "").strip().lower()
    designation_lower = (designation or "").strip().lower()
    if table_lower == "chairman" or designation_lower == "chairman":
        return "chairperson"
    if table_lower == "whole time member" or designation_lower == "whole time member":
        return "wtm"
    if table_lower == "executive director" or "executive director" in designation_lower:
        return "executive_director"
    if "regional director" in designation_lower:
        return "regional_director"
    if "chief vigilance officer" in designation_lower or table_lower == "chief vigilance officer":
        return "chief_vigilance_officer"
    if designation:
        return "staff"
    return None


def _clean_office_heading(node) -> str | None:
    if node is None:
        return None
    heading = normalize_whitespace(node.get_text(" ", strip=True))
    if not heading:
        return None
    heading = heading.replace(" ,", ",")
    return heading


def _parse_directory_offices(
    soup: BeautifulSoup,
    *,
    source_url: str,
    source_type: str,
) -> list[DirectoryOfficeRecord]:
    table = soup.select_one("table.tel_fax_main")
    if table is None:
        return []

    office_contacts: dict[str, dict[str, str | None]] = {}
    cells = table.select("tbody > tr > td")
    if cells:
        _merge_office_contacts(cells[0], office_contacts, field_name="phone")
    if len(cells) > 1:
        _merge_office_contacts(cells[1], office_contacts, field_name="fax")

    offices: list[DirectoryOfficeRecord] = []
    for office_name, contact in office_contacts.items():
        offices.append(
            DirectoryOfficeRecord(
                source_type=source_type,
                source_url=source_url,
                office_name=office_name,
                office_type=_office_type_from_name(office_name),
                region=_region_from_office_name(office_name),
                city=_city_from_office_name(office_name),
                phone=normalize_phone(contact.get("phone")),
                fax=normalize_phone(contact.get("fax")),
            ).with_hash()
        )
    return offices


def _merge_office_contacts(
    container,
    office_contacts: dict[str, dict[str, str | None]],
    *,
    field_name: str,
) -> None:
    for row in container.select("table tr td"):
        office_name, value = _split_contact_row(row.get_text(" ", strip=True))
        if not office_name or not value:
            continue
        office_contacts.setdefault(office_name, {"phone": None, "fax": None})
        office_contacts[office_name][field_name] = value


def _split_contact_row(value: str | None) -> tuple[str | None, str | None]:
    cleaned = normalize_whitespace(value)
    if not cleaned or ":" not in cleaned:
        return None, None
    office_name, contact_value = cleaned.split(":", 1)
    return normalize_whitespace(office_name), normalize_whitespace(contact_value)


def _dedupe_people(records: list[DirectoryPersonRecord]) -> list[DirectoryPersonRecord]:
    seen: dict[str, DirectoryPersonRecord] = {}
    for record in records:
        if record.row_sha256 is None:
            continue
        seen[record.row_sha256] = record
    return list(seen.values())


def _dedupe_offices(records: list[DirectoryOfficeRecord]) -> list[DirectoryOfficeRecord]:
    seen: dict[str, DirectoryOfficeRecord] = {}
    for record in records:
        if record.row_sha256 is None:
            continue
        seen[record.row_sha256] = record
    return list(seen.values())
