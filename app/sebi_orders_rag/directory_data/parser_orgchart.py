"""Structured parser for the official SEBI organisation structure page."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from .models import (
    DirectoryPageParseResult,
    DirectoryPersonRecord,
    OrgStructureRecord,
    normalize_email,
    normalize_person_name,
    normalize_phone,
    normalize_whitespace,
)
from .sources import SOURCE_ORGCHART


def parse_orgchart_page(
    raw_html: str,
    *,
    source_url: str,
    source_type: str = SOURCE_ORGCHART,
) -> DirectoryPageParseResult:
    """Parse the SEBI organisation structure page into people and leader mappings."""

    soup = BeautifulSoup(raw_html, "html.parser")
    people: list[DirectoryPersonRecord] = []
    mappings: list[OrgStructureRecord] = []

    for info in soup.select("div.orgchart div.tree-m-info"):
        name = normalize_person_name(_child_text(info, "h3"))
        role = normalize_whitespace(_child_text(info, "h4"))
        email = normalize_email(_child_text(info, "h5"))
        if not name or not role:
            continue
        people.append(
            DirectoryPersonRecord(
                source_type=source_type,
                source_url=source_url,
                canonical_name=name,
                designation=role,
                role_group=_role_group_from_title(role),
                email=email,
            ).with_hash()
        )

    for section in soup.select("div.details_1"):
        heading = normalize_whitespace(_child_text(section, "h2"))
        if not heading:
            continue
        leader_name, leader_role = _parse_leader_heading(heading)
        table = section.select_one("table")
        if table is None:
            continue
        name_header = _name_header(table)
        for row in table.select("tbody tr"):
            values = [normalize_whitespace(cell.get_text(" ", strip=True)) for cell in row.select("td")]
            if not values:
                continue
            department_name = values[0]
            person_text = values[1] if len(values) > 1 else None
            contact_text = values[2] if len(values) > 2 else None
            executive_name, designation = _split_person_and_designation(person_text)
            phone = _extract_phone(contact_text)
            email = _extract_email(contact_text)
            mappings.append(
                OrgStructureRecord(
                    source_type=source_type,
                    source_url=source_url,
                    leader_name=leader_name,
                    leader_role=leader_role,
                    department_name=department_name,
                    executive_director_name=executive_name,
                    executive_director_email=email,
                    executive_director_phone=phone,
                ).with_hash()
            )
            if executive_name:
                people.append(
                    DirectoryPersonRecord(
                        source_type=source_type,
                        source_url=source_url,
                        canonical_name=executive_name,
                        designation=designation or name_header,
                        role_group=_role_group_from_title(designation or name_header),
                        department_name=department_name,
                        email=email,
                        phone=phone,
                    ).with_hash()
                )

    return DirectoryPageParseResult(
        people=tuple(_dedupe_people(people)),
        org_structure=tuple(_dedupe_org_rows(mappings)),
    )


def _child_text(node, selector: str) -> str | None:
    child = node.select_one(selector)
    if child is None:
        return None
    return child.get_text(" ", strip=True)


def _parse_leader_heading(value: str) -> tuple[str | None, str | None]:
    if "," not in value:
        return normalize_person_name(value), None
    leader_name, leader_role = value.split(",", 1)
    return normalize_person_name(leader_name), normalize_whitespace(leader_role)


def _name_header(table) -> str | None:
    headers = [normalize_whitespace(th.get_text(" ", strip=True)) for th in table.select("thead th")]
    if len(headers) < 2:
        return None
    return headers[1]


def _split_person_and_designation(value: str | None) -> tuple[str | None, str | None]:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None, None
    if "," not in cleaned:
        return normalize_person_name(cleaned), None
    name, designation = cleaned.split(",", 1)
    return normalize_person_name(name), normalize_whitespace(designation)


def _extract_email(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"([A-Za-z0-9._%+-]+@sebi\.gov\.in)", value, re.IGNORECASE)
    return normalize_email(match.group(1)) if match else None


def _extract_phone(value: str | None) -> str | None:
    if not value:
        return None
    email = _extract_email(value)
    phone_text = value.replace(email or "", "")
    return normalize_phone(phone_text)


def _role_group_from_title(value: str | None) -> str | None:
    cleaned = (value or "").lower()
    if "chair" in cleaned:
        return "chairperson"
    if "whole-time member" in cleaned or "whole time member" in cleaned:
        return "wtm"
    if "executive director" in cleaned:
        return "executive_director"
    if "department / division head" in cleaned or "head" in cleaned:
        return "department_head"
    if "chief vigilance officer" in cleaned:
        return "chief_vigilance_officer"
    return None


def _dedupe_people(records: list[DirectoryPersonRecord]) -> list[DirectoryPersonRecord]:
    seen: dict[str, DirectoryPersonRecord] = {}
    for record in records:
        if record.row_sha256 is None:
            continue
        existing = seen.get(record.row_sha256)
        if existing is None or _person_score(record) > _person_score(existing):
            seen[record.row_sha256] = record
    return list(seen.values())


def _dedupe_org_rows(records: list[OrgStructureRecord]) -> list[OrgStructureRecord]:
    seen: dict[str, OrgStructureRecord] = {}
    for record in records:
        if record.row_sha256 is None:
            continue
        seen[record.row_sha256] = record
    return list(seen.values())


def _person_score(record: DirectoryPersonRecord) -> int:
    return sum(
        1
        for value in (
            record.designation,
            record.role_group,
            record.department_name,
            record.email,
            record.phone,
        )
        if value
    )
