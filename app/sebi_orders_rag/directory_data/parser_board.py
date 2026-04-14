"""Structured parser for the official SEBI board-members page."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from .models import (
    BoardMemberRecord,
    DirectoryPageParseResult,
    normalize_person_name,
    normalize_whitespace,
)
from .sources import SOURCE_BOARD_MEMBERS


def parse_board_page(
    raw_html: str,
    *,
    source_url: str,
    source_type: str = SOURCE_BOARD_MEMBERS,
) -> DirectoryPageParseResult:
    """Parse the official board-members page into structured board-member rows."""

    soup = BeautifulSoup(raw_html, "html.parser")
    members: list[BoardMemberRecord] = []

    featured_member = soup.select_one("div.member-first")
    if isinstance(featured_member, Tag):
        parsed = _parse_member_block(
            featured_member,
            source_url=source_url,
            source_type=source_type,
            default_category="chairperson",
        )
        if parsed is not None:
            members.append(parsed)

    for section in soup.select("div.member-list ul"):
        heading = normalize_whitespace(section.find("h2").get_text(" ", strip=True) if section.find("h2") else None)
        default_category = _category_from_section_heading(heading)
        for item in section.find_all("li", recursive=False):
            parsed = _parse_member_block(
                item,
                source_url=source_url,
                source_type=source_type,
                default_category=default_category,
            )
            if parsed is not None:
                members.append(parsed)

    return DirectoryPageParseResult(board_members=tuple(_dedupe_board_members(members)))


def _parse_member_block(
    node: Tag,
    *,
    source_url: str,
    source_type: str,
    default_category: str | None,
) -> BoardMemberRecord | None:
    name = normalize_person_name(_child_text(node, "h3"))
    primary_role = normalize_whitespace(_child_text(node, "h4"))
    secondary_role = normalize_whitespace(_child_text(node, "h5"))
    if not name or not primary_role:
        return None

    board_role = primary_role if not secondary_role else f"{primary_role} ({secondary_role})"
    category = _infer_category(
        primary_role=primary_role,
        secondary_role=secondary_role,
        default_category=default_category,
    )
    return BoardMemberRecord(
        source_type=source_type,
        source_url=source_url,
        canonical_name=name,
        board_role=board_role,
        category=category,
    ).with_hash()


def _child_text(node: Tag, selector: str) -> str | None:
    child = node.select_one(selector)
    if child is None:
        return None
    return child.get_text(" ", strip=True)


def _category_from_section_heading(value: str | None) -> str | None:
    lowered = (value or "").lower()
    if "whole-time member" in lowered or "whole time member" in lowered:
        return "whole_time_member"
    if "part-time member" in lowered or "part time member" in lowered:
        return "part_time_member"
    if "chair" in lowered:
        return "chairperson"
    return None


def _infer_category(
    *,
    primary_role: str,
    secondary_role: str | None,
    default_category: str | None,
) -> str | None:
    primary_lower = primary_role.lower()
    secondary_lower = (secondary_role or "").lower()

    if "chair" in primary_lower:
        return "chairperson"
    if "whole-time member" in primary_lower or "whole time member" in primary_lower:
        return "whole_time_member"
    if "part-time member" in primary_lower or "part time member" in primary_lower:
        if "reserve bank of india" in secondary_lower:
            return "rbi_nominee"
        if "government of india" in secondary_lower or "ministry" in secondary_lower:
            return "government_nominee"
        return "part_time_member"
    return default_category


def _dedupe_board_members(records: list[BoardMemberRecord]) -> list[BoardMemberRecord]:
    seen: dict[str, BoardMemberRecord] = {}
    for record in records:
        if record.row_sha256:
            seen[record.row_sha256] = record
    return list(seen.values())
