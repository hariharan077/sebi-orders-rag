"""Deterministic extraction of signatory and footer metadata from orders."""

from __future__ import annotations

import re
from datetime import date, datetime

from .models import ExtractedOrderMetadata, MetadataPageText

_WHITESPACE_RE = re.compile(r"\s+")
_SD_RE = re.compile(r"^(?:sd/?-?|digitally signed by|signed)$", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\bdate\s*[:.-]?\s*(?P<value>[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
    re.IGNORECASE,
)
_PLACE_RE = re.compile(r"^\s*place\s*[:.-]?\s*(?P<value>[A-Za-z .'-]{2,40})\s*$", re.IGNORECASE)
_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z.'-]*")
_TITLE_PREFIX_RE = re.compile(r"^(?:mr|mrs|ms|dr|shri|smt)\.?\s+", re.IGNORECASE)
_DESIGNATION_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("whole time member", re.compile(r"\bwhole[- ]time member\b|\bwtm\b", re.IGNORECASE), "board_member"),
    ("chairperson", re.compile(r"\bchair(?:man|person)\b", re.IGNORECASE), "board_member"),
    ("adjudicating officer", re.compile(r"\badjudicating officer\b", re.IGNORECASE), "adjudicating_officer"),
    ("executive director", re.compile(r"\bexecutive director\b|\bed\b", re.IGNORECASE), "executive_officer"),
    ("chief general manager", re.compile(r"\bchief general manager\b|\bcgm\b", re.IGNORECASE), "departmental_officer"),
    ("deputy general manager", re.compile(r"\bdeputy general manager\b|\bdgm\b", re.IGNORECASE), "departmental_officer"),
    ("regional director", re.compile(r"\bregional director\b|\brd\b", re.IGNORECASE), "regional_office"),
)


def extract_signatory_metadata(
    *,
    document_version_id: int,
    pages: tuple[MetadataPageText, ...],
    fallback_order_date: date | None = None,
) -> ExtractedOrderMetadata:
    """Extract signatory, date, place, and issuing-authority fields."""

    if not pages:
        return ExtractedOrderMetadata(document_version_id=document_version_id, order_date=fallback_order_date)

    tail_pages = pages[-4:]
    footer_candidates: list[tuple[int, str | None, str, str]] = []
    place_value: str | None = None
    order_date = fallback_order_date

    for page in tail_pages:
        lines = _normalized_lines(page.text)
        if order_date is None:
            order_date = _extract_order_date(lines)
        if place_value is None:
            place_value = _extract_place(lines)
        for index, line in enumerate(lines):
            designation, authority_type = _extract_designation(line)
            if designation is None:
                continue
            signatory_name = _extract_name(lines, index)
            footer_candidates.append((page.page_no, signatory_name, designation, authority_type))

    deduped = _dedupe_footer_candidates(footer_candidates)
    if not deduped:
        return ExtractedOrderMetadata(
            document_version_id=document_version_id,
            order_date=order_date,
            place=place_value,
            metadata_confidence=0.25 if order_date or place_value else 0.0,
        )

    primary_page, primary_name, primary_designation, authority_type = deduped[-1]
    authority_panel = tuple(
        f"{name}, {designation}" if name else designation
        for _, name, designation, _ in deduped
    )
    confidence = 0.42
    if primary_name:
        confidence += 0.32
    if primary_designation:
        confidence += 0.16
    if order_date:
        confidence += 0.06
    if place_value:
        confidence += 0.04
    return ExtractedOrderMetadata(
        document_version_id=document_version_id,
        signatory_name=primary_name,
        signatory_designation=primary_designation.title(),
        signatory_page_start=primary_page,
        signatory_page_end=primary_page,
        order_date=order_date,
        place=place_value,
        issuing_authority_type=authority_type,
        authority_panel=authority_panel,
        metadata_confidence=min(confidence, 0.98),
    )


def _normalized_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip(" \t|")
        if line:
            lines.append(line)
    return lines


def _extract_order_date(lines: list[str]) -> date | None:
    for line in lines:
        match = _DATE_RE.search(line)
        if match is None:
            continue
        parsed = _parse_date(match.group("value"))
        if parsed is not None:
            return parsed
    return None


def _extract_place(lines: list[str]) -> str | None:
    for line in lines:
        match = _PLACE_RE.search(line)
        if match is None:
            continue
        value = match.group("value").strip(" .,-")
        return value.title() if value.isupper() else value
    return None


def _extract_designation(line: str) -> tuple[str | None, str | None]:
    for designation, pattern, authority_type in _DESIGNATION_PATTERNS:
        if pattern.search(line):
            return designation, authority_type
    return None, None


def _extract_name(lines: list[str], designation_index: int) -> str | None:
    for offset in (1, 2, 3):
        candidate_index = designation_index - offset
        if candidate_index < 0:
            break
        candidate = lines[candidate_index].strip(" ,-")
        if _SD_RE.match(candidate) or candidate.lower().startswith(("date", "place")):
            continue
        if not _looks_like_name(candidate):
            continue
        cleaned = _TITLE_PREFIX_RE.sub("", candidate).strip(" ,-")
        return cleaned.title() if cleaned.isupper() else cleaned
    return None


def _looks_like_name(value: str) -> bool:
    tokens = _NAME_TOKEN_RE.findall(value)
    if len(tokens) < 2:
        return False
    lowered = value.lower()
    return not any(term in lowered for term in ("date", "place", "member", "director", "manager", "officer"))


def _dedupe_footer_candidates(
    candidates: list[tuple[int, str | None, str, str]],
) -> list[tuple[int, str | None, str, str]]:
    ordered: list[tuple[int, str | None, str, str]] = []
    seen: set[tuple[int, str | None, str]] = set()
    for page_no, name, designation, authority_type in sorted(candidates, key=lambda item: (item[0], item[2], item[1] or "")):
        key = (page_no, name, designation)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((page_no, name, designation, authority_type))
    return ordered


def _parse_date(value: str) -> date | None:
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%y", "%d-%m-%y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None
