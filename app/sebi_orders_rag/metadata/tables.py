"""Deterministic table and period-row extraction for SEBI order metadata."""

from __future__ import annotations

import re

from .models import ExtractedPriceMovement, MetadataPageText

_MONTH_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_RE = rf"{_MONTH_RE}\s+\d{{1,2}},\s+\d{{4}}"
_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_RE})\s*[-–]\s*(?P<end>{_DATE_RE})",
    re.IGNORECASE,
)
_ROW_START_RE = re.compile(
    rf"\b(?P<label>\d+)\s+(?P<period>{_DATE_RE}\s*[-–]\s*{_DATE_RE})\b",
    re.IGNORECASE,
)
_TABLE_MARKER_RE = re.compile(
    r"\bS\.?\s*No\.?\b.*?\bPatch/Period\b.*?\bRationale\b.*?\bPrice\b",
    re.IGNORECASE,
)
_PRICE_SENTENCE_RE = re.compile(r"\bThe price\b", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"Rs\.?\s*([0-9][0-9,]*(?:\.\d+)?)", re.IGNORECASE)
_PCT_RE = re.compile(r"([+-]\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_HIGH_PRICE_RE = re.compile(
    r"(?:highest|high)\s+price(?:\s+of)?\s+Rs\.?\s*([0-9][0-9,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_LOW_PRICE_RE = re.compile(
    r"low\s+price(?:\s+of)?\s+Rs\.?\s*([0-9][0-9,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_CLOSED_PRICE_RE = re.compile(
    r"closed\s+at\s+(?:Rs\.?\s*)?([0-9][0-9,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_price_movements(
    *,
    document_version_id: int,
    pages: tuple[MetadataPageText, ...],
) -> tuple[ExtractedPriceMovement, ...]:
    """Extract period-wise price movement rows from table-like order text."""

    rows: list[ExtractedPriceMovement] = []
    seen_row_shas: set[str] = set()
    for page in pages:
        normalized = _normalize_page_text(page.text)
        if not normalized or not _TABLE_MARKER_RE.search(normalized):
            continue
        table_text = normalized[_TABLE_MARKER_RE.search(normalized).start() :]
        page_rows = _extract_price_movement_rows_from_table_text(
            document_version_id=document_version_id,
            page_no=page.page_no,
            table_text=table_text,
        )
        for row in page_rows:
            if row.row_sha256 in seen_row_shas:
                continue
            seen_row_shas.add(row.row_sha256)
            rows.append(row)
    return tuple(rows)


def _extract_price_movement_rows_from_table_text(
    *,
    document_version_id: int,
    page_no: int,
    table_text: str,
) -> tuple[ExtractedPriceMovement, ...]:
    matches = list(_ROW_START_RE.finditer(table_text))
    if not matches:
        return ()

    rows: list[ExtractedPriceMovement] = []
    for index, match in enumerate(matches):
        start_index = match.start()
        end_index = matches[index + 1].start() if index + 1 < len(matches) else len(table_text)
        row_text = table_text[start_index:end_index].strip(" .")
        row = _parse_price_movement_row(
            document_version_id=document_version_id,
            page_no=page_no,
            row_text=row_text,
        )
        if row is not None:
            rows.append(row)
    return tuple(rows)


def _parse_price_movement_row(
    *,
    document_version_id: int,
    page_no: int,
    row_text: str,
) -> ExtractedPriceMovement | None:
    start_match = _ROW_START_RE.match(row_text)
    if start_match is None:
        return None

    label = start_match.group("label")
    period_text = start_match.group("period")
    range_match = _DATE_RANGE_RE.search(period_text)
    if range_match is None:
        return None

    remainder = row_text[start_match.end() :].strip()
    pct_matches = list(_PCT_RE.finditer(remainder))
    pct_match = pct_matches[-1] if pct_matches else None
    pct_change = _parse_float(pct_match.group(1)) if pct_match is not None else None
    body = remainder[: pct_match.start()].strip(" .") if pct_match is not None else remainder
    price_anchor = _PRICE_SENTENCE_RE.search(body)
    rationale = body[: price_anchor.start()].strip(" .") if price_anchor is not None else None

    closed_prices = [_parse_float(match.group(1)) for match in _CLOSED_PRICE_RE.finditer(body)]
    if not closed_prices:
        return None

    high_match = _HIGH_PRICE_RE.search(body)
    low_match = _LOW_PRICE_RE.search(body)
    rationale_text = rationale or None
    if rationale_text is not None:
        rationale_text = re.sub(r"\s+", " ", rationale_text).strip(" .")
    return ExtractedPriceMovement(
        document_version_id=document_version_id,
        period_label=f"Patch {label}",
        period_start_text=_clean_text(range_match.group("start")),
        period_end_text=_clean_text(range_match.group("end")),
        start_price=closed_prices[0],
        high_price=_parse_float(high_match.group(1)) if high_match is not None else None,
        low_price=_parse_float(low_match.group(1)) if low_match is not None else None,
        end_price=closed_prices[-1],
        pct_change=pct_change,
        rationale=rationale_text,
        page_start=page_no,
        page_end=page_no,
    )


def _normalize_page_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    return cleaned or None
