"""Time parsing helpers for SEBI Orders RAG."""

from __future__ import annotations

from datetime import date, datetime


def parse_optional_date(value: str | None) -> date | None:
    """Parse an ISO date string if a value is present."""

    if value is None:
        return None
    return date.fromisoformat(value)


def parse_required_datetime(value: str | None, field_name: str) -> datetime:
    """Parse an ISO datetime string and require timezone awareness."""

    if value is None:
        raise ValueError(f"{field_name} is required")

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include timezone information")
    return parsed
