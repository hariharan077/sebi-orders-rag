"""Helpers for structured SEBI office lookup rendering."""

from __future__ import annotations

from .canonical_models import CanonicalOfficeRecord
from .canonicalize_offices import is_generic_city_office_query, match_canonical_offices, normalize_lookup_key
from ..current_info.query_normalization import StructuredCurrentInfoQuery


def match_offices(
    offices: tuple[CanonicalOfficeRecord, ...],
    *,
    query: str,
    intent: StructuredCurrentInfoQuery,
) -> tuple[CanonicalOfficeRecord, ...]:
    """Resolve office matches for one structured office query."""

    return match_canonical_offices(offices, intent.office_hint or query)


def same_city_matches(
    offices: tuple[CanonicalOfficeRecord, ...],
    *,
    city: str | None,
) -> tuple[CanonicalOfficeRecord, ...]:
    """Return offices from the same normalized city."""

    if not city:
        return ()
    normalized_city = normalize_lookup_key(city)
    return tuple(
        office
        for office in offices
        if office.city and normalize_lookup_key(office.city) == normalized_city
    )


def should_render_city_list(
    *,
    query: str,
    intent: StructuredCurrentInfoQuery,
    primary_office: CanonicalOfficeRecord,
    same_city_offices: tuple[CanonicalOfficeRecord, ...],
) -> bool:
    """Return whether a city query should list multiple official offices."""

    return bool(
        len(same_city_offices) > 1
        and (
            (intent.is_follow_up and intent.extracted_city)
            or is_generic_city_office_query(query, primary_office.city)
        )
    )


def render_single_office_answer(
    office: CanonicalOfficeRecord,
    *,
    intent: StructuredCurrentInfoQuery,
) -> str:
    """Render one structured office answer."""

    details = []
    if office.address:
        details.append(f"address: {office.address}")
    if office.phone and (
        intent.wants_phone or not (intent.wants_address and not intent.wants_email and not intent.wants_fax)
    ):
        details.append(f"phone: {office.phone}")
    if office.email and (intent.wants_email or not intent.wants_address):
        details.append(f"email: {office.email}")
    if office.fax and intent.wants_fax:
        details.append(f"fax: {office.fax}")
    detail_text = "; ".join(details) if details else "no contact details were available"
    return f"{office.office_name}: {detail_text}."


def render_office_summary(office: CanonicalOfficeRecord) -> str:
    """Render one office summary entry for multi-office city answers."""

    parts = [office.office_name]
    if office.address:
        parts.append(office.address)
    if office.phone:
        parts.append(f"phone: {office.phone}")
    return "; ".join(parts) + "."
