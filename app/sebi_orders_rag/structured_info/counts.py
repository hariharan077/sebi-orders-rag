"""Canonical count helpers for structured SEBI current-info lookups."""

from __future__ import annotations

from .aggregates import designation_group_from_query, designation_group_label
from .canonical_models import CanonicalPersonRecord, StructuredInfoSnapshot


def people_for_designation_group(
    snapshot: StructuredInfoSnapshot,
    *,
    designation_group: str,
) -> tuple[CanonicalPersonRecord, ...]:
    """Return canonical people rows for one designation group."""

    return tuple(
        person
        for person in snapshot.people
        if person.designation_group == designation_group
    )


def canonical_count_for_designation(
    snapshot: StructuredInfoSnapshot,
    *,
    designation_hint: str | None,
) -> tuple[int, str | None, str | None, tuple[CanonicalPersonRecord, ...]]:
    """Resolve one designation query against the canonical count model."""

    designation_group = designation_group_from_query(designation_hint)
    if not designation_group:
        return 0, None, None, ()
    people = people_for_designation_group(snapshot, designation_group=designation_group)
    count = len(people)
    return count, designation_group, designation_group_label(designation_group), people


def canonical_role_count(
    snapshot: StructuredInfoSnapshot,
    *,
    role_key: str,
    fallback_people: tuple[CanonicalPersonRecord, ...],
) -> int:
    """Return the canonical role-count table value with a people-row fallback."""

    stored_count = snapshot.role_count_by_key().get(role_key)
    if stored_count is not None:
        return stored_count
    return len(fallback_people)


def contributing_names(
    people: tuple[CanonicalPersonRecord, ...],
) -> tuple[str, ...]:
    """Return stable contributor names for one canonical designation or role group."""

    return tuple(
        sorted(
            person.canonical_name
            for person in people
        )
    )
