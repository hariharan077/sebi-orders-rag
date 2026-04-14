"""Audit helpers for canonical structured SEBI current-info data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..directory_data.models import DirectoryReferenceDataset, normalize_whitespace
from ..repositories.structured_info import StructuredInfoRepository
from .aggregates import designation_group_from_person
from .canonical_models import StructuredInfoSnapshot
from .canonicalize_offices import canonicalize_offices
from .canonicalize_people import (
    canonicalize_people_with_audit,
    normalized_name_key,
    _source_rows,
)

_KEY_GROUP_LABELS: tuple[tuple[str, str], ...] = (
    ("chairperson", "Chairperson"),
    ("whole_time_member", "WTM"),
    ("executive_director", "ED"),
    ("assistant_manager", "Assistant Manager"),
    ("chief_general_manager", "CGM"),
    ("deputy_general_manager", "DGM"),
    ("general_manager", "GM"),
    ("assistant_general_manager", "AGM"),
    ("regional_director", "Regional Director"),
)


@dataclass(frozen=True)
class StructuredInfoAuditReport:
    """Live parity audit for raw and canonical structured-info layers."""

    raw_people_by_source: dict[str, int]
    raw_people_by_designation_text: dict[str, int]
    raw_designation_groups: dict[str, int]
    raw_distinct_people_by_designation_group: dict[str, int]
    canonical_people_by_designation_group: dict[str, int]
    stored_designation_counts: dict[str, int]
    stored_role_counts: dict[str, int]
    key_group_counts: dict[str, int]
    key_group_names: dict[str, tuple[str, ...]]
    count_parity_lines: tuple[str, ...]
    merge_audit_lines: tuple[str, ...]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]

    def as_lines(self) -> list[str]:
        lines = ["Raw people rows by source"]
        for source_type, count in sorted(self.raw_people_by_source.items()):
            lines.append(f"  {source_type}: {count}")

        lines.append("Raw people rows by designation text")
        for designation, count in sorted(self.raw_people_by_designation_text.items()):
            lines.append(f"  {designation}: {count}")

        lines.append("Raw people rows by designation group")
        for group, count in sorted(self.raw_designation_groups.items()):
            lines.append(f"  {group}: {count}")

        lines.append("Raw distinct people by designation group")
        for group, count in sorted(self.raw_distinct_people_by_designation_group.items()):
            lines.append(f"  {group}: {count}")

        lines.append("Canonical people rows by designation_group")
        for group, count in sorted(self.canonical_people_by_designation_group.items()):
            lines.append(f"  {group}: {count}")

        lines.append("Stored aggregate designation counts")
        for group, count in sorted(self.stored_designation_counts.items()):
            lines.append(f"  {group}: {count}")

        lines.append("Stored aggregate role counts")
        for role_key, count in sorted(self.stored_role_counts.items()):
            lines.append(f"  {role_key}: {count}")

        lines.append("Key counts")
        for label, count in sorted(self.key_group_counts.items()):
            lines.append(f"  {label}: {count}")

        lines.append("Names in key groups")
        for label, names in sorted(self.key_group_names.items()):
            rendered = ", ".join(names) if names else "(none)"
            lines.append(f"  {label}: {rendered}")

        lines.append("Count parity")
        lines.extend(f"  {line}" for line in self.count_parity_lines)

        lines.append("Merge audit")
        lines.extend(f"  {line}" for line in self.merge_audit_lines)

        if self.warnings:
            lines.append("Warnings")
            lines.extend(f"  {warning}" for warning in self.warnings)
        if self.failures:
            lines.append("Failures")
            lines.extend(f"  {failure}" for failure in self.failures)
        return lines


def build_audit_report(repository: StructuredInfoRepository) -> StructuredInfoAuditReport:
    """Return a deterministic live audit report for structured current-info."""

    raw_dataset = repository.load_raw_dataset()
    stored_snapshot = repository.load_snapshot()
    merge_audits = canonicalize_people_with_audit(
        raw_dataset,
        offices=canonicalize_offices(raw_dataset.offices),
    )
    source_rows = _source_rows(raw_dataset)
    raw_people_by_source = dict(
        sorted(Counter(row.source_type for row in source_rows).items())
    )
    raw_people_by_designation_text = dict(
        sorted(Counter(_designation_text(row.designation) for row in source_rows).items())
    )
    raw_designation_groups = dict(
        sorted(_raw_designation_group_counts(merge_audits).items())
    )
    raw_distinct_people_by_designation_group = dict(
        sorted(_raw_distinct_people_by_designation_group(merge_audits).items())
    )
    canonical_people_by_designation_group = dict(
        sorted(Counter(person.designation_group for person in stored_snapshot.people).items())
    )
    stored_designation_counts = dict(sorted(stored_snapshot.designation_count_by_group().items()))
    stored_role_counts = dict(sorted(stored_snapshot.role_count_by_key().items()))
    key_group_counts = _key_group_counts(stored_snapshot)
    key_group_names = _key_group_names(stored_snapshot)
    count_parity_lines = _count_parity_lines(
        snapshot=stored_snapshot,
        raw_distinct_people_by_designation_group=raw_distinct_people_by_designation_group,
        stored_designation_counts=stored_designation_counts,
        stored_role_counts=stored_role_counts,
    )
    warnings, failures = _audit_flags(
        raw_dataset=raw_dataset,
        snapshot=stored_snapshot,
        raw_designation_groups=raw_designation_groups,
        raw_distinct_people_by_designation_group=raw_distinct_people_by_designation_group,
        canonical_people_by_designation_group=canonical_people_by_designation_group,
        stored_designation_counts=stored_designation_counts,
        stored_role_counts=stored_role_counts,
    )
    return StructuredInfoAuditReport(
        raw_people_by_source=raw_people_by_source,
        raw_people_by_designation_text=raw_people_by_designation_text,
        raw_designation_groups=raw_designation_groups,
        raw_distinct_people_by_designation_group=raw_distinct_people_by_designation_group,
        canonical_people_by_designation_group=canonical_people_by_designation_group,
        stored_designation_counts=stored_designation_counts,
        stored_role_counts=stored_role_counts,
        key_group_counts=key_group_counts,
        key_group_names=key_group_names,
        count_parity_lines=count_parity_lines,
        merge_audit_lines=tuple(_merge_audit_lines(merge_audits)),
        warnings=warnings,
        failures=failures,
    )


def _raw_designation_group_counts(merge_audits) -> Counter[str]:
    counter: Counter[str] = Counter()
    for audit in merge_audits:
        counter[audit.person.designation_group] += len(audit.source_rows)
    return counter


def _raw_distinct_people_by_designation_group(merge_audits) -> Counter[str]:
    return Counter(audit.person.designation_group for audit in merge_audits)


def _key_group_counts(snapshot: StructuredInfoSnapshot) -> dict[str, int]:
    counts: dict[str, int] = {
        "board_member_count": sum(
            1
            for person in snapshot.people
            if person.designation_group in {"chairperson", "whole_time_member", "board_member"}
        ),
    }
    for group, label in _KEY_GROUP_LABELS:
        counts[f"{label.lower().replace(' ', '_')}_count"] = sum(
            1
            for person in snapshot.people
            if person.designation_group == group
        )
    return counts


def _key_group_names(snapshot: StructuredInfoSnapshot) -> dict[str, tuple[str, ...]]:
    names: dict[str, tuple[str, ...]] = {}
    for group, label in _KEY_GROUP_LABELS:
        names[label] = tuple(
            sorted(
                person.canonical_name
                for person in snapshot.people
                if person.designation_group == group
            )
        )
    names["Board Members"] = tuple(
        sorted(
            person.canonical_name
            for person in snapshot.people
            if person.designation_group in {"chairperson", "whole_time_member", "board_member"}
        )
    )
    return names


def _merge_audit_lines(audits) -> list[str]:
    lines: list[str] = []
    for audit in audits:
        sources = ", ".join(
            f"{row.source_type}:{row.source_row_key}:{row.canonical_name}"
            for row in audit.source_rows
        )
        reasons = ", ".join(audit.merge_reasons) if audit.merge_reasons else "no_merge"
        lines.append(
            f"{audit.person.canonical_name} [{audit.person.designation_group}] <- {sources} | reasons: {reasons}"
        )
    return lines


def _audit_flags(
    *,
    raw_dataset: DirectoryReferenceDataset,
    snapshot: StructuredInfoSnapshot,
    raw_designation_groups: dict[str, int],
    raw_distinct_people_by_designation_group: dict[str, int],
    canonical_people_by_designation_group: dict[str, int],
    stored_designation_counts: dict[str, int],
    stored_role_counts: dict[str, int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    failures: list[str] = []

    raw_board_count = sum(
        raw_distinct_people_by_designation_group.get(group, 0)
        for group in ("chairperson", "whole_time_member", "board_member")
    )
    canonical_board_count = sum(
        1
        for person in snapshot.people
        if person.designation_group in {"chairperson", "whole_time_member", "board_member"}
    )
    if raw_board_count != canonical_board_count:
        failures.append(
            f"Board-member count mismatch: raw={raw_board_count}, canonical={canonical_board_count}."
        )

    raw_wtm_count = raw_distinct_people_by_designation_group.get("whole_time_member", 0)
    canonical_wtm_count = canonical_people_by_designation_group.get("whole_time_member", 0)
    if raw_wtm_count != canonical_wtm_count:
        failures.append(
            f"WTM count mismatch: raw={raw_wtm_count}, canonical={canonical_wtm_count}."
        )

    raw_ed_count = raw_distinct_people_by_designation_group.get("executive_director", 0)
    canonical_ed_count = canonical_people_by_designation_group.get("executive_director", 0)
    if raw_ed_count != canonical_ed_count:
        failures.append(
            f"ED count mismatch: raw={raw_ed_count}, canonical={canonical_ed_count}."
        )

    for group, raw_count in raw_designation_groups.items():
        if raw_count <= 0:
            continue
        if canonical_people_by_designation_group.get(group, 0) <= 0:
            failures.append(
                f'Designation group "{group}" is present in raw rows but missing in canonical rows.'
            )
    for group, canonical_count in canonical_people_by_designation_group.items():
        raw_distinct_count = raw_distinct_people_by_designation_group.get(group, 0)
        if raw_distinct_count and canonical_count > raw_distinct_count:
            failures.append(
                f'Designation group "{group}" has canonical count {canonical_count} greater than raw distinct people count {raw_distinct_count}.'
            )
        if (
            group in {
                "whole_time_member",
                "executive_director",
                "assistant_manager",
                "assistant_general_manager",
                "deputy_general_manager",
                "general_manager",
                "chief_general_manager",
                "regional_director",
            }
            and raw_distinct_count >= 2
            and canonical_count <= max(0, raw_distinct_count - 2)
        ):
            failures.append(
                f'Designation group "{group}" looks suspiciously collapsed: raw distinct={raw_distinct_count}, canonical={canonical_count}.'
            )

    for group, live_count in canonical_people_by_designation_group.items():
        stored_count = stored_designation_counts.get(group, 0)
        if live_count > 0 and stored_count == 0:
            failures.append(
                f'Designation count table drift for "{group}": stored=0 while canonical people rows={live_count}.'
            )
        elif stored_count not in {0, live_count}:
            warnings.append(
                f'Designation count table drift for "{group}": stored={stored_count}, people_rows={live_count}.'
            )

    live_wtm_count = sum(1 for person in snapshot.people if person.designation_group == "whole_time_member")
    live_ed_count = sum(1 for person in snapshot.people if person.designation_group == "executive_director")
    live_board_count = sum(
        1
        for person in snapshot.people
        if person.designation_group in {"chairperson", "whole_time_member", "board_member"}
    )
    for role_key, live_count in (
        ("whole_time_member", live_wtm_count),
        ("executive_director", live_ed_count),
        ("board_member", live_board_count),
    ):
        stored_count = stored_role_counts.get(role_key, 0)
        if live_count > 0 and stored_count == 0:
            failures.append(
                f'Role count table drift for "{role_key}": stored=0 while canonical people rows={live_count}.'
            )
        elif stored_count not in {0, live_count}:
            warnings.append(
                f'Role count table drift for "{role_key}": stored={stored_count}, people_rows={live_count}.'
            )

    return tuple(dict.fromkeys(warnings)), tuple(dict.fromkeys(failures))


def _count_parity_lines(
    *,
    snapshot: StructuredInfoSnapshot,
    raw_distinct_people_by_designation_group: dict[str, int],
    stored_designation_counts: dict[str, int],
    stored_role_counts: dict[str, int],
) -> tuple[str, ...]:
    lines: list[str] = []
    for group, label in _KEY_GROUP_LABELS:
        people = tuple(
            sorted(
                person.canonical_name
                for person in snapshot.people
                if person.designation_group == group
            )
        )
        lines.append(
            f'{label}: raw_distinct_people={raw_distinct_people_by_designation_group.get(group, 0)} people_rows={len(people)} stored_designation_count={stored_designation_counts.get(group, 0)} contributors={", ".join(people) if people else "(none)"}'
        )

    board_people = tuple(
        sorted(
            person.canonical_name
            for person in snapshot.people
            if person.designation_group in {"chairperson", "whole_time_member", "board_member"}
        )
    )
    lines.append(
        "Board Members: "
        f'people_rows={len(board_people)} stored_role_count={stored_role_counts.get("board_member", 0)} '
        f'contributors={", ".join(board_people) if board_people else "(none)"}'
    )
    return tuple(lines)


def _designation_text(value: str | None) -> str:
    cleaned = normalize_whitespace(value)
    return cleaned or "(missing)"
