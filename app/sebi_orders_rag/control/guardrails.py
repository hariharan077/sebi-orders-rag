"""Single-matter retrieval and answer guardrails."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from ..constants import SUBSTANTIVE_SECTION_TYPES
from .models import MixedRecordGuardrailResult, StrictMatterLock


def evaluate_mixed_record_guardrail(
    *,
    strict_lock: StrictMatterLock | None,
    retrieved_items: Iterable[object],
    cited_items: Iterable[object] = (),
) -> MixedRecordGuardrailResult:
    """Assess whether a strict single-matter query stayed inside one matter."""

    retrieved_record_keys = _ordered_record_keys(retrieved_items)
    cited_record_keys = _ordered_record_keys(cited_items)
    locked_record_keys = strict_lock.locked_record_keys if strict_lock else ()
    dominant_record_key = _dominant_record_key(cited_record_keys or retrieved_record_keys)
    substantive_citation_present = any(
        getattr(item, "section_type", None) in SUBSTANTIVE_SECTION_TYPES for item in cited_items
    )
    mixed_retrieval = len(retrieved_record_keys) > 1
    mixed_citations = len(cited_record_keys) > 1
    strict_scope_required = bool(strict_lock and strict_lock.strict_scope_required)
    strict_single_matter = bool(strict_lock and strict_lock.strict_single_matter)
    guardrail_fired = strict_single_matter and (mixed_retrieval or mixed_citations)

    reason_codes: list[str] = []
    if strict_scope_required:
        reason_codes.append("strict_scope_required")
    if strict_single_matter:
        reason_codes.append("strict_single_matter")
    if mixed_retrieval:
        reason_codes.append("mixed_retrieval")
    if mixed_citations:
        reason_codes.append("mixed_citations")
    if strict_scope_required and not strict_single_matter:
        reason_codes.append("strict_scope_unresolved")
    if strict_single_matter and not substantive_citation_present and cited_record_keys:
        reason_codes.append("missing_substantive_citation")

    if strict_scope_required and not strict_single_matter:
        return MixedRecordGuardrailResult(
            strict_scope_required=True,
            strict_single_matter=False,
            locked_record_keys=locked_record_keys,
            retrieved_record_keys=retrieved_record_keys,
            cited_record_keys=cited_record_keys,
            dominant_record_key=dominant_record_key,
            mixed_retrieval=mixed_retrieval,
            mixed_citations=mixed_citations,
            substantive_citation_present=substantive_citation_present,
            guardrail_fired=True,
            single_matter_rule_respected=False,
            should_regenerate_locked_matter=False,
            should_abstain=True,
            reason_codes=tuple(reason_codes),
        )

    if not strict_single_matter:
        return MixedRecordGuardrailResult(
            strict_scope_required=strict_scope_required,
            strict_single_matter=False,
            locked_record_keys=locked_record_keys,
            retrieved_record_keys=retrieved_record_keys,
            cited_record_keys=cited_record_keys,
            dominant_record_key=dominant_record_key,
            mixed_retrieval=mixed_retrieval,
            mixed_citations=mixed_citations,
            substantive_citation_present=substantive_citation_present,
            guardrail_fired=False,
            single_matter_rule_respected=True,
            should_regenerate_locked_matter=False,
            should_abstain=False,
            reason_codes=tuple(reason_codes),
        )

    if not locked_record_keys:
        return MixedRecordGuardrailResult(
            strict_scope_required=strict_scope_required,
            strict_single_matter=strict_single_matter,
            locked_record_keys=(),
            retrieved_record_keys=retrieved_record_keys,
            cited_record_keys=cited_record_keys,
            dominant_record_key=dominant_record_key,
            mixed_retrieval=mixed_retrieval,
            mixed_citations=mixed_citations,
            substantive_citation_present=substantive_citation_present,
            guardrail_fired=True,
            single_matter_rule_respected=False,
            should_regenerate_locked_matter=False,
            should_abstain=True,
            reason_codes=tuple(reason_codes + ["missing_locked_record_key"]),
        )

    locked_set = set(locked_record_keys)
    citations_within_lock = not cited_record_keys or set(cited_record_keys).issubset(locked_set)
    retrieval_within_lock = not retrieved_record_keys or set(retrieved_record_keys).issubset(locked_set)
    single_matter_rule_respected = citations_within_lock and retrieval_within_lock
    should_regenerate = not citations_within_lock or not retrieval_within_lock
    should_abstain = should_regenerate and not substantive_citation_present

    return MixedRecordGuardrailResult(
        strict_scope_required=strict_scope_required,
        strict_single_matter=strict_single_matter,
        locked_record_keys=locked_record_keys,
        retrieved_record_keys=retrieved_record_keys,
        cited_record_keys=cited_record_keys,
        dominant_record_key=dominant_record_key,
        mixed_retrieval=mixed_retrieval,
        mixed_citations=mixed_citations,
        substantive_citation_present=substantive_citation_present,
        guardrail_fired=guardrail_fired or should_regenerate,
        single_matter_rule_respected=single_matter_rule_respected,
        should_regenerate_locked_matter=should_regenerate,
        should_abstain=should_abstain,
        reason_codes=tuple(reason_codes),
    )


def filter_items_to_locked_record_keys(
    items: Iterable[object],
    *,
    locked_record_keys: tuple[str, ...],
) -> tuple[object, ...]:
    """Return only items that belong to the locked matter."""

    if not locked_record_keys:
        return tuple(items)
    locked_set = set(locked_record_keys)
    return tuple(item for item in items if getattr(item, "record_key", None) in locked_set)


def _ordered_record_keys(items: Iterable[object]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        record_key = str(getattr(item, "record_key", "") or "").strip()
        if not record_key or record_key in seen:
            continue
        seen.add(record_key)
        ordered.append(record_key)
    return tuple(ordered)


def _dominant_record_key(record_keys: tuple[str, ...]) -> str | None:
    if not record_keys:
        return None
    counts = Counter(record_keys)
    return counts.most_common(1)[0][0]
