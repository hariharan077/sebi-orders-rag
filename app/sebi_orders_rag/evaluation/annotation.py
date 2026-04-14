"""Dataset annotation and enrichment helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..control import load_control_pack
from ..control.models import ControlPack
from ..eval.triage import load_failure_dump_reference
from .dataset import apply_case_updates, make_case_id
from .schemas import EvaluationCase, GoldNumericFact

EXPECTATION_UPDATE_REASONS = {
    "safer clarify behavior",
    "equivalent acceptable route",
    "better grounded answer than prior expectation",
}


def annotate_cases(
    cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
    *,
    control_pack_root: str | Path | None = None,
    failure_dump_root: str | Path | None = None,
) -> tuple[EvaluationCase, ...]:
    """Enrich cases with control-pack and failure-dump metadata."""

    control_pack = load_control_pack(control_pack_root)
    failure_dump = load_failure_dump_reference(failure_dump_root)
    annotated: list[EvaluationCase] = []
    for case in cases:
        enriched = case
        if control_pack is not None:
            enriched = _annotate_with_control_pack(enriched, control_pack)
        if failure_dump is not None:
            enriched = _annotate_with_failure_dump(enriched, failure_dump.failed_cases_by_query)
        enriched = _annotate_flags(enriched)
        annotated.append(enriched)
    return tuple(annotated)


def merge_case_sources(*sources: list[EvaluationCase] | tuple[EvaluationCase, ...]) -> tuple[EvaluationCase, ...]:
    """Combine multiple case collections."""

    merged: list[EvaluationCase] = []
    for source in sources:
        merged.extend(source)
    return tuple(merged)


def apply_expectation_updates(
    cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
    *,
    updates: dict[str, dict[str, Any]],
) -> tuple[EvaluationCase, ...]:
    """Apply approved expectation updates while preserving an audit trail."""

    normalized_updates: dict[str, dict[str, Any]] = {}
    for case_id, patch in updates.items():
        reason = str(patch.get("annotation_reason") or "").strip()
        if reason not in EXPECTATION_UPDATE_REASONS:
            raise ValueError(
                f"{case_id}: annotation_reason must be one of {sorted(EXPECTATION_UPDATE_REASONS)}"
            )
        comment = str(patch.get("annotation_comment") or "").strip()
        metadata_patch = dict(patch.get("metadata", {}) or {})
        metadata_patch["expectation_update"] = {
            "approved": True,
            "reason": reason,
            "comment": comment or None,
        }
        normalized_patch = dict(patch)
        normalized_patch.pop("annotation_reason", None)
        normalized_patch.pop("annotation_comment", None)
        normalized_patch["metadata"] = metadata_patch
        normalized_updates[case_id] = normalized_patch
    return apply_case_updates(cases, normalized_updates)


def seed_numeric_anchor_cases() -> tuple[EvaluationCase, ...]:
    """Return hand-authored regression anchors for numeric fact evaluation."""

    return (
        EvaluationCase(
            case_id=make_case_id(prefix="manual", query="What was the listing price of DU Digital?"),
            query="What was the listing price of DU Digital?",
            route_family_expected="hierarchical_rag",
            allowed_routes=("hierarchical_rag", "memory_scoped_rag"),
            expected_record_keys=("external:98774",),
            expected_bucket_names=("orders-of-ed-cgm",),
            gold_numeric_facts=(
                GoldNumericFact(
                    fact_type="listing_price",
                    value_numeric=12.0,
                    value_text="Rs.12/share",
                    unit="INR/share",
                    subject="DU Digital",
                ),
            ),
            must_use_metadata=True,
            must_not_use_web=True,
            tags=("numeric", "du_digital", "listing_price"),
            issue_class="gold_fact",
            difficulty="medium",
            notes="Regression anchor for deterministic listing-price answers.",
            prompt_family_expected="metadata-first fact",
            source_case_refs=("manual:du_digital",),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="manual",
                query="How much did DU Digital share price increase?",
            ),
            query="How much did DU Digital share price increase?",
            route_family_expected="hierarchical_rag",
            allowed_routes=("hierarchical_rag", "memory_scoped_rag"),
            expected_record_keys=("external:98774",),
            expected_bucket_names=("orders-of-ed-cgm",),
            gold_numeric_facts=(
                GoldNumericFact(
                    fact_type="percentage_change",
                    value_numeric=1392.5,
                    value_text="1392.5%",
                    unit="percent",
                    subject="DU Digital",
                ),
                GoldNumericFact(
                    fact_type="highest_price",
                    value_numeric=296.05,
                    value_text="Rs.296.05",
                    unit="INR",
                    subject="DU Digital",
                ),
            ),
            must_use_metadata=True,
            must_not_use_web=True,
            tags=("numeric", "du_digital", "percentage_change"),
            issue_class="gold_fact",
            difficulty="medium",
            notes="Regression anchor for price-movement summary answers.",
            prompt_family_expected="metadata-first fact",
            source_case_refs=("manual:du_digital",),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="manual",
                query="How many shares did Mrs. Aruna Dhanuka hold in Mint Investment Limited?",
            ),
            query="How many shares did Mrs. Aruna Dhanuka hold in Mint Investment Limited?",
            route_family_expected="hierarchical_rag",
            allowed_routes=("hierarchical_rag", "memory_scoped_rag"),
            expected_record_keys=(
                "derived:cf3aef80c0c3ee8e4b1628ff189990014631ced544fab8d3990a11546205d3b6",
            ),
            expected_bucket_names=("orders-of-chairperson-members",),
            gold_numeric_facts=(
                GoldNumericFact(
                    fact_type="share_count",
                    value_numeric=565818.0,
                    value_text="5,65,818 shares",
                    unit="shares",
                    subject="Mrs. Aruna Dhanuka",
                ),
                GoldNumericFact(
                    fact_type="holding_percentage",
                    value_numeric=10.21,
                    value_text="10.21%",
                    unit="percent",
                    subject="Mrs. Aruna Dhanuka",
                ),
            ),
            must_use_metadata=True,
            must_not_use_web=True,
            tags=("numeric", "mint_investment", "person_vs_trust"),
            issue_class="gold_fact",
            difficulty="hard",
            notes="Regression anchor for person-vs-trust shareholding phrasing.",
            prompt_family_expected="metadata-first fact",
            expected_failure_buckets=("person-vs-trust confusion",),
            source_case_refs=("manual:mint_investment",),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="manual",
                query="What percentage would Aruna Dhanuka Family Trust hold in Mint Investment Limited?",
            ),
            query="What percentage would Aruna Dhanuka Family Trust hold in Mint Investment Limited?",
            route_family_expected="hierarchical_rag",
            allowed_routes=("hierarchical_rag", "memory_scoped_rag"),
            expected_record_keys=(
                "derived:cf3aef80c0c3ee8e4b1628ff189990014631ced544fab8d3990a11546205d3b6",
            ),
            expected_bucket_names=("orders-of-chairperson-members",),
            gold_numeric_facts=(
                GoldNumericFact(
                    fact_type="holding_percentage",
                    value_numeric=23.93,
                    value_text="23.93%",
                    unit="percent",
                    subject="Aruna Dhanuka Family Trust",
                ),
            ),
            must_use_metadata=True,
            must_not_use_web=True,
            tags=("numeric", "mint_investment", "person_vs_trust"),
            issue_class="gold_fact",
            difficulty="hard",
            notes="Regression anchor for trust-vs-person holdings.",
            prompt_family_expected="metadata-first fact",
            expected_failure_buckets=("person-vs-trust confusion",),
            source_case_refs=("manual:mint_investment",),
        ),
    )


def _annotate_with_control_pack(case: EvaluationCase, control_pack: ControlPack) -> EvaluationCase:
    bucket_names = list(case.expected_bucket_names)
    tags = list(case.tags)
    metadata = dict(case.metadata)
    notes = case.notes
    for record_key in case.expected_record_keys:
        document = control_pack.documents_by_record_key.get(record_key)
        if document is None:
            continue
        if document.bucket_category and document.bucket_category not in bucket_names:
            bucket_names.append(document.bucket_category)
        metadata.setdefault("expected_titles", [])
        metadata["expected_titles"] = list(
            dict.fromkeys([*metadata["expected_titles"], document.exact_title])
        )
    if case.expected_record_keys and "named_matter" not in tags:
        tags.append("named_matter")
    if case.must_abstain and "abstain" not in tags:
        tags.append("abstain")
    source_files = tuple(
        dict.fromkeys((*case.source_files, str(control_pack.root)))
    )
    return replace(
        case,
        expected_bucket_names=tuple(bucket_names),
        tags=tuple(tags),
        metadata=metadata,
        notes=notes,
        source_files=source_files,
    )


def _annotate_with_failure_dump(
    case: EvaluationCase,
    failed_cases_by_query: dict[str, dict[str, Any]],
) -> EvaluationCase:
    reference = failed_cases_by_query.get(case.query)
    if reference is None:
        return case
    metadata = dict(case.metadata)
    metadata["reference_failure_dump"] = {
        "primary_bucket": reference.get("primary_bucket"),
        "equivalent_route_reason": reference.get("equivalent_route_reason"),
    }
    expected_failure_buckets = tuple(
        dict.fromkeys(
            (
                *case.expected_failure_buckets,
                str(reference.get("primary_bucket") or "").strip(),
            )
        )
    )
    tags = tuple(
        dict.fromkeys((*case.tags, "failure_dump_seed"))
    )
    return replace(
        case,
        expected_failure_buckets=tuple(
            item for item in expected_failure_buckets if item
        ),
        tags=tags,
        metadata=metadata,
    )


def _annotate_flags(case: EvaluationCase) -> EvaluationCase:
    tags = set(case.tags)
    query = case.query.lower()
    must_use_metadata = case.must_use_metadata or bool(
        {
            "metadata_signatory_followup",
            "metadata_order_date_followup",
            "numeric",
        }
        & tags
    ) or any(
        phrase in query
        for phrase in (
            "who signed",
            "when was this order passed",
            "settlement amount",
            "listing price",
            "highest price",
            "lowest price",
            "pan",
            "holding",
            "shares",
        )
    )
    must_use_structured_current_info = case.must_use_structured_current_info or bool(
        {"structured_info", "structured_current_info"} & tags
    )
    must_use_official_web = case.must_use_official_web or bool(
        {
            "historical_official_lookup",
            "current_news_lookup",
            "current_official_lookup",
        }
        & tags
    )
    must_not_use_web = case.must_not_use_web or (
        bool(case.expected_record_keys) and not must_use_official_web
    )
    must_abstain = case.must_abstain or case.route_family_expected == "abstain"
    must_clarify = case.must_clarify or case.route_family_expected == "clarify"
    return replace(
        case,
        must_use_metadata=must_use_metadata,
        must_use_structured_current_info=must_use_structured_current_info,
        must_use_official_web=must_use_official_web,
        must_not_use_web=must_not_use_web,
        must_abstain=must_abstain,
        must_clarify=must_clarify,
    )
