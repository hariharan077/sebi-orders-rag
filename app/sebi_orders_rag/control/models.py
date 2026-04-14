"""Typed models for the SEBI Orders retrieval hardening control pack."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentIndexRow:
    """One indexed matter from the generated control pack."""

    record_key: str
    exact_title: str
    bucket_category: str
    order_date: date | None
    main_entities: tuple[str, ...]
    short_summary: str
    summary_source: str
    procedural_type: str | None
    manifest_status: str
    manifest_error: str | None
    ingested: bool
    document_version_id: int | None
    detail_url: str | None
    pdf_url: str | None
    local_filename: str


@dataclass(frozen=True)
class ConfusionPair:
    """Known pair of matters that are easy to confuse."""

    record_key_a: str
    title_a: str
    bucket_a: str
    order_date_a: date | None
    record_key_b: str
    title_b: str
    bucket_b: str
    order_date_b: date | None
    confusion_type: str
    reason: str


@dataclass(frozen=True)
class EvalQueryCase:
    """Control-pack query case for end-to-end evaluation."""

    query: str
    expected_route_mode: str
    expected_record_key: str | None
    expected_title: str | None
    comparison_allowed: bool
    notes: str
    reuse_previous_session: bool = False
    session_group: str | None = None


@dataclass(frozen=True)
class WrongAnswerExample:
    """Observed contaminated answer captured as a regression case."""

    user_query: str
    expected_record_key: str | None
    expected_title: str | None
    incorrectly_pulled_record_keys: tuple[str, ...]
    incorrectly_pulled_titles: tuple[str, ...]
    observed_answer_status: str
    observed_confidence: float
    observed_route_mode: str
    tool_output: str
    what_it_should_have_answered: str


@dataclass(frozen=True)
class EntityAliasRow:
    """Alias row mapping entity-name variants to one or more record keys."""

    canonical_name: str
    short_name: str | None
    abbreviations: tuple[str, ...]
    old_name: str | None
    new_name: str | None
    related_record_keys: tuple[str, ...]
    related_titles: tuple[str, ...]


@dataclass(frozen=True)
class StrictAnswerRule:
    """Parsed strict answer rule metadata."""

    text: str
    strict_single_matter_required: bool


@dataclass(frozen=True)
class MatterLockCandidate:
    """One deterministic candidate considered for a strict matter lock."""

    record_key: str
    title: str
    bucket_name: str
    document_version_id: int | None
    canonical_entities: tuple[str, ...]
    score: float
    exact_title_match: bool = False
    record_key_match: bool = False
    matched_aliases: tuple[str, ...] = ()
    matched_entity_terms: tuple[str, ...] = ()
    title_similarity: float = 0.0
    title_overlap_ratio: float = 0.0


@dataclass(frozen=True)
class StrictMatterLock:
    """Deterministic lock decision for a named-matter query."""

    named_matter_query: bool = False
    strict_scope_required: bool = False
    strict_single_matter: bool = False
    ambiguous: bool = False
    comparison_intent: bool = False
    comparison_terms: tuple[str, ...] = ()
    matched_aliases: tuple[str, ...] = ()
    matched_entities: tuple[str, ...] = ()
    matched_titles: tuple[str, ...] = ()
    locked_record_keys: tuple[str, ...] = ()
    candidates: tuple[MatterLockCandidate, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MixedRecordGuardrailResult:
    """Answer-time mixed-record diagnostics for strict named-matter queries."""

    strict_scope_required: bool
    strict_single_matter: bool
    locked_record_keys: tuple[str, ...]
    retrieved_record_keys: tuple[str, ...]
    cited_record_keys: tuple[str, ...]
    dominant_record_key: str | None
    mixed_retrieval: bool
    mixed_citations: bool
    substantive_citation_present: bool
    guardrail_fired: bool
    single_matter_rule_respected: bool
    should_regenerate_locked_matter: bool
    should_abstain: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControlPack:
    """Fully loaded control pack plus convenience indexes."""

    root: Path
    document_index: tuple[DocumentIndexRow, ...]
    confusion_pairs: tuple[ConfusionPair, ...]
    eval_queries: tuple[EvalQueryCase, ...]
    wrong_answer_examples: tuple[WrongAnswerExample, ...]
    entity_aliases: tuple[EntityAliasRow, ...]
    strict_answer_rule: StrictAnswerRule
    documents_by_record_key: dict[str, DocumentIndexRow] = field(default_factory=dict)
    aliases_by_record_key: dict[str, tuple[EntityAliasRow, ...]] = field(default_factory=dict)
    alias_variants: dict[str, tuple[EntityAliasRow, ...]] = field(default_factory=dict)
    confusion_map: dict[str, tuple[ConfusionPair, ...]] = field(default_factory=dict)


def dataclass_asdict(value: Any) -> Any:
    """Convert nested dataclasses to plain structures for debug/eval output."""

    if isinstance(value, tuple):
        return [dataclass_asdict(item) for item in value]
    if isinstance(value, list):
        return [dataclass_asdict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dataclass_asdict(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: dataclass_asdict(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value
