"""Control-pack helpers for SEBI Orders retrieval hardening."""

from .candidate_selection import (
    SAT_COURT_BUCKETS,
    ExactLookupResolution,
    build_matter_clarification_candidates,
    build_person_clarification_candidates,
    looks_like_sat_court_query,
    render_clarification_candidate_lines,
    select_exact_lookup_resolution,
    sort_exact_lookup_candidates,
)
from .exact_match import confusion_penalty_map, detect_comparison_terms, resolve_strict_matter_lock
from .guardrails import evaluate_mixed_record_guardrail, filter_items_to_locked_record_keys
from .loader import load_control_pack
from .models import (
    ConfusionPair,
    ControlPack,
    DocumentIndexRow,
    EntityAliasRow,
    EvalQueryCase,
    MatterLockCandidate,
    MixedRecordGuardrailResult,
    StrictAnswerRule,
    StrictMatterLock,
    WrongAnswerExample,
    dataclass_asdict,
)

__all__ = [
    "ConfusionPair",
    "ControlPack",
    "DocumentIndexRow",
    "EntityAliasRow",
    "ExactLookupResolution",
    "EvalQueryCase",
    "MatterLockCandidate",
    "MixedRecordGuardrailResult",
    "SAT_COURT_BUCKETS",
    "StrictAnswerRule",
    "StrictMatterLock",
    "WrongAnswerExample",
    "build_matter_clarification_candidates",
    "build_person_clarification_candidates",
    "confusion_penalty_map",
    "dataclass_asdict",
    "detect_comparison_terms",
    "evaluate_mixed_record_guardrail",
    "filter_items_to_locked_record_keys",
    "load_control_pack",
    "looks_like_sat_court_query",
    "render_clarification_candidate_lines",
    "resolve_strict_matter_lock",
    "select_exact_lookup_resolution",
    "sort_exact_lookup_candidates",
]
