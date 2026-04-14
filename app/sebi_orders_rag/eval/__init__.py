"""Eval helpers and packaged sample cases for Phase 4."""
"""Evaluation helpers for SEBI Orders RAG."""

from .evaluator import ControlPackEvaluator, summary_as_dict
from .report import EvalCaseResult, EvalSummary, render_summary
from .triage import (
    FailureDumpReference,
    TriageDecision,
    load_failure_dump_reference,
    resolve_failure_dump_root,
    triage_eval_result,
)

__all__ = [
    "ControlPackEvaluator",
    "EvalCaseResult",
    "EvalSummary",
    "FailureDumpReference",
    "TriageDecision",
    "load_failure_dump_reference",
    "render_summary",
    "resolve_failure_dump_root",
    "summary_as_dict",
    "triage_eval_result",
]
