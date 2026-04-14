"""Eval-triage helpers driven by the captured SEBI failure dump."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report import EvalCaseResult

TRIAGE_BUCKET_PRIORITY: tuple[str, ...] = (
    "contamination",
    "stale expectation",
    "weak metadata support",
    "wrong candidate ranking",
    "wrong route",
    "wrong answer despite correct route",
    "unsupported query / expected abstain",
)
_NAMED_SINGLE_MATTER_ROUTES = frozenset(
    {"exact_lookup", "hierarchical_rag", "memory_scoped_rag"}
)
_GENERAL_EXPLANATORY_ROUTES = frozenset({"direct_llm", "general_knowledge"})
_CURRENT_INFO_ROUTES = frozenset({"structured_current_info", "current_official_lookup"})
_LATEST_NEWS_ROUTES = frozenset({"current_news_lookup", "current_official_lookup"})


@dataclass(frozen=True)
class FailureDumpReference:
    """Reference failure dump used to keep stale expectations explicit."""

    root: Path
    summary: dict[str, Any]
    failed_cases_by_query: dict[str, dict[str, Any]]
    markdown_report: str


@dataclass(frozen=True)
class TriageDecision:
    """Bucket assignment plus route-equivalence reasoning for one eval result."""

    primary_bucket: str
    bucket_tags: tuple[str, ...]
    stale_expectation: bool
    true_bug: bool
    equivalent_route_passed: bool = False
    equivalent_route_reason: str | None = None
    reference_bucket: str | None = None


def resolve_failure_dump_root(
    root: str | Path | None,
    *,
    search_root: Path | None = None,
) -> Path | None:
    """Resolve the configured failure-dump directory if present."""

    if root is not None:
        candidate = Path(root).expanduser().resolve(strict=False)
        return candidate if candidate.exists() else None
    base_dir = (search_root or Path.cwd()).resolve(strict=False)
    artifacts_dir = base_dir / "artifacts"
    candidates = sorted(
        (
            path
            for path in artifacts_dir.glob("sebi_eval_failure_dump_*")
            if path.is_dir()
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_failure_dump_reference(root: str | Path | None) -> FailureDumpReference | None:
    """Load the prior failure dump so stale expectations stay transparent."""

    if root is None:
        return None
    resolved_root = Path(root).expanduser().resolve(strict=False)
    summary_path = resolved_root / "summary.json"
    failed_cases_path = resolved_root / "failed_cases.json"
    failed_cases_md_path = resolved_root / "failed_cases.md"
    if not (summary_path.exists() and failed_cases_path.exists() and failed_cases_md_path.exists()):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed_cases_payload = json.loads(failed_cases_path.read_text(encoding="utf-8"))
    markdown_report = failed_cases_md_path.read_text(encoding="utf-8")
    failed_cases_by_query = {
        str(item.get("query") or ""): item
        for item in failed_cases_payload.get("failed_cases", [])
        if str(item.get("query") or "")
    }
    return FailureDumpReference(
        root=resolved_root,
        summary=summary,
        failed_cases_by_query=failed_cases_by_query,
        markdown_report=markdown_report,
    )


def triage_eval_result(
    result: EvalCaseResult,
    *,
    reference: FailureDumpReference | None = None,
) -> TriageDecision:
    """Classify one failed or passed result into an eval-triage bucket."""

    reference_case = reference.failed_cases_by_query.get(result.query) if reference else None
    reference_bucket = (
        str(reference_case.get("primary_bucket") or "").strip() or None
        if reference_case is not None
        else None
    )
    equivalent_route_passed, equivalent_route_reason = _equivalent_route_reason(result)

    reason_text = " ".join(result.reasons).lower()
    route_debug = dict(result.debug.get("route_debug", {}) or {})
    metadata_debug = dict(result.debug.get("metadata_debug", {}) or {})
    candidate_list_used = bool(result.debug.get("candidate_list_debug", {}).get("used"))

    contamination = bool(
        not result.single_matter_rule_respected or "contaminated" in reason_text
    )
    weak_metadata_support = bool(
        metadata_debug.get("used")
        or (
            result.expected_record_key
            and (
                route_debug.get("asks_order_signatory")
                or route_debug.get("asks_order_date")
                or route_debug.get("asks_legal_provisions")
                or route_debug.get("asks_order_observations")
            )
            and result.answer_status == "abstained"
        )
    )
    route_only_failure = _route_only_failure(result)
    stale_expectation = bool(
        (reference_bucket == "stale expectation" and (candidate_list_used or equivalent_route_passed))
        or (equivalent_route_passed and route_only_failure)
    )
    unsupported_query = bool(
        result.expected_route_mode == "abstain"
        and result.answer_status in {"abstained", "clarify"}
        and not candidate_list_used
    )
    wrong_candidate_ranking = bool(
        not contamination
        and candidate_list_used
        and bool(result.expected_record_key)
        and not stale_expectation
    )
    wrong_route = bool(
        not contamination
        and bool(result.expected_route_mode)
        and result.actual_route_mode != result.expected_route_mode
        and not equivalent_route_passed
        and not wrong_candidate_ranking
        and not stale_expectation
    )
    wrong_answer = bool(
        not contamination
        and not wrong_route
        and not wrong_candidate_ranking
        and not weak_metadata_support
        and not stale_expectation
        and (
            result.passed is False
            or result.case_kind == "wrong_answer_regression"
        )
    )

    tags: list[str] = []
    if equivalent_route_passed:
        tags.append("equivalent acceptable route")
    if contamination:
        tags.append("contamination")
    if stale_expectation:
        tags.append("stale expectation")
    if weak_metadata_support:
        tags.append("weak metadata support")
    if wrong_candidate_ranking:
        tags.append("wrong candidate ranking")
    if wrong_route:
        tags.append("wrong route")
    if wrong_answer:
        tags.append("wrong answer despite correct route")
    if unsupported_query and not tags:
        tags.append("unsupported query / expected abstain")
    if not tags:
        tags.append("wrong answer despite correct route")

    primary_bucket = next(
        bucket for bucket in TRIAGE_BUCKET_PRIORITY if bucket in tags
    )
    return TriageDecision(
        primary_bucket=primary_bucket,
        bucket_tags=tuple(dict.fromkeys(tags)),
        stale_expectation=stale_expectation,
        true_bug=not stale_expectation,
        equivalent_route_passed=equivalent_route_passed,
        equivalent_route_reason=equivalent_route_reason,
        reference_bucket=reference_bucket,
    )


def triage_bucket_counts(results: tuple[EvalCaseResult, ...]) -> dict[str, int]:
    """Return a compact bucket-count mapping for summary output."""

    counts = Counter(
        str(result.triage_bucket or "unclassified")
        for result in results
        if not result.passed or result.stale_expectation
    )
    return dict(sorted(counts.items()))


def _equivalent_route_reason(result: EvalCaseResult) -> tuple[bool, str | None]:
    if not result.expected_route_mode or result.actual_route_mode == result.expected_route_mode:
        return False, None

    route_debug = dict(result.debug.get("route_debug", {}) or {})
    if (
        result.expected_route_mode in _GENERAL_EXPLANATORY_ROUTES
        and result.actual_route_mode in _GENERAL_EXPLANATORY_ROUTES
        and route_debug.get("appears_general_explanatory")
        and not route_debug.get("appears_matter_specific")
    ):
        return True, "general_explanatory_equivalent"

    if (
        result.expected_route_mode in _NAMED_SINGLE_MATTER_ROUTES
        and result.actual_route_mode in _NAMED_SINGLE_MATTER_ROUTES
        and _named_single_matter_grounded(result)
    ):
        return True, "named_single_matter_equivalent"

    if (
        result.answer_status == "clarify"
        and bool(result.debug.get("candidate_list_debug", {}).get("used"))
        and route_debug.get("strict_scope_required")
        and not route_debug.get("strict_single_matter")
    ):
        return True, "ambiguous_named_matter_clarify"

    if (
        result.expected_route_mode in _CURRENT_INFO_ROUTES
        and result.actual_route_mode in _CURRENT_INFO_ROUTES
        and (
            route_debug.get("appears_structured_current_info")
            or route_debug.get("appears_current_official_lookup")
        )
    ):
        return True, "current_info_equivalent"

    if (
        result.expected_route_mode in _LATEST_NEWS_ROUTES
        and result.actual_route_mode in _LATEST_NEWS_ROUTES
        and route_debug.get("appears_current_news_lookup")
    ):
        return True, "latest_news_equivalent"

    return False, None


def _named_single_matter_grounded(result: EvalCaseResult) -> bool:
    expected_record_key = str(result.expected_record_key or "").strip()
    if not expected_record_key or not result.single_matter_rule_respected:
        return False
    actual_record_keys = tuple(
        dict.fromkeys(
            (*result.actual_cited_record_keys, *result.actual_active_record_keys)
        )
    )
    if not actual_record_keys:
        return False
    return set(actual_record_keys) == {expected_record_key}


def _route_only_failure(result: EvalCaseResult) -> bool:
    if result.passed:
        return False
    normalized_reasons = [
        str(reason).strip().lower()
        for reason in result.reasons
        if str(reason).strip()
    ]
    if not normalized_reasons:
        return False
    return all(
        reason.startswith("expected route ")
        or reason == "strict single-matter lock did not trigger"
        for reason in normalized_reasons
    )
