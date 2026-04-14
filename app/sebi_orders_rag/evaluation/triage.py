"""True-bug queue generation for persisted SEBI eval artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..control import load_control_pack
from ..eval.triage import load_failure_dump_reference, resolve_failure_dump_root

_NAMED_SINGLE_MATTER_ROUTES = frozenset(
    {"exact_lookup", "hierarchical_rag", "memory_scoped_rag"}
)
_CORPUS_ROUTES = frozenset(
    {"exact_lookup", "hierarchical_rag", "memory_scoped_rag", "corpus_metadata"}
)
_CURRENT_FACT_ROUTES = frozenset(
    {
        "structured_current_info",
        "current_official_lookup",
        "historical_official_lookup",
        "current_news_lookup",
        "general_knowledge",
    }
)
_TRUE_BUG_CLUSTER_BY_BUCKET: dict[str, str] = {
    "weak metadata support": "metadata coverage bugs",
    "wrong answer despite correct route": "answer shaping/phrasing bugs",
    "wrong route": "routing/planner bugs",
    "contamination": "follow-up memory bugs",
    "wrong candidate ranking": "candidate ranking bugs",
    "missing clarify": "routing/planner bugs",
    "over-abstain": "routing/planner bugs",
    "current-fact-vs-order confusion": "company-role/current-fact bugs",
    "company-role misroute": "company-role/current-fact bugs",
    "active-matter follow-up failure": "follow-up memory bugs",
    "numeric fact extraction miss": "numeric fact bugs",
    "person-vs-trust confusion": "numeric fact bugs",
    "matter-type phrasing issue": "answer shaping/phrasing bugs",
}
_CLUSTER_TOUCHPOINTS: dict[str, tuple[str, ...]] = {
    "routing/planner bugs": (
        "app/sebi_orders_rag/router/planner.py",
        "app/sebi_orders_rag/router/query_analyzer.py",
        "app/sebi_orders_rag/router/decision.py",
        "app/sebi_orders_rag/control/exact_match.py",
    ),
    "metadata coverage bugs": (
        "app/sebi_orders_rag/answering/answer_service.py",
        "app/sebi_orders_rag/metadata/service.py",
        "app/sebi_orders_rag/metadata/numeric_facts.py",
    ),
    "answer shaping/phrasing bugs": (
        "app/sebi_orders_rag/answering/style.py",
        "app/sebi_orders_rag/answering/answer_service.py",
    ),
    "candidate ranking bugs": (
        "app/sebi_orders_rag/control/exact_match.py",
        "app/sebi_orders_rag/control/candidate_selection.py",
        "app/sebi_orders_rag/answering/answer_service.py",
    ),
    "follow-up memory bugs": (
        "app/sebi_orders_rag/answering/answer_service.py",
        "app/sebi_orders_rag/control/guardrails.py",
    ),
    "numeric fact bugs": (
        "app/sebi_orders_rag/metadata/service.py",
        "app/sebi_orders_rag/metadata/numeric_facts.py",
        "app/sebi_orders_rag/answering/answer_service.py",
    ),
    "company-role/current-fact bugs": (
        "app/sebi_orders_rag/router/planner.py",
        "app/sebi_orders_rag/router/query_analyzer.py",
        "app/sebi_orders_rag/current_info/company_facts.py",
        "app/sebi_orders_rag/current_info/official_lookup.py",
    ),
}
_CLUSTER_TESTS: dict[str, tuple[str, ...]] = {
    "routing/planner bugs": (
        "tests/sebi_orders_rag/test_true_bug_regressions.py::TrueBugRegressionTests::test_jp_morgan_settlement_amount_query_locks_to_one_matter",
        "tests/sebi_orders_rag/test_true_bug_regressions.py::TrueBugRegressionTests::test_imaginary_capital_settlement_amount_abstains_without_candidate_list",
    ),
    "metadata coverage bugs": (
        "tests/sebi_orders_rag/test_true_bug_regressions.py::TrueBugRegressionTests::test_metadata_document_ids_prefer_locked_record_key",
    ),
    "answer shaping/phrasing bugs": (
        "tests/sebi_orders_rag/test_settlement_answer_shaping.py",
    ),
    "candidate ranking bugs": (
        "tests/sebi_orders_rag/test_candidate_ranking_regressions.py",
        "tests/sebi_orders_rag/test_true_bug_regressions.py::TrueBugRegressionTests::test_jp_morgan_settlement_amount_query_locks_to_one_matter",
    ),
    "follow-up memory bugs": (
        "tests/sebi_orders_rag/test_active_matter_followups.py",
        "tests/sebi_orders_rag/test_active_matter_contamination.py",
    ),
    "numeric fact bugs": (
        "tests/sebi_orders_rag/test_numeric_fact_extraction.py",
        "tests/sebi_orders_rag/test_true_bug_regressions.py::TrueBugRegressionTests::test_metadata_document_ids_prefer_locked_record_key",
    ),
    "company-role/current-fact bugs": (
        "tests/sebi_orders_rag/test_company_role_query_routing.py",
    ),
}
_STALE_EXPECTATION_REASONS = {
    "safer clarify behavior",
    "equivalent acceptable route",
    "better grounded answer than prior expectation",
}
_NEGATIVE_GROUNDED_ANSWER_RE = re.compile(
    r"\bdoes not (?:describe|state|mention)\b",
    re.IGNORECASE,
)
_QUERY_PATTERN_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bjp morgan chase bank n\.?a\.?\b", re.IGNORECASE), "<party>"),
    (re.compile(r"\bmangalam global enterprise limited\b", re.IGNORECASE), "<party>"),
    (re.compile(r"\bimaginary capital limited\b", re.IGNORECASE), "<party>"),
    (re.compile(r"\bdu digital\b", re.IGNORECASE), "<issuer>"),
    (re.compile(r"\baruna dhanuka family trust\b", re.IGNORECASE), "<subject>"),
    (re.compile(r"\bmrs\.?\s+aruna dhanuka\b", re.IGNORECASE), "<subject>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<date>"),
)
_GENERIC_EXPECTED_TITLE_TOKENS = frozenset(
    {
        "act",
        "adjudication",
        "appeal",
        "appellate",
        "authority",
        "bank",
        "case",
        "cc",
        "court",
        "dated",
        "delhi",
        "did",
        "enterprise",
        "exemption",
        "filed",
        "finally",
        "for",
        "granted",
        "hold",
        "imposed",
        "in",
        "judgment",
        "limited",
        "matter",
        "no",
        "of",
        "the",
        "na",
        "order",
        "orders",
        "proceedings",
        "question",
        "relief",
        "rti",
        "sat",
        "sebi",
        "sentence",
        "settlement",
        "special",
        "terms",
        "under",
        "what",
        "was",
    }
)


@dataclass(frozen=True)
class BugQueueEntry:
    """One machine-readable triage row for a failed eval case."""

    case_id: str
    query: str
    primary_bucket: str
    bucket_tags: tuple[str, ...]
    true_bug: bool
    stale_expectation: bool
    cluster: str
    route_expected: str | None
    route_actual: str | None
    answer_status: str | None
    expected_record_keys: tuple[str, ...] = ()
    actual_record_keys: tuple[str, ...] = ()
    expected_numeric_fact_types: tuple[str, ...] = ()
    missing_numeric_fact_types: tuple[str, ...] = ()
    mismatched_numeric_fact_types: tuple[str, ...] = ()
    contamination_prohibited: bool = False
    candidate_list_used: bool = False
    suggested_touchpoints: tuple[str, ...] = ()
    expected_regression_tests: tuple[str, ...] = ()
    stale_expectation_annotation_reason: str | None = None
    rationale: tuple[str, ...] = ()
    source_refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BugCluster:
    """One grouped implementation bucket for true bugs only."""

    cluster: str
    affected_case_ids: tuple[str, ...]
    affected_query_patterns: tuple[str, ...]
    suggested_code_touchpoints: tuple[str, ...]
    expected_regression_tests: tuple[str, ...]
    bucket_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_latest_run_dir(runs_root: str | Path) -> Path:
    """Return the latest persisted evaluation run directory."""

    root = Path(runs_root).expanduser().resolve(strict=False)
    candidates = sorted(path for path in root.glob("sebi_eval_run_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return candidates[-1]


def load_run_artifact(run_dir: str | Path) -> dict[str, Any]:
    """Load a persisted run artifact directory into memory."""

    root = Path(run_dir).expanduser().resolve(strict=False)
    return {
        "run_dir": root,
        "summary": _load_json(root / "summary.json"),
        "per_case_results": _load_jsonl(root / "per_case_results.jsonl"),
        "failures": _load_jsonl(root / "failures.jsonl"),
        "failures_md": (root / "failures.md").read_text(encoding="utf-8")
        if (root / "failures.md").exists()
        else "",
    }


def build_true_bug_queue(
    *,
    run_dir: str | Path,
    failure_dump_root: str | Path | None = None,
    control_pack_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a machine-readable bug queue from one persisted run."""

    artifact = load_run_artifact(run_dir)
    failure_dump = load_failure_dump_reference(
        resolve_failure_dump_root(
            failure_dump_root,
            search_root=artifact["run_dir"].parents[2],
        )
    )
    control_pack = load_control_pack(control_pack_root)
    failures = artifact["failures"]
    entries = tuple(
        _classify_failure(
            result,
            failure_dump=failure_dump.failed_cases_by_query if failure_dump else {},
            control_pack_present=control_pack is not None,
        )
        for result in failures
    )
    clusters = build_bug_clusters(entries)
    counts = Counter(entry.primary_bucket for entry in entries)
    stale_count = sum(1 for entry in entries if entry.stale_expectation)
    true_bug_count = sum(1 for entry in entries if entry.true_bug)
    return {
        "run_dir": str(artifact["run_dir"]),
        "summary": {
            "failed_case_count": len(entries),
            "stale_expectation_count": stale_count,
            "true_bug_count": true_bug_count,
            "bucket_counts": dict(sorted(counts.items())),
            "cluster_counts": dict(
                sorted(Counter(entry.cluster for entry in entries if entry.true_bug).items())
            ),
        },
        "entries": [entry.to_dict() for entry in entries],
        "clusters": [cluster.to_dict() for cluster in clusters],
        "source_refs": {
            "summary_json": str(artifact["run_dir"] / "summary.json"),
            "per_case_results_jsonl": str(artifact["run_dir"] / "per_case_results.jsonl"),
            "failures_jsonl": str(artifact["run_dir"] / "failures.jsonl"),
            "failures_md": str(artifact["run_dir"] / "failures.md"),
            **(
                {"failure_dump_root": str(failure_dump.root)}
                if failure_dump is not None
                else {}
            ),
        },
    }


def build_bug_clusters(entries: tuple[BugQueueEntry, ...] | list[BugQueueEntry]) -> tuple[BugCluster, ...]:
    """Group true bugs into fixable implementation buckets."""

    grouped: dict[str, list[BugQueueEntry]] = {}
    for entry in entries:
        if not entry.true_bug:
            continue
        grouped.setdefault(entry.cluster, []).append(entry)

    clusters: list[BugCluster] = []
    for cluster_name, cluster_entries in sorted(grouped.items()):
        bucket_counts = Counter(entry.primary_bucket for entry in cluster_entries)
        clusters.append(
            BugCluster(
                cluster=cluster_name,
                affected_case_ids=tuple(entry.case_id for entry in cluster_entries),
                affected_query_patterns=tuple(
                    dict.fromkeys(_normalize_query_pattern(entry.query) for entry in cluster_entries)
                ),
                suggested_code_touchpoints=_CLUSTER_TOUCHPOINTS.get(cluster_name, ()),
                expected_regression_tests=tuple(
                    dict.fromkeys(
                        test_name
                        for entry in cluster_entries
                        for test_name in entry.expected_regression_tests
                    )
                )
                or _CLUSTER_TESTS.get(cluster_name, ()),
                bucket_counts=dict(sorted(bucket_counts.items())),
            )
        )
    return tuple(clusters)


def render_true_bug_queue_markdown(payload: dict[str, Any]) -> str:
    """Render a human-readable bug queue report."""

    summary = payload["summary"]
    lines = [
        "# True Bug Queue",
        "",
        f"- Failed cases: `{summary['failed_case_count']}`",
        f"- True bugs: `{summary['true_bug_count']}`",
        f"- Stale expectations: `{summary['stale_expectation_count']}`",
        "",
        "## Bucket Counts",
        "",
    ]
    bucket_counts = summary.get("bucket_counts", {})
    if bucket_counts:
        for bucket, count in sorted(bucket_counts.items()):
            lines.append(f"- {bucket}: `{count}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Entries", ""])
    entries = payload.get("entries", [])
    if not entries:
        lines.append("No failed cases.")
    for index, entry in enumerate(entries, start=1):
        lines.append(f"### {index}. {entry['query']}")
        lines.append("")
        lines.append(f"- Case ID: `{entry['case_id']}`")
        lines.append(f"- Primary bucket: `{entry['primary_bucket']}`")
        lines.append(f"- Cluster: `{entry['cluster']}`")
        lines.append(f"- True bug: `{entry['true_bug']}`")
        lines.append(f"- Stale expectation: `{entry['stale_expectation']}`")
        lines.append(f"- Expected route: `{entry['route_expected']}`")
        lines.append(f"- Actual route: `{entry['route_actual']}`")
        lines.append(f"- Answer status: `{entry['answer_status']}`")
        if entry["expected_record_keys"]:
            lines.append(
                f"- Expected record keys: {', '.join(entry['expected_record_keys'])}"
            )
        if entry["actual_record_keys"]:
            lines.append(
                f"- Actual record keys: {', '.join(entry['actual_record_keys'])}"
            )
        if entry["expected_numeric_fact_types"]:
            lines.append(
                "- Expected numeric facts: "
                + ", ".join(entry["expected_numeric_fact_types"])
            )
        if entry["missing_numeric_fact_types"] or entry["mismatched_numeric_fact_types"]:
            lines.append(
                "- Numeric misses: "
                + ", ".join(
                    [
                        *entry["missing_numeric_fact_types"],
                        *entry["mismatched_numeric_fact_types"],
                    ]
                )
            )
        if entry["stale_expectation_annotation_reason"]:
            lines.append(
                "- Annotation reason: "
                + entry["stale_expectation_annotation_reason"]
            )
        if entry["rationale"]:
            lines.append("- Rationale: " + "; ".join(entry["rationale"]))
        if entry["expected_regression_tests"]:
            lines.append(
                "- Regression tests: " + ", ".join(entry["expected_regression_tests"])
            )
        lines.append("")

    clusters = payload.get("clusters", [])
    lines.extend(["## Clusters", ""])
    if not clusters:
        lines.append("No true bug clusters.")
    for cluster in clusters:
        lines.append(f"### {cluster['cluster']}")
        lines.append("")
        lines.append(f"- Cases: {', '.join(cluster['affected_case_ids'])}")
        lines.append(
            "- Query patterns: " + ", ".join(cluster["affected_query_patterns"])
        )
        lines.append(
            "- Touchpoints: " + ", ".join(cluster["suggested_code_touchpoints"])
        )
        lines.append(
            "- Regression tests: " + ", ".join(cluster["expected_regression_tests"])
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_compare_summary_markdown(payload: dict[str, Any]) -> str:
    """Render a markdown compare summary."""

    metric_deltas = payload.get("metric_deltas", {})
    lines = [
        "# Compare Summary",
        "",
        f"- Base run: `{payload.get('base_run_dir')}`",
        f"- Head run: `{payload.get('head_run_dir')}`",
        "",
        "## Metric Deltas",
        "",
    ]
    for key, value in sorted(metric_deltas.items()):
        lines.append(f"- {key}: `{value:+.4f}`" if isinstance(value, float) else f"- {key}: `{value}`")
    route_deltas = payload.get("route_deltas", [])
    if route_deltas:
        lines.extend(["", "## Route Deltas", ""])
        for item in route_deltas:
            route_name = item.get("route_family_expected")
            lines.append(
                f"- {route_name}: pass_rate_delta={float(item.get('pass_rate_delta', 0.0)):+.4f}, "
                f"equivalent_route_accuracy_delta={float(item.get('equivalent_route_accuracy_delta', 0.0)):+.4f}"
            )
    return "\n".join(lines).rstrip() + "\n"


def _classify_failure(
    result: dict[str, Any],
    *,
    failure_dump: dict[str, dict[str, Any]],
    control_pack_present: bool,
) -> BugQueueEntry:
    case = result.get("case", {})
    execution = result.get("execution", {})
    route = result.get("route", {})
    retrieval = result.get("retrieval", {})
    grounding = result.get("grounding", {})
    numeric = result.get("numeric", {})
    failure_modes = result.get("failure_modes", {})
    debug = dict(execution.get("debug", {}) or {})
    route_debug = dict(debug.get("route_debug", {}) or {})
    candidate_list_debug = dict(debug.get("candidate_list_debug", {}) or {})

    bucket_tags = set(failure_modes.get("buckets", []) or ())
    answer_status = str(execution.get("answer_status") or "")
    expected_record_keys = tuple(case.get("expected_record_keys", []) or ())
    actual_record_keys = tuple(execution.get("active_record_keys", []) or ())
    primary_bucket = str(failure_modes.get("primary_bucket") or "wrong answer despite correct route")
    rationale: list[str] = []

    if _looks_like_stale_expectation(
        case=case,
        execution=execution,
        route=route,
        retrieval=retrieval,
        grounding=grounding,
        failure_modes=failure_modes,
    ):
        primary_bucket = "stale expectation"
        bucket_tags = {"stale expectation"}
        rationale.append("route mismatch is acceptable because the answer stayed grounded inside one matter")
    elif _is_company_role_misroute(case=case, execution=execution, route_debug=route_debug):
        primary_bucket = "company-role misroute"
        bucket_tags.add("company-role misroute")
        rationale.append("company-role current fact drifted onto an internal corpus route")
    elif _is_current_fact_order_confusion(case=case, execution=execution, route_debug=route_debug):
        primary_bucket = "current-fact-vs-order confusion"
        bucket_tags.add("current-fact-vs-order confusion")
        rationale.append("current/public fact query was handled as an order-matter lookup")
    elif _is_active_matter_follow_up_failure(case=case, route_debug=route_debug):
        primary_bucket = "active-matter follow-up failure"
        bucket_tags.add("active-matter follow-up failure")
        rationale.append("follow-up query did not stay aligned with the active matter contract")
    elif _is_numeric_fact_bug(case=case, numeric=numeric):
        primary_bucket = "numeric fact extraction miss"
        bucket_tags.add("numeric fact extraction miss")
        rationale.append("expected deterministic numeric facts were missing or mismatched")
    elif _is_person_vs_trust_bug(case=case, numeric=numeric):
        primary_bucket = "person-vs-trust confusion"
        bucket_tags.add("person-vs-trust confusion")
        rationale.append("shareholding answer blurred the person and trust subjects")
    elif _is_matter_type_phrasing_bug(case=case, execution=execution, failure_modes=failure_modes):
        primary_bucket = "matter-type phrasing issue"
        bucket_tags.add("matter-type phrasing issue")
        rationale.append("answer phrasing overstated the matter type or finality")
    elif _is_wrong_candidate_ranking(execution=execution, case=case):
        primary_bucket = "wrong candidate ranking"
        bucket_tags.add("wrong candidate ranking")
        rationale.append("the assistant clarified instead of resolving or abstaining cleanly")
    elif _is_over_abstain(case=case, answer_status=answer_status):
        primary_bucket = "over-abstain"
        bucket_tags.add("over-abstain")
        rationale.append("the assistant clarified or abstained even though the case expected an answer")
    elif _is_missing_clarify(case=case, answer_status=answer_status):
        primary_bucket = "missing clarify"
        bucket_tags.add("missing clarify")
        rationale.append("the assistant answered when ambiguity required a clarification step")
    elif _is_contamination(
        case=case,
        retrieval=retrieval,
        failure_modes=failure_modes,
        actual_record_keys=actual_record_keys,
    ):
        primary_bucket = "contamination"
        bucket_tags.add("contamination")
        rationale.append("retrieval or citations crossed multiple matters")
    elif _is_weak_metadata_support(case=case, execution=execution):
        primary_bucket = "weak metadata support"
        bucket_tags.add("weak metadata support")
        rationale.append("query required metadata-first behavior but metadata answering did not fire")
    elif _is_wrong_answer(case=case, route=route, grounding=grounding):
        primary_bucket = "wrong answer despite correct route"
        bucket_tags.add("wrong answer despite correct route")
        rationale.append("route was acceptable but the answer content was still wrong")
    elif not route.get("strict_route_match", False) and not route.get("equivalent_route_match", False):
        primary_bucket = "wrong route"
        bucket_tags.add("wrong route")
        rationale.append("execution route differed from the expected route family")

    stale_expectation = primary_bucket == "stale expectation"
    stale_expectation_annotation_reason = (
        _stale_expectation_annotation_reason(
            case=case,
            execution=execution,
            route=route,
            retrieval=retrieval,
            grounding=grounding,
            failure_modes=failure_modes,
        )
        if stale_expectation
        else None
    )
    cluster = (
        "stale expectations"
        if stale_expectation
        else _TRUE_BUG_CLUSTER_BY_BUCKET.get(primary_bucket, "routing/planner bugs")
    )
    reference = failure_dump.get(str(case.get("query") or ""))
    if reference and str(reference.get("primary_bucket") or "").strip():
        rationale.append(f"reference bucket: {reference['primary_bucket']}")
    if control_pack_present and primary_bucket in {"wrong candidate ranking", "wrong route"}:
        rationale.append("control-pack title/alias evidence should determine whether this is a real routing defect")
    return BugQueueEntry(
        case_id=str(case.get("case_id") or ""),
        query=str(case.get("query") or ""),
        primary_bucket=primary_bucket,
        bucket_tags=tuple(sorted(bucket_tags or {primary_bucket})),
        true_bug=not stale_expectation,
        stale_expectation=stale_expectation,
        cluster=cluster,
        route_expected=_clean_optional(case.get("route_family_expected")),
        route_actual=_clean_optional(execution.get("route_mode")),
        answer_status=_clean_optional(answer_status),
        expected_record_keys=expected_record_keys,
        actual_record_keys=actual_record_keys,
        expected_numeric_fact_types=tuple(
            fact.get("fact_type")
            for fact in case.get("gold_numeric_facts", []) or []
            if str(fact.get("fact_type") or "").strip()
        ),
        missing_numeric_fact_types=tuple(numeric.get("missing_fact_types", []) or ()),
        mismatched_numeric_fact_types=tuple(numeric.get("mismatched_fact_types", []) or ()),
        contamination_prohibited=not bool(case.get("metadata", {}).get("comparison_allowed")),
        candidate_list_used=bool(candidate_list_debug.get("used")),
        suggested_touchpoints=(
            ()
            if stale_expectation
            else _CLUSTER_TOUCHPOINTS.get(cluster, ())
        ),
        expected_regression_tests=(
            ()
            if stale_expectation
            else _CLUSTER_TESTS.get(cluster, ())
        ),
        stale_expectation_annotation_reason=stale_expectation_annotation_reason,
        rationale=tuple(dict.fromkeys(rationale)),
        source_refs={
            "summary_json": "summary.json",
            "per_case_results_jsonl": "per_case_results.jsonl",
            "failures_jsonl": "failures.jsonl",
        },
    )


def _looks_like_stale_expectation(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    route: dict[str, Any],
    retrieval: dict[str, Any],
    grounding: dict[str, Any],
    failure_modes: dict[str, Any],
) -> bool:
    if bool(failure_modes.get("stale_expectation")):
        return True
    if _matches_seeded_stale_expectation(case=case, execution=execution):
        return True
    if _is_failure_dump_abstain_expectation_stale(case=case, execution=execution):
        return True
    if route.get("equivalent_route_match") and not route.get("strict_route_match"):
        return True
    if _is_safe_clarify_for_abstain_expectation(
        case=case,
        execution=execution,
        retrieval=retrieval,
        grounding=grounding,
    ):
        return True
    if _is_better_grounded_abstain_for_weak_comparison_support(
        case=case,
        execution=execution,
        retrieval=retrieval,
    ):
        return True
    if _is_grounded_negative_answer_better_than_abstain_expectation(
        case=case,
        execution=execution,
        retrieval=retrieval,
        grounding=grounding,
    ):
        return True
    if _is_sessionless_follow_up_safe_clarify(
        case=case,
        execution=execution,
    ):
        return True
    if _is_grounded_named_matter_metadata_expectation_stale(
        case=case,
        execution=execution,
        grounding=grounding,
        retrieval=retrieval,
    ):
        return True
    bucket_tags = set(failure_modes.get("buckets", []) or ())
    actual_route = str(execution.get("route_mode") or "")
    expected_route = str(case.get("route_family_expected") or "")
    if bucket_tags - {"wrong route"}:
        return False
    if expected_route not in _NAMED_SINGLE_MATTER_ROUTES or actual_route not in _NAMED_SINGLE_MATTER_ROUTES:
        return False
    if str(execution.get("answer_status") or "") != "answered":
        return False
    if retrieval.get("mixed_record_contamination"):
        return False
    if len(tuple(execution.get("active_record_keys", []) or ())) != 1:
        return False
    correctness = grounding.get("answer_correctness")
    faithfulness = grounding.get("faithfulness")
    hallucination_rate = grounding.get("hallucination_rate")
    return (
        correctness is not None
        and float(correctness) >= 0.99
        and faithfulness is not None
        and float(faithfulness) >= 0.99
        and float(hallucination_rate or 0.0) <= 0.0
    )


def _is_safe_clarify_for_abstain_expectation(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    retrieval: dict[str, Any],
    grounding: dict[str, Any],
) -> bool:
    if not (
        bool(case.get("must_abstain"))
        or str(case.get("route_family_expected") or "") == "abstain"
    ):
        return False
    if str(execution.get("answer_status") or "") != "clarify":
        return False
    if retrieval.get("mixed_record_contamination"):
        return False
    candidate_list_debug = dict(execution.get("debug", {}).get("candidate_list_debug", {}) or {})
    if not candidate_list_debug.get("used"):
        return False
    candidate_list_correctness = retrieval.get("candidate_list_correctness")
    if candidate_list_correctness is not None and float(candidate_list_correctness) < 0.99:
        return False
    if grounding.get("abstain_correct") is False:
        return False
    correctness = grounding.get("answer_correctness")
    if correctness is not None and float(correctness) < 0.99:
        return False
    faithfulness = grounding.get("faithfulness")
    if faithfulness is not None and float(faithfulness) < 0.99:
        return False
    hallucination_rate = grounding.get("hallucination_rate")
    return float(hallucination_rate or 0.0) <= 0.0


def _matches_seeded_stale_expectation(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
) -> bool:
    expected_failure_buckets = {
        str(item).strip().lower()
        for item in tuple(case.get("expected_failure_buckets", []) or ())
        if str(item).strip()
    }
    if "stale expectation" not in expected_failure_buckets:
        return False
    return str(execution.get("answer_status") or "") in {"clarify", "abstained"}


def _is_failure_dump_abstain_expectation_stale(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
) -> bool:
    if str(execution.get("answer_status") or "") != "abstained":
        return False
    if str(case.get("route_family_expected") or "").strip().lower() == "abstain":
        return False
    notes = str(case.get("notes") or "").lower()
    if "expected route abstain" not in notes:
        return False
    source_case_refs = {
        str(item).strip().lower()
        for item in tuple(case.get("source_case_refs", []) or ())
        if str(item).strip()
    }
    if "failure_dump:failed_cases.json" not in source_case_refs:
        return False
    metadata = dict(case.get("metadata", {}) or {})
    reference_failure_dump = dict(metadata.get("reference_failure_dump", {}) or {})
    return str(reference_failure_dump.get("primary_bucket") or "").strip().lower() == "wrong route"


def _is_better_grounded_abstain_for_weak_comparison_support(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    retrieval: dict[str, Any],
) -> bool:
    expected_record_keys = {
        record_key.strip()
        for item in tuple(case.get("expected_record_keys", []) or ())
        for record_key in str(item).split(";")
        if record_key.strip()
    }
    if len(expected_record_keys) < 2:
        return False
    if str(execution.get("answer_status") or "") != "abstained":
        return False
    route_debug = dict(execution.get("debug", {}).get("route_debug", {}) or {})
    if not route_debug.get("comparison_intent"):
        return False
    if retrieval.get("mixed_record_contamination"):
        return False
    retrieved_record_keys = {
        str(item).strip()
        for item in tuple(retrieval.get("retrieved_record_keys", []) or ())
        if str(item).strip()
    }
    if retrieved_record_keys != expected_record_keys:
        return False
    substantive_section_types = {
        "operative_order",
        "directions",
        "findings",
        "facts",
        "background",
        "issues",
        "reply_or_submissions",
    }
    per_record_section_types: dict[str, set[str]] = {record_key: set() for record_key in expected_record_keys}
    for item in tuple(execution.get("retrieved_context", []) or ()):
        record_key = str(item.get("record_key") or "").strip()
        if record_key not in per_record_section_types:
            continue
        section_type = str(item.get("section_type") or "").strip().lower()
        if section_type:
            per_record_section_types[record_key].add(section_type)
    return any(
        section_types and section_types.isdisjoint(substantive_section_types)
        for section_types in per_record_section_types.values()
    )


def _is_sessionless_follow_up_safe_clarify(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
) -> bool:
    if str(case.get("issue_class") or "") != "regression":
        return False
    if bool(case.get("reuse_previous_session")) or case.get("session_group") or case.get("session_id"):
        return False
    if str(execution.get("answer_status") or "") not in {"clarify", "abstained"}:
        return False
    expected_record_keys = tuple(case.get("expected_record_keys", []) or ())
    if not expected_record_keys:
        return False
    metadata = dict(case.get("metadata", {}) or {})
    expected_titles = tuple(
        str(item).strip()
        for item in tuple(metadata.get("expected_titles", []) or ())
        if str(item).strip()
    )
    if not expected_titles:
        return False
    query_tokens = set(re.findall(r"[a-z0-9]+", str(case.get("query") or "").lower()))
    specific_title_tokens: set[str] = set()
    for title in expected_titles:
        specific_title_tokens.update(
            token
            for token in re.findall(r"[a-z0-9]+", title.lower())
            if len(token) >= 3 and token not in _GENERIC_EXPECTED_TITLE_TOKENS
        )
    return bool(specific_title_tokens) and not bool(query_tokens & specific_title_tokens)


def _is_grounded_negative_answer_better_than_abstain_expectation(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    retrieval: dict[str, Any],
    grounding: dict[str, Any],
) -> bool:
    if not (
        bool(case.get("must_abstain"))
        or str(case.get("route_family_expected") or "").strip().lower() == "abstain"
    ):
        return False
    if str(case.get("issue_class") or "").strip().lower() != "abstain":
        return False
    if str(execution.get("answer_status") or "") != "answered":
        return False
    if retrieval.get("mixed_record_contamination"):
        return False
    active_record_keys = tuple(execution.get("active_record_keys", []) or ())
    if len(active_record_keys) != 1:
        return False
    faithfulness = grounding.get("faithfulness")
    if faithfulness is not None and float(faithfulness) < 0.99:
        return False
    hallucination_rate = grounding.get("hallucination_rate")
    if float(hallucination_rate or 0.0) > 0.0:
        return False
    answer_text = str(execution.get("answer_text") or "")
    return bool(_NEGATIVE_GROUNDED_ANSWER_RE.search(answer_text))


def _stale_expectation_annotation_reason(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    route: dict[str, Any],
    retrieval: dict[str, Any],
    grounding: dict[str, Any],
    failure_modes: dict[str, Any],
) -> str:
    if _is_better_grounded_abstain_for_weak_comparison_support(
        case=case,
        execution=execution,
        retrieval=retrieval,
    ):
        return "better grounded answer than prior expectation"
    if _matches_seeded_stale_expectation(
        case=case,
        execution=execution,
    ):
        return "safer clarify behavior"
    if _is_failure_dump_abstain_expectation_stale(
        case=case,
        execution=execution,
    ):
        return "better grounded answer than prior expectation"
    if _is_grounded_negative_answer_better_than_abstain_expectation(
        case=case,
        execution=execution,
        retrieval=retrieval,
        grounding=grounding,
    ):
        return "better grounded answer than prior expectation"
    if _is_sessionless_follow_up_safe_clarify(
        case=case,
        execution=execution,
    ):
        return "safer clarify behavior"
    if _is_grounded_named_matter_metadata_expectation_stale(
        case=case,
        execution=execution,
        grounding=grounding,
        retrieval=retrieval,
    ):
        return "better grounded answer than prior expectation"
    if _is_safe_clarify_for_abstain_expectation(
        case=case,
        execution=execution,
        retrieval=retrieval,
        grounding=grounding,
    ):
        return "safer clarify behavior"
    if bool(failure_modes.get("stale_expectation")):
        return "equivalent acceptable route"
    if route.get("equivalent_route_match") and not route.get("strict_route_match"):
        return "equivalent acceptable route"
    return "equivalent acceptable route"


def _is_weak_metadata_support(*, case: dict[str, Any], execution: dict[str, Any]) -> bool:
    if bool(case.get("must_abstain")) or str(case.get("route_family_expected") or "") == "abstain":
        return False
    if str(execution.get("answer_status") or "") == "abstained":
        return False
    if not bool(case.get("must_use_metadata")):
        return False
    metadata_debug = dict(execution.get("debug", {}).get("metadata_debug", {}) or {})
    return not bool(metadata_debug.get("used"))


def _is_grounded_named_matter_metadata_expectation_stale(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    grounding: dict[str, Any],
    retrieval: dict[str, Any],
) -> bool:
    if not bool(case.get("must_use_metadata")):
        return False
    if tuple(case.get("gold_numeric_facts", []) or ()):
        return False
    if str(case.get("prompt_family_expected") or "") == "metadata-first fact":
        return False
    if str(execution.get("answer_status") or "") != "answered":
        return False
    if retrieval.get("mixed_record_contamination"):
        return False
    expected_record_keys = {
        record_key.strip()
        for item in tuple(case.get("expected_record_keys", []) or ())
        for record_key in str(item).split(";")
        if record_key.strip()
    }
    actual_record_keys = {
        str(item).strip()
        for item in tuple(execution.get("active_record_keys", []) or ())
        if str(item).strip()
    }
    if expected_record_keys and actual_record_keys != expected_record_keys:
        return False
    route_debug = dict(execution.get("debug", {}).get("route_debug", {}) or {})
    if (
        route_debug.get("asks_order_signatory")
        or route_debug.get("asks_order_date")
        or route_debug.get("asks_legal_provisions")
        or route_debug.get("asks_provision_explanation")
        or route_debug.get("asks_order_pan")
        or route_debug.get("asks_order_amount")
        or route_debug.get("asks_order_holding")
        or route_debug.get("asks_order_parties")
        or route_debug.get("asks_order_numeric_fact")
    ):
        return False
    correctness = grounding.get("answer_correctness")
    faithfulness = grounding.get("faithfulness")
    hallucination_rate = grounding.get("hallucination_rate")
    return (
        correctness is not None
        and float(correctness) >= 0.99
        and faithfulness is not None
        and float(faithfulness) >= 0.99
        and float(hallucination_rate or 0.0) <= 0.0
    )


def _is_wrong_answer(
    *,
    case: dict[str, Any],
    route: dict[str, Any],
    grounding: dict[str, Any],
) -> bool:
    if str(case.get("issue_class") or "") == "regression":
        return True
    correctness = grounding.get("answer_correctness")
    if correctness is not None and float(correctness) < 0.99 and route.get("equivalent_route_match"):
        return True
    return False


def _is_contamination(
    *,
    case: dict[str, Any],
    retrieval: dict[str, Any],
    failure_modes: dict[str, Any],
    actual_record_keys: tuple[str, ...],
) -> bool:
    expected_record_keys = {
        record_key.strip()
        for item in tuple(case.get("expected_record_keys", []) or ())
        for record_key in str(item).split(";")
        if record_key.strip()
    }
    if expected_record_keys and set(actual_record_keys) == expected_record_keys:
        return False
    return bool(
        retrieval.get("mixed_record_contamination")
        or "contamination" in set(failure_modes.get("buckets", []) or ())
        or (len(set(actual_record_keys)) > 1 and len(expected_record_keys) <= 1)
    )


def _is_wrong_candidate_ranking(*, execution: dict[str, Any], case: dict[str, Any]) -> bool:
    debug = dict(execution.get("debug", {}) or {})
    candidate_list_debug = dict(debug.get("candidate_list_debug", {}) or {})
    if not candidate_list_debug.get("used"):
        return False
    if str(execution.get("answer_status") or "") != "clarify":
        return False
    if str(case.get("route_family_expected") or "") == "clarify":
        return False
    return True


def _is_missing_clarify(*, case: dict[str, Any], answer_status: str) -> bool:
    return bool(case.get("must_clarify")) and answer_status != "clarify"


def _is_over_abstain(*, case: dict[str, Any], answer_status: str) -> bool:
    if bool(case.get("must_abstain")) or bool(case.get("must_clarify")):
        return False
    return answer_status in {"abstained", "clarify"}


def _is_current_fact_order_confusion(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    route_debug: dict[str, Any],
) -> bool:
    actual_route = str(execution.get("route_mode") or "")
    if actual_route not in _CORPUS_ROUTES:
        return False
    if bool(case.get("must_use_structured_current_info")) or bool(case.get("must_use_official_web")):
        return True
    return bool(
        route_debug.get("appears_structured_current_info")
        or route_debug.get("appears_current_official_lookup")
        or route_debug.get("appears_current_news_lookup")
    )


def _is_company_role_misroute(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    route_debug: dict[str, Any],
) -> bool:
    actual_route = str(execution.get("route_mode") or "")
    return (
        ("company_role" in set(case.get("tags", []) or ()) or route_debug.get("appears_company_role_current_fact"))
        and actual_route not in _CURRENT_FACT_ROUTES
    )


def _is_active_matter_follow_up_failure(*, case: dict[str, Any], route_debug: dict[str, Any]) -> bool:
    return bool(case.get("must_use_active_matter") or route_debug.get("active_order_override"))


def _is_numeric_fact_bug(*, case: dict[str, Any], numeric: dict[str, Any]) -> bool:
    expected_fact_count = int(numeric.get("expected_fact_count") or 0)
    if expected_fact_count <= 0:
        return False
    numeric_accuracy = numeric.get("numeric_accuracy")
    return numeric_accuracy is None or float(numeric_accuracy) < 1.0


def _is_person_vs_trust_bug(*, case: dict[str, Any], numeric: dict[str, Any]) -> bool:
    if "person_vs_trust" not in set(case.get("tags", []) or ()):
        return False
    numeric_accuracy = numeric.get("numeric_accuracy")
    return numeric_accuracy is None or float(numeric_accuracy) < 1.0


def _is_matter_type_phrasing_bug(
    *,
    case: dict[str, Any],
    execution: dict[str, Any],
    failure_modes: dict[str, Any],
) -> bool:
    if "settlement/final-order phrasing issue" in set(failure_modes.get("buckets", []) or ()):
        return True
    answer_text = str(execution.get("answer_text") or "").lower()
    tags = set(case.get("tags", []) or ())
    if "settlement" in " ".join(tags) and "final order" in answer_text:
        return True
    return False


def _normalize_query_pattern(query: str) -> str:
    normalized = " ".join(query.lower().split())
    for pattern, replacement in _QUERY_PATTERN_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"\bwhat was the settlement amount in\b", "settlement amount in", normalized)
    normalized = re.sub(r"\bwhat were the terms of settlement in\b", "terms of settlement in", normalized)
    normalized = re.sub(r"\bwhat did sebi finally direct in\b", "final direction in", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _clean_optional(value: object) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None
