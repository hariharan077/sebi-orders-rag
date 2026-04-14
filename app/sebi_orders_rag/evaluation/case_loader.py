"""Seed dataset loaders built from existing repository assets."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..control import load_control_pack
from ..eval.cases import load_eval_cases as load_packaged_phase4_cases
from ..eval.triage import load_failure_dump_reference
from .annotation import annotate_cases, merge_case_sources, seed_numeric_anchor_cases
from .dataset import case_from_dict, make_case_id, merge_datasets
from .schemas import EvaluationCase

_SEMANTIC_REGRESSION_GUIDANCE_RE = re.compile(
    r"should abstain|should not relabel|not an exemption order|no .* order is in scope",
    re.IGNORECASE,
)


def build_seed_dataset(
    *,
    control_pack_root: str | Path | None = None,
    failure_dump_root: str | Path | None = None,
    sample_eval_cases_path: str | Path | None = None,
) -> tuple[list[EvaluationCase], list[str]]:
    """Build a merged seed dataset from existing assets."""

    source_files: list[str] = []
    cases: list[EvaluationCase] = []

    sample_cases = _load_phase4_eval_cases(sample_eval_cases_path)
    cases.extend(sample_cases)
    if sample_eval_cases_path is not None:
        source_files.append(str(Path(sample_eval_cases_path).expanduser().resolve(strict=False)))
    else:
        source_files.append(str((Path(__file__).resolve().parents[1] / "eval" / "sample_eval_cases.jsonl")))

    control_pack = load_control_pack(control_pack_root)
    if control_pack is not None:
        source_files.append(str(control_pack.root))
        cases.extend(_load_control_pack_eval_cases(control_pack.root))
        cases.extend(_load_control_pack_wrong_answer_cases(control_pack.root))

    if failure_dump_root is not None:
        resolved = Path(failure_dump_root).expanduser().resolve(strict=False)
        if resolved.exists():
            source_files.append(str(resolved))
            cases.extend(_load_failure_dump_cases(resolved))

    cases.extend(seed_numeric_anchor_cases())
    annotated = annotate_cases(
        cases,
        control_pack_root=control_pack_root,
        failure_dump_root=failure_dump_root,
    )
    deduped = merge_datasets(
        name="sebi_eval_seed",
        cases=annotated,
        source_files=source_files,
    )
    return list(deduped.cases), list(deduped.source_files)


def build_dataset(
    *,
    name: str,
    control_pack_root: str | Path | None = None,
    failure_dump_root: str | Path | None = None,
    sample_eval_cases_path: str | Path | None = None,
) -> tuple[tuple[EvaluationCase, ...], list[str], str]:
    """Return annotated seed cases plus source files and version."""

    cases, source_files = build_seed_dataset(
        control_pack_root=control_pack_root,
        failure_dump_root=failure_dump_root,
        sample_eval_cases_path=sample_eval_cases_path,
    )
    dataset = merge_datasets(name=name, cases=cases, source_files=source_files)
    return dataset.cases, list(dataset.source_files), dataset.version


def _load_phase4_eval_cases(path: str | Path | None) -> list[EvaluationCase]:
    rows = load_packaged_phase4_cases(Path(path) if path is not None else None)
    cases: list[EvaluationCase] = []
    for row in rows:
        payload = dict(row)
        payload["case_id"] = make_case_id(prefix="phase4", query=str(payload.get("query", "")))
        payload["route_family_expected"] = payload.pop("expected_route_mode", None)
        payload["issue_class"] = _phase4_issue_class(payload)
        payload["allowed_routes"] = (
            [payload["route_family_expected"]]
            if payload.get("route_family_expected")
            else []
        )
        cases.append(case_from_dict(payload))
    return cases


def _load_control_pack_eval_cases(root: Path) -> list[EvaluationCase]:
    path = root / "eval_queries.jsonl"
    if not path.exists():
        return []
    cases: list[EvaluationCase] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        query = str(payload.get("query", "")).strip()
        route = str(payload.get("expected_route_mode", "")).strip() or None
        record_key = str(payload.get("expected_record_key", "")).strip()
        issue_class = "clarify" if route == "clarify" else "abstain" if route == "abstain" else "gold_fact" if record_key else "open_ended"
        cases.append(
            case_from_dict(
                {
                    "case_id": make_case_id(prefix="control_eval", query=query),
                    "query": query,
                    "route_family_expected": route,
                    "allowed_routes": ([route] if route else []),
                    "expected_record_keys": ([record_key] if record_key else []),
                    "must_abstain": route == "abstain",
                    "must_clarify": route == "clarify",
                    "session_group": payload.get("session_group"),
                    "reuse_previous_session": bool(payload.get("reuse_previous_session", False)),
                    "tags": ["control_pack_eval"],
                    "issue_class": issue_class,
                    "difficulty": "medium",
                    "notes": str(payload.get("notes", "")).strip(),
                    "source_case_refs": ["control_pack:eval_queries.jsonl"],
                    "source_files": [str(path)],
                }
            )
        )
    return cases


def _load_control_pack_wrong_answer_cases(root: Path) -> list[EvaluationCase]:
    path = root / "wrong_answer_examples.jsonl"
    if not path.exists():
        return []
    cases: list[EvaluationCase] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        query = str(payload.get("user_query", "")).strip()
        expected_record_key = str(payload.get("expected_record_key", "")).strip()
        incorrect_keys = [
            item.strip()
            for item in str(payload.get("incorrectly_pulled_record_keys", "")).replace(";", ",").split(",")
            if item.strip()
        ]
        answer_guidance = str(payload.get("what_it_should_have_answered", "")).strip()
        cases.append(
            case_from_dict(
                {
                    "case_id": make_case_id(prefix="wrong_answer", query=query),
                    "query": query,
                    "route_family_expected": (
                        "clarify" if not expected_record_key else "exact_lookup"
                    ),
                    "allowed_routes": (
                        ["clarify", "abstain"]
                        if not expected_record_key
                        else ["exact_lookup", "hierarchical_rag", "memory_scoped_rag"]
                    ),
                    "expected_record_keys": ([expected_record_key] if expected_record_key else []),
                    "must_clarify": not expected_record_key,
                    "tags": ["wrong_answer_regression"],
                    "issue_class": "regression",
                    "difficulty": "hard",
                    "notes": answer_guidance,
                    "gold_answer_short": (
                        answer_guidance if _SEMANTIC_REGRESSION_GUIDANCE_RE.search(answer_guidance) else None
                    ),
                    "expected_failure_buckets": (
                        ["contamination"] if incorrect_keys else []
                    ),
                    "metadata": {
                        "incorrect_record_keys": incorrect_keys,
                        "observed_route_mode": payload.get("observed_route_mode"),
                        "observed_answer_status": payload.get("observed_answer_status"),
                        "answer_guidance": answer_guidance or None,
                    },
                    "source_case_refs": ["control_pack:wrong_answer_examples.jsonl"],
                    "source_files": [str(path)],
                }
            )
        )
    return cases


def _load_failure_dump_cases(root: Path) -> list[EvaluationCase]:
    reference = load_failure_dump_reference(root)
    if reference is None:
        return []
    cases: list[EvaluationCase] = []
    for query, item in sorted(reference.failed_cases_by_query.items()):
        route = str(item.get("expected_route_mode", "") or "").strip() or None
        case_payload = {
            "case_id": make_case_id(prefix="failure_dump", query=query),
            "query": query,
            "route_family_expected": route,
            "allowed_routes": ([route] if route else []),
            "expected_record_keys": [
                str(item.get("expected_record_key") or "").strip()
            ]
            if str(item.get("expected_record_key") or "").strip()
            else [],
            "must_abstain": route == "abstain",
            "must_clarify": route == "clarify",
            "tags": ["failure_dump_seed"],
            "issue_class": "regression",
            "difficulty": "hard",
            "notes": "; ".join(item.get("reasons", [])),
            "expected_failure_buckets": [
                str(item.get("primary_bucket") or "").strip()
            ]
            if str(item.get("primary_bucket") or "").strip()
            else [],
            "source_case_refs": ["failure_dump:failed_cases.json"],
            "source_files": [str(root)],
        }
        cases.append(case_from_dict(case_payload))
    return cases


def _phase4_issue_class(payload: dict[str, object]) -> str:
    route = str(payload.get("route_family_expected", "") or "")
    if route == "clarify":
        return "clarify"
    if route == "abstain":
        return "abstain"
    if payload.get("gold_numeric_facts"):
        return "gold_fact"
    if payload.get("min_citations"):
        return "open_ended"
    return "open_ended"
