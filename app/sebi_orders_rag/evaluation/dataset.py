"""Dataset loading, validation, serialization, and deduplication."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .schemas import EvaluationCase, EvaluationDataset, GoldNumericFact

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SCAFFOLD_GOLD_ANSWER_SHORTS = {
    "Exact title lookup should lock to one record.",
    "Explicit comparison request permits multi-record grounding.",
    "Follow-up should stay anchored to prior record.",
    "General explanatory query; no record lock expected.",
    "Named substantive query should stay within the named matter.",
    "Session-seeding exact lookup.",
    "The system should abstain rather than fuse loosely related matters.",
}


def make_case_id(*, prefix: str, query: str, suffix: str | None = None) -> str:
    """Build a stable case identifier from a query string."""

    slug = _SLUG_RE.sub("-", query.lower()).strip("-")
    slug = slug[:72] or "case"
    if suffix:
        suffix_slug = _SLUG_RE.sub("-", suffix.lower()).strip("-")
        return f"{prefix}:{slug}:{suffix_slug}"
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}:{slug}:{digest}"


def load_dataset(path: str | Path, *, name: str | None = None) -> EvaluationDataset:
    """Load one JSONL dataset from disk."""

    dataset_path = Path(path).expanduser().resolve(strict=False)
    cases = tuple(load_cases(dataset_path))
    payload = dataset_to_version_payload(cases)
    version = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return EvaluationDataset(
        name=name or dataset_path.stem,
        version=version,
        cases=cases,
        source_files=(str(dataset_path),),
    )


def load_cases(path: str | Path) -> list[EvaluationCase]:
    """Load JSONL cases from disk."""

    rows: list[EvaluationCase] = []
    dataset_path = Path(path).expanduser().resolve(strict=False)
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(case_from_dict(json.loads(line)))
    return rows


def write_dataset(path: str | Path, dataset: EvaluationDataset) -> Path:
    """Persist one dataset JSONL file to disk."""

    return write_cases(path, dataset.cases)


def write_cases(path: str | Path, cases: Sequence[EvaluationCase]) -> Path:
    """Persist cases to JSONL."""

    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case.to_dict(), sort_keys=True) for case in cases]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def merge_datasets(
    *,
    name: str,
    cases: Iterable[EvaluationCase],
    source_files: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
) -> EvaluationDataset:
    """Create one deduplicated dataset."""

    deduped = tuple(dedupe_cases(cases))
    return EvaluationDataset(
        name=name,
        version=hashlib.sha1(dataset_to_version_payload(deduped).encode("utf-8")).hexdigest()[:12],
        cases=deduped,
        source_files=tuple(dict.fromkeys(source_files)),
        metadata=dict(metadata or {}),
    )


def dedupe_cases(cases: Iterable[EvaluationCase]) -> list[EvaluationCase]:
    """Keep the last occurrence for each case id."""

    by_id: dict[str, EvaluationCase] = {}
    for case in cases:
        by_id[case.case_id] = case
    return list(by_id.values())


def case_from_dict(payload: dict[str, Any]) -> EvaluationCase:
    """Parse one case payload from JSON."""

    numeric_facts = tuple(
        GoldNumericFact(**dict(item))
        for item in _list_of_dicts(payload.get("gold_numeric_facts"))
    )
    tags = tuple(str(item).strip() for item in payload.get("tags", []) if str(item).strip())
    allowed_routes = tuple(
        str(item).strip() for item in payload.get("allowed_routes", []) if str(item).strip()
    )
    expected_record_keys = _parse_expected_record_keys(
        payload.get("expected_record_keys", payload.get("expected_record_key", []))
    )
    expected_bucket_names = tuple(
        str(item).strip() for item in payload.get("expected_bucket_names", []) if str(item).strip()
    )
    source_files = tuple(
        str(item).strip() for item in payload.get("source_files", []) if str(item).strip()
    )
    source_case_refs = tuple(
        str(item).strip() for item in payload.get("source_case_refs", []) if str(item).strip()
    )
    expected_failure_buckets = tuple(
        str(item).strip()
        for item in payload.get("expected_failure_buckets", [])
        if str(item).strip()
    )
    route_family_expected = _clean_optional(
        payload.get("route_family_expected", payload.get("expected_route_mode"))
    )
    issue_class = str(payload.get("issue_class", "open_ended")).strip() or "open_ended"
    case_id = _clean_optional(payload.get("case_id"))
    query = str(payload.get("query", "")).strip()
    gold_answer_short = _normalize_gold_answer_short(payload.get("gold_answer_short"))
    metadata = dict(payload.get("metadata", {}) or {})
    if issue_class == "regression" and gold_answer_short and not str(
        metadata.get("answer_guidance") or ""
    ).strip():
        metadata["answer_guidance"] = gold_answer_short
    if not case_id:
        case_id = make_case_id(prefix="eval", query=query)
    return EvaluationCase(
        case_id=case_id,
        query=query,
        session_id=_clean_optional(payload.get("session_id")),
        session_group=_clean_optional(payload.get("session_group")),
        reuse_previous_session=bool(payload.get("reuse_previous_session", False)),
        route_family_expected=route_family_expected,
        allowed_routes=allowed_routes,
        expected_record_keys=expected_record_keys,
        expected_bucket_names=expected_bucket_names,
        gold_answer_short=gold_answer_short,
        gold_answer_long=_clean_optional(payload.get("gold_answer_long")),
        gold_numeric_facts=numeric_facts,
        must_abstain=bool(payload.get("must_abstain", False)),
        must_clarify=bool(payload.get("must_clarify", False)),
        must_use_active_matter=bool(payload.get("must_use_active_matter", False)),
        must_use_metadata=bool(payload.get("must_use_metadata", False)),
        must_use_structured_current_info=bool(
            payload.get("must_use_structured_current_info", False)
        ),
        must_use_official_web=bool(payload.get("must_use_official_web", False)),
        must_not_use_web=bool(payload.get("must_not_use_web", False)),
        tags=tags,
        issue_class=issue_class,
        difficulty=str(payload.get("difficulty", "medium")).strip() or "medium",
        notes=str(payload.get("notes", "")).strip(),
        prompt_family_expected=_clean_optional(payload.get("prompt_family_expected")),
        expected_failure_buckets=expected_failure_buckets,
        source_files=source_files,
        source_case_refs=source_case_refs,
        metadata=metadata,
    )


def validate_case(case: EvaluationCase) -> list[str]:
    """Return validation errors for one case."""

    errors: list[str] = []
    if not case.case_id.strip():
        errors.append("case_id is required")
    if not case.query.strip():
        errors.append(f"{case.case_id}: query is required")
    if case.must_abstain and case.must_clarify:
        errors.append(f"{case.case_id}: cannot require both abstain and clarify")
    if case.must_use_official_web and case.must_not_use_web:
        errors.append(f"{case.case_id}: cannot require and forbid web usage")
    if case.route_family_expected and case.allowed_routes:
        if case.route_family_expected not in case.allowed_routes:
            errors.append(
                f"{case.case_id}: route_family_expected must appear in allowed_routes when both are set"
            )
    if case.must_clarify and case.expected_record_keys and case.issue_class == "redteam":
        return errors
    for item in case.gold_numeric_facts:
        if not item.fact_type.strip():
            errors.append(f"{case.case_id}: gold_numeric_fact.fact_type is required")
        if item.value_numeric is None and not (item.value_text or "").strip():
            errors.append(
                f"{case.case_id}: gold_numeric_fact requires value_numeric or value_text"
            )
    expectation_update = dict(case.metadata.get("expectation_update", {}) or {})
    if expectation_update.get("approved"):
        reason = str(expectation_update.get("reason") or "").strip()
        allowed_reasons = {
            "safer clarify behavior",
            "equivalent acceptable route",
            "better grounded answer than prior expectation",
        }
        if reason not in allowed_reasons:
            errors.append(
                f"{case.case_id}: expectation_update.reason must be one of {sorted(allowed_reasons)}"
            )
    return errors


def validate_dataset(cases: Sequence[EvaluationCase]) -> list[str]:
    """Return validation errors across a dataset."""

    errors: list[str] = []
    seen_ids: set[str] = set()
    for case in cases:
        if case.case_id in seen_ids:
            errors.append(f"duplicate case_id: {case.case_id}")
        seen_ids.add(case.case_id)
        errors.extend(validate_case(case))
    return errors


def dataset_to_version_payload(cases: Sequence[EvaluationCase]) -> str:
    """Return a stable JSON payload for dataset hashing."""

    serializable = [case.to_dict() for case in cases]
    return json.dumps(serializable, sort_keys=True, separators=(",", ":"))


def filter_cases(
    cases: Sequence[EvaluationCase],
    *,
    include_tags: Sequence[str] = (),
    exclude_tags: Sequence[str] = (),
    limit: int | None = None,
) -> tuple[EvaluationCase, ...]:
    """Filter cases by tag and optional limit."""

    include = {item.strip() for item in include_tags if item.strip()}
    exclude = {item.strip() for item in exclude_tags if item.strip()}
    selected: list[EvaluationCase] = []
    for case in cases:
        tags = set(case.tags)
        if include and not include.issubset(tags):
            continue
        if exclude and tags & exclude:
            continue
        selected.append(case)
        if limit is not None and len(selected) >= limit:
            break
    return tuple(selected)


def apply_case_updates(
    cases: Sequence[EvaluationCase],
    updates: dict[str, dict[str, Any]],
) -> tuple[EvaluationCase, ...]:
    """Apply partial updates by case id."""

    updated: list[EvaluationCase] = []
    for case in cases:
        patch = updates.get(case.case_id)
        if not patch:
            updated.append(case)
            continue
        merged = case.to_dict()
        merged.update(patch)
        updated.append(case_from_dict(merged))
    return tuple(updated)


def add_case_sources(case: EvaluationCase, *sources: str) -> EvaluationCase:
    """Append source files to one case."""

    merged_sources = tuple(dict.fromkeys((*case.source_files, *[item for item in sources if item])))
    return replace(case, source_files=merged_sources)


def _clean_optional(value: object) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _parse_expected_record_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        items = value
    elif value is None:
        items = ()
    else:
        items = (value,)
    parsed: list[str] = []
    for item in items:
        for candidate in str(item).split(";"):
            cleaned = candidate.strip()
            if cleaned:
                parsed.append(cleaned)
    return tuple(dict.fromkeys(parsed))


def _normalize_gold_answer_short(value: object) -> str | None:
    cleaned = _clean_optional(value)
    if cleaned in _SCAFFOLD_GOLD_ANSWER_SHORTS:
        return None
    return cleaned
