"""Packaged Phase 4 eval case loading."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SAMPLE_EVAL_CASES_PATH = Path(__file__).with_name("sample_eval_cases.jsonl")
REQUIRED_SAMPLE_TAGS = frozenset(
    {
        "exact_title_lookup",
        "free_form_settlement_outcome",
        "free_form_settlement_amount",
        "party_specific_settlement",
        "follow_up_after_exact_lookup",
        "wrong_settlement_party_abstain",
        "generic_settlement_explanation",
        "current_news_lookup",
        "historical_official_lookup",
        "sebi_income_official_lookup",
        "sebi_charge_official_lookup",
        "general_person_lookup",
        "named_order_no_web_override",
    }
)


def load_eval_cases(path: Path | None = None) -> list[dict[str, object]]:
    """Load JSONL eval cases from disk."""

    resolved_path = path or DEFAULT_SAMPLE_EVAL_CASES_PATH
    cases: list[dict[str, object]] = []
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def validate_eval_cases(cases: list[dict[str, object]]) -> list[str]:
    """Return validation errors for the packaged sample eval cases."""

    errors: list[str] = []
    if len(cases) < 10:
        errors.append("expected at least 10 eval cases")

    seen_tags: set[str] = set()
    for index, case in enumerate(cases, start=1):
        query = str(case.get("query", "")).strip()
        if not query:
            errors.append(f"case {index} is missing query")
        tags = case.get("tags", [])
        if not isinstance(tags, list) or not tags:
            errors.append(f"case {index} is missing tags")
            continue
        seen_tags.update(str(tag) for tag in tags)

    missing_tags = REQUIRED_SAMPLE_TAGS - seen_tags
    if missing_tags:
        errors.append(
            "missing required sample tags: " + ", ".join(sorted(missing_tags))
        )
    return errors
