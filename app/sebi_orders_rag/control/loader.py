"""Control-pack loading helpers for SEBI Orders retrieval hardening."""

from __future__ import annotations

import csv
import json
from datetime import date
from functools import lru_cache
from pathlib import Path
import re

from ..exceptions import ConfigurationError
from ..normalization import generate_order_alias_variants
from .models import (
    ConfusionPair,
    ControlPack,
    DocumentIndexRow,
    EntityAliasRow,
    EvalQueryCase,
    StrictAnswerRule,
    WrongAnswerExample,
)

_CONTROL_PACK_FILES = (
    "document_index.csv",
    "confusion_list.csv",
    "eval_queries.jsonl",
    "wrong_answer_examples.jsonl",
    "entity_aliases.csv",
    "strict_answer_rule.md",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_control_pack(root: str | Path | None) -> ControlPack | None:
    """Load and cache one control pack from disk."""

    if root is None:
        return None
    resolved_root = Path(root).expanduser().resolve(strict=False)
    return _load_control_pack_cached(str(resolved_root))


@lru_cache(maxsize=4)
def _load_control_pack_cached(root: str) -> ControlPack:
    resolved_root = Path(root)
    _validate_control_pack_root(resolved_root)

    document_index = tuple(_load_document_index(resolved_root / "document_index.csv"))
    confusion_pairs = tuple(_load_confusion_pairs(resolved_root / "confusion_list.csv"))
    eval_queries = tuple(_load_eval_queries(resolved_root / "eval_queries.jsonl"))
    wrong_answer_examples = tuple(
        _load_wrong_answer_examples(resolved_root / "wrong_answer_examples.jsonl")
    )
    entity_aliases = tuple(_load_entity_aliases(resolved_root / "entity_aliases.csv"))
    strict_answer_rule = _load_strict_answer_rule(resolved_root / "strict_answer_rule.md")

    documents_by_record_key = {row.record_key: row for row in document_index}
    aliases_by_record_key: dict[str, list[EntityAliasRow]] = {}
    alias_variants: dict[str, list[EntityAliasRow]] = {}
    for alias_row in entity_aliases:
        for record_key in alias_row.related_record_keys:
            aliases_by_record_key.setdefault(record_key, []).append(alias_row)
        for variant in _alias_variants(alias_row):
            alias_variants.setdefault(variant, []).append(alias_row)

    confusion_map: dict[str, list[ConfusionPair]] = {}
    for pair in confusion_pairs:
        confusion_map.setdefault(pair.record_key_a, []).append(pair)
        confusion_map.setdefault(pair.record_key_b, []).append(pair)

    return ControlPack(
        root=resolved_root,
        document_index=document_index,
        confusion_pairs=confusion_pairs,
        eval_queries=eval_queries,
        wrong_answer_examples=wrong_answer_examples,
        entity_aliases=entity_aliases,
        strict_answer_rule=strict_answer_rule,
        documents_by_record_key=documents_by_record_key,
        aliases_by_record_key={
            key: tuple(value) for key, value in aliases_by_record_key.items()
        },
        alias_variants={key: tuple(value) for key, value in alias_variants.items()},
        confusion_map={key: tuple(value) for key, value in confusion_map.items()},
    )


def _validate_control_pack_root(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise ConfigurationError(f"Control pack root does not exist: {root}")
    missing = [name for name in _CONTROL_PACK_FILES if not (root / name).exists()]
    if missing:
        raise ConfigurationError(
            f"Control pack root is missing required files: {', '.join(sorted(missing))}"
        )


def _load_document_index(path: Path) -> list[DocumentIndexRow]:
    rows: list[DocumentIndexRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            rows.append(
                DocumentIndexRow(
                    record_key=str(raw_row.get("record_key", "")).strip(),
                    exact_title=str(raw_row.get("exact_title", "")).strip(),
                    bucket_category=str(raw_row.get("bucket_category", "")).strip(),
                    order_date=_parse_date(raw_row.get("order_date")),
                    main_entities=_split_multi_value(raw_row.get("main_entities")),
                    short_summary=str(raw_row.get("short_summary", "")).strip(),
                    summary_source=str(raw_row.get("summary_source", "")).strip(),
                    procedural_type=_clean_optional(raw_row.get("procedural_type")),
                    manifest_status=str(raw_row.get("manifest_status", "")).strip(),
                    manifest_error=_clean_optional(raw_row.get("manifest_error")),
                    ingested=_parse_bool(raw_row.get("ingested")),
                    document_version_id=_parse_int(raw_row.get("document_version_id")),
                    detail_url=_clean_optional(raw_row.get("detail_url")),
                    pdf_url=_clean_optional(raw_row.get("pdf_url")),
                    local_filename=str(raw_row.get("local_filename", "")).strip(),
                )
            )
    return rows


def _load_confusion_pairs(path: Path) -> list[ConfusionPair]:
    rows: list[ConfusionPair] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            rows.append(
                ConfusionPair(
                    record_key_a=str(raw_row.get("record_key_a", "")).strip(),
                    title_a=str(raw_row.get("title_a", "")).strip(),
                    bucket_a=str(raw_row.get("bucket_a", "")).strip(),
                    order_date_a=_parse_date(raw_row.get("order_date_a")),
                    record_key_b=str(raw_row.get("record_key_b", "")).strip(),
                    title_b=str(raw_row.get("title_b", "")).strip(),
                    bucket_b=str(raw_row.get("bucket_b", "")).strip(),
                    order_date_b=_parse_date(raw_row.get("order_date_b")),
                    confusion_type=str(raw_row.get("confusion_type", "")).strip(),
                    reason=str(raw_row.get("reason", "")).strip(),
                )
            )
    return rows


def _load_eval_queries(path: Path) -> list[EvalQueryCase]:
    cases: list[EvalQueryCase] = []
    for row in _load_jsonl(path):
        cases.append(
            EvalQueryCase(
                query=str(row.get("query", "")).strip(),
                expected_route_mode=str(row.get("expected_route_mode", "")).strip(),
                expected_record_key=_clean_optional(row.get("expected_record_key")),
                expected_title=_clean_optional(row.get("expected_title")),
                comparison_allowed=bool(row.get("comparison_allowed", False)),
                notes=str(row.get("notes", "")).strip(),
                reuse_previous_session=bool(row.get("reuse_previous_session", False)),
                session_group=_clean_optional(row.get("session_group")),
            )
        )
    return cases


def _load_wrong_answer_examples(path: Path) -> list[WrongAnswerExample]:
    cases: list[WrongAnswerExample] = []
    for row in _load_jsonl(path):
        cases.append(
            WrongAnswerExample(
                user_query=str(row.get("user_query", "")).strip(),
                expected_record_key=_clean_optional(row.get("expected_record_key")),
                expected_title=_clean_optional(row.get("expected_title")),
                incorrectly_pulled_record_keys=_split_multi_value(
                    row.get("incorrectly_pulled_record_keys")
                ),
                incorrectly_pulled_titles=_split_multi_value(row.get("incorrectly_pulled_titles")),
                observed_answer_status=str(row.get("observed_answer_status", "")).strip(),
                observed_confidence=float(row.get("observed_confidence", 0.0) or 0.0),
                observed_route_mode=str(row.get("observed_route_mode", "")).strip(),
                tool_output=str(row.get("tool_output", "")).strip(),
                what_it_should_have_answered=str(
                    row.get("what_it_should_have_answered", "")
                ).strip(),
            )
        )
    return cases


def _load_entity_aliases(path: Path) -> list[EntityAliasRow]:
    rows: list[EntityAliasRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            rows.append(
                EntityAliasRow(
                    canonical_name=str(raw_row.get("canonical_name", "")).strip(),
                    short_name=_clean_optional(raw_row.get("short_name")),
                    abbreviations=_split_multi_value(raw_row.get("abbreviations")),
                    old_name=_clean_optional(raw_row.get("old_name")),
                    new_name=_clean_optional(raw_row.get("new_name")),
                    related_record_keys=_split_multi_value(raw_row.get("related_record_keys")),
                    related_titles=_split_multi_value(raw_row.get("related_titles")),
                )
            )
    return rows


def _load_strict_answer_rule(path: Path) -> StrictAnswerRule:
    text = path.read_text(encoding="utf-8").strip()
    normalized_text = " ".join(text.lower().split())
    return StrictAnswerRule(
        text=text,
        strict_single_matter_required=(
            "answer from that matter only" in normalized_text
            and "unless the user explicitly asks to compare" in normalized_text
        ),
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _clean_optional(value: object) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _parse_bool(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _parse_int(value: object) -> int | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return int(cleaned)


def _parse_date(value: object) -> date | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return date.fromisoformat(cleaned)


def _split_multi_value(value: object) -> tuple[str, ...]:
    raw = str(value or "").strip()
    if not raw:
        return ()
    parts = [segment.strip() for segment in raw.replace(";", ",").split(",")]
    return tuple(part for part in parts if part)


def _alias_variants(row: EntityAliasRow) -> tuple[str, ...]:
    variants = [row.canonical_name]
    if row.short_name:
        variants.append(row.short_name)
    variants.extend(row.abbreviations)
    if row.old_name:
        variants.append(row.old_name)
    if row.new_name:
        variants.append(row.new_name)
    expanded: list[str] = []
    for value in variants:
        if not value.strip():
            continue
        expanded.append(_normalize_variant(value))
        expanded.extend(
            _normalize_variant(item)
            for item in generate_order_alias_variants(value)
            if item.strip()
        )
    return tuple(dict.fromkeys(item for item in expanded if item))


def _normalize_variant(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.lower()))
