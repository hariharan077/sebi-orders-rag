#!/usr/bin/env python3
"""Build control artifacts for hardening the SEBI Orders RAG system."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection
from app.sebi_orders_rag.ingestion.manifest_loader import discover_manifest_paths, load_manifest

OUTPUT_DIR_NAME = "sebi_control_pack_2026_04_10"
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:]+$")
_PAGE_NOISE_RE = re.compile(r"\bpage\s+\d+\b", re.IGNORECASE)
_DATE_IN_TITLE_RE = re.compile(
    r"(?i)\b(?:dated|date of decision)\s*[:\-]?\s*[0-9./-]+(?:\s*[a-z]+)?"
)
_VERSUS_RE = re.compile(r"\b(?:vs\.?|versus|v\.)\b", re.IGNORECASE)
_FILED_BY_RE = re.compile(r"(?i)\bfiled by\s+(.+)$")
_IN_RESPECT_OF_AND_MATTER_RE = re.compile(
    r"(?i)\bin respect of\s+(.+?)\s+in the matter of\s+(.+)$"
)
_IN_THE_MATTER_RE = re.compile(r"(?i)\bin the matter of\s+(.+)$")
_IN_RESPECT_OF_RE = re.compile(r"(?i)\bin respect of\s+(.+)$")
_OLD_NAME_RE = re.compile(r"(?i)\bformerly known as\s+(.+)$")
_PROPRIETOR_RE = re.compile(r"(?i)\((?:prop(?:rietor)?\.?:?|proprietor:)\s*([^)]+)\)")

LEGAL_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bprivate limited\b"), "Pvt. Ltd."),
    (re.compile(r"(?i)\bpvt\.?\s+ltd\.?\b"), "Pvt. Ltd."),
    (re.compile(r"(?i)\blimited\b"), "Ltd."),
    (re.compile(r"(?i)\bltd\.?\b"), "Ltd."),
    (re.compile(r"(?i)\bhuf\b"), "HUF"),
    (re.compile(r"(?i)\bn\.a\.?\b"), "N.A."),
)

SUMMARY_FALLBACKS = {
    "orders-of-aa-under-rti-act": "RTI appellate order concerning {entity}.",
    "orders-of-ao": "Adjudication order concerning {entity}.",
    "orders-of-chairperson-members": "Chairperson/Whole Time Member order concerning {entity}.",
    "orders-of-corporatisation-demutualisation-scheme": (
        "Corporatisation/demutualisation scheme order concerning {entity}."
    ),
    "orders-of-courts": "Court order concerning {entity}.",
    "orders-of-ed-cgm": "Enforcement order concerning {entity}.",
    "orders-of-sat": "SAT order or appeal concerning {entity}.",
    "orders-of-special-courts": "Special Court judgment or sentencing order concerning {entity}.",
    "orders-under-regulation-30a": "Order under Regulation 30A concerning {entity}.",
    "settlement-orders": "Settlement order concerning {entity}.",
}


@dataclass(frozen=True)
class ParsedNodeText:
    procedural_type: str | None
    major_headings: tuple[str, ...]
    opening_lines: tuple[str, ...]


@dataclass(frozen=True)
class DocumentInfo:
    record_key: str
    title: str
    bucket_name: str
    order_date: str | None
    external_record_id: str | None
    detail_url: str | None
    pdf_url: str | None
    local_filename: str
    manifest_status: str
    manifest_error: str | None
    ingested: bool
    document_version_id: int | None
    procedural_type: str | None
    major_headings: tuple[str, ...]
    opening_lines: tuple[str, ...]
    main_entities: tuple[str, ...]
    short_summary: str
    summary_source: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build index, confusion set, eval cases, alias map, and failure examples."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / OUTPUT_DIR_NAME,
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--skip-live-failures",
        action="store_true",
        help="Skip live current-tool failure mining.",
    )
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    settings = SebiOrdersRagSettings.from_env()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [load_manifest(path) for path in discover_manifest_paths(settings.data_root)]
    manifest_rows = [row for manifest in manifests for row in manifest.rows]
    db_rows = fetch_current_document_rows(settings)

    documents = build_document_index(manifest_rows=manifest_rows, db_rows=db_rows)
    confusion_rows = build_confusion_rows(documents)
    eval_rows = build_eval_queries(documents)
    alias_rows = build_alias_rows(documents)
    failure_rows = (
        []
        if args.skip_live_failures
        else build_failure_rows(settings=settings, documents=documents)
    )

    write_document_index(output_dir / "document_index.csv", documents)
    write_csv(output_dir / "confusion_list.csv", confusion_rows)
    write_jsonl(output_dir / "eval_queries.jsonl", eval_rows)
    write_csv(output_dir / "entity_aliases.csv", alias_rows)
    write_jsonl(output_dir / "wrong_answer_examples.jsonl", failure_rows)
    write_strict_answer_rule(output_dir / "strict_answer_rule.md")
    write_readme(
        output_path=output_dir / "README.md",
        documents=documents,
        confusion_rows=confusion_rows,
        eval_rows=eval_rows,
        alias_rows=alias_rows,
        failure_rows=failure_rows,
    )

    print(f"wrote: {output_dir}")
    print(f"document_index.csv rows: {len(documents)}")
    print(f"confusion_list.csv rows: {len(confusion_rows)}")
    print(f"eval_queries.jsonl rows: {len(eval_rows)}")
    print(f"entity_aliases.csv rows: {len(alias_rows)}")
    print(f"wrong_answer_examples.jsonl rows: {len(failure_rows)}")
    return 0


def fetch_current_document_rows(settings: SebiOrdersRagSettings) -> dict[str, dict[str, Any]]:
    sql = """
        SELECT
            sd.record_key,
            sd.bucket_name,
            sd.external_record_id,
            sd.current_version_id,
            dv.document_version_id,
            dv.order_date::text,
            dv.title,
            dv.detail_url,
            dv.pdf_url,
            dv.local_filename,
            dv.manifest_status,
            dn.node_text
        FROM source_documents sd
        LEFT JOIN document_versions dv
            ON dv.document_version_id = sd.current_version_id
        LEFT JOIN document_nodes dn
            ON dn.document_version_id = dv.document_version_id
    """
    rows: dict[str, dict[str, Any]] = {}
    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            for row in cursor.fetchall():
                rows[str(row[0])] = {
                    "record_key": str(row[0]),
                    "bucket_name": row[1],
                    "external_record_id": row[2],
                    "current_version_id": row[3],
                    "document_version_id": row[4],
                    "order_date": row[5],
                    "title": row[6],
                    "detail_url": row[7],
                    "pdf_url": row[8],
                    "local_filename": row[9],
                    "manifest_status": row[10],
                    "node_text": row[11],
                }
    return rows


def build_document_index(
    *,
    manifest_rows: list[Any],
    db_rows: dict[str, dict[str, Any]],
) -> list[DocumentInfo]:
    documents: list[DocumentInfo] = []
    for row in sorted(
        manifest_rows,
        key=lambda item: (
            item.bucket_name,
            item.order_date.isoformat() if item.order_date else "",
            item.title.lower(),
            item.record_key,
        ),
    ):
        db_row = db_rows.get(row.record_key)
        node = parse_node_text(db_row.get("node_text") if db_row else None)
        title = db_row["title"] if db_row and db_row.get("title") else row.title
        entities = extract_main_entities(title)
        summary, summary_source = build_short_summary(
            title=title,
            bucket_name=row.bucket_name,
            entities=entities,
            node=node,
        )
        documents.append(
            DocumentInfo(
                record_key=row.record_key,
                title=title,
                bucket_name=row.bucket_name,
                order_date=(row.order_date.isoformat() if row.order_date else None),
                external_record_id=row.external_record_id,
                detail_url=row.detail_url,
                pdf_url=row.pdf_url or None,
                local_filename=row.local_filename,
                manifest_status=row.manifest_status,
                manifest_error=row.error,
                ingested=bool(db_row and db_row.get("document_version_id")),
                document_version_id=db_row.get("document_version_id") if db_row else None,
                procedural_type=node.procedural_type,
                major_headings=node.major_headings,
                opening_lines=node.opening_lines,
                main_entities=entities,
                short_summary=summary,
                summary_source=summary_source,
            )
        )
    return documents


def parse_node_text(node_text: str | None) -> ParsedNodeText:
    if not node_text:
        return ParsedNodeText(
            procedural_type=None,
            major_headings=(),
            opening_lines=(),
        )

    procedural_type: str | None = None
    major_headings: tuple[str, ...] = ()
    opening_lines: list[str] = []
    in_opening_lines = False

    for raw_line in node_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Procedural type:"):
            procedural_type = line.partition(":")[2].strip() or None
            in_opening_lines = False
            continue
        if line.startswith("Major headings:"):
            headings = line.partition(":")[2].strip()
            major_headings = tuple(
                heading.strip()
                for heading in headings.split(";")
                if heading.strip()
            )
            in_opening_lines = False
            continue
        if line.startswith("Opening lines:"):
            in_opening_lines = True
            continue
        if in_opening_lines and line.startswith("- "):
            opening_lines.append(clean_line(line[2:]))
            continue
        if in_opening_lines:
            in_opening_lines = False

    return ParsedNodeText(
        procedural_type=procedural_type,
        major_headings=major_headings,
        opening_lines=tuple(opening_lines),
    )


def extract_main_entities(title: str) -> tuple[str, ...]:
    working_title = clean_line(title)
    entities: list[str] = []

    versus_parts = _VERSUS_RE.split(working_title, maxsplit=1)
    if len(versus_parts) == 2:
        entities.extend(clean_entity(part) for part in versus_parts)

    if not entities:
        filed_by_match = _FILED_BY_RE.search(working_title)
        if filed_by_match:
            entities.append(clean_entity(filed_by_match.group(1)))

    if not entities:
        combined_match = _IN_RESPECT_OF_AND_MATTER_RE.search(working_title)
        if combined_match:
            for part in combined_match.groups():
                entities.extend(split_matter_phrase(clean_entity(part)))

    if not entities:
        matter_match = _IN_THE_MATTER_RE.search(working_title)
        if matter_match:
            entities.extend(split_matter_phrase(clean_entity(matter_match.group(1))))

    if not entities:
        respect_match = _IN_RESPECT_OF_RE.search(working_title)
        if respect_match:
            entities.extend(split_matter_phrase(clean_entity(respect_match.group(1))))

    if not entities:
        entities.extend(split_matter_phrase(clean_entity(strip_order_prefix(working_title))))

    expanded: list[str] = []
    for entity in entities:
        if not entity:
            continue
        expanded.append(entity)
        proprietor_match = _PROPRIETOR_RE.search(entity)
        if proprietor_match:
            expanded.append(clean_entity(proprietor_match.group(1)))

    return tuple(dedupe_preserving_order(item for item in expanded if item))


def clean_entity(value: str) -> str:
    entity = clean_line(value)
    entity = _OLD_NAME_RE.sub("", entity).strip()
    entity = re.sub(r"\(\s*[^)]*?(?:writ|appeal|petition|case|cc no|cnr no)[^)]*\)", "", entity, flags=re.IGNORECASE)
    entity = re.sub(r"(?i)\b(the matter of|matter of|order in the matter of|order in respect of)\b", "", entity)
    entity = re.sub(r"(?i)\b(corrigendum to the|corrigendum|summary settlement order|settlement order|confirmatory order|final order|adjudication order|revocation order|exemption order|enquiry proceedings|enquiry order|appeal no\.?\s*[0-9&\s.-]+of\s+\d{4})\b", "", entity)
    entity = entity.strip(" ,.-")
    entity = _TRAILING_PUNCT_RE.sub("", entity).strip()
    return entity


def split_matter_phrase(value: str) -> list[str]:
    phrase = clean_entity(value)
    if not phrase:
        return []

    specific_patterns = (
        re.compile(r"(?i)^alleged insider trading in the scrip of (.+?) by (.+)$"),
        re.compile(r"(?i)^front[- ]running trades of big client\s*[-:]?\s*(.+?) by (.+)$"),
        re.compile(r"(?i)^front running of trades of big client by certain entities of (.+)$"),
        re.compile(r"(?i)^stock recommendation tips in the scrip of (.+)$"),
        re.compile(r"(?i)^inspection of (.+)$"),
        re.compile(r"(?i)^unregistered investment advisory activities by (.+)$"),
        re.compile(r"(?i)^unregistered investment advisory by (.+)$"),
        re.compile(r"(?i)^research analyst(?:\.)?$"),
    )
    for pattern in specific_patterns:
        match = pattern.match(phrase)
        if not match:
            continue
        parts = [clean_entity(group) for group in match.groups() if clean_entity(group)]
        return dedupe_preserving_order(parts) or [phrase]

    by_parts = re.split(r"(?i)\s+by\s+", phrase, maxsplit=1)
    if len(by_parts) == 2 and any(
        marker in by_parts[0].lower()
        for marker in (
            "insider trading",
            "front running",
            "investment advisory",
            "research analyst",
            "scrip of",
            "inspection of",
        )
    ):
        return dedupe_preserving_order(clean_entity(part) for part in by_parts if clean_entity(part))

    return [phrase]


def strip_order_prefix(title: str) -> str:
    cleaned = clean_line(title)
    cleaned = _DATE_IN_TITLE_RE.sub("", cleaned).strip()
    prefixes = (
        "Judgment dated",
        "Sentencing Order dated",
        "Order passed by the Hon’ble",
        "Order passed by the Hon'ble",
        "Order dated",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            if " in " in cleaned:
                cleaned = cleaned.split(" in ", 1)[1].strip()
            break
    return cleaned


def build_short_summary(
    *,
    title: str,
    bucket_name: str,
    entities: tuple[str, ...],
    node: ParsedNodeText,
) -> tuple[str, str]:
    for line in node.opening_lines:
        candidate = normalize_summary_candidate(line, title=title)
        if candidate:
            return candidate, "opening_line"

    if node.procedural_type:
        entity = entities[0] if entities else strip_order_prefix(title)
        return (
            shorten(
                f"{node.procedural_type.capitalize()} concerning {entity}."
            ),
            "template",
        )

    entity = entities[0] if entities else strip_order_prefix(title)
    template = SUMMARY_FALLBACKS.get(bucket_name, "Order concerning {entity}.")
    return shorten(template.format(entity=entity)), "template"


def normalize_summary_candidate(line: str, *, title: str) -> str | None:
    cleaned = clean_line(line).strip("_- ")
    if len(cleaned) < 40:
        return None
    lowered = cleaned.lower()
    title_lower = clean_line(title).lower()
    if lowered == title_lower:
        return None
    if lowered.startswith("(under the ") and lowered.endswith(")"):
        return None
    if lowered.startswith(
        (
            "under section",
            "under sections",
            "under sub-sections",
            "under sub sections",
            "order under section",
            "before the",
            "dated :",
            "date of decision",
            "mumbai, the",
            "sl. no.",
            "noticee no.",
        )
    ):
        return None
    if lowered.endswith((": appellant", ": respondent")):
        return None
    if _PAGE_NOISE_RE.search(lowered):
        return None
    if not any(char.isalpha() for char in cleaned):
        return None
    if sum(char.isupper() for char in cleaned) > max(8, len(cleaned) // 2):
        return None
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    return shorten(cleaned)


def clean_line(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.replace("\u00a0", " ")).strip()


def shorten(value: str, *, limit: int = 240) -> str:
    cleaned = clean_line(value)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def dedupe_preserving_order(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def find_doc(documents: list[DocumentInfo], record_key: str) -> DocumentInfo:
    for document in documents:
        if document.record_key == record_key:
            return document
    raise KeyError(f"unknown record_key: {record_key}")


def build_confusion_rows(documents: list[DocumentInfo]) -> list[dict[str, Any]]:
    curated_pairs = [
        (
            "derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1",
            "external:98714",
            "unrelated_named_matter_contamination",
            "Named IPO-style query for Vishvaraj can drift into the Varyaa order because both include VCL/IPO-proceeds language.",
        ),
        (
            "derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e",
            "external:100209",
            "same_matter_original_vs_corrigendum",
            "Original Hardcastle exemption order and its corrigendum are same-matter siblings with different operative text.",
        ),
        (
            "derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46",
            "derived:458ed16723c4cbaf6336c5ae33b5c1d541e717bb30d3bd51a3d583cb70382d00",
            "same_matter_original_vs_corrigendum",
            "Pacheli confirmatory order and its corrigendum differ only by correction, so entity-only lookups can merge them.",
        ),
        (
            "derived:c462f86f79c03e2904ad5041ecb780788b245f8c34a228dd3743f4a2b3630615",
            "derived:8eda45e08355e2932a9b5bfae163546a5241002a83f1fee7af1bdc40437afb5b",
            "same_group_different_target_company",
            "Both Refex exemption orders were issued on the same date and share the same proposed acquirer, but target companies differ.",
        ),
        (
            "external:30189",
            "external:30161",
            "similar_title_different_facts",
            "Prime Broking Company (India) Limited and Prime Broking Company Limited are distinct SAT matters with nearly identical titles.",
        ),
        (
            "external:30222",
            "external:30223",
            "duplicate_title_multiple_records",
            "Two SAT rows share the exact same displayed title for Prime Broking Company (India) Limited.",
        ),
        (
            "external:93689",
            "external:95606",
            "repeated_generic_title",
            "Four Regulation 30A orders use the exact title 'certain Investment Advisers' across different dates.",
        ),
        (
            "external:95606",
            "external:96551",
            "repeated_generic_title",
            "Different dates but identical title and bucket make these easy to blend without date locking.",
        ),
        (
            "external:96551",
            "external:97984",
            "repeated_generic_title",
            "Same generic class-order title, different record/date.",
        ),
        (
            "external:100486",
            "external:100429",
            "same_bucket_same_order_type",
            "JP Morgan and DDP-Standard Chartered are both settlement orders for large financial institutions and share similar operative language.",
        ),
        (
            "external:100535",
            "external:100648",
            "same_theme_different_front_running_matter",
            "Both concern front running of big-client trades but involve different entities and factual patterns.",
        ),
        (
            "derived:551259f200f62065e076213d712072bed57ea9c610044f61b542791220f62c09",
            "derived:3b2168897812899ee830f8977c450f8744ebba7be33de1151fa07dbc1a5ccad3",
            "same_company_different_agel_insider_trading_matter",
            "Both orders concern alleged insider trading in Adani Green Energy Limited but different noticee groups.",
        ),
        (
            "external:88411",
            "external:88412",
            "judgment_vs_sentencing",
            "Kisley Plantation has separate judgment and sentencing documents for the same prosecution.",
        ),
        (
            "external:87947",
            "external:87948",
            "judgment_vs_sentencing",
            "Neelgiri Forest has separate judgment and sentencing documents for the same prosecution.",
        ),
        (
            "external:30026",
            "external:30036",
            "same_party_multiple_sat_matters",
            "Suresh Bharrat appears in two SAT records on different dates.",
        ),
        (
            "external:30023",
            "external:30159",
            "same_party_multiple_sat_matters",
            "SVS Securities Pvt. Ltd. appears in two SAT records with nearly identical party naming.",
        ),
        (
            "external:100722",
            "external:100482",
            "similar_person_names_rti",
            "Rajat Kumar and Rajat Galav are both RTI appeals with identical template structure.",
        ),
        (
            "external:100694",
            "external:100722",
            "same_bucket_same_template_rti",
            "Rajendra Prasad and Rajat Kumar RTI appeals share the same boilerplate and decision framing.",
        ),
        (
            "external:100650",
            "external:100656",
            "same_family_name_same_matter_cluster",
            "Rajeev Chandak HUF and Rashmi Chandak are separate AO respondents in the same illiquid stock options matter.",
        ),
        (
            "external:100650",
            "external:100633",
            "same_matter_cluster",
            "Rajeev Chandak HUF and Rekha M Agarwal are separate AO respondents within the same BSE illiquid options cluster.",
        ),
        (
            "external:100605",
            "external:100609",
            "same_generic_matter_cluster",
            "ITI Securities Broking Limited and ATS Share Brokers Private Limited are distinct respondents in the TradeTron/algo-platform cluster.",
        ),
        (
            "external:100601",
            "external:100611",
            "same_generic_matter_cluster",
            "Ganganagar Commodity Limited and First Global Stockbroking Private Limited are distinct respondents in the same TradeTron cluster.",
        ),
        (
            "external:100851",
            "external:100579",
            "similar_advisory_compliance_theme",
            "Wealthmax and Elite Investment Advisory are different advisory-compliance orders that can blend under generic advisory queries.",
        ),
        (
            "external:100464",
            "external:100465",
            "research_theme_sibling_orders",
            "CapitalVia Global Research and Seema Jain are separate research-related enforcement matters issued on the same date.",
        ),
        (
            "external:93065",
            "external:100465",
            "generic_research_analyst_title_vs_named_ra_order",
            "The class order for certain Research Analysts can blend with named research-analyst enforcement records.",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for left_key, right_key, confusion_type, reason in curated_pairs:
        left = find_doc(documents, left_key)
        right = find_doc(documents, right_key)
        rows.append(
            {
                "record_key_a": left.record_key,
                "title_a": left.title,
                "bucket_a": left.bucket_name,
                "order_date_a": left.order_date or "",
                "record_key_b": right.record_key,
                "title_b": right.title,
                "bucket_b": right.bucket_name,
                "order_date_b": right.order_date or "",
                "confusion_type": confusion_type,
                "reason": reason,
            }
        )
    return rows


def build_eval_queries(documents: list[DocumentInfo]) -> list[dict[str, Any]]:
    by_key = {doc.record_key: doc for doc in documents}
    rows: list[dict[str, Any]] = []

    direct_queries = [
        "What is a settlement order?",
        "What is an adjudication order under the SEBI Act?",
        "Explain what a corrigendum does in a SEBI order.",
        "What is an RTI appellate order?",
        "What is an exemption order under the takeover regulations?",
        "What is a SAT order?",
        "What is an ex-parte interim order?",
        "Explain orders issued under Regulation 30A.",
    ]
    for query in direct_queries:
        rows.append(
            {
                "query": query,
                "expected_route_mode": "direct_llm",
                "expected_record_key": "",
                "expected_title": "",
                "comparison_allowed": False,
                "notes": "General explanatory query; no record lock expected.",
            }
        )

    representative_keys = [
        "external:100722",
        "external:100846",
        "external:100685",
        "external:10784",
        "external:100736",
        "external:100770",
        "external:30222",
        "external:99769",
        "external:100851",
        "external:100669",
        "external:100486",
        "derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1",
        "derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e",
        "derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46",
        "external:100648",
        "external:88411",
        "external:87947",
        "external:93065",
        "external:30161",
        "external:100605",
    ]
    for key in representative_keys:
        doc = by_key[key]
        rows.append(
            {
                "query": doc.title,
                "expected_route_mode": "exact_lookup",
                "expected_record_key": doc.record_key,
                "expected_title": doc.title,
                "comparison_allowed": False,
                "notes": "Exact title lookup should lock to one record.",
            }
        )
        hierarchical_query = build_hierarchical_query(doc)
        rows.append(
            {
                "query": hierarchical_query,
                "expected_route_mode": "hierarchical_rag",
                "expected_record_key": doc.record_key,
                "expected_title": doc.title,
                "comparison_allowed": False,
                "notes": "Named substantive query should stay within the named matter.",
            }
        )

    memory_groups = [
        ("external:100486", "What was the settlement amount?"),
        ("external:100669", "What did SEBI finally direct?"),
        ("derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e", "What exemption was granted?"),
        ("external:100722", "What did the appellate authority decide?"),
        ("external:88411", "What sentence was imposed?"),
    ]
    for group_index, (record_key, follow_up_query) in enumerate(memory_groups, start=1):
        doc = by_key[record_key]
        rows.append(
            {
                "query": doc.title,
                "expected_route_mode": "exact_lookup",
                "expected_record_key": doc.record_key,
                "expected_title": doc.title,
                "comparison_allowed": False,
                "session_group": f"group_{group_index}",
                "notes": "Session-seeding exact lookup.",
            }
        )
        rows.append(
            {
                "query": follow_up_query,
                "expected_route_mode": "memory_scoped_rag",
                "expected_record_key": doc.record_key,
                "expected_title": doc.title,
                "comparison_allowed": False,
                "session_group": f"group_{group_index}",
                "reuse_previous_session": True,
                "notes": "Follow-up should stay anchored to prior record.",
            }
        )

    abstain_queries = [
        "What was the settlement amount in the Imaginary Capital Limited settlement order?",
        "Tell me more about Appeal No. 9999 of 2026 filed by Nonexistent Person.",
        "What exemption did SEBI grant in the Fake Metals Limited matter?",
        "Tell me more about the RTI appeal filed by Prime Broking Company India Limited.",
        "What was the settlement amount in the Vishvaraj Environment Limited matter?",
        "What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
        "Tell me more about Adani Green Energy Limited.",
        "Tell me more about Cochin Stock Exchange Limited.",
    ]
    for query in abstain_queries:
        rows.append(
            {
                "query": query,
                "expected_route_mode": "abstain",
                "expected_record_key": "",
                "expected_title": "",
                "comparison_allowed": False,
                "notes": "The system should abstain rather than fuse loosely related matters.",
            }
        )

    comparison_rows = [
        (
            "Compare the Refex Industries Limited and Refex Renewables & Infrastructure Limited exemption orders.",
            [
                "derived:c462f86f79c03e2904ad5041ecb780788b245f8c34a228dd3743f4a2b3630615",
                "derived:8eda45e08355e2932a9b5bfae163546a5241002a83f1fee7af1bdc40437afb5b",
            ],
        ),
        (
            "Compare the JP Morgan Chase Bank N.A. and DDP- Standard Chartered Bank settlement orders.",
            ["external:100486", "external:100429"],
        ),
        (
            "Compare the Kisley Plantation Limited judgment and sentencing order.",
            ["external:88411", "external:88412"],
        ),
        (
            "Compare the Neelgiri Forest Ltd judgment and sentencing order.",
            ["external:87947", "external:87948"],
        ),
        (
            "Compare the Chaturvedi Group and Sarvottam Securities front-running orders.",
            ["external:100648", "external:100535"],
        ),
    ]
    for query, keys in comparison_rows:
        rows.append(
            {
                "query": query,
                "expected_route_mode": "hierarchical_rag",
                "expected_record_key": ";".join(keys),
                "expected_title": "; ".join(by_key[key].title for key in keys),
                "comparison_allowed": True,
                "notes": "Explicit comparison request permits multi-record grounding.",
            }
        )

    return rows


def build_hierarchical_query(doc: DocumentInfo) -> str:
    entity = doc.main_entities[0] if doc.main_entities else strip_order_prefix(doc.title)
    if doc.bucket_name == "settlement-orders":
        return f"What did SEBI finally direct for {entity}?"
    if doc.bucket_name == "orders-of-aa-under-rti-act":
        return f"What did the appellate authority decide for {entity}?"
    if doc.bucket_name == "orders-of-ao":
        return f"What was SEBI's finding against {entity}?"
    if doc.bucket_name == "orders-of-chairperson-members":
        return f"What did SEBI order for {entity}?"
    if doc.bucket_name == "orders-of-corporatisation-demutualisation-scheme":
        return f"What scheme approval or direction was issued for {entity}?"
    if doc.bucket_name == "orders-of-courts":
        return f"What did the court decide in the matter concerning {entity}?"
    if doc.bucket_name == "orders-of-ed-cgm":
        return f"What action did SEBI take against {entity}?"
    if doc.bucket_name == "orders-of-sat":
        return f"What happened in the SAT matter concerning {entity}?"
    if doc.bucket_name == "orders-of-special-courts":
        return f"What did the Special Court hold concerning {entity}?"
    if doc.bucket_name == "orders-under-regulation-30a":
        return f"What non-compliance did SEBI identify for {entity}?"
    return f"What happened in the matter concerning {entity}?"


def build_alias_rows(documents: list[DocumentInfo]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc in documents:
        for entity in doc.main_entities:
            full_name = entity
            old_name, primary_name = split_old_name(full_name)
            short_name = strip_legal_suffix(primary_name)
            abbreviations = build_abbreviation_variants(primary_name)
            key = (primary_name.lower(), doc.record_key)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "canonical_name": primary_name,
                    "short_name": short_name,
                    "abbreviations": "; ".join(abbreviations),
                    "old_name": old_name or "",
                    "new_name": primary_name if old_name else "",
                    "related_record_keys": doc.record_key,
                    "related_titles": doc.title,
                }
            )
    rows.sort(key=lambda row: (row["canonical_name"].lower(), row["related_record_keys"]))
    return rows


def split_old_name(full_name: str) -> tuple[str | None, str]:
    match = _OLD_NAME_RE.search(full_name)
    if not match:
        return None, full_name
    old_name = clean_entity(match.group(1))
    new_name = clean_entity(full_name[: match.start()].strip(" ,.-"))
    return (old_name or None, new_name or full_name)


def strip_legal_suffix(name: str) -> str:
    shortened = name
    for pattern, _replacement in LEGAL_SUFFIX_PATTERNS:
        shortened = pattern.sub("", shortened)
    shortened = clean_line(shortened).strip(" ,.-")
    return shortened or name


def build_abbreviation_variants(name: str) -> list[str]:
    variants: list[str] = []
    for pattern, replacement in LEGAL_SUFFIX_PATTERNS:
        if pattern.search(name):
            variants.append(clean_line(pattern.sub(replacement, name)))
    proprietor_match = _PROPRIETOR_RE.search(name)
    if proprietor_match:
        variants.append(clean_entity(proprietor_match.group(1)))
    return dedupe_preserving_order(variants)


def build_failure_rows(
    *,
    settings: SebiOrdersRagSettings,
    documents: list[DocumentInfo],
) -> list[dict[str, Any]]:
    queries = [
        {
            "query": "Tell me more about the IPO of Vishvaraj Environment Limited",
            "expected_record_key": "derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1",
            "expected_behavior": "Answer only from the Vishvaraj matter; do not import Varyaa facts.",
        },
        {
            "query": "Tell me more about Hardcastle and Waud Manufacturing Ltd.",
            "expected_record_key": "derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e",
            "expected_behavior": "Stay on the base Hardcastle exemption order unless the user asks for the corrigendum.",
        },
        {
            "query": "Tell me more about Prime Broking Company India Limited",
            "expected_record_key": "external:30189",
            "expected_behavior": "Lock to the '(India) Limited' matter, not the distinct 'Prime Broking Company Limited' record.",
        },
        {
            "query": "Tell me more about Certain Investment Advisers",
            "expected_record_key": "",
            "expected_behavior": "Ambiguous class-order title; ask for date/title or abstain instead of merging records.",
        },
        {
            "query": "Tell me more about Adani Green Energy Limited by Pranav Adani",
            "expected_record_key": "derived:551259f200f62065e076213d712072bed57ea9c610044f61b542791220f62c09",
            "expected_behavior": "Stay within the Pranav Adani AGEL order; do not merge the Vinod Bahety order.",
        },
        {
            "query": "Tell me more about front running trades of big client Sarvottam Securities Private Limited",
            "expected_record_key": "external:100535",
            "expected_behavior": "Stay within the Sarvottam matter; do not blend the Chaturvedi Group order.",
        },
        {
            "query": "Tell me more about Kisley Plantation Limited",
            "expected_record_key": "external:88411",
            "expected_behavior": "Judgment and sentencing are separate records; a named query should not silently merge them.",
        },
        {
            "query": "Tell me more about Neelgiri Forest Ltd",
            "expected_record_key": "external:87947",
            "expected_behavior": "Judgment and sentencing are separate records; a named query should not silently merge them.",
        },
        {
            "query": "Tell me more about Suresh Bharrat",
            "expected_record_key": "external:30026",
            "expected_behavior": "Stay on the Suresh Bharrat SAT matter or abstain; do not answer from the unrelated Suresh G. Motwani prosecution.",
        },
        {
            "query": "Tell me more about the RTI appeal filed by Prime Broking Company India Limited.",
            "expected_record_key": "",
            "expected_behavior": "No RTI appeal exists for this entity in the corpus; abstain instead of surfacing Prime Broking SAT matters.",
        },
        {
            "query": "What exemption did SEBI grant in the JP Morgan Chase Bank N.A. matter?",
            "expected_record_key": "external:100486",
            "expected_behavior": "This is a settlement order, not an exemption order; answer should abstain or explicitly say no exemption order is in scope.",
        },
        {
            "query": "What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
            "expected_record_key": "derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46",
            "expected_behavior": "The answer should not relabel preferential allotment financing as IPO proceeds.",
        },
    ]

    by_key = {doc.record_key: doc for doc in documents}
    failure_rows: list[dict[str, Any]] = []
    with get_connection(settings) as connection:
        service = AdaptiveRagAnswerService(settings=settings, connection=connection)
        for candidate in queries:
            payload = service.answer_query(query=candidate["query"])
            connection.commit()
            expected_record_key = str(candidate["expected_record_key"]).strip()
            expected_doc = by_key.get(expected_record_key) if expected_record_key else None
            actual_keys = dedupe_preserving_order(payload.active_record_keys)
            wrong_keys = [key for key in actual_keys if key != expected_record_key]
            if payload.answer_status == "abstained" and not wrong_keys:
                continue
            failure_rows.append(
                {
                    "user_query": candidate["query"],
                    "tool_output": payload.answer_text,
                    "observed_route_mode": payload.route_mode,
                    "observed_answer_status": payload.answer_status,
                    "observed_confidence": payload.confidence,
                    "expected_record_key": expected_record_key,
                    "expected_title": expected_doc.title if expected_doc else "",
                    "what_it_should_have_answered": build_expected_behavior_text(
                        expected_doc=expected_doc,
                        expectation=str(candidate["expected_behavior"]),
                    ),
                    "incorrectly_pulled_record_keys": "; ".join(wrong_keys),
                    "incorrectly_pulled_titles": "; ".join(
                        by_key[key].title for key in wrong_keys if key in by_key
                    ),
                }
            )
    return failure_rows


def build_expected_behavior_text(
    *,
    expected_doc: DocumentInfo | None,
    expectation: str,
) -> str:
    if expected_doc is None:
        return expectation
    return (
        f"{expectation} Expected anchor: {expected_doc.title} "
        f"({expected_doc.record_key}). Summary anchor: {expected_doc.short_summary}"
    )


def write_document_index(path: Path, documents: list[DocumentInfo]) -> None:
    rows = [
        {
            "record_key": doc.record_key,
            "exact_title": doc.title,
            "bucket_category": doc.bucket_name,
            "order_date": doc.order_date or "",
            "main_entities": "; ".join(doc.main_entities),
            "short_summary": doc.short_summary,
            "summary_source": doc.summary_source,
            "procedural_type": doc.procedural_type or "",
            "manifest_status": doc.manifest_status,
            "manifest_error": doc.manifest_error or "",
            "ingested": doc.ingested,
            "document_version_id": doc.document_version_id or "",
            "detail_url": doc.detail_url or "",
            "pdf_url": doc.pdf_url or "",
            "local_filename": doc.local_filename,
        }
        for doc in documents
    ]
    write_csv(path, rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_strict_answer_rule(path: Path) -> None:
    text = """# Strict Answer Rule

If the query names a specific matter, entity, or title, answer from that matter only unless the user explicitly asks to compare.

Operational implications:

1. Apply an exact entity/title lock before broader retrieval.
2. Do not merge facts across record keys for named queries unless the query is explicitly comparative.
3. If multiple unrelated record keys appear in a named-query draft answer, treat it as likely contamination and regenerate or abstain.
4. If no grounded support exists inside the locked matter, abstain rather than borrowing thematically similar facts from another matter.
"""
    path.write_text(text, encoding="utf-8")


def write_readme(
    *,
    output_path: Path,
    documents: list[DocumentInfo],
    confusion_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    alias_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    downloaded = sum(1 for doc in documents if doc.manifest_status == "downloaded")
    ingested = sum(1 for doc in documents if doc.ingested)
    text = f"""# SEBI Control Pack

Generated on 2026-04-10 for the 235-row SEBI Orders corpus.

## Included artifacts

- `document_index.csv`: compact per-document index with record key, title, bucket, date, entities, and summary.
- `confusion_list.csv`: curated easy-to-confuse record pairs for retrieval hardening.
- `eval_queries.jsonl`: broad eval set covering direct LLM, exact lookup, hierarchical RAG, memory-scoped RAG, abstain, and explicit comparison.
- `entity_aliases.csv`: canonical entity names with short-name and suffix variants.
- `wrong_answer_examples.jsonl`: current-tool bad outputs useful for regression testing.
- `strict_answer_rule.md`: named-matter grounding rule to adopt explicitly.

## Corpus notes

- Manifest rows: {len(documents)}
- Downloaded rows: {downloaded}
- Ingested rows with current DB versions: {ingested}
- Missing/not-downloaded rows: {len(documents) - downloaded}

## Counts

- Confusion examples: {len(confusion_rows)}
- Eval queries: {len(eval_rows)}
- Alias rows: {len(alias_rows)}
- Wrong-answer examples: {len(failure_rows)}
"""
    output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
