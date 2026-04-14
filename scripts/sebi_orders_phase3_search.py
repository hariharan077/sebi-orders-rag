#!/usr/bin/env python3
"""CLI entrypoint for SEBI Orders RAG Phase 3 search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection
from app.sebi_orders_rag.exceptions import ConfigurationError
from app.sebi_orders_rag.logging_utils import configure_logging
from app.sebi_orders_rag.retrieval.hierarchical_search import HierarchicalSearchService
from app.sebi_orders_rag.schemas import MetadataFilterInput


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 3 search CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run hierarchical search over SEBI Orders Phase 3 retrieval layers.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query text.",
    )
    parser.add_argument(
        "--record-key",
        help="Optional source_documents.record_key filter.",
    )
    parser.add_argument(
        "--bucket-name",
        help="Optional source_documents.bucket_name filter.",
    )
    parser.add_argument(
        "--top-k-docs",
        type=int,
        help="Override SEBI_ORDERS_RAG_RETRIEVAL_TOP_K_DOCS.",
    )
    parser.add_argument(
        "--top-k-sections",
        type=int,
        help="Override SEBI_ORDERS_RAG_RETRIEVAL_TOP_K_SECTIONS.",
    )
    parser.add_argument(
        "--top-k-chunks",
        type=int,
        help="Override SEBI_ORDERS_RAG_RETRIEVAL_TOP_K_CHUNKS.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
        settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
        configure_logging(settings.log_level)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        with get_connection(settings) as connection:
            service = HierarchicalSearchService(settings=settings, connection=connection)
            result = service.search(
                query=args.query,
                filters=MetadataFilterInput(
                    record_key=args.record_key,
                    bucket_name=args.bucket_name,
                ),
                top_k_docs=args.top_k_docs,
                top_k_sections=args.top_k_sections,
                top_k_chunks=args.top_k_chunks,
            )
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Query: {args.query}")
    print(
        "Detected query intent: "
        f"{result.query_intent.intent.value}"
        f"{_matched_terms_suffix(result.query_intent.matched_terms)}"
    )
    if result.query_intent.settlement_terms:
        print(
            "Settlement detection: "
            f"focused={str(result.query_intent.settlement_focused).lower()} "
            f"terms={', '.join(result.query_intent.settlement_terms)}"
        )
    if result.query_intent.entity_terms:
        print(f"Entity terms: {', '.join(result.query_intent.entity_terms)}")
    print("")
    _print_document_hits(result.documents)
    print("")
    _print_section_hits(result.sections)
    print("")
    _print_chunk_hits(result.chunks)
    return 0


def _print_document_hits(hits: tuple[object, ...]) -> None:
    print("Top document hits")
    if not hits:
        print("  (no document hits)")
        return
    for index, hit in enumerate(hits, start=1):
        print(
            "  "
            f"{index}. final={hit.score.final_score:.6f} "
            f"record_key={hit.record_key} bucket={hit.bucket_name} "
            f"document_version_id={hit.document_version_id}"
        )
        print(f"     title={hit.title}")
        if hit.order_date is not None:
            print(f"     order_date={hit.order_date.isoformat()}")
        if hit.external_record_id:
            print(f"     external_record_id={hit.external_record_id}")
        print(
            "     "
            f"base={hit.score.base_score:.6f} "
            f"bucket_adjustment={hit.score.bucket_adjustment:.3f} "
            f"query_alignment={hit.score.query_alignment_adjustment:.3f} "
            f"lexical={hit.score.lexical_score:.6f} "
            f"vector={hit.score.vector_score:.6f} "
            f"parent={hit.score.parent_score:.6f} "
            f"fts={hit.score.fts_score:.6f} "
            f"trigram={hit.score.trigram_score:.6f}"
        )


def _print_section_hits(hits: tuple[object, ...]) -> None:
    print("Top section hits")
    if not hits:
        print("  (no section hits)")
        return
    for index, hit in enumerate(hits, start=1):
        print(
            "  "
            f"{index}. final={hit.score.final_score:.6f} "
            f"record_key={hit.record_key} section_type={hit.section_type} "
            f"pages={hit.page_start}-{hit.page_end}"
        )
        print(
            "     "
            f"document_version_id={hit.document_version_id} "
            f"section_key={hit.section_key}"
        )
        print(
            f"     title={hit.section_title or hit.heading_path or hit.title}"
        )
        if hit.heading_path:
            print(f"     heading_path={hit.heading_path}")
        print(
            "     "
            f"base={hit.score.base_score:.6f} "
            f"bucket_adjustment={hit.score.bucket_adjustment:.3f} "
            f"lexical={hit.score.lexical_score:.6f} "
            f"vector={hit.score.vector_score:.6f} "
            f"parent={hit.score.parent_score:.6f}"
        )
        print(
            "     "
            f"query_alignment={hit.score.query_alignment_adjustment:.3f} "
            f"section_prior={hit.score.section_prior:.3f} "
            f"query_adjustment={hit.score.query_intent_adjustment:.3f} "
            f"content_adjustment={hit.score.content_adjustment:.3f} "
            f"diversity_adjustment={hit.score.diversity_adjustment:.6f} "
            f"final={hit.score.final_score:.6f}"
        )


def _print_chunk_hits(hits: tuple[object, ...]) -> None:
    print("Top chunk hits")
    if not hits:
        print("  (no chunk hits)")
        return
    for index, hit in enumerate(hits, start=1):
        preview = _preview(hit.chunk_text)
        print(
            "  "
            f"{index}. final={hit.score.final_score:.6f} "
            f"record_key={hit.record_key} section_type={hit.section_type} "
            f"pages={hit.page_start}-{hit.page_end}"
        )
        print(
            "     "
            f"document_version_id={hit.document_version_id} "
            f"chunk_id={hit.chunk_id} "
            f"section_key={hit.section_key or '-'}"
        )
        if hit.section_title:
            print(f"     section_title={hit.section_title}")
        print(
            "     "
            f"base={hit.score.base_score:.6f} "
            f"bucket_adjustment={hit.score.bucket_adjustment:.3f} "
            f"lexical={hit.score.lexical_score:.6f} "
            f"vector={hit.score.vector_score:.6f} "
            f"parent={hit.score.parent_score:.6f}"
        )
        print(
            "     "
            f"query_alignment={hit.score.query_alignment_adjustment:.3f} "
            f"section_prior={hit.score.section_prior:.3f} "
            f"query_adjustment={hit.score.query_intent_adjustment:.3f} "
            f"content_adjustment={hit.score.content_adjustment:.3f} "
            f"diversity_adjustment={hit.score.diversity_adjustment:.6f} "
            f"final={hit.score.final_score:.6f}"
        )
        print(f"     preview={preview}")


def _preview(text: str, limit: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _matched_terms_suffix(matched_terms: tuple[str, ...]) -> str:
    if not matched_terms:
        return ""
    return f" (matched: {', '.join(matched_terms)})"


if __name__ == "__main__":
    raise SystemExit(main())
