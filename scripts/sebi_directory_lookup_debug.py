#!/usr/bin/env python3
"""Debug one structured SEBI current-info lookup against the local Postgres dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.directory_data.service import DirectoryReferenceQueryService, _classify_query
from app.sebi_orders_rag.repositories.directory import DirectoryRepository
from app.sebi_orders_rag.schemas import ChatSessionStateRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect structured-directory query normalization and match diagnostics.",
    )
    parser.add_argument("query", help="The user query to inspect.")
    parser.add_argument(
        "--follow-up-family",
        choices=("office_contact",),
        help="Optional current-info family to inject into session state.",
    )
    parser.add_argument(
        "--follow-up-query",
        default=None,
        help="Optional prior query text when simulating a follow-up context.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = SebiOrdersRagSettings.from_env(data_root_override=repo_root)
    session_state = _build_session_state(
        family=args.follow_up_family,
        prior_query=args.follow_up_query,
    )
    intent = _classify_query(args.query, session_state=session_state)

    print(f"query: {args.query}")
    print(f"detected query family: {intent.lookup_type}")
    print(f"normalized query: {intent.normalized_query}")
    print(f"extracted city: {intent.extracted_city or '-'}")
    print(f"extracted person: {intent.person_name or '-'}")
    print(f"designation hint: {intent.designation_hint or '-'}")
    print(f"role tokens: {', '.join(intent.role_tokens) if intent.role_tokens else '-'}")
    print(f"office tokens: {', '.join(intent.office_tokens) if intent.office_tokens else '-'}")
    print(f"office follow-up: {intent.is_follow_up}")

    with get_connection(settings) as connection:
        initialize_phase4_schema(connection, settings)
        repository = DirectoryRepository(connection)
        service = DirectoryReferenceQueryService(
            dataset_loader=repository.load_active_dataset,
            provider_name="structured_directory",
        )
        result = service.lookup(query=args.query, session_state=session_state)

    debug = dict(result.debug)
    print(f"answer status: {result.answer_status}")
    print(f"lookup type: {result.lookup_type}")
    print(f"answer path: {debug.get('answer_path') or '-'}")
    print(f"fallback reason: {debug.get('fallback_reason') or '-'}")
    print(f"matched people rows: {debug.get('matched_people_rows_count', 0)}")
    print(f"matched office rows: {debug.get('matched_office_rows_count', 0)}")
    print(f"matched board rows: {debug.get('matched_board_rows_count', 0)}")
    print(f"matched people refs: {_render_refs(debug.get('matched_people'))}")
    print(f"matched office refs: {_render_refs(debug.get('matched_offices'))}")
    print(f"matched board refs: {_render_refs(debug.get('matched_board'))}")
    print("answer text:")
    print(result.answer_text)
    print("debug payload:")
    print(json.dumps(debug, indent=2, sort_keys=True))


def _build_session_state(*, family: str | None, prior_query: str | None) -> ChatSessionStateRecord | None:
    if family is None:
        return None
    return ChatSessionStateRecord(
        session_id=uuid4(),
        current_lookup_family=family,
        current_lookup_query=prior_query,
    )


def _render_refs(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "-"
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "-"))
        row_ids = item.get("row_ids") or []
        parts.append(f"{name} ({', '.join(str(row_id) for row_id in row_ids) or 'no ids'})")
    return "; ".join(parts) if parts else "-"


if __name__ == "__main__":
    main()
