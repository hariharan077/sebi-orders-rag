#!/usr/bin/env python3
"""CLI entrypoint for SEBI Orders RAG Phase 4 chat."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.exceptions import ConfigurationError
from app.sebi_orders_rag.logging_utils import configure_logging
from app.sebi_orders_rag.schemas import ChatAnswerPayload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an adaptive Phase 4 SEBI Orders chat query.",
    )
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument("--session-id", help="Optional existing session UUID.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.",
    )
    parser.add_argument(
        "--control-pack-root",
        type=Path,
        help="Optional override for SEBI_ORDERS_RAG_CONTROL_PACK_ROOT.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
        settings = SebiOrdersRagSettings.from_env(
            data_root_override=args.data_root,
            control_pack_root_override=args.control_pack_root,
        )
        configure_logging(settings.log_level)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        with get_connection(settings) as connection:
            initialize_phase4_schema(connection, settings)
            connection.commit()
            service = AdaptiveRagAnswerService(settings=settings, connection=connection)
            payload = service.answer_query(
                query=args.query,
                session_id=UUID(args.session_id) if args.session_id else None,
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_payload_to_dict(payload), indent=2))
        return 0

    print(f"session_id: {payload.session_id}")
    print(f"route_mode: {payload.route_mode}")
    print(f"query_intent: {payload.query_intent}")
    print(f"confidence: {payload.confidence:.4f}")
    print(f"answer_status: {payload.answer_status}")
    print("")
    print(payload.answer_text)
    if payload.clarification_candidates:
        print("")
        print("clarification_candidates:")
        for candidate in payload.clarification_candidates:
            print(
                f"  {candidate.candidate_index}. {candidate.title}"
                f" | date={candidate.order_date.isoformat() if candidate.order_date else '-'}"
                f" | bucket={candidate.bucket_name or '-'}"
                f" | record_key={candidate.record_key or '-'}"
            )
    if payload.citations:
        print("")
        print("citations:")
        for citation in payload.citations:
            print(
                f"  [{citation.citation_number}] {citation.record_key} "
                f"pp. {citation.page_start}-{citation.page_end} "
                f"{citation.section_type} chunk_id={citation.chunk_id}"
            )
    return 0


def _payload_to_dict(payload: ChatAnswerPayload) -> dict[str, object]:
    return {
        "session_id": str(payload.session_id),
        "route_mode": payload.route_mode,
        "query_intent": payload.query_intent,
        "answer_text": payload.answer_text,
        "confidence": payload.confidence,
        "answer_status": payload.answer_status,
        "clarification_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.candidate_index,
                "candidate_type": candidate.candidate_type,
                "title": candidate.title,
                "record_key": candidate.record_key,
                "bucket_name": candidate.bucket_name,
                "order_date": (
                    candidate.order_date.isoformat()
                    if candidate.order_date is not None
                    else None
                ),
                "document_version_id": candidate.document_version_id,
                "descriptor": candidate.descriptor,
            }
            for candidate in payload.clarification_candidates
        ],
        "citations": [citation.__dict__ for citation in payload.citations],
        "retrieved_chunk_ids": list(payload.retrieved_chunk_ids),
        "active_record_keys": list(payload.active_record_keys),
        "debug": payload.debug,
    }


if __name__ == "__main__":
    raise SystemExit(main())
