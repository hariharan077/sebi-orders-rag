#!/usr/bin/env python3
"""Debug SEBI order lookup, clarify selection, SAT/court routing, and corpus metadata routing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.current_info.query_normalization import normalize_current_info_query
from app.sebi_orders_rag.db import get_connection, initialize_phase4_schema
from app.sebi_orders_rag.repositories.sessions import ChatSessionRepository
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionStateRecord


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Query to inspect.")
    parser.add_argument("--session-id", default=None, help="Optional existing session UUID to reuse.")
    parser.add_argument("--record-key", default=None, help="Optional active record_key for follow-up routing.")
    parser.add_argument(
        "--current-lookup-family",
        default=None,
        help="Optional current-info follow-up family.",
    )
    parser.add_argument(
        "--current-lookup-query",
        default=None,
        help="Optional previous current-info query.",
    )
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    settings = SebiOrdersRagSettings.from_env(data_root_override=PROJECT_ROOT)
    control_pack = load_control_pack(settings.control_pack_root)
    router = AdaptiveQueryRouter(control_pack=control_pack)
    resolved_session_id = UUID(args.session_id) if args.session_id else uuid4()

    with get_connection(settings) as connection:
        initialize_phase4_schema(connection, settings)
        connection.commit()
        session_repository = ChatSessionRepository(connection)
        stored_state = session_repository.get_session_state(session_id=resolved_session_id)
        session_state = stored_state or _build_session_state(
            record_key=args.record_key,
            current_lookup_family=args.current_lookup_family,
            current_lookup_query=args.current_lookup_query,
        )
        decision = router.decide(query=args.query, session_state=session_state)
        analysis = decision.analysis
        current_info_query = normalize_current_info_query(args.query, session_state=session_state)
        answer_service = AdaptiveRagAnswerService(settings=settings, connection=connection)
        payload = answer_service.answer_query(query=args.query, session_id=resolved_session_id)
        updated_state = session_repository.get_session_state(session_id=payload.session_id)

    print(f"query: {args.query}")
    print(f"session_id: {resolved_session_id}")
    print(f"router chosen route: {decision.route_mode}")
    print(f"final route after answer-time guardrails: {payload.route_mode}")
    print(f"final answer status: {payload.answer_status}")
    print(f"detected query family: {analysis.query_family}")
    print(f"normalized query: {analysis.normalized_query}")
    print(
        "normalized expansions: "
        + (", ".join(analysis.normalized_expansions) if analysis.normalized_expansions else "-")
    )
    print(
        "matched abbreviations: "
        + (", ".join(analysis.matched_abbreviations) if analysis.matched_abbreviations else "-")
    )
    print(f"current-info family: {current_info_query.lookup_type}")
    print(f"active-order override: {analysis.active_order_override}")
    print(f"fresh-query override: {analysis.fresh_query_override}")
    print(f"active-matter follow-up intent: {analysis.active_matter_follow_up_intent}")
    print(f"clarification context active before route: {bool(session_state and session_state.clarification_context)}")
    print(f"SAT/court route priority fired: {analysis.appears_sat_court_style}")
    print(f"corpus metadata route fired: {analysis.appears_corpus_metadata_query}")
    print(f"candidate bucket priors applied: {bool(analysis.sat_court_signals)}")
    print(
        "reason codes: "
        + (", ".join(decision.reason_codes) if decision.reason_codes else "-")
    )
    print(
        "strict matter candidates: "
        + json.dumps(
            [
                {
                    "record_key": item.record_key,
                    "title": item.title,
                    "bucket_name": item.bucket_name,
                    "score": item.score,
                }
                for item in analysis.strict_matter_lock.candidates
            ],
            indent=2,
        )
    )
    print(
        "clarification debug: "
        + json.dumps(payload.debug.get("clarification_debug", {}), indent=2, sort_keys=True)
    )
    print(
        "candidate list debug: "
        + json.dumps(payload.debug.get("candidate_list_debug", {}), indent=2, sort_keys=True)
    )
    print(
        "corpus metadata debug: "
        + json.dumps(payload.debug.get("corpus_metadata_debug", {}), indent=2, sort_keys=True)
    )
    print(
        "exact lookup debug: "
        + json.dumps(payload.debug.get("exact_lookup_debug", {}), indent=2, sort_keys=True)
    )
    print(
        "route debug: "
        + json.dumps(payload.debug.get("route_debug", {}), indent=2, sort_keys=True)
    )
    print(
        "clarification context active after answer: "
        + str(bool(updated_state and updated_state.clarification_context))
    )
    if updated_state and updated_state.clarification_context:
        print(
            "stored clarification candidates: "
            + json.dumps(
                [
                    {
                        "candidate_index": candidate.candidate_index,
                        "title": candidate.title,
                        "record_key": candidate.record_key,
                        "bucket_name": candidate.bucket_name,
                        "order_date": (
                            candidate.order_date.isoformat()
                            if candidate.order_date is not None
                            else None
                        ),
                    }
                    for candidate in updated_state.clarification_context.candidates
                ],
                indent=2,
            )
        )
    print(f"final answer: {payload.answer_text}")
    return 0


def _build_session_state(
    *,
    record_key: str | None,
    current_lookup_family: str | None,
    current_lookup_query: str | None,
) -> ChatSessionStateRecord | None:
    if not any((record_key, current_lookup_family)):
        return None
    return ChatSessionStateRecord(
        session_id=uuid4(),
        active_record_keys=((record_key,) if record_key else ()),
        current_lookup_family=current_lookup_family,
        current_lookup_query=current_lookup_query,
    )


if __name__ == "__main__":
    raise SystemExit(main())
