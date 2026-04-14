"""Session-state helpers for grounded follow-up handling."""

from __future__ import annotations

from uuid import UUID

from ..schemas import ChatSessionSnapshot, ChatSessionStateRecord


def empty_session_state(session_id: UUID) -> ChatSessionStateRecord:
    """Return an empty grounded session state for a new session."""

    return ChatSessionStateRecord(session_id=session_id)


def snapshot_with_state(
    snapshot: ChatSessionSnapshot,
    state: ChatSessionStateRecord | None,
) -> ChatSessionSnapshot:
    """Attach a normalized state object to a session snapshot."""

    normalized_state = state or empty_session_state(snapshot.session_id)
    return ChatSessionSnapshot(
        session_id=snapshot.session_id,
        user_name=snapshot.user_name,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
        state=normalized_state,
    )


def has_active_scope(state: ChatSessionStateRecord | None) -> bool:
    """Return whether the session carries grounded document scope."""

    if state is None:
        return False
    return bool(state.active_document_ids or state.active_record_keys)
