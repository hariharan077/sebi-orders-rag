"""Thin portal-facing wrapper over the Phase 4 adaptive RAG layer."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID, uuid4

from ..sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from ..sebi_orders_rag.config import SebiOrdersRagSettings
from ..sebi_orders_rag.db import ensure_phase4_schema_initialized, get_connection, initialize_phase4_schema
from ..sebi_orders_rag.repositories.sessions import ChatSessionRepository
from ..sebi_orders_rag.schemas import (
    ChatAnswerPayload,
    ChatSessionListEntry,
    ChatSessionSnapshot,
    ChatTurnRecord,
    Citation,
    ClarificationCandidate,
)


@dataclass(frozen=True)
class SebiOrdersChatCitation:
    """Portal-safe citation payload rendered beneath assistant messages."""

    citation_number: int
    record_key: str
    title: str
    page_start: int | None
    page_end: int | None
    detail_url: str | None
    pdf_url: str | None
    source_url: str | None
    source_title: str | None = None
    domain: str | None = None
    source_type: str | None = None
    snippet: str | None = None
    title_url: str | None = None
    page_url: str | None = None


@dataclass(frozen=True)
class SebiOrdersChatClarificationCandidate:
    """Portal-safe clarification candidate payload."""

    candidate_id: str
    candidate_index: int
    candidate_type: str
    title: str
    record_key: str | None = None
    bucket_name: str | None = None
    order_date: str | None = None
    document_version_id: int | None = None
    descriptor: str | None = None


@dataclass(frozen=True)
class SebiOrdersChatQueryResult:
    """Thin portal response payload for one chat turn."""

    session_id: UUID
    answer_text: str
    route_mode: str
    query_intent: str
    confidence: float
    citations: tuple[SebiOrdersChatCitation, ...]
    active_record_keys: tuple[str, ...]
    answer_status: str = "answered"
    clarification_candidates: tuple[SebiOrdersChatClarificationCandidate, ...] = ()


@dataclass(frozen=True)
class SebiOrdersChatSessionState:
    """Portal-safe session state metadata."""

    active_record_keys: tuple[str, ...]
    active_document_ids: tuple[int, ...]


@dataclass(frozen=True)
class SebiOrdersChatSession:
    """Portal-safe session snapshot."""

    session_id: UUID
    created_at: str
    updated_at: str
    state: SebiOrdersChatSessionState


@dataclass(frozen=True)
class SebiOrdersChatTurn:
    """Portal-safe visible chat turn rebuilt from persisted answer logs."""

    created_at: str | None
    user_message: str
    assistant_message: str
    route_mode: str
    query_intent: str | None
    confidence: float
    citations: tuple[SebiOrdersChatCitation, ...]


@dataclass(frozen=True)
class SebiOrdersChatSessionSummary:
    """Portal-safe recent session metadata for the sidebar."""

    session_id: UUID
    title: str
    preview_text: str
    created_at: str
    updated_at: str
    last_message_at: str | None
    turn_count: int


@dataclass(frozen=True)
class SebiOrdersChatSessionHistory:
    """Portal-safe persisted session plus its reconstructed visible turns."""

    session: SebiOrdersChatSession
    turns: tuple[SebiOrdersChatTurn, ...]


SettingsLoader = Callable[[], SebiOrdersRagSettings]
ConnectionFactory = Callable[[SebiOrdersRagSettings], AbstractContextManager[Any]]
SchemaInitializer = Callable[[Any, SebiOrdersRagSettings], None]
AnswerServiceFactory = Callable[[SebiOrdersRagSettings, Any], AdaptiveRagAnswerService]
SessionRepositoryFactory = Callable[[Any], ChatSessionRepository]


class SebiOrdersRagService:
    """Bridge the portal layer to the existing Phase 4 answer pipeline."""

    def __init__(
        self,
        *,
        settings_loader: SettingsLoader,
        connection_factory: ConnectionFactory = get_connection,
        schema_initializer: SchemaInitializer = initialize_phase4_schema,
        answer_service_factory: AnswerServiceFactory | None = None,
        session_repository_factory: SessionRepositoryFactory = ChatSessionRepository,
    ) -> None:
        self._settings_loader = settings_loader
        self._connection_factory = connection_factory
        self._schema_initializer = schema_initializer
        self._answer_service_factory = answer_service_factory or _default_answer_service_factory
        self._session_repository_factory = session_repository_factory

    def query(
        self,
        *,
        message: str,
        session_id: UUID | None = None,
    ) -> SebiOrdersChatQueryResult:
        """Execute a portal chat turn through the existing adaptive RAG service."""

        settings = self._settings_loader()
        with self._connection_factory(settings) as connection:
            self._initialize_connection(connection, settings)
            answer_service = self._answer_service_factory(settings, connection)
            payload = answer_service.answer_query(query=message, session_id=session_id)
        return _to_chat_query_result(payload)

    def create_session(
        self,
        *,
        session_id: UUID | None = None,
        user_name: str | None = None,
    ) -> SebiOrdersChatSession:
        """Create a chat session for the portal if one does not already exist."""

        settings = self._settings_loader()
        resolved_session_id = session_id or uuid4()
        with self._connection_factory(settings) as connection:
            self._initialize_connection(connection, settings)
            repository = self._session_repository_factory(connection)
            repository.create_session_if_missing(
                session_id=resolved_session_id,
                user_name=user_name,
            )
            _commit_if_supported(connection)
            snapshot = repository.get_session_snapshot(session_id=resolved_session_id)

        if snapshot is None:  # pragma: no cover - defensive path
            raise RuntimeError("failed to create or load SEBI Orders chat session")
        return _to_chat_session(snapshot)

    def list_sessions(self, *, limit: int = 50) -> tuple[SebiOrdersChatSessionSummary, ...]:
        """Return recent persisted chats for the portal sidebar."""

        settings = self._settings_loader()
        with self._connection_factory(settings) as connection:
            self._initialize_connection(connection, settings)
            repository = self._session_repository_factory(connection)
            entries = repository.list_recent_sessions(limit=limit)
        return tuple(_to_chat_session_summary(entry) for entry in entries)

    def get_session_history(self, *, session_id: UUID) -> SebiOrdersChatSessionHistory | None:
        """Return one persisted session and its visible answer-log turns."""

        settings = self._settings_loader()
        with self._connection_factory(settings) as connection:
            self._initialize_connection(connection, settings)
            repository = self._session_repository_factory(connection)
            snapshot = repository.get_session_snapshot(session_id=session_id)
            if snapshot is None:
                return None
            turns = repository.get_session_turns(session_id=session_id)
        return SebiOrdersChatSessionHistory(
            session=_to_chat_session(snapshot),
            turns=tuple(_to_chat_turn(turn) for turn in turns),
        )

    def _initialize_connection(self, connection: Any, settings: SebiOrdersRagSettings) -> None:
        if self._schema_initializer is initialize_phase4_schema:
            ensure_phase4_schema_initialized(settings)
            return
        self._schema_initializer(connection, settings)
        _commit_if_supported(connection)


def _default_answer_service_factory(
    settings: SebiOrdersRagSettings,
    connection: Any,
) -> AdaptiveRagAnswerService:
    return AdaptiveRagAnswerService(settings=settings, connection=connection)


def _commit_if_supported(connection: Any) -> None:
    commit = getattr(connection, "commit", None)
    if callable(commit):
        commit()


def _to_chat_query_result(payload: ChatAnswerPayload) -> SebiOrdersChatQueryResult:
    return SebiOrdersChatQueryResult(
        session_id=payload.session_id,
        answer_text=payload.answer_text,
        route_mode=payload.route_mode,
        query_intent=payload.query_intent,
        confidence=payload.confidence,
        citations=tuple(_to_chat_citation(citation) for citation in payload.citations),
        active_record_keys=tuple(payload.active_record_keys),
        answer_status=payload.answer_status,
        clarification_candidates=tuple(
            _to_chat_clarification_candidate(candidate)
            for candidate in payload.clarification_candidates
        ),
    )


def _to_chat_citation(citation: Citation) -> SebiOrdersChatCitation:
    return SebiOrdersChatCitation(
        citation_number=citation.citation_number,
        record_key=citation.record_key,
        title=citation.title,
        page_start=citation.page_start,
        page_end=citation.page_end,
        detail_url=citation.detail_url,
        pdf_url=citation.pdf_url,
        source_url=citation.source_url,
        source_title=citation.source_title,
        domain=citation.domain,
        source_type=citation.source_type,
        snippet=citation.snippet,
        title_url=_resolve_title_url(citation),
        page_url=_resolve_page_url(citation),
    )


def _to_chat_clarification_candidate(
    candidate: ClarificationCandidate,
) -> SebiOrdersChatClarificationCandidate:
    return SebiOrdersChatClarificationCandidate(
        candidate_id=candidate.candidate_id,
        candidate_index=candidate.candidate_index,
        candidate_type=candidate.candidate_type,
        title=candidate.title,
        record_key=candidate.record_key,
        bucket_name=candidate.bucket_name,
        order_date=candidate.order_date.isoformat() if candidate.order_date else None,
        document_version_id=candidate.document_version_id,
        descriptor=candidate.descriptor,
    )


def _to_chat_session(snapshot: ChatSessionSnapshot) -> SebiOrdersChatSession:
    state = snapshot.state
    return SebiOrdersChatSession(
        session_id=snapshot.session_id,
        created_at=snapshot.created_at.isoformat(),
        updated_at=snapshot.updated_at.isoformat(),
        state=SebiOrdersChatSessionState(
            active_record_keys=tuple(state.active_record_keys) if state else (),
            active_document_ids=tuple(state.active_document_ids) if state else (),
        ),
    )


def _to_chat_turn(turn: ChatTurnRecord) -> SebiOrdersChatTurn:
    return SebiOrdersChatTurn(
        created_at=turn.created_at.isoformat() if turn.created_at else None,
        user_message=turn.user_query,
        assistant_message=turn.answer_text,
        route_mode=turn.route_mode,
        query_intent=turn.query_intent,
        confidence=turn.answer_confidence,
        citations=tuple(_to_chat_citation(citation) for citation in turn.citations),
    )


def _to_chat_session_summary(entry: ChatSessionListEntry) -> SebiOrdersChatSessionSummary:
    title = _condense_text(entry.first_user_query, fallback="New chat")
    preview_source = entry.latest_user_query or entry.first_user_query
    return SebiOrdersChatSessionSummary(
        session_id=entry.session_id,
        title=title,
        preview_text=_condense_text(preview_source, fallback="Waiting for the first question."),
        created_at=entry.created_at.isoformat(),
        updated_at=entry.updated_at.isoformat(),
        last_message_at=entry.last_message_at.isoformat() if entry.last_message_at else None,
        turn_count=entry.turn_count,
    )


def _condense_text(value: str | None, *, fallback: str) -> str:
    if value is None:
        return fallback
    condensed = " ".join(part for part in value.split())
    return condensed or fallback


def _resolve_title_url(citation: Citation) -> str | None:
    if citation.detail_url:
        return citation.detail_url
    if citation.pdf_url:
        return citation.pdf_url
    return citation.source_url


def _resolve_page_url(citation: Citation) -> str | None:
    if citation.page_start is None:
        return None
    if citation.pdf_url:
        separator = "" if "#" in citation.pdf_url else "#"
        return f"{citation.pdf_url}{separator}page={citation.page_start}"
    if citation.detail_url:
        return citation.detail_url
    return citation.source_url
