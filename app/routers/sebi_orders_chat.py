"""Thin FastAPI router for the internal SEBI Orders beta chat page."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from ..dependencies.sebi_orders_chat_dependencies import (
    get_sebi_orders_chat_service,
    get_sebi_orders_templates,
)
from ..services.sebi_orders_rag_service import SebiOrdersRagService

try:  # pragma: no cover - runtime import
    from fastapi import APIRouter, Depends, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - depends on runtime
    raise RuntimeError(
        "fastapi is required for the SEBI Orders portal router. "
        "Install the dependencies from requirements-sebi-orders-rag.txt."
    ) from exc

router = APIRouter(prefix="/sebi-orders/chat", tags=["sebi-orders-chat"])

_CORPUS_NOTE = "Using the current fully processed local corpus of 215 downloaded PDFs."
_SESSION_ID_PLACEHOLDER = "00000000-0000-0000-0000-000000000000"


class CitationResponse(BaseModel):
    citation_number: int
    record_key: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    detail_url: Optional[str] = None
    pdf_url: Optional[str] = None
    source_url: Optional[str] = None
    title_url: Optional[str] = None
    page_url: Optional[str] = None


class ClarificationCandidateResponse(BaseModel):
    candidate_id: str
    candidate_index: int
    candidate_type: str
    title: str
    record_key: Optional[str] = None
    bucket_name: Optional[str] = None
    order_date: Optional[str] = None
    document_version_id: Optional[int] = None
    descriptor: Optional[str] = None


class ChatQueryRequest(BaseModel):
    session_id: Optional[UUID] = None
    message: str = Field(min_length=1)


class ChatQueryResponse(BaseModel):
    session_id: UUID
    answer_text: str
    route_mode: str
    query_intent: str
    confidence: float
    citations: List[CitationResponse]
    active_record_keys: List[str]
    answer_status: str = "answered"
    clarification_candidates: List[ClarificationCandidateResponse] = Field(default_factory=list)


class ChatSessionRequest(BaseModel):
    session_id: Optional[UUID] = None
    user_name: Optional[str] = None


class ChatSessionStateResponse(BaseModel):
    active_record_keys: List[str]
    active_document_ids: List[int]


class ChatSessionResponse(BaseModel):
    session_id: UUID
    created_at: str
    updated_at: str
    state: ChatSessionStateResponse


class ChatSessionSummaryResponse(BaseModel):
    session_id: UUID
    title: str
    preview_text: str
    created_at: str
    updated_at: str
    last_message_at: Optional[str] = None
    turn_count: int


class ChatSessionsResponse(BaseModel):
    sessions: List[ChatSessionSummaryResponse]


class ChatTurnResponse(BaseModel):
    created_at: Optional[str] = None
    user_message: str
    assistant_message: str
    route_mode: str
    query_intent: Optional[str] = None
    confidence: float
    citations: List[CitationResponse]


class ChatSessionHistoryResponse(BaseModel):
    session: ChatSessionResponse
    turns: List[ChatTurnResponse]


@router.get("", response_class=HTMLResponse, name="render_sebi_orders_chat")
def render_sebi_orders_chat(
    request: Request,
    templates=Depends(get_sebi_orders_templates),
) -> HTMLResponse:
    session_history_endpoint_template = str(
        request.url_for(
            "get_sebi_orders_chat_session_history",
            session_id=_SESSION_ID_PLACEHOLDER,
        )
    ).replace(_SESSION_ID_PLACEHOLDER, "__SESSION_ID__")
    return templates.TemplateResponse(
        request=request,
        name="sebi_orders/chat.html",
        context={
            "page_title": "SEBI Knowledge Hub",
            "corpus_note": _CORPUS_NOTE,
            "query_endpoint": request.url_for("query_sebi_orders_chat"),
            "session_endpoint": request.url_for("create_sebi_orders_chat_session"),
            "sessions_endpoint": request.url_for("list_sebi_orders_chat_sessions"),
            "session_history_endpoint_template": session_history_endpoint_template,
        },
    )


@router.post("/query", response_model=ChatQueryResponse, name="query_sebi_orders_chat")
def query_sebi_orders_chat(
    request: ChatQueryRequest,
    service: SebiOrdersRagService = Depends(get_sebi_orders_chat_service),
) -> ChatQueryResponse:
    result = service.query(message=request.message, session_id=request.session_id)
    return ChatQueryResponse(
        session_id=result.session_id,
        answer_text=result.answer_text,
        route_mode=result.route_mode,
        query_intent=result.query_intent,
        confidence=result.confidence,
        citations=[
            CitationResponse(
                citation_number=citation.citation_number,
                record_key=citation.record_key,
                title=citation.title,
                page_start=citation.page_start,
                page_end=citation.page_end,
                detail_url=citation.detail_url,
                pdf_url=citation.pdf_url,
                source_url=citation.source_url,
                title_url=citation.title_url,
                page_url=citation.page_url,
            )
            for citation in result.citations
        ],
        active_record_keys=list(result.active_record_keys),
        answer_status=result.answer_status,
        clarification_candidates=[
            ClarificationCandidateResponse(
                candidate_id=candidate.candidate_id,
                candidate_index=candidate.candidate_index,
                candidate_type=candidate.candidate_type,
                title=candidate.title,
                record_key=candidate.record_key,
                bucket_name=candidate.bucket_name,
                order_date=candidate.order_date,
                document_version_id=candidate.document_version_id,
                descriptor=candidate.descriptor,
            )
            for candidate in result.clarification_candidates
        ],
    )


@router.post("/session", response_model=ChatSessionResponse, name="create_sebi_orders_chat_session")
def create_sebi_orders_chat_session(
    request: ChatSessionRequest,
    service: SebiOrdersRagService = Depends(get_sebi_orders_chat_service),
) -> ChatSessionResponse:
    session = service.create_session(
        session_id=request.session_id,
        user_name=request.user_name,
    )
    return ChatSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        state=ChatSessionStateResponse(
            active_record_keys=list(session.state.active_record_keys),
            active_document_ids=list(session.state.active_document_ids),
        ),
    )


@router.get("/sessions", response_model=ChatSessionsResponse, name="list_sebi_orders_chat_sessions")
def list_sebi_orders_chat_sessions(
    limit: int = 50,
    service: SebiOrdersRagService = Depends(get_sebi_orders_chat_service),
) -> ChatSessionsResponse:
    sessions = service.list_sessions(limit=max(1, min(limit, 100)))
    return ChatSessionsResponse(
        sessions=[
            ChatSessionSummaryResponse(
                session_id=session.session_id,
                title=session.title,
                preview_text=session.preview_text,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_message_at=session.last_message_at,
                turn_count=session.turn_count,
            )
            for session in sessions
        ]
    )


@router.get(
    "/session/{session_id}/history",
    response_model=ChatSessionHistoryResponse,
    name="get_sebi_orders_chat_session_history",
)
def get_sebi_orders_chat_session_history(
    session_id: UUID,
    service: SebiOrdersRagService = Depends(get_sebi_orders_chat_service),
) -> ChatSessionHistoryResponse:
    history = service.get_session_history(session_id=session_id)
    if history is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="session not found")
    return ChatSessionHistoryResponse(
        session=ChatSessionResponse(
            session_id=history.session.session_id,
            created_at=history.session.created_at,
            updated_at=history.session.updated_at,
            state=ChatSessionStateResponse(
                active_record_keys=list(history.session.state.active_record_keys),
                active_document_ids=list(history.session.state.active_document_ids),
            ),
        ),
        turns=[
            ChatTurnResponse(
                created_at=turn.created_at,
                user_message=turn.user_message,
                assistant_message=turn.assistant_message,
                route_mode=turn.route_mode,
                query_intent=turn.query_intent,
                confidence=turn.confidence,
                citations=[
                    CitationResponse(
                        citation_number=citation.citation_number,
                        record_key=citation.record_key,
                        title=citation.title,
                        page_start=citation.page_start,
                        page_end=citation.page_end,
                        detail_url=citation.detail_url,
                        pdf_url=citation.pdf_url,
                        source_url=citation.source_url,
                        title_url=citation.title_url,
                        page_url=citation.page_url,
                    )
                    for citation in turn.citations
                ],
            )
            for turn in history.turns
        ],
    )
