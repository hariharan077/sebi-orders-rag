"""Local Phase 4 chat routes for standalone validation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from typing import Any
from uuid import UUID

from ..answering.answer_service import AdaptiveRagAnswerService
from ..config import SebiOrdersRagSettings, load_env_file
from ..db import ensure_phase4_schema_initialized, get_connection
from ..repositories.sessions import ChatSessionRepository
from ..schemas import ChatAnswerPayload, ChatSessionSnapshot

try:  # pragma: no cover - depends on local runtime
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - depends on local runtime
    raise RuntimeError(
        "fastapi is required for Phase 4 API routes. "
        "Install the dependencies from requirements-sebi-orders-rag.txt."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
router = APIRouter(prefix="/chat", tags=["chat"])


class CitationModel(BaseModel):
    citation_number: int
    record_key: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_type: Optional[str] = None
    document_version_id: Optional[int] = None
    chunk_id: Optional[int] = None
    detail_url: Optional[str] = None
    pdf_url: Optional[str] = None
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    domain: Optional[str] = None
    source_type: Optional[str] = None
    snippet: Optional[str] = None


class ClarificationCandidateModel(BaseModel):
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
    query: str = Field(min_length=1)


class ChatQueryResponse(BaseModel):
    session_id: UUID
    route_mode: str
    query_intent: str
    answer_text: str
    confidence: float
    citations: List[CitationModel]
    retrieved_chunk_ids: List[int]
    active_record_keys: List[str]
    answer_status: str
    clarification_candidates: List[ClarificationCandidateModel] = Field(default_factory=list)


class ChatSessionRequest(BaseModel):
    session_id: Optional[UUID] = None
    user_name: Optional[str] = None


class ChatSessionStateResponse(BaseModel):
    active_document_ids: List[int]
    active_record_keys: List[str]
    active_entities: List[str]
    active_bucket_names: List[str]
    last_chunk_ids: List[int]
    last_citation_chunk_ids: List[int]
    grounded_summary: Optional[str]


class ChatSessionResponse(BaseModel):
    session_id: UUID
    user_name: Optional[str]
    created_at: str
    updated_at: str
    state: ChatSessionStateResponse


@router.post("/query", response_model=ChatQueryResponse)
def query_chat(request: ChatQueryRequest) -> ChatQueryResponse:
    settings = _load_settings()
    ensure_phase4_schema_initialized(settings)
    with get_connection(settings) as connection:
        service = AdaptiveRagAnswerService(settings=settings, connection=connection)
        payload = service.answer_query(query=request.query, session_id=request.session_id)
    return _payload_to_response(payload)


@router.post("/session", response_model=ChatSessionResponse)
def create_session(request: ChatSessionRequest) -> ChatSessionResponse:
    from uuid import uuid4

    settings = _load_settings()
    ensure_phase4_schema_initialized(settings)
    session_id = request.session_id or uuid4()
    with get_connection(settings) as connection:
        repository = ChatSessionRepository(connection)
        repository.create_session_if_missing(session_id=session_id, user_name=request.user_name)
        connection.commit()
        snapshot = repository.get_session_snapshot(session_id=session_id)
    if snapshot is None:  # pragma: no cover - defensive path
        raise HTTPException(status_code=500, detail="failed to create session")
    return _snapshot_to_response(snapshot)


@router.get("/session/{session_id}", response_model=ChatSessionResponse)
def get_session(session_id: UUID) -> ChatSessionResponse:
    settings = _load_settings()
    ensure_phase4_schema_initialized(settings)
    with get_connection(settings) as connection:
        repository = ChatSessionRepository(connection)
        snapshot = repository.get_session_snapshot(session_id=session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _snapshot_to_response(snapshot)


def _load_settings() -> SebiOrdersRagSettings:
    load_env_file(PROJECT_ROOT / ".env")
    return SebiOrdersRagSettings.from_env()


def _payload_to_response(payload: ChatAnswerPayload) -> ChatQueryResponse:
    return ChatQueryResponse(
        session_id=payload.session_id,
        route_mode=payload.route_mode,
        query_intent=payload.query_intent,
        answer_text=payload.answer_text,
        confidence=payload.confidence,
        citations=[CitationModel(**citation.__dict__) for citation in payload.citations],
        retrieved_chunk_ids=list(payload.retrieved_chunk_ids),
        active_record_keys=list(payload.active_record_keys),
        answer_status=payload.answer_status,
        clarification_candidates=[
            ClarificationCandidateModel(
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
            for candidate in payload.clarification_candidates
        ],
    )


def _snapshot_to_response(snapshot: ChatSessionSnapshot) -> ChatSessionResponse:
    state = snapshot.state
    if state is None:
        state_response = ChatSessionStateResponse(
            active_document_ids=[],
            active_record_keys=[],
            active_entities=[],
            active_bucket_names=[],
            last_chunk_ids=[],
            last_citation_chunk_ids=[],
            grounded_summary=None,
        )
    else:
        state_response = ChatSessionStateResponse(
            active_document_ids=list(state.active_document_ids),
            active_record_keys=list(state.active_record_keys),
            active_entities=list(state.active_entities),
            active_bucket_names=list(state.active_bucket_names),
            last_chunk_ids=list(state.last_chunk_ids),
            last_citation_chunk_ids=list(state.last_citation_chunk_ids),
            grounded_summary=state.grounded_summary,
        )
    return ChatSessionResponse(
        session_id=snapshot.session_id,
        user_name=snapshot.user_name,
        created_at=snapshot.created_at.isoformat(),
        updated_at=snapshot.updated_at.isoformat(),
        state=state_response,
    )
