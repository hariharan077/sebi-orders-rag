"""Grounded session memory management for adaptive RAG."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from uuid import UUID, uuid4

from ..schemas import (
    ChatSessionSnapshot,
    ChatSessionStateRecord,
    Citation,
    ClarificationCandidate,
    ClarificationContext,
    MetadataFilterInput,
    PromptContextChunk,
)
from .session_state import empty_session_state, snapshot_with_state

_TITLE_ENTITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfiled by (?P<entity>.+)$", re.IGNORECASE),
    re.compile(r"\bin the matter of (?P<entity>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<entity>.+?)\s+(?:vs\.?|versus|v\.)\s+.+$", re.IGNORECASE),
)
_CLARIFICATION_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "one",
        "in",
        "of",
        "on",
        "for",
        "this",
        "that",
        "case",
        "matter",
        "order",
        "appeal",
        "please",
        "pick",
        "choose",
        "show",
        "me",
        "vs",
        "versus",
        "v",
        "sebi",
    }
)


class GroundedMemoryService:
    """Read and update grounded session memory without free-form summaries."""

    def __init__(self, *, session_repository: object, retrieval_repository: object) -> None:
        self._sessions = session_repository
        self._retrieval = retrieval_repository

    def get_or_create_session(self, *, session_id: UUID | None) -> ChatSessionSnapshot:
        """Return an existing session snapshot or create a new one."""

        resolved_session_id = session_id or uuid4()
        self._sessions.create_session_if_missing(
            session_id=resolved_session_id,
            user_name=None,
        )
        snapshot = self._sessions.get_session_snapshot(session_id=resolved_session_id)
        if snapshot is None:  # pragma: no cover - defensive path
            raise RuntimeError("chat session creation did not persist")
        return snapshot_with_state(snapshot, snapshot.state)

    def get_session_snapshot(self, *, session_id: UUID) -> ChatSessionSnapshot | None:
        """Return the session snapshot if it exists."""

        snapshot = self._sessions.get_session_snapshot(session_id=session_id)
        if snapshot is None:
            return None
        return snapshot_with_state(snapshot, snapshot.state)

    def build_memory_filters(self, *, state: ChatSessionStateRecord) -> MetadataFilterInput:
        """Resolve current version ids for the session's active grounded scope."""

        current_version_ids = state.active_document_version_ids or self._retrieval.resolve_current_document_version_ids(
            record_keys=state.active_record_keys,
            document_ids=state.active_document_ids,
        )
        return MetadataFilterInput(document_version_ids=current_version_ids)

    def update_from_grounded_answer(
        self,
        *,
        session_id: UUID,
        context_chunks: tuple[PromptContextChunk, ...],
        citations: tuple[Citation, ...],
    ) -> ChatSessionStateRecord:
        """Persist grounded session memory using only retrieved and cited text."""

        if not context_chunks or not citations:
            existing = self._sessions.get_session_state(session_id=session_id)
            return existing or empty_session_state(session_id)

        cited_chunk_ids = {citation.chunk_id for citation in citations}
        cited_context = [chunk for chunk in context_chunks if chunk.chunk_id in cited_chunk_ids]
        active_chunks = cited_context or list(context_chunks)

        active_document_ids = tuple(_ordered_unique(chunk.document_id for chunk in active_chunks))
        active_document_version_ids = tuple(
            _ordered_unique(chunk.document_version_id for chunk in active_chunks)
        )
        active_record_keys = tuple(_ordered_unique(chunk.record_key for chunk in active_chunks))
        active_bucket_names = tuple(_ordered_unique(chunk.bucket_name for chunk in active_chunks))
        active_entities = tuple(
            _ordered_unique(
                entity
                for chunk in active_chunks
                for entity in _extract_entities(chunk.title)
            )
        )
        last_chunk_ids = tuple(chunk.chunk_id for chunk in context_chunks)
        last_citation_chunk_ids = tuple(chunk.chunk_id for chunk in cited_context)
        grounded_summary = _build_grounded_summary(cited_context)
        existing = self._sessions.get_session_state(session_id=session_id)

        self._sessions.upsert_session_state(
            session_id=session_id,
            active_document_ids=active_document_ids,
            active_document_version_ids=active_document_version_ids,
            active_record_keys=active_record_keys,
            active_entities=active_entities,
            active_bucket_names=active_bucket_names,
            active_primary_title=active_chunks[0].title if active_chunks else (existing.active_primary_title if existing else None),
            active_primary_entity=active_entities[0] if active_entities else (existing.active_primary_entity if existing else None),
            active_signatory_name=existing.active_signatory_name if existing else None,
            active_signatory_designation=existing.active_signatory_designation if existing else None,
            active_order_date=existing.active_order_date if existing else None,
            active_order_place=existing.active_order_place if existing else None,
            active_legal_provisions=existing.active_legal_provisions if existing else (),
            last_chunk_ids=last_chunk_ids,
            last_citation_chunk_ids=last_citation_chunk_ids,
            grounded_summary=grounded_summary,
            current_lookup_family=existing.current_lookup_family if existing else None,
            current_lookup_focus=existing.current_lookup_focus if existing else None,
            current_lookup_query=existing.current_lookup_query if existing else None,
            clarification_context=existing.clarification_context if existing else None,
        )
        updated = self._sessions.get_session_state(session_id=session_id)
        return updated or empty_session_state(session_id)

    def update_current_lookup_context(
        self,
        *,
        session_id: UUID,
        family: str,
        focus: str | None = None,
        query: str | None = None,
    ) -> ChatSessionStateRecord:
        """Persist narrow current-info lookup context without touching grounded scope."""

        existing = self._sessions.get_session_state(session_id=session_id) or empty_session_state(session_id)
        self._sessions.upsert_session_state(
            session_id=session_id,
            active_document_ids=existing.active_document_ids,
            active_document_version_ids=existing.active_document_version_ids,
            active_record_keys=existing.active_record_keys,
            active_entities=existing.active_entities,
            active_bucket_names=existing.active_bucket_names,
            active_primary_title=existing.active_primary_title,
            active_primary_entity=existing.active_primary_entity,
            active_signatory_name=existing.active_signatory_name,
            active_signatory_designation=existing.active_signatory_designation,
            active_order_date=existing.active_order_date,
            active_order_place=existing.active_order_place,
            active_legal_provisions=existing.active_legal_provisions,
            last_chunk_ids=existing.last_chunk_ids,
            last_citation_chunk_ids=existing.last_citation_chunk_ids,
            grounded_summary=existing.grounded_summary,
            current_lookup_family=family,
            current_lookup_focus=focus,
            current_lookup_query=query,
            clarification_context=existing.clarification_context,
        )
        updated = self._sessions.get_session_state(session_id=session_id)
        return updated or empty_session_state(session_id)

    def update_active_scope(
        self,
        *,
        session_id: UUID,
        document_ids: tuple[int, ...],
        document_version_ids: tuple[int, ...] = (),
        record_keys: tuple[str, ...],
    ) -> ChatSessionStateRecord:
        """Persist active order scope even when the answer came from extracted metadata."""

        existing = self._sessions.get_session_state(session_id=session_id) or empty_session_state(session_id)
        self._sessions.upsert_session_state(
            session_id=session_id,
            active_document_ids=document_ids or existing.active_document_ids,
            active_document_version_ids=document_version_ids or existing.active_document_version_ids,
            active_record_keys=record_keys or existing.active_record_keys,
            active_entities=existing.active_entities,
            active_bucket_names=existing.active_bucket_names,
            active_primary_title=existing.active_primary_title,
            active_primary_entity=existing.active_primary_entity,
            active_signatory_name=existing.active_signatory_name,
            active_signatory_designation=existing.active_signatory_designation,
            active_order_date=existing.active_order_date,
            active_order_place=existing.active_order_place,
            active_legal_provisions=existing.active_legal_provisions,
            last_chunk_ids=existing.last_chunk_ids,
            last_citation_chunk_ids=existing.last_citation_chunk_ids,
            grounded_summary=existing.grounded_summary,
            current_lookup_family=existing.current_lookup_family,
            current_lookup_focus=existing.current_lookup_focus,
            current_lookup_query=existing.current_lookup_query,
            clarification_context=existing.clarification_context,
        )
        updated = self._sessions.get_session_state(session_id=session_id)
        return updated or empty_session_state(session_id)

    def update_active_matter_context(
        self,
        *,
        session_id: UUID,
        document_ids: tuple[int, ...] = (),
        document_version_ids: tuple[int, ...] = (),
        record_keys: tuple[str, ...] = (),
        entities: tuple[str, ...] = (),
        bucket_names: tuple[str, ...] = (),
        primary_title: str | None = None,
        primary_entity: str | None = None,
        signatory_name: str | None = None,
        signatory_designation: str | None = None,
        order_date=None,
        order_place: str | None = None,
        legal_provisions: tuple[str, ...] = (),
    ) -> ChatSessionStateRecord:
        """Persist richer active-matter metadata for scoped follow-up questions."""

        existing = self._sessions.get_session_state(session_id=session_id) or empty_session_state(session_id)
        self._sessions.upsert_session_state(
            session_id=session_id,
            active_document_ids=document_ids or existing.active_document_ids,
            active_document_version_ids=document_version_ids or existing.active_document_version_ids,
            active_record_keys=record_keys or existing.active_record_keys,
            active_entities=entities or existing.active_entities,
            active_bucket_names=bucket_names or existing.active_bucket_names,
            active_primary_title=primary_title or existing.active_primary_title,
            active_primary_entity=primary_entity or existing.active_primary_entity,
            active_signatory_name=signatory_name or existing.active_signatory_name,
            active_signatory_designation=signatory_designation or existing.active_signatory_designation,
            active_order_date=order_date or existing.active_order_date,
            active_order_place=order_place or existing.active_order_place,
            active_legal_provisions=legal_provisions or existing.active_legal_provisions,
            last_chunk_ids=existing.last_chunk_ids,
            last_citation_chunk_ids=existing.last_citation_chunk_ids,
            grounded_summary=existing.grounded_summary,
            current_lookup_family=existing.current_lookup_family,
            current_lookup_focus=existing.current_lookup_focus,
            current_lookup_query=existing.current_lookup_query,
            clarification_context=existing.clarification_context,
        )
        updated = self._sessions.get_session_state(session_id=session_id)
        return updated or empty_session_state(session_id)

    def clear_current_lookup_context(self, *, session_id: UUID) -> ChatSessionStateRecord:
        """Clear narrow current-info lookup context while preserving grounded scope."""

        existing = self._sessions.get_session_state(session_id=session_id) or empty_session_state(session_id)
        self._sessions.upsert_session_state(
            session_id=session_id,
            active_document_ids=existing.active_document_ids,
            active_document_version_ids=existing.active_document_version_ids,
            active_record_keys=existing.active_record_keys,
            active_entities=existing.active_entities,
            active_bucket_names=existing.active_bucket_names,
            active_primary_title=existing.active_primary_title,
            active_primary_entity=existing.active_primary_entity,
            active_signatory_name=existing.active_signatory_name,
            active_signatory_designation=existing.active_signatory_designation,
            active_order_date=existing.active_order_date,
            active_order_place=existing.active_order_place,
            active_legal_provisions=existing.active_legal_provisions,
            last_chunk_ids=existing.last_chunk_ids,
            last_citation_chunk_ids=existing.last_citation_chunk_ids,
            grounded_summary=existing.grounded_summary,
            current_lookup_family=None,
            current_lookup_focus=None,
            current_lookup_query=None,
            clarification_context=existing.clarification_context,
        )
        updated = self._sessions.get_session_state(session_id=session_id)
        return updated or empty_session_state(session_id)

    def update_clarification_context(
        self,
        *,
        session_id: UUID,
        context: ClarificationContext,
    ) -> ChatSessionStateRecord:
        """Persist the last active clarification candidate set for follow-up selection."""

        existing = self._sessions.get_session_state(session_id=session_id) or empty_session_state(session_id)
        self._sessions.upsert_session_state(
            session_id=session_id,
            active_document_ids=existing.active_document_ids,
            active_document_version_ids=existing.active_document_version_ids,
            active_record_keys=existing.active_record_keys,
            active_entities=existing.active_entities,
            active_bucket_names=existing.active_bucket_names,
            active_primary_title=existing.active_primary_title,
            active_primary_entity=existing.active_primary_entity,
            active_signatory_name=existing.active_signatory_name,
            active_signatory_designation=existing.active_signatory_designation,
            active_order_date=existing.active_order_date,
            active_order_place=existing.active_order_place,
            active_legal_provisions=existing.active_legal_provisions,
            last_chunk_ids=existing.last_chunk_ids,
            last_citation_chunk_ids=existing.last_citation_chunk_ids,
            grounded_summary=existing.grounded_summary,
            current_lookup_family=existing.current_lookup_family,
            current_lookup_focus=existing.current_lookup_focus,
            current_lookup_query=existing.current_lookup_query,
            clarification_context=context,
        )
        updated = self._sessions.get_session_state(session_id=session_id)
        return updated or empty_session_state(session_id)

    def clear_clarification_context(self, *, session_id: UUID) -> ChatSessionStateRecord:
        """Clear the active clarification candidate set while preserving the grounded scope."""

        existing = self._sessions.get_session_state(session_id=session_id) or empty_session_state(session_id)
        self._sessions.upsert_session_state(
            session_id=session_id,
            active_document_ids=existing.active_document_ids,
            active_document_version_ids=existing.active_document_version_ids,
            active_record_keys=existing.active_record_keys,
            active_entities=existing.active_entities,
            active_bucket_names=existing.active_bucket_names,
            active_primary_title=existing.active_primary_title,
            active_primary_entity=existing.active_primary_entity,
            active_signatory_name=existing.active_signatory_name,
            active_signatory_designation=existing.active_signatory_designation,
            active_order_date=existing.active_order_date,
            active_order_place=existing.active_order_place,
            active_legal_provisions=existing.active_legal_provisions,
            last_chunk_ids=existing.last_chunk_ids,
            last_citation_chunk_ids=existing.last_citation_chunk_ids,
            grounded_summary=existing.grounded_summary,
            current_lookup_family=existing.current_lookup_family,
            current_lookup_focus=existing.current_lookup_focus,
            current_lookup_query=existing.current_lookup_query,
            clarification_context=None,
        )
        updated = self._sessions.get_session_state(session_id=session_id)
        return updated or empty_session_state(session_id)

    def resolve_clarification_selection(
        self,
        *,
        query: str,
        state: ChatSessionStateRecord | None,
    ) -> "ClarificationSelectionResult":
        """Resolve a user reply against the active clarification candidate set."""

        context = state.clarification_context if state is not None else None
        if context is None or not context.candidates:
            return ClarificationSelectionResult(
                active_context=False,
                selected_candidates=(),
            )
        normalized_query = " ".join(query.lower().split()).strip()
        if not normalized_query:
            return ClarificationSelectionResult(
                active_context=True,
                selected_candidates=(),
            )

        numeric_index = _parse_numeric_selection(normalized_query)
        if numeric_index is not None:
            matched = tuple(
                candidate
                for candidate in context.candidates
                if candidate.candidate_index == numeric_index
            )
            return ClarificationSelectionResult(
                active_context=True,
                selected_candidates=matched,
                match_reason="candidate_index",
            )

        ordinal_index = _parse_ordinal_selection(normalized_query)
        if ordinal_index is not None:
            matched = tuple(
                candidate
                for candidate in context.candidates
                if candidate.candidate_index == ordinal_index
            )
            return ClarificationSelectionResult(
                active_context=True,
                selected_candidates=matched,
                match_reason="candidate_ordinal",
            )

        matched = tuple(
            candidate
            for candidate in context.candidates
            if _clarification_candidate_matches(candidate, normalized_query)
        )
        if matched or _looks_like_clarification_selection_reply(normalized_query, context):
            return ClarificationSelectionResult(
                active_context=True,
                selected_candidates=matched,
                match_reason="selection_alias" if matched else None,
            )
        return ClarificationSelectionResult(
            active_context=False,
            selected_candidates=(),
        )


@dataclass(frozen=True)
class ClarificationSelectionResult:
    """Outcome of resolving one clarification follow-up reply."""

    active_context: bool
    selected_candidates: tuple[ClarificationCandidate, ...]
    match_reason: str | None = None


def _ordered_unique(values: object) -> list[object]:
    unique: OrderedDict[object, None] = OrderedDict()
    for value in values:
        if value in (None, ""):
            continue
        unique[value] = None
    return list(unique.keys())


def _extract_entities(title: str) -> tuple[str, ...]:
    entities: list[str] = []
    for pattern in _TITLE_ENTITY_PATTERNS:
        match = pattern.search(title.strip())
        if match:
            entities.append(_clean_entity(match.group("entity")))
    if not entities:
        return ()
    return tuple(entity for entity in entities if entity)


def _clean_entity(value: str) -> str:
    return " ".join(value.split()).strip(" -,.")


def _build_grounded_summary(chunks: list[PromptContextChunk]) -> str | None:
    if not chunks:
        return None

    lines: list[str] = []
    for chunk in chunks[:3]:
        excerpt = " ".join(chunk.chunk_text.split())
        if len(excerpt) > 220:
            excerpt = excerpt[:217].rstrip() + "..."
        lines.append(
            f"{chunk.record_key} | {chunk.section_type} | pp. {chunk.page_start}-{chunk.page_end}: {excerpt}"
        )
    return "\n".join(lines)


def _parse_numeric_selection(normalized_query: str) -> int | None:
    if normalized_query.isdigit():
        return int(normalized_query)
    return None


def _parse_ordinal_selection(normalized_query: str) -> int | None:
    ordinal_map = {
        "first": 1,
        "the first": 1,
        "the first one": 1,
        "second": 2,
        "the second": 2,
        "the second one": 2,
        "third": 3,
        "the third": 3,
        "the third one": 3,
        "fourth": 4,
        "the fourth": 4,
        "the fourth one": 4,
        "fifth": 5,
        "the fifth": 5,
        "the fifth one": 5,
    }
    return ordinal_map.get(normalized_query)


def _clarification_candidate_matches(
    candidate: ClarificationCandidate,
    normalized_query: str,
) -> bool:
    candidate_aliases = {
        alias
        for alias in candidate.selection_aliases
        if alias
    }
    if normalized_query in candidate_aliases:
        return True
    if candidate.title and normalized_query in " ".join(candidate.title.lower().split()):
        return True
    if candidate.descriptor and normalized_query in " ".join(candidate.descriptor.lower().split()):
        return True
    query_tokens = _selection_tokens(normalized_query)
    if query_tokens:
        candidate_tokens = _selection_tokens(
            " ".join(
                part
                for part in (
                    candidate.title,
                    candidate.descriptor,
                    " ".join(candidate.selection_aliases),
                )
                if part
            )
        )
        if query_tokens.issubset(candidate_tokens):
            return True
    return any(normalized_query in alias for alias in candidate_aliases)


def _looks_like_clarification_selection_reply(
    normalized_query: str,
    context: ClarificationContext,
) -> bool:
    if normalized_query.isdigit() or _parse_ordinal_selection(normalized_query) is not None:
        return True
    query_tokens = _selection_tokens(normalized_query)
    if not query_tokens:
        return False
    if len(query_tokens) > 4:
        return False
    candidate_tokens = set()
    for candidate in context.candidates:
        candidate_tokens.update(
            _selection_tokens(
                " ".join(
                    part
                    for part in (
                        candidate.title,
                        candidate.descriptor,
                        " ".join(candidate.selection_aliases),
                    )
                    if part
                )
            )
        )
    return bool(query_tokens & candidate_tokens)


def _selection_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token and token not in _CLARIFICATION_STOPWORDS
    }
