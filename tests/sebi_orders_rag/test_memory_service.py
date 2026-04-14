from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from app.sebi_orders_rag.memory.memory_service import GroundedMemoryService
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, Citation, PromptContextChunk


class MemoryServiceTests(unittest.TestCase):
    def test_updates_grounded_memory_after_retrieval_answer(self) -> None:
        session_id = uuid4()
        session_repository = _FakeSessionRepository(session_id)
        retrieval_repository = _FakeRetrievalRepository()
        service = GroundedMemoryService(
            session_repository=session_repository,
            retrieval_repository=retrieval_repository,
        )
        service.get_or_create_session(session_id=session_id)

        context_chunks = (
            PromptContextChunk(
                citation_number=1,
                chunk_id=11,
                document_version_id=101,
                document_id=201,
                record_key="external:100725",
                bucket_name="orders-of-aa-under-rti-act",
                title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                page_start=1,
                page_end=2,
                section_type="operative_order",
                section_title="ORDER",
                detail_url="https://example.com/detail/100725",
                pdf_url="https://example.com/pdf/100725.pdf",
                chunk_text="The appeal is disposed of with directions to provide the requested reply.",
                token_count=42,
                score=0.09,
            ),
        )
        citations = (
            Citation(
                citation_number=1,
                record_key="external:100725",
                title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                page_start=1,
                page_end=2,
                section_type="operative_order",
                document_version_id=101,
                chunk_id=11,
                detail_url="https://example.com/detail/100725",
                pdf_url="https://example.com/pdf/100725.pdf",
            ),
        )

        updated = service.update_from_grounded_answer(
            session_id=session_id,
            context_chunks=context_chunks,
            citations=citations,
        )

        self.assertEqual(updated.active_document_ids, (201,))
        self.assertEqual(updated.active_record_keys, ("external:100725",))
        self.assertEqual(updated.last_citation_chunk_ids, (11,))
        self.assertIn("external:100725", updated.grounded_summary or "")
        self.assertIn("provide the requested reply", updated.grounded_summary or "")

    def test_updates_and_clears_current_lookup_context(self) -> None:
        session_id = uuid4()
        session_repository = _FakeSessionRepository(session_id)
        retrieval_repository = _FakeRetrievalRepository()
        service = GroundedMemoryService(
            session_repository=session_repository,
            retrieval_repository=retrieval_repository,
        )
        service.get_or_create_session(session_id=session_id)

        updated = service.update_current_lookup_context(
            session_id=session_id,
            family="office_contact",
            focus="Mumbai",
            query="where is sebi office in mumbai",
        )
        cleared = service.clear_current_lookup_context(session_id=session_id)

        self.assertEqual(updated.current_lookup_family, "office_contact")
        self.assertEqual(updated.current_lookup_focus, "Mumbai")
        self.assertEqual(cleared.current_lookup_family, None)
        self.assertEqual(cleared.current_lookup_focus, None)


class _FakeSessionRepository:
    def __init__(self, session_id) -> None:
        now = datetime.now(timezone.utc)
        self.snapshot = ChatSessionSnapshot(
            session_id=session_id,
            user_name=None,
            created_at=now,
            updated_at=now,
            state=None,
        )
        self.state = None

    def create_session_if_missing(self, *, session_id, user_name) -> None:
        self.snapshot = ChatSessionSnapshot(
            session_id=session_id,
            user_name=user_name,
            created_at=self.snapshot.created_at,
            updated_at=datetime.now(timezone.utc),
            state=self.state,
        )

    def get_session_snapshot(self, *, session_id):
        return ChatSessionSnapshot(
            session_id=self.snapshot.session_id,
            user_name=self.snapshot.user_name,
            created_at=self.snapshot.created_at,
            updated_at=self.snapshot.updated_at,
            state=self.state,
        )

    def get_session_state(self, *, session_id):
        return self.state

    def upsert_session_state(self, **kwargs) -> None:
        from app.sebi_orders_rag.schemas import ChatSessionStateRecord

        self.state = ChatSessionStateRecord(
            session_id=kwargs["session_id"],
            active_document_ids=tuple(kwargs["active_document_ids"]),
            active_record_keys=tuple(kwargs["active_record_keys"]),
            active_entities=tuple(kwargs["active_entities"]),
            active_bucket_names=tuple(kwargs["active_bucket_names"]),
            last_chunk_ids=tuple(kwargs["last_chunk_ids"]),
            last_citation_chunk_ids=tuple(kwargs["last_citation_chunk_ids"]),
            grounded_summary=kwargs["grounded_summary"],
            current_lookup_family=kwargs.get("current_lookup_family"),
            current_lookup_focus=kwargs.get("current_lookup_focus"),
            current_lookup_query=kwargs.get("current_lookup_query"),
        )


class _FakeRetrievalRepository:
    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return (101,)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
