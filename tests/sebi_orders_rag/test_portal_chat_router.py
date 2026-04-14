from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from app.dependencies.sebi_orders_chat_dependencies import get_sebi_orders_chat_service
from app.routers.sebi_orders_chat import router
from app.services.sebi_orders_rag_service import (
    SebiOrdersChatCitation,
    SebiOrdersChatQueryResult,
    SebiOrdersChatSessionHistory,
    SebiOrdersChatSession,
    SebiOrdersChatSessionSummary,
    SebiOrdersChatSessionState,
    SebiOrdersChatTurn,
)


class PortalChatRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _FakePortalService()
        self.app = FastAPI()
        self.app.mount(
            "/static",
            StaticFiles(directory=str(Path("app/static").resolve())),
            name="static",
        )
        self.app.include_router(router)
        self.app.dependency_overrides[get_sebi_orders_chat_service] = lambda: self.service
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.client.close()

    def test_get_chat_page_renders_sidebar_history_shell(self) -> None:
        response = self.client.get("/sebi-orders/chat")

        self.assertEqual(response.status_code, 200)
        self.assertIn("SEBI Knowledge Hub", response.text)
        self.assertIn("Chat History", response.text)
        self.assertIn("New chat", response.text)
        self.assertNotIn("Reopen prior chats, continue follow-ups, or start a clean thread.", response.text)
        self.assertNotIn("Using the current fully processed local corpus of 215 downloaded PDFs.", response.text)
        self.assertNotIn(
            "Ask a question to start a persistent conversation. Saved chats stay available in the sidebar.",
            response.text,
        )
        self.assertNotIn("Beta: currently running on the downloaded local corpus of processed SEBI orders.", response.text)

    def test_query_endpoint_returns_portal_payload(self) -> None:
        response = self.client.post(
            "/sebi-orders/chat/query",
            json={"message": "What is a settlement order?"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route_mode"], "general_knowledge")
        self.assertEqual(payload["query_intent"], "general_knowledge")
        self.assertEqual(payload["active_record_keys"], [])
        self.assertEqual(payload["citations"], [])

    def test_session_endpoint_returns_session_snapshot(self) -> None:
        response = self.client.post("/sebi-orders/chat/session", json={})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["state"]["active_record_keys"], ["external:100725"])
        self.assertEqual(payload["state"]["active_document_ids"], [201])

    def test_sessions_endpoint_returns_recent_chat_summaries(self) -> None:
        response = self.client.get("/sebi-orders/chat/sessions")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["sessions"][0]["title"], "Appeal No. 6798 of 2026 filed by Hariom Yadav")
        self.assertEqual(payload["sessions"][0]["turn_count"], 1)

    def test_history_endpoint_returns_reconstructed_turns(self) -> None:
        response = self.client.get(f"/sebi-orders/chat/session/{self.service.session_id}/history")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session"]["session_id"], str(self.service.session_id))
        self.assertEqual(payload["turns"][0]["user_message"], "Appeal No. 6798 of 2026 filed by Hariom Yadav")
        self.assertEqual(payload["turns"][0]["citations"][0]["record_key"], "external:100725")
        self.assertEqual(
            payload["turns"][0]["citations"][0]["title_url"],
            "https://example.com/detail/100725",
        )
        self.assertEqual(
            payload["turns"][0]["citations"][0]["page_url"],
            "https://example.com/pdf/100725.pdf#page=1",
        )


class _FakePortalService:
    def __init__(self) -> None:
        self.session_id = uuid4()

    def query(self, *, message: str, session_id: UUID | None = None) -> SebiOrdersChatQueryResult:
        return SebiOrdersChatQueryResult(
            session_id=session_id or self.session_id,
            answer_text="A settlement order resolves enforcement proceedings through settlement.",
            route_mode="general_knowledge",
            query_intent="general_knowledge",
            confidence=0.81,
            citations=(),
            active_record_keys=(),
        )

    def create_session(self, *, session_id: UUID | None = None, user_name: str | None = None) -> SebiOrdersChatSession:
        now = datetime(2026, 4, 10, 10, 30, tzinfo=timezone.utc).isoformat()
        return SebiOrdersChatSession(
            session_id=session_id or self.session_id,
            created_at=now,
            updated_at=now,
            state=SebiOrdersChatSessionState(
                active_record_keys=("external:100725",),
                active_document_ids=(201,),
            ),
        )

    def list_sessions(self, *, limit: int = 50) -> tuple[SebiOrdersChatSessionSummary, ...]:
        now = datetime(2026, 4, 10, 10, 30, tzinfo=timezone.utc).isoformat()
        return (
            SebiOrdersChatSessionSummary(
                session_id=self.session_id,
                title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                preview_text="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                created_at=now,
                updated_at=now,
                last_message_at=now,
                turn_count=1,
            ),
        )

    def get_session_history(self, *, session_id: UUID) -> SebiOrdersChatSessionHistory | None:
        now = datetime(2026, 4, 10, 10, 30, tzinfo=timezone.utc).isoformat()
        return SebiOrdersChatSessionHistory(
            session=self.create_session(session_id=session_id),
            turns=(
                SebiOrdersChatTurn(
                    created_at=now,
                    user_message="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                    assistant_message="The appeal was dismissed.",
                    route_mode="exact_lookup",
                    query_intent="document_lookup",
                    confidence=0.88,
                    citations=(
                        SebiOrdersChatCitation(
                            citation_number=1,
                            record_key="external:100725",
                            title="Appeal No. 6798 of 2026 filed by Hariom Yadav",
                            page_start=1,
                            page_end=2,
                            detail_url="https://example.com/detail/100725",
                            pdf_url="https://example.com/pdf/100725.pdf",
                            source_url=None,
                            title_url="https://example.com/detail/100725",
                            page_url="https://example.com/pdf/100725.pdf#page=1",
                        ),
                    ),
                ),
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
