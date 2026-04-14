from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.control.models import ControlPack, DocumentIndexRow, EntityAliasRow, StrictAnswerRule
from app.sebi_orders_rag.metadata.models import MetadataChunkText, StoredOrderMetadata
from app.sebi_orders_rag.metadata.service import OrderMetadataService
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot, ChatSessionStateRecord


class PatelFollowUpRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control_pack = _build_control_pack()

    def test_router_keeps_deictic_key_details_follow_up_inside_active_matter(self) -> None:
        router = AdaptiveQueryRouter(control_pack=self.control_pack)
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_record_keys=("external:99695",),
            active_document_version_ids=(99695,),
            active_primary_title="CONFIRMATORY ORDER IN THE MATTER OF M/S PATEL WEALTH ADVISORS PRIVATE LIMITED",
        )

        decision = router.decide(
            query="could you explain the key details of this case?",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "memory_scoped_rag")
        self.assertTrue(decision.analysis.active_order_override)
        self.assertFalse(decision.analysis.strict_scope_required)
        self.assertIn("active_matter_deictic_follow_up", decision.analysis.strict_lock_reason_codes)

    def test_action_taken_follow_up_prefers_confirmatory_order_outcome(self) -> None:
        session_id = uuid4()
        state = ChatSessionStateRecord(
            session_id=session_id,
            active_record_keys=("external:99695",),
            active_document_version_ids=(99695,),
            active_primary_title="CONFIRMATORY ORDER IN THE MATTER OF M/S PATEL WEALTH ADVISORS PRIVATE LIMITED",
        )
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(state=state),
            answer_repository=_FakeAnswerRepository(),
            metadata_service=OrderMetadataService(repository=_PatelMetadataRepository()),
            llm_client=_ExplodingLlmClient(),
            router=AdaptiveQueryRouter(control_pack=self.control_pack),
        )

        payload = service.answer_query(
            query="what was the action taken",
            session_id=session_id,
        )

        self.assertEqual(payload.route_mode, "memory_scoped_rag")
        self.assertEqual(payload.answer_status, "answered")
        self.assertTrue(payload.debug["metadata_debug"]["used"])
        self.assertEqual(payload.debug["metadata_debug"]["follow_up_intent"], "final_direction")
        self.assertIn("confirms the directions issued vide the Interim Order dated April 28, 2025", payload.answer_text)
        self.assertIn("modified to the extent", payload.answer_text)
        self.assertIn("shall stand revoked", payload.answer_text)
        self.assertIn("remain in force", payload.answer_text)
        self.assertNotIn("NSE's Order dated April 16, 2025", payload.answer_text)
        self.assertNotIn("which.", payload.answer_text)


def _build_control_pack() -> ControlPack:
    patel_doc = DocumentIndexRow(
        record_key="external:99695",
        exact_title="CONFIRMATORY ORDER IN THE MATTER OF M/S PATEL WEALTH ADVISORS PRIVATE LIMITED",
        bucket_category="orders-of-whole-time-member",
        order_date=date(2025, 5, 1),
        main_entities=("Patel Wealth Advisors Private Limited",),
        short_summary="Confirmatory order concerning Patel Wealth Advisors Private Limited.",
        summary_source="fixture",
        procedural_type="confirmatory_order",
        manifest_status="ingested",
        manifest_error=None,
        ingested=True,
        document_version_id=99695,
        detail_url="https://example.com/patel-wealth",
        pdf_url="https://example.com/patel-wealth.pdf",
        local_filename="patel_wealth.pdf",
    )
    alias_row = EntityAliasRow(
        canonical_name="Patel Wealth Advisors Private Limited",
        short_name="Patel Wealth Advisors",
        abbreviations=("PWAPL",),
        old_name=None,
        new_name=None,
        related_record_keys=(patel_doc.record_key,),
        related_titles=(patel_doc.exact_title,),
    )
    return ControlPack(
        root=Path(".").resolve(),
        document_index=(patel_doc,),
        confusion_pairs=(),
        eval_queries=(),
        wrong_answer_examples=(),
        entity_aliases=(alias_row,),
        strict_answer_rule=StrictAnswerRule(
            text="Stay inside one matter.",
            strict_single_matter_required=True,
        ),
        documents_by_record_key={patel_doc.record_key: patel_doc},
        aliases_by_record_key={patel_doc.record_key: (alias_row,)},
        alias_variants={
            "patel wealth advisors": (alias_row,),
            "patel wealth advisors private limited": (alias_row,),
            "pwapl": (alias_row,),
        },
        confusion_map={},
    )


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return (99695,)


class _FakeSessionRepository:
    def __init__(self, *, state: ChatSessionStateRecord) -> None:
        self._state = state
        now = datetime.now(timezone.utc)
        self._snapshot = ChatSessionSnapshot(
            session_id=state.session_id,
            user_name=None,
            created_at=now,
            updated_at=now,
            state=state,
        )

    def create_session_if_missing(self, *, session_id, user_name):
        return None

    def get_session_snapshot(self, *, session_id):
        return self._snapshot

    def get_session_state(self, *, session_id):
        return self._state

    def upsert_session_state(self, **kwargs):
        self._state = ChatSessionStateRecord(
            session_id=kwargs["session_id"],
            active_document_ids=tuple(kwargs.get("active_document_ids", ())),
            active_document_version_ids=tuple(kwargs.get("active_document_version_ids", ())),
            active_record_keys=tuple(kwargs.get("active_record_keys", ())),
            active_entities=tuple(kwargs.get("active_entities", ())),
            active_bucket_names=tuple(kwargs.get("active_bucket_names", ())),
            active_primary_title=kwargs.get("active_primary_title"),
            active_primary_entity=kwargs.get("active_primary_entity"),
            active_signatory_name=kwargs.get("active_signatory_name"),
            active_signatory_designation=kwargs.get("active_signatory_designation"),
            active_order_date=kwargs.get("active_order_date"),
            active_order_place=kwargs.get("active_order_place"),
            active_legal_provisions=tuple(kwargs.get("active_legal_provisions", ())),
            last_chunk_ids=tuple(kwargs.get("last_chunk_ids", ())),
            last_citation_chunk_ids=tuple(kwargs.get("last_citation_chunk_ids", ())),
            grounded_summary=kwargs.get("grounded_summary"),
            current_lookup_family=kwargs.get("current_lookup_family"),
            current_lookup_focus=kwargs.get("current_lookup_focus"),
            current_lookup_query=kwargs.get("current_lookup_query"),
            clarification_context=kwargs.get("clarification_context"),
        )


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("Patel follow-ups should not fall back to general retrieval")


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("Patel follow-ups should not call the LLM")


class _PatelMetadataRepository:
    def fetch_order_metadata(self, *, document_version_ids):
        if 99695 not in set(document_version_ids):
            return ()
        return (
            StoredOrderMetadata(
                document_version_id=99695,
                document_id=99695,
                record_key="external:99695",
                title="CONFIRMATORY ORDER IN THE MATTER OF M/S PATEL WEALTH ADVISORS PRIVATE LIMITED",
                detail_url="https://example.com/patel-wealth",
                pdf_url="https://example.com/patel-wealth.pdf",
                order_date=date(2025, 5, 1),
            ),
        )

    def load_chunks(self, *, document_version_id: int):
        if document_version_id != 99695:
            return ()
        return (
            MetadataChunkText(
                chunk_id=1377,
                page_start=10,
                page_end=10,
                section_type="directions",
                section_title="Directions",
                text=(
                    "directions as per NSE's Order dated April 16, 2025 (NSE Order), which\n\n"
                    "only issued a warning to Noticee no.1, holding that, \"In view of the fact that Noticee has taken action at its end...\""
                ),
            ),
            MetadataChunkText(
                chunk_id=1389,
                page_start=18,
                page_end=18,
                section_type="operative_order",
                section_title="Order",
                text=(
                    "24. I note that as an interim measure, Noticees have been restrained, inter alia, from buying, selling or dealing in securities. "
                    "In view of these submissions, I direct the investigating authority to confirm the facts essential to determine the role of these directors.\n\n"
                    "25. Further, I note that the Noticees have complied with the directions stipulated in the Interim Order and deposited the alleged unlawful gains as required. "
                    "Accordingly, I proceed to modify the direction issued in this regards in the Interim Order."
                ),
            ),
            MetadataChunkText(
                chunk_id=1390,
                page_start=18,
                page_end=19,
                section_type="operative_order",
                section_title="Order",
                text=(
                    "E. ORDER\n\n"
                    "26. In view of the above, I, in exercise of the powers conferred upon me under subsections (1) and (4) of section 11 and sub-section (1) of section 11B read with\n\n"
                    "section 19 of the SEBI Act, 1992, hereby confirm the directions issued vide the Interim Order dated April 28, 2025.\n\n"
                    "27. Further, the directions issued vide the Interim Order in sub-para II of para 86 qua Noticee nos. 2 to 5 are modified to the extent that the restraint imposed on them from accessing the securities market and from buying, selling or otherwise dealing\n\n"
                    "in securities, directly or indirectly, shall stand revoked. However, directions in the\n\n"
                    "said para, qua the Noticee No. 1 will remain in force.\n\n"
                    "28. The observations made in the present Order are tentative in nature, pending detailed investigation."
                ),
            ),
        )

    def fetch_legal_provisions(self, *, document_version_ids):
        return ()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
