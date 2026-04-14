from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.control import load_control_pack
from app.sebi_orders_rag.control.models import ControlPack, EvalQueryCase, StrictAnswerRule
from app.sebi_orders_rag.eval import ControlPackEvaluator
from app.sebi_orders_rag.retrieval.scoring import ChunkSearchHit, HierarchicalSearchResult, ScoreBreakdown
from app.sebi_orders_rag.schemas import ChatSessionSnapshot
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class WrongAnswerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(REAL_CONTROL_PACK)
        assert cls.pack is not None

    def test_wrong_answer_examples_no_longer_cite_known_contaminated_record_keys(self) -> None:
        for example in self.pack.wrong_answer_examples:
            with self.subTest(query=example.user_query):
                service = _build_regression_service(pack=self.pack, example=example)
                payload = service.answer_query(query=example.user_query, session_id=uuid4())
                cited_record_keys = {citation.record_key for citation in payload.citations}

                self.assertFalse(
                    set(example.incorrectly_pulled_record_keys) & cited_record_keys
                )
                if example.expected_record_key:
                    self.assertTrue(
                        payload.answer_status in {"abstained", "clarify"}
                        or cited_record_keys == {example.expected_record_key}
                    )
                else:
                    self.assertIn(payload.answer_status, {"abstained", "clarify"})

    def test_evaluator_reports_single_matter_for_strict_query(self) -> None:
        example = next(
            item
            for item in self.pack.wrong_answer_examples
            if item.expected_record_key
            == "derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1"
        )
        service = _build_regression_service(pack=self.pack, example=example)
        mini_pack = ControlPack(
            root=self.pack.root,
            document_index=self.pack.document_index,
            confusion_pairs=self.pack.confusion_pairs,
            eval_queries=(
                EvalQueryCase(
                    query=example.user_query,
                    expected_route_mode="exact_lookup",
                    expected_record_key=example.expected_record_key,
                    expected_title=example.expected_title,
                    comparison_allowed=False,
                    notes="strict query smoke",
                ),
            ),
            wrong_answer_examples=(),
            entity_aliases=self.pack.entity_aliases,
            strict_answer_rule=StrictAnswerRule(
                text=self.pack.strict_answer_rule.text,
                strict_single_matter_required=True,
            ),
            documents_by_record_key=self.pack.documents_by_record_key,
            aliases_by_record_key=self.pack.aliases_by_record_key,
            alias_variants=self.pack.alias_variants,
            confusion_map=self.pack.confusion_map,
        )

        summary = ControlPackEvaluator(service=service, control_pack=mini_pack).run(
            run_eval_queries=True,
            run_regressions=False,
        )

        self.assertEqual(summary.total_cases, 1)
        self.assertEqual(summary.failed_cases, 0)
        self.assertTrue(summary.results[0].strict_single_matter_triggered)
        self.assertTrue(summary.results[0].single_matter_rule_respected)


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _RegressionSearchService:
    def __init__(self, *, pack, example) -> None:
        self._pack = pack
        self._example = example

    def search(self, *, query: str, filters=None, top_k_docs=None, top_k_sections=None, top_k_chunks=None, strict_matter_lock=None):
        chunks: list[ChunkSearchHit] = []
        for index, record_key in enumerate(self._example.incorrectly_pulled_record_keys[:1], start=1):
            document = self._pack.documents_by_record_key.get(record_key)
            if document is not None:
                chunks.append(
                    _chunk_from_document(
                        document,
                        chunk_id=2000 + index,
                        score=0.097 - (index * 0.001),
                    )
                )

        if self._example.expected_record_key:
            document = self._pack.documents_by_record_key[self._example.expected_record_key]
            chunks.append(
                _chunk_from_document(
                    document,
                    chunk_id=2100,
                    score=0.089,
                )
            )

        return HierarchicalSearchResult(
            query=query,
            documents=(),
            sections=(),
            chunks=tuple(chunks),
            debug={
                "strict_single_matter": bool(strict_matter_lock and strict_matter_lock.strict_single_matter),
            },
        )


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _FakeSessionRepository:
    def __init__(self) -> None:
        self.snapshots = {}
        self.states = {}

    def create_session_if_missing(self, *, session_id, user_name):
        now = datetime.now(timezone.utc)
        self.snapshots.setdefault(
            session_id,
            ChatSessionSnapshot(
                session_id=session_id,
                user_name=user_name,
                created_at=now,
                updated_at=now,
                state=self.states.get(session_id),
            ),
        )

    def get_session_snapshot(self, *, session_id):
        return self.snapshots.get(session_id)

    def get_session_state(self, *, session_id):
        return self.states.get(session_id)

    def upsert_session_state(self, **kwargs) -> None:
        return None


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _FakeLlmClient:
    def complete_json(self, prompt):
        return {
            "answer_status": "answered",
            "answer_text": "Grounded answer [1].",
            "cited_numbers": [1],
        }


def _chunk_from_document(document, *, chunk_id: int, score: float) -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        document_version_id=document.document_version_id or chunk_id,
        document_id=document.document_version_id or chunk_id,
        record_key=document.record_key,
        bucket_name=document.bucket_category,
        external_record_id=document.record_key.split(":")[-1],
        order_date=document.order_date or date(2026, 1, 1),
        title=document.exact_title,
        chunk_index=0,
        page_start=1,
        page_end=2,
        section_key="section-1",
        section_type="operative_order",
        section_title="ORDER",
        heading_path="ORDER",
        detail_url=f"https://example.com/detail/{chunk_id}",
        pdf_url=f"https://example.com/pdf/{chunk_id}.pdf",
        chunk_text=document.short_summary or document.exact_title,
        token_count=60,
        score=ScoreBreakdown(
            combined_score=score,
            final_score=score,
        ),
    )


def _build_regression_service(*, pack, example) -> AdaptiveRagAnswerService:
    settings = SebiOrdersRagSettings(
        db_dsn="postgresql://unused",
        data_root=Path(".").resolve(),
        control_pack_root=REAL_CONTROL_PACK,
        low_confidence_threshold=0.35,
        enable_memory=True,
    )
    return AdaptiveRagAnswerService(
        settings=settings,
        connection=_FakeConnection(),
        search_service=_RegressionSearchService(pack=pack, example=example),
        retrieval_repository=_FakeRetrievalRepository(),
        session_repository=_FakeSessionRepository(),
        answer_repository=_FakeAnswerRepository(),
        llm_client=_FakeLlmClient(),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
