from __future__ import annotations

import unittest
from csv import DictWriter
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from app.sebi_orders_rag.answering.answer_service import AdaptiveRagAnswerService
from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.constants import MANIFEST_COLUMNS
from app.sebi_orders_rag.corpus_stats import (
    BucketCorpusStats,
    CorpusStatsRepository,
    CorpusStatsService,
    CorpusStatsSnapshot,
)
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionSnapshot


class CorpusMetadataQueryTests(unittest.TestCase):
    def test_router_sends_sat_count_question_to_corpus_metadata(self) -> None:
        decision = AdaptiveQueryRouter().decide(query="how many sat cases do we have")

        self.assertEqual(decision.route_mode, "corpus_metadata")
        self.assertTrue(decision.analysis.appears_corpus_metadata_query)

    def test_answer_service_answers_sat_count_from_corpus_stats(self) -> None:
        service = AdaptiveRagAnswerService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                enable_memory=True,
            ),
            connection=_FakeConnection(),
            search_service=_ExplodingSearchService(),
            retrieval_repository=_FakeRetrievalRepository(),
            session_repository=_FakeSessionRepository(),
            answer_repository=_FakeAnswerRepository(),
            corpus_stats_service=CorpusStatsService(repository=_FakeCorpusStatsRepository()),
            llm_client=_ExplodingLlmClient(),
        )

        payload = service.answer_query(query="how many sat cases do we have", session_id=uuid4())

        self.assertEqual(payload.route_mode, "corpus_metadata")
        self.assertEqual(payload.answer_status, "answered")
        self.assertIn("12 records in orders-of-sat", payload.answer_text)
        self.assertTrue(payload.debug["corpus_metadata_debug"]["used"])

    def test_date_range_query_for_sat_bucket_answers_directly(self) -> None:
        answer = CorpusStatsService(repository=_FakeCorpusStatsRepository()).answer_query(
            "from which dates for sat cases we have"
        )

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("2014-01-03 to 2026-03-18", answer.answer_text)

    def test_repository_does_not_count_empty_local_filename_as_local_pdf(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            bucket_dir = data_root / "sebi-orders-pdfs" / "orders-of-sat"
            bucket_dir.mkdir(parents=True)
            (bucket_dir / "available.pdf").write_bytes(b"%PDF-1.4")

            manifest_path = bucket_dir / "orders_manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "record_key": "external:1",
                        "bucket_name": "orders-of-sat",
                        "order_date": "2015-06-29",
                        "title": "Available SAT order",
                        "external_record_id": "1",
                        "detail_url": "https://example.com/1",
                        "pdf_url": "https://example.com/1.pdf",
                        "local_filename": "available.pdf",
                        "status": "downloaded",
                        "error": "",
                        "first_seen_at": "2026-04-09T17:06:07.668891+00:00",
                        "last_seen_at": "2026-04-09T17:06:07.668891+00:00",
                    }
                )
                writer.writerow(
                    {
                        "record_key": "external:2",
                        "bucket_name": "orders-of-sat",
                        "order_date": "2015-06-30",
                        "title": "Missing SAT order",
                        "external_record_id": "2",
                        "detail_url": "https://example.com/2",
                        "pdf_url": "",
                        "local_filename": "",
                        "status": "failed",
                        "error": "detail page did not expose a PDF attachment",
                        "first_seen_at": "2026-04-09T17:06:07.668891+00:00",
                        "last_seen_at": "2026-04-09T17:06:07.668891+00:00",
                    }
                )

            snapshot = CorpusStatsRepository(data_root=data_root).compute_snapshot()
            sat_bucket = next(
                item for item in snapshot.bucket_stats if item.bucket_name == "orders-of-sat"
            )

            self.assertEqual(sat_bucket.total_manifest_rows, 2)
            self.assertEqual(sat_bucket.local_pdf_count, 1)
            self.assertEqual(sat_bucket.missing_count, 1)


class _FakeConnection:
    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _ExplodingSearchService:
    def search(self, **kwargs):
        raise AssertionError("corpus metadata route should not use retrieval")


class _FakeRetrievalRepository:
    def find_exact_lookup_candidates(self, *, query: str, limit: int, query_variants=None):
        return []

    def resolve_current_document_version_ids(self, *, record_keys=(), document_ids=()):
        return ()


class _FakeSessionRepository:
    def create_session_if_missing(self, *, session_id, user_name):
        now = datetime.now(timezone.utc)
        self._snapshot = ChatSessionSnapshot(
            session_id=session_id,
            user_name=user_name,
            created_at=now,
            updated_at=now,
            state=None,
        )

    def get_session_snapshot(self, *, session_id):
        return getattr(self, "_snapshot", None)

    def get_session_state(self, *, session_id):
        return None

    def upsert_session_state(self, **kwargs):
        return None


class _FakeAnswerRepository:
    def insert_retrieval_log(self, **kwargs):
        return 1

    def insert_answer_log(self, **kwargs):
        return 1


class _ExplodingLlmClient:
    def complete_json(self, prompt):
        raise AssertionError("corpus metadata route should not call the LLM")


class _FakeCorpusStatsRepository(CorpusStatsRepository):
    def __init__(self) -> None:
        pass

    def load_snapshot(self) -> CorpusStatsSnapshot:
        return CorpusStatsSnapshot(
            generated_at=datetime(2026, 4, 12, 8, 0, tzinfo=timezone.utc),
            bucket_stats=(
                BucketCorpusStats(
                    bucket_name="orders-of-sat",
                    total_manifest_rows=12,
                    local_pdf_count=10,
                    missing_count=2,
                    min_date=date(2014, 1, 3),
                    max_date=date(2026, 3, 18),
                ),
                BucketCorpusStats(
                    bucket_name="settlement-orders",
                    total_manifest_rows=8,
                    local_pdf_count=8,
                    missing_count=0,
                    min_date=date(2017, 5, 5),
                    max_date=date(2026, 2, 11),
                ),
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
