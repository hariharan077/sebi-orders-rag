from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.db import get_connection, initialize_directory_reference_schema
from app.sebi_orders_rag.directory_data.models import (
    DirectoryOfficeRecord,
    DirectoryPersonRecord,
    FetchedDirectorySource,
    OrgStructureRecord,
    ReferenceSnapshotRecord,
)
from app.sebi_orders_rag.directory_data.service import DirectoryIngestionService
from app.sebi_orders_rag.repositories.directory import DirectoryRepository

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DirectoryRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_env_file(PROJECT_ROOT / ".env")
        cls.settings = SebiOrdersRagSettings.from_env()
        with get_connection(cls.settings) as connection:
            initialize_directory_reference_schema(connection, cls.settings)
            connection.commit()

    def setUp(self) -> None:
        self.connection = psycopg.connect(self.settings.db_dsn)
        self.repository = DirectoryRepository(self.connection)

    def tearDown(self) -> None:
        self.connection.rollback()
        self.connection.close()

    def test_insert_snapshot_is_idempotent_for_same_source_hash(self) -> None:
        one = self.repository.insert_snapshot(
            source_type="test_directory_repository",
            source_url="https://example.com/directory",
            content_sha256="a" * 64,
            fetch_status="success",
            parse_status="success",
            raw_html="<html></html>",
            error=None,
        )
        two = self.repository.insert_snapshot(
            source_type="test_directory_repository",
            source_url="https://example.com/directory",
            content_sha256="a" * 64,
            fetch_status="success",
            parse_status="success",
            raw_html="<html></html>",
            error=None,
        )

        self.assertEqual(one.snapshot_id, two.snapshot_id)

    def test_replace_source_records_upserts_and_marks_missing_rows_inactive(self) -> None:
        source_type = "test_directory_repository"
        source_url = "https://example.com/directory"

        snapshot_one = self.repository.insert_snapshot(
            source_type=source_type,
            source_url=source_url,
            content_sha256="b" * 64,
            fetch_status="success",
            parse_status="success",
            raw_html="<html>one</html>",
            error=None,
        )
        initial_people = (
            DirectoryPersonRecord(
                source_type=source_type,
                source_url=source_url,
                canonical_name="Tuhin Kanta Pandey",
                designation="Chairman",
                role_group="chairperson",
            ).with_hash(),
            DirectoryPersonRecord(
                source_type=source_type,
                source_url=source_url,
                canonical_name="Amarjeet Singh",
                designation="Whole Time Member",
                role_group="wtm",
            ).with_hash(),
        )
        initial_offices = (
            DirectoryOfficeRecord(
                source_type=source_type,
                source_url=source_url,
                office_name="Southern Regional Office (SRO)",
                office_type="regional_office",
                region="south",
                address="Anna Salai, Chennai",
            ).with_hash(),
        )
        initial_org = (
            OrgStructureRecord(
                source_type=source_type,
                source_url=source_url,
                leader_name="Amarjeet Singh",
                leader_role="Whole Time Member",
                department_name="Investment Management Department",
                executive_director_name="Manoj Kumar",
            ).with_hash(),
        )
        self.repository.replace_source_records(
            source_type=source_type,
            snapshot_id=snapshot_one.snapshot_id,
            people=initial_people,
            offices=initial_offices,
            org_structure=initial_org,
        )

        snapshot_two = self.repository.insert_snapshot(
            source_type=source_type,
            source_url=source_url,
            content_sha256="c" * 64,
            fetch_status="success",
            parse_status="success",
            raw_html="<html>two</html>",
            error=None,
        )
        next_people = (
            DirectoryPersonRecord(
                source_type=source_type,
                source_url=source_url,
                canonical_name="Tuhin Kanta Pandey",
                designation="Chairman",
                role_group="chairperson",
            ).with_hash(),
        )
        self.repository.replace_source_records(
            source_type=source_type,
            snapshot_id=snapshot_two.snapshot_id,
            people=next_people,
            offices=(),
            org_structure=(),
        )

        active_people = [
            record
            for record in self.repository.list_active_people()
            if record.source_type == source_type
        ]
        active_offices = [
            record
            for record in self.repository.list_active_offices()
            if record.source_type == source_type
        ]
        active_org = [
            record
            for record in self.repository.list_active_org_structure()
            if record.source_type == source_type
        ]

        self.assertEqual(len(active_people), 1)
        self.assertEqual(active_people[0].canonical_name, "Tuhin Kanta Pandey")
        self.assertEqual(active_offices, [])
        self.assertEqual(active_org, [])

        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM sebi_people WHERE source_type = %s AND is_active = FALSE",
                (source_type,),
            )
            inactive_people = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*) FROM sebi_offices WHERE source_type = %s AND is_active = FALSE",
                (source_type,),
            )
            inactive_offices = cursor.fetchone()[0]

        self.assertEqual(inactive_people, 1)
        self.assertEqual(inactive_offices, 1)


class DirectoryRefreshBehaviorTests(unittest.TestCase):
    def test_refresh_skips_parse_and_upsert_when_content_hash_is_unchanged(self) -> None:
        repository = _FakeDirectoryRepository()
        service = DirectoryIngestionService(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=PROJECT_ROOT,
            ),
            connection=None,
            repository=repository,
            fetcher=_FakeDirectoryFetcher(
                {
                    "directory": FetchedDirectorySource(
                        source_type="directory",
                        title="SEBI Directory",
                        source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
                        raw_html="<html><body><div class='portlet1 box1 green'></div></body></html>",
                        content_sha256="d" * 64,
                    )
                }
            ),
        )

        summary = service.run(apply=True, source="directory")

        self.assertEqual(summary.source_summaries[0].parse_status, "skipped_unchanged")
        self.assertEqual(repository.replace_calls, 0)


class _FakeDirectoryRepository:
    def __init__(self) -> None:
        self.replace_calls = 0

    def get_latest_snapshot_for_source(self, *, source_type: str) -> ReferenceSnapshotRecord | None:
        return ReferenceSnapshotRecord(
            snapshot_id=11,
            source_type=source_type,
            source_url="https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp",
            fetched_at=datetime.now(timezone.utc),
            content_sha256="d" * 64,
            fetch_status="success",
            parse_status="success",
            raw_html="<html></html>",
            error=None,
        )

    def insert_snapshot(self, **kwargs) -> ReferenceSnapshotRecord:
        return ReferenceSnapshotRecord(
            snapshot_id=11,
            source_type=kwargs["source_type"],
            source_url=kwargs["source_url"],
            fetched_at=datetime.now(timezone.utc),
            content_sha256=kwargs["content_sha256"],
            fetch_status=kwargs["fetch_status"],
            parse_status=kwargs["parse_status"],
            raw_html=kwargs["raw_html"],
            error=kwargs["error"],
        )

    def replace_source_records(self, **kwargs) -> None:
        self.replace_calls += 1


class _FakeDirectoryFetcher:
    def __init__(self, payload_by_source_type: dict[str, FetchedDirectorySource]) -> None:
        self._payload_by_source_type = payload_by_source_type

    def fetch(self, source) -> FetchedDirectorySource:
        return self._payload_by_source_type[source.source_type]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
