"""Persistence helpers for structured SEBI directory reference data."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..directory_data.models import (
    BoardMemberRecord,
    DirectoryOfficeRecord,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
    OrgStructureRecord,
    ReferenceSnapshotRecord,
)


class DirectoryRepository:
    """Read and write structured SEBI directory/reference rows in Postgres."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def insert_snapshot(
        self,
        *,
        source_type: str,
        source_url: str,
        content_sha256: str,
        fetch_status: str,
        parse_status: str,
        raw_html: str | None,
        error: str | None,
    ) -> ReferenceSnapshotRecord:
        sql = """
            INSERT INTO sebi_reference_snapshots (
                source_type,
                source_url,
                content_sha256,
                fetch_status,
                parse_status,
                raw_html,
                error
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_type, content_sha256) DO UPDATE
            SET
                source_url = EXCLUDED.source_url,
                fetched_at = NOW(),
                fetch_status = EXCLUDED.fetch_status,
                parse_status = EXCLUDED.parse_status,
                raw_html = EXCLUDED.raw_html,
                error = EXCLUDED.error
            RETURNING
                snapshot_id,
                source_type,
                source_url,
                fetched_at,
                content_sha256,
                fetch_status,
                parse_status,
                raw_html,
                error
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    source_type,
                    source_url,
                    content_sha256,
                    fetch_status,
                    parse_status,
                    raw_html,
                    error,
                ),
            )
            row = cursor.fetchone()
        return ReferenceSnapshotRecord(*row)

    def get_latest_snapshot_for_source(self, *, source_type: str) -> ReferenceSnapshotRecord | None:
        sql = """
            SELECT
                snapshot_id,
                source_type,
                source_url,
                fetched_at,
                content_sha256,
                fetch_status,
                parse_status,
                raw_html,
                error
            FROM sebi_reference_snapshots
            WHERE source_type = %s
            ORDER BY fetched_at DESC, snapshot_id DESC
            LIMIT 1
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql, (source_type,))
            row = cursor.fetchone()
        return ReferenceSnapshotRecord(*row) if row else None

    def replace_source_records(
        self,
        *,
        source_type: str,
        snapshot_id: int,
        people: Sequence[DirectoryPersonRecord] = (),
        board_members: Sequence[BoardMemberRecord] = (),
        offices: Sequence[DirectoryOfficeRecord] = (),
        org_structure: Sequence[OrgStructureRecord] = (),
    ) -> None:
        self._replace_people(source_type=source_type, snapshot_id=snapshot_id, records=people)
        self._replace_board_members(source_type=source_type, snapshot_id=snapshot_id, records=board_members)
        self._replace_offices(source_type=source_type, snapshot_id=snapshot_id, records=offices)
        self._replace_org_structure(source_type=source_type, snapshot_id=snapshot_id, records=org_structure)

    def load_active_dataset(self) -> DirectoryReferenceDataset:
        return DirectoryReferenceDataset(
            people=self.list_active_people(),
            board_members=self.list_active_board_members(),
            offices=self.list_active_offices(),
            org_structure=self.list_active_org_structure(),
        )

    def list_active_people(self) -> tuple[DirectoryPersonRecord, ...]:
        sql = """
            SELECT
                person_id,
                source_type,
                source_url,
                snapshot_id,
                canonical_name,
                designation,
                role_group,
                department_name,
                office_name,
                email,
                phone,
                date_of_joining,
                staff_no,
                is_active,
                row_sha256
            FROM sebi_people
            WHERE is_active = TRUE
            ORDER BY canonical_name ASC, source_type ASC, updated_at DESC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            DirectoryPersonRecord(
                person_id=int(row[0]),
                source_type=row[1],
                source_url=row[2],
                snapshot_id=row[3],
                canonical_name=row[4],
                designation=row[5],
                role_group=row[6],
                department_name=row[7],
                office_name=row[8],
                email=row[9],
                phone=row[10],
                date_of_joining=row[11],
                staff_no=row[12],
                is_active=bool(row[13]),
                row_sha256=row[14],
            )
            for row in rows
        )

    def list_active_board_members(self) -> tuple[BoardMemberRecord, ...]:
        sql = """
            SELECT
                board_member_id,
                source_url,
                snapshot_id,
                canonical_name,
                board_role,
                category,
                is_active,
                row_sha256,
                source_type
            FROM sebi_board_members
            WHERE is_active = TRUE
            ORDER BY canonical_name ASC, updated_at DESC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            BoardMemberRecord(
                board_member_id=int(row[0]),
                source_url=row[1],
                snapshot_id=row[2],
                canonical_name=row[3],
                board_role=row[4],
                category=row[5],
                is_active=bool(row[6]),
                row_sha256=row[7],
                source_type=row[8],
            )
            for row in rows
        )

    def list_active_offices(self) -> tuple[DirectoryOfficeRecord, ...]:
        sql = """
            SELECT
                office_id,
                source_type,
                source_url,
                snapshot_id,
                office_name,
                office_type,
                region,
                address,
                email,
                phone,
                fax,
                city,
                state,
                is_active,
                row_sha256
            FROM sebi_offices
            WHERE is_active = TRUE
            ORDER BY office_name ASC, source_type ASC, updated_at DESC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            DirectoryOfficeRecord(
                office_id=int(row[0]),
                source_type=row[1],
                source_url=row[2],
                snapshot_id=row[3],
                office_name=row[4],
                office_type=row[5],
                region=row[6],
                address=row[7],
                email=row[8],
                phone=row[9],
                fax=row[10],
                city=row[11],
                state=row[12],
                is_active=bool(row[13]),
                row_sha256=row[14],
            )
            for row in rows
        )

    def list_active_org_structure(self) -> tuple[OrgStructureRecord, ...]:
        sql = """
            SELECT
                org_id,
                source_type,
                source_url,
                snapshot_id,
                leader_name,
                leader_role,
                department_name,
                executive_director_name,
                executive_director_email,
                executive_director_phone,
                is_active,
                row_sha256
            FROM sebi_org_structure
            WHERE is_active = TRUE
            ORDER BY leader_name ASC, department_name ASC, updated_at DESC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            OrgStructureRecord(
                org_id=int(row[0]),
                source_type=row[1],
                source_url=row[2],
                snapshot_id=row[3],
                leader_name=row[4],
                leader_role=row[5],
                department_name=row[6],
                executive_director_name=row[7],
                executive_director_email=row[8],
                executive_director_phone=row[9],
                is_active=bool(row[10]),
                row_sha256=row[11],
            )
            for row in rows
        )

    def _replace_people(
        self,
        *,
        source_type: str,
        snapshot_id: int,
        records: Sequence[DirectoryPersonRecord],
    ) -> None:
        sql = """
            INSERT INTO sebi_people (
                source_type,
                source_url,
                snapshot_id,
                canonical_name,
                designation,
                role_group,
                department_name,
                office_name,
                email,
                phone,
                date_of_joining,
                staff_no,
                is_active,
                row_sha256
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s)
            ON CONFLICT (source_type, row_sha256) DO UPDATE
            SET
                source_url = EXCLUDED.source_url,
                snapshot_id = EXCLUDED.snapshot_id,
                canonical_name = EXCLUDED.canonical_name,
                designation = EXCLUDED.designation,
                role_group = EXCLUDED.role_group,
                department_name = EXCLUDED.department_name,
                office_name = EXCLUDED.office_name,
                email = EXCLUDED.email,
                phone = EXCLUDED.phone,
                date_of_joining = EXCLUDED.date_of_joining,
                staff_no = EXCLUDED.staff_no,
                is_active = TRUE,
                updated_at = NOW()
        """
        self._upsert_records(
            sql=sql,
            payloads=[
                (
                    record.source_type,
                    record.source_url,
                    snapshot_id,
                    record.canonical_name,
                    record.designation,
                    record.role_group,
                    record.department_name,
                    record.office_name,
                    record.email,
                    record.phone,
                    record.date_of_joining,
                    record.staff_no,
                    record.row_sha256,
                )
                for record in records
                if record.row_sha256
            ],
        )
        self._mark_missing_inactive(
            table_name="sebi_people",
            source_type=source_type,
            row_hashes=[record.row_sha256 for record in records if record.row_sha256],
        )

    def _replace_board_members(
        self,
        *,
        source_type: str,
        snapshot_id: int,
        records: Sequence[BoardMemberRecord],
    ) -> None:
        sql = """
            INSERT INTO sebi_board_members (
                source_type,
                source_url,
                snapshot_id,
                canonical_name,
                board_role,
                category,
                is_active,
                row_sha256
            )
            VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s)
            ON CONFLICT (source_type, row_sha256) DO UPDATE
            SET
                source_url = EXCLUDED.source_url,
                snapshot_id = EXCLUDED.snapshot_id,
                canonical_name = EXCLUDED.canonical_name,
                board_role = EXCLUDED.board_role,
                category = EXCLUDED.category,
                is_active = TRUE,
                updated_at = NOW()
        """
        self._upsert_records(
            sql=sql,
            payloads=[
                (
                    record.source_type,
                    record.source_url,
                    snapshot_id,
                    record.canonical_name,
                    record.board_role,
                    record.category,
                    record.row_sha256,
                )
                for record in records
                if record.row_sha256
            ],
        )
        self._mark_missing_inactive(
            table_name="sebi_board_members",
            source_type=source_type,
            row_hashes=[record.row_sha256 for record in records if record.row_sha256],
        )

    def _replace_offices(
        self,
        *,
        source_type: str,
        snapshot_id: int,
        records: Sequence[DirectoryOfficeRecord],
    ) -> None:
        sql = """
            INSERT INTO sebi_offices (
                source_type,
                source_url,
                snapshot_id,
                office_name,
                office_type,
                region,
                address,
                email,
                phone,
                fax,
                city,
                state,
                is_active,
                row_sha256
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s)
            ON CONFLICT (source_type, row_sha256) DO UPDATE
            SET
                source_url = EXCLUDED.source_url,
                snapshot_id = EXCLUDED.snapshot_id,
                office_name = EXCLUDED.office_name,
                office_type = EXCLUDED.office_type,
                region = EXCLUDED.region,
                address = EXCLUDED.address,
                email = EXCLUDED.email,
                phone = EXCLUDED.phone,
                fax = EXCLUDED.fax,
                city = EXCLUDED.city,
                state = EXCLUDED.state,
                is_active = TRUE,
                updated_at = NOW()
        """
        self._upsert_records(
            sql=sql,
            payloads=[
                (
                    record.source_type,
                    record.source_url,
                    snapshot_id,
                    record.office_name,
                    record.office_type,
                    record.region,
                    record.address,
                    record.email,
                    record.phone,
                    record.fax,
                    record.city,
                    record.state,
                    record.row_sha256,
                )
                for record in records
                if record.row_sha256
            ],
        )
        self._mark_missing_inactive(
            table_name="sebi_offices",
            source_type=source_type,
            row_hashes=[record.row_sha256 for record in records if record.row_sha256],
        )

    def _replace_org_structure(
        self,
        *,
        source_type: str,
        snapshot_id: int,
        records: Sequence[OrgStructureRecord],
    ) -> None:
        sql = """
            INSERT INTO sebi_org_structure (
                source_type,
                source_url,
                snapshot_id,
                leader_name,
                leader_role,
                department_name,
                executive_director_name,
                executive_director_email,
                executive_director_phone,
                is_active,
                row_sha256
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s)
            ON CONFLICT (source_type, row_sha256) DO UPDATE
            SET
                source_url = EXCLUDED.source_url,
                snapshot_id = EXCLUDED.snapshot_id,
                leader_name = EXCLUDED.leader_name,
                leader_role = EXCLUDED.leader_role,
                department_name = EXCLUDED.department_name,
                executive_director_name = EXCLUDED.executive_director_name,
                executive_director_email = EXCLUDED.executive_director_email,
                executive_director_phone = EXCLUDED.executive_director_phone,
                is_active = TRUE,
                updated_at = NOW()
        """
        self._upsert_records(
            sql=sql,
            payloads=[
                (
                    record.source_type,
                    record.source_url,
                    snapshot_id,
                    record.leader_name,
                    record.leader_role,
                    record.department_name,
                    record.executive_director_name,
                    record.executive_director_email,
                    record.executive_director_phone,
                    record.row_sha256,
                )
                for record in records
                if record.row_sha256
            ],
        )
        self._mark_missing_inactive(
            table_name="sebi_org_structure",
            source_type=source_type,
            row_hashes=[record.row_sha256 for record in records if record.row_sha256],
        )

    def _upsert_records(self, *, sql: str, payloads: Iterable[tuple[Any, ...]]) -> None:
        rows = list(payloads)
        if not rows:
            return
        with self._connection.cursor() as cursor:
            cursor.executemany(sql, rows)

    def _mark_missing_inactive(
        self,
        *,
        table_name: str,
        source_type: str,
        row_hashes: Sequence[str],
    ) -> None:
        with self._connection.cursor() as cursor:
            if row_hashes:
                cursor.execute(
                    f"""
                    UPDATE {table_name}
                    SET
                        is_active = FALSE,
                        updated_at = NOW()
                    WHERE source_type = %s
                      AND is_active = TRUE
                      AND NOT (row_sha256 = ANY(%s))
                    """,
                    (source_type, list(row_hashes)),
                )
            else:
                cursor.execute(
                    f"""
                    UPDATE {table_name}
                    SET
                        is_active = FALSE,
                        updated_at = NOW()
                    WHERE source_type = %s
                      AND is_active = TRUE
                    """,
                    (source_type,),
                )
