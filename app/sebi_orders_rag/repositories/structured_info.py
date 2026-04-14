"""Repository for canonical structured SEBI current-info data."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..directory_data.models import DirectoryReferenceDataset, OrgStructureRecord
from ..repositories.directory import DirectoryRepository
from ..structured_info.aggregates import build_snapshot
from ..structured_info.canonical_models import (
    CanonicalOfficeRecord,
    CanonicalPersonRecord,
    DepartmentCountRecord,
    DesignationCountRecord,
    OfficeCountRecord,
    RoleCountRecord,
    StructuredInfoSnapshot,
)
from ..structured_info.canonicalize_offices import canonicalize_offices
from ..structured_info.canonicalize_people import canonicalize_people


class StructuredInfoRepository:
    """Load, persist, and refresh canonical structured current-info tables."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._directory_repository = DirectoryRepository(connection)

    def load_raw_dataset(self) -> DirectoryReferenceDataset:
        return self._directory_repository.load_active_dataset()

    def build_snapshot_from_raw_dataset(
        self,
        raw_dataset: DirectoryReferenceDataset,
    ) -> StructuredInfoSnapshot:
        return build_structured_info_snapshot(raw_dataset)

    def refresh_from_raw_dataset(
        self,
        raw_dataset: DirectoryReferenceDataset | None = None,
    ) -> StructuredInfoSnapshot:
        dataset = raw_dataset or self.load_raw_dataset()
        snapshot = self.build_snapshot_from_raw_dataset(dataset)
        self._replace_canonical_people(snapshot.people)
        self._replace_canonical_offices(snapshot.offices)
        self._replace_designation_counts(snapshot.designation_counts)
        self._replace_office_counts(snapshot.office_counts)
        self._replace_department_counts(snapshot.department_counts)
        self._replace_role_counts(snapshot.role_counts)
        return snapshot

    def load_snapshot(self) -> StructuredInfoSnapshot:
        raw_dataset = self.load_raw_dataset()
        try:
            people = self.list_canonical_people()
            offices = self.list_canonical_offices()
            designation_counts = self.list_designation_counts()
            office_counts = self.list_office_counts()
            department_counts = self.list_department_counts()
            role_counts = self.list_role_counts()
        except Exception:
            return self.build_snapshot_from_raw_dataset(raw_dataset)
        if not people and (raw_dataset.people or raw_dataset.board_members or raw_dataset.offices):
            return self.build_snapshot_from_raw_dataset(raw_dataset)
        return StructuredInfoSnapshot(
            people=people,
            offices=offices,
            org_structure=raw_dataset.org_structure,
            raw_people=raw_dataset.people,
            raw_board_members=raw_dataset.board_members,
            raw_offices=raw_dataset.offices,
            designation_counts=designation_counts,
            office_counts=office_counts,
            department_counts=department_counts,
            role_counts=role_counts,
            raw_people_count=len(raw_dataset.people),
            raw_board_member_count=len(raw_dataset.board_members),
            raw_office_count=len(raw_dataset.offices),
            raw_org_count=len(raw_dataset.org_structure),
        )

    def list_canonical_people(self) -> tuple[CanonicalPersonRecord, ...]:
        sql = """
            SELECT
                canonical_person_id,
                canonical_name,
                normalized_name_key,
                aliases,
                designation,
                designation_group,
                department_name,
                department_aliases,
                office_name,
                office_city,
                office_region,
                office_aliases,
                email,
                phone,
                date_of_joining,
                staff_no,
                role_group,
                board_role,
                board_category,
                source_priority,
                active_status,
                source_types,
                source_urls,
                source_row_count,
                merge_notes,
                merged_row_keys
            FROM sebi_canonical_people
            WHERE active_status = TRUE
            ORDER BY canonical_name ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            CanonicalPersonRecord(
                canonical_person_id=row[0],
                canonical_name=row[1],
                normalized_name_key=row[2],
                aliases=_json_text_tuple(row[3]),
                designation=row[4],
                designation_group=row[5],
                department_name=row[6],
                department_aliases=_json_text_tuple(row[7]),
                office_name=row[8],
                office_city=row[9],
                office_region=row[10],
                office_aliases=_json_text_tuple(row[11]),
                email=row[12],
                phone=row[13],
                date_of_joining=row[14],
                staff_no=row[15],
                role_group=row[16],
                board_role=row[17],
                board_category=row[18],
                source_priority=int(row[19] or 0),
                active_status=bool(row[20]),
                source_types=_json_text_tuple(row[21]),
                source_urls=_json_text_tuple(row[22]),
                source_row_count=int(row[23] or 0),
                merge_notes=_json_text_tuple(row[24]),
                merged_row_keys=_json_text_tuple(row[25]),
            )
            for row in rows
        )

    def list_canonical_offices(self) -> tuple[CanonicalOfficeRecord, ...]:
        sql = """
            SELECT
                canonical_office_id,
                office_name,
                normalized_office_key,
                office_aliases,
                office_type,
                city,
                state,
                region,
                address,
                email,
                phone,
                fax,
                source_priority,
                active_status,
                source_types,
                source_urls,
                source_row_count
            FROM sebi_canonical_offices
            WHERE active_status = TRUE
            ORDER BY office_name ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            CanonicalOfficeRecord(
                canonical_office_id=row[0],
                office_name=row[1],
                normalized_office_key=row[2],
                office_aliases=_json_text_tuple(row[3]),
                office_type=row[4],
                city=row[5],
                state=row[6],
                region=row[7],
                address=row[8],
                email=row[9],
                phone=row[10],
                fax=row[11],
                source_priority=int(row[12] or 0),
                active_status=bool(row[13]),
                source_types=_json_text_tuple(row[14]),
                source_urls=_json_text_tuple(row[15]),
                source_row_count=int(row[16] or 0),
            )
            for row in rows
        )

    def list_designation_counts(self) -> tuple[DesignationCountRecord, ...]:
        sql = """
            SELECT designation_group, designation_label, people_count
            FROM sebi_designation_counts
            ORDER BY designation_group ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            DesignationCountRecord(
                designation_group=row[0],
                designation_label=row[1],
                people_count=int(row[2] or 0),
            )
            for row in rows
        )

    def list_office_counts(self) -> tuple[OfficeCountRecord, ...]:
        sql = """
            SELECT canonical_office_id, office_name, city, region, people_count, office_count
            FROM sebi_office_counts
            ORDER BY office_name ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            OfficeCountRecord(
                canonical_office_id=row[0],
                office_name=row[1],
                city=row[2],
                region=row[3],
                people_count=int(row[4] or 0),
                office_count=int(row[5] or 0),
            )
            for row in rows
        )

    def list_department_counts(self) -> tuple[DepartmentCountRecord, ...]:
        sql = """
            SELECT department_name, people_count
            FROM sebi_department_counts
            ORDER BY department_name ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            DepartmentCountRecord(
                department_name=row[0],
                people_count=int(row[1] or 0),
            )
            for row in rows
        )

    def list_role_counts(self) -> tuple[RoleCountRecord, ...]:
        sql = """
            SELECT role_key, role_label, people_count
            FROM sebi_role_counts
            ORDER BY role_key ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return tuple(
            RoleCountRecord(
                role_key=row[0],
                role_label=row[1],
                people_count=int(row[2] or 0),
            )
            for row in rows
        )

    def raw_row_counts_by_source(self) -> dict[str, int]:
        sql = """
            SELECT source_type, COUNT(*)
            FROM (
                SELECT source_type FROM sebi_people WHERE is_active = TRUE
                UNION ALL
                SELECT source_type FROM sebi_board_members WHERE is_active = TRUE
                UNION ALL
                SELECT source_type FROM sebi_offices WHERE is_active = TRUE
                UNION ALL
                SELECT source_type FROM sebi_org_structure WHERE is_active = TRUE
            ) rows
            GROUP BY source_type
            ORDER BY source_type ASC
        """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return {row[0]: int(row[1] or 0) for row in rows}

    def _replace_canonical_people(
        self,
        people: tuple[CanonicalPersonRecord, ...],
    ) -> None:
        delete_sql = "DELETE FROM sebi_canonical_people"
        insert_sql = """
            INSERT INTO sebi_canonical_people (
                canonical_person_id,
                canonical_name,
                normalized_name_key,
                aliases,
                designation,
                designation_group,
                department_name,
                department_aliases,
                office_name,
                office_city,
                office_region,
                office_aliases,
                email,
                phone,
                date_of_joining,
                staff_no,
                role_group,
                board_role,
                board_category,
                source_priority,
                active_status,
                source_types,
                source_urls,
                source_row_count,
                merge_notes,
                merged_row_keys
            )
            VALUES (
                %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s::jsonb
            )
        """
        with self._connection.cursor() as cursor:
            cursor.execute(delete_sql)
            cursor.executemany(
                insert_sql,
                [
                    (
                        person.canonical_person_id,
                        person.canonical_name,
                        person.normalized_name_key,
                        json.dumps(list(person.aliases)),
                        person.designation,
                        person.designation_group,
                        person.department_name,
                        json.dumps(list(person.department_aliases)),
                        person.office_name,
                        person.office_city,
                        person.office_region,
                        json.dumps(list(person.office_aliases)),
                        person.email,
                        person.phone,
                        person.date_of_joining,
                        person.staff_no,
                        person.role_group,
                        person.board_role,
                        person.board_category,
                        person.source_priority,
                        person.active_status,
                        json.dumps(list(person.source_types)),
                        json.dumps(list(person.source_urls)),
                        person.source_row_count,
                        json.dumps(list(person.merge_notes)),
                        json.dumps(list(person.merged_row_keys)),
                    )
                    for person in people
                ],
            )

    def _replace_canonical_offices(
        self,
        offices: tuple[CanonicalOfficeRecord, ...],
    ) -> None:
        delete_sql = "DELETE FROM sebi_canonical_offices"
        insert_sql = """
            INSERT INTO sebi_canonical_offices (
                canonical_office_id,
                office_name,
                normalized_office_key,
                office_aliases,
                office_type,
                city,
                state,
                region,
                address,
                email,
                phone,
                fax,
                source_priority,
                active_status,
                source_types,
                source_urls,
                source_row_count
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
        """
        with self._connection.cursor() as cursor:
            cursor.execute(delete_sql)
            cursor.executemany(
                insert_sql,
                [
                    (
                        office.canonical_office_id,
                        office.office_name,
                        office.normalized_office_key,
                        json.dumps(list(office.office_aliases)),
                        office.office_type,
                        office.city,
                        office.state,
                        office.region,
                        office.address,
                        office.email,
                        office.phone,
                        office.fax,
                        office.source_priority,
                        office.active_status,
                        json.dumps(list(office.source_types)),
                        json.dumps(list(office.source_urls)),
                        office.source_row_count,
                    )
                    for office in offices
                ],
            )

    def _replace_designation_counts(
        self,
        counts: tuple[DesignationCountRecord, ...],
    ) -> None:
        delete_sql = "DELETE FROM sebi_designation_counts"
        insert_sql = """
            INSERT INTO sebi_designation_counts (designation_group, designation_label, people_count)
            VALUES (%s, %s, %s)
        """
        with self._connection.cursor() as cursor:
            cursor.execute(delete_sql)
            cursor.executemany(
                insert_sql,
                [
                    (record.designation_group, record.designation_label, record.people_count)
                    for record in counts
                ],
            )

    def _replace_office_counts(
        self,
        counts: tuple[OfficeCountRecord, ...],
    ) -> None:
        delete_sql = "DELETE FROM sebi_office_counts"
        insert_sql = """
            INSERT INTO sebi_office_counts (
                canonical_office_id,
                office_name,
                city,
                region,
                people_count,
                office_count
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self._connection.cursor() as cursor:
            cursor.execute(delete_sql)
            cursor.executemany(
                insert_sql,
                [
                    (
                        record.canonical_office_id,
                        record.office_name,
                        record.city,
                        record.region,
                        record.people_count,
                        record.office_count,
                    )
                    for record in counts
                ],
            )

    def _replace_department_counts(
        self,
        counts: tuple[DepartmentCountRecord, ...],
    ) -> None:
        delete_sql = "DELETE FROM sebi_department_counts"
        insert_sql = """
            INSERT INTO sebi_department_counts (department_name, people_count)
            VALUES (%s, %s)
        """
        with self._connection.cursor() as cursor:
            cursor.execute(delete_sql)
            cursor.executemany(
                insert_sql,
                [
                    (record.department_name, record.people_count)
                    for record in counts
                ],
            )

    def _replace_role_counts(
        self,
        counts: tuple[RoleCountRecord, ...],
    ) -> None:
        delete_sql = "DELETE FROM sebi_role_counts"
        insert_sql = """
            INSERT INTO sebi_role_counts (role_key, role_label, people_count)
            VALUES (%s, %s, %s)
        """
        with self._connection.cursor() as cursor:
            cursor.execute(delete_sql)
            cursor.executemany(
                insert_sql,
                [
                    (record.role_key, record.role_label, record.people_count)
                    for record in counts
                ],
            )


def _json_text_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item not in (None, ""))
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return (value,)
        if isinstance(loaded, list):
            return tuple(str(item) for item in loaded if item not in (None, ""))
            return (str(loaded),)
    return (str(value),)


def build_structured_info_snapshot(
    raw_dataset: DirectoryReferenceDataset,
) -> StructuredInfoSnapshot:
    """Build a canonical structured-info snapshot without requiring a repository instance."""

    offices = canonicalize_offices(raw_dataset.offices)
    people = canonicalize_people(raw_dataset, offices=offices)
    return build_snapshot(
        raw_dataset=raw_dataset,
        people=people,
        offices=offices,
        org_structure=raw_dataset.org_structure,
    )
