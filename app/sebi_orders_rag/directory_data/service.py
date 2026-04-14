"""Ingestion and query services for structured SEBI reference data."""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable

from ..config import SebiOrdersRagSettings
from ..current_info.provider import CurrentInfoResult, CurrentInfoSource
from ..current_info.query_normalization import StructuredCurrentInfoQuery, normalize_current_info_query
from ..normalization import FuzzyBand, FuzzyCandidate, FuzzyMatchResult, rank_fuzzy_candidates
from ..repositories.directory import DirectoryRepository
from ..repositories.structured_info import build_structured_info_snapshot
from ..schemas import ChatSessionStateRecord
from ..structured_info.query_service import StructuredInfoQueryService
from ..web_fallback.ranking import extract_domain
from .canonicalize import (
    CanonicalOfficeRecord,
    CanonicalPersonRecord,
    CanonicalReferenceDataset,
    canonicalize_reference_dataset,
    is_generic_city_office_query,
    match_canonical_offices,
    normalize_department,
    normalize_designation_key,
    normalize_lookup_key,
)
from .fetcher import OfficialDirectoryHtmlFetcher
from .models import (
    BoardMemberRecord,
    DirectoryIngestionSummary,
    DirectoryOfficeRecord,
    DirectoryPageParseResult,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
    DirectorySourceRunSummary,
    DirectorySourceDefinition,
    FetchedDirectorySource,
    OrgStructureRecord,
)
from .sources import (
    SOURCE_BOARD_MEMBERS,
    SOURCE_CONTACT_US,
    SOURCE_DIRECTORY,
    SOURCE_ORGCHART,
    SOURCE_REGIONAL_OFFICES,
    configured_directory_sources,
    source_title_for_type,
)


class DirectoryIngestionService:
    """Fetch, parse, snapshot, and upsert structured SEBI reference data."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        connection,
        repository: DirectoryRepository | None = None,
        fetcher: OfficialDirectoryHtmlFetcher | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository or DirectoryRepository(connection)
        self._fetcher = fetcher or OfficialDirectoryHtmlFetcher(
            timeout_seconds=settings.directory_timeout_seconds,
            user_agent=settings.directory_user_agent,
        )

    def run(
        self,
        *,
        apply: bool,
        source: str = "all",
    ) -> DirectoryIngestionSummary:
        summary = DirectoryIngestionSummary()
        for source_def in _selected_sources(self._settings, source=source):
            item = DirectorySourceRunSummary(
                source_type=source_def.source_type,
                source_url=source_def.url,
            )
            try:
                fetched = self._fetcher.fetch(source_def)
                item.fetch_status = "success"
            except Exception as exc:
                item.fetch_status = "failed"
                item.parse_status = "skipped"
                item.error = str(exc)
                if apply:
                    snapshot = self._repository.insert_snapshot(
                        source_type=source_def.source_type,
                        source_url=source_def.url,
                        content_sha256=hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                        fetch_status=item.fetch_status,
                        parse_status=item.parse_status,
                        raw_html=None,
                        error=item.error,
                    )
                    item.snapshot_id = snapshot.snapshot_id
                summary.source_summaries.append(item)
                continue

            latest_snapshot = (
                self._repository.get_latest_snapshot_for_source(source_type=source_def.source_type)
                if apply
                else None
            )
            if (
                latest_snapshot is not None
                and latest_snapshot.content_sha256 == fetched.content_sha256
                and latest_snapshot.parse_status in {"success", "skipped_unchanged"}
            ):
                item.parse_status = "skipped_unchanged"
                if apply:
                    snapshot = self._repository.insert_snapshot(
                        source_type=source_def.source_type,
                        source_url=source_def.url,
                        content_sha256=fetched.content_sha256,
                        fetch_status=item.fetch_status,
                        parse_status=item.parse_status,
                        raw_html=fetched.raw_html,
                        error=None,
                    )
                    item.snapshot_id = snapshot.snapshot_id
                summary.source_summaries.append(item)
                continue

            try:
                parsed = _parse_source(source_def.source_type, fetched)
                item.parse_status = "success"
                item.people_rows = len(parsed.people)
                item.board_rows = len(parsed.board_members)
                item.office_rows = len(parsed.offices)
                item.org_rows = len(parsed.org_structure)
            except Exception as exc:
                parsed = DirectoryPageParseResult()
                item.parse_status = "failed"
                item.error = str(exc)

            if apply:
                snapshot = self._repository.insert_snapshot(
                    source_type=source_def.source_type,
                    source_url=source_def.url,
                    content_sha256=fetched.content_sha256,
                    fetch_status=item.fetch_status,
                    parse_status=item.parse_status,
                    raw_html=fetched.raw_html,
                    error=item.error,
                )
                item.snapshot_id = snapshot.snapshot_id
                if item.parse_status == "success":
                    self._repository.replace_source_records(
                        source_type=source_def.source_type,
                        snapshot_id=snapshot.snapshot_id,
                        people=_attach_snapshot_id(parsed.people, snapshot.snapshot_id),
                        board_members=_attach_snapshot_id(parsed.board_members, snapshot.snapshot_id),
                        offices=_attach_snapshot_id(parsed.offices, snapshot.snapshot_id),
                        org_structure=_attach_snapshot_id(parsed.org_structure, snapshot.snapshot_id),
                    )
            summary.source_summaries.append(item)
        return summary


class DirectoryReferenceQueryService:
    """Answer structured people/org/office questions directly from ingested rows."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], DirectoryReferenceDataset],
        provider_name: str,
    ) -> None:
        self._dataset_loader = dataset_loader
        self._provider_name = provider_name
        self._delegate = StructuredInfoQueryService(
            snapshot_loader=lambda: build_structured_info_snapshot(self._dataset_loader()),
            provider_name=provider_name,
        )

    def supports_query(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> bool:
        return self._delegate.supports_query(query=query, session_state=session_state)

    def lookup(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult:
        return self._delegate.lookup(query=query, session_state=session_state)

    def _answer_board_members(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        if not dataset.board_members:
            return _insufficient_result(
                self._provider_name,
                "board_members",
                intent=intent,
                fallback_reason="no_board_members",
            )

        chairperson = next((record for record in dataset.board_members if record.board_category == "chairperson"), None)
        wtms = [record.canonical_name for record in dataset.board_members if record.board_category == "whole_time_member"]
        part_time = [
            record.canonical_name
            for record in dataset.board_members
            if record.board_category in {"government_nominee", "rbi_nominee", "part_time_member"}
        ]
        parts = [f"SEBI currently has {len(dataset.board_members)} board members."]
        if chairperson is not None:
            parts.append(f"The Chairperson is {chairperson.canonical_name}.")
        if wtms:
            parts.append(f"The Whole-Time Members are {_render_name_list(wtms)}.")
        if part_time:
            parts.append(f"The Part-Time Members are {_render_name_list(part_time)}.")
        supporting_records = [
            source_record
            for person in dataset.board_members
            for source_record in person.source_records
            if getattr(source_record, "source_type", None) == SOURCE_BOARD_MEMBERS
        ]
        return _answered_result(
            answer_text=" ".join(parts),
            provider_name=self._provider_name,
            lookup_type="board_members",
            records=supporting_records,
            confidence=0.97,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=dataset.board_members,
                matched_board=dataset.board_members,
                answer_path="board_members:list",
            ),
        )

    def _answer_chairperson(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        person = next(
            (
                record
                for record in dataset.people
                if record.board_category == "chairperson" or record.role_group == "chairperson"
            ),
            None,
        )
        if person is None:
            return _insufficient_result(
                self._provider_name,
                "chairperson",
                intent=intent,
                fallback_reason="chairperson_missing",
            )
        answer = f"SEBI's Chairperson is {person.canonical_name}."
        return _answered_result(
            answer_text=_append_person_details(answer, person, intent=intent),
            provider_name=self._provider_name,
            lookup_type="chairperson",
            records=(person,),
            confidence=0.97,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=(person,),
                matched_board=(person,) if person.is_board_member else (),
                answer_path="chairperson:single_match",
            ),
        )

    def _answer_wtms(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        members = tuple(record for record in dataset.people if record.role_group == "wtm")
        if not members:
            return _insufficient_result(
                self._provider_name,
                "wtm_list",
                intent=intent,
                fallback_reason="wtm_missing",
            )
        answer = f"SEBI currently has {len(members)} Whole-Time Members: {_render_name_list([member.canonical_name for member in members])}."
        return _answered_result(
            answer_text=answer,
            provider_name=self._provider_name,
            lookup_type="wtm_list",
            records=members,
            confidence=0.96,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=members,
                matched_board=members,
                answer_path="wtm:list",
            ),
        )

    def _answer_executive_directors(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        directors = tuple(record for record in dataset.people if record.role_group == "executive_director")
        if not directors:
            return _insufficient_result(
                self._provider_name,
                "ed_list",
                intent=intent,
                fallback_reason="executive_directors_missing",
            )
        rendered = [_render_person_with_department(person) for person in directors]
        answer = f"SEBI currently has {len(directors)} Executive Directors in the ingested official directory: {'; '.join(rendered)}."
        return _answered_result(
            answer_text=answer,
            provider_name=self._provider_name,
            lookup_type="ed_list",
            records=directors,
            confidence=0.91,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=directors,
                answer_path="executive_directors:list",
            ),
        )

    def _answer_leadership(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        wtm_result = self._answer_wtms(
            dataset=dataset,
            raw_dataset=raw_dataset,
            intent=intent,
            query=query,
        )
        ed_result = self._answer_executive_directors(
            dataset=dataset,
            raw_dataset=raw_dataset,
            intent=intent,
            query=query,
        )
        answer = " ".join(
            result.answer_text
            for result in (wtm_result, ed_result)
            if result.answer_status == "answered" and result.answer_text
        )
        if not answer:
            return _insufficient_result(
                self._provider_name,
                "leadership_list",
                intent=intent,
                fallback_reason="leadership_missing",
            )
        return CurrentInfoResult(
            answer_status="answered",
            answer_text=answer,
            sources=tuple({*wtm_result.sources, *ed_result.sources}),
            confidence=min(wtm_result.confidence, ed_result.confidence),
            provider_name=self._provider_name,
            lookup_type="leadership_list",
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=tuple(record for record in dataset.people if record.role_group in {"wtm", "executive_director"}),
                matched_board=tuple(record for record in dataset.board_members if record.role_group == "wtm"),
                answer_path="leadership:list",
            ),
        )

    def _answer_org_structure(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        if not dataset.org_structure and not dataset.people:
            return _insufficient_result(
                self._provider_name,
                "org_structure",
                intent=intent,
                fallback_reason="org_structure_missing",
            )

        chair = next((record for record in dataset.people if record.role_group == "chairperson"), None)
        wtms = [record.canonical_name for record in dataset.people if record.role_group == "wtm"]
        grouped = _group_org_structure(dataset.org_structure)
        parts: list[str] = []
        if chair is not None:
            parts.append(f"{chair.canonical_name} is listed as Chairperson.")
        if wtms:
            parts.append(f"The Whole-Time Members are {_render_name_list(wtms)}.")
        if grouped:
            mappings = []
            for leader_name, departments in grouped.items():
                rendered_departments = ", ".join(departments[:4])
                if len(departments) > 4:
                    rendered_departments += ", and others"
                mappings.append(f"{leader_name} oversees {rendered_departments}")
            parts.append("Organisation-structure mappings include " + "; ".join(mappings) + ".")
        return _answered_result(
            answer_text=" ".join(parts),
            provider_name=self._provider_name,
            lookup_type="org_structure",
            records=list(dataset.people) + list(dataset.org_structure),
            confidence=0.92,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=dataset.people,
                answer_path="org_structure:summary",
            ),
        )

    def _answer_office_details(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        matches = match_canonical_offices(dataset.offices, intent.office_hint or query)
        if not matches:
            return _insufficient_result(
                self._provider_name,
                "office_contact",
                intent=intent,
                fallback_reason="office_match_missing",
            )

        primary = matches[0]
        if intent.is_follow_up and intent.extracted_city:
            same_city = tuple(record for record in matches if normalize_lookup_key(record.city) == normalize_lookup_key(primary.city))
            if len(same_city) > 1:
                answer = (
                    f"SEBI has multiple official offices in {primary.city}: "
                    + " ".join(_render_office_summary(record) for record in same_city[:5])
                )
                return _answered_result(
                    answer_text=answer,
                    provider_name=self._provider_name,
                    lookup_type="office_contact",
                    records=same_city,
                    confidence=0.95,
                    debug=_build_lookup_debug(
                        intent=intent,
                        matched_offices=same_city,
                        answer_path="office:follow_up_multi_city_list",
                    ),
                )
        if is_generic_city_office_query(query, primary.city):
            same_city = tuple(record for record in matches if record.city == primary.city)
            if len(same_city) > 1:
                answer = (
                    f"SEBI has multiple official offices in {primary.city}: "
                    + " ".join(_render_office_summary(record) for record in same_city[:5])
                )
                return _answered_result(
                    answer_text=answer,
                    provider_name=self._provider_name,
                    lookup_type="office_contact",
                    records=same_city,
                    confidence=0.95,
                    debug=_build_lookup_debug(
                        intent=intent,
                        matched_offices=same_city,
                        answer_path="office:multi_city_list",
                    ),
                )

        answer = _render_single_office_answer(primary, intent=intent)
        return _answered_result(
            answer_text=answer,
            provider_name=self._provider_name,
            lookup_type="office_contact",
            records=(primary,),
            confidence=0.95,
            debug=_build_lookup_debug(
                intent=intent,
                matched_offices=matches,
                answer_path="office:single_best_match",
            ),
        )

    def _answer_regional_director(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        matches = match_canonical_offices(dataset.offices, intent.office_hint or query)
        if not matches:
            return _insufficient_result(
                self._provider_name,
                "regional_director",
                intent=intent,
                fallback_reason="regional_office_missing",
            )

        office = matches[0]
        office_key = normalize_lookup_key(office.canonical_name)
        directors = tuple(
            person
            for person in dataset.people
            if person.role_group == "regional_director"
            and person.office_name
            and (
                office_key in normalize_lookup_key(person.office_name)
                or normalize_lookup_key(person.office_name) in office_key
                or (office.city and normalize_lookup_key(office.city) in normalize_lookup_key(person.office_name))
            )
        )
        if not directors:
            return _insufficient_result(
                self._provider_name,
                "regional_director",
                intent=intent,
                matched_offices=(office,),
                fallback_reason="regional_director_missing",
            )
        person = directors[0]
        answer = f"The listed Regional Director for {office.canonical_name} is {person.canonical_name}"
        if person.designation:
            answer += f" ({person.designation})"
        details = []
        if person.phone:
            details.append(f"phone: {person.phone}")
        if person.email:
            details.append(f"email: {person.email}")
        if details:
            answer += "; " + "; ".join(details)
        answer += "."
        return _answered_result(
            answer_text=answer,
            provider_name=self._provider_name,
            lookup_type="regional_director",
            records=(person, office),
            confidence=0.95,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=(person,),
                matched_offices=(office,),
                answer_path="regional_director:single_match",
            ),
        )

    def _answer_person_lookup(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        filtered_people = _filter_people(dataset.people, intent=intent)
        if intent.unsupported_reason:
            return CurrentInfoResult(
                answer_status="insufficient_context",
                answer_text=intent.unsupported_reason,
                provider_name=self._provider_name,
                lookup_type="person_lookup",
                debug=_build_lookup_debug(
                    intent=intent,
                    matched_people=filtered_people,
                    fallback_reason="unsupported_people_filter",
                    answer_path="person:unsupported_filter",
                ),
            )

        if not intent.person_name:
            if not filtered_people:
                return CurrentInfoResult(
                    answer_status="insufficient_context",
                    answer_text="No matching current directory entry was found in the ingested official SEBI data.",
                    provider_name=self._provider_name,
                    lookup_type="person_lookup",
                    debug=_build_lookup_debug(
                        intent=intent,
                        fallback_reason="person_filter_missing",
                    ),
                )
            if len(filtered_people) == 1:
                person = filtered_people[0]
                answer = f"{_display_person_name(person.canonical_name)} is listed as {person.designation or person.board_role or 'SEBI official'}"
                if person.department_name:
                    answer += f" in {person.department_name}"
                if person.office_name:
                    answer += f" ({person.office_name})"
                answer += "."
                return _answered_result(
                    answer_text=_append_person_details(answer, person, intent=intent),
                    provider_name=self._provider_name,
                    lookup_type="person_lookup",
                    records=(person,),
                    confidence=0.91,
                    debug=_build_lookup_debug(
                        intent=intent,
                        matched_people=(person,),
                        answer_path="person:single_filtered_match",
                    ),
                )
            listed_people = "; ".join(_render_person_with_department(person) for person in filtered_people[:8])
            if len(filtered_people) > 8:
                listed_people += "; and others"
            scope_label = intent.department_hint or intent.designation_hint or "the requested current SEBI directory filter"
            answer = (
                f"The ingested official SEBI directory currently lists {len(filtered_people)} matching entries for {scope_label}: "
                f"{listed_people}."
            )
            return _answered_result(
                answer_text=answer,
                provider_name=self._provider_name,
                lookup_type="person_lookup",
                records=filtered_people,
                confidence=0.84,
                debug=_build_lookup_debug(
                    intent=intent,
                    matched_people=filtered_people,
                    answer_path="person:filtered_list",
                ),
            )

        search_people = (
            filtered_people
            if (intent.department_hint or intent.designation_hint)
            else dataset.people
        )
        match_result = _match_people(search_people, intent.person_name)
        candidate_rows = match_result.candidates
        matches = tuple(candidate.value for candidate in candidate_rows)
        if not matches:
            return CurrentInfoResult(
                answer_status="insufficient_context",
                answer_text="No matching current directory entry was found in the ingested official SEBI data.",
                provider_name=self._provider_name,
                lookup_type="person_lookup",
                debug=_build_lookup_debug(
                    intent=intent,
                    fallback_reason="person_match_missing",
                    fuzzy_candidates=candidate_rows,
                ),
            )

        top_candidate = candidate_rows[0]
        if (
            match_result.band == "medium"
            and len(matches) == 1
            and _can_promote_filtered_single_token_match(intent=intent, candidate=top_candidate)
        ):
            match_result = FuzzyMatchResult(
                query=match_result.query,
                normalized_query=match_result.normalized_query,
                candidates=match_result.candidates,
                confident_match=top_candidate.value,
                ambiguous=False,
                band="high",
                clarification_candidate=None,
            )

        if match_result.band == "medium":
            candidate = match_result.clarification_candidate or top_candidate
            return CurrentInfoResult(
                answer_status="insufficient_context",
                answer_text=f"Did you mean {_render_person_clarification(candidate.value)}?",
                provider_name=self._provider_name,
                lookup_type="person_lookup",
                debug=_build_lookup_debug(
                    intent=intent,
                    matched_people=(candidate.value,),
                    fallback_reason="person_match_clarify",
                    answer_path="person:clarify",
                    fuzzy_candidates=candidate_rows,
                    fuzzy_band=match_result.band,
                ),
            )

        if match_result.band == "low":
            return CurrentInfoResult(
                answer_status="insufficient_context",
                answer_text="No matching current directory entry was found in the ingested official SEBI data.",
                provider_name=self._provider_name,
                lookup_type="person_lookup",
                debug=_build_lookup_debug(
                    intent=intent,
                    fallback_reason="person_match_low_confidence",
                    answer_path="person:abstain_low_confidence",
                    fuzzy_candidates=candidate_rows,
                    fuzzy_band=match_result.band,
                ),
            )

        confident_candidate = match_result.confident_match
        if confident_candidate is None and len(matches) > 1:
            return CurrentInfoResult(
                answer_status="insufficient_context",
                answer_text=(
                    f"Multiple current directory entries match {intent.person_name}: "
                    f"{_render_name_list([person.canonical_name for person in matches[:5]])}. "
                    "Please specify the full name or office."
                ),
                provider_name=self._provider_name,
                lookup_type="person_lookup",
                debug=_build_lookup_debug(
                    intent=intent,
                    matched_people=matches,
                    fallback_reason="person_match_ambiguous",
                    answer_path="person:ambiguous",
                    fuzzy_candidates=candidate_rows,
                    fuzzy_band=match_result.band,
                ),
            )

        person = confident_candidate or matches[0]
        title = person.designation or person.board_role or "SEBI official"
        display_name = _display_person_name(person.canonical_name)
        answer = f"{display_name} is listed as {title}"
        if person.department_name:
            answer += f" in {person.department_name}"
        if person.office_name:
            answer += f" ({person.office_name})"
        answer += "."
        return _answered_result(
            answer_text=_append_person_details(answer, person, intent=intent),
            provider_name=self._provider_name,
            lookup_type="person_lookup",
            records=(person,),
            confidence=0.92,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=(person,),
                matched_board=(person,) if person.is_board_member else (),
                answer_path="person:single_match" if confident_candidate else "person:cautious_single_candidate",
                fuzzy_candidates=candidate_rows,
                fuzzy_band=match_result.band,
            ),
        )

    def _answer_designation_count(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        if not intent.designation_hint:
            return _insufficient_result(
                self._provider_name,
                "designation_count",
                intent=intent,
                fallback_reason="designation_missing",
            )
        target_key = normalize_designation_key(intent.designation_hint)
        matches = tuple(
            person
            for person in dataset.people
            if _designation_matches(person.designation, target_key)
        )
        answer = (
            f'The ingested public SEBI directory currently lists {len(matches)} '
            f'entries matching the designation "{intent.designation_hint}".'
        )
        return _answered_result(
            answer_text=answer,
            provider_name=self._provider_name,
            lookup_type="designation_count",
            records=matches,
            confidence=0.83,
            debug=_build_lookup_debug(
                intent=intent,
                matched_people=matches,
                answer_path="designation_count:exact_match",
            ),
        )

    def _answer_total_strength(
        self,
        *,
        dataset: CanonicalReferenceDataset,
        raw_dataset: DirectoryReferenceDataset,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        return _answered_result(
            answer_text="The public directory data does not reliably establish total institutional strength.",
            provider_name=self._provider_name,
            lookup_type="total_strength",
            records=(),
            confidence=0.48,
            debug=_build_lookup_debug(
                intent=intent,
                answer_path="total_strength:cautious_decline",
            ),
        )


def _append_person_details(answer: str, person: CanonicalPersonRecord, *, intent: StructuredCurrentInfoQuery) -> str:
    details = []
    if intent.wants_joining_date and person.date_of_joining:
        details.append(f"date of joining: {person.date_of_joining}")
    if intent.wants_phone and person.phone:
        details.append(f"phone: {person.phone}")
    if intent.wants_email and person.email:
        details.append(f"email: {person.email}")
    if not details:
        return answer
    return answer.rstrip(".") + "; " + "; ".join(details) + "."


def _render_person_with_department(person: CanonicalPersonRecord) -> str:
    if person.department_name:
        return f"{_display_person_name(person.canonical_name)} ({person.department_name})"
    return _display_person_name(person.canonical_name)


def _render_person_clarification(person: CanonicalPersonRecord) -> str:
    parts = [_display_person_name(person.canonical_name)]
    if person.designation:
        parts.append(person.designation)
    rendered = ", ".join(parts)
    if person.department_name and person.office_name:
        return f"{rendered} in {person.department_name} ({person.office_name})"
    if person.department_name:
        return f"{rendered} in {person.department_name}"
    if person.office_name:
        return f"{rendered} ({person.office_name})"
    return rendered


def _render_name_list(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _display_person_name(value: str) -> str:
    return value.strip(" ,.")


def _render_single_office_answer(office: CanonicalOfficeRecord, *, intent: StructuredCurrentInfoQuery) -> str:
    details = []
    if office.address:
        details.append(f"address: {office.address}")
    if office.phone and (intent.wants_phone or not (intent.wants_address and not intent.wants_email and not intent.wants_fax)):
        details.append(f"phone: {office.phone}")
    if office.email and (intent.wants_email or not intent.wants_address):
        details.append(f"email: {office.email}")
    if office.fax and intent.wants_fax:
        details.append(f"fax: {office.fax}")
    rendered_details = "; ".join(details) if details else "official contact details are listed on the SEBI site"
    return f"{office.canonical_name}: {rendered_details}."


def _render_office_summary(office: CanonicalOfficeRecord) -> str:
    details = [office.canonical_name]
    if office.address:
        details.append(f"address: {office.address}")
    if office.phone:
        details.append(f"phone: {office.phone}")
    return "; ".join(details) + "."


def _answered_result(
    *,
    answer_text: str,
    provider_name: str,
    lookup_type: str,
    records: Iterable[object],
    confidence: float,
    debug: dict[str, object] | None = None,
) -> CurrentInfoResult:
    sources = _sources_for_records(records)
    return CurrentInfoResult(
        answer_status="answered",
        answer_text=answer_text,
        sources=sources,
        confidence=confidence,
        provider_name=provider_name,
        lookup_type=lookup_type,
        debug={
            "source_count": len(sources),
            **(debug or {}),
        },
    )


def _insufficient_result(
    provider_name: str,
    lookup_type: str,
    *,
    intent: StructuredCurrentInfoQuery,
    matched_people: Iterable[CanonicalPersonRecord] = (),
    matched_offices: Iterable[CanonicalOfficeRecord] = (),
    matched_board: Iterable[CanonicalPersonRecord] = (),
    fallback_reason: str,
) -> CurrentInfoResult:
    return CurrentInfoResult(
        answer_status="insufficient_context",
        answer_text=(
            "I could not find a confident structured SEBI people, board, organisation, "
            "or office-contact match for that query."
        ),
        provider_name=provider_name,
        lookup_type=lookup_type,
        debug=_build_lookup_debug(
            intent=intent,
            matched_people=tuple(matched_people),
            matched_offices=tuple(matched_offices),
            matched_board=tuple(matched_board),
            fallback_reason=fallback_reason,
        ),
    )


def _classify_query(
    query: str,
    session_state: ChatSessionStateRecord | None = None,
) -> StructuredCurrentInfoQuery:
    return normalize_current_info_query(query, session_state=session_state)


def _match_people(
    people: tuple[CanonicalPersonRecord, ...],
    person_name: str,
) -> FuzzyMatchResult[CanonicalPersonRecord]:
    return rank_fuzzy_candidates(
        person_name,
        people,
        key=lambda person: person.canonical_name,
        min_score=0.7,
        medium_score=0.8,
        confident_score=0.84,
        ambiguity_gap=0.05,
    )


def _designation_matches(designation: str | None, target_key: str) -> bool:
    designation_key = normalize_designation_key(designation)
    if not designation_key or not target_key:
        return False
    return designation_key == target_key or designation_key.startswith(f"{target_key} ")


def _filter_people(
    people: tuple[CanonicalPersonRecord, ...],
    *,
    intent: StructuredCurrentInfoQuery,
) -> tuple[CanonicalPersonRecord, ...]:
    filtered = list(people)
    if intent.department_hint:
        department_key = normalize_department(intent.department_hint)
        filtered = [
            person
            for person in filtered
            if department_key
            and (
                person.department_key == department_key
                or normalize_department(person.department_name) == department_key
                or normalize_department(person.office_name) == department_key
                or normalize_lookup_key(department_key) in normalize_lookup_key(person.department_name)
                or normalize_lookup_key(department_key) in normalize_lookup_key(person.office_name)
            )
        ]
    if intent.designation_hint:
        designation_key = normalize_designation_key(intent.designation_hint)
        filtered = [
            person
            for person in filtered
            if designation_key and _designation_matches(person.designation, designation_key)
        ]
    return tuple(filtered)


def _can_promote_filtered_single_token_match(
    *,
    intent: StructuredCurrentInfoQuery,
    candidate: FuzzyCandidate[CanonicalPersonRecord],
) -> bool:
    if not intent.person_name or " " in intent.person_name.strip():
        return False
    if not (intent.department_hint or intent.designation_hint):
        return False
    query_token = normalize_lookup_key(intent.person_name)
    candidate_tokens = set(candidate.normalized_name.split())
    return query_token in candidate_tokens


def _build_lookup_debug(
    *,
    intent: StructuredCurrentInfoQuery,
    matched_people: Iterable[CanonicalPersonRecord] = (),
    matched_offices: Iterable[CanonicalOfficeRecord] = (),
    matched_board: Iterable[CanonicalPersonRecord] = (),
    fallback_reason: str | None = None,
    answer_path: str | None = None,
    fuzzy_candidates: Iterable[FuzzyCandidate[CanonicalPersonRecord]] = (),
    fuzzy_band: FuzzyBand | None = None,
) -> dict[str, object]:
    people_matches = tuple(matched_people)
    office_matches = tuple(matched_offices)
    board_matches = tuple(matched_board)
    return {
        "normalized_query": intent.normalized_query,
        "detected_query_family": intent.lookup_type,
        "extracted_city": intent.extracted_city,
        "extracted_person_name": intent.person_name,
        "department_hint": intent.department_hint,
        "designation_hint": intent.designation_hint,
        "extracted_role_tokens": list(intent.role_tokens),
        "office_tokens": list(intent.office_tokens),
        "is_follow_up": intent.is_follow_up,
        "matched_people_rows_count": len(people_matches),
        "matched_office_rows_count": len(office_matches),
        "matched_board_rows_count": len(board_matches),
        "matched_people": _debug_people_refs(people_matches),
        "matched_offices": _debug_office_refs(office_matches),
        "matched_board": _debug_people_refs(board_matches),
        "normalized_expansions": list(intent.normalized_expansions),
        "matched_abbreviations": list(intent.matched_abbreviations),
        "fuzzy_candidates": _debug_fuzzy_candidates(fuzzy_candidates),
        "fuzzy_band": fuzzy_band,
        "unsupported_reason": intent.unsupported_reason,
        "fallback_reason": fallback_reason,
        "answer_path": answer_path,
    }


def _debug_people_refs(records: Iterable[CanonicalPersonRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row_ids = [
            getattr(source_record, "person_id", None) or getattr(source_record, "board_member_id", None)
            for source_record in record.source_records
        ]
        rows.append(
            {
                "name": record.canonical_name,
                "row_ids": [row_id for row_id in row_ids if row_id is not None],
            }
        )
    return rows


def _debug_office_refs(records: Iterable[CanonicalOfficeRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "name": record.canonical_name,
                "row_ids": [
                    getattr(source_record, "office_id", None)
                    for source_record in record.source_records
                    if getattr(source_record, "office_id", None) is not None
                ],
            }
        )
    return rows


def _debug_fuzzy_candidates(
    candidates: Iterable[FuzzyCandidate[CanonicalPersonRecord]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        rows.append(
            {
                "name": candidate.display_name,
                "score": candidate.score,
                "match_type": candidate.match_type,
                "token_overlap": candidate.token_overlap,
                "first_token_similarity": candidate.first_token_similarity,
                "last_token_similarity": candidate.last_token_similarity,
            }
        )
    return rows


def _selected_sources(settings: SebiOrdersRagSettings, *, source: str) -> tuple[DirectorySourceDefinition, ...]:
    sources = configured_directory_sources(settings)
    if source == "all":
        return sources
    return tuple(item for item in sources if item.source_type == source)


def _parse_source(source_type: str, fetched: FetchedDirectorySource) -> DirectoryPageParseResult:
    parser = {
        SOURCE_DIRECTORY: lambda html_text: __import__(
            "app.sebi_orders_rag.directory_data.parser_directory",
            fromlist=["parse_directory_page"],
        ).parse_directory_page(
            html_text,
            source_url=fetched.source_url,
            source_type=source_type,
        ),
        SOURCE_ORGCHART: lambda html_text: __import__(
            "app.sebi_orders_rag.directory_data.parser_orgchart",
            fromlist=["parse_orgchart_page"],
        ).parse_orgchart_page(
            html_text,
            source_url=fetched.source_url,
            source_type=source_type,
        ),
        SOURCE_REGIONAL_OFFICES: lambda html_text: __import__(
            "app.sebi_orders_rag.directory_data.parser_offices",
            fromlist=["parse_regional_offices_page"],
        ).parse_regional_offices_page(
            html_text,
            source_url=fetched.source_url,
            source_type=source_type,
        ),
        SOURCE_CONTACT_US: lambda html_text: __import__(
            "app.sebi_orders_rag.directory_data.parser_offices",
            fromlist=["parse_contact_us_page"],
        ).parse_contact_us_page(
            html_text,
            source_url=fetched.source_url,
            source_type=source_type,
        ),
        SOURCE_BOARD_MEMBERS: lambda html_text: __import__(
            "app.sebi_orders_rag.directory_data.parser_board",
            fromlist=["parse_board_page"],
        ).parse_board_page(
            html_text,
            source_url=fetched.source_url,
            source_type=source_type,
        ),
    }[source_type]
    return parser(fetched.raw_html)


def _attach_snapshot_id(records: Iterable[object], snapshot_id: int) -> tuple[object, ...]:
    attached = []
    for record in records:
        attached.append(record.__class__(**{**record.__dict__, "snapshot_id": snapshot_id}))
    return tuple(attached)


def _group_org_structure(records: Iterable[OrgStructureRecord]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        if not record.leader_name or not record.department_name:
            continue
        grouped.setdefault(record.leader_name, [])
        if record.department_name not in grouped[record.leader_name]:
            grouped[record.leader_name].append(record.department_name)
    return grouped


def _sources_for_records(records: Iterable[object]) -> tuple[CurrentInfoSource, ...]:
    seen: dict[tuple[str, str], CurrentInfoSource] = {}
    for record in _flatten_records(records):
        source_type = getattr(record, "source_type", None)
        source_url = getattr(record, "source_url", None)
        if not source_type or not source_url:
            continue
        key = (source_type, source_url)
        if key in seen:
            continue
        seen[key] = CurrentInfoSource(
            title=source_title_for_type(source_type),
            url=source_url,
            record_key=f"official:{source_type}",
            domain=extract_domain(source_url),
            source_type="structured",
        )
    return tuple(seen.values())


def _flatten_records(records: Iterable[object]) -> tuple[object, ...]:
    flattened: list[object] = []
    for record in records:
        if hasattr(record, "source_records"):
            flattened.extend(getattr(record, "source_records"))
        else:
            flattened.append(record)
    return tuple(flattened)
