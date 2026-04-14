"""Unified canonical query service for structured SEBI current information."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ..current_info.provider import CurrentInfoResult, CurrentInfoSource
from ..current_info.query_normalization import StructuredCurrentInfoQuery, normalize_current_info_query
from ..directory_data.models import DirectoryReferenceDataset
from ..directory_data.sources import source_title_for_type
from ..schemas import ChatSessionStateRecord
from ..web_fallback.ranking import extract_domain
from .canonical_models import CanonicalOfficeRecord, CanonicalPersonRecord, StructuredInfoSnapshot
from .canonicalize_offices import is_generic_city_office_query, match_canonical_offices, normalize_lookup_key
from .counts import canonical_count_for_designation, canonical_role_count, contributing_names
from .office_lookup import (
    match_offices,
    render_office_summary,
    render_single_office_answer,
    same_city_matches,
    should_render_city_list,
)
from .people_lookup import (
    filter_people,
    lookup_staff_no,
    render_clarification,
    render_person_answer,
    render_person_with_context,
    render_staff_lookup_answer,
    resolve_person_match,
)


class StructuredInfoQueryService:
    """Answer structured current-information questions from one canonical snapshot."""

    def __init__(
        self,
        *,
        snapshot_loader: Callable[[], StructuredInfoSnapshot],
        provider_name: str,
    ) -> None:
        self._snapshot_loader = snapshot_loader
        self._provider_name = provider_name

    def supports_query(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> bool:
        return classify_structured_info_query(query, session_state=session_state).lookup_type != "unsupported"

    def lookup(
        self,
        *,
        query: str,
        session_state: ChatSessionStateRecord | None = None,
    ) -> CurrentInfoResult:
        intent = classify_structured_info_query(query, session_state=session_state)
        if intent.lookup_type == "unsupported":
            return self._insufficient(
                intent=intent,
                answer_text=(
                    "Structured SEBI reference lookup is intended for current people, board, "
                    "organisation, and office-contact questions."
                ),
                fallback_reason="unsupported_query",
            )

        snapshot = self._snapshot_loader()
        if not (snapshot.people or snapshot.offices or snapshot.org_structure):
            return self._insufficient(
                intent=intent,
                answer_text=(
                    "No canonical current SEBI people, board, or office data is currently available."
                ),
                fallback_reason="snapshot_missing",
            )

        handlers = {
            "board_members": self._answer_board_members,
            "chairperson": self._answer_chairperson,
            "wtm_list": self._answer_wtms,
            "ed_list": self._answer_executive_directors,
            "leadership_list": self._answer_leadership,
            "org_structure": self._answer_org_structure,
            "office_contact": self._answer_office_details,
            "regional_director": self._answer_regional_director,
            "person_lookup": self._answer_person_lookup,
            "staff_id_lookup": self._answer_staff_id_lookup,
            "designation_count": self._answer_designation_count,
            "total_strength": self._answer_total_strength,
        }
        return handlers[intent.lookup_type](snapshot=snapshot, intent=intent, query=query)

    def _answer_board_members(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        members = tuple(
            person
            for person in snapshot.people
            if person.designation_group in {"chairperson", "whole_time_member", "board_member"}
        )
        if not members:
            return self._insufficient(intent=intent, fallback_reason="board_members_missing")
        chairperson = next((person for person in members if person.designation_group == "chairperson"), None)
        wtms = [person.canonical_name for person in members if person.designation_group == "whole_time_member"]
        others = [
            person.canonical_name
            for person in members
            if person.designation_group == "board_member"
        ]
        parts = [f"SEBI currently has {len(members)} board members."]
        if chairperson is not None:
            parts.append(f"The Chairperson is {chairperson.canonical_name}.")
        if wtms:
            parts.append(f"The Whole-Time Members are {_render_name_list(wtms)}.")
        if others:
            parts.append(f"The other board members are {_render_name_list(others)}.")
        return self._answered(
            answer_text=" ".join(parts),
            lookup_type="board_members",
            records=members,
            confidence=0.97,
            preferred_source_types=("board_members",),
                debug=self._debug(
                    intent=intent,
                    matched_people=members,
                    count_debug={
                        "group": "board_member",
                        "count": len(members),
                        "contributing_names": list(contributing_names(members)),
                    },
                    answer_path="board_members:canonical_list",
                ),
        )

    def _answer_chairperson(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        person = next((person for person in snapshot.people if person.designation_group == "chairperson"), None)
        if person is None:
            return self._insufficient(intent=intent, fallback_reason="chairperson_missing")
        answer = f"SEBI's Chairperson is {person.canonical_name}."
        return self._answered(
            answer_text=self._append_person_details(answer, person, intent=intent),
            lookup_type="chairperson",
            records=(person,),
            confidence=0.97,
            debug=self._debug(
                intent=intent,
                matched_people=(person,),
                answer_path="chairperson:single_match",
            ),
        )

    def _answer_wtms(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        members = tuple(
            sorted(
                (person for person in snapshot.people if person.designation_group == "whole_time_member"),
                key=lambda person: person.canonical_name,
            )
        )
        if not members:
            return self._insufficient(intent=intent, fallback_reason="whole_time_members_missing")
        count = canonical_role_count(
            snapshot,
            role_key="whole_time_member",
            fallback_people=members,
        )
        answer = (
            f"SEBI currently has {count} Whole-Time Members: "
            f"{_render_name_list([person.canonical_name for person in members])}."
        )
        return self._answered(
            answer_text=answer,
            lookup_type="wtm_list",
            records=members,
            confidence=0.96,
                debug=self._debug(
                    intent=intent,
                    matched_people=members,
                    count_debug={
                        "group": "whole_time_member",
                        "count": count,
                        "contributing_names": list(contributing_names(members)),
                    },
                    answer_path="wtm:canonical_count",
                ),
        )

    def _answer_executive_directors(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        people = tuple(
            sorted(
                (person for person in snapshot.people if person.designation_group == "executive_director"),
                key=lambda person: (person.department_name or "", person.canonical_name),
            )
        )
        if not people:
            return self._insufficient(intent=intent, fallback_reason="executive_directors_missing")
        count = canonical_role_count(
            snapshot,
            role_key="executive_director",
            fallback_people=people,
        )
        rendered = "; ".join(render_person_with_context(person) for person in people)
        answer = (
            f"SEBI currently has {count} Executive Directors in the canonical current-info layer: "
            f"{rendered}."
        )
        return self._answered(
            answer_text=answer,
            lookup_type="ed_list",
            records=people,
            confidence=0.94,
                debug=self._debug(
                    intent=intent,
                    matched_people=people,
                    count_debug={
                        "group": "executive_director",
                        "count": count,
                        "contributing_names": list(contributing_names(people)),
                    },
                    answer_path="executive_director:canonical_count",
                ),
        )

    def _answer_leadership(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        wtm_result = self._answer_wtms(snapshot=snapshot, intent=intent, query=query)
        ed_result = self._answer_executive_directors(snapshot=snapshot, intent=intent, query=query)
        answer_text = " ".join(
            result.answer_text
            for result in (wtm_result, ed_result)
            if result.answer_status == "answered" and result.answer_text
        )
        if not answer_text:
            return self._insufficient(intent=intent, fallback_reason="leadership_missing")
        return CurrentInfoResult(
            answer_status="answered",
            answer_text=answer_text,
            sources=tuple({*wtm_result.sources, *ed_result.sources}),
            confidence=min(wtm_result.confidence, ed_result.confidence),
            provider_name=self._provider_name,
            lookup_type="leadership_list",
            debug=self._debug(
                intent=intent,
                matched_people=tuple(
                    person
                    for person in snapshot.people
                    if person.designation_group in {"whole_time_member", "executive_director"}
                ),
                answer_path="leadership:combined",
            ),
        )

    def _answer_org_structure(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        parts: list[str] = []
        chair = next((person for person in snapshot.people if person.designation_group == "chairperson"), None)
        wtms = [person.canonical_name for person in snapshot.people if person.designation_group == "whole_time_member"]
        if chair is not None:
            parts.append(f"{chair.canonical_name} is listed as Chairperson.")
        if wtms:
            parts.append(f"The Whole-Time Members are {_render_name_list(wtms)}.")
        grouped = _group_org_structure(snapshot.org_structure)
        if grouped:
            rendered = []
            for leader_name, departments in sorted(grouped.items()):
                label = ", ".join(departments[:4])
                if len(departments) > 4:
                    label += ", and others"
                rendered.append(f"{leader_name} oversees {label}")
            parts.append("Organisation-structure mappings include " + "; ".join(rendered) + ".")
        if not parts:
            return self._insufficient(intent=intent, fallback_reason="org_structure_missing")
        return self._answered(
            answer_text=" ".join(parts),
            lookup_type="org_structure",
            records=tuple(snapshot.people) + tuple(snapshot.org_structure),
            confidence=0.92,
            debug=self._debug(
                intent=intent,
                matched_people=snapshot.people,
                answer_path="org_structure:summary",
            ),
        )

    def _answer_office_details(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        matches = match_offices(snapshot.offices, query=query, intent=intent)
        if not matches:
            return self._insufficient(intent=intent, fallback_reason="office_match_missing")
        primary = matches[0]
        same_city = same_city_matches(matches, city=primary.city)
        if should_render_city_list(
            query=query,
            intent=intent,
            primary_office=primary,
            same_city_offices=same_city,
        ):
            answer = (
                f"SEBI has multiple official offices in {primary.city}: "
                + " ".join(render_office_summary(office) for office in same_city[:5])
            )
            return self._answered(
                answer_text=answer,
                lookup_type="office_contact",
                records=same_city,
                confidence=0.95,
                debug=self._debug(
                    intent=intent,
                    matched_offices=same_city,
                    answer_path="office:multi_city_list",
                ),
            )
        return self._answered(
            answer_text=render_single_office_answer(primary, intent=intent),
            lookup_type="office_contact",
            records=(primary,),
            confidence=0.95,
            debug=self._debug(
                intent=intent,
                matched_offices=matches,
                answer_path="office:single_best_match",
            ),
        )

    def _answer_staff_id_lookup(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        if not intent.staff_no:
            return self._insufficient(
                intent=intent,
                answer_text="No staff ID was detected in that query.",
                fallback_reason="staff_id_missing",
            )
        staff_match = lookup_staff_no(snapshot, staff_no=intent.staff_no)
        if staff_match is None:
            return self._insufficient(
                intent=intent,
                answer_text=f"No matching current directory entry was found for staff ID {intent.staff_no}.",
                fallback_reason="staff_id_no_match",
            )
        records: tuple[object, ...] = staff_match.canonical_people or staff_match.raw_rows
        return self._answered(
            answer_text=render_staff_lookup_answer(staff_match),
            lookup_type="staff_id_lookup",
            records=records,
            confidence=0.96,
            debug=self._debug(
                intent=intent,
                matched_people=staff_match.canonical_people,
                raw_staff_rows=staff_match.raw_rows,
                answer_path="staff_id:raw_structured_lookup",
            ),
        )

    def _answer_regional_director(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        offices = match_canonical_offices(snapshot.offices, intent.office_hint or query)
        if not offices:
            return self._insufficient(intent=intent, fallback_reason="regional_office_missing")
        office = offices[0]
        people = tuple(
            person
            for person in snapshot.people
            if person.designation_group == "regional_director"
            and (
                (person.office_name and normalize_lookup_key(office.office_name) in normalize_lookup_key(person.office_name))
                or (office.city and person.office_city and normalize_lookup_key(person.office_city) == normalize_lookup_key(office.city))
                or (office.city and person.office_name and normalize_lookup_key(office.city) in normalize_lookup_key(person.office_name))
            )
        )
        if not people:
            return self._insufficient(
                intent=intent,
                matched_offices=(office,),
                fallback_reason="regional_director_missing",
            )
        person = people[0]
        answer = f"The listed Regional Director for {office.office_name} is {person.canonical_name}"
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
        return self._answered(
            answer_text=answer,
            lookup_type="regional_director",
            records=(person, office),
            confidence=0.95,
            debug=self._debug(
                intent=intent,
                matched_people=(person,),
                matched_offices=(office,),
                answer_path="regional_director:single_match",
            ),
        )

    def _answer_person_lookup(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        if intent.unsupported_reason:
            return self._insufficient(
                intent=intent,
                answer_text=intent.unsupported_reason,
                fallback_reason="unsupported_people_filter",
            )

        filtered_people = filter_people(snapshot.people, intent=intent)
        if not intent.person_name:
            if not filtered_people:
                return self._insufficient(
                    intent=intent,
                    answer_text=_NO_MATCH_CURRENT_PERSON,
                    fallback_reason="person_filter_missing",
                )
            if len(filtered_people) == 1:
                person = filtered_people[0]
                answer = render_person_answer(person)
                return self._answered(
                    answer_text=self._append_person_details(answer, person, intent=intent),
                    lookup_type="person_lookup",
                    records=(person,),
                    confidence=0.92,
                    debug=self._debug(
                        intent=intent,
                        matched_people=(person,),
                        answer_path="person:single_filtered_match",
                    ),
                )
            scope_label = intent.department_hint or intent.designation_hint or "the requested current SEBI directory filter"
            answer = (
                f"The canonical current-info layer lists {len(filtered_people)} matching entries for {scope_label}: "
                + "; ".join(render_person_with_context(person) for person in filtered_people[:8])
            )
            if len(filtered_people) > 8:
                answer += "; and others."
            else:
                answer += "."
            return self._answered(
                answer_text=answer,
                lookup_type="person_lookup",
                records=filtered_people,
                confidence=0.85,
                debug=self._debug(
                    intent=intent,
                    matched_people=filtered_people,
                    answer_path="person:filtered_list",
                ),
            )

        match_result = resolve_person_match(snapshot, intent=intent)
        if match_result.status in {"exact", "high"} and match_result.matches:
            person = match_result.matches[0]
            return self._answered(
                answer_text=self._append_person_details(render_person_answer(person), person, intent=intent),
                lookup_type="person_lookup",
                records=(person,),
                confidence=0.93 if match_result.status == "exact" else 0.89,
                debug=self._debug(
                    intent=intent,
                    matched_people=(person,),
                    fuzzy_candidates=match_result.fuzzy_candidates,
                    fuzzy_band=match_result.fuzzy_band,
                    person_match_status=match_result.match_stage or match_result.status,
                    answer_path=f"person:{match_result.status}_match",
                ),
            )
        if match_result.status == "clarify" and match_result.clarification_candidates:
            rendered = render_clarification(match_result.clarification_candidates)
            return self._insufficient(
                intent=intent,
                answer_text=f"Did you mean {rendered}?",
                matched_people=match_result.clarification_candidates,
                fuzzy_candidates=match_result.fuzzy_candidates,
                fuzzy_band=match_result.fuzzy_band,
                person_match_status=match_result.match_stage or match_result.status,
                fallback_reason="person_match_clarify",
                answer_path="person:clarify",
            )
        return self._insufficient(
            intent=intent,
            answer_text=_NO_MATCH_CURRENT_PERSON,
            fuzzy_candidates=match_result.fuzzy_candidates,
            fuzzy_band=match_result.fuzzy_band,
            person_match_status=match_result.match_stage or match_result.status,
            fallback_reason="person_match_missing",
            answer_path="person:no_match",
        )

    def _answer_designation_count(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        count, designation_group, label, people = canonical_count_for_designation(
            snapshot,
            designation_hint=intent.designation_hint,
        )
        if not designation_group or label is None:
            return self._insufficient(
                intent=intent,
                fallback_reason="designation_group_missing",
            )
        stored_count = snapshot.designation_count_by_group().get(designation_group, 0)
        answer = (
            f'The canonical current-info layer currently lists {count} '
            f'entries matching the designation "{label}".'
        )
        return self._answered(
            answer_text=answer,
            lookup_type="designation_count",
            records=people,
            confidence=0.93,
            debug=self._debug(
                intent=intent,
                matched_people=people,
                count_debug={
                    "group": designation_group,
                    "count": count,
                    "stored_count": stored_count,
                    "count_table_parity_ok": stored_count == count,
                    "contributing_names": list(contributing_names(people)),
                },
                answer_path="designation_count:canonical_group",
            ),
        )

    def _answer_total_strength(
        self,
        *,
        snapshot: StructuredInfoSnapshot,
        intent: StructuredCurrentInfoQuery,
        query: str,
    ) -> CurrentInfoResult:
        return self._answered(
            answer_text="The public structured current-info data does not reliably establish total institutional strength.",
            lookup_type="total_strength",
            records=(),
            confidence=0.48,
            debug=self._debug(
                intent=intent,
                answer_path="total_strength:cautious_decline",
            ),
        )

    def _append_person_details(
        self,
        answer: str,
        person: CanonicalPersonRecord,
        *,
        intent: StructuredCurrentInfoQuery,
    ) -> str:
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

    def _answered(
        self,
        *,
        answer_text: str,
        lookup_type: str,
        records: Iterable[object],
        confidence: float,
        preferred_source_types: tuple[str, ...] = (),
        debug: dict[str, object],
    ) -> CurrentInfoResult:
        sources = _sources_for_records(records, preferred_source_types=preferred_source_types)
        return CurrentInfoResult(
            answer_status="answered",
            answer_text=answer_text,
            sources=sources,
            confidence=confidence,
            provider_name=self._provider_name,
            lookup_type=lookup_type,
            debug={
                "source_count": len(sources),
                **debug,
            },
        )

    def _insufficient(
        self,
        *,
        intent: StructuredCurrentInfoQuery,
        answer_text: str | None = None,
        matched_people: Iterable[CanonicalPersonRecord] = (),
        matched_offices: Iterable[CanonicalOfficeRecord] = (),
        fuzzy_candidates=(),
        fuzzy_band: str | None = None,
        person_match_status: str | None = None,
        fallback_reason: str,
        answer_path: str | None = None,
    ) -> CurrentInfoResult:
        return CurrentInfoResult(
            answer_status="insufficient_context",
            answer_text=answer_text or _GENERIC_STRUCTURED_MISS,
            provider_name=self._provider_name,
            lookup_type=intent.lookup_type,
            debug=self._debug(
                intent=intent,
                matched_people=tuple(matched_people),
                matched_offices=tuple(matched_offices),
                fuzzy_candidates=fuzzy_candidates,
                fuzzy_band=fuzzy_band,
                person_match_status=person_match_status,
                fallback_reason=fallback_reason,
                answer_path=answer_path,
            ),
        )

    def _debug(
        self,
        *,
        intent: StructuredCurrentInfoQuery,
        matched_people: Iterable[CanonicalPersonRecord] = (),
        matched_offices: Iterable[CanonicalOfficeRecord] = (),
        raw_staff_rows=(),
        fuzzy_candidates=(),
        fuzzy_band: str | None = None,
        person_match_status: str | None = None,
        count_debug: dict[str, object] | None = None,
        fallback_reason: str | None = None,
        answer_path: str | None = None,
    ) -> dict[str, object]:
        people = tuple(matched_people)
        offices = tuple(matched_offices)
        raw_staff_matches = tuple(raw_staff_rows)
        return {
            "normalized_query": intent.normalized_query,
            "detected_query_family": intent.lookup_type,
            "extracted_city": intent.extracted_city,
            "extracted_person_name": intent.person_name,
            "extracted_staff_no": intent.staff_no,
            "department_hint": intent.department_hint,
            "designation_hint": intent.designation_hint,
            "extracted_role_tokens": list(intent.role_tokens),
            "office_tokens": list(intent.office_tokens),
            "is_follow_up": intent.is_follow_up,
            "matched_people_rows_count": len(people),
            "matched_office_rows_count": len(offices),
            "matched_people": [
                {
                    "name": person.canonical_name,
                    "canonical_person_id": person.canonical_person_id,
                    "designation_group": person.designation_group,
                    "designation": person.designation,
                    "office_name": person.office_name,
                    "department_name": person.department_name,
                    "staff_no": person.staff_no,
                    "merge_notes": list(person.merge_notes),
                    "merged_row_keys": list(person.merged_row_keys),
                }
                for person in people
            ],
            "matched_offices": [
                {"name": office.office_name, "canonical_office_id": office.canonical_office_id}
                for office in offices
            ],
            "raw_staff_rows": [
                {
                    "name": row.canonical_name,
                    "designation": row.designation,
                    "department_name": row.department_name,
                    "office_name": row.office_name,
                    "staff_no": row.staff_no,
                    "row_sha256": row.row_sha256,
                    "source_type": row.source_type,
                }
                for row in raw_staff_matches
            ],
            "normalized_expansions": list(intent.normalized_expansions),
            "matched_abbreviations": list(intent.matched_abbreviations),
            "fuzzy_candidates": [
                {
                    "name": candidate.display_name,
                    "score": candidate.score,
                    "match_type": candidate.match_type,
                    "token_overlap": candidate.token_overlap,
                    "first_token_similarity": candidate.first_token_similarity,
                    "last_token_similarity": candidate.last_token_similarity,
                }
                for candidate in fuzzy_candidates
            ],
            "fuzzy_band": fuzzy_band,
            "person_match_status": person_match_status,
            "count_debug": count_debug or {},
            "unsupported_reason": intent.unsupported_reason,
            "fallback_reason": fallback_reason,
            "answer_path": answer_path,
        }


def classify_structured_info_query(
    query: str,
    session_state: ChatSessionStateRecord | None = None,
) -> StructuredCurrentInfoQuery:
    """Return the normalized canonical structured-info intent for one query."""

    return normalize_current_info_query(query, session_state=session_state)


_GENERIC_STRUCTURED_MISS = (
    "I could not find a confident canonical SEBI people, board, organisation, or office match for that query."
)
_NO_MATCH_CURRENT_PERSON = (
    "No matching current directory entry was found in the ingested official SEBI data."
)


def _render_name_list(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"




def _group_org_structure(records) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        if not getattr(record, "leader_name", None) or not getattr(record, "department_name", None):
            continue
        grouped.setdefault(record.leader_name, [])
        if record.department_name not in grouped[record.leader_name]:
            grouped[record.leader_name].append(record.department_name)
    return grouped


def _sources_for_records(
    records: Iterable[object],
    *,
    preferred_source_types: tuple[str, ...] = (),
) -> tuple[CurrentInfoSource, ...]:
    seen: dict[tuple[str, str], CurrentInfoSource] = {}
    for record in records:
        source_types = tuple(getattr(record, "source_types", ()) or ())
        source_urls = tuple(getattr(record, "source_urls", ()) or ())
        if source_types and source_urls:
            for source_type, source_url in zip(source_types, source_urls):
                if not source_type or not source_url:
                    continue
                seen.setdefault(
                    (source_type, source_url),
                    CurrentInfoSource(
                        title=source_title_for_type(source_type),
                        url=source_url,
                        record_key=f"official:{source_type}",
                        domain=extract_domain(source_url),
                        source_type="structured",
                    ),
                )
            continue
        source_type = getattr(record, "source_type", None)
        source_url = getattr(record, "source_url", None)
        if not source_type or not source_url:
            continue
        seen.setdefault(
            (source_type, source_url),
            CurrentInfoSource(
                title=source_title_for_type(source_type),
                url=source_url,
                record_key=f"official:{source_type}",
                domain=extract_domain(source_url),
                source_type="structured",
            ),
        )
    ordered = list(seen.values())
    preferred = {value for value in preferred_source_types if value}
    ordered.sort(
        key=lambda source: (
            0 if source.record_key.replace("official:", "") in preferred else 1,
            source.record_key,
        )
    )
    return tuple(ordered)
