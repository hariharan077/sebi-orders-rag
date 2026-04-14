"""Deterministic normalization for structured current-information queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..directory_data.canonicalize import normalize_designation, normalize_lookup_key
from ..directory_data.models import normalize_person_name, normalize_whitespace
from ..normalization import (
    expand_query,
    normalize_department_alias,
    normalize_designation_alias,
)
from ..schemas import ChatSessionStateRecord

_HOW_MANY_RE = re.compile(r"\bhow\s+ma(?:ny|y)\b", re.IGNORECASE)
_SEBI_TERM_RE = re.compile(r"\bsebi\b", re.IGNORECASE)
_BOARD_RE = re.compile(r"\bboard(?:\s+members?)?\b", re.IGNORECASE)
_CHAIRPERSON_RE = re.compile(r"\bchair(?:man|person)\b|\bheads?\s+sebi\b", re.IGNORECASE)
_WTM_RE = re.compile(r"\bwhole[- ]time members?\b|\bwhole time members?\b|\bwtms?\b", re.IGNORECASE)
_ED_RE = re.compile(r"\bexecutive directors?\b|\beds?\b", re.IGNORECASE)
_ORG_RE = re.compile(
    r"\borganisation structure\b|\borganization structure\b|\borg chart\b|\borganisational structure\b",
    re.IGNORECASE,
)
_OFFICE_RE = re.compile(
    r"\baddress\b|\blocation\b|\blocated\b|\bwhere is\b|\bwhere are\b|\boffice\b|\bcontact\b|\bphone\b|\btelephone\b|\bfax\b|\bemail\b",
    re.IGNORECASE,
)
_REGIONAL_DIRECTOR_RE = re.compile(r"\bregional director\b|\brd\b", re.IGNORECASE)
_JOINING_DATE_RE = re.compile(r"\b(?:date of joining|joining date|when did)\b", re.IGNORECASE)
_WHO_ARE_THEY_RE = re.compile(r"\bwho are they\b|\band who are they\b|\blist them\b", re.IGNORECASE)
_STAFF_ID_RE = re.compile(
    r"\b(?:whose|who(?:'s| is)?|which(?:\s+person)?\s+has)\s+staff\s*(?:id|number|no\.?)\s*(?:is|:)?\s*(?P<staff_no>[a-z0-9/-]{2,24})\b",
    re.IGNORECASE,
)
_STAFF_ID_SIMPLE_RE = re.compile(
    r"\bstaff\s*(?:id|number|no\.?)\s*(?:is|:)?\s*(?P<staff_no>[a-z0-9/-]{2,24})\b",
    re.IGNORECASE,
)
_CALLED_PERSON_RE = re.compile(
    r"\b(?:is there|do you have|do we have)\s+(?:(?:an?|the)\s+)?(?P<designation>[a-z][a-z -]+?)\s+called\s+(?P<name>[a-z][a-z .'-]{1,80})\b",
    re.IGNORECASE,
)
_PERSON_NAME_IN_SEBI_RE = re.compile(
    r"\bwho\s+is\s+(?P<name>[a-z][a-z .'-]{1,80})\s+(?:in|at)\s+sebi\b",
    re.IGNORECASE,
)
_PERSON_NAME_FROM_SEBI_RE = re.compile(
    r"\bwho\s+is\s+(?P<name>[a-z][a-z .'-]{1,80})\s+from\s+sebi\b",
    re.IGNORECASE,
)
_PLAIN_NAME_IN_SEBI_RE = re.compile(
    r"^(?P<name>[a-z]+(?:\s+[a-z]+){0,3})\s+(?:in|at)\s+sebi\b$",
    re.IGNORECASE,
)
_PLAIN_NAME_FROM_SEBI_RE = re.compile(
    r"^(?P<name>[a-z]+(?:\s+[a-z]+){0,3})\s+from\s+sebi\b$",
    re.IGNORECASE,
)
_PLAIN_NAME_SEBI_SUFFIX_RE = re.compile(
    r"^(?P<name>[a-z]+(?:\s+[a-z]+){0,3})\s+sebi\b$",
    re.IGNORECASE,
)
_JOIN_PERSON_RE = re.compile(r"\bwhen did\s+(?P<name>[a-z][a-z .'-]{1,80})\s+join\b", re.IGNORECASE)
_NUMBER_PERSON_RE = re.compile(
    r"\bwhat(?:'s| is)?\s+(?P<name>[a-z][a-z .'-]{1,80})'?s\s+(?:number|phone)\b",
    re.IGNORECASE,
)
_DETAIL_PERSON_RE = re.compile(
    r"\b(?:phone|number|email|mail|contact details?)\b.*?\b(?P<name>[a-z][a-z .'-]{1,80})\b",
    re.IGNORECASE,
)
_PLAIN_WHO_IS_PERSON_RE = re.compile(
    r"^\s*who\s+is\s+(?P<name>[a-z][a-z .'-]{1,80})\s*\??$",
    re.IGNORECASE,
)
_PERSON_IN_DEPARTMENT_RE = re.compile(
    r"\bis\s+there\s+(?:an?\s+|the\s+)?(?P<name>[a-z][a-z .'-]{1,80})\s+in\s+(?P<department>[a-z0-9 .&'/-]{2,80})\b",
    re.IGNORECASE,
)
_DESIGNATION_CALLED_PERSON_RE = re.compile(
    r"^\s*(?P<designation>[a-z][a-z -]{2,80})\s+called\s+(?P<name>[a-z][a-z .'-]{1,80})\s*\??$",
    re.IGNORECASE,
)
_DEPARTMENT_PEOPLE_RE = re.compile(
    r"\b(?:who\s+is|who\s+are|who'?s)\s+in\s+(?P<department>[a-z0-9 .&'/-]{2,80})\b",
    re.IGNORECASE,
)
_DIVISION_FILTER_RE = re.compile(r"\bdivision chief\b|\bid\s*\d+\b", re.IGNORECASE)
_DESIGNATION_OF_PERSON_RE = re.compile(
    r"\b(?:what(?:'s| is)\s+the\s+designation\s+of|designation\s+of)\s+(?P<name>[a-z][a-z .'-]{1,80})\b",
    re.IGNORECASE,
)
_PERSON_DESIGNATION_RE = re.compile(
    r"\b(?P<name>[a-z][a-z .'-]{1,80})'?s\s+designation\b",
    re.IGNORECASE,
)
_HOW_MANY_DESIGNATION_RE = re.compile(
    r"\bhow\s+ma(?:ny|y)\s+(?P<designation>[a-z][a-z -]{1,80}?)(?:\s+are\b|\s+is\b|\s+serving\b|\s+listed\b|\s+in\s+sebi\b)",
    re.IGNORECASE,
)
_FOLLOW_UP_PREFIXES = ("in ", "what about ", "and ", "about ")
_OFFICE_ALIAS_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("head office", ("head office", "headquarters")),
    ("regional office", ("regional office", "regional offices")),
    ("local office", ("local office", "local offices")),
    ("sebi bhavan", ("sebi bhavan",)),
    ("sebi bhavan ii", ("sebi bhavan ii", "sebi bhavan 2", "bhavan ii", "bhavan 2")),
    ("bkc", ("bkc", "bandra kurla complex")),
    ("nariman point", ("nariman point",)),
    ("ncl office", ("ncl office",)),
    ("nro", ("nro", "northern regional office")),
    ("sro", ("sro", "southern regional office")),
    ("ero", ("ero", "eastern regional office")),
    ("wro", ("wro", "western regional office")),
)
_CITY_ALIASES: tuple[tuple[str, str], ...] = (
    ("mumbai", "Mumbai"),
    ("bombay", "Mumbai"),
    ("new delhi", "Delhi"),
    ("delhi", "Delhi"),
    ("kolkata", "Kolkata"),
    ("calcutta", "Kolkata"),
    ("chennai", "Chennai"),
    ("madras", "Chennai"),
    ("ahmedabad", "Ahmedabad"),
    ("indore", "Indore"),
)
_IGNORED_ROLE_TOKENS = {
    "currently",
    "current",
    "listed",
    "public",
    "sebi",
    "serving",
    "there",
    "total",
}
_ORDERISH_QUERY_TERMS = {
    "order",
    "appeal",
    "matter",
    "case",
    "petition",
    "settlement",
    "judgment",
    "vs",
    "versus",
}
_ORDERISH_OFFICE_BLOCK_PHRASES = (
    "judgment dated",
    "judgement dated",
    "special court",
    "hon ble",
    "honble",
    "cnr no",
    "cc no",
    "in the matter of",
)
_PERSON_FRAGMENT_BLOCKLIST = {
    "address",
    "board",
    "chairperson",
    "contact",
    "delhi",
    "designation",
    "director",
    "email",
    "join",
    "manager",
    "member",
    "mumbai",
    "number",
    "office",
    "phone",
    "regional",
    "sebi",
}
_HISTORICAL_CUE_RE = re.compile(r"\b(?:previous|former|erstwhile|was)\b", re.IGNORECASE)


@dataclass(frozen=True)
class _ExtractedPersonQuery:
    person_name: str | None = None
    designation_hint: str | None = None
    department_hint: str | None = None
    wants_names: bool = False
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class StructuredCurrentInfoQuery:
    """Normalized structured-query intent shared by routing and lookup."""

    raw_query: str
    normalized_query: str
    query_family: str = "unsupported"
    extracted_city: str | None = None
    extracted_person_name: str | None = None
    extracted_staff_no: str | None = None
    designation_hint: str | None = None
    department_hint: str | None = None
    office_alias: str | None = None
    role_tokens: tuple[str, ...] = ()
    office_tokens: tuple[str, ...] = ()
    wants_address: bool = False
    wants_phone: bool = False
    wants_email: bool = False
    wants_fax: bool = False
    wants_joining_date: bool = False
    wants_count: bool = False
    wants_names: bool = False
    is_follow_up: bool = False
    normalized_expansions: tuple[str, ...] = ()
    matched_abbreviations: tuple[str, ...] = ()
    unsupported_reason: str | None = None

    @property
    def lookup_type(self) -> str:
        return self.query_family

    @property
    def person_name(self) -> str | None:
        return self.extracted_person_name

    @property
    def staff_no(self) -> str | None:
        return self.extracted_staff_no

    @property
    def office_hint(self) -> str | None:
        return self.raw_query


def normalize_current_info_query(
    query: str,
    *,
    session_state: ChatSessionStateRecord | None = None,
) -> StructuredCurrentInfoQuery:
    """Return a normalized structured-query interpretation for one user query."""

    expansion = expand_query(query, contexts=("current_people", "current_offices"))
    normalized_query = normalize_lookup_key(expansion.normalized_query)
    wants_address = any(
        phrase in normalized_query
        for phrase in ("address", "location", "located", "where is", "where are")
    )
    wants_phone = any(token in normalized_query for token in ("phone", "telephone", "number", "contact"))
    wants_email = any(token in normalized_query for token in ("email", "mail"))
    wants_fax = "fax" in normalized_query
    wants_joining_date = bool(_JOINING_DATE_RE.search(query))
    wants_count = bool(_HOW_MANY_RE.search(normalized_query)) or "total strength" in normalized_query
    wants_names = bool(_WHO_ARE_THEY_RE.search(normalized_query))
    extracted_city = _extract_city(normalized_query)
    office_alias = _extract_office_alias(normalized_query)
    office_tokens = tuple(
        token
        for token in (normalize_lookup_key(extracted_city), office_alias)
        if token
    )
    role_tokens = _extract_role_tokens(normalized_query)
    is_follow_up = _is_office_follow_up(normalized_query, session_state=session_state)

    staff_no = _extract_staff_id_query(query, normalized_query)
    if staff_no is not None:
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="staff_id_lookup",
            extracted_staff_no=staff_no,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    person_query = _extract_person_query(query, normalized_query)
    if person_query is not None:
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="person_lookup",
            extracted_person_name=person_query.person_name,
            designation_hint=person_query.designation_hint,
            department_hint=person_query.department_hint,
            role_tokens=role_tokens,
            wants_phone=wants_phone,
            wants_email=wants_email,
            wants_joining_date=wants_joining_date,
            wants_names=wants_names or person_query.wants_names,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
            unsupported_reason=person_query.unsupported_reason,
        )

    if _REGIONAL_DIRECTOR_RE.search(normalized_query):
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="regional_director",
            extracted_city=extracted_city,
            office_alias=office_alias,
            office_tokens=office_tokens,
            wants_phone=wants_phone,
            wants_email=wants_email,
            is_follow_up=is_follow_up,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    if _BOARD_RE.search(normalized_query) and "member" in normalized_query:
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="board_members",
            wants_count=wants_count,
            wants_names=wants_names,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    if _CHAIRPERSON_RE.search(normalized_query) and not _HISTORICAL_CUE_RE.search(normalized_query):
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="chairperson",
            wants_phone=wants_phone,
            wants_email=wants_email,
            wants_joining_date=wants_joining_date,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    has_wtm = bool(_WTM_RE.search(normalized_query))
    has_ed = bool(_ED_RE.search(normalized_query))
    if has_wtm and has_ed:
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="leadership_list",
            role_tokens=role_tokens,
            wants_count=True,
            wants_names=True,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )
    if has_wtm:
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="wtm_list",
            role_tokens=role_tokens,
            wants_count=True,
            wants_names=True,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )
    if has_ed:
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="ed_list",
            role_tokens=role_tokens,
            wants_count=True,
            wants_names=True,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    if _ORG_RE.search(normalized_query):
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="org_structure",
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    if "total strength" in normalized_query:
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="total_strength",
            wants_count=True,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    aggregate_designation = _extract_aggregate_designation(query, normalized_query)
    if aggregate_designation is not None:
        normalized_designation = _normalize_designation_phrase(aggregate_designation)
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="designation_count",
            designation_hint=normalized_designation,
            role_tokens=_role_tokens_for_designation(normalized_designation),
            wants_count=True,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    if _looks_like_office_query(
        normalized_query,
        extracted_city=extracted_city,
        office_alias=office_alias,
        is_follow_up=is_follow_up,
    ):
        return StructuredCurrentInfoQuery(
            raw_query=query,
            normalized_query=normalized_query,
            query_family="office_contact",
            extracted_city=extracted_city,
            office_alias=office_alias,
            office_tokens=office_tokens,
            wants_address=wants_address or "office" in normalized_query,
            wants_phone=wants_phone,
            wants_email=wants_email,
            wants_fax=wants_fax,
            is_follow_up=is_follow_up,
            normalized_expansions=expansion.expansions,
            matched_abbreviations=expansion.matched_abbreviations,
        )

    return StructuredCurrentInfoQuery(
        raw_query=query,
        normalized_query=normalized_query,
        query_family="unsupported",
        extracted_city=extracted_city,
        office_alias=office_alias,
        role_tokens=role_tokens,
        office_tokens=office_tokens,
        wants_address=wants_address,
        wants_phone=wants_phone,
        wants_email=wants_email,
        wants_fax=wants_fax,
        wants_joining_date=wants_joining_date,
        wants_count=wants_count,
        wants_names=wants_names,
        is_follow_up=is_follow_up,
        normalized_expansions=expansion.expansions,
        matched_abbreviations=expansion.matched_abbreviations,
    )


def _extract_person_query(query: str, normalized_query: str) -> _ExtractedPersonQuery | None:
    if _BOARD_RE.search(normalized_query) and "member" in normalized_query:
        return None
    if _CHAIRPERSON_RE.search(normalized_query) and not _HISTORICAL_CUE_RE.search(normalized_query):
        return None
    if _WTM_RE.search(normalized_query) or _ED_RE.search(normalized_query):
        return None
    if _REGIONAL_DIRECTOR_RE.search(normalized_query):
        return None
    if _ORG_RE.search(normalized_query):
        return None

    for pattern in (_PERSON_IN_DEPARTMENT_RE, _DESIGNATION_CALLED_PERSON_RE):
        match = pattern.search(query)
        if match is None:
            continue
        raw_name = normalize_person_name(normalize_whitespace(match.group("name")))
        if not raw_name or not _is_valid_person_name_candidate(raw_name):
            continue
        designation_hint = _normalize_designation_hint(match.groupdict().get("designation"))
        department_hint = normalize_department_alias(match.groupdict().get("department"))
        return _ExtractedPersonQuery(
            person_name=raw_name,
            designation_hint=designation_hint,
            department_hint=department_hint,
        )

    department_only = _extract_department_only_people_query(query)
    if department_only is not None:
        unsupported_reason = None
        if _DIVISION_FILTER_RE.search(query):
            unsupported_reason = (
                "The ingested official directory does not expose division-level hierarchy "
                "filters such as division-chief or ID-based department lookups."
            )
        return _ExtractedPersonQuery(
            department_hint=department_only,
            wants_names=True,
            unsupported_reason=unsupported_reason,
        )

    for pattern in (
        _CALLED_PERSON_RE,
        _DESIGNATION_OF_PERSON_RE,
        _PERSON_DESIGNATION_RE,
        _PERSON_NAME_IN_SEBI_RE,
        _PERSON_NAME_FROM_SEBI_RE,
        _PLAIN_NAME_IN_SEBI_RE,
        _PLAIN_NAME_FROM_SEBI_RE,
        _PLAIN_NAME_SEBI_SUFFIX_RE,
        _JOIN_PERSON_RE,
        _NUMBER_PERSON_RE,
        _DETAIL_PERSON_RE,
        _PLAIN_WHO_IS_PERSON_RE,
    ):
        match = pattern.search(query)
        if match is None:
            continue
        raw_name = normalize_person_name(normalize_whitespace(match.group("name")))
        if not raw_name or not _is_valid_person_name_candidate(raw_name):
            continue
        return _ExtractedPersonQuery(
            person_name=raw_name,
            designation_hint=_normalize_designation_hint(match.groupdict().get("designation")),
        )
    if _looks_like_person_name_fragment(query, normalized_query):
        normalized_name = normalize_person_name(normalize_whitespace(query))
        if normalized_name and _is_valid_person_name_candidate(normalized_name):
            return _ExtractedPersonQuery(person_name=normalized_name)
    return None


def _extract_staff_id_query(query: str, normalized_query: str) -> str | None:
    for pattern in (_STAFF_ID_RE, _STAFF_ID_SIMPLE_RE):
        match = pattern.search(query)
        if match is not None:
            return normalize_whitespace(match.group("staff_no"))
    if any(token in normalized_query for token in ("staff id", "staff number", "staff no")):
        match = _STAFF_ID_SIMPLE_RE.search(normalized_query)
        if match is not None:
            return normalize_whitespace(match.group("staff_no"))
    return None


def _is_valid_person_name_candidate(value: str) -> bool:
    normalized = normalize_lookup_key(value)
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return False
    if tokens[0] in {"what", "who", "how", "which", "where", "why", "when", "is", "are", "the"}:
        return False
    if any(token in _PERSON_FRAGMENT_BLOCKLIST for token in tokens):
        return False
    return True


def _extract_department_only_people_query(query: str) -> str | None:
    match = _DEPARTMENT_PEOPLE_RE.search(query)
    if match is not None:
        return normalize_department_alias(match.group("department"))
    if _DIVISION_FILTER_RE.search(query) and " of " in query.lower():
        _, _, tail = query.lower().partition(" of ")
        return normalize_department_alias(tail)
    return None


def _extract_aggregate_designation(query: str, normalized_query: str) -> str | None:
    match = _HOW_MANY_DESIGNATION_RE.search(normalized_query or query)
    if match is None:
        return None
    value = normalize_whitespace(match.group("designation"))
    if not value:
        return None
    lowered = normalize_lookup_key(value)
    if lowered in {"board members", "board member", "wtm", "wtms", "whole time members", "whole time member", "executive directors", "executive director"}:
        return None
    return value


def _extract_role_tokens(normalized_query: str) -> tuple[str, ...]:
    tokens: list[str] = []
    if _WTM_RE.search(normalized_query):
        tokens.append("wtm")
    if _ED_RE.search(normalized_query):
        tokens.append("executive_director")
    if _BOARD_RE.search(normalized_query) and "member" in normalized_query:
        tokens.append("board_member")
    if "assistant manager" in normalized_query:
        tokens.append("assistant_manager")
    if "deputy general manager" in normalized_query:
        tokens.append("deputy_general_manager")
    if "chief general manager" in normalized_query:
        tokens.append("chief_general_manager")
    if "regional director" in normalized_query:
        tokens.append("regional_director")
    return tuple(dict.fromkeys(tokens))


def _role_tokens_for_designation(designation: str | None) -> tuple[str, ...]:
    if not designation:
        return ()
    key = normalize_lookup_key(designation)
    return tuple(
        token
        for token in key.split()
        if token and token not in _IGNORED_ROLE_TOKENS
    )


def _normalize_designation_phrase(value: str) -> str:
    expansion = expand_query(value, contexts=("current_people",))
    cleaned = normalize_lookup_key(expansion.normalized_query)
    if cleaned.endswith("s") and not cleaned.endswith("ss"):
        cleaned = cleaned[:-1]
    return _normalize_designation_hint(cleaned) or cleaned.title()


def _normalize_designation_hint(value: str | None) -> str | None:
    normalized = normalize_designation_alias(value)
    if normalized is not None:
        return normalized
    return normalize_designation(normalize_whitespace(value))


def _extract_city(normalized_query: str) -> str | None:
    for alias, canonical in _CITY_ALIASES:
        if alias in normalized_query:
            return canonical
    return None


def _extract_office_alias(normalized_query: str) -> str | None:
    for label, aliases in _OFFICE_ALIAS_LABELS:
        if any(alias in normalized_query for alias in aliases):
            return label
    return None


def _is_office_follow_up(
    normalized_query: str,
    *,
    session_state: ChatSessionStateRecord | None,
) -> bool:
    if session_state is None or session_state.current_lookup_family != "office_contact":
        return False
    if not normalized_query:
        return False
    has_city_or_alias = _extract_city(normalized_query) is not None or _extract_office_alias(normalized_query) is not None
    if not has_city_or_alias:
        return False
    if len(normalized_query.split()) <= 5:
        return True
    return normalized_query.startswith(_FOLLOW_UP_PREFIXES)


def _looks_like_office_query(
    normalized_query: str,
    *,
    extracted_city: str | None,
    office_alias: str | None,
    is_follow_up: bool,
) -> bool:
    if is_follow_up:
        return True
    tokens = {token for token in normalized_query.split() if token}
    if any(term in tokens for term in _ORDERISH_QUERY_TERMS):
        return False
    if any(phrase in normalized_query for phrase in _ORDERISH_OFFICE_BLOCK_PHRASES):
        return False
    has_city_or_alias = extracted_city is not None or office_alias is not None
    if office_alias in {"head office", "regional office", "local office", "nro", "sro", "ero", "wro"}:
        return True
    if has_city_or_alias and _OFFICE_RE.search(normalized_query):
        return True
    if has_city_or_alias and _SEBI_TERM_RE.search(normalized_query):
        return True
    return False


def _looks_like_person_name_fragment(query: str, normalized_query: str) -> bool:
    tokens = [token for token in normalized_query.split() if token]
    if not normalized_query or any(term in tokens for term in _ORDERISH_QUERY_TERMS):
        return False
    if len(tokens) < 1 or len(tokens) > 4:
        return False
    if tokens[0] in {"what", "who", "how", "which", "where", "why", "when", "explain", "on", "about", "in", "at"}:
        return False
    if any(token in _PERSON_FRAGMENT_BLOCKLIST for token in tokens):
        return False
    if len(tokens) == 1:
        return tokens[0].isalpha() and len(tokens[0]) >= 4
    return all(token.isalpha() and len(token) >= 2 for token in tokens)
