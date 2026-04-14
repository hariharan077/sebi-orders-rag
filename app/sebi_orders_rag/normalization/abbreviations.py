"""Context-sensitive abbreviation expansion for SEBI queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_GLUED_TERM_REWRITES: tuple[tuple[str, str], ...] = (
    ("assistantmanagers", "assistant managers"),
    ("assistantmanager", "assistant manager"),
    ("wholetimemembers", "whole time members"),
    ("wholetimemember", "whole time member"),
    ("boardmembers", "board members"),
    ("boardmember", "board member"),
)


@dataclass(frozen=True)
class AbbreviationMatch:
    """One matched abbreviation or short-form normalization."""

    variant: str
    canonical: str
    family: str
    context: str


@dataclass(frozen=True)
class AbbreviationExpansion:
    """Expanded query plus the matched abbreviation families."""

    raw_query: str
    normalized_query: str
    expansions: tuple[str, ...] = ()
    matches: tuple[AbbreviationMatch, ...] = ()

    @property
    def matched_abbreviations(self) -> tuple[str, ...]:
        values: list[str] = []
        seen: set[str] = set()
        for match in self.matches:
            if match.variant in seen:
                continue
            seen.add(match.variant)
            values.append(match.variant)
        return tuple(values)


@dataclass(frozen=True)
class _AbbreviationRule:
    family: str
    canonical: str
    variants: tuple[str, ...]
    context: str

    @property
    def pattern(self) -> re.Pattern[str]:
        escaped = sorted((re.escape(value) for value in self.variants), key=len, reverse=True)
        return re.compile(rf"\b(?:{'|'.join(escaped)})\b", re.IGNORECASE)


_CURRENT_INFO_RULES: tuple[_AbbreviationRule, ...] = (
    _AbbreviationRule("wtm", "whole time member", ("wtm", "wtms", "whole-time member", "whole-time members"), "current_people"),
    _AbbreviationRule("ed", "executive director", ("ed", "eds", "executive directors"), "current_people"),
    _AbbreviationRule("am", "assistant manager", ("am", "ams", "assistant managers"), "current_people"),
    _AbbreviationRule("dgm", "deputy general manager", ("dgm", "dgm s", "dgms"), "current_people"),
    _AbbreviationRule("cgm", "chief general manager", ("cgm", "cgms"), "current_people"),
    _AbbreviationRule("rd", "regional director", ("rd", "rds"), "current_people"),
    _AbbreviationRule("chairman", "chairperson", ("chairman",), "current_people"),
    _AbbreviationRule("ro", "regional office", ("ro", "ros"), "current_offices"),
    _AbbreviationRule("lo", "local office", ("lo", "los"), "current_offices"),
    _AbbreviationRule("nro", "northern regional office", ("nro",), "current_offices"),
    _AbbreviationRule("sro", "southern regional office", ("sro",), "current_offices"),
    _AbbreviationRule("ero", "eastern regional office", ("ero",), "current_offices"),
    _AbbreviationRule("wro", "western regional office", ("wro",), "current_offices"),
)

_ORDER_RULES: tuple[_AbbreviationRule, ...] = (
    _AbbreviationRule("wtm", "whole time member", ("wtm", "wtms", "whole-time member", "whole-time members"), "order_lookup"),
    _AbbreviationRule("chairman", "chairperson", ("chairman",), "order_lookup"),
    _AbbreviationRule("pfutp", "pfutp regulations", ("pfutp",), "order_legal"),
    _AbbreviationRule("pit", "pit regulations", ("pit",), "order_legal"),
    _AbbreviationRule("icdr", "icdr regulations", ("icdr",), "order_legal"),
    _AbbreviationRule("ia", "investment advisers regulations", ("ia",), "order_legal"),
    _AbbreviationRule("ra", "research analysts regulations", ("ra",), "order_legal"),
    _AbbreviationRule("reit", "reit regulations", ("reit",), "order_legal"),
    _AbbreviationRule("nbfc", "non banking financial company", ("nbfc",), "order_lookup"),
    _AbbreviationRule("drhp", "draft red herring prospectus", ("drhp",), "order_lookup"),
    _AbbreviationRule("ofs", "offer for sale", ("ofs",), "order_lookup"),
    _AbbreviationRule("scn", "show cause notice", ("scn",), "order_lookup"),
    _AbbreviationRule("ao", "adjudicating officer", ("ao",), "order_lookup"),
    _AbbreviationRule("sat", "securities appellate tribunal", ("sat",), "order_lookup"),
)

_ORDER_ED_EXECUTIVE = _AbbreviationRule("ed", "executive director", ("ed", "eds"), "order_lookup")
_ORDER_ED_DEPARTMENT = _AbbreviationRule("ed", "enforcement department", ("ed",), "order_lookup")
_DEPARTMENT_CUE_RE = re.compile(r"\bdepartment\b|\benforcement\b|\binvestigation\b|\binspection\b", re.IGNORECASE)


def expand_abbreviations(
    query: str,
    *,
    contexts: Iterable[str] = (),
) -> AbbreviationExpansion:
    """Expand only the abbreviations relevant to the requested contexts."""

    normalized_query = normalize_abbreviation_text(query)
    if not normalized_query:
        return AbbreviationExpansion(raw_query=query, normalized_query="")

    matched: list[AbbreviationMatch] = []
    expansions: list[str] = [normalized_query]
    current = normalized_query
    for rule in _rules_for_contexts(normalized_query, contexts):
        rule_matches = [normalize_abbreviation_text(item.group(0)) for item in rule.pattern.finditer(current)]
        if not rule_matches:
            continue
        current = rule.pattern.sub(rule.canonical, current)
        current = normalize_abbreviation_text(current)
        if current not in expansions:
            expansions.append(current)
        for variant in dict.fromkeys(rule_matches):
            matched.append(
                AbbreviationMatch(
                    variant=variant,
                    canonical=rule.canonical,
                    family=rule.family,
                    context=rule.context,
                )
            )
    return AbbreviationExpansion(
        raw_query=query,
        normalized_query=current,
        expansions=tuple(expansions),
        matches=tuple(matched),
    )


def normalize_abbreviation_text(value: str) -> str:
    """Normalize text without applying any semantic rewrites."""

    normalized = value.strip().lower()
    for source, target in _GLUED_TERM_REWRITES:
        normalized = normalized.replace(source, target)
    return _WHITESPACE_RE.sub(" ", normalized)


def _rules_for_contexts(
    normalized_query: str,
    contexts: Iterable[str],
) -> tuple[_AbbreviationRule, ...]:
    requested = {value for value in contexts if value}
    rules: list[_AbbreviationRule] = []
    if "current_people" in requested:
        rules.extend(rule for rule in _CURRENT_INFO_RULES if rule.context == "current_people")
    if "current_offices" in requested:
        rules.extend(rule for rule in _CURRENT_INFO_RULES if rule.context == "current_offices")
    if requested.intersection({"order_lookup", "order_legal"}):
        rules.extend(rule for rule in _ORDER_RULES if rule.context in requested or rule.context == "order_lookup")
        rules.append(_ORDER_ED_DEPARTMENT if _DEPARTMENT_CUE_RE.search(normalized_query) else _ORDER_ED_EXECUTIVE)
    return tuple(rules)
