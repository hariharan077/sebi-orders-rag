"""Entity and title alias normalization for SEBI order lookups."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_ORDER_PREFIXES: tuple[str, ...] = (
    "order in the matter of",
    "final order in the matter of",
    "settlement order in the matter of",
    "summary settlement order in the matter of",
    "order in respect of",
    "final order in respect of",
    "adjudication order in respect of",
    "adjudication order in the matter of",
    "enquiry order in the matter of",
    "exemption order in the matter of",
    "confirmatory order in the matter of",
    "revocation order in the matter of",
    "corrigendum to the",
    "in the matter of",
)
_VERSUS_RE = re.compile(r"\b(?:vs\.?|versus|v\.)\b")
_MS_RE = re.compile(r"\bm\s*/?\s*s\b", re.IGNORECASE)
_LTD_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (" private limited", " pvt ltd"),
    (" pvt ltd", " private limited"),
    (" private ltd", " private limited"),
    (" ltd", " limited"),
    (" limited", " ltd"),
)
_TERM_VARIANTS: dict[str, str] = {
    "solution": "solutions",
    "solutions": "solution",
    "advisor": "advisors",
    "advisors": "advisor",
    "service": "services",
    "services": "service",
}
_STOP_SUFFIX_TOKENS = {"order", "orders", "matter", "case", "sebi"}
_COMPANY_SUFFIX_TOKENS = {
    "limited",
    "ltd",
    "private",
    "pvt",
    "llp",
    "inc",
    "co",
    "company",
    "industries",
    "solutions",
    "advisor",
    "advisors",
}
_DEPARTMENT_ALIAS_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ITD", ("itd", "information technology department")),
    ("IVD", ("ivd", "investigation department")),
    ("MIRSD", ("mirsd", "market intermediaries regulation and supervision department")),
    ("OIAE", ("oiae", "office of investor assistance and education")),
    ("AFD", ("afd",)),
    ("IMD", ("imd", "investment management department")),
    ("CFD", ("cfd", "corporation finance department", "corporate finance department")),
)
_DESIGNATION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Whole-Time Member", ("wtm", "whole time member", "whole-time member")),
    ("Executive Director", ("ed", "executive director")),
    ("Assistant Manager", ("am", "assistant manager")),
    ("Deputy General Manager", ("dgm", "deputy general manager")),
    ("Chief General Manager", ("cgm", "chief general manager")),
    ("Regional Director", ("rd", "regional director")),
)


def normalize_alias_text(value: str) -> str:
    """Normalize free text into a comparable alias key."""

    cleaned = value.lower()
    cleaned = _MS_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def generate_order_alias_variants(value: str) -> tuple[str, ...]:
    """Generate conservative lookup variants for titles, entities, and person-name fragments."""

    base = normalize_alias_text(value)
    if not base:
        return ()

    variants: list[str] = []
    _append_variant(variants, base)
    _append_variant(variants, _strip_order_prefixes(base))
    _append_variant(variants, _strip_proprietor_clause(base))
    if _VERSUS_RE.search(value):
        versus_parts = _VERSUS_RE.split(base, maxsplit=1)
        if len(versus_parts) == 2:
            left, right = (part.strip() for part in versus_parts)
            _append_variant(variants, left)
            _append_variant(variants, right)
            if left and right:
                _append_variant(variants, f"{left} vs {right}")
                _append_variant(variants, f"{right} vs {left}")
            if right == "sebi" and left:
                _append_variant(variants, f"sebi vs {left}")
                _append_variant(variants, f"in the matter of {left}")
            if left == "sebi" and right:
                _append_variant(variants, f"{right} vs sebi")
                _append_variant(variants, f"in the matter of {right}")

    for current in tuple(variants):
        for source, target in _LTD_REPLACEMENTS:
            if source in f" {current} ":
                _append_variant(variants, current.replace(source.strip(), target.strip()))
        _append_variant(variants, _drop_trailing_initial(current))
        _append_variant(variants, _drop_person_trailing_token(current))
        for token, alternate in _TERM_VARIANTS.items():
            if f" {token} " in f" {current} ":
                _append_variant(variants, current.replace(token, alternate, 1))
        _append_variant(variants, _strip_terminal_order_terms(current))
    return tuple(variants)


def _strip_order_prefixes(value: str) -> str:
    current = value
    changed = True
    while changed:
        changed = False
        for prefix in _ORDER_PREFIXES:
            if current.startswith(prefix + " "):
                current = current[len(prefix) + 1 :].strip()
                changed = True
                break
    return current


def _strip_proprietor_clause(value: str) -> str:
    cleaned = re.sub(r"\bproprietor\b.*$", "", value, flags=re.IGNORECASE).strip()
    return cleaned or value


def _drop_trailing_initial(value: str) -> str:
    tokens = value.split()
    if len(tokens) >= 3 and len(tokens[-1]) == 1 and tokens[-1].isalpha():
        return " ".join(tokens[:-1])
    return value


def _drop_person_trailing_token(value: str) -> str:
    tokens = value.split()
    if len(tokens) != 3:
        return value
    if any(token in _COMPANY_SUFFIX_TOKENS for token in tokens):
        return value
    if all(token.isalpha() for token in tokens):
        return " ".join(tokens[:2])
    return value


def _strip_terminal_order_terms(value: str) -> str:
    tokens = value.split()
    while len(tokens) > 2 and tokens[-1] in _STOP_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _append_variant(variants: list[str], value: str) -> None:
    normalized = normalize_alias_text(value)
    if normalized and normalized not in variants:
        variants.append(normalized)


def normalize_department_alias(value: str | None) -> str | None:
    """Normalize SEBI department aliases into one stable uppercase label."""

    normalized = normalize_alias_text(value or "")
    if not normalized:
        return None
    for canonical, variants in _DEPARTMENT_ALIAS_PATTERNS:
        if any(
            normalized == variant
            or normalized.startswith(f"{variant} ")
            or f" {variant} " in f" {normalized} "
            for variant in variants
        ):
            return canonical
    if normalized.isalpha() and 2 <= len(normalized) <= 6:
        return normalized.upper()
    return None


def normalize_designation_alias(value: str | None) -> str | None:
    """Normalize short-form designation aliases into one stable display label."""

    normalized = normalize_alias_text(value or "")
    if not normalized:
        return None
    for canonical, variants in _DESIGNATION_ALIASES:
        if any(
            normalized == variant
            or normalized.startswith(f"{variant} ")
            or f" {variant} " in f" {normalized} "
            for variant in variants
        ):
            return canonical
    return None
