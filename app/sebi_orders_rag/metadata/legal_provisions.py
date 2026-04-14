"""Deterministic extraction and explanation of cited legal provisions."""

from __future__ import annotations

import re

from .models import ExtractedLegalProvision, MetadataChunkText, StoredLegalProvision

_STATUTE_ALIASES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "SEBI Act, 1992",
        ("sebi act", "securities and exchange board of india act", "sebi act 1992"),
        "The SEBI Act is the main statute that gives SEBI regulatory and enforcement powers over the securities market.",
    ),
    (
        "PFUTP Regulations",
        ("pfutp", "pfutp regulations", "fraudulent and unfair trade practices regulations"),
        "The PFUTP Regulations prohibit fraudulent, manipulative, deceptive, or unfair trade practices in the securities market.",
    ),
    (
        "PIT Regulations",
        ("pit", "pit regulations", "insider trading regulations"),
        "The PIT Regulations govern unpublished price sensitive information and prohibit insider trading.",
    ),
    (
        "ICDR Regulations",
        ("icdr", "icdr regulations", "issue of capital and disclosure requirements regulations"),
        "The ICDR Regulations govern public issues, disclosures, and issuer compliance requirements.",
    ),
    (
        "Investment Advisers Regulations",
        ("ia", "ia regulations", "investment advisers regulations"),
        "The Investment Advisers Regulations govern registration, conduct, and client-facing obligations of investment advisers.",
    ),
    (
        "Research Analysts Regulations",
        ("ra", "ra regulations", "research analysts regulations"),
        "The Research Analysts Regulations govern registration, disclosures, and conduct standards for research analysts.",
    ),
    (
        "REIT Regulations",
        ("reit", "reit regulations", "real estate investment trusts regulations"),
        "The REIT Regulations govern the structure, disclosure, and operation of real estate investment trusts.",
    ),
    (
        "Settlement Regulations",
        ("settlement regulations", "sebi settlement regulations"),
        "The Settlement Regulations govern how eligible enforcement proceedings may be resolved through settlement with SEBI.",
    ),
)
_STATUTE_PATTERN = "|".join(
    re.escape(alias)
    for _, aliases, _ in _STATUTE_ALIASES
    for alias in sorted(aliases, key=len, reverse=True)
)
_PROVISION_BLOCK_RE = re.compile(
    rf"\b(?P<kind>sections?|regulations?|rules?|schedules?)\s+"
    r"(?P<refs>[0-9a-zA-Z()\/.,\- ]{1,160}?)\s+of\s+(?:the\s+)?"
    rf"(?P<statute>{_STATUTE_PATTERN})\b",
    re.IGNORECASE,
)
_CONNECTOR_RE = re.compile(r"\b(?:and|read with)\b|[,;/]")
_NUMBER_TOKEN_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z()./-]*")


def extract_legal_provisions(
    *,
    document_version_id: int,
    chunks: tuple[MetadataChunkText, ...],
) -> tuple[ExtractedLegalProvision, ...]:
    """Extract distinct cited provisions from chunk text."""

    rows: dict[str, ExtractedLegalProvision] = {}
    for chunk in chunks:
        for match in _PROVISION_BLOCK_RE.finditer(chunk.text):
            statute_name = _canonical_statute_name(match.group("statute"))
            kind = match.group("kind").lower()
            provision_type = _normalize_provision_type(kind)
            snippet = _build_snippet(chunk.text, match.start(), match.end())
            for reference in _split_provision_refs(match.group("refs"), kind=provision_type):
                row = ExtractedLegalProvision(
                    document_version_id=document_version_id,
                    statute_name=statute_name,
                    section_or_regulation=reference,
                    provision_type=provision_type,
                    text_snippet=snippet,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
                rows.setdefault(row.row_sha256, row)
    return tuple(rows.values())


def explain_provisions(provisions: tuple[StoredLegalProvision, ...]) -> tuple[str, ...]:
    """Return concise plain-language explanations for stored provisions."""

    explanations: list[str] = []
    for provision in provisions:
        base = _statute_explanation(provision.statute_name)
        specific = _specific_explanation(
            statute_name=provision.statute_name,
            provision_ref=provision.section_or_regulation,
        )
        if specific:
            explanations.append(
                f"{provision.section_or_regulation} of {provision.statute_name}: {specific}"
            )
        else:
            explanations.append(
                f"{provision.section_or_regulation} of {provision.statute_name}: {base}"
            )
    return tuple(explanations)


def _canonical_statute_name(value: str) -> str:
    normalized = value.lower().strip()
    for canonical, aliases, _ in _STATUTE_ALIASES:
        if normalized in aliases:
            return canonical
    return value.strip()


def _normalize_provision_type(value: str) -> str:
    singular = value.lower().rstrip("s")
    if singular == "section":
        return "section"
    if singular == "regulation":
        return "regulation"
    if singular == "rule":
        return "rule"
    return "schedule"


def _split_provision_refs(value: str, *, kind: str) -> tuple[str, ...]:
    refs: list[str] = []
    for segment in _CONNECTOR_RE.split(value):
        for token in _NUMBER_TOKEN_RE.findall(segment):
            if token.lower() in {"section", "sections", "regulation", "regulations", "rule", "rules", "schedule", "schedules"}:
                continue
            refs.append(f"{kind.title()} {token}")
    return tuple(dict.fromkeys(refs))


def _build_snippet(text: str, start: int, end: int, width: int = 220) -> str:
    left = max(0, start - width // 3)
    right = min(len(text), end + width)
    snippet = re.sub(r"\s+", " ", text[left:right]).strip()
    return snippet[:width].rstrip()


def _statute_explanation(statute_name: str) -> str:
    for canonical, _, explanation in _STATUTE_ALIASES:
        if statute_name == canonical:
            return explanation
    return "This provision is one of the legal grounds cited in the order."


def _specific_explanation(*, statute_name: str, provision_ref: str) -> str | None:
    reference = provision_ref.lower()
    if statute_name == "SEBI Act, 1992":
        if reference.startswith("section 11b") or reference.startswith("section 11 "):
            return "These provisions deal with SEBI's power to issue directions and take protective action."
        if reference.startswith("section 12a"):
            return "These provisions prohibit fraudulent, manipulative, or deceptive conduct in securities transactions."
    if statute_name == "PFUTP Regulations":
        if reference.startswith("regulation 3"):
            return "This provision broadly prohibits fraudulent and unfair trade practices in the securities market."
        if reference.startswith("regulation 4"):
            return "This provision targets manipulative trades, misleading acts, and other deceptive market conduct."
    if statute_name == "PIT Regulations":
        return "These provisions regulate unpublished price sensitive information and insider-trading restrictions."
    return None
