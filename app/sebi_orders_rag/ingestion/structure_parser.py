"""Deterministic legal-structure parsing for SEBI order documents."""

from __future__ import annotations

import re

from ..schemas import ExtractedPage, HeadingMatch, ParsedDocument, SectionType, StructuredBlock
from ..utils.strings import uppercase_ratio
from .token_count import token_count

_NUMBERED_HEADING_RE = re.compile(r"^(?P<label>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<title>.+)$")
_LETTER_HEADING_RE = re.compile(r"^\((?P<label>[A-Za-z])\)\s+(?P<title>.+)$")
_ISSUE_HEADING_RE = re.compile(r"^(?P<label>Issue\s+[IVXLCM0-9]+)\b[:.) -]*(?P<title>.*)$", re.IGNORECASE)
_TOKEN_SPLIT_RE = re.compile(r"\s+")
_NUMBERED_LABEL_PREFIX_RE = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|\([A-Za-z0-9]+\)|Issue\s+[IVXLCM0-9]+\b)", re.IGNORECASE)

_SECTION_PATTERNS: tuple[tuple[re.Pattern[str], SectionType], ...] = (
    (
        re.compile(
            r"\b(summary\s+settlement\s+order|settlement\s+order|"
            r"ex-?parte\s+interim\s+order|final\s+order|interim\s+order|"
            r"operative\s+order|order)\b",
            re.IGNORECASE,
        ),
        "operative_order",
    ),
    (re.compile(r"\bannexure\b", re.IGNORECASE), "annexure"),
    (re.compile(r"\bshow\s+cause\s+notice\b", re.IGNORECASE), "show_cause_notice"),
    (re.compile(r"\bbackground\b", re.IGNORECASE), "background"),
    (re.compile(r"\bfacts?\b", re.IGNORECASE), "facts"),
    (re.compile(r"\ballegations?\b", re.IGNORECASE), "allegations"),
    (re.compile(r"\b(reply|replies|submissions?)\b", re.IGNORECASE), "reply_or_submissions"),
    (
        re.compile(
            r"\b(findings?|consideration of issues and findings)\b",
            re.IGNORECASE,
        ),
        "findings",
    ),
    (re.compile(r"\bissues?\b", re.IGNORECASE), "issues"),
    (re.compile(r"\bdirections?\b", re.IGNORECASE), "directions"),
)
_CONNECTOR_WORDS = {
    "of",
    "and",
    "or",
    "the",
    "to",
    "for",
    "in",
    "on",
    "with",
    "by",
    "at",
    "under",
}
_TABLE_KEYWORDS = (
    "isin",
    "pan",
    "settlement application number",
    "name of the applicant",
    "date of allotment",
    "outstanding amount",
    "sr.no",
    "sr no",
    "no. of",
)


def detect_heading(
    line: str,
    *,
    min_heading_caps_ratio: float,
) -> HeadingMatch | None:
    """Return heading metadata when a line looks like a legal section heading."""

    candidate = line.strip().strip(":")
    if not candidate or len(candidate) > 160:
        return None

    section_type = _classify_section_type(candidate)
    numbered_match = _NUMBERED_HEADING_RE.match(candidate)
    if numbered_match and _looks_like_heading_title(numbered_match.group("title"), allow_sentence_case=False):
        label = numbered_match.group("label")
        level = label.count(".") + 1
        return HeadingMatch(title=candidate, section_type=section_type, level=level)

    letter_match = _LETTER_HEADING_RE.match(candidate)
    if letter_match and _looks_like_heading_title(letter_match.group("title"), allow_sentence_case=False):
        return HeadingMatch(title=candidate, section_type=section_type, level=3)

    issue_match = _ISSUE_HEADING_RE.match(candidate)
    if issue_match:
        return HeadingMatch(title=candidate, section_type=section_type, level=2)

    if _looks_like_short_caps_heading(candidate, min_heading_caps_ratio=min_heading_caps_ratio):
        return HeadingMatch(title=candidate, section_type=section_type, level=1)

    if section_type != "other" and _looks_like_heading_title(candidate, allow_sentence_case=True):
        return HeadingMatch(title=candidate, section_type=section_type, level=1)

    return None


def parse_document_structure(
    pages: tuple[ExtractedPage, ...],
    *,
    min_heading_caps_ratio: float,
    model_name: str,
) -> ParsedDocument:
    """Convert normalized pages into an ordered block structure."""

    blocks: list[StructuredBlock] = []
    current_heading_path: tuple[str, ...] = ()
    current_section_title: str | None = None
    current_section_type: SectionType = "header"
    block_index = 0

    for page in pages:
        raw_blocks = [part.strip() for part in page.final_text.split("\n\n") if part.strip()]
        pending_table_blocks: list[str] = []

        def flush_table_blocks() -> None:
            nonlocal block_index
            if not pending_table_blocks:
                return
            table_text = "\n".join(pending_table_blocks)
            blocks.append(
                StructuredBlock(
                    block_index=block_index,
                    page_no=page.page_no,
                    block_type="table_block",
                    text=table_text,
                    token_count=token_count(table_text, model_name=model_name),
                    section_type="table_block",
                    section_title=current_section_title,
                    heading_path=current_heading_path,
                    heading_level=None,
                )
            )
            block_index += 1
            pending_table_blocks.clear()

        for raw_block in raw_blocks:
            heading = detect_heading(
                raw_block,
                min_heading_caps_ratio=min_heading_caps_ratio,
            )
            if heading is not None:
                flush_table_blocks()
                resolved_level = _resolve_heading_level(
                    current_heading_path,
                    heading.title,
                    heading.level,
                )
                current_heading_path = _update_heading_path(
                    current_heading_path,
                    heading.title,
                    resolved_level,
                )
                current_section_title = heading.title
                current_section_type = heading.section_type
                blocks.append(
                    StructuredBlock(
                        block_index=block_index,
                        page_no=page.page_no,
                        block_type="heading",
                        text=heading.title,
                        token_count=token_count(heading.title, model_name=model_name),
                        section_type=current_section_type,
                        section_title=current_section_title,
                        heading_path=current_heading_path,
                        heading_level=resolved_level,
                    )
                )
                block_index += 1
                continue

            if _is_table_like_block(raw_block):
                pending_table_blocks.append(raw_block)
                continue

            flush_table_blocks()
            blocks.append(
                StructuredBlock(
                    block_index=block_index,
                    page_no=page.page_no,
                    block_type="paragraph",
                    text=raw_block,
                    token_count=token_count(raw_block, model_name=model_name),
                    section_type=current_section_type,
                    section_title=current_section_title,
                    heading_path=current_heading_path,
                    heading_level=None,
                )
            )
            block_index += 1

        flush_table_blocks()

    return ParsedDocument(blocks=tuple(blocks))


def _classify_section_type(text: str) -> SectionType:
    for pattern, section_type in _SECTION_PATTERNS:
        if pattern.search(text):
            return section_type
    return "other"


def _looks_like_short_caps_heading(text: str, *, min_heading_caps_ratio: float) -> bool:
    if len(text.split()) > 18:
        return False
    if text.endswith("."):
        return False
    if len(text.split()) <= 3 and _classify_section_type(text) == "other":
        return False
    return uppercase_ratio(text) >= min_heading_caps_ratio


def _looks_like_heading_title(text: str, *, allow_sentence_case: bool) -> bool:
    normalized = text.strip().strip(":")
    if not normalized or normalized.endswith("."):
        return False
    words = [word for word in _TOKEN_SPLIT_RE.split(normalized) if word]
    if len(words) > 18:
        return False

    scored_words = 0
    heading_words = 0
    for word in words:
        cleaned = word.strip("(),.;:-")
        if not cleaned:
            continue
        if cleaned.isdigit():
            continue
        scored_words += 1
        lower = cleaned.lower()
        if lower in _CONNECTOR_WORDS:
            heading_words += 1
            continue
        if cleaned.isupper() or cleaned[:1].isupper():
            heading_words += 1

    if scored_words == 0:
        return False

    ratio = heading_words / scored_words
    if ratio >= 0.75:
        return True
    return allow_sentence_case and ratio >= 0.50


def _is_table_like_block(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    lowered_text = text.lower()
    keyword_hits = sum(1 for keyword in _TABLE_KEYWORDS if keyword in lowered_text)
    if "|" in text:
        return True
    if keyword_hits >= 2:
        return True
    if len(lines) < 2:
        return False
    numeric_heavy_lines = 0
    for line in lines:
        cells = [cell for cell in re.split(r"\s{2,}", line) if cell]
        if len(cells) >= 3:
            numeric_heavy_lines += 1
    return numeric_heavy_lines >= 2


def _update_heading_path(
    current_path: tuple[str, ...],
    title: str,
    level: int,
) -> tuple[str, ...]:
    normalized_level = max(1, level)
    kept_path = current_path[: normalized_level - 1]
    return kept_path + (title,)


def _resolve_heading_level(
    current_path: tuple[str, ...],
    title: str,
    detected_level: int,
) -> int:
    if (
        detected_level == 1
        and current_path
        and _NUMBERED_LABEL_PREFIX_RE.match(title)
        and not _NUMBERED_LABEL_PREFIX_RE.match(current_path[-1])
    ):
        return 2
    return detected_level
