"""Text normalization helpers for PDF extraction output."""

from __future__ import annotations

import re

from ..utils.strings import collapse_inline_whitespace, normalize_newlines

_DISPOSABLE_LINE_PATTERNS = (
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE),
    re.compile(r"^\d+\s*/\s*\d+$"),
)
_NUMBERED_PARAGRAPH_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?|\([A-Za-z0-9]+\)|[IVXLCMivxlcm]+\.)\s+\S"
)
_ISSUE_LABEL_RE = re.compile(r"^issue\s+[IVXLCM0-9]+\b", re.IGNORECASE)
_SHORT_ALL_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,:;()/&.'-]{1,120}$")
_KNOWN_HEADING_PHRASE_RE = re.compile(
    r"\b("
    r"background|facts?|allegations?|show cause notice|reply|replies|"
    r"submissions?|findings?|directions?|order|operative order|annexure|"
    r"consideration of issues and findings"
    r")\b",
    re.IGNORECASE,
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_extracted_text(text: str) -> str:
    """Normalize PDF text while preserving legal numbering and headings."""

    if not text:
        return ""

    sanitized = (
        normalize_newlines(text)
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u00ad", "")
    )

    output_blocks: list[str] = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        output_blocks.append(_merge_paragraph_lines(paragraph_buffer))
        paragraph_buffer.clear()

    for raw_line in sanitized.split("\n"):
        line = collapse_inline_whitespace(raw_line)
        if not line:
            flush_paragraph()
            continue
        if _is_disposable_line(line):
            flush_paragraph()
            continue
        if _looks_like_heading_candidate(line):
            flush_paragraph()
            output_blocks.append(line)
            continue
        if _starts_new_paragraph(line) and paragraph_buffer:
            flush_paragraph()
        paragraph_buffer.append(line)

    flush_paragraph()

    normalized = "\n\n".join(block for block in output_blocks if block)
    normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def _merge_paragraph_lines(lines: list[str]) -> str:
    if not lines:
        return ""

    merged = lines[0]
    for line in lines[1:]:
        if merged.endswith("-") and line[:1].islower():
            merged = f"{merged[:-1]}{line}"
        else:
            merged = f"{merged} {line}"
    return merged.strip()


def _is_disposable_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in _DISPOSABLE_LINE_PATTERNS)


def _starts_new_paragraph(line: str) -> bool:
    return bool(_NUMBERED_PARAGRAPH_RE.match(line) or _ISSUE_LABEL_RE.match(line))


def _looks_like_heading_candidate(line: str) -> bool:
    if len(line) > 140:
        return False
    if _SHORT_ALL_CAPS_RE.match(line):
        return True
    if line.endswith((".", ";", ",")):
        return False
    return bool(_KNOWN_HEADING_PHRASE_RE.search(line) and len(line.split()) <= 12)
