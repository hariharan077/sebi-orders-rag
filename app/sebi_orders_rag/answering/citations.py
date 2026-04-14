"""Citation helpers for retrieval-grounded and web-backed answers."""

from __future__ import annotations

import re

from ..schemas import Citation, PromptContextChunk
from ..web_fallback.ranking import extract_domain

_CITATION_RE = re.compile(r"\[(\d+)\]")
_INLINE_CITATION_GROUP_RE = re.compile(r"(?:\s*\[(\d+)\])+")


def build_citations(context_chunks: tuple[PromptContextChunk, ...]) -> tuple[Citation, ...]:
    """Return the structured citation objects for prompt context chunks."""

    return tuple(
        Citation(
            citation_number=chunk.citation_number,
            record_key=chunk.record_key,
            title=chunk.title,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_type=chunk.section_type,
            document_version_id=chunk.document_version_id,
            chunk_id=chunk.chunk_id,
            detail_url=chunk.detail_url,
            pdf_url=chunk.pdf_url,
        )
        for chunk in context_chunks
    )


def build_external_citations(sources: tuple[object, ...]) -> tuple[Citation, ...]:
    """Return structured citations for non-corpus sources."""

    citations: list[Citation] = []
    for index, source in enumerate(sources, start=1):
        url = str(getattr(source, "url", getattr(source, "source_url", "")) or "").strip()
        title = str(
            getattr(source, "title", getattr(source, "source_title", "")) or url
        ).strip()
        citations.append(
            Citation(
                citation_number=index,
                record_key=str(getattr(source, "record_key", "") or ""),
                title=title,
                page_start=None,
                page_end=None,
                section_type=None,
                document_version_id=None,
                chunk_id=None,
                detail_url=None,
                pdf_url=None,
                source_url=url or None,
                source_title=title,
                domain=str(getattr(source, "domain", "") or extract_domain(url) or ""),
                source_type=str(getattr(source, "source_type", "") or ""),
                snippet=getattr(source, "snippet", None),
            )
        )
    return tuple(citations)


def extract_citation_numbers(answer_text: str) -> tuple[int, ...]:
    """Extract inline citation markers like ``[1]`` in stable order."""

    seen: set[int] = set()
    ordered: list[int] = []
    for match in _CITATION_RE.finditer(answer_text):
        number = int(match.group(1))
        if number in seen:
            continue
        seen.add(number)
        ordered.append(number)
    return tuple(ordered)


def filter_citations(
    citations: tuple[Citation, ...],
    *,
    cited_numbers: tuple[int, ...],
) -> tuple[Citation, ...]:
    """Return only the citations referenced in the answer text."""

    allowed = set(cited_numbers)
    return tuple(citation for citation in citations if citation.citation_number in allowed)


def resolve_cited_context_chunks(
    context_chunks: tuple[PromptContextChunk, ...],
    *,
    cited_numbers: tuple[int, ...],
) -> tuple[PromptContextChunk, ...]:
    """Return the prompt context chunks referenced by inline citation markers."""

    allowed = set(cited_numbers)
    return tuple(chunk for chunk in context_chunks if chunk.citation_number in allowed)


def strip_inline_citation_markers(answer_text: str) -> str:
    """Remove inline numeric citation markers like ``[1][2]`` from answer text."""

    cleaned = _INLINE_CITATION_GROUP_RE.sub("", answer_text or "")
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
