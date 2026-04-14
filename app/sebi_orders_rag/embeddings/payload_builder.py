"""Deterministic payload builders for Phase 3 hierarchical embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..ingestion.token_count import split_token_windows, token_count
from ..schemas import EmbeddingCandidate, NodePayload, SectionGroupInput, StoredChunk
from ..utils.strings import collapse_inline_whitespace

_MAX_MAJOR_HEADINGS = 8
_MAX_OPENING_LINES = 6
_MAX_EMBEDDING_NODE_TOKENS = 7800


def build_document_node_payload(
    document: EmbeddingCandidate,
    *,
    sections: Sequence[SectionGroupInput],
    chunks: Sequence[StoredChunk],
    model_name: str,
) -> NodePayload:
    """Build one deterministic document-level node payload."""

    major_headings = _collect_major_headings(sections)
    opening_lines = _collect_opening_lines(chunks)
    lines = [
        f"Document title: {document.title}",
        f"Bucket: {document.bucket_name}",
        f"Record key: {document.record_key}",
    ]
    if document.order_date is not None:
        lines.append(f"Order date: {document.order_date.isoformat()}")
    if document.external_record_id:
        lines.append(f"External record id: {document.external_record_id}")

    procedural_type = infer_procedural_type(
        title=document.title,
        bucket_name=document.bucket_name,
        major_headings=major_headings,
    )
    if procedural_type is not None:
        lines.append(f"Procedural type: {procedural_type}")
    if major_headings:
        lines.append("Major headings: " + "; ".join(major_headings))
    if opening_lines:
        lines.append("Opening lines:")
        lines.extend(f"- {line}" for line in opening_lines)

    node_text, text_truncated = _cap_embedding_node_text(
        "\n".join(lines).strip(),
        model_name=model_name,
    )
    return NodePayload(
        node_text=node_text,
        token_count=token_count(node_text, model_name=model_name),
        metadata={
            "record_key": document.record_key,
            "bucket_name": document.bucket_name,
            "external_record_id": document.external_record_id,
            "major_headings": list(major_headings),
            "opening_lines": list(opening_lines),
            "procedural_type": procedural_type,
            "text_truncated": text_truncated,
        },
    )


def build_section_node_payload(
    document: EmbeddingCandidate,
    section: SectionGroupInput,
    *,
    model_name: str,
) -> NodePayload:
    """Build one deterministic section-level node payload."""

    lines = [
        f"Document title: {document.title}",
        f"Bucket: {document.bucket_name}",
        f"Record key: {document.record_key}",
        f"Section key: {section.section_key}",
        f"Section type: {section.section_type}",
    ]
    if section.section_title:
        lines.append(f"Section title: {section.section_title}")
    if section.heading_path:
        lines.append(f"Heading path: {section.heading_path}")
    lines.append(f"Page range: {section.page_start}-{section.page_end}")
    lines.append("Section text:")
    lines.append(_combine_section_chunk_texts(section.chunks))

    node_text, text_truncated = _cap_embedding_node_text(
        "\n".join(lines).strip(),
        model_name=model_name,
    )
    return NodePayload(
        node_text=node_text,
        token_count=token_count(node_text, model_name=model_name),
        metadata={
            "section_key": section.section_key,
            "section_type": section.section_type,
            "section_title": section.section_title,
            "heading_path": section.heading_path,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "chunk_ids": [chunk.chunk_id for chunk in section.chunks],
            "text_truncated": text_truncated,
        },
    )


def build_chunk_embedding_text(chunk: StoredChunk) -> str:
    """Build embedding text for one chunk with light section context."""

    prefix_lines = [f"Section type: {chunk.section_type}"]
    if chunk.section_title:
        prefix_lines.append(f"Section title: {chunk.section_title}")
    if chunk.heading_path and chunk.heading_path != chunk.section_title:
        prefix_lines.append(f"Heading path: {chunk.heading_path}")
    return "\n".join(prefix_lines + ["", chunk.chunk_text.strip()]).strip()


def infer_procedural_type(
    *,
    title: str,
    bucket_name: str,
    major_headings: Sequence[str],
) -> str | None:
    """Infer a stable procedural type string from title and bucket metadata."""

    lowered_title = title.lower()
    lowered_bucket = bucket_name.lower()
    lowered_headings = " ".join(heading.lower() for heading in major_headings)

    heuristics = (
        ("rti appeal order", ("rti", "appellate authority")),
        ("settlement order", ("settlement order",)),
        ("adjudication order", ("adjudication order",)),
        ("appeal order", ("appeal",)),
        ("final order", ("final order",)),
        ("interim order", ("interim order", "ex-parte interim order")),
        ("confirmatory order", ("confirmatory order",)),
        ("exemption order", ("exemption order",)),
        ("enquiry order", ("enquiry order",)),
        ("revocation order", ("revocation order",)),
    )
    combined_text = " ".join((lowered_title, lowered_bucket, lowered_headings))
    for label, patterns in heuristics:
        if any(pattern in combined_text for pattern in patterns):
            return label
    if lowered_bucket:
        return lowered_bucket.replace("-", " ")
    return None


def _collect_major_headings(sections: Sequence[SectionGroupInput]) -> tuple[str, ...]:
    headings: list[str] = []
    seen: set[str] = set()
    for section in sections:
        candidate = section.section_title or section.heading_path or section.section_type
        cleaned = collapse_inline_whitespace(candidate or "")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        headings.append(cleaned)
        if len(headings) >= _MAX_MAJOR_HEADINGS:
            break
    return tuple(headings)


def _collect_opening_lines(chunks: Sequence[StoredChunk]) -> tuple[str, ...]:
    opening_lines: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        for raw_line in _chunk_lines(chunk):
            cleaned = collapse_inline_whitespace(raw_line)
            if len(cleaned) < 20:
                continue
            if cleaned.upper() == cleaned and len(cleaned.split()) <= 10:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            opening_lines.append(cleaned)
            if len(opening_lines) >= _MAX_OPENING_LINES:
                return tuple(opening_lines)
    return tuple(opening_lines)


def _chunk_lines(chunk: StoredChunk) -> Iterable[str]:
    for line in chunk.chunk_text.splitlines():
        if line.strip():
            yield line


def _combine_section_chunk_texts(chunks: Sequence[StoredChunk]) -> str:
    return "\n\n".join(chunk.chunk_text.strip() for chunk in chunks if chunk.chunk_text.strip())


def _cap_embedding_node_text(text: str, *, model_name: str) -> tuple[str, bool]:
    if token_count(text, model_name=model_name) <= _MAX_EMBEDDING_NODE_TOKENS:
        return text, False
    windows = split_token_windows(
        text,
        model_name=model_name,
        max_tokens=_MAX_EMBEDDING_NODE_TOKENS,
        overlap_tokens=0,
    )
    if not windows:
        return text, False
    return windows[0], True
