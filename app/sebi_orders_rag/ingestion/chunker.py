"""Section-aware legal chunking for Phase 2."""

from __future__ import annotations

import re
from dataclasses import replace

from ..schemas import ChunkRecord, ParsedDocument, StructuredBlock
from ..utils.strings import sha256_hexdigest
from .token_count import split_token_windows, token_count

_OPERATIVE_SECTION_TYPES = {"directions", "operative_order"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(])")


def build_chunks(
    parsed_document: ParsedDocument,
    *,
    model_name: str,
    target_chunk_tokens: int,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> tuple[ChunkRecord, ...]:
    """Build deterministic section-aware chunks from parsed document blocks."""

    if not parsed_document.blocks:
        return ()

    sections = _group_major_sections(parsed_document.blocks)
    chunk_records: list[ChunkRecord] = []
    chunk_index = 0

    for section_blocks in sections:
        expanded_blocks = _expand_oversized_blocks(
            section_blocks,
            model_name=model_name,
            target_chunk_tokens=target_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
        )
        section_chunk_blocks = _chunk_section(
            expanded_blocks,
            model_name=model_name,
            target_chunk_tokens=target_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        for chunk_blocks in section_chunk_blocks:
            chunk_text = _render_chunk_text(chunk_blocks)
            chunk_records.append(
                ChunkRecord(
                    chunk_index=chunk_index,
                    page_start=min(block.page_no for block in chunk_blocks),
                    page_end=max(block.page_no for block in chunk_blocks),
                    section_type=_chunk_section_type(chunk_blocks),
                    section_title=_chunk_section_title(chunk_blocks),
                    heading_path=_chunk_heading_path(chunk_blocks),
                    chunk_text=chunk_text,
                    chunk_sha256=sha256_hexdigest(chunk_text),
                    token_count=token_count(chunk_text, model_name=model_name),
                )
            )
            chunk_index += 1

    return tuple(chunk_records)


def _group_major_sections(blocks: tuple[StructuredBlock, ...]) -> list[list[StructuredBlock]]:
    sections: list[list[StructuredBlock]] = []
    current: list[StructuredBlock] = []

    for block in blocks:
        if current and _starts_new_major_section(block, current):
            sections.append(current)
            current = []
        current.append(block)

    if current:
        sections.append(current)
    return sections


def _starts_new_major_section(block: StructuredBlock, current: list[StructuredBlock]) -> bool:
    if block.block_type != "heading":
        return False
    if not current:
        return False
    if all(existing.block_type == "heading" for existing in current):
        return False
    current_first = current[0]
    if current_first.section_type == "header":
        return True
    if block.heading_level is not None and block.heading_level <= 1:
        return True
    return block.section_type != current_first.section_type and block.section_type != "table_block"


def _expand_oversized_blocks(
    blocks: list[StructuredBlock],
    *,
    model_name: str,
    target_chunk_tokens: int,
    max_chunk_tokens: int,
) -> list[StructuredBlock]:
    expanded: list[StructuredBlock] = []
    for block in blocks:
        if block.token_count <= max_chunk_tokens:
            expanded.append(block)
            continue
        expanded.extend(
            _split_oversized_block(
                block,
                model_name=model_name,
                target_chunk_tokens=target_chunk_tokens,
                max_chunk_tokens=max_chunk_tokens,
            )
        )
    return expanded


def _split_oversized_block(
    block: StructuredBlock,
    *,
    model_name: str,
    target_chunk_tokens: int,
    max_chunk_tokens: int,
) -> list[StructuredBlock]:
    if block.block_type == "table_block":
        windows = split_token_windows(
            block.text,
            model_name=model_name,
            max_tokens=max_chunk_tokens,
            overlap_tokens=0,
        )
        return [
            replace(
                block,
                block_index=(block.block_index * 1000) + index,
                text=window,
                token_count=token_count(window, model_name=model_name),
            )
            for index, window in enumerate(windows)
        ]

    units = _split_block_units(block.text)
    if len(units) == 1:
        units = split_token_windows(
            block.text,
            model_name=model_name,
            max_tokens=max_chunk_tokens,
            overlap_tokens=0,
        )

    parts: list[str] = []
    current_units: list[str] = []
    for unit in units:
        candidate_units = current_units + [unit]
        candidate_text = " ".join(candidate_units).strip()
        candidate_tokens = token_count(candidate_text, model_name=model_name)
        if current_units and candidate_tokens > target_chunk_tokens:
            parts.append(" ".join(current_units).strip())
            current_units = [unit]
            continue
        if candidate_tokens > max_chunk_tokens:
            if current_units:
                parts.append(" ".join(current_units).strip())
                current_units = []
            parts.extend(
                split_token_windows(
                    unit,
                    model_name=model_name,
                    max_tokens=max_chunk_tokens,
                    overlap_tokens=0,
                )
            )
            continue
        current_units = candidate_units

    if current_units:
        parts.append(" ".join(current_units).strip())

    return [
        replace(
            block,
            block_index=(block.block_index * 1000) + index,
            text=part,
            token_count=token_count(part, model_name=model_name),
        )
        for index, part in enumerate(parts)
        if part
    ]


def _split_block_units(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences or [text.strip()]


def _chunk_section(
    blocks: list[StructuredBlock],
    *,
    model_name: str,
    target_chunk_tokens: int,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> list[list[StructuredBlock]]:
    if not blocks:
        return []

    full_section_tokens = token_count(_render_chunk_text(blocks), model_name=model_name)
    if blocks[0].section_type in _OPERATIVE_SECTION_TYPES and full_section_tokens <= max_chunk_tokens:
        return [blocks]

    chunks: list[list[StructuredBlock]] = []
    current: list[StructuredBlock] = []

    for block in blocks:
        if block.block_type == "table_block":
            if current:
                chunks.append(current)
                current = []
            chunks.append([block])
            continue

        if not current:
            current = [block]
            continue

        candidate = current + [block]
        candidate_tokens = token_count(_render_chunk_text(candidate), model_name=model_name)
        if candidate_tokens <= target_chunk_tokens:
            current = candidate
            continue
        if len(current) == 1 and candidate_tokens <= max_chunk_tokens:
            current = candidate
            continue

        chunks.append(current)
        current = _build_overlap(current, overlap_tokens=overlap_tokens)
        current = _trim_overlap_to_fit(
            current,
            incoming_block=block,
            model_name=model_name,
            max_chunk_tokens=max_chunk_tokens,
        )
        current.append(block)

    if current:
        chunks.append(current)

    normalized_chunks: list[list[StructuredBlock]] = []
    for chunk in chunks:
        chunk_text = _render_chunk_text(chunk)
        if token_count(chunk_text, model_name=model_name) <= max_chunk_tokens:
            normalized_chunks.append(chunk)
            continue

        windows = split_token_windows(
            chunk_text,
            model_name=model_name,
            max_tokens=max_chunk_tokens,
            overlap_tokens=0,
        )
        template = chunk[0]
        for index, window in enumerate(windows):
            normalized_chunks.append(
                [
                    replace(
                        template,
                        block_index=(template.block_index * 1000) + index,
                        text=window,
                        token_count=token_count(window, model_name=model_name),
                    )
                ]
            )

    return normalized_chunks


def _build_overlap(blocks: list[StructuredBlock], *, overlap_tokens: int) -> list[StructuredBlock]:
    if overlap_tokens <= 0:
        return []

    overlap: list[StructuredBlock] = []
    running_tokens = 0
    for block in reversed(blocks):
        overlap.insert(0, block)
        running_tokens += block.token_count
        if running_tokens >= overlap_tokens:
            break
    return overlap


def _trim_overlap_to_fit(
    overlap: list[StructuredBlock],
    *,
    incoming_block: StructuredBlock,
    model_name: str,
    max_chunk_tokens: int,
) -> list[StructuredBlock]:
    trimmed = list(overlap)
    while trimmed:
        candidate = trimmed + [incoming_block]
        if token_count(_render_chunk_text(candidate), model_name=model_name) <= max_chunk_tokens:
            break
        trimmed.pop(0)
    return trimmed


def _render_chunk_text(blocks: list[StructuredBlock]) -> str:
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()


def _chunk_section_type(blocks: list[StructuredBlock]) -> str:
    if len(blocks) == 1 and blocks[0].block_type == "table_block":
        return "table_block"
    return _preferred_metadata_block(blocks).section_type


def _chunk_section_title(blocks: list[StructuredBlock]) -> str | None:
    metadata_block = _preferred_metadata_block(blocks)
    return metadata_block.section_title


def _chunk_heading_path(blocks: list[StructuredBlock]) -> tuple[str, ...]:
    return _preferred_metadata_block(blocks).heading_path


def _preferred_metadata_block(blocks: list[StructuredBlock]) -> StructuredBlock:
    for block in reversed(blocks):
        if block.block_type == "heading" and block.section_type not in {"other", "table_block"}:
            return block
    for block in reversed(blocks):
        if block.block_type == "heading" and block.section_type != "table_block":
            return block
    for block in blocks:
        if block.section_type not in {"other", "table_block"}:
            return block
    for block in blocks:
        if block.section_type != "table_block":
            return block
    return blocks[0]
