"""Deterministic post-processing for Phase 2 chunk quality cleanup."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import ChunkRecord, SectionType
from ..utils.strings import collapse_inline_whitespace, sha256_hexdigest, uppercase_ratio
from .token_count import token_count

_SUBSTANTIVE_SECTION_TYPES = frozenset(
    {
        "background",
        "facts",
        "allegations",
        "show_cause_notice",
        "reply_or_submissions",
        "issues",
        "findings",
        "directions",
        "operative_order",
    }
)
_NON_SUBSTANTIVE_SECTION_TYPES = frozenset({"header", "other"})
_PROTECTED_MINIMUM_MERGE_SECTION_TYPES = frozenset(
    {"table_block", "annexure", "operative_order", "directions"}
)
_HEADING_ONLY_EXACT_MATCHES = frozenset(
    {
        "ORDER",
        "BACKGROUND",
        "FACTS",
        "FINDINGS",
        "DIRECTIONS",
        "SHOW CAUSE NOTICE",
        "ISSUE",
        "ISSUES",
        "ANNEXURE",
        "ANNEXURE A",
        "ANNEXURE B",
    }
)
_HEADING_ONLY_RE = re.compile(
    r"^(?:"
    r"ORDER|BACKGROUND|FACTS|FINDINGS|DIRECTIONS|SHOW CAUSE NOTICE|"
    r"ISSUE(?:S)?(?:\s+[A-Z0-9IVXLCM.-]+)?|"
    r"\d+(?:\.\d+)*[.)]?\s+[A-Z][A-Z0-9 ,()/:-]*|"
    r"[IVXLCM0-9]+[.)]?\s+[A-Z][A-Z0-9 ,()/:-]*"
    r")$",
    re.IGNORECASE,
)
_FOOTER_SIGNAL_RE = re.compile(
    r"\b(?:date|place)\s*:|"
    r"\bappellate authority\b|"
    r"\bsecurities and exchange board of india\b|"
    r"\bright to information act\b",
    re.IGNORECASE,
)
_KNOWN_FOOTER_PHRASES = (
    "APPELLATE AUTHORITY UNDER THE RTI ACT",
    "APPELLATE AUTHORITY",
    "SECURITIES AND EXCHANGE BOARD OF INDIA",
    "UNDER THE RIGHT TO INFORMATION ACT, 2005",
    "UNDER THE RTI ACT",
)


@dataclass(frozen=True)
class ChunkPostprocessSummary:
    """Stable stats for one document-level post-processing run."""

    initial_chunk_count: int
    final_chunk_count: int
    merges_applied: int
    suppressed_chunks: int


@dataclass(frozen=True)
class ChunkPostprocessResult:
    """Post-processed chunks plus stable effect counts."""

    chunks: tuple[ChunkRecord, ...]
    summary: ChunkPostprocessSummary


def postprocess_chunks(
    chunks: tuple[ChunkRecord, ...],
    *,
    page_count: int,
    model_name: str,
    max_chunk_tokens: int,
) -> ChunkPostprocessResult:
    """Apply a focused merge/suppression pass without changing core chunking."""

    if not chunks:
        return ChunkPostprocessResult(
            chunks=(),
            summary=ChunkPostprocessSummary(
                initial_chunk_count=0,
                final_chunk_count=0,
                merges_applied=0,
                suppressed_chunks=0,
            ),
        )

    working = list(chunks)
    merges_applied = 0
    suppressed_chunks = 0

    leading_merges = _merge_leading_caption_fragments(
        working,
        model_name=model_name,
    )
    merges_applied += leading_merges

    heading_merges = _merge_heading_only_chunks(
        working,
        model_name=model_name,
        max_chunk_tokens=max_chunk_tokens,
    )
    merges_applied += heading_merges

    footer_merges, footer_suppressed = _cleanup_trailing_footer_chunks(
        working,
        model_name=model_name,
        max_chunk_tokens=max_chunk_tokens,
    )
    merges_applied += footer_merges
    suppressed_chunks += footer_suppressed

    minimum_merges = _merge_small_cleanup_chunks(
        working,
        model_name=model_name,
        max_chunk_tokens=max_chunk_tokens,
        token_threshold=80,
        only_non_substantive=False,
    )
    merges_applied += minimum_merges

    if page_count <= 3 and len(working) > 6:
        extra_merges = _merge_small_cleanup_chunks(
            working,
            model_name=model_name,
            max_chunk_tokens=max_chunk_tokens,
            token_threshold=120,
            only_non_substantive=True,
        )
        merges_applied += extra_merges

    normalized_chunks = _reindex_chunks(
        working,
        model_name=model_name,
    )
    return ChunkPostprocessResult(
        chunks=tuple(normalized_chunks),
        summary=ChunkPostprocessSummary(
            initial_chunk_count=len(chunks),
            final_chunk_count=len(normalized_chunks),
            merges_applied=merges_applied,
            suppressed_chunks=suppressed_chunks,
        ),
    )


def _merge_leading_caption_fragments(
    chunks: list[ChunkRecord],
    *,
    model_name: str,
) -> int:
    first_substantive_index = next(
        (
            index
            for index, chunk in enumerate(chunks)
            if _is_substantive_section(chunk.section_type)
        ),
        len(chunks),
    )
    candidate_count = 0

    for index in range(first_substantive_index):
        chunk = chunks[index]
        if not _is_leading_caption_candidate(chunk):
            break
        if index > 0 and not _is_adjacent_page_span(chunks[index - 1], chunk):
            break
        candidate_count += 1

    if candidate_count <= 1:
        return 0

    merged_chunk = chunks[0]
    for chunk in chunks[1:candidate_count]:
        merged_chunk = _merge_pair(
            merged_chunk,
            chunk,
            model_name=model_name,
            metadata_strategy="leading_caption",
        )

    chunks[:candidate_count] = [merged_chunk]
    return candidate_count - 1


def _merge_heading_only_chunks(
    chunks: list[ChunkRecord],
    *,
    model_name: str,
    max_chunk_tokens: int,
) -> int:
    merges_applied = 0
    index = 0
    while index < len(chunks) - 1:
        current = chunks[index]
        following = chunks[index + 1]
        if not _is_heading_only_chunk(current):
            index += 1
            continue
        if not _is_heading_follow_merge_candidate(current, following):
            index += 1
            continue

        combined_text = _combine_chunk_text(current.chunk_text, following.chunk_text)
        combined_tokens = token_count(combined_text, model_name=model_name)
        if not _is_merge_size_acceptable(
            combined_tokens=combined_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overflow_tolerance=min(current.token_count, 40),
        ):
            index += 1
            continue

        chunks[index + 1] = _merge_pair(
            current,
            following,
            model_name=model_name,
            metadata_strategy="right",
        )
        del chunks[index]
        merges_applied += 1
    return merges_applied


def _cleanup_trailing_footer_chunks(
    chunks: list[ChunkRecord],
    *,
    model_name: str,
    max_chunk_tokens: int,
) -> tuple[int, int]:
    merges_applied = 0
    suppressed_chunks = 0
    index = len(chunks) - 1

    while index >= 0 and chunks:
        chunk = chunks[index]
        if not _is_trailing_footer_candidate(chunk):
            break

        if _is_pure_footer_boilerplate(chunk):
            del chunks[index]
            suppressed_chunks += 1
            index = min(index - 1, len(chunks) - 1)
            continue

        previous_index = index - 1
        if previous_index < 0:
            break

        previous = chunks[previous_index]
        combined_text = _combine_chunk_text(previous.chunk_text, chunk.chunk_text)
        combined_tokens = token_count(combined_text, model_name=model_name)
        if not _is_merge_size_acceptable(
            combined_tokens=combined_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overflow_tolerance=min(chunk.token_count, 40),
        ):
            break

        chunks[previous_index] = _merge_pair(
            previous,
            chunk,
            model_name=model_name,
            metadata_strategy="left",
        )
        del chunks[index]
        merges_applied += 1
        index = previous_index

    return merges_applied, suppressed_chunks


def _merge_small_cleanup_chunks(
    chunks: list[ChunkRecord],
    *,
    model_name: str,
    max_chunk_tokens: int,
    token_threshold: int,
    only_non_substantive: bool,
) -> int:
    merges_applied = 0
    index = 0

    while index < len(chunks):
        candidate = chunks[index]
        if (
            index == 0
            and _is_non_substantive_section(candidate.section_type)
            and len(chunks) > 1
            and _is_substantive_section(chunks[1].section_type)
        ):
            index += 1
            continue
        if not _is_small_cleanup_candidate(
            candidate,
            token_threshold=token_threshold,
            only_non_substantive=only_non_substantive,
        ):
            index += 1
            continue

        direction = _pick_cleanup_merge_direction(
            chunks,
            candidate_index=index,
            model_name=model_name,
            max_chunk_tokens=max_chunk_tokens,
        )
        if direction is None:
            index += 1
            continue

        if direction == "prev":
            previous = chunks[index - 1]
            chunks[index - 1] = _merge_pair(
                previous,
                candidate,
                model_name=model_name,
                metadata_strategy="left",
            )
            del chunks[index]
            merges_applied += 1
            index = max(index - 1, 0)
            continue

        following = chunks[index + 1]
        chunks[index + 1] = _merge_pair(
            candidate,
            following,
            model_name=model_name,
            metadata_strategy="right",
        )
        del chunks[index]
        merges_applied += 1

    return merges_applied


def _pick_cleanup_merge_direction(
    chunks: list[ChunkRecord],
    *,
    candidate_index: int,
    model_name: str,
    max_chunk_tokens: int,
) -> str | None:
    candidate = chunks[candidate_index]
    options: list[tuple[int, str]] = []

    if candidate_index > 0:
        previous = chunks[candidate_index - 1]
        previous_score = _score_cleanup_merge(
            anchor=previous,
            candidate=candidate,
            model_name=model_name,
            max_chunk_tokens=max_chunk_tokens,
            direction="prev",
        )
        if previous_score is not None:
            options.append((previous_score, "prev"))

    if candidate_index + 1 < len(chunks):
        following = chunks[candidate_index + 1]
        following_score = _score_cleanup_merge(
            anchor=following,
            candidate=candidate,
            model_name=model_name,
            max_chunk_tokens=max_chunk_tokens,
            direction="next",
        )
        if following_score is not None:
            options.append((following_score, "next"))

    if not options:
        return None

    options.sort(key=lambda item: (item[0], item[1] == "prev"), reverse=True)
    return options[0][1]


def _score_cleanup_merge(
    *,
    anchor: ChunkRecord,
    candidate: ChunkRecord,
    model_name: str,
    max_chunk_tokens: int,
    direction: str,
) -> int | None:
    if anchor.section_type == "table_block":
        return None
    if _page_gap_too_large(left=anchor if direction == "prev" else candidate, right=candidate if direction == "prev" else anchor):
        return None

    combined_text = (
        _combine_chunk_text(anchor.chunk_text, candidate.chunk_text)
        if direction == "prev"
        else _combine_chunk_text(candidate.chunk_text, anchor.chunk_text)
    )
    combined_tokens = token_count(combined_text, model_name=model_name)
    if combined_tokens > max_chunk_tokens:
        return None

    score = 0
    if anchor.section_type == candidate.section_type:
        score += 5
    if _is_non_substantive_section(anchor.section_type) and _is_non_substantive_section(
        candidate.section_type
    ):
        score += 4
    if direction == "prev" and _is_non_substantive_section(candidate.section_type):
        score += 2
    if direction == "next" and _is_non_substantive_section(candidate.section_type):
        score += 3
    if direction == "next" and _is_heading_only_chunk(candidate):
        score += 4
    if direction == "next" and _is_substantive_section(anchor.section_type):
        score += 2
    if direction == "prev" and _is_substantive_section(anchor.section_type):
        score += 1
    return score


def _merge_pair(
    left: ChunkRecord,
    right: ChunkRecord,
    *,
    model_name: str,
    metadata_strategy: str,
) -> ChunkRecord:
    chunk_text = _combine_chunk_text(left.chunk_text, right.chunk_text)
    section_type, section_title, heading_path = _resolve_merge_metadata(
        left,
        right,
        metadata_strategy=metadata_strategy,
    )
    return ChunkRecord(
        chunk_index=min(left.chunk_index, right.chunk_index),
        page_start=min(left.page_start, right.page_start),
        page_end=max(left.page_end, right.page_end),
        section_type=section_type,
        section_title=section_title,
        heading_path=heading_path,
        chunk_text=chunk_text,
        chunk_sha256=sha256_hexdigest(chunk_text),
        token_count=token_count(chunk_text, model_name=model_name),
    )


def _resolve_merge_metadata(
    left: ChunkRecord,
    right: ChunkRecord,
    *,
    metadata_strategy: str,
) -> tuple[SectionType, str | None, tuple[str, ...]]:
    if metadata_strategy == "right":
        return right.section_type, right.section_title, right.heading_path
    if metadata_strategy == "leading_caption":
        section_title = left.section_title or right.section_title
        heading_path = left.heading_path or right.heading_path
        return "header", section_title, heading_path
    return left.section_type, left.section_title, left.heading_path


def _reindex_chunks(
    chunks: list[ChunkRecord],
    *,
    model_name: str,
) -> list[ChunkRecord]:
    normalized_chunks: list[ChunkRecord] = []
    for index, chunk in enumerate(chunks):
        chunk_text = chunk.chunk_text.strip()
        normalized_chunks.append(
            ChunkRecord(
                chunk_index=index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_type=chunk.section_type,
                section_title=chunk.section_title,
                heading_path=chunk.heading_path,
                chunk_text=chunk_text,
                chunk_sha256=sha256_hexdigest(chunk_text),
                token_count=token_count(chunk_text, model_name=model_name),
            )
        )
    return normalized_chunks


def _is_leading_caption_candidate(chunk: ChunkRecord) -> bool:
    return (
        chunk.section_type in _NON_SUBSTANTIVE_SECTION_TYPES
        and chunk.token_count < 120
        and chunk.page_start <= 2
        and chunk.page_end <= 2
    )


def _is_heading_follow_merge_candidate(
    current: ChunkRecord,
    following: ChunkRecord,
) -> bool:
    if following.section_type in {"table_block", "annexure"}:
        return False
    if _page_gap_too_large(left=current, right=following):
        return False
    if not _is_substantive_section(following.section_type):
        return False
    return current.section_type == following.section_type or current.section_type in {
        "header",
        "other",
    }


def _is_small_cleanup_candidate(
    chunk: ChunkRecord,
    *,
    token_threshold: int,
    only_non_substantive: bool,
) -> bool:
    if chunk.token_count >= token_threshold:
        return False
    if chunk.section_type in _PROTECTED_MINIMUM_MERGE_SECTION_TYPES:
        return False
    if only_non_substantive and not _is_non_substantive_section(chunk.section_type):
        return False
    return True


def _is_heading_only_chunk(chunk: ChunkRecord) -> bool:
    if chunk.token_count >= 40:
        return False
    if chunk.section_type in {"table_block", "annexure"}:
        return False

    normalized = _normalize_inline_text(chunk.chunk_text)
    if not normalized or normalized.endswith("."):
        return False

    word_count = len(normalized.split())
    if word_count == 0 or word_count > 14:
        return False
    if normalized.upper() in _HEADING_ONLY_EXACT_MATCHES:
        return True
    if _HEADING_ONLY_RE.match(normalized):
        return True
    return word_count <= 8 and uppercase_ratio(normalized) >= 0.80


def _is_trailing_footer_candidate(chunk: ChunkRecord) -> bool:
    if chunk.section_type not in _NON_SUBSTANTIVE_SECTION_TYPES:
        return False
    if chunk.token_count >= 80:
        return False
    return _looks_like_footer_text(chunk.chunk_text)


def _looks_like_footer_text(text: str) -> bool:
    normalized = _normalize_inline_text(text)
    if not normalized:
        return False
    if _FOOTER_SIGNAL_RE.search(normalized):
        return True
    return _is_pure_footer_boilerplate_text(normalized)


def _is_pure_footer_boilerplate(chunk: ChunkRecord) -> bool:
    return _is_pure_footer_boilerplate_text(_normalize_inline_text(chunk.chunk_text))


def _is_pure_footer_boilerplate_text(text: str) -> bool:
    normalized = text.upper().strip()
    if not normalized:
        return False
    if "DATE:" in normalized or "PLACE:" in normalized:
        return False

    stripped = normalized
    for phrase in _KNOWN_FOOTER_PHRASES:
        stripped = stripped.replace(phrase, " ")
    stripped = re.sub(r"[^A-Z0-9]+", " ", stripped)
    stripped = collapse_inline_whitespace(stripped)
    return stripped == ""


def _combine_chunk_text(left_text: str, right_text: str) -> str:
    left_clean = left_text.strip()
    right_clean = right_text.strip()
    if left_clean and right_clean:
        return f"{left_clean}\n\n{right_clean}"
    return left_clean or right_clean


def _normalize_inline_text(text: str) -> str:
    return collapse_inline_whitespace(text.replace("\n", " ")).strip()


def _is_merge_size_acceptable(
    *,
    combined_tokens: int,
    max_chunk_tokens: int,
    overflow_tolerance: int,
) -> bool:
    return combined_tokens <= max_chunk_tokens + overflow_tolerance


def _is_substantive_section(section_type: SectionType) -> bool:
    return section_type in _SUBSTANTIVE_SECTION_TYPES


def _is_non_substantive_section(section_type: SectionType) -> bool:
    return section_type in _NON_SUBSTANTIVE_SECTION_TYPES


def _is_adjacent_page_span(left: ChunkRecord, right: ChunkRecord) -> bool:
    return not _page_gap_too_large(left=left, right=right) and right.page_end <= 2


def _page_gap_too_large(left: ChunkRecord, right: ChunkRecord) -> bool:
    return right.page_start > left.page_end + 1
