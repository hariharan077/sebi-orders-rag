"""Pure chunk QA logic and repository-backed inspection helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import mean

from ..config import SebiOrdersRagSettings
from ..constants import DEFAULT_MAX_CHUNK_TOKENS
from ..repositories.qa import (
    BucketAggregateRow,
    ChunkQaRepository,
    ChunkRow,
    CorpusAggregateRow,
    ProcessedDocumentVersionRow,
)
from ..utils.strings import uppercase_ratio

TINY_CHUNK_FLAG = "tiny_chunk"
OVERSIZED_CHUNK_FLAG = "oversized_chunk"
HEADING_ONLY_CHUNK_FLAG = "heading_only_chunk"
SHORT_DOC_OVERFRAGMENTED_FLAG = "short_doc_overfragmented"
CHUNK_DENSITY_HIGH_FLAG = "chunk_density_high"
DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG = "duplicate_chunk_text_in_doc"
SUSPICIOUS_SECTION_JUMP_FLAG = "suspicious_section_jump"

CHUNK_FLAG_ORDER = (
    OVERSIZED_CHUNK_FLAG,
    DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG,
    HEADING_ONLY_CHUNK_FLAG,
    TINY_CHUNK_FLAG,
    SUSPICIOUS_SECTION_JUMP_FLAG,
)
DOCUMENT_FLAG_ORDER = (
    SHORT_DOC_OVERFRAGMENTED_FLAG,
    CHUNK_DENSITY_HIGH_FLAG,
    DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG,
    SUSPICIOUS_SECTION_JUMP_FLAG,
)
ALL_FLAG_ORDER = DOCUMENT_FLAG_ORDER + CHUNK_FLAG_ORDER

SEVERITY_WEIGHTS = {
    OVERSIZED_CHUNK_FLAG: 4,
    DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG: 4,
    SHORT_DOC_OVERFRAGMENTED_FLAG: 3,
    CHUNK_DENSITY_HIGH_FLAG: 2,
    HEADING_ONLY_CHUNK_FLAG: 1,
    TINY_CHUNK_FLAG: 1,
    SUSPICIOUS_SECTION_JUMP_FLAG: 1,
}
TINY_CHUNK_EXEMPT_SECTION_TYPES = frozenset(
    {"header", "table_block", "annexure", "operative_order", "directions"}
)
HEADING_ONLY_EXACT_MATCHES = frozenset(
    {
        "ORDER",
        "BACKGROUND",
        "FACTS",
        "FINDINGS",
        "DIRECTIONS",
        "ANNEXURE",
        "ANNEXURE A",
        "ANNEXURE B",
        "ISSUE",
        "ISSUES",
    }
)
HEADING_ONLY_KEYWORD_RE = re.compile(r"^(ORDER|BACKGROUND|ISSUE(?:S)?(?:\s+[A-Z0-9IVXLC-]+)?)$")


@dataclass(frozen=True)
class ChunkInspectionResult:
    """Terminal-friendly inspection view for one chunk."""

    chunk_index: int
    section_type: str
    section_title: str | None
    heading_path: tuple[str, ...]
    page_start: int
    page_end: int
    token_count: int
    flags: tuple[str, ...]
    first_text_preview: str
    last_text_preview: str


@dataclass(frozen=True)
class DocumentAuditResult:
    """Audit result for one processed document version."""

    document_version_id: int
    record_key: str
    title: str
    bucket_name: str
    order_date: date | None
    page_count: int
    chunk_count: int
    average_tokens_per_chunk: float
    severity_score: int
    document_flags: tuple[str, ...]
    document_flag_counts: Mapping[str, int]
    chunk_flag_counts: Mapping[str, int]
    chunks: tuple[ChunkInspectionResult, ...]

    @property
    def is_flagged(self) -> bool:
        """Return whether the document has any QA findings."""

        return self.severity_score > 0 or bool(self.document_flags)


@dataclass(frozen=True)
class CorpusAuditResult:
    """Audit result for a document scope."""

    summary: CorpusAggregateRow
    per_bucket: tuple[BucketAggregateRow, ...]
    documents: tuple[DocumentAuditResult, ...]

    @property
    def flagged_documents(self) -> tuple[DocumentAuditResult, ...]:
        """Return documents with non-zero severity ordered by priority."""

        flagged = [document for document in self.documents if document.is_flagged]
        flagged.sort(key=lambda document: (-document.severity_score, document.document_version_id))
        return tuple(flagged)


class ChunkAuditAnalyzer:
    """Pure deterministic QA rules for legal chunk inspection."""

    def __init__(
        self,
        *,
        oversized_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        tiny_chunk_tokens: int = 80,
        high_density_token_threshold: float = 120.0,
    ) -> None:
        self._oversized_chunk_tokens = oversized_chunk_tokens
        self._tiny_chunk_tokens = tiny_chunk_tokens
        self._high_density_token_threshold = high_density_token_threshold

    def audit_document(
        self,
        *,
        document: ProcessedDocumentVersionRow,
        chunks: Sequence[ChunkRow],
    ) -> DocumentAuditResult:
        """Audit one processed document version in memory."""

        ordered_chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_index))
        chunk_flags: dict[int, set[str]] = {
            chunk.chunk_index: set() for chunk in ordered_chunks
        }
        document_flag_counts: Counter[str] = Counter()

        for chunk in ordered_chunks:
            if self._is_tiny_chunk(chunk):
                chunk_flags[chunk.chunk_index].add(TINY_CHUNK_FLAG)
            if chunk.token_count > self._oversized_chunk_tokens:
                chunk_flags[chunk.chunk_index].add(OVERSIZED_CHUNK_FLAG)
            if is_heading_only_chunk_text(chunk.chunk_text, token_count=chunk.token_count):
                chunk_flags[chunk.chunk_index].add(HEADING_ONLY_CHUNK_FLAG)

        duplicate_event_count = self._flag_duplicate_chunks(
            chunks=ordered_chunks,
            chunk_flags=chunk_flags,
        )
        if duplicate_event_count:
            document_flag_counts[DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG] = duplicate_event_count

        suspicious_jump_count = self._flag_suspicious_section_jumps(
            chunks=ordered_chunks,
            chunk_flags=chunk_flags,
        )
        if suspicious_jump_count:
            document_flag_counts[SUSPICIOUS_SECTION_JUMP_FLAG] = suspicious_jump_count

        page_count = document.page_count or max(
            (chunk.page_end for chunk in ordered_chunks),
            default=0,
        )
        chunk_count = len(ordered_chunks)
        average_tokens_per_chunk = (
            mean(chunk.token_count for chunk in ordered_chunks)
            if ordered_chunks
            else 0.0
        )

        if page_count <= 3 and chunk_count > 6:
            document_flag_counts[SHORT_DOC_OVERFRAGMENTED_FLAG] = 1
        if average_tokens_per_chunk < self._high_density_token_threshold and chunk_count >= 6:
            document_flag_counts[CHUNK_DENSITY_HIGH_FLAG] = 1

        chunk_flag_counts = Counter[str]()
        for flags in chunk_flags.values():
            chunk_flag_counts.update(flags)

        severity_score = calculate_severity_score(
            chunk_flag_counts=chunk_flag_counts,
            document_flag_counts=document_flag_counts,
        )
        inspection_chunks = tuple(
            ChunkInspectionResult(
                chunk_index=chunk.chunk_index,
                section_type=chunk.section_type,
                section_title=chunk.section_title,
                heading_path=chunk.heading_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                token_count=chunk.token_count,
                flags=_ordered_flags(chunk_flags[chunk.chunk_index], CHUNK_FLAG_ORDER),
                first_text_preview=build_text_preview(chunk.chunk_text, from_start=True),
                last_text_preview=build_text_preview(chunk.chunk_text, from_start=False),
            )
            for chunk in ordered_chunks
        )

        return DocumentAuditResult(
            document_version_id=document.document_version_id,
            record_key=document.record_key,
            title=document.title,
            bucket_name=document.bucket_name,
            order_date=document.order_date,
            page_count=page_count,
            chunk_count=chunk_count,
            average_tokens_per_chunk=average_tokens_per_chunk,
            severity_score=severity_score,
            document_flags=_ordered_flags(document_flag_counts.keys(), DOCUMENT_FLAG_ORDER),
            document_flag_counts=_ordered_count_mapping(document_flag_counts, DOCUMENT_FLAG_ORDER),
            chunk_flag_counts=_ordered_count_mapping(chunk_flag_counts, CHUNK_FLAG_ORDER),
            chunks=inspection_chunks,
        )

    def _is_tiny_chunk(self, chunk: ChunkRow) -> bool:
        return (
            chunk.token_count < self._tiny_chunk_tokens
            and chunk.section_type not in TINY_CHUNK_EXEMPT_SECTION_TYPES
        )

    @staticmethod
    def _flag_duplicate_chunks(
        *,
        chunks: Sequence[ChunkRow],
        chunk_flags: Mapping[int, set[str]],
    ) -> int:
        duplicates: dict[str, list[ChunkRow]] = defaultdict(list)
        for chunk in chunks:
            duplicates[chunk.chunk_sha256].append(chunk)

        duplicate_event_count = 0
        for duplicate_group in duplicates.values():
            if len(duplicate_group) <= 1:
                continue
            duplicate_event_count += len(duplicate_group) - 1
            for chunk in duplicate_group:
                chunk_flags[chunk.chunk_index].add(DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG)
        return duplicate_event_count

    @staticmethod
    def _flag_suspicious_section_jumps(
        *,
        chunks: Sequence[ChunkRow],
        chunk_flags: Mapping[int, set[str]],
    ) -> int:
        suspicious_count = 0
        for current_index in range(1, len(chunks) - 1):
            previous_chunk = chunks[current_index - 1]
            current_chunk = chunks[current_index]
            next_chunk = chunks[current_index + 1]

            if previous_chunk.section_type != next_chunk.section_type:
                continue
            if previous_chunk.section_type in {"other", "table_block"}:
                continue
            if current_chunk.section_type == previous_chunk.section_type:
                continue

            page_span = max(
                previous_chunk.page_end,
                current_chunk.page_end,
                next_chunk.page_end,
            ) - min(
                previous_chunk.page_start,
                current_chunk.page_start,
                next_chunk.page_start,
            )
            if page_span > 1:
                continue
            if (
                current_chunk.section_type not in {"other", "header"}
                and current_chunk.token_count >= 120
            ):
                continue

            suspicious_count += 1
            chunk_flags[current_chunk.chunk_index].add(SUSPICIOUS_SECTION_JUMP_FLAG)
        return suspicious_count


class ChunkAuditService:
    """Repository-backed chunk QA inspection service."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        connection: object,
        analyzer: ChunkAuditAnalyzer | None = None,
    ) -> None:
        self._settings = settings
        self._repository = ChunkQaRepository(connection)
        self._analyzer = analyzer or ChunkAuditAnalyzer(
            oversized_chunk_tokens=settings.max_chunk_tokens
        )

    def inspect_document(
        self,
        *,
        document_version_id: int | None = None,
        record_key: str | None = None,
    ) -> DocumentAuditResult:
        """Inspect a single document selected by version id or record key."""

        if document_version_id is None and record_key is None:
            raise ValueError("Expected document_version_id or record_key for inspection")

        if document_version_id is not None:
            matches = self._repository.list_processed_document_versions(
                document_version_ids=[document_version_id]
            )
            if not matches:
                raise LookupError(
                    f"No processed document_version found for id {document_version_id}"
                )
            return self._audit_processed_document(matches[0])

        assert record_key is not None
        matches = self._repository.list_processed_document_versions(record_keys=[record_key])
        if not matches:
            raise LookupError(f"No processed document_version found for record_key {record_key!r}")
        return self._audit_processed_document(matches[-1])

    def audit_scope(
        self,
        *,
        document_version_ids: Sequence[int] | None = None,
        record_keys: Sequence[str] | None = None,
        bucket_name: str | None = None,
        limit: int | None = None,
        sample_per_bucket: int | None = None,
    ) -> CorpusAuditResult:
        """Audit a processed document scope selected by explicit ids or sampling."""

        selected_document_ids = tuple(
            _dedupe_preserving_order(
                self._repository.sample_document_version_ids(
                    document_version_ids=document_version_ids,
                    record_keys=record_keys,
                    bucket_name=bucket_name,
                    limit=limit,
                    sample_per_bucket=sample_per_bucket,
                )
            )
        )
        if not selected_document_ids:
            return CorpusAuditResult(
                summary=CorpusAggregateRow(
                    processed_document_versions=0,
                    total_chunks=0,
                    avg_chunks_per_document=None,
                    median_chunks_per_document=None,
                    avg_tokens_per_chunk=None,
                    median_tokens_per_chunk=None,
                    min_tokens_per_chunk=None,
                    max_tokens_per_chunk=None,
                ),
                per_bucket=(),
                documents=(),
            )

        processed_documents = self._repository.list_processed_document_versions(
            document_version_ids=selected_document_ids
        )
        documents_by_id = {
            document.document_version_id: document for document in processed_documents
        }
        ordered_documents = [
            documents_by_id[document_version_id]
            for document_version_id in selected_document_ids
            if document_version_id in documents_by_id
        ]
        document_results = tuple(
            self._audit_processed_document(document) for document in ordered_documents
        )

        return CorpusAuditResult(
            summary=self._repository.get_corpus_aggregates(
                document_version_ids=selected_document_ids
            ),
            per_bucket=tuple(
                self._repository.get_per_bucket_aggregates(
                    document_version_ids=selected_document_ids
                )
            ),
            documents=document_results,
        )

    def _audit_processed_document(
        self,
        document: ProcessedDocumentVersionRow,
    ) -> DocumentAuditResult:
        chunks = self._repository.list_chunks_for_document_version(
            document_version_id=document.document_version_id
        )
        return self._analyzer.audit_document(document=document, chunks=chunks)


def is_heading_only_chunk_text(text: str, *, token_count: int) -> bool:
    """Return whether a short chunk appears to be mostly a heading/title."""

    normalized_text = text.strip()
    if not normalized_text:
        return False
    if token_count > 24:
        return False

    lines = [line.strip(" :-\t") for line in normalized_text.splitlines() if line.strip()]
    if not lines:
        return False

    joined_lines = " ".join(lines).strip()
    if len(joined_lines) > 160:
        return False

    uppercase_joined = joined_lines.upper()
    if uppercase_joined in HEADING_ONLY_EXACT_MATCHES:
        return True
    if HEADING_ONLY_KEYWORD_RE.fullmatch(uppercase_joined):
        return True
    if re.fullmatch(r"ISSUE(?:S)?\s+[A-Z0-9IVXLC-]+", uppercase_joined):
        return True
    if re.search(r"[.!?;]", joined_lines):
        return False

    if len(lines) <= 2 and uppercase_ratio(joined_lines) >= 0.75:
        return True

    if len(lines) <= 3 and all(len(line) <= 80 for line in lines):
        uppercase_lines = sum(1 for line in lines if uppercase_ratio(line) >= 0.75)
        if uppercase_lines == len(lines):
            return True

    return False


def build_text_preview(text: str, *, from_start: bool, max_chars: int = 200) -> str:
    """Build a compact one-line preview from the start or end of chunk text."""

    snippet = text[:max_chars] if from_start else text[-max_chars:]
    return " ".join(snippet.split())


def calculate_severity_score(
    *,
    chunk_flag_counts: Mapping[str, int],
    document_flag_counts: Mapping[str, int],
) -> int:
    """Calculate the deterministic severity score for one document."""

    return (
        chunk_flag_counts.get(OVERSIZED_CHUNK_FLAG, 0) * SEVERITY_WEIGHTS[OVERSIZED_CHUNK_FLAG]
        + document_flag_counts.get(DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG, 0)
        * SEVERITY_WEIGHTS[DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG]
        + document_flag_counts.get(SHORT_DOC_OVERFRAGMENTED_FLAG, 0)
        * SEVERITY_WEIGHTS[SHORT_DOC_OVERFRAGMENTED_FLAG]
        + document_flag_counts.get(CHUNK_DENSITY_HIGH_FLAG, 0)
        * SEVERITY_WEIGHTS[CHUNK_DENSITY_HIGH_FLAG]
        + chunk_flag_counts.get(HEADING_ONLY_CHUNK_FLAG, 0)
        * SEVERITY_WEIGHTS[HEADING_ONLY_CHUNK_FLAG]
        + chunk_flag_counts.get(TINY_CHUNK_FLAG, 0) * SEVERITY_WEIGHTS[TINY_CHUNK_FLAG]
        + document_flag_counts.get(SUSPICIOUS_SECTION_JUMP_FLAG, 0)
        * SEVERITY_WEIGHTS[SUSPICIOUS_SECTION_JUMP_FLAG]
    )


def _ordered_flags(flags: Sequence[str] | set[str], order: Sequence[str]) -> tuple[str, ...]:
    known_flags = [flag for flag in order if flag in flags]
    extra_flags = sorted(flag for flag in flags if flag not in order)
    return tuple(known_flags + extra_flags)


def _ordered_count_mapping(
    counts: Mapping[str, int],
    order: Sequence[str],
) -> Mapping[str, int]:
    ordered_keys = [flag for flag in order if counts.get(flag, 0) > 0]
    ordered_keys.extend(
        sorted(flag for flag, count in counts.items() if count > 0 and flag not in order)
    )
    return {flag: int(counts[flag]) for flag in ordered_keys}


def _dedupe_preserving_order(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
