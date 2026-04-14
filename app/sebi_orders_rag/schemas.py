"""Typed domain schemas for SEBI Orders RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from .control.models import StrictMatterLock


@dataclass(frozen=True)
class ManifestRow:
    """Normalized manifest row resolved against a local corpus folder."""

    record_key: str
    bucket_name: str
    order_date: date | None
    title: str
    external_record_id: str | None
    detail_url: str | None
    pdf_url: str
    local_filename: str
    manifest_status: str
    error: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    manifest_path: Path
    local_path: Path
    row_number: int


@dataclass(frozen=True)
class LoadedManifest:
    """A parsed manifest file and its valid rows."""

    path: Path
    bucket_name: str
    rows: tuple[ManifestRow, ...]
    invalid_rows: int = 0


@dataclass(frozen=True)
class FileFingerprint:
    """Fingerprint metadata for a local file."""

    file_size_bytes: int
    file_sha256: str


@dataclass(frozen=True)
class LocalFileSnapshot:
    """Local file availability and fingerprint details."""

    path: Path
    exists: bool
    fingerprint: FileFingerprint | None = None


@dataclass(frozen=True)
class SourceDocumentRecord:
    """Row materialized from the source_documents table."""

    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    current_version_id: int | None
    is_active: bool


@dataclass(frozen=True)
class DocumentVersionRecord:
    """Row materialized from the document_versions table."""

    document_version_id: int
    document_id: int
    order_date: date | None
    title: str
    detail_url: str | None
    pdf_url: str
    local_filename: str
    local_path: str
    file_size_bytes: int
    file_sha256: str
    manifest_status: str
    parser_name: str
    parser_version: str
    extraction_status: str
    ocr_used: bool
    page_count: int | None
    extracted_char_count: int | None
    ingest_status: str
    ingest_error: str | None
    ingested_at: datetime | None
    created_at: datetime
    chunking_version: str | None = None
    chunk_count: int | None = None
    embedding_status: str | None = None
    embedding_error: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedded_at: datetime | None = None


@dataclass(frozen=True)
class PlannedDocumentVersionCreate:
    """Planned document_versions insert for a new file hash."""

    order_date: date | None
    title: str
    detail_url: str | None
    pdf_url: str
    local_filename: str
    local_path: str
    file_size_bytes: int
    file_sha256: str
    manifest_status: str
    parser_name: str
    parser_version: str
    extraction_status: str
    ingest_status: str


PlannerAction = Literal[
    "skip_missing_file",
    "create_document_and_version",
    "create_version",
    "reuse_version",
]


@dataclass(frozen=True)
class PlannerDecision:
    """Pure planning output for a single manifest row."""

    action: PlannerAction
    create_document: bool
    create_version: bool
    reuse_existing_version: bool
    current_version_id: int | None
    version_to_create: PlannedDocumentVersionCreate | None


@dataclass
class Phase1Summary:
    """Aggregated Phase 1 execution summary."""

    manifests_found: int = 0
    rows_processed: int = 0
    pdfs_present: int = 0
    pdfs_missing: int = 0
    documents_inserted: int = 0
    document_versions_inserted: int = 0
    existing_versions_reused: int = 0
    rows_skipped_due_to_missing_files: int = 0
    invalid_rows: int = 0

    def as_lines(self) -> list[str]:
        """Render summary counts in a stable CLI-friendly order."""

        lines = [
            f"manifests found: {self.manifests_found}",
            f"rows processed: {self.rows_processed}",
            f"PDFs present: {self.pdfs_present}",
            f"PDFs missing: {self.pdfs_missing}",
            f"documents inserted: {self.documents_inserted}",
            f"document versions inserted: {self.document_versions_inserted}",
            f"existing versions reused: {self.existing_versions_reused}",
            f"rows skipped due to missing files: {self.rows_skipped_due_to_missing_files}",
        ]
        if self.invalid_rows:
            lines.append(f"invalid rows skipped: {self.invalid_rows}")
        return lines


SectionType = Literal[
    "header",
    "background",
    "facts",
    "allegations",
    "show_cause_notice",
    "reply_or_submissions",
    "issues",
    "findings",
    "directions",
    "operative_order",
    "annexure",
    "table_block",
    "other",
]

BlockType = Literal["heading", "paragraph", "table_block"]


@dataclass(frozen=True)
class PendingDocumentVersion:
    """Candidate document version selected for Phase 2 processing."""

    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    order_date: date | None
    title: str
    detail_url: str | None
    pdf_url: str
    local_filename: str
    local_path: str
    file_size_bytes: int
    file_sha256: str
    manifest_status: str
    parser_name: str
    parser_version: str
    extraction_status: str
    ocr_used: bool
    page_count: int | None
    extracted_char_count: int | None
    ingest_status: str
    ingest_error: str | None
    ingested_at: datetime | None
    created_at: datetime
    chunking_version: str | None
    chunk_count: int | None
    embedding_status: str | None = None
    embedding_error: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedded_at: datetime | None = None


@dataclass(frozen=True)
class ExtractedPage:
    """Normalized page-level extraction output before structure parsing."""

    page_no: int
    extracted_text: str
    ocr_text: str | None
    final_text: str
    char_count: int
    token_count: int
    low_text: bool
    page_sha256: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Page-wise extraction result for a single PDF document."""

    pages: tuple[ExtractedPage, ...]
    ocr_used: bool

    @property
    def page_count(self) -> int:
        """Return the number of extracted pages."""

        return len(self.pages)

    @property
    def extracted_char_count(self) -> int:
        """Return the sum of final extracted characters across all pages."""

        return sum(page.char_count for page in self.pages)


@dataclass(frozen=True)
class HeadingMatch:
    """Heading detection result for a single line of text."""

    title: str
    section_type: SectionType
    level: int


@dataclass(frozen=True)
class StructuredBlock:
    """Logical document block derived from ordered extracted pages."""

    block_index: int
    page_no: int
    block_type: BlockType
    text: str
    token_count: int
    section_type: SectionType
    section_title: str | None
    heading_path: tuple[str, ...]
    heading_level: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Intermediate document structure used for section-aware chunking."""

    blocks: tuple[StructuredBlock, ...]


@dataclass(frozen=True)
class ChunkRecord:
    """Chunk payload ready for insertion into document_chunks."""

    chunk_index: int
    page_start: int
    page_end: int
    section_type: SectionType
    section_title: str | None
    heading_path: tuple[str, ...]
    chunk_text: str
    chunk_sha256: str
    token_count: int
    section_key: str | None = None
    chunk_metadata: dict[str, Any] | None = None


@dataclass
class Phase2Summary:
    """Aggregated Phase 2 execution summary."""

    documents_selected: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    pages_inserted: int = 0
    chunks_inserted: int = 0
    ocr_documents: int = 0
    skipped_missing_files: int = 0

    def as_lines(self) -> list[str]:
        """Render summary counts in a stable CLI-friendly order."""

        return [
            f"documents selected: {self.documents_selected}",
            f"documents processed: {self.documents_processed}",
            f"documents failed: {self.documents_failed}",
            f"pages inserted: {self.pages_inserted}",
            f"chunks inserted: {self.chunks_inserted}",
            f"OCR documents: {self.ocr_documents}",
            f"skipped missing files: {self.skipped_missing_files}",
        ]


@dataclass(frozen=True)
class EmbeddingCandidate:
    """Document version candidate selected for Phase 3 embeddings."""

    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    order_date: date | None
    title: str
    detail_url: str | None
    pdf_url: str
    local_filename: str
    local_path: str
    ingest_status: str
    chunking_version: str | None
    chunk_count: int | None
    embedding_status: str | None
    embedding_error: str | None
    embedding_model: str | None
    embedding_dim: int | None
    embedded_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class StoredChunk:
    """Persisted chunk row plus document context used for Phase 3."""

    chunk_id: int
    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    order_date: date | None
    title: str
    chunk_index: int
    page_start: int
    page_end: int
    section_type: str
    section_title: str | None
    heading_path: str | None
    section_key: str | None
    chunk_text: str
    chunk_sha256: str
    token_count: int
    chunk_metadata: dict[str, Any]
    embedding_model: str | None = None
    embedding_created_at: datetime | None = None


@dataclass(frozen=True)
class SectionGroupInput:
    """Logical section aggregation derived from ordered chunks."""

    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    order_date: date | None
    title: str
    section_key: str
    section_type: str
    section_title: str | None
    heading_path: str | None
    page_start: int
    page_end: int
    chunks: tuple[StoredChunk, ...]


@dataclass(frozen=True)
class NodePayload:
    """Deterministic text payload prepared for node embeddings."""

    node_text: str
    token_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentNodeUpsert:
    """Database write payload for a document-level node."""

    document_version_id: int
    node_text: str
    token_count: int
    embedding: tuple[float, ...]
    embedding_model: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SectionNodeUpsert:
    """Database write payload for a section-level node."""

    document_version_id: int
    section_key: str
    section_type: str
    section_title: str | None
    heading_path: str | None
    page_start: int
    page_end: int
    node_text: str
    token_count: int
    embedding: tuple[float, ...]
    embedding_model: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkEmbeddingUpdate:
    """Chunk embedding update payload written in place."""

    chunk_id: int
    section_key: str
    chunk_metadata: dict[str, Any]
    embedding: tuple[float, ...]
    embedding_model: str


@dataclass(frozen=True)
class MetadataFilterInput:
    """Normalized metadata filters applied before lexical/vector retrieval."""

    record_key: str | None = None
    bucket_name: str | None = None
    document_version_ids: tuple[int, ...] = ()
    section_keys: tuple[str, ...] = ()
    section_types: tuple[str, ...] = ()


@dataclass
class Phase3EmbeddingSummary:
    """Aggregated Phase 3 embedding execution summary."""

    documents_selected: int = 0
    documents_embedded: int = 0
    documents_failed: int = 0
    document_nodes_written: int = 0
    section_nodes_written: int = 0
    chunk_embeddings_updated: int = 0

    def as_lines(self) -> list[str]:
        """Render summary counts in a stable CLI-friendly order."""

        return [
            f"documents selected: {self.documents_selected}",
            f"documents embedded: {self.documents_embedded}",
            f"documents failed: {self.documents_failed}",
            f"document nodes written: {self.document_nodes_written}",
            f"section nodes written: {self.section_nodes_written}",
            f"chunk embeddings updated: {self.chunk_embeddings_updated}",
        ]


RouteMode = Literal[
    "smalltalk",
    "general_knowledge",
    "structured_current_info",
    "current_official_lookup",
    "current_news_lookup",
    "historical_official_lookup",
    "direct_llm",
    "exact_lookup",
    "hierarchical_rag",
    "memory_scoped_rag",
    "corpus_metadata",
    "clarify",
    "abstain",
]

PlannerRoute = Literal[
    "structured_current_info",
    "order_metadata",
    "order_corpus_rag",
    "official_web",
    "current_news",
    "general_knowledge",
    "clarify",
    "abstain",
]

AnswerStatus = Literal[
    "answered",
    "cautious",
    "insufficient_context",
    "clarify",
    "abstained",
]

ClarificationCandidateType = Literal["matter", "person"]


@dataclass(frozen=True)
class ClarificationCandidate:
    """One explicit user-selectable clarification candidate."""

    candidate_id: str
    candidate_index: int
    candidate_type: ClarificationCandidateType
    title: str
    record_key: str | None = None
    bucket_name: str | None = None
    order_date: date | None = None
    document_version_id: int | None = None
    descriptor: str | None = None
    resolution_query: str | None = None
    canonical_person_id: str | None = None
    selection_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClarificationContext:
    """Persisted clarification prompt and its candidate set."""

    source_query: str
    source_route_mode: str
    source_query_intent: str
    candidate_type: ClarificationCandidateType
    candidates: tuple[ClarificationCandidate, ...]


@dataclass(frozen=True)
class ChatSessionStateRecord:
    """Persisted grounded session memory used by Phase 4 follow-ups."""

    session_id: UUID
    active_document_ids: tuple[int, ...] = ()
    active_document_version_ids: tuple[int, ...] = ()
    active_record_keys: tuple[str, ...] = ()
    active_entities: tuple[str, ...] = ()
    active_bucket_names: tuple[str, ...] = ()
    active_primary_title: str | None = None
    active_primary_entity: str | None = None
    active_signatory_name: str | None = None
    active_signatory_designation: str | None = None
    active_order_date: date | None = None
    active_order_place: str | None = None
    active_legal_provisions: tuple[str, ...] = ()
    last_chunk_ids: tuple[int, ...] = ()
    last_citation_chunk_ids: tuple[int, ...] = ()
    grounded_summary: str | None = None
    current_lookup_family: str | None = None
    current_lookup_focus: str | None = None
    current_lookup_query: str | None = None
    clarification_context: ClarificationContext | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ChatSessionSnapshot:
    """Top-level session row plus current grounded state."""

    session_id: UUID
    user_name: str | None
    created_at: datetime
    updated_at: datetime
    state: ChatSessionStateRecord | None = None


@dataclass(frozen=True)
class ChatSessionListEntry:
    """Persisted session metadata used to render recent-chat history."""

    session_id: UUID
    created_at: datetime
    updated_at: datetime
    first_user_query: str | None = None
    latest_user_query: str | None = None
    last_message_at: datetime | None = None
    turn_count: int = 0


@dataclass(frozen=True)
class QueryAnalysis:
    """Deterministic feature extraction used by the Phase 4 router."""

    raw_query: str
    normalized_query: str
    query_family: str = "ambiguous"
    normalized_expansions: tuple[str, ...] = ()
    matched_abbreviations: tuple[str, ...] = ()
    title_or_party_lookup_signals: tuple[str, ...] = ()
    procedural_or_outcome_signals: tuple[str, ...] = ()
    settlement_signals: tuple[str, ...] = ()
    general_explanatory_signals: tuple[str, ...] = ()
    smalltalk_signals: tuple[str, ...] = ()
    structured_current_info_signals: tuple[str, ...] = ()
    current_official_lookup_signals: tuple[str, ...] = ()
    current_news_signals: tuple[str, ...] = ()
    historical_official_signals: tuple[str, ...] = ()
    current_public_fact_signals: tuple[str, ...] = ()
    company_role_signals: tuple[str, ...] = ()
    order_context_override_signals: tuple[str, ...] = ()
    brief_summary_signals: tuple[str, ...] = ()
    current_info_query_family: str | None = None
    current_info_follow_up: bool = False
    follow_up_signals: tuple[str, ...] = ()
    matter_reference_signals: tuple[str, ...] = ()
    sat_court_signals: tuple[str, ...] = ()
    corpus_metadata_signals: tuple[str, ...] = ()
    asks_order_signatory: bool = False
    asks_order_date: bool = False
    asks_legal_provisions: bool = False
    asks_provision_explanation: bool = False
    asks_order_pan: bool = False
    asks_order_amount: bool = False
    asks_order_holding: bool = False
    asks_order_parties: bool = False
    asks_order_observations: bool = False
    asks_order_numeric_fact: bool = False
    active_matter_follow_up_intent: str | None = None
    active_order_override: bool = False
    fresh_query_override: bool = False
    likely_follow_up: bool = False
    has_active_documents: bool = False
    has_active_record_keys: bool = False
    has_session_scope: bool = False
    has_active_clarification: bool = False
    mentions_sebi: bool = False
    appears_smalltalk: bool = False
    appears_structured_current_info: bool = False
    appears_current_official_lookup: bool = False
    appears_current_news_lookup: bool = False
    appears_historical_official_lookup: bool = False
    appears_corpus_metadata_query: bool = False
    appears_sat_court_style: bool = False
    appears_non_sebi_person_query: bool = False
    appears_company_role_current_fact: bool = False
    appears_general_explanatory: bool = False
    appears_matter_specific: bool = False
    appears_settlement_specific: bool = False
    asks_brief_summary: bool = False
    requires_live_information: bool = False
    comparison_intent: bool = False
    comparison_terms: tuple[str, ...] = ()
    strict_scope_required: bool = False
    strict_single_matter: bool = False
    strict_lock_record_keys: tuple[str, ...] = ()
    strict_lock_titles: tuple[str, ...] = ()
    strict_lock_matched_aliases: tuple[str, ...] = ()
    strict_lock_matched_entities: tuple[str, ...] = ()
    strict_lock_reason_codes: tuple[str, ...] = ()
    strict_lock_ambiguous: bool = False
    strict_matter_lock: StrictMatterLock = field(default_factory=StrictMatterLock)


@dataclass(frozen=True)
class QueryPlan:
    """Canonical planner output used to choose the execution path."""

    route: PlannerRoute
    reason: str
    confidence: float
    use_structured_db: bool = False
    use_order_metadata: bool = False
    use_order_rag: bool = False
    use_official_web: bool = False
    use_general_web: bool = False
    force_fresh_named_matter_override: bool = False


@dataclass(frozen=True)
class RouteDecision:
    """Final deterministic routing choice for one query."""

    route_mode: RouteMode
    query_intent: str
    analysis: QueryAnalysis
    reason_codes: tuple[str, ...] = ()
    plan: QueryPlan | None = None


@dataclass(frozen=True)
class ExactLookupCandidate:
    """Document identity candidate returned by exact-lookup SQL."""

    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    external_record_id: str | None
    order_date: date | None
    title: str
    match_score: float


@dataclass(frozen=True)
class PromptContextChunk:
    """Context chunk included in a grounded answer prompt."""

    citation_number: int
    chunk_id: int
    document_version_id: int
    document_id: int
    record_key: str
    bucket_name: str
    title: str
    page_start: int
    page_end: int
    section_type: str
    section_title: str | None
    detail_url: str | None
    pdf_url: str | None
    chunk_text: str
    token_count: int
    score: float


@dataclass(frozen=True)
class Citation:
    """Structured citation mapped from inline markers like ``[1]``."""

    citation_number: int
    record_key: str
    title: str
    page_start: int | None
    page_end: int | None
    section_type: str | None
    document_version_id: int | None
    chunk_id: int | None
    detail_url: str | None = None
    pdf_url: str | None = None
    source_url: str | None = None
    source_title: str | None = None
    domain: str | None = None
    source_type: str | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class ChatTurnRecord:
    """Persisted answer-log row reconstructed as one visible chat turn."""

    answer_id: int
    session_id: UUID | None
    user_query: str
    route_mode: str
    query_intent: str | None
    answer_text: str
    answer_confidence: float
    citations: tuple[Citation, ...] = ()
    created_at: datetime | None = None


@dataclass(frozen=True)
class ChatAnswerPayload:
    """Phase 4 answer payload returned by the API and CLI."""

    session_id: UUID
    route_mode: RouteMode
    query_intent: str
    answer_text: str
    confidence: float
    citations: tuple[Citation, ...] = ()
    retrieved_chunk_ids: tuple[int, ...] = ()
    active_record_keys: tuple[str, ...] = ()
    answer_status: AnswerStatus = "answered"
    clarification_candidates: tuple[ClarificationCandidate, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)
