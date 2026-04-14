"""Chunk QA helpers for the SEBI Orders RAG project."""

from .chunk_audit import (
    CHUNK_DENSITY_HIGH_FLAG,
    DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG,
    HEADING_ONLY_CHUNK_FLAG,
    OVERSIZED_CHUNK_FLAG,
    SHORT_DOC_OVERFRAGMENTED_FLAG,
    SUSPICIOUS_SECTION_JUMP_FLAG,
    TINY_CHUNK_FLAG,
    ChunkAuditAnalyzer,
    ChunkAuditService,
    ChunkInspectionResult,
    CorpusAuditResult,
    DocumentAuditResult,
    build_text_preview,
    calculate_severity_score,
    is_heading_only_chunk_text,
)
from .report_formatter import (
    render_corpus_audit_json,
    render_corpus_audit_report,
    render_document_json,
    render_document_report,
    write_report,
)

__all__ = [
    "CHUNK_DENSITY_HIGH_FLAG",
    "DUPLICATE_CHUNK_TEXT_IN_DOC_FLAG",
    "HEADING_ONLY_CHUNK_FLAG",
    "OVERSIZED_CHUNK_FLAG",
    "SHORT_DOC_OVERFRAGMENTED_FLAG",
    "SUSPICIOUS_SECTION_JUMP_FLAG",
    "TINY_CHUNK_FLAG",
    "ChunkAuditAnalyzer",
    "ChunkAuditService",
    "ChunkInspectionResult",
    "CorpusAuditResult",
    "DocumentAuditResult",
    "build_text_preview",
    "calculate_severity_score",
    "is_heading_only_chunk_text",
    "render_corpus_audit_json",
    "render_corpus_audit_report",
    "render_document_json",
    "render_document_report",
    "write_report",
]
