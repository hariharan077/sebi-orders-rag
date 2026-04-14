"""Application service for Phase 2 extraction and chunk persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import SebiOrdersRagSettings
from ..schemas import ChunkRecord, ExtractedDocument, PendingDocumentVersion, Phase2Summary
from ..repositories.chunks import DocumentChunkRepository
from ..repositories.documents import DocumentRepository
from ..repositories.pages import DocumentPageRepository
from .chunker import build_chunks
from .chunk_postprocess import ChunkPostprocessSummary, postprocess_chunks
from .pdf_extract import extract_pdf_document
from .structure_parser import parse_document_structure

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedDocument:
    """In-memory Phase 2 result for one document version."""

    extracted_document: ExtractedDocument
    chunks: tuple[ChunkRecord, ...]
    chunk_postprocess_summary: ChunkPostprocessSummary


class Phase2IngestionService:
    """Coordinate Phase 2 extraction, structure parsing, chunking, and persistence."""

    def __init__(self, *, settings: SebiOrdersRagSettings, connection: Any) -> None:
        self._settings = settings
        self._connection = connection
        self._documents = DocumentRepository(connection)
        self._pages = DocumentPageRepository(connection)
        self._chunks = DocumentChunkRepository(connection)

    def run(
        self,
        *,
        apply: bool,
        record_key: str | None = None,
        document_version_id: int | None = None,
        limit: int | None = None,
    ) -> Phase2Summary:
        """Run Phase 2 for pending/failed document versions."""

        summary = Phase2Summary()
        candidates = self._documents.list_pending_versions(
            record_key=record_key,
            document_version_id=document_version_id,
            limit=None,
        )

        for candidate in candidates:
            pdf_path = Path(candidate.local_path)
            if not pdf_path.exists():
                summary.skipped_missing_files += 1
                LOGGER.warning(
                    "Skipping document_version_id=%s because the local file is missing: %s",
                    candidate.document_version_id,
                    pdf_path,
                )
                if apply:
                    self._mark_failed(
                        candidate,
                        f"Local file missing at processing time: {pdf_path}",
                    )
                continue

            if limit is not None and summary.documents_selected >= limit:
                break

            summary.documents_selected += 1
            if not apply:
                LOGGER.info(
                    "[dry-run] Would process document_version_id=%s record_key=%s path=%s",
                    candidate.document_version_id,
                    candidate.record_key,
                    pdf_path,
                )
                continue

            LOGGER.info(
                "Processing document_version_id=%s record_key=%s",
                candidate.document_version_id,
                candidate.record_key,
            )

            try:
                processed = self._process_document(pdf_path)
                LOGGER.info(
                    "Chunk post-process document_version_id=%s initial_chunks=%s final_chunks=%s merges=%s suppressed=%s",
                    candidate.document_version_id,
                    processed.chunk_postprocess_summary.initial_chunk_count,
                    processed.chunk_postprocess_summary.final_chunk_count,
                    processed.chunk_postprocess_summary.merges_applied,
                    processed.chunk_postprocess_summary.suppressed_chunks,
                )
                pages_inserted = self._pages.replace_pages(
                    document_version_id=candidate.document_version_id,
                    pages=processed.extracted_document.pages,
                )
                chunks_inserted = self._chunks.replace_chunks(
                    document_version_id=candidate.document_version_id,
                    chunks=processed.chunks,
                )
                self._documents.mark_version_done(
                    document_version_id=candidate.document_version_id,
                    parser_name=self._settings.parser_name,
                    parser_version=self._settings.parser_version,
                    chunking_version=self._settings.chunking_version,
                    page_count=processed.extracted_document.page_count,
                    extracted_char_count=processed.extracted_document.extracted_char_count,
                    ocr_used=processed.extracted_document.ocr_used,
                    chunk_count=len(processed.chunks),
                )
                self._connection.commit()
            except Exception as exc:
                self._connection.rollback()
                LOGGER.exception(
                    "Phase 2 failed for document_version_id=%s",
                    candidate.document_version_id,
                )
                self._mark_failed(candidate, f"{type(exc).__name__}: {exc}")
                summary.documents_failed += 1
                continue

            summary.documents_processed += 1
            summary.pages_inserted += pages_inserted
            summary.chunks_inserted += chunks_inserted
            if processed.extracted_document.ocr_used:
                summary.ocr_documents += 1

        return summary

    def _process_document(self, pdf_path: Path) -> ProcessedDocument:
        extracted_document = extract_pdf_document(pdf_path, settings=self._settings)
        parsed_document = parse_document_structure(
            extracted_document.pages,
            min_heading_caps_ratio=self._settings.min_heading_caps_ratio,
            model_name=self._settings.embedding_model,
        )
        chunks = build_chunks(
            parsed_document,
            model_name=self._settings.embedding_model,
            target_chunk_tokens=self._settings.target_chunk_tokens,
            max_chunk_tokens=self._settings.max_chunk_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
        )
        postprocessed = postprocess_chunks(
            chunks,
            page_count=extracted_document.page_count,
            model_name=self._settings.embedding_model,
            max_chunk_tokens=self._settings.max_chunk_tokens,
        )
        return ProcessedDocument(
            extracted_document=extracted_document,
            chunks=postprocessed.chunks,
            chunk_postprocess_summary=postprocessed.summary,
        )

    def _mark_failed(self, candidate: PendingDocumentVersion, ingest_error: str) -> None:
        try:
            self._documents.mark_version_failed(
                document_version_id=candidate.document_version_id,
                parser_name=self._settings.parser_name,
                parser_version=self._settings.parser_version,
                chunking_version=self._settings.chunking_version,
                ingest_error=ingest_error,
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            LOGGER.exception(
                "Failed to record Phase 2 failure state for document_version_id=%s",
                candidate.document_version_id,
            )
