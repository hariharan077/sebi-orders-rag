"""PDF extraction pipeline for Phase 2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import SebiOrdersRagSettings
from ..exceptions import MissingDependencyError
from ..schemas import ExtractedDocument, ExtractedPage
from ..utils.strings import sha256_hexdigest
from .ocr import ocr_dependencies_available, ocr_page
from .text_normalizer import normalize_extracted_text
from .token_count import token_count

LOGGER = logging.getLogger(__name__)


def _import_pymupdf() -> Any:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise MissingDependencyError(
            "PyMuPDF is required for Phase 2 PDF extraction. "
            "Install the dependencies from requirements-sebi-orders-rag.txt."
        ) from exc
    return fitz


def extract_pdf_document(
    pdf_path: Path,
    *,
    settings: SebiOrdersRagSettings,
) -> ExtractedDocument:
    """Extract page-by-page text from a PDF with optional OCR fallback."""

    fitz = _import_pymupdf()
    pages: list[ExtractedPage] = []
    any_ocr_used = False

    if settings.enable_ocr and not ocr_dependencies_available():
        LOGGER.warning(
            "OCR is enabled but pytesseract is unavailable; using the OpenAI OCR fallback when possible."
        )

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            extracted_text = normalize_extracted_text(page.get_text("text") or "")
            low_text = len(extracted_text) < settings.low_text_char_threshold
            ocr_text: str | None = None
            final_text = extracted_text

            if settings.enable_ocr and low_text:
                ocr_result = ocr_page(
                    page,
                    model_name=settings.chat_model,
                    openai_api_key=settings.openai_api_key,
                    openai_base_url=settings.openai_base_url,
                )
                if ocr_result:
                    ocr_text = normalize_extracted_text(ocr_result)
                    any_ocr_used = True
                    if len(ocr_text) >= len(final_text):
                        final_text = ocr_text

            page_token_count = token_count(final_text, model_name=settings.embedding_model)
            pages.append(
                ExtractedPage(
                    page_no=page_index,
                    extracted_text=extracted_text,
                    ocr_text=ocr_text,
                    final_text=final_text,
                    char_count=len(final_text),
                    token_count=page_token_count,
                    low_text=low_text,
                    page_sha256=sha256_hexdigest(final_text),
                )
            )

    return ExtractedDocument(pages=tuple(pages), ocr_used=any_ocr_used)
