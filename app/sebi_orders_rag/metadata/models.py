"""Typed models for extracted order metadata, legal provisions, and numeric facts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class MetadataPageText:
    """Page-level text used by the metadata extractor."""

    page_no: int
    text: str


@dataclass(frozen=True)
class MetadataChunkText:
    """Chunk-level text used for legal-provision extraction."""

    chunk_id: int
    page_start: int
    page_end: int
    text: str
    section_type: str | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class MetadataExtractionTarget:
    """One persisted document selected for metadata extraction."""

    document_version_id: int
    document_id: int
    record_key: str
    title: str
    order_date: date | None = None
    detail_url: str | None = None
    pdf_url: str | None = None


@dataclass(frozen=True)
class ExtractedOrderMetadata:
    """Extracted metadata before persistence."""

    document_version_id: int
    signatory_name: str | None = None
    signatory_designation: str | None = None
    signatory_page_start: int | None = None
    signatory_page_end: int | None = None
    order_date: date | None = None
    place: str | None = None
    issuing_authority_type: str | None = None
    authority_panel: tuple[str, ...] = ()
    metadata_confidence: float = 0.0
    extraction_version: str = "metadata_v1"


@dataclass(frozen=True)
class ExtractedLegalProvision:
    """Extracted legal provision before persistence."""

    document_version_id: int
    statute_name: str
    section_or_regulation: str
    provision_type: str
    text_snippet: str
    page_start: int | None = None
    page_end: int | None = None

    @property
    def row_sha256(self) -> str:
        payload = "|".join(
            str(value or "")
            for value in (
                self.document_version_id,
                self.statute_name,
                self.section_or_regulation,
                self.provision_type,
                self.page_start,
                self.page_end,
                self.text_snippet,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractedNumericFact:
    """Deterministically extracted numeric or structured fact before persistence."""

    document_version_id: int
    fact_type: str
    subject: str | None = None
    value_text: str | None = None
    value_numeric: float | None = None
    unit: str | None = None
    context_label: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    @property
    def row_sha256(self) -> str:
        payload = "|".join(
            str(value or "")
            for value in (
                self.document_version_id,
                self.fact_type,
                self.subject,
                self.value_text,
                self.value_numeric,
                self.unit,
                self.context_label,
                self.page_start,
                self.page_end,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractedPriceMovement:
    """Deterministically extracted period-wise price movement row before persistence."""

    document_version_id: int
    period_label: str
    period_start_text: str | None = None
    period_end_text: str | None = None
    start_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    end_price: float | None = None
    pct_change: float | None = None
    rationale: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    @property
    def row_sha256(self) -> str:
        payload = "|".join(
            str(value or "")
            for value in (
                self.document_version_id,
                self.period_label,
                self.period_start_text,
                self.period_end_text,
                self.start_price,
                self.high_price,
                self.low_price,
                self.end_price,
                self.pct_change,
                self.rationale,
                self.page_start,
                self.page_end,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractedMetadataBundle:
    """One deterministic extraction pass across metadata, provisions, and numeric facts."""

    order_metadata: ExtractedOrderMetadata
    legal_provisions: tuple[ExtractedLegalProvision, ...] = ()
    numeric_facts: tuple[ExtractedNumericFact, ...] = ()
    price_movements: tuple[ExtractedPriceMovement, ...] = ()


@dataclass(frozen=True)
class StoredOrderMetadata:
    """Persisted order-level metadata joined with document identity."""

    document_version_id: int
    document_id: int
    record_key: str
    title: str
    detail_url: str | None
    pdf_url: str | None
    signatory_name: str | None = None
    signatory_designation: str | None = None
    signatory_page_start: int | None = None
    signatory_page_end: int | None = None
    order_date: date | None = None
    place: str | None = None
    issuing_authority_type: str | None = None
    authority_panel: tuple[str, ...] = ()
    metadata_confidence: float = 0.0
    extraction_version: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class StoredLegalProvision:
    """Persisted provision row joined with document identity."""

    provision_id: int
    document_version_id: int
    document_id: int
    record_key: str
    title: str
    detail_url: str | None
    pdf_url: str | None
    statute_name: str
    section_or_regulation: str
    provision_type: str
    text_snippet: str
    page_start: int | None = None
    page_end: int | None = None
    row_sha256: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class StoredNumericFact:
    """Persisted numeric fact row joined with document identity."""

    numeric_fact_id: int
    document_version_id: int
    document_id: int
    record_key: str
    title: str
    detail_url: str | None
    pdf_url: str | None
    fact_type: str
    subject: str | None = None
    value_text: str | None = None
    value_numeric: float | None = None
    unit: str | None = None
    context_label: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    row_sha256: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class StoredPriceMovement:
    """Persisted price-movement row joined with document identity."""

    price_movement_id: int
    document_version_id: int
    document_id: int
    record_key: str
    title: str
    detail_url: str | None
    pdf_url: str | None
    period_label: str
    period_start_text: str | None = None
    period_end_text: str | None = None
    start_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    end_price: float | None = None
    pct_change: float | None = None
    rationale: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    row_sha256: str | None = None
    updated_at: datetime | None = None
