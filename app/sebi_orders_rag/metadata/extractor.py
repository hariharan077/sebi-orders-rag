"""High-level metadata extraction orchestration for one order."""

from __future__ import annotations

from datetime import date

from .legal_provisions import extract_legal_provisions
from .models import ExtractedMetadataBundle, MetadataChunkText, MetadataPageText
from .numeric_facts import extract_numeric_facts
from .signatory import extract_signatory_metadata
from .tables import extract_price_movements


def extract_order_metadata_bundle(
    *,
    document_version_id: int,
    pages: tuple[MetadataPageText, ...],
    chunks: tuple[MetadataChunkText, ...],
    fallback_order_date: date | None = None,
    title: str | None = None,
) -> ExtractedMetadataBundle:
    """Extract order-level metadata and legal provisions in one deterministic pass."""

    metadata = extract_signatory_metadata(
        document_version_id=document_version_id,
        pages=pages,
        fallback_order_date=fallback_order_date,
    )
    provisions = extract_legal_provisions(
        document_version_id=document_version_id,
        chunks=chunks,
    )
    price_movements = extract_price_movements(
        document_version_id=document_version_id,
        pages=pages,
    )
    numeric_facts = extract_numeric_facts(
        document_version_id=document_version_id,
        pages=pages,
        chunks=chunks,
        price_movements=price_movements,
        title=title,
    )
    return ExtractedMetadataBundle(
        order_metadata=metadata,
        legal_provisions=provisions,
        numeric_facts=numeric_facts,
        price_movements=price_movements,
    )
