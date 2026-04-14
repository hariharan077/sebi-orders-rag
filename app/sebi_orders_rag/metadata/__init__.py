"""Metadata extraction helpers for SEBI order hardening."""

from .extractor import extract_order_metadata_bundle
from .legal_provisions import explain_provisions, extract_legal_provisions
from .models import (
    ExtractedMetadataBundle,
    ExtractedLegalProvision,
    ExtractedNumericFact,
    ExtractedOrderMetadata,
    ExtractedPriceMovement,
    MetadataChunkText,
    MetadataExtractionTarget,
    MetadataPageText,
    StoredLegalProvision,
    StoredNumericFact,
    StoredOrderMetadata,
    StoredPriceMovement,
)
from .service import MetadataAnswer, OrderMetadataService
from .signatory import extract_signatory_metadata

__all__ = [
    "ExtractedLegalProvision",
    "ExtractedMetadataBundle",
    "ExtractedNumericFact",
    "ExtractedOrderMetadata",
    "ExtractedPriceMovement",
    "MetadataAnswer",
    "MetadataChunkText",
    "MetadataExtractionTarget",
    "MetadataPageText",
    "OrderMetadataService",
    "StoredLegalProvision",
    "StoredNumericFact",
    "StoredOrderMetadata",
    "StoredPriceMovement",
    "explain_provisions",
    "extract_legal_provisions",
    "extract_order_metadata_bundle",
    "extract_signatory_metadata",
]
