"""Custom exceptions for the SEBI Orders RAG package."""

from __future__ import annotations

from pathlib import Path


class SebiOrdersRagError(Exception):
    """Base exception for SEBI Orders RAG errors."""


class ConfigurationError(SebiOrdersRagError):
    """Raised when required runtime configuration is invalid or missing."""


class MissingDependencyError(SebiOrdersRagError):
    """Raised when an optional runtime dependency is required but unavailable."""


class SchemaInitializationError(SebiOrdersRagError):
    """Raised when the retrieval schema cannot be initialized."""


class ManifestValidationError(SebiOrdersRagError):
    """Raised when a manifest file or row does not match the expected format."""

    def __init__(
        self,
        message: str,
        *,
        manifest_path: Path | None = None,
        row_number: int | None = None,
    ) -> None:
        location_parts: list[str] = []
        if manifest_path is not None:
            location_parts.append(str(manifest_path))
        if row_number is not None:
            location_parts.append(f"row {row_number}")
        location = " ".join(location_parts)
        if location:
            super().__init__(f"{message} ({location})")
        else:
            super().__init__(message)
        self.manifest_path = manifest_path
        self.row_number = row_number
