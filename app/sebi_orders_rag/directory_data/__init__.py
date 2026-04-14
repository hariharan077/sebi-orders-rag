"""Structured SEBI directory and office-reference package."""

from .models import (
    BoardMemberRecord,
    DirectoryIngestionSummary,
    DirectoryOfficeRecord,
    DirectoryPageParseResult,
    DirectoryPersonRecord,
    DirectoryReferenceDataset,
    DirectorySourceDefinition,
    DirectorySourceRunSummary,
    FetchedDirectorySource,
    OrgStructureRecord,
    ReferenceSnapshotRecord,
)
from .sources import (
    SOURCE_BOARD_MEMBERS,
    SOURCE_CONTACT_US,
    SOURCE_DIRECTORY,
    SOURCE_ORGCHART,
    SOURCE_REGIONAL_OFFICES,
    configured_directory_sources,
)

__all__ = [
    "BoardMemberRecord",
    "DirectoryIngestionSummary",
    "DirectoryOfficeRecord",
    "DirectoryPageParseResult",
    "DirectoryPersonRecord",
    "DirectoryReferenceDataset",
    "DirectorySourceDefinition",
    "DirectorySourceRunSummary",
    "FetchedDirectorySource",
    "OrgStructureRecord",
    "ReferenceSnapshotRecord",
    "SOURCE_BOARD_MEMBERS",
    "SOURCE_CONTACT_US",
    "SOURCE_DIRECTORY",
    "SOURCE_ORGCHART",
    "SOURCE_REGIONAL_OFFICES",
    "configured_directory_sources",
]
