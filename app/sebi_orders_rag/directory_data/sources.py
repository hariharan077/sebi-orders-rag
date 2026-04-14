"""Configured official SEBI structured-reference source definitions."""

from __future__ import annotations

from ..config import SebiOrdersRagSettings
from .models import DirectorySourceDefinition

SOURCE_DIRECTORY = "directory"
SOURCE_ORGCHART = "orgchart"
SOURCE_REGIONAL_OFFICES = "regional_offices"
SOURCE_CONTACT_US = "contact_us"
SOURCE_BOARD_MEMBERS = "board_members"


def configured_directory_sources(
    settings: SebiOrdersRagSettings,
) -> tuple[DirectorySourceDefinition, ...]:
    """Return the configured official source pages for structured reference data."""

    return (
        DirectorySourceDefinition(
            source_type=SOURCE_DIRECTORY,
            title="SEBI Directory",
            url=settings.directory_source_directory_url,
        ),
        DirectorySourceDefinition(
            source_type=SOURCE_ORGCHART,
            title="SEBI Organisation Structure",
            url=settings.directory_source_orgchart_url,
        ),
        DirectorySourceDefinition(
            source_type=SOURCE_REGIONAL_OFFICES,
            title="SEBI Regional Offices",
            url=settings.directory_source_regional_offices_url,
        ),
        DirectorySourceDefinition(
            source_type=SOURCE_CONTACT_US,
            title="SEBI Contact Us",
            url=settings.directory_source_contact_us_url,
        ),
        DirectorySourceDefinition(
            source_type=SOURCE_BOARD_MEMBERS,
            title="SEBI Board Members",
            url=settings.directory_source_board_members_url,
        ),
    )


def source_title_for_type(source_type: str) -> str:
    """Return a stable human-readable label for one configured source type."""

    lookup = {
        SOURCE_DIRECTORY: "SEBI Directory",
        SOURCE_ORGCHART: "SEBI Organisation Structure",
        SOURCE_REGIONAL_OFFICES: "SEBI Regional Offices",
        SOURCE_CONTACT_US: "SEBI Contact Us",
        SOURCE_BOARD_MEMBERS: "SEBI Board Members",
    }
    return lookup.get(source_type, source_type.replace("_", " ").title())
