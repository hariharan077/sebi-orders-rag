"""Current-information provider selection helpers."""

from __future__ import annotations

from typing import Any

from ..config import SebiOrdersRagSettings
from .company_facts import CompanyRoleQuery, detect_company_role_order_context, parse_company_role_query
from .history_lookup import HistoricalOfficialLookupProvider
from .news_lookup import CurrentNewsLookupProvider
from .provider import (
    CurrentInfoProvider,
    CurrentInfoResult,
    CurrentInfoSource,
    UnavailableCurrentInfoProvider,
)


def build_current_info_provider(
    settings: SebiOrdersRagSettings,
    *,
    connection: Any | None = None,
) -> CurrentInfoProvider:
    """Return the configured current-information provider for this environment."""

    if not settings.current_lookup_enabled:
        return UnavailableCurrentInfoProvider(
            reason=(
                "Current official lookup is disabled in this environment. "
                "Please verify on the official SEBI or Government of India website."
            )
        )
    from .official_lookup import OfficialWebsiteCurrentInfoProvider

    return OfficialWebsiteCurrentInfoProvider(
        settings=settings,
        connection=connection,
    )


__all__ = [
    "CompanyRoleQuery",
    "CurrentInfoProvider",
    "CurrentInfoResult",
    "CurrentInfoSource",
    "CurrentNewsLookupProvider",
    "HistoricalOfficialLookupProvider",
    "UnavailableCurrentInfoProvider",
    "build_current_info_provider",
    "detect_company_role_order_context",
    "parse_company_role_query",
]
