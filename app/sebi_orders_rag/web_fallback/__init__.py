"""Controlled web fallback interfaces."""

from .general_search import GeneralWebSearchProvider
from .models import WebSearchRequest, WebSearchResult, WebSearchSource
from .official_search import OfficialWebSearchProvider
from .provider import (
    UnavailableWebSearchProvider,
    WebSearchProvider,
    build_general_web_search_provider,
    build_official_web_search_provider,
)

__all__ = [
    "GeneralWebSearchProvider",
    "OfficialWebSearchProvider",
    "UnavailableWebSearchProvider",
    "WebSearchProvider",
    "WebSearchRequest",
    "WebSearchResult",
    "WebSearchSource",
    "build_general_web_search_provider",
    "build_official_web_search_provider",
]
