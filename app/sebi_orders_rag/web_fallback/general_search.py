"""Broader web-search provider used only on explicitly allowed routes."""

from __future__ import annotations

from .provider import WebSearchProvider


class GeneralWebSearchProvider(WebSearchProvider):
    """Thin wrapper kept for explicit general-search imports."""

    def __init__(self, *, delegate: WebSearchProvider) -> None:
        self._delegate = delegate

    def search(self, *, request):
        return self._delegate.search(request=request)
