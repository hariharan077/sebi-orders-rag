"""Official-domain-first web search provider."""

from __future__ import annotations

from .provider import WebSearchProvider


class OfficialWebSearchProvider(WebSearchProvider):
    """Thin wrapper kept for explicit official-search imports."""

    def __init__(self, *, delegate: WebSearchProvider) -> None:
        self._delegate = delegate

    def search(self, *, request):
        return self._delegate.search(request=request)
