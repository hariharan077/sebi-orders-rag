"""HTML fetcher for official SEBI structured-reference pages."""

from __future__ import annotations

import hashlib
import ssl
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

from .models import DirectorySourceDefinition, FetchedDirectorySource


class OfficialDirectoryHtmlFetcher:
    """Fetch official HTML pages with a stable user agent and TLS fallback."""

    def __init__(self, *, timeout_seconds: float, user_agent: str) -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent

    def fetch(self, source: DirectorySourceDefinition) -> FetchedDirectorySource:
        request = Request(
            source.url,
            headers={"User-Agent": self._user_agent},
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                raw_bytes = response.read()
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            if not isinstance(reason, ssl.SSLCertVerificationError):
                raise
            with urlopen(  # noqa: S310
                request,
                timeout=self._timeout_seconds,
                context=ssl._create_unverified_context(),
            ) as response:
                raw_bytes = response.read()

        raw_html = raw_bytes.decode("utf-8", errors="ignore")
        return FetchedDirectorySource(
            source_type=source.source_type,
            title=source.title,
            source_url=source.url,
            raw_html=raw_html,
            content_sha256=hashlib.sha256(raw_html.encode("utf-8")).hexdigest(),
            fetched_at=datetime.now(timezone.utc),
        )
