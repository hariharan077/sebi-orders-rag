"""Domain normalization and ranking for web fallback sources."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .models import WebSearchSource

_MONTH_INDEX = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_YEAR_RE = re.compile(r"/(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)-(?P<year>\d{4})/", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"/(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?:/|$)")
_YEAR_RE = re.compile(r"(?<!\d)(?P<year>20\d{2}|19\d{2})(?!\d)")
_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"})


def canonicalize_source_url(url: str) -> str:
    """Return a stable URL form for source deduplication and display."""

    stripped = (url or "").strip()
    if not stripped:
        return ""
    parsed = urlparse(stripped)
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key and key.lower() not in _TRACKING_QUERY_KEYS
        and not any(key.lower().startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES)
    ]
    normalized_path = parsed.path or ""
    if normalized_path not in {"", "/"}:
        normalized_path = normalized_path.rstrip("/")
    normalized = parsed._replace(
        netloc=(parsed.netloc or "").lower(),
        path=normalized_path,
        query=urlencode(filtered_query, doseq=True),
        fragment="",
    )
    return urlunparse(normalized)


def extract_domain(url: str) -> str:
    """Return the normalized hostname for one URL."""

    if not url:
        return ""
    parsed = urlparse(url)
    return (parsed.netloc or "").lower().removeprefix("www.").removeprefix("m.")


def is_domain_allowed(domain: str, allowed_domains: tuple[str, ...]) -> bool:
    """Return whether one hostname is inside the allowed-domain policy."""

    normalized_domain = (domain or "").lower().removeprefix("www.")
    for allowed in allowed_domains:
        normalized_allowed = allowed.lower().removeprefix("www.")
        if (
            normalized_domain == normalized_allowed
            or normalized_domain.endswith(f".{normalized_allowed}")
        ):
            return True
    return False


def rank_web_sources(
    sources: tuple[WebSearchSource, ...],
    *,
    allowed_domains: tuple[str, ...] = (),
    unique_per_domain: bool = False,
) -> tuple[WebSearchSource, ...]:
    """Return web sources in stable priority order with deduplication."""

    deduped: dict[str, WebSearchSource] = {}
    for source in sources:
        canonical_url = canonicalize_source_url(source.source_url)
        if not canonical_url:
            continue
        existing = deduped.get(canonical_url)
        if existing is None or _source_title_priority(source) < _source_title_priority(existing):
            deduped[canonical_url] = source

    ordered = tuple(
        sorted(
            deduped.values(),
            key=lambda source: (
                _domain_priority(source.domain, allowed_domains=allowed_domains),
                _source_title_priority(source),
                _url_page_priority(source.source_url),
                _url_recency_key(source.source_url),
                source.source_title.lower(),
                source.source_url,
            ),
        )
    )
    if not unique_per_domain:
        return ordered

    domain_deduped: list[WebSearchSource] = []
    seen_domains: set[str] = set()
    for source in ordered:
        normalized_domain = (source.domain or "").lower().removeprefix("www.")
        if normalized_domain in seen_domains:
            continue
        seen_domains.add(normalized_domain)
        domain_deduped.append(source)
    return tuple(domain_deduped)


def _domain_priority(domain: str, *, allowed_domains: tuple[str, ...]) -> tuple[int, int]:
    if allowed_domains:
        matched_allowed = _matched_allowed_domain_rank(domain, allowed_domains)
        if matched_allowed is not None:
            return matched_allowed
    if domain.endswith(".gov.in") or domain.endswith(".nic.in"):
        return (0, 1)
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return (1, 0)
    return (2, 0)


def _matched_allowed_domain_rank(
    domain: str,
    allowed_domains: tuple[str, ...],
) -> tuple[int, int] | None:
    normalized_domain = (domain or "").lower().removeprefix("www.")
    matches: list[tuple[int, int]] = []
    for index, allowed in enumerate(allowed_domains):
        normalized_allowed = allowed.lower().removeprefix("www.")
        if (
            normalized_domain == normalized_allowed
            or normalized_domain.endswith(f".{normalized_allowed}")
        ):
            specificity = -len(normalized_allowed.split("."))
            matches.append((index, specificity))
    if not matches:
        return None
    return min(matches)


def _url_page_priority(url: str) -> int:
    normalized = url.lower()
    if "homeaction.do" in normalized and "search=" in normalized:
        return 2
    if any(
        marker in normalized
        for marker in (
            "/press-releases/",
            "/circulars/",
            "/public-notices/",
            "/regulations/",
            "/acts/",
            "/reports-and-statistics/",
            "/publications/",
            "/attachdocs/",
        )
    ):
        return 0
    return 1


def _url_recency_key(url: str) -> tuple[int, int, int]:
    normalized = url.lower()

    iso_match = _ISO_DATE_RE.search(normalized)
    if iso_match is not None:
        return (
            -int(iso_match.group("year")),
            -int(iso_match.group("month")),
            -int(iso_match.group("day")),
        )

    month_match = _MONTH_YEAR_RE.search(normalized)
    if month_match is not None:
        return (
            -int(month_match.group("year")),
            -_MONTH_INDEX[month_match.group("month")[:3].lower()],
            0,
        )

    year_match = _YEAR_RE.search(normalized)
    if year_match is not None:
        return (-int(year_match.group("year")), 0, 0)

    return (0, 0, 0)


def _source_title_priority(source: WebSearchSource) -> int:
    title = (source.source_title or "").strip().lower()
    domain = (source.domain or "").strip().lower()
    if not title or title == domain:
        return 1
    return 0
