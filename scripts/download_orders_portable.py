#!/usr/bin/env python3
"""Portable single-file downloader for SEBI order PDFs.

This script is self-contained and uses only the Python standard library.

Quick start:
  python download_orders_portable.py examples
  python download_orders_portable.py backfill --output-dir ./sebi-orders-pdfs
  python download_orders_portable.py backfill --category orders-of-sat
  python download_orders_portable.py backfill --category-range orders-of-sat:25
  python download_orders_portable.py backfill --category-range orders-of-sat:25 \
      --category-range orders-of-chairperson-members:1-3
  python download_orders_portable.py update --category orders-of-sat

Environment overrides:
  SEBI_ORDERS_TIMEOUT_SECONDS=30
  SEBI_ORDERS_USER_AGENT="SEBIOrdersRAG/0.1 (+https://www.sebi.gov.in/enforcement/orders.html)"
  SEBI_ORDERS_OUTPUT_DIR=./sebi-orders-pdfs

Scheduler note:
  This script does not install cron jobs. Use the `examples` command for a
  commented 6:00 PM IST cron line you can copy into `crontab -e` later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import mimetypes
import os
import posixpath
import re
import shutil
import ssl
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit
from urllib.request import (
    HTTPCookieProcessor,
    HTTPSHandler,
    Request,
    build_opener,
    urlopen,
)

RunMode = Literal["backfill", "update"]

try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11
    UTC = timezone.utc  # noqa: UP017

SEBI_ORDERS_ROOT_URL = "https://www.sebi.gov.in/enforcement/orders.html"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = "SEBIOrdersRAG/0.1 (+https://www.sebi.gov.in/enforcement/orders.html)"
DEFAULT_OUTPUT_DIR = Path("./sebi-orders-pdfs")
MANIFEST_FILE_NAME = "orders_manifest.csv"
DEFAULT_UPDATE_RECENT_PAGES = 10
DEFAULT_UNCHANGED_PAGE_THRESHOLD = 3

_UNSAFE_SCHEMES = ("javascript:", "mailto:", "tel:")
_NON_ALNUM_PATTERN = re.compile(r"[^0-9a-zA-Z\s]+")
_DEFAULT_SECTION_NAME = "Enforcement"
_DEFAULT_SUBSECTION_NAME = "Orders"
_DEFAULT_DEPARTMENT_ID = "-1"
_DEFAULT_INTMID = "-1"
_FILE_SUFFIXES = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip")
_AJAX_SEPARATOR = "#@#"
_HIDDEN_INPUT_RE = re.compile(
    r"<input[^>]+name=['\"](?P<name>[^'\"]+)['\"][^>]+value=['\"]?(?P<value>[^'\">\s]+)",
    re.IGNORECASE,
)
_LAST_PAGE_RE = re.compile(
    r"searchFormNewsList\(\s*['\"]n['\"]\s*,\s*['\"](?P<value>\d+)['\"]\s*\)"
    r".*?title=['\"]Last['\"]",
    re.IGNORECASE,
)
_NEXT_LINK_RE = re.compile(r"title=['\"]Next['\"]", re.IGNORECASE)
_EXTERNAL_ID_PATH_RE = re.compile(r"_(?P<external_id>\d+)\.html?$", re.IGNORECASE)
_FILE_QUERY_KEYS = ("file", "url", "src")
_CHUNK_SIZE_BYTES = 64 * 1024
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
_DERIVED_ID_LENGTH = 16
_MAX_SLUG_LENGTH = 80
_MANIFEST_FIELD_NAMES = (
    "record_key",
    "bucket_name",
    "order_date",
    "title",
    "external_record_id",
    "detail_url",
    "pdf_url",
    "local_filename",
    "status",
    "error",
    "first_seen_at",
    "last_seen_at",
)
_MATERIAL_MANIFEST_FIELDS = (
    "record_key",
    "bucket_name",
    "order_date",
    "title",
    "external_record_id",
    "detail_url",
    "pdf_url",
    "local_filename",
    "status",
    "error",
    "first_seen_at",
)


class OrdersDownloadFatalError(RuntimeError):
    """Raised when the downloader cannot continue safely."""


class RootCategorySyncError(RuntimeError):
    """Raised when root-category discovery cannot complete."""


class ListingCrawlError(RuntimeError):
    """Raised when a listing page cannot be fetched or parsed safely."""


class DetailPageFetchError(RuntimeError):
    """Raised when a detail page cannot be fetched safely."""


class FileDownloadError(RuntimeError):
    """Raised when a file cannot be downloaded safely."""

    def __init__(self, message: str, *, reason_code: str = "unknown") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class TargetCategoryDefinition:
    """Phase 1 SEBI category definition."""

    bucket_name: str
    default_label: str
    crawl_priority: int
    aliases: tuple[str, ...]


PHASE1_TARGET_CATEGORIES: tuple[TargetCategoryDefinition, ...] = (
    TargetCategoryDefinition(
        bucket_name="orders-of-sat",
        default_label="Orders of SAT",
        crawl_priority=10,
        aliases=("orders of sat",),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-of-chairperson-members",
        default_label="Orders of Chairperson/Members",
        crawl_priority=20,
        aliases=("orders of chairperson members",),
    ),
    TargetCategoryDefinition(
        bucket_name="settlement-orders",
        default_label="Settlement Order",
        crawl_priority=30,
        aliases=("settlement order", "settlement orders"),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-of-aa-under-rti-act",
        default_label="Orders of AA under the RTI Act",
        crawl_priority=40,
        aliases=(
            "orders of aa under the rti act",
            "orders of aa under rti act",
        ),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-of-corporatisation-demutualisation-scheme",
        default_label="Orders of Corporatisation / Demutualisation Scheme",
        crawl_priority=50,
        aliases=(
            "orders of corporatisation demutualisation scheme",
            "orders of corporatisation / demutualisation scheme",
        ),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-of-ao",
        default_label="Orders of AO",
        crawl_priority=60,
        aliases=(
            "orders of ao",
            "orders of adjudicating officer",
            "orders of adjudicating officers",
        ),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-of-courts",
        default_label="Orders of Courts",
        crawl_priority=70,
        aliases=("orders of courts",),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-of-special-courts",
        default_label="Orders Of Special Courts",
        crawl_priority=80,
        aliases=(
            "orders of special courts",
            "orders of special court",
        ),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-of-ed-cgm",
        default_label="Orders of ED / CGM (Quasi-Judicial Authorities)",
        crawl_priority=90,
        aliases=(
            "orders of ed cgm",
            "orders of ed cgm quasi judicial authorities",
            "orders of ed / cgm",
            "orders of ed / cgm quasi judicial authorities",
        ),
    ),
    TargetCategoryDefinition(
        bucket_name="orders-under-regulation-30a",
        default_label="Orders under Regulation 30A of the SEBI (Intermediaries) Regulations, 2008",
        crawl_priority=100,
        aliases=(
            "orders under regulation 30a of the sebi intermediaries regulations 2008",
            "orders under regulation 30a",
        ),
    ),
)

SUPPORTED_BUCKET_NAMES = tuple(target.bucket_name for target in PHASE1_TARGET_CATEGORIES)


class RootPageFetcher(Protocol):
    def fetch(self, url: str) -> RootPageFetchResult: ...


class ListingPageFetcherProtocol(Protocol):
    def fetch_initial_page(self, context: ListingCategoryContext) -> ListingFetchResult: ...

    def fetch_page(
        self,
        *,
        context: ListingCategoryContext,
        page_number: int,
        pagination: ListingPaginationState,
    ) -> ListingFetchResult: ...


class DetailPageFetcher(Protocol):
    def fetch(self, *, url: str, referer_url: str | None = None) -> DetailPageFetchResult: ...


class FileDownloader(Protocol):
    def fetch(self, *, url: str, referer_url: str | None = None) -> DownloadedFile: ...


@dataclass(frozen=True)
class ParsedAnchorLink:
    label: str
    href: str
    url: str
    normalized_label: str


@dataclass(frozen=True)
class RootPageFetchResult:
    url: str
    html: str
    fetched_at: datetime
    content_hash: str


@dataclass(frozen=True)
class DiscoveredCategoryLink:
    bucket_name: str
    label: str
    listing_url: str


@dataclass(frozen=True)
class ListingFetchResult:
    request_url: str
    referer_url: str | None
    page_number: int
    http_status: int
    content_bytes: bytes
    text: str
    fetched_at: datetime
    content_hash: str


@dataclass(frozen=True)
class ListingCategoryContext:
    source_category_id: int
    bucket_name: str
    category_name: str
    root_url: str
    listing_url: str
    sid: str
    ssid: str
    smid: str
    section_name: str = _DEFAULT_SECTION_NAME
    subsection_name: str = _DEFAULT_SUBSECTION_NAME

    @property
    def ajax_url(self) -> str:
        return urljoin(self.listing_url, "/sebiweb/ajax/home/getnewslistinfo.jsp")

    @property
    def ssidhidden(self) -> str:
        return self.ssid


@dataclass(frozen=True)
class ListingPaginationState:
    next_value: str
    total_pages: int | None
    has_next: bool


@dataclass(frozen=True)
class ParsedListingRow:
    row_index: int
    external_record_id: str | None
    title: str
    order_date: date | None
    detail_url: str | None
    direct_file_url: str | None
    link_url: str | None
    raw_metadata_json: dict[str, str | int | None]


@dataclass(frozen=True)
class ParsedListingPage:
    rows: tuple[ParsedListingRow, ...]
    pagination: ListingPaginationState


@dataclass(frozen=True)
class ListingPageRequest:
    page_number: int
    request_url: str
    referer_url: str | None
    method: Literal["GET", "POST"]
    form_data: dict[str, str] | None = None


@dataclass(frozen=True)
class DetailPageFetchResult:
    request_url: str
    referer_url: str | None
    http_status: int
    content_bytes: bytes
    text: str
    fetched_at: datetime
    content_hash: str


@dataclass(frozen=True)
class ParsedDetailPage:
    title: str | None
    order_date: date | None
    bucket_label: str | None
    attached_file_url: str | None
    raw_metadata_json: dict[str, object]


@dataclass(frozen=True)
class DownloadedFile:
    request_url: str
    response_url: str
    http_status: int
    fetched_at: datetime
    temp_path: Path
    file_size_bytes: int
    sha256: str
    mime_type: str | None
    content_disposition: str | None
    file_name: str | None

    def cleanup(self) -> None:
        self.temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ResolvedOrderFile:
    order_date: str
    title: str
    external_record_id: str
    detail_url: str
    pdf_url: str


@dataclass(frozen=True)
class ManifestRow:
    record_key: str
    bucket_name: str
    order_date: str
    title: str
    external_record_id: str
    detail_url: str
    pdf_url: str
    local_filename: str
    status: str
    error: str
    first_seen_at: str
    last_seen_at: str


@dataclass(frozen=True)
class PageProcessResult:
    discovered_count: int
    downloaded_count: int
    skipped_count: int
    failed_count: int
    page_changed: bool


@dataclass(frozen=True)
class PageWindow:
    start_page: int
    end_page: int


@dataclass(frozen=True)
class CategoryRangeArgument:
    bucket_name: str
    page_window: PageWindow


@dataclass(frozen=True)
class SelectedCategoryRequest:
    category: DiscoveredCategoryLink
    page_window: PageWindow | None


@dataclass(frozen=True)
class CategoryRunResult:
    bucket_name: str
    label: str
    pages_crawled: int
    pages_processed: int
    requested_pages: str
    effective_pages: str
    discovered_count: int
    downloaded_count: int
    skipped_count: int
    failed_count: int
    manifest_path: Path


@dataclass(frozen=True)
class OrdersDownloadRunResult:
    mode: RunMode
    output_dir: Path
    category_results: tuple[CategoryRunResult, ...]

    @property
    def discovered_count(self) -> int:
        return sum(item.discovered_count for item in self.category_results)

    @property
    def downloaded_count(self) -> int:
        return sum(item.downloaded_count for item in self.category_results)

    @property
    def skipped_count(self) -> int:
        return sum(item.skipped_count for item in self.category_results)

    @property
    def failed_count(self) -> int:
        return sum(item.failed_count for item in self.category_results)


def _default_timeout_seconds() -> int:
    raw_value = os.getenv("SEBI_ORDERS_TIMEOUT_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise OrdersDownloadFatalError("SEBI_ORDERS_TIMEOUT_SECONDS must be an integer") from exc
    if parsed < 1:
        raise OrdersDownloadFatalError("SEBI_ORDERS_TIMEOUT_SECONDS must be at least 1")
    return parsed


def _default_user_agent() -> str:
    return os.getenv("SEBI_ORDERS_USER_AGENT", DEFAULT_USER_AGENT)


def _default_output_dir() -> Path:
    raw_value = os.getenv("SEBI_ORDERS_OUTPUT_DIR")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_OUTPUT_DIR
    return Path(raw_value)


def build_examples_text(prog: str) -> str:
    """Return copy-paste usage examples for the standalone script."""

    command = f"python {prog}"
    return (
        "Examples:\n"
        f"  {command} backfill --output-dir ./sebi-orders-pdfs\n"
        f"  {command} backfill --category orders-of-sat\n"
        f"  {command} backfill --category-range orders-of-sat:25\n"
        f"  {command} backfill --category settlement-orders --page-start 3 --page-end 5\n"
        f"  {command} backfill --category-range orders-of-sat:25 "
        "--category-range orders-of-chairperson-members:1-3\n"
        f"  {command} update --category orders-of-sat\n"
        "\n"
        "6:00 PM IST cron example (copy into `crontab -e` later, not installed by this script):\n"
        f"  0 18 * * * cd /path/to/project && {command} backfill --category orders-of-sat "
        ">> ./logs/sebi-orders.log 2>&1\n"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the portable downloader."""

    parser = argparse.ArgumentParser(
        prog="download_orders_portable.py",
        description=(
            "Portable stdlib-only downloader for SEBI Phase 1 order-category PDFs. "
            "Run the `examples` subcommand for copy-paste commands."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    backfill = subparsers.add_parser(
        "backfill",
        help="Crawl reachable pages for each selected category.",
        description="Backfill one, many, or all SEBI Phase 1 order categories.",
    )
    _add_common_arguments(backfill, include_page_window_options=True)

    update = subparsers.add_parser(
        "update",
        help="Incrementally recrawl recent pages for each selected category.",
        description="Incrementally recrawl recent pages for selected categories.",
    )
    _add_common_arguments(update, include_page_window_options=False)

    subparsers.add_parser(
        "examples",
        help="Print copy-paste usage examples, including a cron template.",
        description="Print copy-paste usage examples, including a 6:00 PM IST cron line.",
    )

    return parser


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, include_page_window_options: bool
) -> None:
    parser.add_argument(
        "--category",
        action="append",
        choices=sorted(SUPPORTED_BUCKET_NAMES),
        default=None,
        help="Repeat to crawl only specific category bucket names.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional hard cap on the number of pages to crawl per category.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Root directory that will contain one folder per selected category.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_default_timeout_seconds(),
        help="HTTP timeout in seconds. Defaults to env SEBI_ORDERS_TIMEOUT_SECONDS or 30.",
    )
    parser.add_argument(
        "--user-agent",
        default=_default_user_agent(),
        help="HTTP user agent. Defaults to env SEBI_ORDERS_USER_AGENT or the built-in SEBI agent.",
    )
    if include_page_window_options:
        parser.add_argument(
            "--page-start",
            type=int,
            default=None,
            help="Inclusive 1-based start page applied to selected categories.",
        )
        parser.add_argument(
            "--page-end",
            type=int,
            default=None,
            help="Inclusive 1-based end page applied to selected categories.",
        )
        parser.add_argument(
            "--category-range",
            action="append",
            type=parse_category_range_argument,
            default=None,
            help="Repeatable per-category page window in the form bucket:start-end or bucket:page.",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the portable downloader CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "examples":
        print(build_examples_text(Path(sys.argv[0]).name))
        return 0

    try:
        result = run_download_orders(
            mode=args.mode,
            categories=args.category,
            max_pages=args.max_pages,
            output_dir=args.output_dir,
            page_start=getattr(args, "page_start", None),
            page_end=getattr(args, "page_end", None),
            category_page_windows=build_category_page_windows(
                getattr(args, "category_range", None)
            ),
            timeout_seconds=args.timeout_seconds,
            user_agent=args.user_agent,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except OrdersDownloadFatalError as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1

    for category_result in result.category_results:
        print(
            (
                f"[{category_result.bucket_name}] "
                f"requested_pages={category_result.requested_pages} "
                f"effective_pages={category_result.effective_pages} "
                f"fetched_pages={category_result.pages_crawled} "
                f"processed_pages={category_result.pages_processed} "
                f"discovered={category_result.discovered_count} "
                f"downloaded={category_result.downloaded_count} "
                f"skipped={category_result.skipped_count} "
                f"failed={category_result.failed_count} "
                f"manifest={category_result.manifest_path}"
            ),
            file=sys.stdout,
        )

    print(
        (
            f"Completed {result.mode} run: "
            f"categories={len(result.category_results)} "
            f"discovered={result.discovered_count} "
            f"downloaded={result.downloaded_count} "
            f"skipped={result.skipped_count} "
            f"failed={result.failed_count} "
            f"output_dir={result.output_dir}"
        ),
        file=sys.stdout,
    )
    return 0


def normalize_text(value: str) -> str:
    """Return a lowercase, whitespace-normalized label for matching."""

    normalized = value.replace("&", " and ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
    return " ".join(normalized.lower().split())


def normalize_http_url(base_url: str, candidate: str) -> str | None:
    """Resolve and normalize a candidate HTTP URL against a base URL."""

    stripped_candidate = candidate.strip()
    if not stripped_candidate:
        return None

    lowered_candidate = stripped_candidate.lower()
    if lowered_candidate.startswith(_UNSAFE_SCHEMES):
        return None

    absolute_url = urljoin(base_url, stripped_candidate)
    parts = urlsplit(absolute_url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None

    normalized_path = posixpath.normpath(parts.path or "/")
    if normalized_path == ".":
        normalized_path = "/"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    if parts.path.endswith("/") and not normalized_path.endswith("/"):
        normalized_path = f"{normalized_path}/"

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            normalized_path,
            parts.query,
            "",
        )
    )


class _AnchorLinkParser(HTMLParser):
    """Collect text-bearing anchor tags from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href is None:
            return
        self._active_href = href
        self._active_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._active_href is None:
            return
        self._active_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._active_href is None:
            return
        label = " ".join("".join(self._active_text_parts).split())
        if label:
            self.links.append((self._active_href, label))
        self._active_href = None
        self._active_text_parts = []


def parse_anchor_links(html: str, base_url: str) -> list[ParsedAnchorLink]:
    """Parse anchor tags from HTML and normalize their URLs."""

    parser = _AnchorLinkParser()
    parser.feed(html)

    parsed_links: list[ParsedAnchorLink] = []
    for href, label in parser.links:
        normalized_url = normalize_http_url(base_url=base_url, candidate=href)
        if normalized_url is None:
            continue
        parsed_links.append(
            ParsedAnchorLink(
                label=label,
                href=href,
                url=normalized_url,
                normalized_label=normalize_text(label),
            )
        )
    return parsed_links


class HttpRootPageFetcher:
    """HTTP fetcher for the SEBI orders root page."""

    def __init__(self, *, timeout_seconds: int, user_agent: str) -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._ssl_context = ssl.create_default_context()

    def fetch(self, url: str) -> RootPageFetchResult:
        request = Request(
            url,
            headers={"User-Agent": self._user_agent},
            method="GET",
        )
        try:
            with urlopen(
                request,
                timeout=self._timeout_seconds,
                context=self._ssl_context,
            ) as response:
                response_body = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                html = response_body.decode(charset, errors="replace")
        except (HTTPError, URLError) as exc:
            raise RootCategorySyncError(f"failed to fetch SEBI orders root page: {exc}") from exc

        fetched_at = datetime.now(UTC)
        return RootPageFetchResult(
            url=url,
            html=html,
            fetched_at=fetched_at,
            content_hash=hashlib.sha256(response_body).hexdigest(),
        )


def parse_root_category_links(
    html: str, root_url: str = SEBI_ORDERS_ROOT_URL
) -> list[ParsedAnchorLink]:
    """Parse all text-bearing anchors from the SEBI root orders page."""

    return parse_anchor_links(html=html, base_url=root_url)


def _match_score(target: TargetCategoryDefinition, link: ParsedAnchorLink) -> int:
    score = 0
    normalized_label = link.normalized_label

    if any(normalized_label == alias for alias in target.aliases):
        score += 100
    elif any(alias in normalized_label or normalized_label in alias for alias in target.aliases):
        score += 60

    if target.bucket_name == "orders-of-sat":
        if "orders" in normalized_label and "sat" in f" {normalized_label} ":
            score += 35
    elif target.bucket_name == "orders-of-chairperson-members":
        if "chairperson" in normalized_label and "member" in normalized_label:
            score += 30
        if "order" in normalized_label:
            score += 5
    elif target.bucket_name == "settlement-orders":
        if "settlement" in normalized_label and "order" in normalized_label:
            score += 30
    elif target.bucket_name == "orders-of-aa-under-rti-act":
        if " aa " in f" {normalized_label} " and "rti" in normalized_label:
            score += 35
        if "act" in normalized_label:
            score += 10
    elif target.bucket_name == "orders-of-corporatisation-demutualisation-scheme":
        if "corporatisation" in normalized_label and "demutualisation" in normalized_label:
            score += 35
        if "scheme" in normalized_label:
            score += 10
    elif target.bucket_name == "orders-of-ed-cgm":
        if (
            "order" in normalized_label
            and " ed " in f" {normalized_label} "
            and " cgm " in f" {normalized_label} "
        ):
            score += 30
        if "quasi" in normalized_label and "judicial" in normalized_label:
            score += 20
    elif target.bucket_name == "orders-of-ao":
        if "order" in normalized_label and " ao " in f" {normalized_label} ":
            score += 30
        if "order" in normalized_label and "adjudicating officer" in normalized_label:
            score += 30
    elif target.bucket_name == "orders-of-courts":
        if "orders" in normalized_label and "courts" in normalized_label:
            score += 30
    elif target.bucket_name == "orders-of-special-courts":
        if "orders" in normalized_label and "special courts" in normalized_label:
            score += 35
    elif target.bucket_name == "orders-under-regulation-30a":
        if "regulation 30a" in normalized_label:
            score += 40
        if "intermediaries" in normalized_label and "2008" in normalized_label:
            score += 10

    if "dolisting=yes" in link.url.lower():
        score += 5

    return score


def select_phase1_category_links(
    links: list[ParsedAnchorLink],
) -> dict[str, DiscoveredCategoryLink]:
    """Select the best-matching Phase 1 target links from parsed root anchors."""

    selected: dict[str, DiscoveredCategoryLink] = {}
    best_scores: dict[str, int] = {}

    for link in links:
        for target in PHASE1_TARGET_CATEGORIES:
            score = _match_score(target=target, link=link)
            if score <= 0:
                continue
            current_best = best_scores.get(target.bucket_name, -1)
            if score <= current_best:
                continue
            best_scores[target.bucket_name] = score
            selected[target.bucket_name] = DiscoveredCategoryLink(
                bucket_name=target.bucket_name,
                label=link.label,
                listing_url=link.url,
            )

    return selected


def build_listing_category_context(
    *,
    source_category_id: int,
    bucket_name: str,
    category_name: str,
    root_url: str,
    listing_url: str,
) -> ListingCategoryContext:
    """Build listing crawl context from a listing URL."""

    query = parse_qs(urlsplit(listing_url).query)
    sid = _first_query_value(query, "sid")
    ssid = _first_query_value(query, "ssid")
    smid = _first_query_value(query, "smid")
    if not sid or not ssid or not smid:
        raise ListingCrawlError(
            f"listing URL does not include the expected sid/ssid/smid parameters: {listing_url}"
        )

    return ListingCategoryContext(
        source_category_id=source_category_id,
        bucket_name=bucket_name,
        category_name=category_name,
        root_url=root_url,
        listing_url=listing_url,
        sid=sid,
        ssid=ssid,
        smid=smid,
    )


def build_initial_listing_request(context: ListingCategoryContext) -> ListingPageRequest:
    return ListingPageRequest(
        page_number=1,
        request_url=context.listing_url,
        referer_url=context.root_url,
        method="GET",
    )


def build_paginated_listing_request(
    *,
    context: ListingCategoryContext,
    page_number: int,
    pagination: ListingPaginationState,
) -> ListingPageRequest:
    if page_number < 2:
        raise ListingCrawlError(f"invalid paginated page number: {page_number}")

    return ListingPageRequest(
        page_number=page_number,
        request_url=context.ajax_url,
        referer_url=context.listing_url,
        method="POST",
        form_data={
            "nextValue": pagination.next_value,
            "next": "n",
            "search": "",
            "fromDate": "",
            "toDate": "",
            "fromYear": "",
            "toYear": "",
            "deptId": _DEFAULT_DEPARTMENT_ID,
            "sid": context.sid,
            "ssid": context.ssid,
            "smid": context.smid,
            "ssidhidden": context.ssidhidden,
            "intmid": _DEFAULT_INTMID,
            "sText": context.section_name,
            "ssText": context.subsection_name,
            "smText": context.category_name,
            "doDirect": str(page_number - 1),
        },
    )


class SebiListingSessionClient:
    """Cookie-aware HTTP client for SEBI listing pages."""

    def __init__(self, *, timeout_seconds: int, user_agent: str) -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._ssl_context = ssl.create_default_context()
        cookie_jar = CookieJar()
        self._opener = build_opener(
            HTTPCookieProcessor(cookie_jar),
            HTTPSHandler(context=self._ssl_context),
        )

    def fetch(self, request: ListingPageRequest) -> ListingFetchResult:
        encoded_data = None
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if request.referer_url:
            headers["Referer"] = request.referer_url
        if request.form_data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            encoded_data = _encode_form_data(request.form_data)

        http_request = Request(
            request.request_url,
            data=encoded_data,
            headers=headers,
            method=request.method,
        )

        try:
            with self._opener.open(http_request, timeout=self._timeout_seconds) as response:
                content_bytes = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                http_status = getattr(response, "status", response.getcode())
        except HTTPError as exc:
            exc.read()
            raise ListingCrawlError(
                f"listing fetch failed with HTTP {exc.code} for {request.request_url}"
            ) from exc
        except URLError as exc:
            raise ListingCrawlError(
                f"listing fetch failed for {request.request_url}: {exc}"
            ) from exc

        fetched_at = datetime.now(UTC)
        return ListingFetchResult(
            request_url=request.request_url,
            referer_url=request.referer_url,
            page_number=request.page_number,
            http_status=http_status,
            content_bytes=content_bytes,
            text=content_bytes.decode(charset, errors="replace"),
            fetched_at=fetched_at,
            content_hash=hashlib.sha256(content_bytes).hexdigest(),
        )


class ListingPageFetcher:
    """Fetch initial and deeper listing pages with shared session state."""

    def __init__(self, client: SebiListingSessionClient) -> None:
        self._client = client

    def fetch_initial_page(self, context: ListingCategoryContext) -> ListingFetchResult:
        return self._client.fetch(build_initial_listing_request(context))

    def fetch_page(
        self,
        *,
        context: ListingCategoryContext,
        page_number: int,
        pagination: ListingPaginationState,
    ) -> ListingFetchResult:
        return self._client.fetch(
            build_paginated_listing_request(
                context=context,
                page_number=page_number,
                pagination=pagination,
            )
        )


def parse_listing_page(*, html: str, base_url: str) -> ParsedListingPage:
    """Parse rows and pagination state from a SEBI listing response."""

    listing_markup = extract_listing_markup(html)
    row_parser = _ListingRowParser()
    row_parser.feed(listing_markup)

    rows: list[ParsedListingRow] = []
    for row_index, parsed_row in enumerate(row_parser.rows, start=1):
        normalized_link_url = (
            normalize_http_url(base_url=base_url, candidate=parsed_row.href)
            if parsed_row.href
            else None
        )
        link_kind = classify_listing_link(normalized_link_url)
        title = clean_listing_text(parsed_row.anchor_title or parsed_row.anchor_text)
        if not title:
            continue

        order_date = parse_listing_date(parsed_row.date_text)
        external_record_id = extract_external_record_id(normalized_link_url)
        detail_url = normalized_link_url if link_kind == "detail" else None
        direct_file_url = normalized_link_url if link_kind == "direct_file" else None

        rows.append(
            ParsedListingRow(
                row_index=row_index,
                external_record_id=external_record_id,
                title=title,
                order_date=order_date,
                detail_url=detail_url,
                direct_file_url=direct_file_url,
                link_url=normalized_link_url,
                raw_metadata_json={
                    "row_index": row_index,
                    "raw_date_text": clean_listing_text(parsed_row.date_text),
                    "raw_href": parsed_row.href,
                    "anchor_title": clean_listing_text(parsed_row.anchor_title),
                    "anchor_text": clean_listing_text(parsed_row.anchor_text),
                    "link_kind": link_kind,
                },
            )
        )

    return ParsedListingPage(
        rows=tuple(rows),
        pagination=parse_listing_pagination(listing_markup),
    )


def extract_listing_markup(html: str) -> str:
    listing_markup, _, _ = html.partition(_AJAX_SEPARATOR)
    return listing_markup.strip()


def parse_listing_pagination(html: str) -> ListingPaginationState:
    hidden_inputs = {
        match.group("name"): match.group("value") for match in _HIDDEN_INPUT_RE.finditer(html)
    }
    next_value = hidden_inputs.get("nextValue", "1")
    total_pages = _parse_total_pages(hidden_inputs, html)
    has_next = bool(_NEXT_LINK_RE.search(html))
    return ListingPaginationState(
        next_value=next_value,
        total_pages=total_pages,
        has_next=has_next,
    )


def parse_listing_date(value: str) -> date | None:
    cleaned = clean_listing_text(value)
    if not cleaned:
        return None
    for format_string in (
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(cleaned, format_string).date()
        except ValueError:
            continue
    return None


def classify_listing_link(url: str | None) -> Literal["detail", "direct_file", "unknown"]:
    if url is None:
        return "unknown"
    normalized_path = urlsplit(url).path.lower()
    if normalized_path.endswith(_FILE_SUFFIXES):
        return "direct_file"
    return "detail"


def extract_external_record_id(url: str | None) -> str | None:
    if url is None:
        return None

    path = urlsplit(url).path
    path_match = _EXTERNAL_ID_PATH_RE.search(path)
    if path_match is not None:
        return path_match.group("external_id")

    query = parse_qs(urlsplit(url).query)
    for key in ("id", "newsid", "recordid", "entryid"):
        value = _first_query_value(query, key)
        if value:
            return value
    return None


def build_source_record_key(
    *,
    bucket_name: str,
    order_date: date | None,
    title: str,
    link_url: str | None,
    external_record_id: str | None,
) -> str:
    if external_record_id:
        return f"external:{external_record_id}"

    normalized_order_date = order_date.isoformat() if order_date is not None else ""
    normalized_title = normalize_text(title)
    normalized_link = link_url or ""
    digest = hashlib.sha256(
        "\x1f".join(
            [
                bucket_name,
                normalized_order_date,
                normalized_title,
                normalized_link,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"derived:{digest}"


def clean_listing_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def _parse_total_pages(hidden_inputs: dict[str, str], html: str) -> int | None:
    total_pages_value = hidden_inputs.get("totalpage")
    if total_pages_value and total_pages_value.isdigit():
        return int(total_pages_value)
    last_page_match = _LAST_PAGE_RE.search(html)
    if last_page_match is not None:
        return int(last_page_match.group("value")) + 1
    return None


def _encode_form_data(form_data: dict[str, str]) -> bytes:
    from urllib.parse import urlencode

    return urlencode(form_data).encode("utf-8")


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


@dataclass(frozen=True)
class _RawListingRow:
    date_text: str
    href: str | None
    anchor_title: str | None
    anchor_text: str


class _ListingRowParser(HTMLParser):
    """Collect listing rows from the SEBI listing table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_RawListingRow] = []
        self._inside_listing_table = False
        self._inside_row = False
        self._inside_cell = False
        self._inside_anchor = False
        self._current_cell_parts: list[str] = []
        self._current_cells: list[str] = []
        self._current_href: str | None = None
        self._current_anchor_title: str | None = None
        self._current_anchor_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        attributes = dict(attrs)

        if normalized_tag == "table" and attributes.get("id") == "sample_1":
            self._inside_listing_table = True
            return
        if not self._inside_listing_table:
            return
        if normalized_tag == "tr":
            self._inside_row = True
            self._current_cells = []
            self._current_href = None
            self._current_anchor_title = None
            self._current_anchor_parts = []
            return
        if not self._inside_row:
            return
        if normalized_tag == "td":
            self._inside_cell = True
            self._current_cell_parts = []
            return
        if normalized_tag == "a":
            self._inside_anchor = True
            self._current_href = attributes.get("href")
            self._current_anchor_title = attributes.get("title")
            self._current_anchor_parts = []

    def handle_data(self, data: str) -> None:
        if self._inside_cell:
            self._current_cell_parts.append(data)
        if self._inside_anchor:
            self._current_anchor_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()

        if normalized_tag == "table" and self._inside_listing_table:
            self._inside_listing_table = False
            return
        if not self._inside_listing_table:
            return
        if normalized_tag == "a":
            self._inside_anchor = False
            return
        if normalized_tag == "td" and self._inside_cell:
            self._inside_cell = False
            self._current_cells.append(clean_listing_text("".join(self._current_cell_parts)))
            self._current_cell_parts = []
            return
        if normalized_tag == "tr" and self._inside_row:
            self._inside_row = False
            if self._inside_cell:
                self._inside_cell = False
                self._current_cells.append(clean_listing_text("".join(self._current_cell_parts)))
                self._current_cell_parts = []
            if len(self._current_cells) < 2:
                return
            self.rows.append(
                _RawListingRow(
                    date_text=self._current_cells[0],
                    href=self._current_href,
                    anchor_title=self._current_anchor_title,
                    anchor_text=clean_listing_text("".join(self._current_anchor_parts)),
                )
            )


class SebiDetailPageClient:
    """HTTP client for SEBI detail pages."""

    def __init__(self, *, timeout_seconds: int, user_agent: str) -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._ssl_context = ssl.create_default_context()
        self._opener = build_opener(HTTPSHandler(context=self._ssl_context))

    def fetch(self, *, url: str, referer_url: str | None = None) -> DetailPageFetchResult:
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if referer_url:
            headers["Referer"] = referer_url
        request = Request(url, headers=headers, method="GET")
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                content_bytes = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                http_status = getattr(response, "status", response.getcode())
        except HTTPError as exc:
            exc.read()
            raise DetailPageFetchError(
                f"detail page fetch failed with HTTP {exc.code} for {url}"
            ) from exc
        except URLError as exc:
            raise DetailPageFetchError(f"detail page fetch failed for {url}: {exc}") from exc

        fetched_at = datetime.now(UTC)
        return DetailPageFetchResult(
            request_url=url,
            referer_url=referer_url,
            http_status=http_status,
            content_bytes=content_bytes,
            text=content_bytes.decode(charset, errors="replace"),
            fetched_at=fetched_at,
            content_hash=hashlib.sha256(content_bytes).hexdigest(),
        )


def normalize_detail_page_url(*, base_url: str, candidate: str) -> str | None:
    return normalize_http_url(base_url=base_url, candidate=candidate)


def normalize_file_url(*, base_url: str, candidate: str) -> str | None:
    normalized_url = normalize_http_url(base_url=base_url, candidate=candidate)
    if normalized_url is None:
        return None
    nested_file_url = _extract_nested_file_url(normalized_url)
    if nested_file_url is not None:
        return nested_file_url
    if _is_direct_file_url(normalized_url):
        return normalized_url
    return None


def parse_detail_page(*, html: str, base_url: str) -> ParsedDetailPage:
    parser = _DetailPageParser()
    parser.feed(html)

    title = _first_non_empty(
        parser.heading_text,
        _clean_page_title(parser.meta_title),
        _clean_page_title(parser.page_title),
    )
    order_date = _first_parsed_date(parser.date_candidates)
    bucket_label = clean_listing_text(parser.bucket_label) or None
    attachment_candidates = _collect_attachment_candidates(parser=parser, base_url=base_url)
    attached_file_url = attachment_candidates[0][0] if attachment_candidates else None

    return ParsedDetailPage(
        title=title,
        order_date=order_date,
        bucket_label=bucket_label,
        attached_file_url=attached_file_url,
        raw_metadata_json={
            "page_title": parser.page_title,
            "meta_title": parser.meta_title,
            "meta_keywords": parser.meta_keywords,
            "bucket_label": bucket_label,
            "date_candidates": list(parser.date_candidates),
            "attachment_candidates": [item[0] for item in attachment_candidates],
            "attachment_source": attachment_candidates[0][1] if attachment_candidates else None,
        },
    )


def _collect_attachment_candidates(
    *, parser: _DetailPageParser, base_url: str
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    for iframe_url in parser.iframe_sources:
        normalized_url = normalize_file_url(base_url=base_url, candidate=iframe_url)
        if normalized_url is None or normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        candidates.append((normalized_url, "iframe"))

    for anchor_url in parser.anchor_hrefs:
        normalized_url = normalize_file_url(base_url=base_url, candidate=anchor_url)
        if normalized_url is None or normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        candidates.append((normalized_url, "anchor"))

    return candidates


def _extract_nested_file_url(candidate_url: str) -> str | None:
    pending_url = candidate_url
    seen_urls: set[str] = set()

    while pending_url not in seen_urls:
        seen_urls.add(pending_url)
        query = parse_qs(urlsplit(pending_url).query)
        nested_candidate: str | None = None

        for key in _FILE_QUERY_KEYS:
            values = query.get(key)
            if not values:
                continue
            nested_candidate = normalize_http_url(base_url=pending_url, candidate=values[0])
            if nested_candidate is None:
                continue
            if _is_direct_file_url(nested_candidate):
                return nested_candidate
            break

        if nested_candidate is None:
            break
        pending_url = nested_candidate

    return None


def _is_direct_file_url(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(_FILE_SUFFIXES)


def _first_parsed_date(values: tuple[str, ...]) -> date | None:
    for value in values:
        parsed = parse_listing_date(value)
        if parsed is not None:
            return parsed
    return None


def _clean_page_title(value: str | None) -> str | None:
    cleaned = clean_listing_text(value)
    if not cleaned:
        return None
    if cleaned.lower().startswith("sebi | "):
        cleaned = cleaned[7:].strip()
    return cleaned or None


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        cleaned = clean_listing_text(value)
        if cleaned:
            return cleaned
    return None


class _DetailPageParser(HTMLParser):
    """Extract metadata and attachment candidates from detail HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page_title: str | None = None
        self.meta_title: str | None = None
        self.meta_keywords: str | None = None
        self.heading_text: str | None = None
        self.bucket_label: str | None = None
        self.date_candidates: tuple[str, ...] = ()
        self.iframe_sources: list[str] = []
        self.anchor_hrefs: list[str] = []
        self._collected_date_candidates: list[str] = []
        self._inside_title = False
        self._inside_h1 = False
        self._inside_h5 = False
        self._inside_bucket_label = False
        self._title_parts: list[str] = []
        self._h1_parts: list[str] = []
        self._h5_parts: list[str] = []
        self._bucket_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        attributes = {key.lower(): value for key, value in attrs if value is not None}

        if normalized_tag == "title":
            self._inside_title = True
            self._title_parts = []
            return
        if normalized_tag == "meta":
            self._handle_meta(attributes)
            return
        if normalized_tag == "h1":
            self._inside_h1 = True
            self._h1_parts = []
            return
        if normalized_tag == "h5":
            self._inside_h5 = True
            self._h5_parts = []
            return
        if normalized_tag == "li" and _has_class(attributes.get("class"), "active_page"):
            self._inside_bucket_label = True
            self._bucket_parts = []
            return
        if normalized_tag == "iframe":
            src = attributes.get("src")
            if src:
                self.iframe_sources.append(src)
            return
        if normalized_tag == "a":
            href = attributes.get("href")
            if href:
                self.anchor_hrefs.append(href)

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self._title_parts.append(data)
        if self._inside_h1:
            self._h1_parts.append(data)
        if self._inside_h5:
            self._h5_parts.append(data)
        if self._inside_bucket_label:
            self._bucket_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()

        if normalized_tag == "title" and self._inside_title:
            self._inside_title = False
            self.page_title = clean_listing_text("".join(self._title_parts)) or None
            self._title_parts = []
            return
        if normalized_tag == "h1" and self._inside_h1:
            self._inside_h1 = False
            heading_text = clean_listing_text("".join(self._h1_parts))
            if heading_text and self.heading_text is None:
                self.heading_text = heading_text
            self._h1_parts = []
            return
        if normalized_tag == "h5" and self._inside_h5:
            self._inside_h5 = False
            h5_text = clean_listing_text("".join(self._h5_parts))
            if h5_text:
                self._collected_date_candidates.append(h5_text)
                self.date_candidates = tuple(self._collected_date_candidates)
            self._h5_parts = []
            return
        if normalized_tag == "li" and self._inside_bucket_label:
            self._inside_bucket_label = False
            bucket_text = clean_listing_text("".join(self._bucket_parts))
            if bucket_text and self.bucket_label is None:
                self.bucket_label = bucket_text
            self._bucket_parts = []

    def _handle_meta(self, attributes: dict[str, str]) -> None:
        meta_name = (attributes.get("name") or attributes.get("property") or "").lower()
        meta_content = clean_listing_text(attributes.get("content"))
        if not meta_name or not meta_content:
            return
        if meta_name in {"title", "og:title"} and self.meta_title is None:
            self.meta_title = meta_content
        elif meta_name == "keywords" and self.meta_keywords is None:
            self.meta_keywords = meta_content


def _has_class(class_value: str | None, expected: str) -> bool:
    if not class_value:
        return False
    return expected in {part.strip() for part in class_value.split()}


class HttpFileDownloader:
    """HTTP downloader that streams files to disk with retry handling."""

    def __init__(
        self,
        *,
        timeout_seconds: int,
        user_agent: str,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
        opener: Any | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        if opener is not None:
            self._opener = opener
            return
        ssl_context = ssl.create_default_context()
        self._opener = build_opener(HTTPSHandler(context=ssl_context))

    def fetch(self, *, url: str, referer_url: str | None = None) -> DownloadedFile:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._fetch_once(url=url, referer_url=referer_url)
            except HTTPError as exc:
                exc.read()
                if attempt < self._max_attempts and _is_retryable_http_status(exc.code):
                    time.sleep(self._retry_backoff_seconds * attempt)
                    continue
                reason_code = "forbidden" if exc.code == 403 else "http_error"
                raise FileDownloadError(
                    f"file download failed with HTTP {exc.code} for {url}",
                    reason_code=reason_code,
                ) from exc
            except URLError as exc:
                if attempt < self._max_attempts:
                    time.sleep(self._retry_backoff_seconds * attempt)
                    continue
                raise FileDownloadError(
                    f"file download failed for {url}: {exc}",
                    reason_code="timeout" if "timed out" in str(exc).lower() else "network_error",
                ) from exc

        raise FileDownloadError(
            f"file download exhausted retries for {url}",
            reason_code="network_error",
        )

    def _fetch_once(self, *, url: str, referer_url: str | None = None) -> DownloadedFile:
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
        }
        if referer_url:
            headers["Referer"] = referer_url

        request = Request(url, headers=headers, method="GET")
        temp_path: Path | None = None

        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                named_temp_file = tempfile.NamedTemporaryFile(
                    prefix="sebi-order-file-",
                    suffix=".download",
                    delete=False,
                )
                temp_path = Path(named_temp_file.name)
                sha256 = hashlib.sha256()
                file_size_bytes = 0

                with named_temp_file:
                    while True:
                        chunk = response.read(_CHUNK_SIZE_BYTES)
                        if not chunk:
                            break
                        named_temp_file.write(chunk)
                        sha256.update(chunk)
                        file_size_bytes += len(chunk)

                response_url = _response_url(response=response, fallback_url=url)
                content_disposition = response.headers.get("Content-Disposition")
                file_name = infer_download_file_name(
                    url=response_url,
                    content_disposition=content_disposition,
                )
                mime_type = infer_download_mime_type(
                    content_type=response.headers.get("Content-Type"),
                    file_name=file_name,
                    url=response_url,
                )

                return DownloadedFile(
                    request_url=url,
                    response_url=response_url,
                    http_status=getattr(response, "status", response.getcode()),
                    fetched_at=datetime.now(UTC),
                    temp_path=temp_path,
                    file_size_bytes=file_size_bytes,
                    sha256=sha256.hexdigest(),
                    mime_type=mime_type,
                    content_disposition=content_disposition,
                    file_name=file_name,
                )
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise


def infer_download_file_name(*, url: str, content_disposition: str | None) -> str | None:
    header_file_name = infer_file_name_from_content_disposition(content_disposition)
    if header_file_name:
        return header_file_name
    path_file_name = PurePosixPath(unquote(urlsplit(url).path)).name
    return path_file_name or None


def infer_file_name_from_content_disposition(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None

    filename_star_match = re.search(
        r"filename\*\s*=\s*[^']*''(?P<value>[^;]+)",
        content_disposition,
        flags=re.IGNORECASE,
    )
    if filename_star_match:
        return _basename(unquote(filename_star_match.group("value").strip().strip('"')))

    filename_match = re.search(
        r'filename\s*=\s*"(?P<quoted>[^"]+)"|filename\s*=\s*(?P<bare>[^;]+)',
        content_disposition,
        flags=re.IGNORECASE,
    )
    if filename_match:
        file_name = filename_match.group("quoted") or filename_match.group("bare")
        return _basename(file_name.strip().strip('"'))

    return None


def infer_download_mime_type(
    *,
    content_type: str | None,
    file_name: str | None,
    url: str,
) -> str | None:
    normalized_content_type = normalize_content_type(content_type)
    if normalized_content_type is not None:
        return normalized_content_type
    for candidate in (file_name, url):
        if not candidate:
            continue
        inferred, _ = mimetypes.guess_type(candidate)
        if inferred is not None:
            return inferred
    return None


def normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()
    return normalized or None


def _basename(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).name


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def _response_url(*, response: Any, fallback_url: str) -> str:
    try:
        response_url = response.geturl()
    except AttributeError:
        return fallback_url
    return response_url or fallback_url


def parse_category_range_argument(value: str) -> CategoryRangeArgument:
    """Parse one CLI category page window."""

    bucket_name, separator, page_spec = value.partition(":")
    if not separator or not bucket_name or not page_spec:
        raise argparse.ArgumentTypeError(
            "category ranges must look like bucket:start-end or bucket:page"
        )
    if bucket_name not in SUPPORTED_BUCKET_NAMES:
        raise argparse.ArgumentTypeError(f"unknown category bucket: {bucket_name}")

    page_spec = page_spec.strip()
    if "-" in page_spec:
        start_text, end_text = page_spec.split("-", maxsplit=1)
    else:
        start_text = page_spec
        end_text = page_spec

    try:
        start_page = int(start_text)
        end_page = int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid page range for {bucket_name}: {page_spec}"
        ) from exc

    return CategoryRangeArgument(
        bucket_name=bucket_name,
        page_window=validate_page_window(
            start_page=start_page,
            end_page=end_page,
            option_label=f"category range {bucket_name}",
            exception_type=argparse.ArgumentTypeError,
        ),
    )


def build_category_page_windows(
    category_ranges: Sequence[CategoryRangeArgument] | None,
) -> dict[str, PageWindow]:
    if not category_ranges:
        return {}
    page_windows: dict[str, PageWindow] = {}
    for category_range in category_ranges:
        if category_range.bucket_name in page_windows:
            raise OrdersDownloadFatalError(
                f"duplicate page range provided for {category_range.bucket_name}"
            )
        page_windows[category_range.bucket_name] = category_range.page_window
    return page_windows


def resolve_global_page_window(
    *,
    page_start: int | None,
    page_end: int | None,
) -> PageWindow | None:
    if page_start is None and page_end is None:
        return None
    if page_start is None or page_end is None:
        raise OrdersDownloadFatalError("page_start and page_end must be provided together")
    return validate_page_window(
        start_page=page_start,
        end_page=page_end,
        option_label="global page range",
        exception_type=OrdersDownloadFatalError,
    )


def validate_page_window(
    *,
    start_page: int,
    end_page: int,
    option_label: str,
    exception_type: type[Exception],
) -> PageWindow:
    if start_page < 1:
        raise exception_type(f"{option_label} start page must be at least 1")
    if end_page < 1:
        raise exception_type(f"{option_label} end page must be at least 1")
    if end_page < start_page:
        raise exception_type(f"{option_label} end page must be greater than or equal to start page")
    return PageWindow(start_page=start_page, end_page=end_page)


def validate_category_page_windows(
    category_page_windows: dict[str, PageWindow],
) -> dict[str, PageWindow]:
    validated_windows: dict[str, PageWindow] = {}
    for bucket_name, page_window in category_page_windows.items():
        if bucket_name not in SUPPORTED_BUCKET_NAMES:
            raise OrdersDownloadFatalError(f"unknown category bucket: {bucket_name}")
        validated_windows[bucket_name] = validate_page_window(
            start_page=page_window.start_page,
            end_page=page_window.end_page,
            option_label=f"category range {bucket_name}",
            exception_type=OrdersDownloadFatalError,
        )
    return validated_windows


def run_download_orders(
    *,
    mode: RunMode,
    categories: Sequence[str] | None = None,
    max_pages: int | None = None,
    output_dir: Path | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    category_page_windows: dict[str, PageWindow] | None = None,
    category_listing_url_overrides: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    user_agent: str | None = None,
    root_fetcher: RootPageFetcher | None = None,
    listing_fetcher: ListingPageFetcherProtocol | None = None,
    detail_fetcher: DetailPageFetcher | None = None,
    downloader: FileDownloader | None = None,
    update_recent_pages: int = DEFAULT_UPDATE_RECENT_PAGES,
    unchanged_page_threshold: int = DEFAULT_UNCHANGED_PAGE_THRESHOLD,
    now_factory: Callable[[], datetime] | None = None,
    stdout: object | None = None,
    stderr: object | None = None,
) -> OrdersDownloadRunResult:
    """Execute a multi-category backfill or incremental update run."""

    del stdout
    if max_pages is not None and max_pages < 1:
        raise OrdersDownloadFatalError("max_pages must be at least 1")
    if update_recent_pages < 1:
        raise OrdersDownloadFatalError("update_recent_pages must be at least 1")
    if unchanged_page_threshold < 1:
        raise OrdersDownloadFatalError("unchanged_page_threshold must be at least 1")

    resolved_global_page_window = resolve_global_page_window(
        page_start=page_start,
        page_end=page_end,
    )
    resolved_category_page_windows = validate_category_page_windows(category_page_windows or {})
    if mode == "update" and (
        resolved_global_page_window is not None or resolved_category_page_windows
    ):
        raise OrdersDownloadFatalError("page-range options are only supported for backfill runs")
    if max_pages is not None and (
        resolved_global_page_window is not None or resolved_category_page_windows
    ):
        raise OrdersDownloadFatalError("max_pages cannot be combined with page-range options")

    resolved_timeout_seconds = timeout_seconds or _default_timeout_seconds()
    if resolved_timeout_seconds < 1:
        raise OrdersDownloadFatalError("timeout_seconds must be at least 1")
    resolved_user_agent = user_agent or _default_user_agent()
    resolved_output_dir = output_dir or _default_output_dir()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_now_factory = now_factory or (lambda: datetime.now(UTC))

    root_page_fetcher = root_fetcher or HttpRootPageFetcher(
        timeout_seconds=resolved_timeout_seconds,
        user_agent=resolved_user_agent,
    )
    resolved_listing_fetcher = listing_fetcher or ListingPageFetcher(
        SebiListingSessionClient(
            timeout_seconds=resolved_timeout_seconds,
            user_agent=resolved_user_agent,
        )
    )
    resolved_detail_fetcher = detail_fetcher or SebiDetailPageClient(
        timeout_seconds=resolved_timeout_seconds,
        user_agent=resolved_user_agent,
    )
    resolved_downloader = downloader or HttpFileDownloader(
        timeout_seconds=resolved_timeout_seconds,
        user_agent=resolved_user_agent,
    )

    discovered_categories = discover_phase1_categories(root_fetcher=root_page_fetcher)
    if category_listing_url_overrides:
        discovered_categories = apply_category_listing_url_overrides(
            discovered_categories=discovered_categories,
            category_listing_url_overrides=category_listing_url_overrides,
        )
    selected_category_requests = select_requested_category_requests(
        discovered_categories=discovered_categories,
        requested_categories=categories,
        global_page_window=resolved_global_page_window,
        category_page_windows=resolved_category_page_windows,
    )

    category_results: list[CategoryRunResult] = []
    for selected_request in selected_category_requests:
        category_results.append(
            run_category_download(
                category=selected_request.category,
                mode=mode,
                output_dir=resolved_output_dir,
                max_pages=max_pages,
                requested_page_window=selected_request.page_window,
                listing_fetcher=resolved_listing_fetcher,
                detail_fetcher=resolved_detail_fetcher,
                downloader=resolved_downloader,
                update_recent_pages=update_recent_pages,
                unchanged_page_threshold=unchanged_page_threshold,
                now_factory=resolved_now_factory,
                stderr=stderr,
            )
        )

    return OrdersDownloadRunResult(
        mode=mode,
        output_dir=resolved_output_dir,
        category_results=tuple(category_results),
    )


def discover_phase1_categories(
    *,
    root_fetcher: RootPageFetcher,
    root_url: str = SEBI_ORDERS_ROOT_URL,
) -> dict[str, DiscoveredCategoryLink]:
    try:
        root_page = root_fetcher.fetch(root_url)
    except RootCategorySyncError as exc:
        raise OrdersDownloadFatalError(str(exc)) from exc

    parsed_links = parse_root_category_links(html=root_page.html, root_url=root_page.url)
    return select_phase1_category_links(parsed_links)


def apply_category_listing_url_overrides(
    *,
    discovered_categories: dict[str, DiscoveredCategoryLink],
    category_listing_url_overrides: dict[str, str],
) -> dict[str, DiscoveredCategoryLink]:
    resolved_categories = dict(discovered_categories)
    for bucket_name, listing_url in category_listing_url_overrides.items():
        existing_category = resolved_categories.get(bucket_name)
        if existing_category is None:
            raise OrdersDownloadFatalError(
                f"cannot override listing URL for undiscovered category: {bucket_name}"
            )
        normalized_listing_url = normalize_http_url(
            base_url=SEBI_ORDERS_ROOT_URL,
            candidate=listing_url,
        )
        if normalized_listing_url is None:
            raise OrdersDownloadFatalError(
                f"invalid listing URL override for {bucket_name}: {listing_url}"
            )
        resolved_categories[bucket_name] = DiscoveredCategoryLink(
            bucket_name=existing_category.bucket_name,
            label=existing_category.label,
            listing_url=normalized_listing_url,
        )
    return resolved_categories


def select_requested_categories(
    *,
    discovered_categories: dict[str, DiscoveredCategoryLink],
    requested_categories: Sequence[str] | None,
) -> list[DiscoveredCategoryLink]:
    if requested_categories is None:
        missing = [
            bucket_name
            for bucket_name in SUPPORTED_BUCKET_NAMES
            if bucket_name not in discovered_categories
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise OrdersDownloadFatalError(
                f"could not discover all Phase 1 categories from the SEBI root page: {missing_text}"
            )
        return [discovered_categories[bucket_name] for bucket_name in SUPPORTED_BUCKET_NAMES]

    deduped_requested: list[str] = []
    seen_categories: set[str] = set()
    for bucket_name in requested_categories:
        if bucket_name in seen_categories:
            continue
        seen_categories.add(bucket_name)
        deduped_requested.append(bucket_name)

    missing = [
        bucket_name for bucket_name in deduped_requested if bucket_name not in discovered_categories
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise OrdersDownloadFatalError(
            f"requested categories were not discovered from the SEBI root page: {missing_text}"
        )
    return [discovered_categories[bucket_name] for bucket_name in deduped_requested]


def select_requested_category_requests(
    *,
    discovered_categories: dict[str, DiscoveredCategoryLink],
    requested_categories: Sequence[str] | None,
    global_page_window: PageWindow | None,
    category_page_windows: dict[str, PageWindow],
) -> list[SelectedCategoryRequest]:
    requested_bucket_names: list[str] | None
    if requested_categories is not None:
        requested_bucket_names = list(requested_categories)
        for bucket_name in category_page_windows:
            if bucket_name not in requested_bucket_names:
                requested_bucket_names.append(bucket_name)
    elif global_page_window is not None:
        requested_bucket_names = None
    elif category_page_windows:
        requested_bucket_names = list(category_page_windows)
    else:
        requested_bucket_names = None

    selected_categories = select_requested_categories(
        discovered_categories=discovered_categories,
        requested_categories=requested_bucket_names,
    )
    return [
        SelectedCategoryRequest(
            category=category,
            page_window=category_page_windows.get(category.bucket_name, global_page_window),
        )
        for category in selected_categories
    ]


def run_category_download(
    *,
    category: DiscoveredCategoryLink,
    mode: RunMode,
    output_dir: Path,
    max_pages: int | None,
    requested_page_window: PageWindow | None,
    listing_fetcher: ListingPageFetcherProtocol,
    detail_fetcher: DetailPageFetcher,
    downloader: FileDownloader,
    update_recent_pages: int,
    unchanged_page_threshold: int,
    now_factory: Callable[[], datetime],
    stderr: object | None,
) -> CategoryRunResult:
    category_dir = output_dir / category.bucket_name
    category_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = category_dir / MANIFEST_FILE_NAME
    manifest_rows = load_manifest(manifest_path)

    context = build_listing_category_context(
        source_category_id=0,
        bucket_name=category.bucket_name,
        category_name=category.label,
        root_url=SEBI_ORDERS_ROOT_URL,
        listing_url=category.listing_url,
    )

    pages_crawled = 0
    pages_processed = 0
    discovered_count = 0
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    consecutive_unchanged_pages = 0
    first_processed_page: int | None = None
    last_processed_page: int | None = None
    minimum_update_pages = _minimum_update_pages(
        max_pages=max_pages,
        update_recent_pages=update_recent_pages,
    )
    requested_pages = format_requested_pages(
        mode=mode,
        max_pages=max_pages,
        requested_page_window=requested_page_window,
    )

    page_number = 1
    pagination: ListingPaginationState | None = None
    while True:
        fetched_page = fetch_listing_page(
            fetcher=listing_fetcher,
            context=context,
            page_number=page_number,
            pagination=pagination,
        )
        pages_crawled += 1
        parsed_page = parse_listing_page(html=fetched_page.text, base_url=context.listing_url)
        pagination = parsed_page.pagination

        if requested_page_window is not None:
            total_pages = pagination.total_pages
            if (
                total_pages is not None
                and page_number == 1
                and requested_page_window.start_page > total_pages
            ):
                raise OrdersDownloadFatalError(
                    f"requested start page {requested_page_window.start_page} for "
                    f"{category.bucket_name} exceeds live page count {total_pages}"
                )
            if page_number < requested_page_window.start_page:
                if not pagination.has_next:
                    raise OrdersDownloadFatalError(
                        f"requested start page {requested_page_window.start_page} for "
                        f"{category.bucket_name} exceeds live page count {page_number}"
                    )
                page_number += 1
                continue

        page_result = process_listing_page(
            category=category,
            parsed_page=parsed_page,
            category_dir=category_dir,
            manifest_rows=manifest_rows,
            detail_fetcher=detail_fetcher,
            downloader=downloader,
            referer_url=context.listing_url,
            now_factory=now_factory,
            stderr=stderr,
        )
        pages_processed += 1
        first_processed_page = page_number if first_processed_page is None else first_processed_page
        last_processed_page = page_number
        discovered_count += page_result.discovered_count
        downloaded_count += page_result.downloaded_count
        skipped_count += page_result.skipped_count
        failed_count += page_result.failed_count

        if requested_page_window is not None:
            if page_number >= requested_page_window.end_page:
                break
            if not pagination.has_next:
                break
            page_number += 1
            continue

        if max_pages is not None and page_number >= max_pages:
            break
        if not pagination.has_next:
            break
        if mode == "update":
            if page_result.page_changed:
                consecutive_unchanged_pages = 0
            else:
                consecutive_unchanged_pages += 1
            if (
                pages_crawled >= minimum_update_pages
                and consecutive_unchanged_pages >= unchanged_page_threshold
            ):
                break
        page_number += 1

    write_manifest(manifest_path=manifest_path, manifest_rows=manifest_rows)
    return CategoryRunResult(
        bucket_name=category.bucket_name,
        label=category.label,
        pages_crawled=pages_crawled,
        pages_processed=pages_processed,
        requested_pages=requested_pages,
        effective_pages=format_effective_pages(
            first_processed_page=first_processed_page,
            last_processed_page=last_processed_page,
        ),
        discovered_count=discovered_count,
        downloaded_count=downloaded_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        manifest_path=manifest_path,
    )


def _minimum_update_pages(*, max_pages: int | None, update_recent_pages: int) -> int:
    if max_pages is None:
        return update_recent_pages
    return min(max_pages, update_recent_pages)


def format_requested_pages(
    *,
    mode: RunMode,
    max_pages: int | None,
    requested_page_window: PageWindow | None,
) -> str:
    if requested_page_window is not None:
        return format_page_window(requested_page_window)
    if mode == "update":
        return "incremental"
    if max_pages is not None:
        return format_page_window(PageWindow(start_page=1, end_page=max_pages))
    return "all"


def format_effective_pages(
    *,
    first_processed_page: int | None,
    last_processed_page: int | None,
) -> str:
    if first_processed_page is None or last_processed_page is None:
        return "none"
    return format_page_window(
        PageWindow(start_page=first_processed_page, end_page=last_processed_page)
    )


def format_page_window(page_window: PageWindow) -> str:
    if page_window.start_page == page_window.end_page:
        return str(page_window.start_page)
    return f"{page_window.start_page}-{page_window.end_page}"


def fetch_listing_page(
    *,
    fetcher: ListingPageFetcherProtocol,
    context: ListingCategoryContext,
    page_number: int,
    pagination: ListingPaginationState | None,
) -> ListingFetchResult:
    try:
        if page_number == 1:
            return fetcher.fetch_initial_page(context)
        if pagination is None:
            raise OrdersDownloadFatalError(
                f"missing pagination state before fetching page {page_number}"
            )
        return fetcher.fetch_page(
            context=context,
            page_number=page_number,
            pagination=pagination,
        )
    except Exception as exc:
        raise OrdersDownloadFatalError(
            f"failed to fetch page {page_number} for {context.bucket_name}: {exc}"
        ) from exc


def process_listing_page(
    *,
    category: DiscoveredCategoryLink,
    parsed_page: ParsedListingPage,
    category_dir: Path,
    manifest_rows: dict[str, ManifestRow],
    detail_fetcher: DetailPageFetcher,
    downloader: FileDownloader,
    referer_url: str | None,
    now_factory: Callable[[], datetime],
    stderr: object | None,
) -> PageProcessResult:
    page_changed = False
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    for row in parsed_page.rows:
        record_key = build_record_key(category.bucket_name, row)
        previous_row = manifest_rows.get(record_key)

        try:
            resolved = resolve_listing_row(
                row=row,
                detail_fetcher=detail_fetcher,
                referer_url=referer_url,
            )
            desired_filename = build_output_file_name(
                record_key=record_key,
                resolved=resolved,
            )
            next_manifest_row, action = upsert_successful_row(
                previous_row=previous_row,
                record_key=record_key,
                bucket_name=category.bucket_name,
                resolved=resolved,
                desired_filename=desired_filename,
                category_dir=category_dir,
                downloader=downloader,
                now_factory=now_factory,
            )
            manifest_rows[record_key] = next_manifest_row
            if action == "downloaded":
                downloaded_count += 1
            else:
                skipped_count += 1
            page_changed |= materially_changed(previous_row, next_manifest_row)
        except Exception as exc:
            failed_count += 1
            failed_row = build_failed_manifest_row(
                previous_row=previous_row,
                record_key=record_key,
                bucket_name=category.bucket_name,
                row=row,
                error=str(exc),
                now_factory=now_factory,
            )
            manifest_rows[record_key] = failed_row
            page_changed |= materially_changed(previous_row, failed_row)
            if stderr is not None:
                print(f"Failed: [{category.bucket_name}] {row.title}: {exc}", file=stderr)

    return PageProcessResult(
        discovered_count=len(parsed_page.rows),
        downloaded_count=downloaded_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        page_changed=page_changed,
    )


def build_record_key(bucket_name: str, row: ParsedListingRow) -> str:
    return build_source_record_key(
        bucket_name=bucket_name,
        order_date=row.order_date,
        title=row.title,
        link_url=row.link_url or row.detail_url or row.direct_file_url,
        external_record_id=row.external_record_id,
    )


def resolve_listing_row(
    *,
    row: ParsedListingRow,
    detail_fetcher: DetailPageFetcher,
    referer_url: str | None,
) -> ResolvedOrderFile:
    if row.direct_file_url is not None:
        pdf_url = normalize_file_url(
            base_url=row.link_url or row.direct_file_url,
            candidate=row.direct_file_url,
        )
        if pdf_url is None or not pdf_url.lower().endswith(".pdf"):
            raise ValueError("listing row did not expose a downloadable PDF")
        return ResolvedOrderFile(
            order_date=row.order_date.isoformat() if row.order_date else "",
            title=row.title,
            external_record_id=row.external_record_id or "",
            detail_url=row.detail_url or "",
            pdf_url=pdf_url,
        )

    if row.detail_url is None:
        raise ValueError("listing row has neither a detail page nor a direct PDF link")

    try:
        fetch_result = detail_fetcher.fetch(url=row.detail_url, referer_url=referer_url)
    except DetailPageFetchError as exc:
        raise ValueError(f"detail page fetch failed: {exc}") from exc

    parsed_detail = parse_detail_page(html=fetch_result.text, base_url=row.detail_url)
    pdf_url = parsed_detail.attached_file_url
    if pdf_url is None or not pdf_url.lower().endswith(".pdf"):
        raise ValueError("detail page did not expose a PDF attachment")

    return ResolvedOrderFile(
        order_date=(parsed_detail.order_date or row.order_date).isoformat()
        if (parsed_detail.order_date or row.order_date)
        else "",
        title=parsed_detail.title or row.title,
        external_record_id=row.external_record_id or "",
        detail_url=row.detail_url,
        pdf_url=pdf_url,
    )


def upsert_successful_row(
    *,
    previous_row: ManifestRow | None,
    record_key: str,
    bucket_name: str,
    resolved: ResolvedOrderFile,
    desired_filename: str,
    category_dir: Path,
    downloader: FileDownloader,
    now_factory: Callable[[], datetime],
) -> tuple[ManifestRow, Literal["downloaded", "skipped"]]:
    timestamp = now_factory().isoformat()
    desired_path = category_dir / desired_filename
    previous_path = (
        category_dir / previous_row.local_filename
        if previous_row is not None and previous_row.local_filename
        else None
    )
    existing_pdf_url = previous_row.pdf_url if previous_row is not None else ""

    if previous_row is not None and existing_pdf_url == resolved.pdf_url:
        if desired_path.exists():
            return (
                build_downloaded_manifest_row(
                    previous_row=previous_row,
                    record_key=record_key,
                    bucket_name=bucket_name,
                    resolved=resolved,
                    local_filename=desired_filename,
                    now_iso=timestamp,
                ),
                "skipped",
            )
        if previous_path is not None and previous_path.exists() and previous_path != desired_path:
            if desired_path.exists() and desired_path != previous_path:
                previous_path.unlink(missing_ok=True)
            else:
                previous_path.replace(desired_path)
            return (
                build_downloaded_manifest_row(
                    previous_row=previous_row,
                    record_key=record_key,
                    bucket_name=bucket_name,
                    resolved=resolved,
                    local_filename=desired_filename,
                    now_iso=timestamp,
                ),
                "skipped",
            )

    download_pdf(
        downloader=downloader,
        pdf_url=resolved.pdf_url,
        detail_url=resolved.detail_url or None,
        destination=desired_path,
    )
    if previous_path is not None and previous_path.exists() and previous_path != desired_path:
        previous_path.unlink(missing_ok=True)

    return (
        build_downloaded_manifest_row(
            previous_row=previous_row,
            record_key=record_key,
            bucket_name=bucket_name,
            resolved=resolved,
            local_filename=desired_filename,
            now_iso=timestamp,
        ),
        "downloaded",
    )


def build_downloaded_manifest_row(
    *,
    previous_row: ManifestRow | None,
    record_key: str,
    bucket_name: str,
    resolved: ResolvedOrderFile,
    local_filename: str,
    now_iso: str,
) -> ManifestRow:
    first_seen_at = previous_row.first_seen_at if previous_row is not None else now_iso
    return ManifestRow(
        record_key=record_key,
        bucket_name=bucket_name,
        order_date=resolved.order_date,
        title=resolved.title,
        external_record_id=resolved.external_record_id,
        detail_url=resolved.detail_url,
        pdf_url=resolved.pdf_url,
        local_filename=local_filename,
        status="downloaded",
        error="",
        first_seen_at=first_seen_at,
        last_seen_at=now_iso,
    )


def build_failed_manifest_row(
    *,
    previous_row: ManifestRow | None,
    record_key: str,
    bucket_name: str,
    row: ParsedListingRow,
    error: str,
    now_factory: Callable[[], datetime],
) -> ManifestRow:
    now_iso = now_factory().isoformat()
    first_seen_at = previous_row.first_seen_at if previous_row is not None else now_iso
    return ManifestRow(
        record_key=record_key,
        bucket_name=bucket_name,
        order_date=row.order_date.isoformat() if row.order_date else "",
        title=row.title,
        external_record_id=row.external_record_id or "",
        detail_url=row.detail_url or (previous_row.detail_url if previous_row else ""),
        pdf_url=(previous_row.pdf_url if previous_row is not None else ""),
        local_filename=(previous_row.local_filename if previous_row is not None else ""),
        status="failed",
        error=error,
        first_seen_at=first_seen_at,
        last_seen_at=now_iso,
    )


def materially_changed(previous_row: ManifestRow | None, next_row: ManifestRow) -> bool:
    if previous_row is None:
        return True
    return _material_tuple(previous_row) != _material_tuple(next_row)


def _material_tuple(row: ManifestRow) -> tuple[str, ...]:
    return tuple(getattr(row, field_name) for field_name in _MATERIAL_MANIFEST_FIELDS)


def build_output_file_name(*, record_key: str, resolved: ResolvedOrderFile) -> str:
    date_part = resolved.order_date or "unknown-date"
    record_id = derive_record_id(record_key)
    slug = slugify_title(resolved.title)
    return f"{date_part}__{record_id}__{slug}.pdf"


def derive_record_id(record_key: str) -> str:
    prefix, _, value = record_key.partition(":")
    if prefix == "external" and value:
        return value
    if prefix == "derived" and value:
        return value[:_DERIVED_ID_LENGTH]
    fallback = hashlib.sha256(record_key.encode("utf-8")).hexdigest()
    return fallback[:_DERIVED_ID_LENGTH]


def slugify_title(title: str) -> str:
    normalized = "-".join(normalize_text(title).split())
    if not normalized:
        return "order"
    return normalized[:_MAX_SLUG_LENGTH].strip("-") or "order"


def download_pdf(
    *,
    downloader: FileDownloader,
    pdf_url: str,
    detail_url: str | None,
    destination: Path,
) -> None:
    download_result: DownloadedFile | None = None
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_result = downloader.fetch(url=pdf_url, referer_url=detail_url)
        if not looks_like_pdf(download_result):
            raise ValueError(
                "resolved URL did not return a PDF "
                f"(mime_type={download_result.mime_type!r}, "
                f"response_url={download_result.response_url!r})"
            )
        shutil.copyfile(download_result.temp_path, destination)
    except FileDownloadError as exc:
        raise ValueError(f"pdf download failed: {exc}") from exc
    finally:
        if download_result is not None:
            download_result.cleanup()


def looks_like_pdf(download_result: DownloadedFile) -> bool:
    if download_result.mime_type == "application/pdf":
        return True
    if (download_result.file_name or "").lower().endswith(".pdf"):
        return True
    return download_result.response_url.lower().endswith(".pdf")


def load_manifest(manifest_path: Path) -> dict[str, ManifestRow]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        manifest_rows: dict[str, ManifestRow] = {}
        for row in reader:
            manifest_row = ManifestRow(
                record_key=row.get("record_key", ""),
                bucket_name=row.get("bucket_name", ""),
                order_date=row.get("order_date", ""),
                title=row.get("title", ""),
                external_record_id=row.get("external_record_id", ""),
                detail_url=row.get("detail_url", ""),
                pdf_url=row.get("pdf_url", ""),
                local_filename=row.get("local_filename", ""),
                status=row.get("status", ""),
                error=row.get("error", ""),
                first_seen_at=row.get("first_seen_at", ""),
                last_seen_at=row.get("last_seen_at", ""),
            )
            manifest_rows[manifest_row.record_key] = manifest_row
        return manifest_rows


def write_manifest(*, manifest_path: Path, manifest_rows: dict[str, ManifestRow]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        manifest_rows.values(),
        key=lambda row: (row.order_date, row.title.lower(), row.record_key),
        reverse=True,
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELD_NAMES)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(asdict(row))


if __name__ == "__main__":
    raise SystemExit(main())
