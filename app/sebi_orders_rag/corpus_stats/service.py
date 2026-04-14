"""Direct-answer service for bucket counts, date ranges, and local PDF coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BucketCorpusStats, CorpusStatsAnswer, CorpusStatsSnapshot

_COUNT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\bcount\b", re.IGNORECASE),
)
_DATE_RANGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdate range\b", re.IGNORECASE),
    re.compile(r"\bfrom which dates\b", re.IGNORECASE),
    re.compile(r"\bfrom what dates\b", re.IGNORECASE),
)
_LOCAL_PDF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blocal pdfs?\b", re.IGNORECASE),
    re.compile(r"\bpdf availability\b", re.IGNORECASE),
)
_CATEGORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow many categories\b", re.IGNORECASE),
    re.compile(r"\bwhich categories\b", re.IGNORECASE),
    re.compile(r"\bbuckets?\b", re.IGNORECASE),
)
_BUCKET_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("orders-of-sat", ("sat", "sat case", "sat cases")),
    ("orders-of-courts", ("court", "courts", "court case", "court cases")),
    ("orders-of-special-courts", ("special court", "special courts")),
    ("settlement-orders", ("settlement", "settlement order", "settlement orders")),
    ("orders-under-regulation-30a", ("regulation 30a", "30a order", "30a orders")),
)


@dataclass(frozen=True)
class CorpusStatsQuery:
    """Parsed corpus-stat intent."""

    metric: str
    bucket_name: str | None = None


class CorpusStatsService:
    """Answer coverage and date-range questions from stored corpus metadata."""

    def __init__(self, *, repository) -> None:
        self._repository = repository

    def supports_query(self, query: str) -> bool:
        """Return whether a query should route to the corpus-stats layer."""

        plan = self.classify_query(query)
        return plan is not None

    def classify_query(self, query: str) -> CorpusStatsQuery | None:
        """Return the parsed corpus-stat query plan when supported."""

        normalized = " ".join(query.lower().split())
        bucket_name = _resolve_bucket_name(normalized)
        if any(pattern.search(normalized) for pattern in _DATE_RANGE_PATTERNS):
            if bucket_name is None:
                return None
            return CorpusStatsQuery(metric="date_range", bucket_name=bucket_name)
        if any(pattern.search(normalized) for pattern in _LOCAL_PDF_PATTERNS):
            if bucket_name is None:
                return None
            return CorpusStatsQuery(metric="local_pdf_count", bucket_name=bucket_name)
        if any(pattern.search(normalized) for pattern in _CATEGORY_PATTERNS):
            return CorpusStatsQuery(metric="category_count")
        if any(pattern.search(normalized) for pattern in _COUNT_PATTERNS):
            if bucket_name is None:
                return None
            return CorpusStatsQuery(metric="manifest_count", bucket_name=bucket_name)
        return None

    def answer_query(self, query: str) -> CorpusStatsAnswer | None:
        """Answer a supported coverage query from the persisted snapshot."""

        plan = self.classify_query(query)
        if plan is None:
            return None
        snapshot = self._repository.load_snapshot()
        if plan.metric == "category_count":
            return self._answer_category_count(snapshot)
        bucket = _bucket_from_snapshot(snapshot, plan.bucket_name)
        if bucket is None:
            return None
        if plan.metric == "manifest_count":
            return CorpusStatsAnswer(
                answer_text=(
                    f"The local corpus currently has {bucket.total_manifest_rows} "
                    f"records in {bucket.bucket_name}."
                ),
                debug={
                    "metric": plan.metric,
                    "bucket_name": bucket.bucket_name,
                    "value": bucket.total_manifest_rows,
                },
            )
        if plan.metric == "local_pdf_count":
            return CorpusStatsAnswer(
                answer_text=(
                    f"The local corpus currently has {bucket.local_pdf_count} local PDFs "
                    f"available in {bucket.bucket_name}."
                ),
                debug={
                    "metric": plan.metric,
                    "bucket_name": bucket.bucket_name,
                    "value": bucket.local_pdf_count,
                    "missing_count": bucket.missing_count,
                },
            )
        return CorpusStatsAnswer(
            answer_text=_render_date_range(bucket),
            debug={
                "metric": plan.metric,
                "bucket_name": bucket.bucket_name,
                "min_date": bucket.min_date.isoformat() if bucket.min_date else None,
                "max_date": bucket.max_date.isoformat() if bucket.max_date else None,
            },
        )

    @staticmethod
    def _answer_category_count(snapshot: CorpusStatsSnapshot) -> CorpusStatsAnswer:
        bucket_names = tuple(item.bucket_name for item in snapshot.bucket_stats)
        return CorpusStatsAnswer(
            answer_text=(
                f"The local corpus currently spans {len(bucket_names)} order categories: "
                + ", ".join(bucket_names)
                + "."
            ),
            debug={
                "metric": "category_count",
                "bucket_count": len(bucket_names),
                "bucket_names": list(bucket_names),
            },
        )


def _resolve_bucket_name(normalized_query: str) -> str | None:
    for bucket_name, aliases in _BUCKET_ALIASES:
        if any(alias in normalized_query for alias in aliases):
            return bucket_name
    return None


def _bucket_from_snapshot(
    snapshot: CorpusStatsSnapshot,
    bucket_name: str | None,
) -> BucketCorpusStats | None:
    if bucket_name is None:
        return None
    for item in snapshot.bucket_stats:
        if item.bucket_name == bucket_name:
            return item
    return None


def _render_date_range(bucket: BucketCorpusStats) -> str:
    if bucket.min_date is None or bucket.max_date is None:
        return f"I could not determine a clean date range for {bucket.bucket_name} from the stored manifest metadata."
    return (
        f"For {bucket.bucket_name}, the local corpus currently spans "
        f"{bucket.min_date.isoformat()} to {bucket.max_date.isoformat()}."
    )
