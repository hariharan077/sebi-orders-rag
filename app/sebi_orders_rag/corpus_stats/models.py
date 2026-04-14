"""Typed corpus-stat models for direct metadata answers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class BucketCorpusStats:
    """One bucket's corpus coverage and date-range metrics."""

    bucket_name: str
    total_manifest_rows: int
    local_pdf_count: int
    missing_count: int
    min_date: date | None = None
    max_date: date | None = None


@dataclass(frozen=True)
class CorpusStatsSnapshot:
    """Stored or computed corpus-stat snapshot."""

    generated_at: datetime
    bucket_stats: tuple[BucketCorpusStats, ...]


@dataclass(frozen=True)
class CorpusStatsAnswer:
    """Direct answer produced from corpus metadata instead of RAG."""

    answer_text: str
    debug: dict[str, object]
