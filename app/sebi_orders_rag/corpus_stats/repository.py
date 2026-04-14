"""Manifest-backed repository for SEBI corpus coverage statistics."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..ingestion.manifest_loader import discover_manifest_paths, load_manifest
from .models import BucketCorpusStats, CorpusStatsSnapshot


class CorpusStatsRepository:
    """Compute and persist lightweight corpus stats from local manifests."""

    def __init__(self, *, data_root: Path, snapshot_path: Path | None = None) -> None:
        self._data_root = data_root
        self._snapshot_path = snapshot_path or (
            _resolve_project_root(data_root) / "artifacts" / "sebi_corpus_stats.json"
        )

    @property
    def snapshot_path(self) -> Path:
        return self._snapshot_path

    def compute_snapshot(self) -> CorpusStatsSnapshot:
        """Compute a fresh stats snapshot from manifest files and local PDFs."""

        grouped_rows: dict[str, list[object]] = defaultdict(list)
        for manifest_path in discover_manifest_paths(_resolve_corpus_root(self._data_root)):
            loaded = load_manifest(manifest_path)
            grouped_rows[loaded.bucket_name].extend(loaded.rows)

        bucket_stats: list[BucketCorpusStats] = []
        for bucket_name in sorted(grouped_rows):
            rows = grouped_rows[bucket_name]
            dated_rows = [row.order_date for row in rows if row.order_date is not None]
            local_pdf_count = sum(
                1
                for row in rows
                if row.local_filename and row.local_path.is_file()
            )
            total_rows = len(rows)
            bucket_stats.append(
                BucketCorpusStats(
                    bucket_name=bucket_name,
                    total_manifest_rows=total_rows,
                    local_pdf_count=local_pdf_count,
                    missing_count=max(total_rows - local_pdf_count, 0),
                    min_date=min(dated_rows) if dated_rows else None,
                    max_date=max(dated_rows) if dated_rows else None,
                )
            )
        return CorpusStatsSnapshot(
            generated_at=datetime.now(timezone.utc),
            bucket_stats=tuple(bucket_stats),
        )

    def load_snapshot(self) -> CorpusStatsSnapshot:
        """Load the persisted snapshot when present, else compute one."""

        if not self._snapshot_path.exists():
            return self.compute_snapshot()
        with self._snapshot_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        bucket_stats = tuple(
            BucketCorpusStats(
                bucket_name=str(item["bucket_name"]),
                total_manifest_rows=int(item["total_manifest_rows"]),
                local_pdf_count=int(item["local_pdf_count"]),
                missing_count=int(item["missing_count"]),
                min_date=_optional_date(item.get("min_date")),
                max_date=_optional_date(item.get("max_date")),
            )
            for item in payload.get("bucket_stats", [])
            if isinstance(item, dict)
        )
        if not bucket_stats:
            return self.compute_snapshot()
        return CorpusStatsSnapshot(
            generated_at=_parse_generated_at(payload.get("generated_at")),
            bucket_stats=bucket_stats,
        )

    def save_snapshot(self, snapshot: CorpusStatsSnapshot) -> Path:
        """Persist one snapshot under the project artifacts directory."""

        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": snapshot.generated_at.isoformat(),
            "bucket_stats": [
                {
                    "bucket_name": item.bucket_name,
                    "total_manifest_rows": item.total_manifest_rows,
                    "local_pdf_count": item.local_pdf_count,
                    "missing_count": item.missing_count,
                    "min_date": item.min_date.isoformat() if item.min_date else None,
                    "max_date": item.max_date.isoformat() if item.max_date else None,
                }
                for item in snapshot.bucket_stats
            ],
        }
        with self._snapshot_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return self._snapshot_path


def _optional_date(value: object):
    if value in (None, ""):
        return None
    from datetime import date

    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _parse_generated_at(value: object) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _resolve_corpus_root(data_root: Path) -> Path:
    normalized = data_root.resolve()
    if normalized.name == "sebi-orders-pdfs":
        return normalized
    candidate = normalized / "sebi-orders-pdfs"
    return candidate if candidate.exists() else normalized


def _resolve_project_root(data_root: Path) -> Path:
    normalized = data_root.resolve()
    if normalized.name == "sebi-orders-pdfs":
        return normalized.parent
    return normalized
