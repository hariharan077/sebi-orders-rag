#!/usr/bin/env python3
"""Refresh the stored SEBI corpus bucket stats snapshot from local manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from app.sebi_orders_rag.corpus_stats import CorpusStatsRepository

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Optional override for SEBI_ORDERS_RAG_DATA_ROOT.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        load_env_file(PROJECT_ROOT / ".env")
        settings = SebiOrdersRagSettings.from_env(data_root_override=args.data_root)
        repository = CorpusStatsRepository(data_root=settings.data_root)
        snapshot = repository.compute_snapshot()
        path = repository.save_snapshot(snapshot)
        print(f"saved: {path}")
        print(f"generated_at: {snapshot.generated_at.isoformat()}")
        for bucket in snapshot.bucket_stats:
            print(
                f"{bucket.bucket_name}"
                f" | manifest_rows={bucket.total_manifest_rows}"
                f" | local_pdfs={bucket.local_pdf_count}"
                f" | missing={bucket.missing_count}"
                f" | min_date={bucket.min_date.isoformat() if bucket.min_date else '-'}"
                f" | max_date={bucket.max_date.isoformat() if bucket.max_date else '-'}"
            )
        return 0
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(
            "Corpus stats refresh finished with a safe failure summary: "
            f"{type(exc).__name__}: {' '.join(str(exc).split()) or 'no details available'}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
