"""Human-readable and JSON report formatting for chunk QA output."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .chunk_audit import CorpusAuditResult, DocumentAuditResult


def render_document_report(document: DocumentAuditResult) -> str:
    """Render a single-document inspection report for terminal output."""

    lines = ["Chunk QA Inspection", ""]
    lines.extend(_render_document_metadata(document))
    lines.append("")
    lines.append("Chunks")
    for chunk in document.chunks:
        flags = ", ".join(chunk.flags) if chunk.flags else "none"
        heading_path = " > ".join(chunk.heading_path) if chunk.heading_path else "-"
        section_title = chunk.section_title or "-"
        lines.append(
            f"[{chunk.chunk_index:03d}] pages={chunk.page_start}-{chunk.page_end} "
            f"section={chunk.section_type} tokens={chunk.token_count} flags={flags}"
        )
        lines.append(f"  section_title: {section_title}")
        lines.append(f"  heading_path: {heading_path}")
        lines.append(f"  first_200: {chunk.first_text_preview}")
        lines.append(f"  last_200: {chunk.last_text_preview}")
    return "\n".join(lines)


def render_corpus_audit_report(
    report: CorpusAuditResult,
    *,
    scope_label: str,
    show_chunks: bool,
    only_flagged: bool,
) -> str:
    """Render a multi-document audit report for terminal output."""

    lines = ["Chunk QA Audit", f"scope: {scope_label}", ""]
    lines.extend(_render_summary(report))
    lines.append("")
    lines.extend(_render_per_bucket(report))
    lines.append("")
    lines.extend(_render_flagged_documents(report, show_chunks=show_chunks))

    if not only_flagged:
        clean_documents = [document for document in report.documents if not document.is_flagged]
        lines.append("")
        lines.append("Clean Documents")
        if clean_documents:
            for document in clean_documents:
                lines.append(
                    f"- dv={document.document_version_id} record_key={document.record_key} "
                    f"bucket={document.bucket_name} pages={document.page_count} "
                    f"chunks={document.chunk_count} avg_tokens/chunk={_format_float(document.average_tokens_per_chunk)}"
                )
        else:
            lines.append("- none")

    return "\n".join(lines)


def render_document_json(document: DocumentAuditResult) -> str:
    """Render a single-document inspection report as JSON."""

    payload = {
        "mode": "document_inspection",
        "document": _document_to_dict(document, include_chunks=True),
    }
    return json.dumps(payload, indent=2, default=_json_default)


def render_corpus_audit_json(
    report: CorpusAuditResult,
    *,
    scope_label: str,
    show_chunks: bool,
    only_flagged: bool,
) -> str:
    """Render a multi-document audit report as JSON."""

    documents = [
        _document_to_dict(document, include_chunks=show_chunks)
        for document in report.documents
        if document.is_flagged or not only_flagged
    ]
    payload = {
        "mode": "audit",
        "scope": scope_label,
        "summary": _summary_to_dict(report),
        "per_bucket": [_bucket_to_dict(bucket) for bucket in report.per_bucket],
        "flagged_documents": [
            _document_to_dict(document, include_chunks=show_chunks)
            for document in report.flagged_documents
        ],
        "documents": documents,
    }
    return json.dumps(payload, indent=2, default=_json_default)


def write_report(output_path: Path, content: str) -> None:
    """Persist a rendered report to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _render_document_metadata(document: DocumentAuditResult) -> list[str]:
    return [
        f"document_version_id: {document.document_version_id}",
        f"record_key: {document.record_key}",
        f"title: {document.title}",
        f"bucket_name: {document.bucket_name}",
        f"order_date: {_format_nullable(document.order_date)}",
        f"page_count: {document.page_count}",
        f"chunk_count: {document.chunk_count}",
        f"avg_tokens_per_chunk: {_format_float(document.average_tokens_per_chunk)}",
        f"severity_score: {document.severity_score}",
        f"document_flags: {_format_flags(document.document_flag_counts)}",
        f"chunk_flag_counts: {_format_flags(document.chunk_flag_counts)}",
    ]


def _render_summary(report: CorpusAuditResult) -> list[str]:
    summary = report.summary
    return [
        "Summary",
        f"processed document_versions: {summary.processed_document_versions}",
        f"total chunks: {summary.total_chunks}",
        f"avg chunks/doc: {_format_float(summary.avg_chunks_per_document)}",
        f"median chunks/doc: {_format_float(summary.median_chunks_per_document)}",
        f"avg tokens/chunk: {_format_float(summary.avg_tokens_per_chunk)}",
        f"median tokens/chunk: {_format_float(summary.median_tokens_per_chunk)}",
        f"min/max tokens/chunk: {_format_nullable(summary.min_tokens_per_chunk)} / "
        f"{_format_nullable(summary.max_tokens_per_chunk)}",
    ]


def _render_per_bucket(report: CorpusAuditResult) -> list[str]:
    lines = ["Per Bucket"]
    if not report.per_bucket:
        lines.append("- none")
        return lines

    for bucket in report.per_bucket:
        lines.append(
            f"- {bucket.bucket_name}: docs={bucket.processed_documents} "
            f"chunks={bucket.total_chunks} avg_chunks/doc={_format_float(bucket.avg_chunks_per_document)} "
            f"avg_tokens/chunk={_format_float(bucket.avg_tokens_per_chunk)}"
        )
    return lines


def _render_flagged_documents(
    report: CorpusAuditResult,
    *,
    show_chunks: bool,
) -> list[str]:
    lines = ["Flagged Documents"]
    flagged_documents = report.flagged_documents
    if not flagged_documents:
        lines.append("- none")
        return lines

    for index, document in enumerate(flagged_documents, start=1):
        lines.append(
            f"{index}. score={document.severity_score} dv={document.document_version_id} "
            f"record_key={document.record_key}"
        )
        lines.append(f"   title: {document.title}")
        lines.append(
            f"   bucket={document.bucket_name} order_date={_format_nullable(document.order_date)} "
            f"pages={document.page_count} chunks={document.chunk_count}"
        )
        lines.append(
            f"   avg_tokens/chunk={_format_float(document.average_tokens_per_chunk)}"
        )
        lines.append(f"   document_flags={_format_flags(document.document_flag_counts)}")
        lines.append(f"   chunk_flag_counts={_format_flags(document.chunk_flag_counts)}")

        if show_chunks:
            for chunk in document.chunks:
                if not chunk.flags:
                    continue
                flags = ", ".join(chunk.flags)
                lines.append(
                    f"   - chunk[{chunk.chunk_index}] pages={chunk.page_start}-{chunk.page_end} "
                    f"section={chunk.section_type} tokens={chunk.token_count} flags={flags}"
                )
                lines.append(f"     first_200={chunk.first_text_preview}")
                lines.append(f"     last_200={chunk.last_text_preview}")
    return lines


def _summary_to_dict(report: CorpusAuditResult) -> dict[str, Any]:
    summary = report.summary
    return {
        "processed_document_versions": summary.processed_document_versions,
        "total_chunks": summary.total_chunks,
        "avg_chunks_per_document": summary.avg_chunks_per_document,
        "median_chunks_per_document": summary.median_chunks_per_document,
        "avg_tokens_per_chunk": summary.avg_tokens_per_chunk,
        "median_tokens_per_chunk": summary.median_tokens_per_chunk,
        "min_tokens_per_chunk": summary.min_tokens_per_chunk,
        "max_tokens_per_chunk": summary.max_tokens_per_chunk,
    }


def _bucket_to_dict(bucket: Any) -> dict[str, Any]:
    return {
        "bucket_name": bucket.bucket_name,
        "processed_documents": bucket.processed_documents,
        "total_chunks": bucket.total_chunks,
        "avg_chunks_per_document": bucket.avg_chunks_per_document,
        "avg_tokens_per_chunk": bucket.avg_tokens_per_chunk,
    }


def _document_to_dict(
    document: DocumentAuditResult,
    *,
    include_chunks: bool,
) -> dict[str, Any]:
    payload = {
        "document_version_id": document.document_version_id,
        "record_key": document.record_key,
        "title": document.title,
        "bucket_name": document.bucket_name,
        "order_date": document.order_date,
        "page_count": document.page_count,
        "chunk_count": document.chunk_count,
        "average_tokens_per_chunk": document.average_tokens_per_chunk,
        "severity_score": document.severity_score,
        "document_flags": dict(document.document_flag_counts),
        "chunk_flag_counts": dict(document.chunk_flag_counts),
        "is_flagged": document.is_flagged,
    }
    if include_chunks:
        payload["chunks"] = [
            {
                "chunk_index": chunk.chunk_index,
                "section_type": chunk.section_type,
                "section_title": chunk.section_title,
                "heading_path": list(chunk.heading_path),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "flags": list(chunk.flags),
                "first_text_preview": chunk.first_text_preview,
                "last_text_preview": chunk.last_text_preview,
            }
            for chunk in document.chunks
        ]
    return payload


def _format_flags(flag_counts: Any) -> str:
    if not flag_counts:
        return "none"
    return ", ".join(f"{flag}={count}" for flag, count in flag_counts.items())


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_nullable(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
