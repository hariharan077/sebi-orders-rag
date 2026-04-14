"""Deterministic confidence scoring for grounded Phase 4 answers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..constants import SUBSTANTIVE_SECTION_TYPES
from ..schemas import PromptContextChunk
from ..web_fallback.ranking import is_domain_allowed

_SETTLEMENT_BUCKET_NAME = "settlement-orders"


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Deterministic confidence decision for one answer."""

    confidence: float
    should_abstain: bool
    should_hedge: bool


def assess_retrieval_confidence(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
    cited_context_chunks: tuple[PromptContextChunk, ...],
    answer_status: str,
    threshold: float,
    strict_scope_required: bool = False,
    strict_single_matter: bool = False,
    locked_record_keys: tuple[str, ...] = (),
) -> ConfidenceAssessment:
    """Score grounded answer support from retrieval quality and citation support."""

    if not context_chunks or not cited_context_chunks or answer_status != "answered":
        return ConfidenceAssessment(confidence=0.0, should_abstain=True, should_hedge=False)
    if strict_scope_required and not strict_single_matter:
        return ConfidenceAssessment(confidence=0.0, should_abstain=True, should_hedge=False)

    ordered_scores = sorted((chunk.score for chunk in context_chunks), reverse=True)
    top_score = ordered_scores[0]
    if top_score < 0.01:
        return ConfidenceAssessment(
            confidence=round(_clamp(top_score / 0.08), 4),
            should_abstain=True,
            should_hedge=False,
        )
    second_score = ordered_scores[1] if len(ordered_scores) > 1 else top_score * 0.85
    score_quality = _clamp(top_score / 0.08)
    margin_quality = _clamp((top_score - second_score) / 0.02)
    substantive_quality = (
        sum(1 for chunk in cited_context_chunks if chunk.section_type in SUBSTANTIVE_SECTION_TYPES)
        / len(cited_context_chunks)
    )
    agreement_quality = _agreement_quality(cited_context_chunks)
    strict_matter_quality = _strict_matter_quality(
        cited_context_chunks=cited_context_chunks,
        strict_single_matter=strict_single_matter,
        locked_record_keys=locked_record_keys,
    )
    settlement_support_quality = _settlement_support_quality(
        context_chunks=context_chunks,
        cited_context_chunks=cited_context_chunks,
    )
    support_quality = 1.0
    confidence = round(
        _clamp(
            (score_quality * 0.35)
            + (margin_quality * 0.15)
            + (substantive_quality * 0.20)
            + (agreement_quality * 0.15)
            + (strict_matter_quality * 0.15)
            + (settlement_support_quality * 0.05)
            + (support_quality * 0.05)
        ),
        4,
    )
    abstain_threshold = threshold * 0.75
    if (
        settlement_support_quality >= 0.75
        and substantive_quality >= 0.5
        and top_score >= 0.04
    ):
        abstain_threshold = threshold * 0.70
    should_abstain = confidence < abstain_threshold
    should_hedge = not should_abstain and confidence < threshold
    return ConfidenceAssessment(
        confidence=confidence,
        should_abstain=should_abstain,
        should_hedge=should_hedge,
    )


def assess_direct_llm_confidence(answer_text: str) -> float:
    """Return a conservative confidence estimate for direct explanatory answers."""

    normalized = answer_text.lower()
    confidence = 0.62
    if any(phrase in normalized for phrase in ("may", "might", "depends", "uncertain")):
        confidence -= 0.10
    if any(phrase in normalized for phrase in ("i do not know", "not sure", "insufficient")):
        confidence -= 0.22
    if len(answer_text.split()) < 12:
        confidence -= 0.05
    return round(_clamp(confidence), 4)


def assess_web_fallback_confidence(
    *,
    answer_status: str,
    sources: tuple[object, ...],
    preferred_source_type: str,
    preferred_domains: tuple[str, ...] = (),
) -> ConfidenceAssessment:
    """Return a conservative confidence estimate for web-backed answers."""

    if answer_status != "answered" or not sources:
        return ConfidenceAssessment(confidence=0.0, should_abstain=True, should_hedge=False)

    unique_urls = {
        str(getattr(source, "source_url", getattr(source, "url", "")) or "").strip()
        for source in sources
        if str(getattr(source, "source_url", getattr(source, "url", "")) or "").strip()
    }
    unique_count = len(unique_urls)
    official_match_count = sum(
        1
        for source in sources
        if str(getattr(source, "source_type", "") or "") == "official_web"
        and is_domain_allowed(
            str(getattr(source, "domain", "") or ""),
            preferred_domains,
        )
    )
    authoritative_count = sum(
        1
        for source in sources
        if _is_authoritative_domain(str(getattr(source, "domain", "") or ""))
    )

    if preferred_source_type == "official_web":
        if preferred_domains and official_match_count == 0:
            return ConfidenceAssessment(confidence=0.0, should_abstain=True, should_hedge=False)
        confidence = _clamp(
            0.78
            + min(max(unique_count - 1, 0), 2) * 0.05
            + min(max(official_match_count - 1, 0), 1) * 0.04
        )
        confidence = round(confidence, 4)
        should_abstain = confidence < 0.72
        should_hedge = not should_abstain and confidence < 0.83
        return ConfidenceAssessment(
            confidence=confidence,
            should_abstain=should_abstain,
            should_hedge=should_hedge,
        )

    confidence = _clamp(
        0.56
        + min(max(unique_count - 1, 0), 2) * 0.06
        + min(authoritative_count, 2) * 0.04
    )
    confidence = round(confidence, 4)
    should_abstain = confidence < 0.58
    should_hedge = not should_abstain and confidence < 0.68
    return ConfidenceAssessment(
        confidence=confidence,
        should_abstain=should_abstain,
        should_hedge=should_hedge,
    )


def _agreement_quality(cited_context_chunks: tuple[PromptContextChunk, ...]) -> float:
    counts = Counter(chunk.record_key for chunk in cited_context_chunks)
    if not counts:
        return 0.0
    if len(counts) == 1:
        return 1.0
    dominant_share = max(counts.values()) / sum(counts.values())
    if dominant_share >= 0.67:
        return 0.65
    return 0.35


def _settlement_support_quality(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
    cited_context_chunks: tuple[PromptContextChunk, ...],
) -> float:
    settlement_context = [
        chunk for chunk in context_chunks[:4] if chunk.bucket_name == _SETTLEMENT_BUCKET_NAME
    ]
    if not settlement_context:
        return 0.0

    cited_settlement_chunks = [
        chunk for chunk in cited_context_chunks if chunk.bucket_name == _SETTLEMENT_BUCKET_NAME
    ]
    cited_substantive_ratio = (
        sum(1 for chunk in cited_settlement_chunks if chunk.section_type in SUBSTANTIVE_SECTION_TYPES)
        / len(cited_settlement_chunks)
        if cited_settlement_chunks
        else 0.0
    )
    context_share = len(settlement_context) / min(len(context_chunks), 4)
    return _clamp(max(cited_substantive_ratio, context_share * 0.8))


def _strict_matter_quality(
    *,
    cited_context_chunks: tuple[PromptContextChunk, ...],
    strict_single_matter: bool,
    locked_record_keys: tuple[str, ...],
) -> float:
    if not strict_single_matter:
        return 1.0
    if not cited_context_chunks or not locked_record_keys:
        return 0.0
    cited_record_keys = {chunk.record_key for chunk in cited_context_chunks}
    if not cited_record_keys.issubset(set(locked_record_keys)):
        return 0.0
    substantive_citations = [
        chunk for chunk in cited_context_chunks if chunk.section_type in SUBSTANTIVE_SECTION_TYPES
    ]
    if substantive_citations:
        return 1.0
    return 0.55


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_authoritative_domain(domain: str) -> bool:
    normalized = (domain or "").lower()
    if not normalized:
        return False
    return normalized.endswith(
        (
            ".gov.in",
            ".nic.in",
            ".gov",
            ".edu",
            ".org",
        )
    ) or normalized in {"wikipedia.org", "www.wikipedia.org"}
