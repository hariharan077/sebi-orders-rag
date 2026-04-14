"""Answer-time helpers for extracted order metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..answering.style import apply_grounded_wording_caution
from ..router.query_analyzer import analyze_query
from ..schemas import Citation, PromptContextChunk
from .legal_provisions import explain_provisions
from .models import (
    MetadataChunkText,
    StoredLegalProvision,
    StoredNumericFact,
    StoredOrderMetadata,
    StoredPriceMovement,
)

_SIGNATORY_TRIGGER_RE = re.compile(r"(?:\bdate\s*:|\bplace\s*:|\bsigned\b)", re.IGNORECASE)
_SIGNATORY_SKIP_RE = re.compile(
    r"\b(?:securities and exchange board of india|order in the matter of|strictly complied with|copy of this order)\b",
    re.IGNORECASE,
)
_SIGNATORY_NAME_RE = re.compile(r"^[A-Z][A-Z .&'-]{1,80}$")
_INLINE_SIGNATORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"Date:\s*[^|]+?\s+(?P<name>[A-Z][A-Za-z .'-]{2,80}?)\s+Place:\s*[^|]+?\s+(?P<designation>Quasi[- ]Judicial Authority|Adjudicating Officer|Whole Time Member|Regional Director)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Date:\s*[^|]+?\s+Place:\s*[^|]+?\s+(?P<name>[A-Z][A-Za-z .'-]{2,80}?)\s+(?P<designation>Quasi[- ]Judicial Authority|Adjudicating Officer|Whole Time Member|Regional Director)",
        re.IGNORECASE,
    ),
)
_SIGNATORY_DESIGNATION_TOKENS: tuple[str, ...] = (
    "authority",
    "member",
    "chair",
    "director",
    "officer",
    "adjudicating",
    "judicial",
)
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AMOUNT_RE = re.compile(
    r"\b(?:rs\.?|inr)\s*[0-9][0-9,]*(?:\.\d+)?\b|\b[0-9][0-9,]*(?:\.\d+)?\s*(?:crore|lakh)\b",
    re.IGNORECASE,
)
_HOLDING_SIGNAL_RE = re.compile(r"\bshares?\b|\bhold(?:ing|ings)?\b|\bown(?:s|ed|ership)?\b|\b\d+(?:\.\d+)?\s*%\b", re.IGNORECASE)
_PARTY_SIGNAL_RE = re.compile(r"\bpart(?:y|ies)\b|\bnoticees?\b|\brespondents?\b|\bappellants?\b", re.IGNORECASE)
_PRICE_QUERY_RE = re.compile(
    r"\b(?:share price|price movement|listing price|listed at|highest price|lowest price|price before|price after|before and after|patch(?:es)?|each period|period-wise|period wise|increased by|decrease(?:d)? by|how much did .* share price increase)\b",
    re.IGNORECASE,
)
_PERIOD_PRICE_QUERY_RE = re.compile(
    r"\b(?:price movement.*(?:period|patch)|each period|each patch|period-wise|patch-wise)\b",
    re.IGNORECASE,
)
_BEFORE_AFTER_PRICE_RE = re.compile(
    r"\b(?:before and after|price before and after|before the increase|after the increase)\b",
    re.IGNORECASE,
)
_LISTING_PRICE_QUERY_RE = re.compile(r"\b(?:listing price|listed at)\b", re.IGNORECASE)
_HIGHEST_PRICE_QUERY_RE = re.compile(r"\b(?:highest|peak)\s+price\b", re.IGNORECASE)
_LOWEST_PRICE_QUERY_RE = re.compile(r"\blow(?:est)?\s+price\b", re.IGNORECASE)
_PRICE_INCREASE_QUERY_RE = re.compile(
    r"\b(?:how much did .* share price increase|how much .* price increase|price increased by|percentage increase|percent increase)\b",
    re.IGNORECASE,
)
_QUERY_SUBJECT_RE = re.compile(
    r"\b(?:does|did|do)\s+(?P<subject>[a-z][a-z0-9 .&'/-]{2,120}?)\s+(?:own|hold|held)\b",
    re.IGNORECASE,
)
_OBSERVATION_TERM_RE = re.compile(
    r"\b(?:designated authority|(?:^|\s)da(?:\s|$)|enquiry report|enquiry officer|observ(?:e|ed|ation|ations)|find(?:ing|ings)?|conclud(?:e|ed))\b",
    re.IGNORECASE,
)
_DA_TERM_RE = re.compile(r"\b(?:designated authority|(?:^|\s)da(?:\s|$))\b", re.IGNORECASE)
_ENQUIRY_TERM_RE = re.compile(r"\benquiry\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+|\n+")
_DECISION_TERM_RE = re.compile(
    r"\b(?:dismissed|allowed|disposed of|set aside|upheld|quashed|remanded|rejected|accepted|granted)\b",
    re.IGNORECASE,
)
_ACTIVE_MATTER_FOLLOW_UP_TERMS: dict[str, tuple[str, ...]] = {
    "exemption_granted": (
        "exemption",
        "granted",
        "open offer",
        "regulation 11",
        "exempt",
    ),
    "appellate_authority_decision": (
        "appeal",
        "appellate authority",
        "dismissed",
        "allowed",
        "set aside",
        "upheld",
        "disposed of",
    ),
    "da_observed": (
        "designated authority",
        "da",
        "observed",
        "finding",
        "concluded",
        "enquiry",
    ),
    "settlement_amount": (
        "settlement amount",
        "notice of demand",
        "remitted",
        "credit of the same",
        "credit of said amount",
    ),
    "penalty": (
        "penalty",
        "penalties",
        "fine",
        "fined",
        "sentence",
        "imprisonment",
    ),
    "final_direction": (
        "it is hereby ordered",
        "directed",
        "directions",
        "ordered that",
        "final order",
    ),
    "outcome": (
        "dismissed",
        "allowed",
        "restrained",
        "debarred",
        "disposed of",
        "held",
        "directed",
    ),
    "sat_hold": (
        "sat",
        "tribunal",
        "court",
        "held",
        "dismissed",
        "allowed",
        "upheld",
        "quashed",
    ),
}
_ACTIVE_FOLLOW_UP_OPERATIVE_INTENTS = frozenset(
    {"exemption_granted", "settlement_amount", "penalty", "final_direction"}
)
_ACTIVE_FOLLOW_UP_RESULT_INTENTS = frozenset(
    {"appellate_authority_decision", "outcome", "sat_hold"}
)
_SECTION_PRIORS_DEFAULT: dict[str, float] = {
    "operative_order": 3.2,
    "directions": 2.8,
    "findings": 2.4,
    "issues": 1.8,
    "reply_or_submissions": 1.1,
    "facts": 1.0,
    "background": 0.9,
    "other": 0.8,
    "table_block": 0.6,
    "annexure": 0.5,
    "header": 0.3,
}
_SECTION_PRIORS_RESULT: dict[str, float] = {
    "findings": 3.0,
    "operative_order": 2.9,
    "directions": 2.8,
    "issues": 2.0,
    "reply_or_submissions": 1.0,
    "facts": 0.9,
    "background": 0.8,
    "other": 0.7,
    "table_block": 0.5,
    "annexure": 0.4,
    "header": 0.2,
}


@dataclass(frozen=True)
class MetadataAnswer:
    """Direct answer produced from stored order metadata."""

    answer_text: str
    citations: tuple[Citation, ...]
    metadata_type: str
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SemanticFollowUpMatch:
    row: StoredOrderMetadata
    chunk: MetadataChunkText
    answer_text: str
    score: float


class OrderMetadataService:
    """Use extracted metadata first for signatory and legal-provision follow-ups."""

    def __init__(self, *, repository: object) -> None:
        self._repository = repository

    def answer_signatory_question(
        self,
        *,
        document_version_ids: Sequence[int],
    ) -> MetadataAnswer | None:
        metadata_rows = self._repository.fetch_order_metadata(document_version_ids=document_version_ids)
        candidates = [
            row
            for row in metadata_rows
            if row.signatory_name or row.signatory_designation
        ]
        if candidates:
            ordered_candidates = sorted(
                candidates,
                key=lambda item: (-item.metadata_confidence, item.document_version_id),
            )
            row = next(
                (item for item in ordered_candidates if _metadata_signatory_is_usable(item)),
                None,
            )
            if row is not None and row.signatory_name and row.signatory_designation:
                answer = f"The order was signed by {row.signatory_name}, {row.signatory_designation}."
            elif row is not None and row.signatory_name:
                answer = f"The extracted signatory for the order is {row.signatory_name}."
            elif row is not None:
                answer = f"The extracted signatory designation for the order is {row.signatory_designation}."
            else:
                answer = ""
            if row is not None and (row.order_date or row.place):
                details = []
                if row.order_date:
                    details.append(f"date: {row.order_date.isoformat()}")
                if row.place:
                    details.append(f"place: {row.place}")
                answer = answer.rstrip(".") + "; " + "; ".join(details) + "."
            if row is not None:
                return MetadataAnswer(
                    answer_text=answer,
                    citations=(self._metadata_citation(row=row, citation_number=1),),
                    metadata_type="signatory",
                    debug={
                        "document_version_id": row.document_version_id,
                        "metadata_confidence": row.metadata_confidence,
                        "metadata_type": "signatory",
                    },
                )

        fallback_answer = self._answer_signatory_from_footer(metadata_rows=metadata_rows)
        if fallback_answer is not None:
            return fallback_answer
        return None

    def answer_order_date_question(
        self,
        *,
        document_version_ids: Sequence[int],
    ) -> MetadataAnswer | None:
        metadata_rows = self._repository.fetch_order_metadata(document_version_ids=document_version_ids)
        candidates = [row for row in metadata_rows if row.order_date is not None]
        if not candidates:
            return None
        row = sorted(candidates, key=lambda item: (-item.metadata_confidence, item.document_version_id))[0]
        answer = f"The order date recorded for this matter is {row.order_date.isoformat()}."
        return MetadataAnswer(
            answer_text=answer,
            citations=(self._metadata_citation(row=row, citation_number=1),),
            metadata_type="order_date",
            debug={
                "document_version_id": row.document_version_id,
                "metadata_confidence": row.metadata_confidence,
                "metadata_type": "order_date",
            },
        )

    def answer_legal_provisions_question(
        self,
        *,
        document_version_ids: Sequence[int],
        explain: bool,
    ) -> MetadataAnswer | None:
        rows = self._repository.fetch_legal_provisions(document_version_ids=document_version_ids)
        if not rows:
            return None
        grouped = _group_provisions(rows)
        primary_record_key = rows[0].record_key
        primary_title = rows[0].title
        summary = "The active order cites " + _render_grouped_provisions(grouped) + "."
        if explain:
            explanation_lines = []
            seen: set[str] = set()
            for line in explain_provisions(rows[:6]):
                if line in seen:
                    continue
                seen.add(line)
                explanation_lines.append(line)
            if explanation_lines:
                summary += " In plain terms: " + " ".join(explanation_lines[:4])
        citations = tuple(
            self._provision_citation(row=row, citation_number=index)
            for index, row in enumerate(rows[:4], start=1)
        )
        return MetadataAnswer(
            answer_text=summary,
            citations=citations,
            metadata_type="legal_provisions",
            debug={
                "document_version_id": rows[0].document_version_id,
                "record_key": primary_record_key,
                "title": primary_title,
                "metadata_type": "legal_provisions",
                "provision_count": len(rows),
                "explained": explain,
            },
        )

    def answer_exact_fact_question(
        self,
        *,
        query: str,
        document_version_ids: Sequence[int],
    ) -> MetadataAnswer | None:
        load_chunks = getattr(self._repository, "load_chunks", None)
        if not callable(load_chunks):
            return None

        metadata_rows = self._repository.fetch_order_metadata(document_version_ids=document_version_ids)
        if not metadata_rows:
            return None

        normalized_query = " ".join(query.lower().split())
        for row in metadata_rows:
            chunks = tuple(load_chunks(document_version_id=row.document_version_id))
            if not chunks:
                continue
            if _HOLDING_SIGNAL_RE.search(normalized_query):
                answer = _answer_holding_fact(query=query, row=row, chunks=chunks)
                if answer is not None:
                    return answer
            if "pan" in normalized_query:
                answer = _answer_pan_fact(query=query, row=row, chunks=chunks)
                if answer is not None:
                    return answer
            if _AMOUNT_RE.search(normalized_query) or "amount" in normalized_query:
                answer = _answer_amount_fact(query=query, row=row, chunks=chunks)
                if answer is not None:
                    return answer
            if _PARTY_SIGNAL_RE.search(normalized_query):
                answer = _answer_party_fact(query=query, row=row, chunks=chunks)
                if answer is not None:
                    return answer
        return None

    def answer_numeric_fact_question(
        self,
        *,
        query: str,
        document_version_ids: Sequence[int],
    ) -> MetadataAnswer | None:
        numeric_facts = self._repository.fetch_numeric_facts(document_version_ids=document_version_ids)
        price_movements = self._repository.fetch_price_movements(document_version_ids=document_version_ids)
        if not numeric_facts and not price_movements:
            return None

        normalized_query = " ".join(query.lower().split())
        if _PERIOD_PRICE_QUERY_RE.search(normalized_query) and price_movements:
            return _answer_price_movements_by_period(query=query, rows=price_movements)
        if _BEFORE_AFTER_PRICE_RE.search(normalized_query):
            answer = _answer_before_after_price(query=query, facts=numeric_facts, price_movements=price_movements)
            if answer is not None:
                return answer
        if _PRICE_INCREASE_QUERY_RE.search(normalized_query):
            answer = _answer_price_increase(query=query, facts=numeric_facts)
            if answer is not None:
                return answer
        if _HIGHEST_PRICE_QUERY_RE.search(normalized_query):
            answer = _answer_single_numeric_fact(
                query=query,
                facts=numeric_facts,
                fact_types=("highest_price", "percentage_change_from_listing"),
                metadata_type="highest_price",
            )
            if answer is not None:
                return answer
        if _LOWEST_PRICE_QUERY_RE.search(normalized_query):
            answer = _answer_single_numeric_fact(
                query=query,
                facts=numeric_facts,
                fact_types=("lowest_price",),
                metadata_type="lowest_price",
            )
            if answer is not None:
                return answer
        if _LISTING_PRICE_QUERY_RE.search(normalized_query):
            answer = _answer_single_numeric_fact(
                query=query,
                facts=numeric_facts,
                fact_types=("listing_price",),
                metadata_type="listing_price",
            )
            if answer is not None:
                return answer
        if _PRICE_QUERY_RE.search(normalized_query):
            answer = _answer_price_summary(query=query, facts=numeric_facts, price_movements=price_movements)
            if answer is not None:
                return answer
        if _SETTLEMENT_RE.search(normalized_query):
            answer = _answer_single_numeric_fact(
                query=query,
                facts=numeric_facts,
                fact_types=("settlement_amount",),
                metadata_type="settlement_amount",
            )
            if answer is not None:
                return answer
        if _PENALTY_RE.search(normalized_query):
            answer = _answer_single_numeric_fact(
                query=query,
                facts=numeric_facts,
                fact_types=("penalty_amount",),
                metadata_type="penalty_amount",
            )
            if answer is not None:
                return answer
        if _HOLDING_SIGNAL_RE.search(normalized_query):
            answer = _answer_single_numeric_fact(
                query=query,
                facts=numeric_facts,
                fact_types=("share_count", "holding_percentage"),
                metadata_type="holding_numeric_fact",
            )
            if answer is not None:
                return answer
        return None

    def answer_active_matter_follow_up(
        self,
        *,
        query: str,
        document_version_ids: Sequence[int],
        follow_up_intent: str,
    ) -> MetadataAnswer | None:
        load_chunks = getattr(self._repository, "load_chunks", None)
        if not callable(load_chunks):
            return None

        metadata_rows = self._repository.fetch_order_metadata(document_version_ids=document_version_ids)
        if not metadata_rows:
            return None

        if follow_up_intent == "settlement_amount":
            numeric_answer = _answer_single_numeric_fact(
                query=query,
                facts=self._repository.fetch_numeric_facts(
                    document_version_ids=document_version_ids,
                    fact_types=("settlement_amount",),
                ),
                fact_types=("settlement_amount",),
                metadata_type="active_matter_settlement_amount",
            )
            if numeric_answer is not None:
                return numeric_answer
        if follow_up_intent == "penalty":
            numeric_answer = _answer_single_numeric_fact(
                query=query,
                facts=self._repository.fetch_numeric_facts(
                    document_version_ids=document_version_ids,
                    fact_types=("penalty_amount",),
                ),
                fact_types=("penalty_amount",),
                metadata_type="active_matter_penalty",
            )
            if numeric_answer is not None:
                return numeric_answer
        if follow_up_intent == "final_direction":
            final_direction_answer = _answer_final_direction_follow_up(
                rows=metadata_rows,
                load_chunks=load_chunks,
            )
            if final_direction_answer is not None:
                return final_direction_answer

        matches: list[_SemanticFollowUpMatch] = []
        for row in metadata_rows:
            chunks = tuple(load_chunks(document_version_id=row.document_version_id))
            if not chunks:
                continue
            match = _best_semantic_follow_up_match(
                query=query,
                row=row,
                chunks=chunks,
                follow_up_intent=follow_up_intent,
            )
            if match is not None:
                matches.append(match)

        if not matches:
            return None

        best_match = sorted(
            matches,
            key=lambda item: (-item.score, item.row.document_version_id, item.chunk.chunk_id),
        )[0]
        citation = _metadata_chunk_citation(
            row=best_match.row,
            chunk=best_match.chunk,
            citation_number=1,
        )
        prompt_chunk = PromptContextChunk(
            citation_number=1,
            chunk_id=best_match.chunk.chunk_id,
            document_version_id=best_match.row.document_version_id,
            document_id=best_match.row.document_id,
            record_key=best_match.row.record_key,
            bucket_name="metadata",
            title=best_match.row.title,
            page_start=best_match.chunk.page_start,
            page_end=best_match.chunk.page_end,
            section_type=best_match.chunk.section_type or "other",
            section_title=best_match.chunk.section_title,
            detail_url=best_match.row.detail_url,
            pdf_url=best_match.row.pdf_url,
            chunk_text=best_match.chunk.text,
            token_count=max(len(best_match.chunk.text.split()), 1),
            score=best_match.score,
        )
        answer_text, style_debug = apply_grounded_wording_caution(
            answer_text=best_match.answer_text,
            context_chunks=(prompt_chunk,),
            analysis=analyze_query(query),
        )
        return MetadataAnswer(
            answer_text=answer_text,
            citations=(citation,),
            metadata_type=f"active_matter_{follow_up_intent}",
            debug={
                "document_version_id": best_match.row.document_version_id,
                "record_key": best_match.row.record_key,
                "metadata_type": f"active_matter_{follow_up_intent}",
                "follow_up_intent": follow_up_intent,
                "section_type": best_match.chunk.section_type,
                "match_score": round(best_match.score, 4),
                "style_debug": style_debug,
            },
        )

    def answer_observation_question(
        self,
        *,
        query: str,
        document_version_ids: Sequence[int],
    ) -> MetadataAnswer | None:
        load_chunks = getattr(self._repository, "load_chunks", None)
        if not callable(load_chunks):
            return None

        metadata_rows = self._repository.fetch_order_metadata(document_version_ids=document_version_ids)
        if not metadata_rows:
            return None

        for row in metadata_rows:
            chunks = tuple(load_chunks(document_version_id=row.document_version_id))
            if not chunks:
                continue
            relevant_chunks = [chunk for chunk in chunks if _OBSERVATION_TERM_RE.search(chunk.text)]
            selected_chunk = relevant_chunks[0] if relevant_chunks else None
            if selected_chunk is None:
                continue
            answer_text = _render_observation_answer(chunk_text=selected_chunk.text, query=query)
            return MetadataAnswer(
                answer_text=answer_text,
                citations=(
                    Citation(
                        citation_number=1,
                        record_key=row.record_key,
                        title=row.title,
                        page_start=selected_chunk.page_start,
                        page_end=selected_chunk.page_end,
                        section_type="metadata_observation",
                        document_version_id=row.document_version_id,
                        chunk_id=selected_chunk.chunk_id,
                        detail_url=row.detail_url,
                        pdf_url=row.pdf_url,
                    ),
                ),
                metadata_type="order_observations",
                debug={
                    "document_version_id": row.document_version_id,
                    "record_key": row.record_key,
                    "metadata_type": "order_observations",
                },
            )

        primary_row = metadata_rows[0]
        return MetadataAnswer(
            answer_text=_render_missing_observation_answer(query),
            citations=(self._metadata_citation(row=primary_row, citation_number=1),),
            metadata_type="order_observations_missing",
            debug={
                "document_version_id": primary_row.document_version_id,
                "record_key": primary_row.record_key,
                "metadata_type": "order_observations_missing",
            },
        )

    @staticmethod
    def _metadata_citation(*, row: StoredOrderMetadata, citation_number: int) -> Citation:
        return Citation(
            citation_number=citation_number,
            record_key=row.record_key,
            title=row.title,
            page_start=row.signatory_page_start,
            page_end=row.signatory_page_end,
            section_type="metadata_signatory",
            document_version_id=row.document_version_id,
            chunk_id=None,
            detail_url=row.detail_url,
            pdf_url=row.pdf_url,
        )

    @staticmethod
    def _provision_citation(*, row: StoredLegalProvision, citation_number: int) -> Citation:
        return Citation(
            citation_number=citation_number,
            record_key=row.record_key,
            title=row.title,
            page_start=row.page_start,
            page_end=row.page_end,
            section_type="metadata_legal_provision",
            document_version_id=row.document_version_id,
            chunk_id=None,
            detail_url=row.detail_url,
            pdf_url=row.pdf_url,
        )

    def _answer_signatory_from_footer(
        self,
        *,
        metadata_rows: Sequence[StoredOrderMetadata],
    ) -> MetadataAnswer | None:
        load_chunks = getattr(self._repository, "load_chunks", None)
        if not callable(load_chunks):
            return None
        for row in metadata_rows:
            chunks = tuple(load_chunks(document_version_id=row.document_version_id))
            footer_match = _extract_footer_signatory(chunks)
            if footer_match is None:
                continue
            name, designation, page_start, page_end = footer_match
            answer = f"The order was signed by {name}"
            if designation:
                answer += f", {designation}"
            answer += "."
            if row.order_date or row.place:
                details = []
                if row.order_date:
                    details.append(f"date: {row.order_date.isoformat()}")
                if row.place:
                    details.append(f"place: {row.place}")
                if details:
                    answer = answer.rstrip(".") + "; " + "; ".join(details) + "."
            return MetadataAnswer(
                answer_text=answer,
                citations=(
                    Citation(
                        citation_number=1,
                        record_key=row.record_key,
                        title=row.title,
                        page_start=page_start,
                        page_end=page_end,
                        section_type="metadata_signatory",
                        document_version_id=row.document_version_id,
                        chunk_id=None,
                        detail_url=row.detail_url,
                        pdf_url=row.pdf_url,
                    ),
                ),
                metadata_type="signatory",
                debug={
                    "document_version_id": row.document_version_id,
                    "metadata_confidence": row.metadata_confidence,
                    "metadata_type": "signatory_footer_fallback",
                },
            )
        return None


def _group_provisions(rows: Sequence[StoredLegalProvision]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row.statute_name, [])
        if row.section_or_regulation not in grouped[row.statute_name]:
            grouped[row.statute_name].append(row.section_or_regulation)
    return grouped


def _render_observation_answer(*, chunk_text: str, query: str) -> str:
    sentences = [
        " ".join(part.split()).strip()
        for part in _SENTENCE_SPLIT_RE.split(chunk_text)
        if part and _OBSERVATION_TERM_RE.search(part)
    ]
    if not sentences:
        excerpt = " ".join(chunk_text.split())
        sentences = [excerpt[:280].rstrip() + ("..." if len(excerpt) > 280 else "")]
    joined = " ".join(sentences[:2]).strip()
    if _DA_TERM_RE.search(query):
        return f"The indexed text states the Designated Authority observations as follows: {joined}"
    if _ENQUIRY_TERM_RE.search(query):
        return f"The indexed text states the enquiry-report conclusion as follows: {joined}"
    return f"The indexed text records the following observations in this matter: {joined}"


def _render_missing_observation_answer(query: str) -> str:
    if _DA_TERM_RE.search(query):
        return "I could not identify a Designated Authority observation in this matter from the indexed text."
    if _ENQUIRY_TERM_RE.search(query):
        return "I could not identify an enquiry-report conclusion in this matter from the indexed text."
    return "I could not identify a distinct observation section in this matter from the indexed text."


def _render_grouped_provisions(grouped: dict[str, list[str]]) -> str:
    parts = []
    for statute_name, refs in grouped.items():
        if len(refs) == 1:
            parts.append(f"{refs[0]} of {statute_name}")
        elif len(refs) == 2:
            parts.append(f"{refs[0]} and {refs[1]} of {statute_name}")
        else:
            parts.append(f"{', '.join(refs[:-1])}, and {refs[-1]} of {statute_name}")
    if not parts:
        return "no extracted legal provisions"
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts)


def _extract_footer_signatory(
    chunks: Sequence[MetadataChunkText],
) -> tuple[str, str | None, int | None, int | None] | None:
    for chunk in reversed(tuple(chunks)[-6:]):
        inline_match = _extract_inline_footer_signatory(chunk)
        if inline_match is not None:
            return inline_match
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in chunk.text.splitlines()
            if line.strip()
        ]
        if not lines or not any(_SIGNATORY_TRIGGER_RE.search(line) for line in lines):
            continue
        for index, line in enumerate(lines):
            if not _SIGNATORY_TRIGGER_RE.search(line):
                continue
            candidate_lines = [
                candidate
                for candidate in lines[index + 1 : index + 4]
                if candidate and not _SIGNATORY_SKIP_RE.search(candidate)
            ]
            if not candidate_lines:
                continue
            name_line = next(
                (
                    candidate
                    for candidate in candidate_lines
                    if _looks_like_signatory_name(candidate)
                ),
                None,
            )
            if name_line is None:
                continue
            designation_lines = [
                _to_display_case(candidate)
                for candidate in candidate_lines[candidate_lines.index(name_line) + 1 :]
                if _looks_like_designation(candidate)
            ]
            designation = " / ".join(designation_lines) if designation_lines else None
            return (
                _to_display_case(name_line),
                designation,
                chunk.page_start,
                chunk.page_end,
            )
    return None


def _extract_inline_footer_signatory(
    chunk: MetadataChunkText,
) -> tuple[str, str | None, int | None, int | None] | None:
    normalized = re.sub(r"\s*\|\s*", " ", chunk.text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    for pattern in _INLINE_SIGNATORY_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        name = _normalize_signatory_display_name(match.group("name"))
        designation = _normalize_signatory_display_name(match.group("designation"))
        if not name or not designation:
            continue
        if not _looks_like_candidate_person_name(name):
            continue
        return name, designation, chunk.page_start, chunk.page_end
    return None


def _looks_like_signatory_name(value: str) -> bool:
    return bool(
        _SIGNATORY_NAME_RE.fullmatch(value)
        and any(character.isalpha() for character in value)
        and value.upper() == value
    )


def _looks_like_designation(value: str) -> bool:
    normalized = value.strip().lower()
    return any(token in normalized for token in _SIGNATORY_DESIGNATION_TOKENS)


def _to_display_case(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).title()


def _normalize_signatory_display_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip(" ,.;:-"))
    if not cleaned:
        return ""
    return cleaned.title()


def _looks_like_candidate_person_name(value: str) -> bool:
    lowered = value.lower()
    if any(
        token in lowered
        for token in (
            "order in the matter of",
            "copy of this order",
            "securities and exchange board of india",
            "the banks are directed",
            "issue no.",
            "page ",
        )
    ):
        return False
    tokens = [token for token in re.findall(r"[a-z]+", lowered) if token]
    return len(tokens) >= 2


def _metadata_signatory_is_usable(row: StoredOrderMetadata) -> bool:
    if row.signatory_name and _looks_like_candidate_person_name(row.signatory_name):
        return True
    return False


def _answer_holding_fact(
    *,
    query: str,
    row: StoredOrderMetadata,
    chunks: Sequence[MetadataChunkText],
) -> MetadataAnswer | None:
    relevant_chunks = _select_relevant_chunks(query=query, chunks=chunks, signal_re=_HOLDING_SIGNAL_RE)
    if not relevant_chunks:
        return None
    prompt_chunks = tuple(
        PromptContextChunk(
            citation_number=index,
            chunk_id=chunk.chunk_id,
            document_version_id=row.document_version_id,
            document_id=row.document_id,
            record_key=row.record_key,
            bucket_name="orders",
            title=row.title,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_type="metadata_exact_fact",
            section_title=None,
            detail_url=row.detail_url,
            pdf_url=row.pdf_url,
            chunk_text=chunk.text,
            token_count=max(1, len(chunk.text.split())),
            score=1.0,
        )
        for index, chunk in enumerate(relevant_chunks[:3], start=1)
    )
    answer_text, style_debug = apply_grounded_wording_caution(
        answer_text="The order says the queried subject holds shares as stated in the cited text.",
        context_chunks=prompt_chunks,
        analysis=analyze_query(query),
    )
    if not style_debug.get("holding_rewrite_used"):
        return None
    return MetadataAnswer(
        answer_text=answer_text,
        citations=tuple(
            _metadata_chunk_citation(row=row, chunk=chunk, citation_number=index)
            for index, chunk in enumerate(relevant_chunks[:2], start=1)
        ),
        metadata_type="holding_fact",
        debug={
            "document_version_id": row.document_version_id,
            "metadata_type": "holding_fact",
            "style_debug": style_debug,
        },
    )


def _answer_pan_fact(
    *,
    query: str,
    row: StoredOrderMetadata,
    chunks: Sequence[MetadataChunkText],
) -> MetadataAnswer | None:
    relevant_chunks = _select_relevant_chunks(query=query, chunks=chunks, signal_re=_PAN_RE)
    for chunk in relevant_chunks:
        match = _PAN_RE.search(chunk.text.upper())
        if match is None:
            continue
        answer = f"The cited order text lists PAN {match.group(0)}."
        subject = _extract_query_subject(query)
        if subject:
            answer = f"The cited order text lists PAN {match.group(0)} for {subject}."
        return MetadataAnswer(
            answer_text=answer,
            citations=(_metadata_chunk_citation(row=row, chunk=chunk, citation_number=1),),
            metadata_type="pan_fact",
            debug={
                "document_version_id": row.document_version_id,
                "metadata_type": "pan_fact",
            },
        )
    return None


def _answer_amount_fact(
    *,
    query: str,
    row: StoredOrderMetadata,
    chunks: Sequence[MetadataChunkText],
) -> MetadataAnswer | None:
    relevant_chunks = _select_relevant_chunks(query=query, chunks=chunks, signal_re=_AMOUNT_RE)
    for chunk in relevant_chunks:
        matches = tuple(dict.fromkeys(match.group(0) for match in _AMOUNT_RE.finditer(chunk.text)))
        if not matches:
            continue
        rendered = matches[0] if len(matches) == 1 else f"{matches[0]} (and related amounts in the cited text)"
        return MetadataAnswer(
            answer_text=f"The cited order text mentions an amount of {rendered}.",
            citations=(_metadata_chunk_citation(row=row, chunk=chunk, citation_number=1),),
            metadata_type="amount_fact",
            debug={
                "document_version_id": row.document_version_id,
                "metadata_type": "amount_fact",
            },
        )
    return None


def _answer_single_numeric_fact(
    *,
    query: str,
    facts: Sequence[StoredNumericFact],
    fact_types: Sequence[str],
    metadata_type: str,
) -> MetadataAnswer | None:
    primary_facts, _ = _select_primary_numeric_record(facts=facts, price_movements=())
    ordered = [
        fact
        for fact_type in fact_types
        for fact in primary_facts
        if fact.fact_type == fact_type
    ]
    if not ordered:
        return None

    primary = ordered[0]
    if metadata_type == "highest_price":
        pct_fact = next(
            (fact for fact in ordered[1:] if fact.fact_type == "percentage_change_from_listing"),
            None,
        )
        answer_text = f"The highest price recorded in this matter was {_display_numeric_fact(primary)}"
        if primary.context_label:
            answer_text += f" {primary.context_label}"
        if pct_fact is not None:
            answer_text += f", which was {_display_numeric_fact(pct_fact)} of the listing price"
        answer_text += "."
        citations = tuple(
            _numeric_fact_citation(fact=fact, citation_number=index)
            for index, fact in enumerate((primary, *(tuple([pct_fact]) if pct_fact else ())), start=1)
        )
    elif metadata_type == "listing_price":
        answer_text = f"The listing price recorded in this matter was {_display_numeric_fact(primary)}"
        if primary.context_label:
            answer_text += f" {primary.context_label}"
        answer_text += "."
        citations = (_numeric_fact_citation(fact=primary, citation_number=1),)
    elif metadata_type in {"settlement_amount", "active_matter_settlement_amount"}:
        answer_text = f"The settlement amount recorded in this matter was {_display_numeric_fact(primary)}."
        citations = (_numeric_fact_citation(fact=primary, citation_number=1),)
    elif metadata_type in {"penalty_amount", "active_matter_penalty"}:
        answer_text = f"The penalty amount recorded in this matter was {_display_numeric_fact(primary)}."
        citations = (_numeric_fact_citation(fact=primary, citation_number=1),)
    elif metadata_type == "holding_numeric_fact":
        rendered = " and ".join(_display_numeric_fact(fact) for fact in ordered[:2])
        answer_text = f"The extracted holding figures in this matter include {rendered}."
        citations = tuple(
            _numeric_fact_citation(fact=fact, citation_number=index)
            for index, fact in enumerate(ordered[:2], start=1)
        )
    else:
        answer_text = f"The extracted figure in this matter is {_display_numeric_fact(primary)}."
        citations = (_numeric_fact_citation(fact=primary, citation_number=1),)
    return MetadataAnswer(
        answer_text=answer_text,
        citations=citations,
        metadata_type=metadata_type,
        debug={
            "document_version_id": primary.document_version_id,
            "record_key": primary.record_key,
            "metadata_type": metadata_type,
            "fact_types": list(dict.fromkeys(fact.fact_type for fact in ordered[:3])),
        },
    )


def _answer_price_increase(
    *,
    query: str,
    facts: Sequence[StoredNumericFact],
) -> MetadataAnswer | None:
    del query
    primary_facts, _ = _select_primary_numeric_record(facts=facts, price_movements=())
    listing = _first_fact(primary_facts, "listing_price")
    closing = _first_fact(primary_facts, "closing_price")
    pct = _first_fact(primary_facts, "percentage_change")
    highest = _first_fact(primary_facts, "highest_price")
    highest_pct = _first_fact(primary_facts, "percentage_change_from_listing")
    if listing is None or pct is None:
        return None
    answer_text = (
        f"The share price increased by {_display_numeric_fact(pct)}"
    )
    if listing is not None and closing is not None:
        answer_text += (
            f", from {_display_numeric_fact(listing)}"
            f" {listing.context_label or ''}".rstrip()
            + f" to {_display_numeric_fact(closing)}"
        )
        if closing.context_label:
            answer_text += _render_context_label(closing.context_label)
    answer_text += "."
    if highest is not None:
        answer_text += f" It also reached a high of {_display_numeric_fact(highest)}"
        if highest.context_label:
            answer_text += f" {highest.context_label}"
        if highest_pct is not None:
            answer_text += f", which was {_display_numeric_fact(highest_pct)} of the listing price"
        answer_text += "."
    citation_facts = tuple(
        fact
        for fact in (listing, closing, pct, highest)
        if fact is not None
    )[:3]
    return MetadataAnswer(
        answer_text=answer_text,
        citations=tuple(
            _numeric_fact_citation(fact=fact, citation_number=index)
            for index, fact in enumerate(citation_facts, start=1)
        ),
        metadata_type="price_increase",
        debug={
            "document_version_id": listing.document_version_id,
            "record_key": listing.record_key,
            "metadata_type": "price_increase",
        },
    )


def _answer_before_after_price(
    *,
    query: str,
    facts: Sequence[StoredNumericFact],
    price_movements: Sequence[StoredPriceMovement],
) -> MetadataAnswer | None:
    del query
    primary_facts, primary_rows = _select_primary_numeric_record(
        facts=facts,
        price_movements=price_movements,
    )
    listing = _first_fact(primary_facts, "listing_price")
    closing = _first_fact(primary_facts, "closing_price")
    highest = _first_fact(primary_facts, "highest_price")
    if listing is None and primary_rows:
        listing = _movement_fact_proxy(primary_rows[0], fact_type="listing_price")
    if closing is None and primary_rows:
        closing = _movement_fact_proxy(primary_rows[-1], fact_type="closing_price")
    if listing is None or closing is None:
        return None

    answer_text = (
        f"The price before the increase was {_display_numeric_fact(listing)}"
        f" {listing.context_label or ''}".rstrip()
        + f", and it was {_display_numeric_fact(closing)}"
    )
    if closing.context_label:
        answer_text += _render_context_label(closing.context_label)
    answer_text += "."
    if highest is not None:
        answer_text += f" The highest price during the period was {_display_numeric_fact(highest)}"
        if highest.context_label:
            answer_text += f" {highest.context_label}"
        answer_text += "."
    citation_facts = tuple(fact for fact in (listing, closing, highest) if fact is not None)
    return MetadataAnswer(
        answer_text=answer_text,
        citations=tuple(
            _numeric_fact_citation(fact=fact, citation_number=index)
            for index, fact in enumerate(citation_facts[:3], start=1)
        ),
        metadata_type="before_after_price",
        debug={
            "document_version_id": listing.document_version_id,
            "record_key": listing.record_key,
            "metadata_type": "before_after_price",
        },
    )


def _answer_price_summary(
    *,
    query: str,
    facts: Sequence[StoredNumericFact],
    price_movements: Sequence[StoredPriceMovement],
) -> MetadataAnswer | None:
    answer = _answer_price_increase(query=query, facts=facts)
    if answer is not None:
        return answer
    return _answer_price_movements_by_period(query=query, rows=price_movements)


def _answer_price_movements_by_period(
    *,
    query: str,
    rows: Sequence[StoredPriceMovement],
) -> MetadataAnswer | None:
    del query
    _, primary_rows = _select_primary_numeric_record(facts=(), price_movements=rows)
    if not primary_rows:
        return None
    ordered_rows = sorted(primary_rows, key=_price_movement_sort_key)
    rendered_rows = []
    for row in ordered_rows:
        parts = [f"{row.period_label}"]
        if row.period_start_text and row.period_end_text:
            parts.append(f"({row.period_start_text} to {row.period_end_text})")
        metrics = []
        if row.start_price is not None:
            metrics.append(f"start Rs.{row.start_price:g}")
        if row.high_price is not None:
            metrics.append(f"high Rs.{row.high_price:g}")
        if row.low_price is not None:
            metrics.append(f"low Rs.{row.low_price:g}")
        if row.end_price is not None:
            metrics.append(f"end Rs.{row.end_price:g}")
        if row.pct_change is not None:
            prefix = "+" if row.pct_change > 0 else ""
            metrics.append(f"{prefix}{row.pct_change:g}%")
        parts.append(": " + ", ".join(metrics))
        rendered_rows.append(" ".join(parts))
    return MetadataAnswer(
        answer_text=" ".join(rendered_rows),
        citations=tuple(
            _price_movement_citation(row=row, citation_number=index)
            for index, row in enumerate(ordered_rows[:4], start=1)
        ),
        metadata_type="price_movements_by_period",
        debug={
            "document_version_id": ordered_rows[0].document_version_id,
            "record_key": ordered_rows[0].record_key,
            "metadata_type": "price_movements_by_period",
            "row_count": len(ordered_rows),
        },
    )


def _answer_final_direction_follow_up(
    *,
    rows: Sequence[StoredOrderMetadata],
    load_chunks,
) -> MetadataAnswer | None:
    best_answer: MetadataAnswer | None = None
    for row in rows:
        chunks = tuple(load_chunks(document_version_id=row.document_version_id))
        if not chunks:
            continue
        relevant_chunks = tuple(
            chunk
            for chunk in chunks
            if (
                (chunk.section_type or "").strip() in {"directions", "operative_order"}
                or re.search(
                    r"\b(?:refund|repayment|debarred|prohibited from accessing the securities market|penalt(?:y|ies)|do hereby issue the following directions)\b",
                    chunk.text,
                    re.IGNORECASE,
                )
            )
        )
        if not relevant_chunks:
            continue
        answer_text = _render_final_direction_follow_up_answer(relevant_chunks)
        if answer_text is None:
            continue
        citations = tuple(
            _metadata_chunk_citation(row=row, chunk=chunk, citation_number=index)
            for index, chunk in enumerate(relevant_chunks[:2], start=1)
        )
        candidate = MetadataAnswer(
            answer_text=answer_text,
            citations=citations,
            metadata_type="active_matter_final_direction",
            debug={
                "document_version_id": row.document_version_id,
                "record_key": row.record_key,
                "metadata_type": "active_matter_final_direction",
                "follow_up_intent": "final_direction",
                "chunk_ids": tuple(chunk.chunk_id for chunk in relevant_chunks[:2]),
            },
        )
        if best_answer is None or len(citations) > len(best_answer.citations):
            best_answer = candidate
    return best_answer


def _render_final_direction_follow_up_answer(
    chunks: Sequence[MetadataChunkText],
) -> str | None:
    combined_text = " ".join(" ".join(chunk.text.split()) for chunk in chunks)
    if not combined_text:
        return None
    has_directional_text = re.search(
        r"\b(?:refund|repayment|debarred|penalt(?:y|ies)|do hereby issue the following directions)\b",
        combined_text,
        re.IGNORECASE,
    )
    if has_directional_text is None:
        return None
    clauses: list[str] = []
    if re.search(r"\brefund|repayment\b", combined_text, re.IGNORECASE):
        clauses.append("ordered refunds of money collected from investors and completion of the repayment process")
    debar_match = re.search(
        r"\bdebarred\b[^.]*?(?:for a period of [^.]+|until[^.]+)?",
        combined_text,
        re.IGNORECASE,
    )
    if debar_match is not None:
        debar_text = debar_match.group(0)
        duration_match = re.search(r"for a period of [^.]+|until[^.]+", debar_text, re.IGNORECASE)
        if duration_match is not None:
            clauses.append(
                "debarred the noticee from the securities market "
                + duration_match.group(0).strip(" .")
            )
        else:
            clauses.append("debarred the noticee from the securities market")
    penalty_anchor = re.search(r"\bpenalt(?:y|ies)\b", combined_text, re.IGNORECASE)
    if penalty_anchor is not None:
        local_window = combined_text[penalty_anchor.start(): penalty_anchor.start() + 500]
        amounts = re.findall(r"Rs\.?\s*[0-9,]+(?:\.[0-9]+)?", local_window, flags=re.IGNORECASE)
        if amounts:
            if len(amounts) == 1:
                clauses.append(f"imposed a monetary penalty of {amounts[0]}")
            else:
                clauses.append(
                    "imposed monetary penalties of "
                    + ", ".join(amounts[:-1])
                    + f" and {amounts[-1]}"
                )
        else:
            clauses.append("imposed monetary penalties")
    if not clauses:
        return None
    if len(clauses) == 1:
        return f"SEBI {clauses[0]}."
    return "SEBI " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}."


def _best_semantic_follow_up_match(
    *,
    query: str,
    row: StoredOrderMetadata,
    chunks: Sequence[MetadataChunkText],
    follow_up_intent: str,
) -> _SemanticFollowUpMatch | None:
    query_terms = tuple(
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) >= 3 and token not in {"what", "the", "was", "did", "this", "that", "order", "matter"}
    )
    best: _SemanticFollowUpMatch | None = None
    for chunk in chunks:
        score = _score_semantic_follow_up_chunk(
            follow_up_intent=follow_up_intent,
            query_terms=query_terms,
            chunk=chunk,
        )
        if score <= 0.0:
            continue
        answer_text = _extract_semantic_follow_up_answer(
            follow_up_intent=follow_up_intent,
            chunk=chunk,
        )
        if answer_text is None:
            continue
        candidate = _SemanticFollowUpMatch(
            row=row,
            chunk=chunk,
            answer_text=answer_text,
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _score_semantic_follow_up_chunk(
    *,
    follow_up_intent: str,
    query_terms: Sequence[str],
    chunk: MetadataChunkText,
) -> float:
    normalized_text = " ".join(chunk.text.lower().split())
    if not normalized_text:
        return 0.0
    section_type = (chunk.section_type or "other").strip()
    if follow_up_intent in _ACTIVE_FOLLOW_UP_RESULT_INTENTS:
        score = _SECTION_PRIORS_RESULT.get(section_type, _SECTION_PRIORS_RESULT["other"])
    else:
        score = _SECTION_PRIORS_DEFAULT.get(section_type, _SECTION_PRIORS_DEFAULT["other"])
    for term in _ACTIVE_MATTER_FOLLOW_UP_TERMS.get(follow_up_intent, ()):
        if term in normalized_text:
            score += 2.2 if " " in term else 1.0
    for token in query_terms:
        if token in normalized_text:
            score += 0.12
    if follow_up_intent == "settlement_amount":
        if _AMOUNT_RE.search(chunk.text):
            score += 1.8
        else:
            score -= 1.5
    elif follow_up_intent == "penalty":
        if _AMOUNT_RE.search(chunk.text):
            score += 1.1
    elif follow_up_intent in _ACTIVE_FOLLOW_UP_RESULT_INTENTS:
        if _DECISION_TERM_RE.search(chunk.text):
            score += 1.4
        else:
            score -= 0.8
    elif follow_up_intent == "exemption_granted":
        if "granted" in normalized_text and "exemption" in normalized_text:
            score += 1.6
    return score


def _extract_semantic_follow_up_answer(
    *,
    follow_up_intent: str,
    chunk: MetadataChunkText,
) -> str | None:
    sentences = [
        _clean_sentence(sentence)
        for sentence in _SENTENCE_SPLIT_RE.split(chunk.text)
    ]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return None
    intent_terms = _ACTIVE_MATTER_FOLLOW_UP_TERMS.get(follow_up_intent, ())
    scored_sentences: list[tuple[float, str]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        score = 0.0
        for term in intent_terms:
            if term in lowered:
                score += 2.0 if " " in term else 0.8
        if follow_up_intent == "settlement_amount" and _AMOUNT_RE.search(sentence):
            score += 2.4
        if follow_up_intent == "penalty" and _AMOUNT_RE.search(sentence):
            score += 1.5
        if follow_up_intent in _ACTIVE_FOLLOW_UP_RESULT_INTENTS and _DECISION_TERM_RE.search(sentence):
            score += 1.7
        if follow_up_intent == "da_observed" and _OBSERVATION_TERM_RE.search(sentence):
            score += 1.5
        if score > 0.0:
            scored_sentences.append((score, sentence))

    if not scored_sentences:
        fallback = sentences[0]
        if follow_up_intent == "settlement_amount" and not _AMOUNT_RE.search(fallback):
            return None
        if follow_up_intent in _ACTIVE_FOLLOW_UP_RESULT_INTENTS and not _DECISION_TERM_RE.search(fallback):
            return None
        return fallback

    scored_sentences.sort(key=lambda item: (-item[0], len(item[1])))
    selected = [scored_sentences[0][1]]
    if len(scored_sentences) > 1 and scored_sentences[1][0] >= max(scored_sentences[0][0] - 1.2, 1.5):
        selected.append(scored_sentences[1][1])
    return " ".join(selected[:2])


def _clean_sentence(value: str) -> str:
    sentence = " ".join(value.split()).strip(" .;")
    if not sentence:
        return ""
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _answer_party_fact(
    *,
    query: str,
    row: StoredOrderMetadata,
    chunks: Sequence[MetadataChunkText],
) -> MetadataAnswer | None:
    del query, chunks
    lowered_title = row.title.lower()
    if " vs " in lowered_title or " versus " in lowered_title or " v " in lowered_title:
        parts = re.split(r"\b(?:vs\.?|versus|v\.)\b", row.title, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            answer = f"The title identifies the matter as between {parts[0].strip()} and {parts[1].strip()}."
            return MetadataAnswer(
                answer_text=answer,
                citations=(Citation(
                    citation_number=1,
                    record_key=row.record_key,
                    title=row.title,
                    page_start=1,
                    page_end=1,
                    section_type="metadata_exact_fact",
                    document_version_id=row.document_version_id,
                    chunk_id=None,
                    detail_url=row.detail_url,
                    pdf_url=row.pdf_url,
                ),),
                metadata_type="party_fact",
                debug={
                    "document_version_id": row.document_version_id,
                    "metadata_type": "party_fact",
                },
            )
    if "in the matter of " in lowered_title:
        subject = row.title[lowered_title.index("in the matter of ") + len("in the matter of ") :].strip(" .")
        if subject:
            return MetadataAnswer(
                answer_text=f"The title identifies the matter as concerning {subject}.",
                citations=(Citation(
                    citation_number=1,
                    record_key=row.record_key,
                    title=row.title,
                    page_start=1,
                    page_end=1,
                    section_type="metadata_exact_fact",
                    document_version_id=row.document_version_id,
                    chunk_id=None,
                    detail_url=row.detail_url,
                    pdf_url=row.pdf_url,
                ),),
                metadata_type="party_fact",
                debug={
                    "document_version_id": row.document_version_id,
                    "metadata_type": "party_fact",
                },
            )
    return None


def _metadata_chunk_citation(
    *,
    row: StoredOrderMetadata,
    chunk: MetadataChunkText,
    citation_number: int,
) -> Citation:
    return Citation(
        citation_number=citation_number,
        record_key=row.record_key,
        title=row.title,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_type="metadata_exact_fact",
        document_version_id=row.document_version_id,
        chunk_id=chunk.chunk_id,
        detail_url=row.detail_url,
        pdf_url=row.pdf_url,
    )


def _numeric_fact_citation(*, fact: StoredNumericFact, citation_number: int) -> Citation:
    return Citation(
        citation_number=citation_number,
        record_key=fact.record_key,
        title=fact.title,
        page_start=fact.page_start,
        page_end=fact.page_end,
        section_type="metadata_numeric_fact",
        document_version_id=fact.document_version_id,
        chunk_id=None,
        detail_url=fact.detail_url,
        pdf_url=fact.pdf_url,
    )


def _price_movement_citation(*, row: StoredPriceMovement, citation_number: int) -> Citation:
    return Citation(
        citation_number=citation_number,
        record_key=row.record_key,
        title=row.title,
        page_start=row.page_start,
        page_end=row.page_end,
        section_type="metadata_price_movement",
        document_version_id=row.document_version_id,
        chunk_id=None,
        detail_url=row.detail_url,
        pdf_url=row.pdf_url,
    )


def _select_relevant_chunks(
    *,
    query: str,
    chunks: Sequence[MetadataChunkText],
    signal_re: re.Pattern[str],
) -> tuple[MetadataChunkText, ...]:
    subject_tokens = tuple(token for token in re.findall(r"[a-z0-9]+", (_extract_query_subject(query) or "").lower()) if token)
    scored: list[tuple[int, MetadataChunkText]] = []
    for chunk in chunks:
        lowered = chunk.text.lower()
        if not signal_re.search(chunk.text):
            continue
        score = 1
        if subject_tokens and all(token in lowered for token in subject_tokens[:2]):
            score += 3
        elif subject_tokens and any(token in lowered for token in subject_tokens):
            score += 1
        if "proposed" in lowered:
            score += 1
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return tuple(chunk for _, chunk in scored[:3])


def _extract_query_subject(query: str) -> str | None:
    match = _QUERY_SUBJECT_RE.search(query)
    if match is None:
        return None
    subject = re.sub(r"\s+", " ", match.group("subject").strip(" ,.-"))
    return subject or None


def _first_fact(
    facts: Sequence[StoredNumericFact],
    fact_type: str,
) -> StoredNumericFact | None:
    return next((fact for fact in facts if fact.fact_type == fact_type), None)


def _display_numeric_fact(fact: StoredNumericFact) -> str:
    return fact.value_text or (str(fact.value_numeric) if fact.value_numeric is not None else "the cited figure")


def _render_context_label(context_label: str) -> str:
    label = context_label.strip()
    if not label:
        return ""
    if label.lower().startswith(("on ", "as on ", "as of ", "during ")):
        return f" {label}"
    return f" for {label}"


def _select_primary_numeric_record(
    *,
    facts: Sequence[StoredNumericFact],
    price_movements: Sequence[StoredPriceMovement],
) -> tuple[tuple[StoredNumericFact, ...], tuple[StoredPriceMovement, ...]]:
    record_scores: dict[str, float] = {}
    for fact in facts:
        record_scores[fact.record_key] = record_scores.get(fact.record_key, 0.0) + 1.0
        if fact.fact_type in {
            "listing_price",
            "closing_price",
            "percentage_change",
            "highest_price",
            "settlement_amount",
            "penalty_amount",
        }:
            record_scores[fact.record_key] += 1.0
    for row in price_movements:
        record_scores[row.record_key] = record_scores.get(row.record_key, 0.0) + 2.0
    if not record_scores:
        return tuple(facts), tuple(price_movements)
    primary_record_key = sorted(
        record_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]
    return (
        tuple(fact for fact in facts if fact.record_key == primary_record_key),
        tuple(row for row in price_movements if row.record_key == primary_record_key),
    )


def _movement_fact_proxy(
    row: StoredPriceMovement,
    *,
    fact_type: str,
) -> StoredNumericFact | None:
    if fact_type == "listing_price":
        value = row.start_price
        context = f"for {row.period_label} ({row.period_start_text})" if row.period_start_text else row.period_label
    else:
        value = row.end_price
        context = f"for {row.period_label} ({row.period_end_text})" if row.period_end_text else row.period_label
    if value is None:
        return None
    return StoredNumericFact(
        numeric_fact_id=-1,
        document_version_id=row.document_version_id,
        document_id=row.document_id,
        record_key=row.record_key,
        title=row.title,
        detail_url=row.detail_url,
        pdf_url=row.pdf_url,
        fact_type=fact_type,
        value_text=f"Rs.{value:g}",
        value_numeric=value,
        unit="INR",
        context_label=context,
        page_start=row.page_start,
        page_end=row.page_end,
        row_sha256=None,
        updated_at=row.updated_at,
    )


def _price_movement_sort_key(row: StoredPriceMovement) -> tuple[int, str]:
    label_match = re.search(r"(\d+)$", row.period_label)
    label_number = int(label_match.group(1)) if label_match is not None else 99
    return (label_number, row.period_start_text or row.period_label)
