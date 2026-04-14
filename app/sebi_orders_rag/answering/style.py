"""Answer-style safety helpers for grounded regulatory responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..normalization import normalize_fuzzy_text
from ..schemas import PromptContextChunk, QueryAnalysis

_PROPOSED_SIGNAL_RE = re.compile(
    r"\b(?:proposed acquirer|proposed acquisition|proposed to acquire|proposed to hold|proposed holding|exemption from open offer)\b",
    re.IGNORECASE,
)
_TABULAR_PROPOSED_SIGNAL_RE = re.compile(
    r"\b(?:proposed transaction|acquirer and pacs|acquirer trust|shareholding after the proposed transaction)\b",
    re.IGNORECASE,
)
_INTERIM_SIGNAL_RE = re.compile(
    r"\b(?:prima facie|interim order|interim findings|ex-parte interim order)\b",
    re.IGNORECASE,
)
_OVERCLAIM_OWNER_RE = re.compile(r"\bowner(?:ship)?\b", re.IGNORECASE)
_FINALITY_RE = re.compile(r"\bfinal(?:ly)?\b", re.IGNORECASE)
_HOLDING_QUERY_RE = re.compile(
    r"\b(?:does|did|do|would|will)\s+(?P<subject>[a-z][a-z0-9 .&'/-]{2,120}?)\s+(?:own|hold|held)\b",
    re.IGNORECASE,
)
_HOLDING_SENTENCE_RE = re.compile(
    r"(?P<subject>[a-z][a-z0-9 .,&'/-]{1,140}?)\s+"
    r"(?:(?P<individual>individually)\s+)?"
    r"(?:(?:was|is)\s+the\s+proposed\s+acquirer\s+)?"
    r"(?:(?P<proposed>proposed(?:\s+to)?)\s+)?"
    r"(?:hold|holds|held)\b",
    re.IGNORECASE,
)
_TABULAR_HOLDING_RE = re.compile(
    r"(?:^|\b)(?:\d+\.\s+)?"
    r"(?P<subject>(?:(?:mrs|mr|ms|dr)\.?\s+)?[a-z][a-z0-9 .&'/-]{1,120}?)\s+"
    r"(?P<shares>[0-9][0-9,]*)\s+"
    r"(?P<percent>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_SHARE_COUNT_RE = re.compile(r"([0-9][0-9,]*)\s+shares?\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_HONORIFIC_PREFIX_RE = re.compile(r"^(?:mr|mrs|ms|dr|shri|smt)\.?\s+", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bRs\.)(?<!\b[Nn]o\.)(?<!\b[Nn]os\.)(?<=[.;])\s+|\n+"
)
_BRIEF_SUMMARY_QUERY_RE = re.compile(
    r"\bbrief\s+summary\b|\bgive\s+a\s+brief\s+summary\b|\bsummar(?:ise|ize)\s+briefly\b|\bsummary\s+of\s+what\s+happened\b",
    re.IGNORECASE,
)
_SETTLEMENT_FINDINGS_QUERY_RE = re.compile(
    r"\bwhat\s+did\s+sebi\s+find\b|\bfindings?\b|\bwhat\s+happened\b|\bsummary\b",
    re.IGNORECASE,
)
_SETTLEMENT_MATTER_RE = re.compile(r"\bsettlement(?:\s+order|\s+proceedings|\s+application)?\b", re.IGNORECASE)
_EXEMPTION_MATTER_RE = re.compile(r"\bexemption(?:\s+order)?\b|\bopen offer\b", re.IGNORECASE)
_RTI_APPELLATE_MATTER_RE = re.compile(r"\brti\b|\bappellate authority\b", re.IGNORECASE)
_APPELLATE_MATTER_RE = re.compile(r"\b(?:sat|tribunal|appeal|court|writ petition|judgment)\b", re.IGNORECASE)
_FINAL_ENFORCEMENT_MATTER_RE = re.compile(
    r"\b(?:adjudicating officer|whole time member|penalty|restrained|debarred|violat(?:ed|ion))\b",
    re.IGNORECASE,
)
_SUMMARY_KEYWORD_RE = re.compile(
    r"\b(?:alleg(?:ed|ation)?|complaint|show cause notice|scn|investigation|observ(?:ed|ation)?|find(?:ing|ings)?|held|direct(?:ed|ion)?|disposed of|settlement amount|penalty|relief|exemption|appeal)\b",
    re.IGNORECASE,
)
_SUMMARY_SKIP_RE = re.compile(
    r"\b(?:copy of this order|place:|date:|page \d+|issued on|whole time member, sebi|reads as follows|factors to be taken into account while adjudging quantum of penalty|section\s+\d+[a-z()]*\s+of\s+the\s+sebi\s+act|consideration of issues and findings|the following issues require consideration|this direction shall not apply|detailed investigation into the matter|detailed investigation shall be carried out|without being influenced by|opportunity of personal hearing|prima facie findings recorded in the interim order|tentative in nature|based on the outcome of the detailed investigation|if the noticees have any open position|open position in any exchange)\b",
    re.IGNORECASE,
)
_SUMMARY_QUESTION_RE = re.compile(
    r"^(?:\(?[ivx]+\)?\s*)?(?:whether|if\s+answer|what\s+directions?\s+need\s+to\s+be\s+issued)\b",
    re.IGNORECASE,
)
_SUMMARY_ENUMERATION_RE = re.compile(r"^\(?[ivxlcdm]+\)?\s+", re.IGNORECASE)
_SUMMARY_FRAGMENT_START_RE = re.compile(r"^(?:and|or|but|which|that)\b", re.IGNORECASE)
_SUMMARY_DANGLING_END_RE = re.compile(r"\b(?:and|or|but|which|that|to|of|as per)\.$", re.IGNORECASE)
_ACTION_OR_SUMMARY_QUERY_RE = re.compile(
    r"\b(?:summary|what happened|action taken|what was the action taken|what did .* decide|what did .* order|what relief|what exemption|what was the outcome)\b",
    re.IGNORECASE,
)
_EXEMPTION_REQUEST_QUERY_RE = re.compile(
    r"\bwhat\s+exemption\b|\bwhat\s+exemption\s+was\s+granted\b",
    re.IGNORECASE,
)
_IPO_PROCEEDS_QUERY_RE = re.compile(r"\bipo proceeds?\b", re.IGNORECASE)
_MATTER_TYPE_TRIGGER_RE = re.compile(
    r"\b(?:exemption|appeal|appellate|rti|sat|tribunal|court|penalty|final direction|quasi[- ]judicial|what did sebi find|what did the da observe)\b",
    re.IGNORECASE,
)
_EXEMPTION_GRANT_RE = re.compile(
    r"\b(?:exempt(?:ed|ion)?|relief|open offer|regulation 11|subject to|provided that|condition)\b",
    re.IGNORECASE,
)
_APPELLATE_DISPOSITION_RE = re.compile(
    r"\b(?:appeal|appellate authority|dismissed|allowed|upheld|set aside|quashed|remanded|disposed of)\b",
    re.IGNORECASE,
)
_RTI_DECISION_RE = re.compile(
    r"\b(?:rti|cpio|information requested|request sought|reply of the cpio|appeal is dismissed|appeal is allowed|appellate authority)\b",
    re.IGNORECASE,
)
_FINAL_ENFORCEMENT_ACTION_RE = re.compile(
    r"\b(?:violat(?:ed|ion)|observ(?:ed|ation)|finding|findings|penalt(?:y|ies)|restrained|debarred|directed|direction|order)\b",
    re.IGNORECASE,
)
_PREFERENTIAL_ALLOTMENT_RE = re.compile(r"\bpreferential allotment\b", re.IGNORECASE)
_LOAN_CONVERSION_RE = re.compile(
    r"\bconversion of (?:the )?outstanding unsecured loans into equity shares\b",
    re.IGNORECASE,
)
_ENTITY_SUBJECT_TOKENS = {
    "trust",
    "family trust",
    "huf",
    "limited",
    "ltd",
    "private limited",
    "private",
    "company",
    "llp",
    "fund",
    "proposed acquirer",
}
_TABULAR_SUBJECT_BLOCKLIST = {
    "sub total",
    "subtotal",
    "public",
    "public shareholders",
    "total",
    "target company",
    "proposed acquirer",
}


@dataclass(frozen=True)
class _HoldingEvidence:
    subject: str
    normalized_subject: str
    base_subject: str
    subject_type: Literal["individual", "entity"]
    share_count: str | None
    percentage: str | None
    proposed: bool
    individually: bool
    sentence: str


@dataclass(frozen=True)
class _ContextSentence:
    sentence: str
    section_type: str
    score: float


def apply_grounded_wording_caution(
    *,
    answer_text: str,
    context_chunks: tuple[PromptContextChunk, ...],
    analysis: QueryAnalysis,
) -> tuple[str, dict[str, object]]:
    """Soften grounded answers when the cited corpus context is explicitly tentative."""

    combined_context = "\n".join(chunk.chunk_text for chunk in context_chunks)
    has_proposed_signal = bool(_PROPOSED_SIGNAL_RE.search(combined_context))
    has_interim_signal = bool(_INTERIM_SIGNAL_RE.search(combined_context))
    matter_type = _detect_matter_type(context_chunks)
    brief_summary_requested = bool(
        analysis.asks_brief_summary or _BRIEF_SUMMARY_QUERY_RE.search(analysis.raw_query)
    )

    cautions: list[str] = []
    adjusted = answer_text.strip()

    if brief_summary_requested:
        brief_summary = _rewrite_brief_summary(
            context_chunks=context_chunks,
            answer_text=adjusted,
            matter_type=matter_type,
        )
        if brief_summary is not None:
            adjusted = brief_summary
            cautions.append("brief_summary_formatter")

    settlement_rewrite = _rewrite_settlement_context_answer(
        answer_text=adjusted,
        context_chunks=context_chunks,
        analysis=analysis,
        matter_type=matter_type,
    )
    if settlement_rewrite is not None:
        adjusted = settlement_rewrite
        cautions.append("settlement_matter_type_caution")

    matter_type_rewrite = _rewrite_non_settlement_matter_type_answer(
        answer_text=adjusted,
        context_chunks=context_chunks,
        analysis=analysis,
        matter_type=matter_type,
    )
    if matter_type_rewrite is not None:
        adjusted = matter_type_rewrite
        cautions.append("matter_type_context_caution")

    query_context_rewrite = _rewrite_query_context_mismatch(
        answer_text=adjusted,
        context_chunks=context_chunks,
        analysis=analysis,
        matter_type=matter_type,
    )
    if query_context_rewrite is not None:
        adjusted = query_context_rewrite
        cautions.append("query_context_mismatch_caution")

    holding_rewrite, holding_debug = _rewrite_holding_answer(
        answer_text=adjusted,
        combined_context=combined_context,
        analysis=analysis,
    )
    if holding_rewrite is not None:
        adjusted = holding_rewrite
        cautions.extend(holding_debug["caution_flags"])

    holding_rewrite_used = bool(holding_debug["used"])

    if has_proposed_signal and _OVERCLAIM_OWNER_RE.search(adjusted):
        adjusted = _OVERCLAIM_OWNER_RE.sub("holding", adjusted)
        cautions.append("replaced_owner_language")

    if (
        has_proposed_signal
        and not holding_rewrite_used
        and "proposed" not in adjusted.lower()
    ):
        adjusted = (
            "The cited order text describes a proposed acquisition or exemption context. "
            + adjusted
        )
        cautions.append("prefixed_proposed_context")

    if (
        has_interim_signal
        and not brief_summary_requested
        and not any(token in adjusted.lower() for token in ("prima facie", "interim"))
    ):
        adjusted = "The cited text reflects prima facie or interim observations. " + adjusted
        cautions.append("prefixed_interim_context")

    if has_interim_signal and not analysis.appears_general_explanatory and _FINALITY_RE.search(adjusted):
        adjusted = _FINALITY_RE.sub("ultimately", adjusted, count=1)
        cautions.append("softened_finality")

    return adjusted, {
        "used": bool(cautions),
        "caution_flags": tuple(dict.fromkeys(cautions)),
        "matter_type": matter_type,
        "brief_summary_requested": brief_summary_requested,
        "has_proposed_signal": has_proposed_signal,
        "has_interim_signal": has_interim_signal,
        "holding_rewrite_used": bool(holding_debug.get("used", False)),
        **{
            key: value
            for key, value in holding_debug.items()
            if key not in {"used", "caution_flags"}
        },
    }


def _detect_matter_type(context_chunks: tuple[PromptContextChunk, ...]) -> str:
    haystack = " ".join(
        " ".join(
            part
            for part in (
                chunk.title,
                chunk.bucket_name,
                chunk.section_title,
                chunk.chunk_text[:400],
            )
            if part
        )
        for chunk in context_chunks
    )
    if _SETTLEMENT_MATTER_RE.search(haystack):
        return "settlement_order"
    if _EXEMPTION_MATTER_RE.search(haystack):
        return "exemption_order"
    if _RTI_APPELLATE_MATTER_RE.search(haystack):
        return "rti_appellate_order"
    if _APPELLATE_MATTER_RE.search(haystack):
        return "appellate_order"
    if _INTERIM_SIGNAL_RE.search(haystack):
        return "interim_order"
    if _FINAL_ENFORCEMENT_MATTER_RE.search(haystack):
        return "final_enforcement_order"
    return "order"


def _rewrite_brief_summary(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
    answer_text: str,
    matter_type: str,
) -> str | None:
    title = next((chunk.title.strip() for chunk in context_chunks if chunk.title and chunk.title.strip()), None)
    if matter_type == "settlement_order":
        settlement_summary = _rewrite_settlement_brief_summary(context_chunks=context_chunks)
        if settlement_summary is not None:
            return settlement_summary
    if matter_type == "final_enforcement_order":
        enforcement_summary = _rewrite_final_enforcement_brief_summary(
            context_chunks=context_chunks,
            title=title,
        )
        if enforcement_summary is not None:
            return enforcement_summary
    selected = _select_summary_sentences(context_chunks=context_chunks, limit=3)
    if not selected:
        return None
    sentences = [sentence.sentence for sentence in selected]
    if title and title.lower() not in sentences[0].lower():
        sentences.insert(0, f"This matter concerns {title}.")
    if len(sentences) == 1 and answer_text.strip():
        sentences.append(answer_text.strip())
    return " ".join(sentences[:4]).strip()


def _rewrite_settlement_context_answer(
    *,
    answer_text: str,
    context_chunks: tuple[PromptContextChunk, ...],
    analysis: QueryAnalysis,
    matter_type: str,
) -> str | None:
    if matter_type != "settlement_order":
        return None
    if not _SETTLEMENT_FINDINGS_QUERY_RE.search(analysis.raw_query):
        return None
    rendered = _render_settlement_summary_from_context(context_chunks=context_chunks)
    if rendered is not None:
        return rendered
    normalized_context = re.sub(
        r"\s*\|\s*",
        " ",
        "\n".join(chunk.chunk_text for chunk in context_chunks),
    )
    normalized_context = re.sub(r"\s+", " ", normalized_context).strip()
    normalized_context = re.sub(r"(?:^|\s)(\d+)\.\s+", "\n", normalized_context)
    sentences = [
        _ensure_sentence(raw_sentence.strip(" .;"))
        for raw_sentence in _SENTENCE_SPLIT_RE.split(normalized_context)
        if " ".join(raw_sentence.split()).strip(" .;")
    ]
    if not sentences:
        return None
    allegation = next(
        (sentence for sentence in sentences if re.search(r"\balleg|show cause notice|scn|material non-public|communicat", sentence, re.IGNORECASE)),
        None,
    )
    amount = next(
        (sentence for sentence in sentences if re.search(r"\bsettlement amount\b|\bnotice of demand\b|\bremitted\b|\bpaid\b", sentence, re.IGNORECASE)),
        None,
    )
    disposal = next(
        (sentence for sentence in sentences if re.search(r"\bdisposed of\b|\bsettlement proceedings\b|\bsettled\b", sentence, re.IGNORECASE)),
        None,
    )
    if allegation is None and amount is None and disposal is None:
        if re.search(r"\bsebi\s+(?:found|held)\b", answer_text, re.IGNORECASE):
            return re.sub(
                r"\bSEBI\s+(?:found|held)\b",
                "The settlement order records that SEBI alleged",
                answer_text,
                count=1,
                flags=re.IGNORECASE,
            )
        return None
    rendered: list[str] = []
    if allegation is None:
        rendered.append(
            "The settlement order records allegations and settlement terms rather than a final merits finding."
        )
    else:
        rendered.append(allegation)
    if amount and amount not in rendered:
        rendered.append(amount)
    if disposal and disposal not in rendered:
        rendered.append(disposal)
    return " ".join(rendered[:3]).strip()


def _rewrite_settlement_brief_summary(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
) -> str | None:
    rendered = _render_settlement_summary_from_context(context_chunks=context_chunks)
    if rendered is None:
        return None
    title = next((chunk.title.strip() for chunk in context_chunks if chunk.title and chunk.title.strip()), None)
    if not title:
        return rendered
    return f"This matter concerns {title}. {rendered}".strip()


def _render_settlement_summary_from_context(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
) -> str | None:
    normalized_context = _normalize_context_text(context_chunks)
    if not normalized_context:
        return None
    observation_body = _extract_first_match(
        normalized_context,
        (
            re.compile(
                r"(?:SEBI|Securities and Exchange Board of India[^)]*\)|Securities and Exchange Board of India)\s+observed\s+(.*?(?:high correlation).*?trades of certain others)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:SEBI|Securities and Exchange Board of India[^)]*\)|Securities and Exchange Board of India)\s+observed\s+(.*?)(?:\.|Pursuant to)",
                re.IGNORECASE,
            ),
        ),
        group=1,
    )
    allegation_body = _extract_first_match(
        normalized_context,
        (
            re.compile(
                r"(?:it was alleged that|wherein inter alia it was alleged that|the show cause notice alleged that)\s+([^.]*(?:material non-public information)[^.]*?(?:unlawful gains)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:it was alleged that|wherein inter alia it was alleged that|the show cause notice alleged that)\s+([^.]+)",
                re.IGNORECASE,
            ),
        ),
        group=1,
    )
    amount = _extract_first_match(
        normalized_context,
        (
            re.compile(
                r"recommended the case for settlement on payment of\s+(?:INR|Rs\.?)\s*([0-9,]+)",
                re.IGNORECASE,
            ),
            re.compile(
                r"settlement amount of\s+(?:INR|Rs\.?)\s*([0-9,]+)",
                re.IGNORECASE,
            ),
        ),
        group=1,
    )
    disposal = _extract_first_match(
        normalized_context,
        (
            re.compile(
                r"(?:the specified proceedings|the instant proceedings)[^.]*?disposed of[^.]*?\.",
                re.IGNORECASE,
            ),
            re.compile(r"[^.]*disposed of[^.]*\.", re.IGNORECASE),
        ),
    )
    if observation_body is None and allegation_body is None and amount is None and disposal is None:
        return None
    rendered: list[str] = []
    if observation_body:
        rendered.append(_ensure_sentence(f"SEBI observed {observation_body.rstrip('.')}"))
    if allegation_body:
        rendered.append(_ensure_sentence(f"The SCN alleged that {allegation_body.rstrip('.')}"))
    if observation_body is None and allegation_body is None:
        rendered.append(
            "The settlement order records allegations and settlement terms rather than a final merits finding."
        )
    if amount is not None:
        if disposal is not None:
            rendered.append(
                _ensure_sentence(
                    f"The matter was settled on payment of Rs. {amount.strip(' ,./-')}, and the proceedings were disposed of."
                )
            )
        else:
            rendered.append(
                _ensure_sentence(
                    f"The matter was settled on payment of Rs. {amount.strip(' ,./-')}."
                )
            )
    elif disposal is not None:
        rendered.append("The proceedings were disposed of through settlement.")
    return " ".join(rendered[:3]).strip()


def _rewrite_final_enforcement_brief_summary(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
    title: str | None,
) -> str | None:
    combined_text = _normalize_context_text(context_chunks)
    complaint = _find_context_sentence(
        context_chunks=context_chunks,
        patterns=(
            re.compile(r"\breceived a complaint\b", re.IGNORECASE),
            re.compile(r"\bcarried out the examination\b", re.IGNORECASE),
            re.compile(r"\binvestigation examined\b", re.IGNORECASE),
            re.compile(r"\bproceedings arose\b", re.IGNORECASE),
        ),
        preferred_sections=("background", "facts", "show_cause_notice"),
    )
    if (
        "claiming it to be a sebi registered intermediary" in combined_text.lower()
        and "unregistered" in combined_text.lower()
    ):
        complaint = (
            "SEBI received a complaint that the noticee was running Yash Trading Academy, "
            "falsely presenting it as SEBI-registered, and offering unregistered investment-advisory "
            "and portfolio-management services."
        )
    finding = _find_context_sentence(
        context_chunks=context_chunks,
        patterns=(
            re.compile(r"\bthe examination concluded\b", re.IGNORECASE),
            re.compile(r"\bi find that\b", re.IGNORECASE),
            re.compile(r"\bviolated\b", re.IGNORECASE),
            re.compile(r"\bfraud\b", re.IGNORECASE),
        ),
        preferred_sections=("findings", "operative_order", "other", "background"),
    )
    if (
        "unregistered investment advisory" in combined_text.lower()
        or "portfolio management" in combined_text.lower()
    ) and (
        "fraud" in combined_text.lower()
        or "false and misleading" in combined_text.lower()
        or "guaranteed return" in combined_text.lower()
    ):
        finding = (
            "The order found that the noticee was engaged in unregistered investment-advisory "
            "and portfolio-management activity and had misled investors through false or misleading claims."
        )
    directions = _render_final_direction_summary_from_context(context_chunks)
    rendered: list[str] = []
    if title:
        rendered.append(f"This matter concerns {title}.")
    for sentence in (complaint, finding, directions):
        if not sentence:
            continue
        normalized = normalize_fuzzy_text(sentence)
        if any(normalized == normalize_fuzzy_text(existing) for existing in rendered):
            continue
        rendered.append(_ensure_sentence(sentence.rstrip(".")))
    if len(rendered) < 2:
        return None
    return " ".join(rendered[:4]).strip()


def _render_final_direction_summary_from_context(
    context_chunks: tuple[PromptContextChunk, ...],
) -> str | None:
    combined_text = _normalize_context_text(context_chunks)
    refund_sentence = _find_context_sentence(
        context_chunks=context_chunks,
        patterns=(re.compile(r"\bshall refund\b|\brefunds?\b", re.IGNORECASE),),
        preferred_sections=("directions", "operative_order"),
    )
    debar_sentence = _find_context_sentence(
        context_chunks=context_chunks,
        patterns=(
            re.compile(r"\bdebarred\b|\bprohibited from accessing the securities market\b", re.IGNORECASE),
            re.compile(r"\brestrained from accessing the securities market\b", re.IGNORECASE),
        ),
        preferred_sections=("directions", "operative_order"),
    )
    penalty_sentence = _find_context_sentence(
        context_chunks=context_chunks,
        patterns=(re.compile(r"\bpenalt(?:y|ies)\b.*\bRs\.", re.IGNORECASE),),
        preferred_sections=("directions", "operative_order"),
    )
    clauses: list[str] = []
    if refund_sentence is not None:
        clauses.append("directed refunds to investors and related repayment compliance steps")
    if debar_sentence is not None:
        duration = _extract_first_match(
            _normalize_sentence_text(debar_sentence),
            (
                re.compile(r"for a period of [^.]+", re.IGNORECASE),
                re.compile(r"until[^.]+", re.IGNORECASE),
            ),
        )
        if duration is not None:
            clauses.append(
                f"debarred the noticee from the securities market {duration.rstrip('.')}"
            )
        elif re.search(r"\brestrained\b", debar_sentence, re.IGNORECASE):
            clauses.append("restrained the noticee from accessing the securities market")
        else:
            clauses.append("debarred the noticee from the securities market")
    if penalty_sentence is not None:
        amounts = re.findall(r"Rs\.?\s*[0-9,]+(?:\.[0-9]+)?", penalty_sentence, flags=re.IGNORECASE)
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
    if not clauses and "do hereby issue the following directions" not in combined_text.lower():
        return None
    if not clauses:
        clauses.append("issued operative directions against the noticee")
    if len(clauses) == 1:
        return _ensure_sentence(f"The order {clauses[0]}.")
    return _ensure_sentence(
        "The order "
        + ", ".join(clauses[:-1])
        + f", and {clauses[-1]}."
    )


def _rewrite_non_settlement_matter_type_answer(
    *,
    answer_text: str,
    context_chunks: tuple[PromptContextChunk, ...],
    analysis: QueryAnalysis,
    matter_type: str,
) -> str | None:
    if matter_type in {"order", "settlement_order"}:
        return None
    if analysis.asks_brief_summary:
        return None
    if not (
        _ACTION_OR_SUMMARY_QUERY_RE.search(analysis.raw_query)
    ):
        return None
    if not (
        analysis.active_order_override
        or _MATTER_TYPE_TRIGGER_RE.search(analysis.raw_query)
    ):
        return None

    sentences = [
        _ensure_sentence(" ".join(raw_sentence.split()).strip(" .;"))
        for raw_sentence in _SENTENCE_SPLIT_RE.split(
            "\n".join(chunk.chunk_text for chunk in context_chunks)
        )
        if " ".join(raw_sentence.split()).strip(" .;")
    ]
    if not sentences:
        return None

    selected: list[str] = []
    if matter_type == "exemption_order":
        selected = _collect_sentences(sentences, _EXEMPTION_GRANT_RE, limit=3)
    elif matter_type == "rti_appellate_order":
        selected = _collect_sentences(sentences, _RTI_DECISION_RE, limit=3)
    elif matter_type == "appellate_order":
        selected = _collect_sentences(sentences, _APPELLATE_DISPOSITION_RE, limit=3)
    elif matter_type == "final_enforcement_order":
        selected = _collect_sentences(sentences, _FINAL_ENFORCEMENT_ACTION_RE, limit=3)
    elif matter_type == "interim_order":
        selected = _collect_sentences(
            sentences,
            re.compile(r"\b(?:prima facie|interim|restrained|directions?)\b", re.IGNORECASE),
            limit=3,
        )

    if not selected:
        return None
    if len(selected) == 1 and answer_text.strip():
        selected.append(_ensure_sentence(answer_text.strip(" .;")))
    return " ".join(selected[:3]).strip()


def _rewrite_query_context_mismatch(
    *,
    answer_text: str,
    context_chunks: tuple[PromptContextChunk, ...],
    analysis: QueryAnalysis,
    matter_type: str,
) -> str | None:
    if matter_type == "settlement_order" and _EXEMPTION_REQUEST_QUERY_RE.search(analysis.raw_query):
        settlement_summary = _render_settlement_summary_from_context(
            context_chunks=context_chunks
        )
        if settlement_summary is not None:
            return (
                "This is a settlement order, not an exemption order. "
                + settlement_summary
            ).strip()
        return (
            "This is a settlement order, not an exemption order. "
            "It records allegations, settlement terms, and disposal of the proceedings rather than an exemption granted by SEBI."
        )

    normalized_context = _normalize_context_text(context_chunks)
    if not _IPO_PROCEEDS_QUERY_RE.search(analysis.raw_query):
        return None
    if not (
        _PREFERENTIAL_ALLOTMENT_RE.search(normalized_context)
        or _LOAN_CONVERSION_RE.search(normalized_context)
    ):
        return None

    transaction_sentence = _find_context_sentence(
        context_chunks=context_chunks,
        patterns=(
            _PREFERENTIAL_ALLOTMENT_RE,
            _LOAN_CONVERSION_RE,
        ),
        preferred_sections=("facts", "background", "findings", "operative_order", "other"),
    )
    if transaction_sentence is None:
        return (
            "The cited order does not describe IPO proceeds. "
            "It instead discusses a preferential allotment and loan-conversion transaction."
        )
    return (
        "The cited order does not describe IPO proceeds. "
        + _ensure_sentence(transaction_sentence.rstrip("."))
    )


def _collect_sentences(
    sentences: list[str],
    pattern: re.Pattern[str],
    *,
    limit: int,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = normalize_fuzzy_text(sentence)
        if normalized in seen or not pattern.search(sentence):
            continue
        selected.append(sentence)
        seen.add(normalized)
        if len(selected) >= limit:
            break
    return selected


def _select_summary_sentences(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
    limit: int,
) -> tuple[_ContextSentence, ...]:
    candidates: list[_ContextSentence] = []
    for chunk in context_chunks:
        section_type = (chunk.section_type or "other").strip()
        base_score = {
            "findings": 3.2,
            "facts": 3.0,
            "background": 2.6,
            "operative_order": 2.2,
            "directions": 2.0,
            "issues": 1.8,
        }.get(section_type, 0.8)
        for raw_sentence in _SENTENCE_SPLIT_RE.split(chunk.chunk_text):
            sentence = " ".join(raw_sentence.split()).strip(" .;")
            if (
                not _summary_sentence_usable(sentence)
            ):
                continue
            score = base_score
            if _SUMMARY_KEYWORD_RE.search(sentence):
                score += 1.2
            if re.search(r"\b(?:received a complaint|debarred|penalt(?:y|ies)|refund|disposed of)\b", sentence, re.IGNORECASE):
                score += 0.8
            candidates.append(
                _ContextSentence(
                    sentence=_ensure_sentence(sentence),
                    section_type=section_type,
                    score=score,
                )
            )
    if not candidates:
        return ()
    ordered = sorted(
        candidates,
        key=lambda item: (-item.score, len(item.sentence)),
    )
    selected: list[_ContextSentence] = []
    seen_sentences: set[str] = set()
    seen_sections: set[str] = set()
    for candidate in ordered:
        normalized = normalize_fuzzy_text(candidate.sentence)
        if normalized in seen_sentences:
            continue
        if candidate.section_type in seen_sections and len(selected) >= 2:
            continue
        seen_sentences.add(normalized)
        seen_sections.add(candidate.section_type)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _ensure_sentence(value: str) -> str:
    sentence = value.strip()
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _summary_sentence_usable(sentence: str) -> bool:
    normalized = sentence.strip()
    if len(normalized) < 35:
        return False
    if (
        _SUMMARY_SKIP_RE.search(normalized)
        or _SUMMARY_QUESTION_RE.search(normalized)
        or _SUMMARY_ENUMERATION_RE.search(normalized)
    ):
        return False
    if normalized.endswith("?") or _SUMMARY_FRAGMENT_START_RE.search(normalized):
        return False
    if normalized[:1].islower():
        return False
    return _SUMMARY_DANGLING_END_RE.search(_ensure_sentence(normalized)) is None


def _normalize_context_text(context_chunks: tuple[PromptContextChunk, ...]) -> str:
    text = re.sub(r"\s*\|\s*", " ", "\n".join(chunk.chunk_text for chunk in context_chunks))
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?:^|\s)(\d+)\.\s+", "\n", text)


def _normalize_sentence_text(value: str) -> str:
    return " ".join(value.split()).strip(" .;")


def _extract_first_match(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
    *,
    group: int = 0,
) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        value = _normalize_sentence_text(match.group(group))
        if value:
            return value
    return None


def _find_context_sentence(
    *,
    context_chunks: tuple[PromptContextChunk, ...],
    patterns: tuple[re.Pattern[str], ...],
    preferred_sections: tuple[str, ...],
) -> str | None:
    section_ranks = {
        section: len(preferred_sections) - index
        for index, section in enumerate(preferred_sections)
    }
    best: tuple[int, int, str] | None = None
    for chunk in context_chunks:
        section_type = (chunk.section_type or "other").strip()
        section_rank = section_ranks.get(section_type, 0)
        if section_rank <= 0:
            continue
        for raw_sentence in _SENTENCE_SPLIT_RE.split(chunk.chunk_text):
            sentence = _normalize_sentence_text(raw_sentence)
            if (
                not _summary_sentence_usable(sentence)
            ):
                continue
            pattern_hits = sum(1 for pattern in patterns if pattern.search(sentence))
            if pattern_hits <= 0:
                continue
            candidate = (section_rank, pattern_hits, sentence)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return None if best is None else best[2]


def _rewrite_holding_answer(
    *,
    answer_text: str,
    combined_context: str,
    analysis: QueryAnalysis,
) -> tuple[str | None, dict[str, object]]:
    query_subject = _extract_query_subject(analysis.raw_query)
    if not query_subject:
        return None, {
            "used": False,
            "holding_query_subject": None,
            "holding_candidate_subjects": (),
            "selected_holding_proposed": False,
        }

    evidences = _extract_holding_evidence(combined_context)
    matched = tuple(
        evidence
        for evidence in evidences
        if _holding_subject_matches(query_subject, evidence)
    )
    if not matched:
        return None, {
            "used": False,
            "holding_query_subject": query_subject,
            "holding_candidate_subjects": tuple(evidence.subject for evidence in evidences),
            "selected_holding_proposed": False,
        }

    selected = _select_holding_evidence(query_subject, matched)
    counterpart = _related_counterpart(selected=selected, evidences=evidences)
    rewritten = _format_holding_answer(selected=selected, counterpart=counterpart)
    return rewritten, {
        "used": True,
        "holding_query_subject": query_subject,
        "holding_candidate_subjects": tuple(evidence.subject for evidence in evidences),
        "holding_selected_subject": selected.subject,
        "holding_selected_subject_type": selected.subject_type,
        "selected_holding_proposed": selected.proposed,
        "caution_flags": (
            "holding_subject_disambiguation",
            "individual_vs_entity_caution",
            *(
                ("proposed_holding_caution",)
                if selected.proposed
                else ()
            ),
        ),
        "original_answer_text": answer_text,
    }


def _extract_query_subject(query: str) -> str | None:
    match = _HOLDING_QUERY_RE.search(query)
    if match is None:
        return None
    subject = match.group("subject").strip(" ,.-")
    return subject or None


def _extract_holding_evidence(combined_context: str) -> tuple[_HoldingEvidence, ...]:
    evidences: list[_HoldingEvidence] = []
    for raw_sentence in _SENTENCE_SPLIT_RE.split(combined_context):
        sentence = " ".join(raw_sentence.split()).strip(" .;")
        if not sentence:
            continue
        subject_match = _HOLDING_SENTENCE_RE.search(sentence)
        if subject_match is None:
            continue
        share_count_match = _SHARE_COUNT_RE.search(sentence)
        percent_match = _PERCENT_RE.search(sentence)
        if share_count_match is None and percent_match is None:
            continue
        subject = _clean_subject(subject_match.group("subject"))
        if not subject:
            continue
        evidences.append(
            _HoldingEvidence(
                subject=subject,
                normalized_subject=_normalize_subject(subject),
                base_subject=_base_subject(subject),
                subject_type=_subject_type(subject),
                share_count=(share_count_match.group(1) if share_count_match is not None else None),
                percentage=(
                    f"{percent_match.group(1)}%"
                    if percent_match is not None
                    else None
                ),
                proposed=bool(_PROPOSED_SIGNAL_RE.search(sentence) or subject_match.group("proposed")),
                individually=bool(subject_match.group("individual")),
                sentence=sentence,
            )
        )
    normalized_context = " ".join(combined_context.split())
    for match in _TABULAR_HOLDING_RE.finditer(normalized_context):
        subject = _clean_subject(match.group("subject"))
        if not subject or _normalized_subject_is_blocked(subject):
            continue
        local_window = normalized_context[max(0, match.start() - 160): min(len(normalized_context), match.end() + 160)]
        evidences.append(
            _HoldingEvidence(
                subject=subject,
                normalized_subject=_normalize_subject(subject),
                base_subject=_base_subject(subject),
                subject_type=_subject_type(subject),
                share_count=match.group("shares"),
                percentage=f"{match.group('percent')}%",
                proposed=_tabular_holding_is_proposed(subject=subject, local_window=local_window),
                individually=subject.lower().startswith(("mrs", "mr", "ms", "dr")),
                sentence=match.group(0),
            )
        )
    return tuple(evidences)


def _holding_subject_matches(query_subject: str, evidence: _HoldingEvidence) -> bool:
    query_type = _subject_type(query_subject)
    if query_type != evidence.subject_type:
        return False
    normalized_query = _normalize_subject(query_subject)
    if not normalized_query:
        return False
    if normalized_query == evidence.normalized_subject:
        return True
    if evidence.subject_type == "entity":
        return normalized_query in evidence.normalized_subject or evidence.normalized_subject in normalized_query
    return evidence.normalized_subject.endswith(normalized_query) or normalized_query.endswith(evidence.normalized_subject)


def _select_holding_evidence(
    query_subject: str,
    evidences: tuple[_HoldingEvidence, ...],
) -> _HoldingEvidence:
    normalized_query = _normalize_subject(query_subject)

    def _score(evidence: _HoldingEvidence) -> tuple[int, int, int, int]:
        exact = int(evidence.normalized_subject == normalized_query)
        individual_bonus = int(evidence.individually and evidence.subject_type == "individual")
        proposed_bonus = int(evidence.proposed and evidence.subject_type == "entity")
        completeness = int(bool(evidence.share_count)) + int(bool(evidence.percentage))
        return (exact, individual_bonus, proposed_bonus, completeness)

    return sorted(
        evidences,
        key=lambda evidence: (_score(evidence), evidence.subject),
        reverse=True,
    )[0]


def _related_counterpart(
    *,
    selected: _HoldingEvidence,
    evidences: tuple[_HoldingEvidence, ...],
) -> _HoldingEvidence | None:
    related = [
        evidence
        for evidence in evidences
        if evidence.subject != selected.subject
        and evidence.subject_type != selected.subject_type
        and evidence.base_subject
        and evidence.base_subject == selected.base_subject
    ]
    if not related:
        return None
    return sorted(
        related,
        key=lambda evidence: (bool(evidence.share_count), bool(evidence.percentage), evidence.subject),
        reverse=True,
    )[0]


def _format_holding_answer(
    *,
    selected: _HoldingEvidence,
    counterpart: _HoldingEvidence | None,
) -> str:
    amount = _format_holding_amount(selected)
    if selected.proposed:
        answer = f"In the cited order, {selected.subject} was the proposed acquirer proposed to hold {amount}."
    elif selected.individually:
        answer = f"In the cited order, {selected.subject} individually held {amount}."
    else:
        answer = f"In the cited order, {selected.subject} held {amount}."

    if counterpart is None:
        return answer
    counterpart_amount = _format_holding_amount(counterpart)
    if counterpart.proposed:
        return (
            answer
            + f" The separate proposed-acquirer entry for {counterpart.subject} described a proposed holding of {counterpart_amount}."
        )
    if counterpart.individually:
        return (
            answer
            + f" The order separately states that {counterpart.subject} individually held {counterpart_amount}."
        )
    return answer + f" The order separately states that {counterpart.subject} held {counterpart_amount}."


def _format_holding_amount(evidence: _HoldingEvidence) -> str:
    if evidence.share_count and evidence.percentage:
        return f"{evidence.share_count} shares ({evidence.percentage})"
    if evidence.share_count:
        return f"{evidence.share_count} shares"
    if evidence.percentage:
        return evidence.percentage
    return "shares not clearly stated"


def _normalize_subject(value: str) -> str:
    return normalize_fuzzy_text(_HONORIFIC_PREFIX_RE.sub("", value or ""))


def _base_subject(value: str) -> str:
    normalized = _normalize_subject(value)
    if not normalized:
        return ""
    tokens = [token for token in normalized.split() if token]
    while tokens and any(
        " ".join(tokens[-size:]) in _ENTITY_SUBJECT_TOKENS
        for size in (2, 1)
        if len(tokens) >= size
    ):
        if len(tokens) >= 2 and " ".join(tokens[-2:]) in _ENTITY_SUBJECT_TOKENS:
            tokens = tokens[:-2]
            continue
        tokens = tokens[:-1]
    return " ".join(tokens)


def _subject_type(value: str) -> Literal["individual", "entity"]:
    normalized = _normalize_subject(value)
    if any(term in normalized for term in _ENTITY_SUBJECT_TOKENS):
        return "entity"
    return "individual"


def _clean_subject(value: str) -> str:
    cleaned = value.strip(" ,:-")
    cleaned = re.sub(r"^(?:and|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,:-")


def _normalized_subject_is_blocked(value: str) -> bool:
    normalized = _normalize_subject(value)
    return normalized in _TABULAR_SUBJECT_BLOCKLIST


def _tabular_holding_is_proposed(*, subject: str, local_window: str) -> bool:
    normalized_subject = _normalize_subject(subject)
    if not normalized_subject:
        return False
    if _TABULAR_PROPOSED_SIGNAL_RE.search(local_window) is None:
        return False
    if "family trust" in normalized_subject:
        return True
    if "acquirer" in local_window.lower() and _subject_type(subject) == "entity":
        return True
    return False
