"""Deterministic extraction of numeric and tabular facts from SEBI orders."""

from __future__ import annotations

import re

from .models import ExtractedNumericFact, ExtractedPriceMovement, MetadataChunkText, MetadataPageText

_MONTH_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_RE = rf"{_MONTH_RE}\s+\d{{1,2}},\s+\d{{4}}"
_DATE_RANGE_TEXT_RE = rf"{_DATE_RE}\s*(?:to|[-–])\s*{_DATE_RE}"
_LISTING_PRICE_RE = re.compile(
    rf"(?:post\s+)?listing\s+on\s+(?P<date>{_DATE_RE})\s+at\s+(?:the\s+)?price\s+of\s+Rs\.?\s*(?P<value>[0-9][0-9,]*(?:\.\d+)?)\s*/?\s*share",
    re.IGNORECASE,
)
_OVERALL_PRICE_MOVE_RE = re.compile(
    rf"increased\s+by\s+(?P<pct>[+-]?\d+(?:\.\d+)?)%\s+during\s+the\s+period\s+(?P<period>{_DATE_RANGE_TEXT_RE}).*?closed\s+at\s+Rs\.?\s*(?P<close>[0-9][0-9,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_HIGHEST_PRICE_RE = re.compile(
    rf"highest\s+price\s+of\s+Rs\.?\s*(?P<price>[0-9][0-9,]*(?:\.\d+)?)\s*,?\s*(?:which\s+is\s+(?P<pct>[+-]?\d+(?:\.\d+)?)%\s+of\s+the\s+listing\s+price,?\s*)?on\s+(?P<date>{_DATE_RE})",
    re.IGNORECASE,
)
_LOWEST_PRICE_RE = re.compile(
    rf"(?:lowest|low)\s+price\s+of\s+Rs\.?\s*(?P<price>[0-9][0-9,]*(?:\.\d+)?)\s*(?:on\s+(?P<date>{_DATE_RE}))?",
    re.IGNORECASE,
)
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AMOUNT_RE = re.compile(
    r"(?:₹\s*[0-9][0-9,]*(?:\.\d+)?(?:/-)?|\b(?:Rs\.?|INR)\s*[0-9][0-9,]*(?:\.\d+)?(?:/-)?\b|\b[0-9][0-9,]*(?:\.\d+)?\s*(?:crore|lakh)\b)",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
_SHARE_COUNT_RE = re.compile(r"([0-9][0-9,]*(?:\.\d+)?)\s+shares?\b", re.IGNORECASE)
_HOLDING_RE = re.compile(r"\bhold(?:ing|ings)?\b|\bshareholding\b|\bownership\b", re.IGNORECASE)
_SETTLEMENT_RE = re.compile(r"\bsettlement amount\b", re.IGNORECASE)
_PENALTY_RE = re.compile(r"\bpenalt(?:y|ies)\b|\bfine\b", re.IGNORECASE)
_FEE_RE = re.compile(r"\bfee(?:s)?\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z])|\n+")
_TITLE_SUBJECT_RE = re.compile(
    r"(?:in\s+the\s+matter\s+of|order\s+in\s+the\s+matter\s+of|final\s+order\s+in\s+the\s+matter\s+of)\s+(?P<subject>.+)",
    re.IGNORECASE,
)
_TRAILING_SUBJECT_NOISE_RE = re.compile(
    r"\s*(?:\(|\[).*$|\s+vide\s+.*$",
    re.IGNORECASE,
)


def extract_numeric_facts(
    *,
    document_version_id: int,
    pages: tuple[MetadataPageText, ...],
    chunks: tuple[MetadataChunkText, ...],
    price_movements: tuple[ExtractedPriceMovement, ...] = (),
    title: str | None = None,
) -> tuple[ExtractedNumericFact, ...]:
    """Extract deterministic numeric facts from narrative text and structured rows."""

    facts: list[ExtractedNumericFact] = []
    seen: set[str] = set()
    subject = _infer_primary_subject(title)

    for page in pages:
        normalized = _normalize_text(page.text)
        if not normalized:
            continue

        listing_match = _LISTING_PRICE_RE.search(normalized)
        if listing_match is not None:
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="listing_price",
                    subject=subject,
                    value_text=f"Rs.{listing_match.group('value')}/share",
                    value_numeric=_parse_float(listing_match.group("value")),
                    unit="INR/share",
                    context_label=f"listed on {_clean_text(listing_match.group('date'))}",
                    page_start=page.page_no,
                    page_end=page.page_no,
                ),
            )

        overall_move_match = _OVERALL_PRICE_MOVE_RE.search(normalized)
        if overall_move_match is not None:
            period_text = _clean_text(overall_move_match.group("period"))
            period_end_text = _extract_period_end_text(period_text)
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="percentage_change",
                    subject=subject,
                    value_text=f"{overall_move_match.group('pct')}%",
                    value_numeric=_parse_float(overall_move_match.group("pct")),
                    unit="percent",
                    context_label=period_text,
                    page_start=page.page_no,
                    page_end=page.page_no,
                ),
            )
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="closing_price",
                    subject=subject,
                    value_text=f"Rs.{overall_move_match.group('close')}",
                    value_numeric=_parse_float(overall_move_match.group("close")),
                    unit="INR",
                    context_label=(f"on {period_end_text}" if period_end_text else period_text),
                    page_start=page.page_no,
                    page_end=page.page_no,
                ),
            )

        highest_match = _HIGHEST_PRICE_RE.search(normalized)
        if highest_match is not None:
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="highest_price",
                    subject=subject,
                    value_text=f"Rs.{highest_match.group('price')}",
                    value_numeric=_parse_float(highest_match.group("price")),
                    unit="INR",
                    context_label=f"on {_clean_text(highest_match.group('date'))}",
                    page_start=page.page_no,
                    page_end=page.page_no,
                ),
            )
            if highest_match.group("pct"):
                _append_fact(
                    facts,
                    seen,
                    ExtractedNumericFact(
                        document_version_id=document_version_id,
                        fact_type="percentage_change_from_listing",
                        subject=subject,
                        value_text=f"{highest_match.group('pct')}%",
                        value_numeric=_parse_float(highest_match.group("pct")),
                        unit="percent",
                        context_label=f"highest price on {_clean_text(highest_match.group('date'))}",
                        page_start=page.page_no,
                        page_end=page.page_no,
                    ),
                )

        for low_match in _LOWEST_PRICE_RE.finditer(normalized):
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="lowest_price",
                    subject=subject,
                    value_text=f"Rs.{low_match.group('price')}",
                    value_numeric=_parse_float(low_match.group("price")),
                    unit="INR",
                    context_label=(
                        f"on {_clean_text(low_match.group('date'))}"
                        if low_match.group("date")
                        else None
                    ),
                    page_start=page.page_no,
                    page_end=page.page_no,
                ),
            )

        for line in page.text.splitlines():
            pan_match = _PAN_RE.search(line.upper())
            if pan_match is None:
                continue
            cleaned_line = " ".join(line.split())
            if cleaned_line.lower().endswith(" pan"):
                continue
            prefix = cleaned_line[: pan_match.start()].strip(" -:")
            if not prefix or prefix.lower() in {"name", "noticee no. name"}:
                continue
            subject_text = re.sub(r"^\d+\s+", "", prefix).strip(" -:")
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="pan",
                    subject=subject_text or subject,
                    value_text=pan_match.group(0),
                    unit="PAN",
                    page_start=page.page_no,
                    page_end=page.page_no,
                ),
            )

    for chunk in chunks:
        for sentence in _iter_sentences(chunk.text):
            lowered = sentence.lower()
            if _SETTLEMENT_RE.search(sentence):
                for amount in _extract_amounts(sentence):
                    _append_fact(
                        facts,
                        seen,
                        ExtractedNumericFact(
                            document_version_id=document_version_id,
                            fact_type="settlement_amount",
                            subject=subject,
                            value_text=amount,
                            value_numeric=_amount_numeric(amount),
                            unit=_amount_unit(amount),
                            context_label=_clean_sentence(sentence),
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                        ),
                    )
            if _PENALTY_RE.search(sentence):
                for amount in _extract_amounts(sentence):
                    _append_fact(
                        facts,
                        seen,
                        ExtractedNumericFact(
                            document_version_id=document_version_id,
                            fact_type="penalty_amount",
                            subject=subject,
                            value_text=amount,
                            value_numeric=_amount_numeric(amount),
                            unit=_amount_unit(amount),
                            context_label=_clean_sentence(sentence),
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                        ),
                    )
            if _FEE_RE.search(sentence):
                for amount in _extract_amounts(sentence):
                    _append_fact(
                        facts,
                        seen,
                        ExtractedNumericFact(
                            document_version_id=document_version_id,
                            fact_type="fee_amount",
                            subject=subject,
                            value_text=amount,
                            value_numeric=_amount_numeric(amount),
                            unit=_amount_unit(amount),
                            context_label=_clean_sentence(sentence),
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                        ),
                    )
            if _HOLDING_RE.search(sentence):
                for share_match in _SHARE_COUNT_RE.finditer(sentence):
                    _append_fact(
                        facts,
                        seen,
                        ExtractedNumericFact(
                            document_version_id=document_version_id,
                            fact_type="share_count",
                            subject=subject,
                            value_text=share_match.group(0),
                            value_numeric=_parse_float(share_match.group(1)),
                            unit="shares",
                            context_label=_clean_sentence(sentence),
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                        ),
                    )
                for pct_match in _PCT_RE.finditer(sentence):
                    _append_fact(
                        facts,
                        seen,
                        ExtractedNumericFact(
                            document_version_id=document_version_id,
                            fact_type="holding_percentage",
                            subject=subject,
                            value_text=f"{pct_match.group(1)}%",
                            value_numeric=_parse_float(pct_match.group(1)),
                            unit="percent",
                            context_label=_clean_sentence(sentence),
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                        ),
                    )

    for row in price_movements:
        context_label = _price_movement_context(row)
        _append_fact(
            facts,
            seen,
            ExtractedNumericFact(
                document_version_id=document_version_id,
                fact_type="period_start_price",
                subject=subject,
                value_text=_currency_text(row.start_price),
                value_numeric=row.start_price,
                unit="INR",
                context_label=context_label,
                page_start=row.page_start,
                page_end=row.page_end,
            ),
        )
        if row.high_price is not None:
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="period_high_price",
                    subject=subject,
                    value_text=_currency_text(row.high_price),
                    value_numeric=row.high_price,
                    unit="INR",
                    context_label=context_label,
                    page_start=row.page_start,
                    page_end=row.page_end,
                ),
            )
        if row.low_price is not None:
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="period_low_price",
                    subject=subject,
                    value_text=_currency_text(row.low_price),
                    value_numeric=row.low_price,
                    unit="INR",
                    context_label=context_label,
                    page_start=row.page_start,
                    page_end=row.page_end,
                ),
            )
        _append_fact(
            facts,
            seen,
            ExtractedNumericFact(
                document_version_id=document_version_id,
                fact_type="period_end_price",
                subject=subject,
                value_text=_currency_text(row.end_price),
                value_numeric=row.end_price,
                unit="INR",
                context_label=context_label,
                page_start=row.page_start,
                page_end=row.page_end,
            ),
        )
        if row.pct_change is not None:
            _append_fact(
                facts,
                seen,
                ExtractedNumericFact(
                    document_version_id=document_version_id,
                    fact_type="period_pct_change",
                    subject=subject,
                    value_text=f"{row.pct_change:g}%",
                    value_numeric=row.pct_change,
                    unit="percent",
                    context_label=context_label,
                    page_start=row.page_start,
                    page_end=row.page_end,
                ),
            )

    return tuple(facts)


def _append_fact(
    facts: list[ExtractedNumericFact],
    seen: set[str],
    fact: ExtractedNumericFact,
) -> None:
    if fact.value_text is None and fact.value_numeric is None:
        return
    if fact.row_sha256 in seen:
        return
    seen.add(fact.row_sha256)
    facts.append(fact)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _infer_primary_subject(title: str | None) -> str | None:
    if not title:
        return None
    cleaned = " ".join(title.split()).strip(" .")
    match = _TITLE_SUBJECT_RE.search(cleaned)
    subject = match.group("subject") if match is not None else cleaned
    subject = _TRAILING_SUBJECT_NOISE_RE.sub("", subject).strip(" .")
    return subject or None


def _iter_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        sentence
        for sentence in (
            " ".join(part.split()).strip()
            for part in _SENTENCE_SPLIT_RE.split(text)
        )
        if sentence
    )


def _extract_amounts(sentence: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0).strip() for match in _AMOUNT_RE.finditer(sentence)))


def _amount_numeric(value: str) -> float | None:
    lowered = value.lower()
    number_match = re.search(r"([0-9][0-9,]*(?:\.\d+)?)", value)
    if number_match is None:
        return None
    numeric = _parse_float(number_match.group(1))
    if numeric is None:
        return None
    if "crore" in lowered:
        return numeric * 10_000_000
    if "lakh" in lowered:
        return numeric * 100_000
    return numeric


def _amount_unit(value: str) -> str:
    lowered = value.lower()
    if "crore" in lowered:
        return "INR_crore"
    if "lakh" in lowered:
        return "INR_lakh"
    return "INR"


def _currency_text(value: float | None) -> str | None:
    if value is None:
        return None
    return f"Rs.{value:g}"


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    return cleaned or None


def _clean_sentence(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    return cleaned[:280]


def _price_movement_context(row: ExtractedPriceMovement) -> str:
    parts = [row.period_label]
    if row.period_start_text and row.period_end_text:
        parts.append(f"{row.period_start_text} to {row.period_end_text}")
    return " | ".join(parts)


def _extract_period_end_text(period_text: str | None) -> str | None:
    if not period_text:
        return None
    matches = re.findall(_DATE_RE, period_text, flags=re.IGNORECASE)
    return matches[-1] if matches else None
