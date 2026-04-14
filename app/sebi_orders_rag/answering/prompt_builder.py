"""Prompt builders for grounded and direct Phase 4 answers."""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import PromptContextChunk, QueryAnalysis


@dataclass(frozen=True)
class PromptPayload:
    """Rendered system and user prompts for one model call."""

    system_prompt: str
    user_prompt: str


def build_retrieval_answer_prompt(
    *,
    query: str,
    context_chunks: tuple[PromptContextChunk, ...],
    analysis: QueryAnalysis | None = None,
    grounded_summary: str | None = None,
    strict_rule_text: str | None = None,
    locked_record_keys: tuple[str, ...] = (),
    comparison_allowed: bool = False,
) -> PromptPayload:
    """Build a strictly grounded prompt for retrieval-backed answers."""

    summary_block = ""
    if grounded_summary:
        summary_block = (
            "Grounded session memory is provided only to resolve follow-up references. "
            "Do not rely on it for claims unless the same claim appears in the context chunks.\n"
            f"{grounded_summary}\n\n"
        )

    context_lines: list[str] = []
    for chunk in context_chunks:
        context_lines.append(
            f"[{chunk.citation_number}] "
            f"record_key={chunk.record_key} | "
            f"title={chunk.title} | "
            f"section_type={chunk.section_type} | "
            f"pages={chunk.page_start}-{chunk.page_end} | "
            f"document_version_id={chunk.document_version_id} | "
            f"chunk_id={chunk.chunk_id}"
        )
        context_lines.append(chunk.chunk_text.strip())
        context_lines.append("")

    strict_rule_block = ""
    if strict_rule_text and locked_record_keys and not comparison_allowed:
        strict_rule_block = (
            "Single-matter rule in effect:\n"
            f"{strict_rule_text.strip()}\n"
            f"Locked matter record_key(s): {', '.join(locked_record_keys)}\n\n"
        )

    extra_rules: list[str] = []
    if analysis is not None and analysis.asks_brief_summary:
        extra_rules.append(
            "10. For brief-summary requests, answer in 3 to 5 concise sentences covering the matter, core issue, what the order says, and the disposition."
        )
    if analysis is not None and analysis.appears_matter_specific:
        extra_rules.append(
            "11. If the matter is a settlement order, describe allegations, settlement amount or terms, and disposal; do not present settlement disposal as a final merits adjudication."
        )
        extra_rules.append(
            "12. If the matter is an exemption order, explain the exemption sought or granted and any conditions. If it is an appellate, SAT, court, or RTI appellate matter, explain the appeal or request and the disposition."
        )

    user_prompt = (
        f"Question:\n{query.strip()}\n\n"
        f"{summary_block}"
        f"{strict_rule_block}"
        "Context chunks:\n"
        f"{chr(10).join(context_lines).strip()}\n\n"
        "Return valid JSON only with this shape:\n"
        '{"answer_status":"answered|insufficient_context","answer_text":"...","cited_numbers":[1,2]}\n\n'
        "Rules:\n"
        "1. Answer only from the context chunks.\n"
        "2. Put supporting chunk numbers only in cited_numbers, not inside answer_text.\n"
        "3. Distinguish allegations or submissions from findings and operative directions.\n"
        "4. If the context is weak or incomplete, set answer_status to insufficient_context and say so briefly.\n"
        "5. Do not invent facts, dates, penalties, holdings, or parties.\n"
        "6. Prefer findings, directions, and operative_order chunks over header material when answering substantive questions.\n"
        "7. If the question names one specific matter and no comparison is requested, stay inside that matter only.\n"
        "8. Preserve legal caution terms from the source text. Do not upgrade proposed, interim, prima facie, or exemption language into final ownership or final findings.\n"
        "9. Keep answer_text concise and readable.\n"
        f"{chr(10).join(extra_rules)}"
    )
    system_prompt = (
        "You answer questions about SEBI orders using retrieved corpus text only. "
        "Do not add unsupported claims. Do not infer beyond the provided context."
    )
    return PromptPayload(system_prompt=system_prompt, user_prompt=user_prompt)


def build_general_knowledge_prompt(*, query: str) -> PromptPayload:
    """Build a direct explanatory prompt for generic non-matter questions."""

    user_prompt = (
        f"Question:\n{query.strip()}\n\n"
        "Return valid JSON only with this shape:\n"
        '{"answer_status":"answered|insufficient_context","answer_text":"..."}\n\n'
        "Rules:\n"
        "1. Give a concise general-knowledge answer.\n"
        "2. Do not claim that you searched or cited the SEBI orders corpus.\n"
        "3. Do not imply access to current official facts unless the question itself is timeless.\n"
        "4. If the question is too broad or ambiguous, set answer_status to insufficient_context.\n"
    )
    system_prompt = (
        "You provide concise general explanations about securities regulation, law, finance, "
        "and economics. Be clear about uncertainty and do not imply corpus grounding."
    )
    return PromptPayload(system_prompt=system_prompt, user_prompt=user_prompt)


def build_direct_answer_prompt(*, query: str) -> PromptPayload:
    """Backward-compatible alias for the generic direct-answer prompt."""

    return build_general_knowledge_prompt(query=query)
