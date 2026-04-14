"""Optional LLM judge for open-ended evaluation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from ..config import SebiOrdersRagSettings
from .schemas import EvaluationCase, JudgeScores, RetrievedContext

_PROMPT_VERSION = "sebi_eval_judge_v1"


class JudgeClient(ABC):
    """Abstract judge interface."""

    @abstractmethod
    def evaluate(
        self,
        *,
        case: EvaluationCase,
        retrieved_context: tuple[RetrievedContext, ...],
        route_mode: str,
        answer_status: str,
        answer_text: str,
    ) -> JudgeScores | None:
        raise NotImplementedError


class NoopJudgeClient(JudgeClient):
    """No-op judge used when LLM judging is disabled."""

    def evaluate(
        self,
        *,
        case: EvaluationCase,
        retrieved_context: tuple[RetrievedContext, ...],
        route_mode: str,
        answer_status: str,
        answer_text: str,
    ) -> JudgeScores | None:
        return None


class OpenAIJudgeClient(JudgeClient):
    """JSON-only judge using the configured OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        model_name: str | None = None,
    ) -> None:
        api_key = (settings.openai_api_key or "").strip()
        if not api_key or api_key.upper() in {"YOUR_KEY", "YOUR_API_KEY"}:
            raise RuntimeError("A real OpenAI API key is required for judge execution.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError("openai is required for judge execution.") from exc
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 2, "timeout": 60.0}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)
        self._model_name = model_name or settings.chat_model

    def evaluate(
        self,
        *,
        case: EvaluationCase,
        retrieved_context: tuple[RetrievedContext, ...],
        route_mode: str,
        answer_status: str,
        answer_text: str,
    ) -> JudgeScores | None:
        if not should_use_judge(case):
            return None
        prompt = build_judge_prompt(
            case=case,
            retrieved_context=retrieved_context,
            route_mode=route_mode,
            answer_status=answer_status,
            answer_text=answer_text,
        )
        response = self._client.chat.completions.create(
            model=self._model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
        )
        raw_content = response.choices[0].message.content or "{}"
        payload = _parse_json_object(raw_content)
        return JudgeScores(
            context_relevance=_as_float(payload.get("context_relevance")),
            context_coverage=_as_float(payload.get("context_coverage")),
            faithfulness=_as_float(payload.get("faithfulness")),
            correctness=_as_float(payload.get("correctness")),
            reasoning_quality=_as_float(payload.get("reasoning_quality")),
            conciseness_relevance=_as_float(payload.get("conciseness_relevance")),
            failure_modes=tuple(
                str(item).strip()
                for item in payload.get("failure_modes", [])
                if str(item).strip()
            ),
            rationale=str(payload.get("rationale", "")).strip() or None,
            model_name=self._model_name,
            prompt_version=_PROMPT_VERSION,
            raw=payload,
        )


def should_use_judge(case: EvaluationCase) -> bool:
    """Return whether one case benefits from LLM judging."""

    if case.issue_class in {"open_ended", "redteam", "regression"}:
        return True
    if case.gold_answer_long or case.gold_answer_short:
        return True
    return False


def build_judge_prompt(
    *,
    case: EvaluationCase,
    retrieved_context: tuple[RetrievedContext, ...],
    route_mode: str,
    answer_status: str,
    answer_text: str,
) -> dict[str, str]:
    """Build the reusable structured-JSON judge prompt."""

    context_rows = []
    for item in retrieved_context[:8]:
        context_rows.append(
            {
                "rank": item.rank,
                "record_key": item.record_key,
                "bucket_name": item.bucket_name,
                "title": item.title,
                "section_type": item.section_type,
                "score": item.score,
                "chunk_text": item.chunk_text,
                "source_type": item.source_type,
                "source_url": item.source_url,
            }
        )
    system = (
        "Evaluate the assistant response for a SEBI assistant. "
        "Return structured JSON only. "
        "Score each dimension from 0 to 5. "
        "Detect failure modes from this list only: hallucination, missing key information, "
        "irrelevant context used, over-generalization, incorrect inference, cross-document contamination, "
        "wrong matter lock, stale session override failure, person-vs-trust confusion, "
        "current-fact-vs-order confusion."
    )
    user = json.dumps(
        {
            "instructions": [
                "Evaluate the assistant response given 1. user query 2. retrieved context 3. model response.",
                "Score: context relevance (0-5), context coverage (0-5), faithfulness / hallucination (0-5), correctness (0-5), reasoning quality (0-5), conciseness/relevance (0-5).",
                "Return JSON keys: context_relevance, context_coverage, faithfulness, correctness, reasoning_quality, conciseness_relevance, failure_modes, rationale.",
            ],
            "case": case.to_dict(),
            "route_mode": route_mode,
            "answer_status": answer_status,
            "retrieved_context": context_rows,
            "model_response": answer_text,
        },
        ensure_ascii=True,
        indent=2,
    )
    return {"system": system, "user": user}


def _parse_json_object(raw_content: str) -> dict[str, Any]:
    stripped = raw_content.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}


def _as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
