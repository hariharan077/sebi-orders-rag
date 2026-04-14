"""Provider interfaces and OpenAI-backed web-search execution."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from ..config import SebiOrdersRagSettings
from ..exceptions import ConfigurationError, MissingDependencyError
from .models import WebSearchRequest, WebSearchResult, WebSearchSource
from .ranking import canonicalize_source_url, extract_domain, is_domain_allowed, rank_web_sources

_COMMON_SEARCH_INSTRUCTIONS = (
    "Use web search only for this answer. "
    "If the sources clearly support an answer, respond with 'ANSWER: ' followed by a concise answer. "
    "If support is weak, outdated, mixed, or missing, respond with 'INSUFFICIENT: ' followed by a brief reason. "
    "Do not include inline bracket markers or a separate sources list."
)
_WEB_SEARCH_MODEL_FALLBACKS: tuple[str, ...] = ("o4-mini", "gpt-5")
_WEB_SEARCH_MAX_OUTPUT_TOKENS = 1600
_SUPPORTED_WEB_SEARCH_PROVIDERS = frozenset({"openai", "openai_responses", "responses"})
_INLINE_MARKDOWN_LINK_RE = re.compile(r"\(\s*\[[^\]]+\]\((https?://[^)]+)\)\s*\)")
_BARE_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


class WebSearchProvider(ABC):
    """Abstract provider for controlled web fallback."""

    @abstractmethod
    def search(self, *, request: WebSearchRequest) -> WebSearchResult:
        """Return one web-search-backed answer."""


class UnavailableWebSearchProvider(WebSearchProvider):
    """Provider used when web search is disabled or unsupported."""

    def __init__(self, *, reason: str, provider_name: str) -> None:
        self._reason = reason
        self._provider_name = provider_name

    def search(self, *, request: WebSearchRequest) -> WebSearchResult:
        return WebSearchResult(
            answer_status="unavailable",
            answer_text=self._reason,
            provider_name=self._provider_name,
            lookup_type=request.lookup_type,
            debug={
                "provider_available": False,
                "query": request.query,
                "allowed_domains": list(request.allowed_domains),
            },
        )


class OpenAIResponsesWebSearchProvider(WebSearchProvider):
    """OpenAI Responses-backed web search wrapped behind a stable interface."""

    def __init__(
        self,
        *,
        settings: SebiOrdersRagSettings,
        provider_name: str,
        default_source_type: str,
        default_allowed_domains: tuple[str, ...] = (),
        default_search_context_size: str = "medium",
        default_max_results: int = 6,
    ) -> None:
        api_key = (settings.openai_api_key or "").strip()
        if not api_key or api_key.upper() in {"YOUR_KEY", "YOUR_API_KEY"}:
            raise ConfigurationError(
                "SEBI_ORDERS_RAG_OPENAI_API_KEY must be configured for web search fallback."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise MissingDependencyError(
                "openai is required for web-search fallback. "
                "Install the dependencies from requirements-sebi-orders-rag.txt."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 2,
            "timeout": settings.web_search_timeout_seconds,
        }
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = OpenAI(**client_kwargs)
        self._settings = settings
        self._provider_name = provider_name
        self._default_source_type = default_source_type
        self._default_allowed_domains = default_allowed_domains
        self._default_search_context_size = default_search_context_size
        self._default_max_results = max(1, int(default_max_results))

    def search(self, *, request: WebSearchRequest) -> WebSearchResult:
        allowed_domains = request.allowed_domains or self._default_allowed_domains
        search_context_size = request.search_context_size or self._default_search_context_size
        max_results = max(1, int(request.max_results or self._default_max_results))
        instructions = "\n\n".join(
            part.strip()
            for part in (_COMMON_SEARCH_INSTRUCTIONS, request.instructions)
            if part and part.strip()
        )

        response = None
        tool_type_used: str | None = None
        model_used: str | None = None
        last_error: str | None = None
        attempt_errors: list[dict[str, str]] = []
        for model_name in self._candidate_models(requires_domain_filters=bool(allowed_domains)):
            for tool_type in ("web_search", "web_search_2025_08_26"):
                if allowed_domains and tool_type != "web_search":
                    continue
                try:
                    response = self._client.responses.create(
                        model=model_name,
                        input=request.query,
                        instructions=instructions,
                        tools=[self._build_tool(tool_type, allowed_domains, search_context_size)],
                        include=["web_search_call.action.sources"],
                        max_output_tokens=_WEB_SEARCH_MAX_OUTPUT_TOKENS,
                        timeout=self._settings.web_search_timeout_seconds,
                        **self._optional_request_kwargs(model_name),
                    )
                    tool_type_used = tool_type
                    model_used = model_name
                    break
                except Exception as exc:  # pragma: no cover - depends on runtime/provider
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempt_errors.append(
                        {
                            "model": model_name,
                            "tool_type": tool_type,
                            "error": last_error,
                        }
                    )
                    response = None
            if response is not None:
                break

        if response is None:
            return WebSearchResult(
                answer_status="unavailable",
                answer_text=(
                    "Web search fallback is configured but unavailable in this environment right now."
                ),
                provider_name=self._provider_name,
                lookup_type=request.lookup_type,
                debug={
                    "provider_available": False,
                    "query": request.query,
                    "allowed_domains": list(allowed_domains),
                    "tool_type": tool_type_used,
                    "error": last_error,
                    "attempts": attempt_errors,
                },
            )

        answer_text, annotations = _extract_answer_and_annotations(response)
        answer_status, normalized_text = _normalize_answer_text(answer_text)
        annotation_sources = _extract_sources_from_annotations(
            annotations,
            source_type=request.source_type,
        )
        response_sources = (
            ()
            if annotation_sources
            else _extract_sources_from_response(
                response,
                source_type=request.source_type,
            )
        )
        raw_sources = _merge_web_sources(annotation_sources, response_sources)
        ranked_sources = rank_web_sources(
            raw_sources,
            allowed_domains=allowed_domains,
            unique_per_domain=request.source_type == "general_web",
        )
        if request.source_type == "official_web" and allowed_domains:
            ranked_sources = tuple(
                source
                for source in ranked_sources
                if is_domain_allowed(source.domain, allowed_domains)
            )
        ranked_sources = ranked_sources[:max_results]

        if answer_status == "answered" and not ranked_sources:
            answer_status = "insufficient_context"
            normalized_text = "I could not verify that from reliable web sources."

        return WebSearchResult(
            answer_status=answer_status,
            answer_text=normalized_text,
            sources=ranked_sources,
            provider_name=self._provider_name,
            lookup_type=request.lookup_type,
            debug={
                "provider_available": True,
                "query": request.query,
                "allowed_domains": list(allowed_domains),
                "search_context_size": search_context_size,
                "model_used": model_used,
                "tool_type": tool_type_used,
                "raw_source_count": len(raw_sources),
                "filtered_source_count": len(ranked_sources),
                "max_results": max_results,
                "source_domains": [source.domain for source in ranked_sources],
                "attempts": attempt_errors,
            },
        )

    @staticmethod
    def _build_tool(
        tool_type: str,
        allowed_domains: tuple[str, ...],
        search_context_size: str,
    ) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "type": tool_type,
            "search_context_size": search_context_size,
        }
        if tool_type == "web_search" and allowed_domains:
            tool["filters"] = {"allowed_domains": list(allowed_domains)}
        return tool

    def _candidate_models(self, *, requires_domain_filters: bool) -> tuple[str, ...]:
        candidates: list[str] = []
        primary_model = (self._settings.chat_model or "").strip()
        if primary_model:
            candidates.append(primary_model)
        if requires_domain_filters:
            candidates.extend(_WEB_SEARCH_MODEL_FALLBACKS)
        else:
            candidates.extend(_WEB_SEARCH_MODEL_FALLBACKS)

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return tuple(deduped)

    @staticmethod
    def _optional_request_kwargs(model_name: str) -> dict[str, Any]:
        normalized = model_name.strip().lower()
        if normalized.startswith("gpt-5") or normalized.startswith("o"):
            return {"reasoning": {"effort": "low"}}
        return {}


def build_official_web_search_provider(
    settings: SebiOrdersRagSettings,
) -> WebSearchProvider:
    """Return the configured official-domain web-search provider."""

    if not settings.web_fallback_enabled:
        return UnavailableWebSearchProvider(
            reason="Web fallback is disabled in this environment.",
            provider_name="official_web_search",
        )
    if not settings.official_web_search_enabled:
        return UnavailableWebSearchProvider(
            reason="Official-domain web search is disabled in this environment.",
            provider_name="official_web_search",
        )
    if settings.web_search_provider not in _SUPPORTED_WEB_SEARCH_PROVIDERS:
        return UnavailableWebSearchProvider(
            reason=(
                "Unsupported web search provider configured: "
                f"{settings.web_search_provider!r}. Supported values: openai."
            ),
            provider_name="official_web_search",
        )
    try:
        return OpenAIResponsesWebSearchProvider(
            settings=settings,
            provider_name="official_web_search",
            default_source_type="official_web",
            default_allowed_domains=settings.official_allowed_domains,
            default_search_context_size="low",
            default_max_results=settings.web_search_max_results,
        )
    except (ConfigurationError, MissingDependencyError) as exc:
        return UnavailableWebSearchProvider(
            reason=str(exc),
            provider_name="official_web_search",
        )


def build_general_web_search_provider(
    settings: SebiOrdersRagSettings,
) -> WebSearchProvider:
    """Return the configured broader web-search provider."""

    if not settings.web_fallback_enabled:
        return UnavailableWebSearchProvider(
            reason="Web fallback is disabled in this environment.",
            provider_name="general_web_search",
        )
    if not settings.general_web_search_enabled:
        return UnavailableWebSearchProvider(
            reason="General web search is disabled in this environment.",
            provider_name="general_web_search",
        )
    if settings.web_search_provider not in _SUPPORTED_WEB_SEARCH_PROVIDERS:
        return UnavailableWebSearchProvider(
            reason=(
                "Unsupported web search provider configured: "
                f"{settings.web_search_provider!r}. Supported values: openai."
            ),
            provider_name="general_web_search",
        )
    try:
        return OpenAIResponsesWebSearchProvider(
            settings=settings,
            provider_name="general_web_search",
            default_source_type="general_web",
            default_allowed_domains=settings.general_allowed_domains,
            default_search_context_size="medium",
            default_max_results=settings.web_search_max_results,
        )
    except (ConfigurationError, MissingDependencyError) as exc:
        return UnavailableWebSearchProvider(
            reason=str(exc),
            provider_name="general_web_search",
        )


def _extract_answer_and_annotations(response: Any) -> tuple[str, tuple[Any, ...]]:
    text_parts: list[str] = []
    annotations: list[Any] = []
    for item in getattr(response, "output", ()) or ():
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", ()) or ():
            if getattr(content, "type", None) != "output_text":
                continue
            text = str(getattr(content, "text", "") or "").strip()
            if text:
                text_parts.append(text)
            annotations.extend(tuple(getattr(content, "annotations", ()) or ()))
    if not text_parts:
        text = str(getattr(response, "output_text", "") or "").strip()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts).strip(), tuple(annotations)


def _normalize_answer_text(answer_text: str) -> tuple[str, str]:
    stripped = (answer_text or "").strip()
    if " INSUFFICIENT:" in stripped and not stripped.upper().startswith("INSUFFICIENT:"):
        stripped = stripped.split(" INSUFFICIENT:", 1)[0].strip()
    upper = stripped.upper()
    if upper.startswith("ANSWER:"):
        return "answered", _strip_inline_markdown_links(stripped[7:].strip())
    if upper.startswith("INSUFFICIENT:"):
        return "insufficient_context", _strip_inline_markdown_links(stripped[13:].strip())
    lowered = stripped.lower()
    if not stripped:
        return "insufficient_context", "I could not find enough reliable support to answer that."
    if lowered.startswith("i could not") or "not enough reliable" in lowered:
        return "insufficient_context", _strip_inline_markdown_links(stripped)
    return "answered", _strip_inline_markdown_links(stripped)


def _extract_sources_from_annotations(
    annotations: tuple[Any, ...],
    *,
    source_type: str,
) -> tuple[WebSearchSource, ...]:
    sources: list[WebSearchSource] = []
    for annotation in annotations:
        if getattr(annotation, "type", None) != "url_citation":
            continue
        url = canonicalize_source_url(str(getattr(annotation, "url", "") or ""))
        title = str(getattr(annotation, "title", "") or "").strip()
        domain = extract_domain(url)
        if not url or not domain:
            continue
        sources.append(
            WebSearchSource(
                source_title=title or domain,
                source_url=url,
                domain=domain,
                source_type=source_type,
                record_key=f"{source_type}:{domain}",
            )
        )
    return tuple(sources)


def _extract_sources_from_response(
    response: Any,
    *,
    source_type: str,
) -> tuple[WebSearchSource, ...]:
    sources: list[WebSearchSource] = []
    for item in getattr(response, "output", ()) or ():
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        for source in getattr(action, "sources", ()) or ():
            url = canonicalize_source_url(str(getattr(source, "url", "") or ""))
            domain = extract_domain(url)
            if not url or not domain:
                continue
            sources.append(
                WebSearchSource(
                    source_title=_source_title_from_url(url),
                    source_url=url,
                    domain=domain,
                    source_type=source_type,
                    record_key=f"{source_type}:{domain}",
                )
            )
    return tuple(sources)


def _merge_web_sources(*source_groups: tuple[WebSearchSource, ...]) -> tuple[WebSearchSource, ...]:
    merged: dict[str, WebSearchSource] = {}
    for group in source_groups:
        for source in group:
            existing = merged.get(source.source_url)
            if existing is None:
                merged[source.source_url] = source
                continue
            if existing.source_title == existing.domain and source.source_title != source.domain:
                merged[source.source_url] = source
    return tuple(merged.values())


def _strip_inline_markdown_links(text: str) -> str:
    without_parenthetical_links = _INLINE_MARKDOWN_LINK_RE.sub("", text)
    without_markdown_links = _BARE_MARKDOWN_LINK_RE.sub(r"\1", without_parenthetical_links)
    normalized = re.sub(r"[ \t]+", " ", without_markdown_links)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    normalized = re.sub(
        r"\s*\(([a-z0-9.-]+\.[a-z]{2,})(?:/[^\s)]*)?\)?$",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _source_title_from_url(url: str) -> str:
    domain = extract_domain(url)
    normalized = canonicalize_source_url(url)
    path = normalized.split("?", 1)[0].rstrip("/").split("/")
    if not path:
        return domain or url
    leaf = next((segment for segment in reversed(path) if segment), "")
    if not leaf:
        return domain or url
    slug = leaf.rsplit(".", 1)[0] if "." in leaf else leaf
    slug = re.sub(r"_\d+$", "", slug)
    slug = slug.replace("-", " ").replace("_", " ").strip()
    slug = re.sub(r"\s+", " ", slug)
    if not slug or slug.isdigit():
        return domain or url
    words = [
        word.upper() if word.isupper() and len(word) <= 5 else word.capitalize()
        for word in slug.split()
    ]
    rendered = " ".join(words).strip()
    return rendered or domain or url
